// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include "gguf_parser.h"
#include "tokenizer.h"
#include "ns_q4k_quant.h"
#include "ns_gguf_quant.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <random>
#include <omp.h>
#include <chrono>
#include <regex>
#include <numeric>

#include <cassert>

static const size_t CLUSTER_SIZE = 256;
// activation_freq stores each cluster's absolute mean-|y| score. Clusters are
// ranked globally across all tensors; the top hot_budget elements -> Q8_0,
// bottom cold_budget -> Q2_K, rest -> Q4_K (see --hot-budget/--cold-budget).

// --dequant-input: treat Q4_K/Q6_K/Q8_0 input tensors as dequantizable to float
static bool g_dequant_input = false;

static bool is_dequant_type(GGMLType ty) {
    return ty == GGMLType::Q4_K || ty == GGMLType::Q6_K || ty == GGMLType::Q8_0;
}

// Byte offset for a given number of elements, per on-disk type.
// Quantized types are block-packed: caller must keep element offsets block-aligned.
static uint64_t elem_offset_bytes(GGMLType ty, size_t elems) {
    switch (ty) {
        case GGMLType::F16:  return (uint64_t)elems * sizeof(uint16_t);
        case GGMLType::Q4_K: return (uint64_t)(elems / QK_K) * sizeof(block_q4_K);
        case GGMLType::Q6_K: return (uint64_t)(elems / QK_K) * sizeof(block_q6_K);
        case GGMLType::Q8_0: return (uint64_t)(elems / QK8_0) * sizeof(block_q8_0);
        default:             return (uint64_t)elems * sizeof(float);
    }
}
struct ClusterOut {
    float activation_freq;
    uint8_t quant_bits;
};

struct TensorOut {
    char name[64];
    uint32_t n_clusters;
    uint32_t quant_type;   // GGMLType value written to the GGUF tensor descriptor
    uint64_t data_offset;  // offset within the temp data stream
    uint64_t data_bytes;
    std::vector<ClusterOut> clusters;
};

// Per-tensor metadata computed in the profiling pass and reused in the
// quantize pass (the global budget sort sits between the two passes).
struct TensorJob {
    int n_rows = 1, n_cols = 1;
    size_t nc = 0;
    size_t row_chunk = 0;
    bool float_path = false;
    bool quantizable = false;
    bool kquant_ok = false, q8_ok = false;
    bool f32_tensor = false;
    bool profiled = false;
    int min_bits = 2, max_bits = 8;
};

static size_t file_size(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return f.tellg();
}

static std::vector<std::string> read_prompts(const std::string& path, int want) {
    std::vector<std::string> out;
    std::ifstream f(path);
    if (f) {
        std::string line;
        while (out.size() < (size_t)want && std::getline(f, line)) {
            if (!line.empty()) out.push_back(line);
        }
    }
    if (out.empty()) {
        std::mt19937 rng(123);
        for (int i = 0; i < want; ++i) {
            out.push_back("prompt " + std::to_string(i) + " placeholder text");
        }
    }
    if (out.size() > (size_t)want) out.resize(want);
    return out;
}

static size_t numel(const GGUFTensor& t) {
    size_t n = 1;
    for (auto d : t.shape) n *= (size_t)d;
    return n;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t mant =  h & 0x03FFu;

    if (exp == 0) {
        if (mant == 0) { uint32_t v = sign; float f; std::memcpy(&f, &v, sizeof(f)); return f; }
        float f = (float)mant * (1.0f / 1024.0f) * (1.0f / 16384.0f);
        return (h & 0x8000u) ? -f : f;
    }
    if (exp == 31) {
        uint32_t v = sign | (0xFFu << 23) | (mant << 13);
        float f; std::memcpy(&f, &v, sizeof(f)); return f;
    }
    uint32_t v = sign | ((exp + 112) << 23) | (mant << 13);
    float f; std::memcpy(&f, &v, sizeof(f)); return f;
}

static bool read_tensor_floats(const GGUFParser& parser, const GGUFTensor& t, std::vector<float>& out) {
    size_t n = numel(t);
    out.resize(n);
    if (t.type == GGMLType::F32) {
        return parser.read_tensor(t, (uint8_t*)out.data(), n * sizeof(float));
    } else if (t.type == GGMLType::F16) {
        std::vector<uint16_t> raw(n);
        if (!parser.read_tensor(t, (uint8_t*)raw.data(), n * sizeof(uint16_t))) return false;
        for (size_t i = 0; i < n; ++i) out[i] = f16_to_f32(raw[i]);
        return true;
    } else if (g_dequant_input && t.type == GGMLType::Q4_K) {
        if (n % QK_K != 0) {
            std::cerr << "read_tensor_floats: " << t.name << " Q4_K not block-aligned (n=" << n << ")" << std::endl;
            return false;
        }
        std::vector<uint8_t> raw((n / QK_K) * sizeof(block_q4_K));
        if (!parser.read_tensor(t, raw.data(), raw.size())) return false;
        dequantize_row_q4_K((const block_q4_K*)raw.data(), out.data(), (int64_t)n);
        return true;
    } else if (g_dequant_input && t.type == GGMLType::Q6_K) {
        if (n % QK_K != 0) {
            std::cerr << "read_tensor_floats: " << t.name << " Q6_K not block-aligned (n=" << n << ")" << std::endl;
            return false;
        }
        std::vector<uint8_t> raw((n / QK_K) * sizeof(block_q6_K));
        if (!parser.read_tensor(t, raw.data(), raw.size())) return false;
        dequantize_row_q6_K((const block_q6_K*)raw.data(), out.data(), (int64_t)n);
        return true;
    } else if (g_dequant_input && t.type == GGMLType::Q8_0) {
        if (n % QK8_0 != 0) {
            std::cerr << "read_tensor_floats: " << t.name << " Q8_0 not block-aligned (n=" << n << ")" << std::endl;
            return false;
        }
        std::vector<uint8_t> raw((n / QK8_0) * sizeof(block_q8_0));
        if (!parser.read_tensor(t, raw.data(), raw.size())) return false;
        dequantize_row_q8_0((const block_q8_0*)raw.data(), out.data(), (int64_t)n);
        return true;
    }
    std::cerr << "read_tensor_floats: unsupported type for " << t.name << std::endl;
    return false;
}

static bool is_preserve(const std::string& name) {
    std::regex re_preserve("^(blk\\.\\d+\\.attn_(q|k|v)\\.(bias))$");
    return std::regex_match(name, re_preserve);
}

// Tensors that must stay F32 in GGUF output: norms and biases are 1D and
// llama.cpp expects them unquantized.
static bool is_f32_tensor(const std::string& name) {
    if (is_preserve(name)) return false;
    static const char* suffixes[] = {
        "attn_norm.weight", "ffn_norm.weight", "output_norm.weight", ".bias"
    };
    for (const char* s : suffixes) {
        size_t sl = strlen(s);
        if (name.size() >= sl && name.compare(name.size() - sl, sl, s) == 0)
            return true;
    }
    return false;
}

// Quantize a float buffer to a GGUF-standard block type. Returns bytes written.
static size_t quantize_to_type(const std::vector<float>& w, GGMLType ty, std::vector<uint8_t>& out) {
    size_t n = w.size();
    switch (ty) {
        case GGMLType::F32:
            out.resize(n * sizeof(float));
            std::memcpy(out.data(), w.data(), n * sizeof(float));
            return out.size();
        case GGMLType::F16: {
            out.resize(n * sizeof(uint16_t));
            uint16_t* h = (uint16_t*)out.data();
            for (size_t i = 0; i < n; ++i) h[i] = f32_to_fp16(w[i]);
            return out.size();
        }
        case GGMLType::Q8_0: {
            out.resize((n / QK8_0) * sizeof(block_q8_0));
            quantize_row_q8_0(w.data(), (block_q8_0*)out.data(), (int64_t)n);
            return out.size();
        }
        case GGMLType::Q4_K: {
            out.resize((n / QK_K) * sizeof(block_q4_K));
            quantize_row_q4_K(w.data(), (block_q4_K*)out.data(), (int64_t)n);
            return out.size();
        }
        case GGMLType::Q2_K: {
            out.resize((n / QK_K) * sizeof(block_q2_K));
            quantize_row_q2_K(w.data(), (block_q2_K*)out.data(), (int64_t)n);
            return out.size();
        }
        default:
            return 0;
    }
}

// Profiles each row by mean |y| across prompts (y_r = dot(w_row, x)), then
// stores each cluster's mean |y| PERCENTILE RANK (0..1 within this tensor) into
// cluster_score. The rank is normalized per-tensor so a global sort distributes
// the hot/warm/cold budget within every tensor — a raw |y| score would instead
// be dominated by tensor scale and push whole low-magnitude tensors to Q2_K.
static void profile_tensor(const float* w, int n_rows, int n_cols,
                           const std::vector<std::vector<float>>& xs,
                           std::vector<float>& cluster_score) {
    size_t total = (size_t)n_rows * (size_t)n_cols;
    size_t n_clusters = (total + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
    cluster_score.assign(n_clusters, 0.5f);
    if (n_rows <= 0 || n_cols <= 0 || xs.empty()) return;

    // Accumulate |y_r| per row across all prompts (parallel over prompts).
    std::vector<float> row_sum(n_rows, 0.0f);
    #pragma omp parallel
    {
        std::vector<float> local(n_rows, 0.0f);
        #pragma omp for nowait
        for (size_t pi = 0; pi < xs.size(); ++pi) {
            const std::vector<float>& x = xs[pi];
            for (int r = 0; r < n_rows; ++r) {
                float s = 0.0f;
                const float* wr = w + (size_t)r * n_cols;
                for (int c = 0; c < n_cols; ++c) s += wr[c] * x[c];
                local[r] += std::fabs(s);
            }
        }
        #pragma omp critical
        {
            for (int r = 0; r < n_rows; ++r) row_sum[r] += local[r];
        }
    }

    // mean_activation[r] = mean |y_r| across prompts
    std::vector<float> mean_act(n_rows);
    float inv = 1.0f / (float)xs.size();
    for (int r = 0; r < n_rows; ++r) mean_act[r] = row_sum[r] * inv;

    // Percentile rank of each row's mean_activation (0 = smallest, 1 = largest)
    std::vector<int> order(n_rows);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return mean_act[a] < mean_act[b]; });
    std::vector<float> row_rank(n_rows);
    float denom = (n_rows > 1) ? (float)(n_rows - 1) : 1.0f;
    for (int i = 0; i < n_rows; ++i) row_rank[order[i]] = (float)i / denom;

    // Each cluster's score = element-weighted mean rank of the rows it overlaps.
    for (size_t ci = 0; ci < n_clusters; ++ci) {
        size_t f0 = ci * CLUSTER_SIZE;
        size_t f1 = std::min(f0 + CLUSTER_SIZE - 1, total - 1);
        int r0 = (int)(f0 / (size_t)n_cols);
        int r1 = (int)(f1 / (size_t)n_cols);
        double acc = 0.0;
        size_t cnt = 0;
        for (int r = r0; r <= r1; ++r) {
            size_t e0 = std::max(f0, (size_t)r * (size_t)n_cols);
            size_t e1 = std::min(f1, (size_t)r * (size_t)n_cols + (size_t)n_cols - 1);
            size_t nn = (e1 >= e0) ? (e1 - e0 + 1) : 0;
            acc += (double)row_rank[r] * (double)nn;
            cnt += nn;
        }
        cluster_score[ci] = cnt ? (float)(acc / (double)cnt) : 0.5f;
    }
}

static std::vector<std::vector<float>> build_prompt_inputs(const BPETokenizer& tok,
                                                           const std::vector<std::string>& prompts,
                                                           int n_embd,
                                                           const std::vector<float>& token_embd,
                                                           int n_vocab) {
    std::vector<std::vector<float>> out;
    out.reserve(prompts.size());

    for (const auto& p : prompts) {
        std::vector<float> x(n_embd, 0.0f);
        std::vector<int32_t> ids;
        bool ok = false;
        if (tok.vocab_size() > 0) {
            ids = tok.encode(p);
            ok = !ids.empty();
        }
        if (ok && !token_embd.empty()) {
            const float* te = token_embd.data();
            for (int id : ids) {
                if (id >= 0 && id < n_vocab) {
                    const float* row = te + (size_t)id * n_embd;
                    for (int i = 0; i < n_embd; ++i) x[i] += row[i];
                }
            }
            for (int i = 0; i < n_embd; ++i) x[i] /= (float)ids.size();
        } else if (!token_embd.empty()) {
            // Use a stable random input seeded by prompt
            std::mt19937 rng((unsigned)std::hash<std::string>{}(p));
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for (int i = 0; i < n_embd; ++i) x[i] = dist(rng);
        } else {
            std::mt19937 rng((unsigned)std::hash<std::string>{}(p));
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for (int i = 0; i < n_embd; ++i) x[i] = dist(rng);
        }
        out.push_back(std::move(x));
    }
    return out;
}

int main(int argc, char** argv) {
    long long cli_hot_budget = 500000000;
    long long cli_cold_budget = 100000000;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dequant-input") g_dequant_input = true;
        else if (a == "--hot-budget" && i + 1 < argc) cli_hot_budget = std::atoll(argv[++i]);
        else if (a == "--cold-budget" && i + 1 < argc) cli_cold_budget = std::atoll(argv[++i]);
        else pos.push_back(a);
    }
    if (pos.size() < 2) {
        std::cerr << "Usage: " << argv[0] << " [--dequant-input] [--hot-budget N] [--cold-budget N] <input.gguf> <output.gguf> [n_prompts [prompts.txt]]" << std::endl;
        return 1;
    }
    const char* in_path = pos[0].c_str();
    const char* out_path = pos[1].c_str();
    int n_prompts = (pos.size() >= 3) ? std::atoi(pos[2].c_str()) : 500;
    if (n_prompts < 0) n_prompts = 0;
    std::string prompts_path = (pos.size() >= 4) ? pos[3] : "prompts.txt";

    auto t0 = std::chrono::steady_clock::now();

    GGUFParser parser;
    if (!parser.load_pread(in_path)) {
        std::cerr << "Failed to load " << in_path << std::endl;
        return 1;
    }

    BPETokenizer tok;
    tok.load(parser);

    const ModelParams& mp = parser.params();
    int n_embd = (int)mp.n_embd;
    if (n_embd == 0) n_embd = 64;

    // --dequant-input: reject tensor types we cannot dequantize (Q2_K, Q1/IQ*, etc.)
    if (g_dequant_input) {
        for (const auto& t : parser.tensors()) {
            if (t.type == GGMLType::F32 || t.type == GGMLType::F16 || is_dequant_type(t.type)) continue;
            std::cerr << "Error: --dequant-input cannot dequantize tensor '" << t.name
                      << "' (type " << (uint32_t)t.type << "). Supported input types: F32, F16, Q4_K, Q6_K, Q8_0."
                      << std::endl;
            return 1;
        }
        std::cout << "--dequant-input: dequantizing Q4_K/Q6_K/Q8_0 tensors to float before profiling" << std::endl;
    }

    bool any_float = false;
    for (const auto& t : parser.tensors()) {
        if (t.type == GGMLType::F32 || t.type == GGMLType::F16 ||
            (g_dequant_input && is_dequant_type(t.type))) { any_float = true; break; }
    }
    bool structural_q4 = !any_float;

    // Read token_embd into memory once for prompt profiling, then release it
    // before the per-tensor loop so only one tensor is in RAM at a time.
    const GGUFTensor* et = nullptr;
    for (const auto& t : parser.tensors()) {
        if (t.name == "token_embd.weight" && (t.type == GGMLType::F32 || t.type == GGMLType::F16 ||
            (g_dequant_input && is_dequant_type(t.type)))) { et = &t; break; }
    }
    std::vector<float> token_embd;
    int n_vocab = 0;
    if (et && et->shape.size() >= 2) {
        size_t n = numel(*et);
        (void)n;
        if (!read_tensor_floats(parser, *et, token_embd)) {
            std::cerr << "Failed to read token_embd" << std::endl;
            return 1;
        }
        n_vocab = (int)et->shape[0];
    }

    std::vector<std::vector<float>> prompt_inputs;
    if (!structural_q4 && n_prompts > 0) {
        auto prompts = read_prompts(prompts_path, n_prompts);
        prompt_inputs = build_prompt_inputs(tok, prompts, n_embd, token_embd, n_vocab);
        std::cout << "Profiled with " << prompt_inputs.size() << " prompt samples" << std::endl;
    }

    // Token embedding is no longer needed after prompt inputs are built
    token_embd.clear();
    token_embd.shrink_to_fit();

    // RAM budget pass: largest float dequant buffer, current + next + 20% headroom
    size_t largest_bytes = 0;
    for (const auto& t : parser.tensors()) {
        size_t b = numel(t) * 4; // F32 dequant estimate
        if (b > largest_bytes) largest_bytes = b;
    }
    size_t working_budget = (size_t)(largest_bytes * 2.0 * 1.20);
    size_t chunk_floats = working_budget / 2 / sizeof(float);
    chunk_floats = (chunk_floats / 256) * 256;
    assert(chunk_floats >= 256);
    std::cout << "RAM budget: " << (working_budget / (1024 * 1024)) << " MB" << std::endl;

    // Stream quantized tensor data to a temporary file; only one tensor is held in RAM
    std::string data_tmp_path = std::string(out_path) + ".data.tmp";
    FILE* data_tmp = fopen(data_tmp_path.c_str(), "w+b");
    if (!data_tmp) {
        std::cerr << "Failed to open temporary data file " << data_tmp_path << std::endl;
        return 1;
    }

    size_t total_hot = 0, total_warm = 0, total_cold = 0;
    size_t n_tensors = parser.tensors().size();

    std::vector<TensorOut> outs(n_tensors);
    std::vector<TensorJob> jobs(n_tensors);

    // PASS 1: per-tensor metadata + mean-|y| profiling per cluster. The score is
    // stored in activation_freq; quantization is deferred until after the
    // global budget sort decides every cluster's bit width.
    for (size_t ti = 0; ti < n_tensors; ++ti) {
        const GGUFTensor& t = parser.tensors()[ti];
        if ((ti % 10) == 0) {
            std::cout << "Profiling [" << (ti + 1) << "/" << n_tensors << "] " << t.name << std::endl;
        }

        TensorJob& J = jobs[ti];
        TensorOut& to = outs[ti];
        std::memset(to.name, 0, sizeof(to.name));
        std::strncpy(to.name, t.name.c_str(), sizeof(to.name) - 1);
        to.n_clusters = 0;
        to.quant_type = 0;
        to.data_offset = 0;
        to.data_bytes = 0;

        J.float_path = (t.type == GGMLType::F32 || t.type == GGMLType::F16) ||
                       (g_dequant_input && is_dequant_type(t.type));
        if (!J.float_path) continue;  // passthrough handled in pass 2

        size_t n_full = numel(t);
        J.nc = (n_full + CLUSTER_SIZE - 1) / CLUSTER_SIZE;

        // Flatten shape; treat shape[0] as input columns and shape[1] as
        // output rows, matching the GGUF layout and the NSRun kernels.
        if (t.shape.size() >= 2) {
            J.n_cols = (int)t.shape[0];
            J.n_rows = (int)t.shape[1];
        } else if (!t.shape.empty()) {
            J.n_rows = (int)t.shape[0];
            J.n_cols = 1;
        }

        bool preserve = is_preserve(t.name);
        J.f32_tensor = is_f32_tensor(t.name);
        bool is2d = (t.shape.size() >= 2);
        // K-quants need ne[0] % 256; Q8_0 needs ne[0] % 32
        J.kquant_ok = is2d && (J.n_cols % (int)QK_K == 0);
        J.q8_ok = is2d && (J.n_cols % (int)QK8_0 == 0);
        J.quantizable = is2d && !preserve && !J.f32_tensor && (J.kquant_ok || J.q8_ok);

        to.n_clusters = (uint32_t)J.nc;
        to.clusters.resize(J.nc);

        // Precision floor: attention and lm_head must never be Q2, and Q8 is
        // wasted because the loader requantizes to Q4_K, so keep them at Q4.
        if (t.name.find("token_embd") != std::string::npos ||
            t.name.find("blk.0.") != std::string::npos ||
            t.name.find(".attn_q.") != std::string::npos ||
            t.name.find(".attn_k.") != std::string::npos ||
            t.name.find(".attn_v.") != std::string::npos ||
            t.name.find(".attn_output.") != std::string::npos) {
            J.min_bits = 8;
            J.max_bits = 8;
        } else if (t.name.find("attn_q") != std::string::npos ||
                   t.name.find("attn_k") != std::string::npos ||
                   t.name.find("attn_v") != std::string::npos ||
                   t.name.find("attn_output") != std::string::npos ||
                   t.name.find("output.weight") != std::string::npos) {
            J.min_bits = 4;
            J.max_bits = 4;
        }

        // Build prompt inputs projected to n_cols once per tensor
        std::vector<std::vector<float>> xs;
        bool do_profile = J.quantizable && n_prompts > 0 && (int)prompt_inputs.size() == n_prompts && J.n_rows > 1 && J.n_cols > 1;
        if (do_profile) {
            for (const auto& p : prompt_inputs) {
                int dim = p.size();
                std::vector<float> x(J.n_cols, 0.0f);
                if (dim == J.n_cols) x = p;
                else {
                    std::mt19937 rng((unsigned)(dim + J.n_cols));
                    std::normal_distribution<float> dist(0.0f, 0.02f);
                    for (float& v : x) v = dist(rng);
                }
                xs.push_back(std::move(x));
            }
        }
        J.profiled = do_profile;

        // Decide chunking along rows to stay within the RAM budget
        J.row_chunk = (size_t)J.n_rows;
        if (n_full > chunk_floats) {
            size_t g = std::gcd((size_t)J.n_cols, (size_t)CLUSTER_SIZE);
            size_t col_step = CLUSTER_SIZE / g;
            size_t rows_per_chunk = chunk_floats / (size_t)J.n_cols;
            size_t rc = (rows_per_chunk / col_step) * col_step;
            if (rc == 0) rc = col_step;
            J.row_chunk = rc;
            std::cout << "  chunking " << t.name << " rows " << J.n_rows
                      << " by " << J.row_chunk << " (cols " << J.n_cols << ")" << std::endl;
        }

        // Profile each chunk: per-cluster mean-|y| score
        for (int row_start = 0; row_start < J.n_rows; row_start += (int)J.row_chunk) {
            int row_end = std::min(row_start + (int)J.row_chunk, J.n_rows);
            int chunk_n_rows = row_end - row_start;
            size_t chunk_n = (size_t)chunk_n_rows * (size_t)J.n_cols;
            size_t chunk_offset_elems = (size_t)row_start * (size_t)J.n_cols;
            size_t cluster_offset = chunk_offset_elems / CLUSTER_SIZE;
            size_t chunk_nc = (chunk_n + CLUSTER_SIZE - 1) / CLUSTER_SIZE;

            GGUFTensor t_chunk = t;
            t_chunk.offset += elem_offset_bytes(t.type, chunk_offset_elems);
            t_chunk.shape = { (uint64_t)chunk_n_rows, (uint64_t)J.n_cols };

            std::vector<float> w;
            if (!read_tensor_floats(parser, t_chunk, w)) {
                std::cerr << "Failed to read " << t.name << " chunk at row " << row_start << std::endl;
                return 1;
            }

            if (do_profile) {
                std::vector<float> score;
                profile_tensor(w.data(), chunk_n_rows, J.n_cols, xs, score);
                for (size_t i = 0; i < score.size(); ++i) {
                    to.clusters[cluster_offset + i].activation_freq = score[i];
                }
            } else {
                for (size_t i = 0; i < chunk_nc; ++i) {
                    to.clusters[cluster_offset + i].activation_freq = 0.0f;
                }
            }
        }
    }

    // GLOBAL BUDGET: sort all profiled, non-forced clusters by mean-|y| score.
    // Top hot_budget elements -> Q8_0, bottom cold_budget -> Q2_K, rest -> Q4_K.
    size_t total_params = 0;
    for (size_t ti = 0; ti < n_tensors; ++ti) {
        if (jobs[ti].quantizable) total_params += numel(parser.tensors()[ti]);
    }
    size_t hot_budget = std::min((size_t)(total_params * 0.20), (size_t)cli_hot_budget);
    size_t cold_budget = std::min((size_t)(total_params * 0.05), (size_t)cli_cold_budget);

    struct ScoreEnt { float score; size_t ti, ci, elems; };
    std::vector<ScoreEnt> ent;
    for (size_t ti = 0; ti < n_tensors; ++ti) {
        const TensorJob& J = jobs[ti];
        if (!J.quantizable || !J.profiled) continue;
        if (J.min_bits == J.max_bits) continue;  // forced override, not budgeted
        size_t n = numel(parser.tensors()[ti]);
        for (size_t ci = 0; ci < J.nc; ++ci) {
            size_t e0 = ci * CLUSTER_SIZE;
            size_t elems = std::min((size_t)CLUSTER_SIZE, n - e0);
            ent.push_back({outs[ti].clusters[ci].activation_freq, ti, ci, elems});
        }
    }
    std::sort(ent.begin(), ent.end(), [](const ScoreEnt& a, const ScoreEnt& b) { return a.score > b.score; });
    for (auto& e : ent) outs[e.ti].clusters[e.ci].quant_bits = 4;  // warm default
    size_t acc = 0, top = 0;
    for (; top < ent.size() && acc < hot_budget; ++top) {
        acc += ent[top].elems;
        outs[ent[top].ti].clusters[ent[top].ci].quant_bits = 8;
    }
    size_t bacc = 0;
    for (size_t j = ent.size(); j > top && bacc < cold_budget; ) {
        --j;
        bacc += ent[j].elems;
        outs[ent[j].ti].clusters[ent[j].ci].quant_bits = 2;
    }
    std::cout << "Budget: total_params=" << total_params
              << " hot_budget=" << hot_budget << " cold_budget=" << cold_budget
              << " sorted_clusters=" << ent.size() << std::endl;

    // Per-tensor dominant class -> output type (forced overrides via min/max_bits)
    for (size_t ti = 0; ti < n_tensors; ++ti) {
        const GGUFTensor& t = parser.tensors()[ti];
        TensorJob& J = jobs[ti];
        TensorOut& to = outs[ti];
        if (!J.float_path) continue;  // passthrough type set in pass 2
        if (!J.quantizable) {
            // Keep F32 source tensors (biases, norms) as F32 — llama.cpp's
            // element-wise add path requires matching operand types/shapes.
            to.quant_type = (t.type == GGMLType::F32 || J.f32_tensor) ? (uint32_t)GGMLType::F32 : (uint32_t)GGMLType::F16;
            continue;
        }
        int n_cold = 0, n_warm = 0, n_hot = 0;
        size_t nfull = numel(t);
        for (size_t c = 0; c < J.nc; ++c) {
            int bits = to.clusters[c].quant_bits;
            if (bits < J.min_bits) bits = J.min_bits;
            if (bits > J.max_bits) bits = J.max_bits;
            to.clusters[c].quant_bits = (uint8_t)bits;
            size_t e0 = c * CLUSTER_SIZE;
            size_t el = std::min((size_t)CLUSTER_SIZE, nfull - e0);
            if (bits == 2) { n_cold++; total_cold += el; }
            else if (bits == 8) { n_hot++; total_hot += el; }
            else { n_warm++; total_warm += el; }
        }
        int bits = 4; // warm default
        if (n_hot >= n_warm && n_hot >= n_cold) bits = 8;
        else if (n_cold > n_warm && n_cold > n_hot) bits = 2;
        if (bits < J.min_bits) bits = J.min_bits;
        if (bits > J.max_bits) bits = J.max_bits;
        GGMLType out_type = (bits == 8) ? GGMLType::Q8_0 : (bits == 2) ? GGMLType::Q2_K : GGMLType::Q4_K;
        if (out_type == GGMLType::Q8_0 && !J.q8_ok) out_type = GGMLType::F16;
        if ((out_type == GGMLType::Q4_K || out_type == GGMLType::Q2_K) && !J.kquant_ok)
            out_type = J.q8_ok ? GGMLType::Q8_0 : GGMLType::F16;
        to.quant_type = (uint32_t)out_type;
    }

    // PASS 2: quantize each tensor to its decided type, stream to temp file
    for (size_t ti = 0; ti < n_tensors; ++ti) {
        const GGUFTensor& t = parser.tensors()[ti];
        TensorJob& J = jobs[ti];
        TensorOut& to = outs[ti];
        if ((ti % 10) == 0) {
            std::cout << "Quantizing [" << (ti + 1) << "/" << n_tensors << "] " << t.name << std::endl;
        }

        if (!J.float_path) {
            // Passthrough: copy raw tensor bytes, keep original GGUF type
            to.quant_type = (uint32_t)t.type;
            to.data_offset = (uint64_t)ftell(data_tmp);

            size_t data_len = 0;
            if (ti + 1 < n_tensors) {
                const GGUFTensor& next_t = parser.tensors()[ti + 1];
                data_len = (size_t)(next_t.offset - t.offset);
            } else {
                data_len = parser.full_file_size() - parser.data_offset() - t.offset;
            }

            std::vector<uint8_t> block(data_len);
            if (data_len > 0) {
                if (!parser.read_tensor(t, block.data(), data_len)) {
                    std::cerr << "Failed to read " << t.name << std::endl;
                    return 1;
                }
                fwrite(block.data(), 1, data_len, data_tmp);
            }
            to.data_bytes = data_len;
            continue;
        }

        GGMLType out_type = (GGMLType)to.quant_type;
        to.data_offset = (uint64_t)ftell(data_tmp);
        uint64_t tensor_bytes = 0;
        for (int row_start = 0; row_start < J.n_rows; row_start += (int)J.row_chunk) {
            int row_end = std::min(row_start + (int)J.row_chunk, J.n_rows);
            int chunk_n_rows = row_end - row_start;
            size_t chunk_offset_elems = (size_t)row_start * (size_t)J.n_cols;

            GGUFTensor t_chunk = t;
            t_chunk.offset += elem_offset_bytes(t.type, chunk_offset_elems);
            t_chunk.shape = { (uint64_t)chunk_n_rows, (uint64_t)J.n_cols };

            std::vector<float> w;
            if (!read_tensor_floats(parser, t_chunk, w)) {
                std::cerr << "Failed to read " << t.name << " chunk at row " << row_start << std::endl;
                return 1;
            }
            std::vector<uint8_t> qbuf;
            size_t wrote = quantize_to_type(w, out_type, qbuf);
            if (wrote == 0 && !w.empty()) {
                std::cerr << "Failed to quantize " << t.name << std::endl;
                return 1;
            }
            if (wrote > 0) fwrite(qbuf.data(), 1, wrote, data_tmp);
            tensor_bytes += wrote;
        }
        to.data_bytes = tensor_bytes;
    }

    // Assemble GGUF: KV section copied verbatim from source, new tensor
    // descriptors, then tensor data already in final block layout in the
    // temp stream.
    fflush(data_tmp);
    fseek(data_tmp, 0, SEEK_SET);

    const auto& src_buf = parser.head_buffer();

    // Walk source GGUF header to find where the KV section ends
    size_t off = 0;
    off += 4; // magic
    off += 4; // version
    off += 8; // tensor_count
    uint64_t src_kv = *(const uint64_t*)(src_buf.data() + off); off += 8;
    for (uint64_t i = 0; i < src_kv; ++i) {
        uint64_t klen = *(const uint64_t*)(src_buf.data() + off); off += 8 + klen;
        uint32_t vt = *(const uint32_t*)(src_buf.data() + off); off += 4;
        switch (vt) {
            case 0: case 1: case 7: off += 1; break;
            case 2: case 3: off += 2; break;
            case 4: case 5: case 6: off += 4; break;
            case 10: case 11: case 12: off += 8; break;
            case 8: { uint64_t sl = *(const uint64_t*)(src_buf.data() + off); off += 8 + sl; } break;
            case 9: {
                uint32_t at = *(const uint32_t*)(src_buf.data() + off); off += 4;
                uint64_t al = *(const uint64_t*)(src_buf.data() + off); off += 8;
                if (at == 8) {
                    for (uint64_t j = 0; j < al; ++j) {
                        uint64_t sl = *(const uint64_t*)(src_buf.data() + off); off += 8 + sl;
                    }
                } else {
                    size_t es = (at <= 1 || at == 7) ? 1 : (at <= 3) ? 2 : (at <= 6) ? 4 : 8;
                    off += al * es;
                }
            } break;
            default: break;
        }
    }
    size_t kv_end = off;

    // Tensor descriptor section size
    size_t desc_size = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
        const GGUFTensor& st = parser.tensors()[i];
        desc_size += 8 + st.name.size() + 4 + st.shape.size() * 8 + 4 + 8;
    }
    size_t header_size = kv_end + desc_size;
    size_t data_start = (header_size + 31) & ~size_t(31);

    // Per-tensor data offsets relative to data section start, 32-aligned
    std::vector<uint64_t> data_offs(outs.size());
    uint64_t cur = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
        data_offs[i] = cur;
        cur += outs[i].data_bytes;
        cur = (cur + 31) & ~uint64_t(31);
    }

    FILE* f = fopen(out_path, "wb");
    if (!f) {
        std::cerr << "Failed to open output " << out_path << std::endl;
        fclose(data_tmp);
        std::remove(data_tmp_path.c_str());
        return 1;
    }

    // KV section verbatim from source (magic, version, tensor_count, kv_count, KVs)
    fwrite(src_buf.data(), 1, kv_end, f);

    // Tensor descriptors
    for (size_t i = 0; i < outs.size(); ++i) {
        const GGUFTensor& st = parser.tensors()[i];
        uint64_t nl = st.name.size();
        fwrite(&nl, 8, 1, f);
        fwrite(st.name.data(), 1, nl, f);
        uint32_t nd = (uint32_t)st.shape.size();
        fwrite(&nd, 4, 1, f);
        for (auto dd : st.shape) { fwrite(&dd, 8, 1, f); }
        fwrite(&outs[i].quant_type, 4, 1, f);
        fwrite(&data_offs[i], 8, 1, f);
    }

    // Pad to data_start
    {
        long pos = ftell(f);
        long pad = (long)data_start - pos;
        if (pad > 0) {
            std::vector<uint8_t> zeros(pad, 0);
            fwrite(zeros.data(), 1, pad, f);
        }
    }

    // Tensor data: stream each tensor's span from temp, pad to 32
    const size_t COPY_BUF = 8 * 1024 * 1024;
    std::vector<uint8_t> copy_buf(COPY_BUF);
    for (size_t i = 0; i < outs.size(); ++i) {
        fseek(data_tmp, (long)outs[i].data_offset, SEEK_SET);
        uint64_t remaining = outs[i].data_bytes;
        while (remaining > 0) {
            size_t want = remaining < COPY_BUF ? (size_t)remaining : COPY_BUF;
            size_t got = fread(copy_buf.data(), 1, want, data_tmp);
            if (got == 0) break;
            fwrite(copy_buf.data(), 1, got, f);
            remaining -= got;
        }
        long p = ftell(f);
        long npad = (32 - (p % 32)) % 32;
        if (npad > 0) {
            std::vector<uint8_t> zeros(npad, 0);
            fwrite(zeros.data(), 1, npad, f);
        }
    }

    fclose(data_tmp);
    fclose(f);
    std::remove(data_tmp_path.c_str());

    std::cout << "GGUF output written: " << out_path << std::endl;

    // Validation: check GGUF magic
    FILE* v = fopen(out_path, "rb");
    if (v) {
        char magic[4] = {0};
        if (fread(magic, 1, 4, v) == 4 && std::memcmp(magic, "GGUF", 4) == 0) {
            std::cout << "Validation: GGUF magic OK, tensors=" << outs.size() << std::endl;
        } else {
            std::cerr << "Validation failed: bad GGUF magic" << std::endl;
        }
        fclose(v);
    }

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    size_t in_bytes = file_size(in_path);
    size_t out_bytes = file_size(out_path);
    std::cout << "Input:  " << in_bytes << " bytes" << std::endl;
    std::cout << "Output: " << out_bytes << " bytes" << std::endl;
    size_t total_clusters = total_hot + total_warm + total_cold;
    if (total_clusters > 0) {
        std::cout << "Cluster distribution:" << std::endl;
        std::cout << "  hot  (8-bit): " << total_hot  << " (" << (100.0 * total_hot  / total_clusters) << "%)" << std::endl;
        std::cout << "  warm (4-bit): " << total_warm << " (" << (100.0 * total_warm / total_clusters) << "%)" << std::endl;
        std::cout << "  cold (2-bit): " << total_cold << " (" << (100.0 * total_cold / total_clusters) << "%)" << std::endl;
    } else if (structural_q4) {
        std::cout << "Structural Q4 pass-through; no variable-rate quantization performed." << std::endl;
    }
    std::cout << "Compile time: " << sec << " s" << std::endl;
    std::cout << (structural_q4 ? "Note: input was not F32; structural Q4 pass-through mode." : "") << std::endl;

    return 0;
}
