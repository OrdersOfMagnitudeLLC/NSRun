#include "nsm_loader.h"
#include "nsm.h"
#include "vendor/ns_dequant.h"
#include "ns_q4k_quant.h"
#include "ns_repack.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <algorithm>
#include <numeric>
#include <zlib.h>
#include <unistd.h>
#include <fcntl.h>

static const size_t CLUSTER_SIZE = 256;
static const size_t PANEL_SIZE = 8 * CLUSTER_SIZE; // 2048 values = 8 rows x 256 columns

static size_t numel(const GGUFTensor& t) {
    size_t n = 1;
    for (auto d : t.shape) n *= (size_t)d;
    return n;
}

static std::string prev_name(const std::string& name) {
    std::regex re("^blk\\.(\\d+)\\.(.+)$");
    std::smatch m;
    if (std::regex_match(name, m, re)) {
        int l = std::stoi(m[1].str());
        if (l > 0) return "blk." + std::to_string(l - 1) + "." + m[2].str();
    }
    return "";
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Return the unpacked packed size for a tensor.  For passthrough (nc==1, raw on disk)
// this equals nt.data_bytes; for per-cluster compressed data it is nc * unit_size.
// use_panels is set when the Q4_K/Q6_K/Q8_0 layout is 2D panel-ordered.
static size_t packed_size_for_nt(const NSMTensor& nt, const GGUFTensor& shape, bool& use_panels, size_t& unit_size) {
    size_t n = numel(shape);
    bool is_2d = shape.shape.size() >= 2;
    size_t n_cols = is_2d ? (size_t)shape.shape[0] : n;
    size_t n_rows = is_2d ? (size_t)shape.shape[1] : 1;
    size_t full_blocks = (n + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
    size_t n_blocks_per_row = (n_cols + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
    size_t n_row_groups = (n_rows + 7) / 8;
    size_t n_panels = n_row_groups * n_blocks_per_row;

    size_t panel_bytes = 0;
    size_t non_panel_bytes = 0;
    use_panels = false;
    unit_size = 0;
    switch (nt.quant_type) {
        case 12: { // Q4_K
            panel_bytes = n_panels * sizeof(ns_q4_Kx8);
            non_panel_bytes = full_blocks * sizeof(block_q4_K);
            unit_size = sizeof(block_q4_K);
            break;
        }
        case 14: { // Q6_K
            panel_bytes = n_panels * sizeof(ns_q6_Kx8);
            non_panel_bytes = full_blocks * sizeof(block_q6_K);
            unit_size = sizeof(block_q6_K);
            break;
        }
        case 8: { // Q8_0
            unit_size = 8 * sizeof(block_q8_0);
            panel_bytes = n_panels * unit_size;
            non_panel_bytes = full_blocks * unit_size;
            break;
        }
        case 1: // F16
            non_panel_bytes = n * sizeof(uint16_t);
            unit_size = CLUSTER_SIZE * sizeof(uint16_t);
            break;
        case 0: // F32
            non_panel_bytes = n * sizeof(float);
            unit_size = CLUSTER_SIZE * sizeof(float);
            break;
        default:
            return 0;
    }

    if (nt.data_bytes == panel_bytes && panel_bytes > 0) {
        use_panels = true;
        return panel_bytes;
    }
    if (nt.data_bytes == non_panel_bytes && non_panel_bytes > 0) {
        use_panels = false;
        return non_panel_bytes;
    }
    // nt.data_bytes is compressed; infer from cluster count.
    if (nt.n_clusters == full_blocks) {
        use_panels = false;
        return non_panel_bytes;
    }
    if (nt.n_clusters == n_panels && panel_bytes > 0) {
        use_panels = true;
        return panel_bytes;
    }
    // After the use_panels checks above, set the per-cluster unit size correctly.
    switch (nt.quant_type) {
        case 12: // Q4_K
            unit_size = use_panels ? sizeof(ns_q4_Kx8) : sizeof(block_q4_K);
            break;
        case 14: // Q6_K
            unit_size = use_panels ? sizeof(ns_q6_Kx8) : sizeof(block_q6_K);
            break;
        case 8:  // Q8_0
            unit_size = use_panels ? 64 * sizeof(block_q8_0) : 8 * sizeof(block_q8_0);
            break;
    }

    return 0;
}

static int zlib_uncompress(const std::vector<uint8_t>& in, std::vector<uint8_t>& out) {
    if (in.empty()) { out.clear(); return Z_STREAM_END; }
    z_stream s;
    std::memset(&s, 0, sizeof(s));
    if (inflateInit(&s) != Z_OK) return Z_STREAM_ERROR;
    s.avail_in = (uInt)in.size();
    s.next_in = (Bytef*)in.data();
    out.resize(in.size());
    s.avail_out = (uInt)out.size();
    s.next_out = out.data();
    int r;
    do {
        r = inflate(&s, Z_NO_FLUSH);
        if (r == Z_STREAM_END) break;
        if (r != Z_OK) { inflateEnd(&s); return r; }
        if (s.avail_out == 0) {
            size_t old = out.size();
            out.resize(old * 2);
            s.next_out = out.data() + old;
            s.avail_out = (uInt)(out.size() - old);
        }
    } while (r != Z_STREAM_END);
    out.resize(s.total_out);
    inflateEnd(&s);
    return r;
}

static bool read_exact_fd(int fd, void* dst, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::read(fd, p + got, n - got);
        if (r <= 0) return false;
        got += (size_t)r;
    }
    return true;
}


static void dequantize_block_q4_K(const block_q4_K* b, float* out, size_t n) {
    const uint8_t* q = b->qs;
    float d = GGML_FP16_TO_FP32(b->d);
    float min = GGML_FP16_TO_FP32(b->dmin);
    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < (int)n; j += 64) {
        get_scale_min_k4(is + 0, b->scales, &sc, &m);
        float d1 = d * sc; float m1 = min * m;
        get_scale_min_k4(is + 1, b->scales, &sc, &m);
        float d2 = d * sc; float m2 = min * m;
        int limit1 = std::min(32, (int)n - j);
        for (int l = 0; l < limit1; ++l) out[j + l] = d1 * (q[l] & 0xF) - m1;
        int limit2 = std::min(32, (int)n - j - 32);
        for (int l = 0; l < limit2; ++l) out[j + 32 + l] = d2 * (q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}

namespace nsm {


bool nsm_dequant_tensor(int fd, const NSMTensor& nt, size_t cl_offset, const GGUFTensor& shape, const uint8_t* prev_base, GGMLType prev_type, size_t prev_n, uint8_t* out_buf, size_t out_bytes, const uint8_t* layer_base, size_t layer_base_offset) {
    size_t n = numel(shape);
    bool is_2d = shape.shape.size() >= 2;

    size_t n_cols = is_2d ? (size_t)shape.shape[0] : n;
    size_t n_rows = is_2d ? (size_t)shape.shape[1] : 1;
    size_t full_blocks = (n + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
    size_t n_blocks_per_row = (n_cols + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
    size_t n_row_groups = (n_rows + 7) / 8;
    size_t n_panels = n_row_groups * n_blocks_per_row;

    bool use_panels = false;
    size_t unit_size = 0;
    size_t expected_bytes = packed_size_for_nt(nt, shape, use_panels, unit_size);
    if (expected_bytes == 0) {
        std::cerr << "nsm_dequant_tensor: cannot determine packed size for " << shape.name
                  << " (quant_type=" << nt.quant_type << ")" << std::endl;
        return false;
    }
    if (out_bytes != expected_bytes) {
        std::cerr << "nsm_dequant_tensor: output buffer size mismatch for " << shape.name
                  << " (expected " << expected_bytes << ", got " << out_bytes << ")" << std::endl;
        return false;
    }

    (void)cl_offset;  // cluster metadata now pre-loaded in shape.clusters
    if (shape.clusters.size() != nt.n_clusters) {
        std::cerr << "nsm_dequant_tensor: cluster metadata missing for " << shape.name
                  << " (expected " << nt.n_clusters << ", got " << shape.clusters.size() << ")" << std::endl;
        return false;
    }
    const std::vector<NSMCluster>& clusters = shape.clusters;

    size_t nc = nt.n_clusters;
    if (nc == 1 && clusters[0].data_bytes == nt.data_bytes && nt.data_bytes == expected_bytes) {
        const NSMCluster& cl = clusters[0];
        std::vector<uint8_t> packed(cl.data_bytes);
        if (cl.data_bytes > 0) {
            size_t abs_off = nt.data_offset + cl.data_offset;
            if (layer_base) {
                size_t rel_off = abs_off - layer_base_offset;
                std::memcpy(packed.data(), layer_base + rel_off, cl.data_bytes);
            } else {
                ::lseek(fd, (off_t)abs_off, SEEK_SET);
                if (!read_exact_fd(fd, packed.data(), cl.data_bytes)) {
                    std::cerr << "nsm_dequant_tensor: failed to read packed data for " << shape.name << std::endl;
                    return false;
                }
            }
        }
        std::memcpy(out_buf, packed.data(), cl.data_bytes);
        return true;
    }

    if (nt.quant_type != 0 && nt.quant_type != 1 && nc != (use_panels ? n_panels : full_blocks)) {
        std::cerr << "nsm_dequant_tensor: cluster count mismatch for " << shape.name
                  << ": expected " << (use_panels ? n_panels : full_blocks)
                  << ", got " << nc << std::endl;
        return false;
    }

    bool large = (nt.data_bytes > (size_t)(1024 * 1024));
    size_t expected_total = 0, actual_total = 0, mismatches = 0;
    bool has_prev = (prev_base != nullptr);
    size_t prev_nc = has_prev ? (prev_n + CLUSTER_SIZE - 1) / CLUSTER_SIZE : 0;

    for (uint32_t c = 0; c < nc; ++c) {
        size_t cluster_off, cluster_len;
        if (use_panels) {
            cluster_off = (size_t)c * PANEL_SIZE;
            cluster_len = (cluster_off >= n) ? 0 : std::min(PANEL_SIZE, n - cluster_off);
        } else {
            cluster_off = (size_t)c * CLUSTER_SIZE;
            cluster_len = (cluster_off >= n) ? 0 : std::min(CLUSTER_SIZE, n - cluster_off);
        }
        const NSMCluster& cl = clusters[c];

        std::vector<uint8_t> packed(cl.data_bytes);
        if (cl.data_bytes > 0) {
            size_t abs_off = nt.data_offset + cl.data_offset;
            if (layer_base) {
                size_t rel_off = abs_off - layer_base_offset;
                std::memcpy(packed.data(), layer_base + rel_off, cl.data_bytes);
            } else {
                ::lseek(fd, (off_t)abs_off, SEEK_SET);
                if (!read_exact_fd(fd, packed.data(), cl.data_bytes)) {
                    std::cerr << "nsm_dequant_tensor: failed to read packed data for " << shape.name << " cluster " << c << std::endl;
                    return false;
                }
            }
        }

        std::vector<uint8_t> bytes;
        int zr = zlib_uncompress(packed, bytes);
        if (zr != Z_STREAM_END) {
            std::cerr << "nsm_dequant_tensor: zlib decompression failed for " << shape.name << " cluster " << c << " rc=" << zr << std::endl;
            return false;
        }

        float scale = cl.scale;
        if (scale == 0.0f) scale = 1.0f;

        size_t expected_payload = 0;
        if (nt.quant_type == 0 || nt.quant_type == 1) {
            size_t elem_size = (nt.quant_type == 0) ? sizeof(float) : sizeof(uint16_t);
            expected_payload = cluster_len * elem_size;
            if (bytes.size() < expected_payload) {
                std::cerr << "nsm_dequant_tensor: " << (nt.quant_type == 0 ? "F32" : "F16")
                          << " data too short for " << shape.name << " cluster " << c
                          << " (expected " << expected_payload << ", got " << bytes.size() << ")" << std::endl;
                return false;
            }
            std::memcpy(out_buf + cl.data_offset, bytes.data(), expected_payload);
        } else if (cl.quant_bits == 2) {
            if (use_panels) {
                std::cerr << "nsm_dequant_tensor: Q2 residuals not supported for 2D " << shape.name << std::endl;
                return false;
            }
            expected_payload = (cluster_len + 3) / 4;
            if (bytes.size() < expected_payload) {
                std::cerr << "nsm_dequant_tensor: Q2 residual data too short for " << shape.name << " cluster " << c
                          << " (expected " << expected_payload << ", got " << bytes.size() << ")" << std::endl;
                return false;
            }
            float block_buf[CLUSTER_SIZE];
            std::memset(block_buf, 0, sizeof(block_buf));
            if (has_prev) {
                if (c < prev_nc) {
                    size_t prev_cluster_len = std::min(CLUSTER_SIZE, prev_n - cluster_off);
                    if (prev_type == GGMLType::Q4_K) {
                        const block_q4_K* prev_block = (const block_q4_K*)(prev_base + c * sizeof(block_q4_K));
                        dequantize_block_q4_K(prev_block, block_buf, prev_cluster_len);
                    } else if (prev_type == GGMLType::F32) {
                        const float* prev_f32 = (const float*)(prev_base + c * CLUSTER_SIZE * sizeof(float));
                        block_q4_K tmp;
                        quantize_block_q4_K(prev_f32, &tmp, prev_cluster_len);
                        dequantize_block_q4_K(&tmp, block_buf, prev_cluster_len);
                    }
                }
            }
            for (size_t k = 0; k < cluster_len; ++k) {
                uint8_t b = bytes[k / 4];
                uint8_t nibble = (b >> ((k % 4) * 2)) & 0x3;
                int8_t q2 = (int8_t)nibble - 1;
                block_buf[k] += (float)q2 * scale;
            }
            for (size_t k = cluster_len; k < CLUSTER_SIZE; ++k) block_buf[k] = 0.0f;
            quantize_block_q4_K(block_buf, (block_q4_K*)(out_buf + c * sizeof(block_q4_K)), cluster_len);
        } else {
            expected_payload = unit_size;
            if (bytes.size() < expected_payload) {
                std::cerr << "nsm_dequant_tensor: packed data too short for " << shape.name << " cluster " << c
                          << " (expected " << expected_payload << ", got " << bytes.size() << ")" << std::endl;
                return false;
            }
            std::memcpy(out_buf + c * unit_size, bytes.data(), unit_size);
        }

        expected_total += expected_payload;
        actual_total += bytes.size();
        if (bytes.size() != expected_payload) mismatches++;
    }


    return true;
}

bool load_nsm_weights(const std::string& nsm_path,
                      const std::vector<GGUFTensor>& shapes,
                      AlignedVectorU8& out_q4k,
                      std::vector<GGUFTensor>& out_tensors,
                      GGUFParser* model) {
    int fd = ::open(nsm_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "nsm_loader: cannot open " << nsm_path << std::endl;
        return false;
    }


    // Read and validate header
    NSMHeader hdr;
    if (!read_exact_fd(fd, &hdr, sizeof(hdr))) {
        std::cerr << "nsm_loader: failed to read header" << std::endl;
        ::close(fd);
        return false;
    }
    if (hdr.magic != NSM_MAGIC) {
        std::cerr << "nsm_loader: bad magic " << std::hex << hdr.magic << std::endl;
        ::close(fd);
        return false;
    }
    if (hdr.version != 2) {
        std::cerr << "nsm_loader: unsupported version " << hdr.version << " (expected 2)" << std::endl;
        ::close(fd);
        return false;
    }

    // Read tensor map
    std::vector<NSMTensor> nsm_tensors(hdr.n_tensors);
    for (uint32_t i = 0; i < hdr.n_tensors; ++i) {
        if (!read_exact_fd(fd, &nsm_tensors[i], sizeof(NSMTensor))) {
            std::cerr << "nsm_loader: failed to read NSMTensor " << i << std::endl;
            ::close(fd);
            return false;
        }
    }

    // Map NSM tensor name -> index, and compute cluster map offset for each
    std::unordered_map<std::string, uint32_t> nsm_name_to_idx;
    std::vector<size_t> cl_offsets(hdr.n_tensors);
    size_t cluster_map_start = sizeof(NSMHeader) + hdr.n_tensors * sizeof(NSMTensor);
    size_t cursor = cluster_map_start;
    for (uint32_t i = 0; i < hdr.n_tensors; ++i) {
        std::string name(nsm_tensors[i].name);
        // Trim null bytes
        size_t z = name.find('\0');
        if (z != std::string::npos) name.resize(z);
        nsm_name_to_idx[name] = i;
        cl_offsets[i] = cursor;
        cursor += (size_t)nsm_tensors[i].n_clusters * sizeof(NSMCluster);
    }


    // Layer offset indexing pass: record each transformer layer's contiguous disk span.
    if (model) {
        model->nsm_layer_offsets.assign(hdr.n_layers, 0);
        model->nsm_layer_bytes.assign(hdr.n_layers, 0);
        std::vector<uint64_t> layer_min(hdr.n_layers, (uint64_t)(-1));
        std::vector<uint64_t> layer_max(hdr.n_layers, 0);
        std::regex re_layer("^blk\\.(\\d+)\\..+$");
        for (uint32_t i = 0; i < hdr.n_tensors; ++i) {
            std::string name(nsm_tensors[i].name);
            size_t z = name.find('\0');
            if (z != std::string::npos) name.resize(z);
            std::smatch m;
            if (std::regex_match(name, m, re_layer)) {
                int l = std::stoi(m[1].str());
                if (l >= 0 && (uint32_t)l < hdr.n_layers) {
                    uint64_t start = nsm_tensors[i].data_offset;
                    uint64_t end   = start + nsm_tensors[i].data_bytes;
                    if (start < layer_min[l]) layer_min[l] = start;
                    if (end > layer_max[l])   layer_max[l] = end;
                }
            }
        }
        for (uint32_t l = 0; l < hdr.n_layers; ++l) {
            if (layer_min[l] != (uint64_t)(-1)) {
                model->nsm_layer_offsets[l] = (size_t)layer_min[l];
                model->nsm_layer_bytes[l]   = (size_t)(layer_max[l] - layer_min[l]);
            }
        }
        model->nsm_layer_index.resize(hdr.n_layers);
        for (uint32_t l = 0; l < hdr.n_layers; ++l) {
            model->nsm_layer_index[l].data_offset = model->nsm_layer_offsets[l];
            model->nsm_layer_index[l].data_bytes  = model->nsm_layer_bytes[l];
        }
    }


    out_q4k.clear();
    out_tensors.clear();
    out_tensors.resize(shapes.size());

    // Map expected shapes by name
    std::unordered_map<std::string, size_t> shape_to_idx;
    for (size_t i = 0; i < shapes.size(); ++i) shape_to_idx[shapes[i].name] = i;

    // Per-shape metadata for Q2 dependency ordering
    std::vector<size_t> shape_n(shapes.size(), 0);
    std::vector<int> shape_layer(shapes.size(), -1);
    std::regex re_layer("^blk\\.(\\d+)\\..+$");

    // Precompute total output bytes and reserve the output buffer.
    // Each keep_in_ram tensor is stored exactly as its nt.data_bytes.
    size_t total_bytes = 0;
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto it = nsm_name_to_idx.find(shapes[i].name);
        if (it == nsm_name_to_idx.end()) continue;
        const NSMTensor& nt = nsm_tensors[it->second];
        size_t n = numel(shapes[i]);
        shape_n[i] = n;
        std::smatch lm;
        if (std::regex_match(shapes[i].name, lm, re_layer)) {
            shape_layer[i] = std::stoi(lm[1].str());
        }
        bool keep_in_ram = (shapes[i].name == "token_embd.weight" ||
                           shapes[i].name == "output.weight" ||
                           shapes[i].name == "lm_head.weight" ||
                           ends_with(shapes[i].name, "attn_norm.weight") ||
                           ends_with(shapes[i].name, "ffn_norm.weight") ||
                           ends_with(shapes[i].name, "output_norm.weight") ||
                           ends_with(shapes[i].name, ".bias"));
        if (keep_in_ram) {
            bool up; size_t us;
            size_t packed = packed_size_for_nt(nt, shapes[i], up, us);
            total_bytes += packed + 64; // 64-byte alignment per tensor
        }
    }
    out_q4k.reserve(total_bytes);

    // Process in numeric layer order so Q2 residuals can be resolved immediately
    std::vector<size_t> order(shapes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (shape_layer[a] != shape_layer[b]) return shape_layer[a] < shape_layer[b];
        return a < b;
    });

    // Pack one shape at a time, freeing its Q4_K buffer before moving to the next.
    for (size_t pos = 0; pos < order.size(); ++pos) {
        size_t si = order[pos];
        const GGUFTensor& shape = shapes[si];
        auto it = nsm_name_to_idx.find(shape.name);
        if (it == nsm_name_to_idx.end()) {
            std::cerr << "nsm_loader: tensor not found in NSM: " << shape.name << std::endl;
            ::close(fd);
            return false;
        }
        uint32_t nti = it->second;
        const NSMTensor& nt = nsm_tensors[nti];
        size_t n = numel(shape);
        shape_n[si] = n;

        bool keep_in_ram = (shape.name == "token_embd.weight" ||
                            shape.name == "output.weight" ||
                            shape.name == "lm_head.weight" ||
                            ends_with(shape.name, "attn_norm.weight") ||
                            ends_with(shape.name, "ffn_norm.weight") ||
                            ends_with(shape.name, "output_norm.weight") ||
                            ends_with(shape.name, ".bias"));
        if (!keep_in_ram) {
            // Metadata only; data lives on disk and is streamed per layer.
            out_tensors[si] = shape;
            out_tensors[si].offset = nt.data_offset;
            out_tensors[si].cl_offset = cl_offsets[nti];
            out_tensors[si].nsm_n_clusters = nt.n_clusters;
            out_tensors[si].nsm_data_bytes = nt.data_bytes;
            out_tensors[si].type = static_cast<GGMLType>(nt.quant_type);
            out_tensors[si].on_disk = true;
            out_tensors[si].clusters.resize(nt.n_clusters);
            ::lseek(fd, (off_t)cl_offsets[nti], SEEK_SET);
            if (!read_exact_fd(fd, out_tensors[si].clusters.data(), nt.n_clusters * sizeof(NSMCluster))) {
                std::cerr << "nsm_loader: failed to read cluster map for " << shape.name << std::endl;
                ::close(fd);
                return false;
            }
            continue;
        }

        GGUFTensor out_t = shape;
        out_t.type = static_cast<GGMLType>(nt.quant_type);

        while (out_q4k.size() % 64) out_q4k.push_back(0);

        out_t.offset = out_q4k.size();

        out_t.clusters.resize(nt.n_clusters);
        ::lseek(fd, (off_t)cl_offsets[nti], SEEK_SET);
        if (!read_exact_fd(fd, out_t.clusters.data(), nt.n_clusters * sizeof(NSMCluster))) {
            std::cerr << "nsm_loader: failed to read cluster map for " << shape.name << std::endl;
            ::close(fd);
            return false;
        }

        bool up; size_t us;
        size_t q4k_buf_bytes = packed_size_for_nt(nt, shape, up, us);
        std::vector<uint8_t> q4k_buf(q4k_buf_bytes, 0);

        std::string pn = prev_name(shape.name);
        const uint8_t* prev_base = nullptr;
        GGMLType prev_type = GGMLType::F32;
        size_t prev_n = 0;
        if (!pn.empty()) {
            auto pit = shape_to_idx.find(pn);
            if (pit != shape_to_idx.end()) {
                size_t prev_idx = pit->second;
                if (out_tensors[prev_idx].type == GGMLType::Q4_K ||
                    out_tensors[prev_idx].type == GGMLType::F32) {
                    prev_base = out_q4k.data() + out_tensors[prev_idx].offset;
                    prev_type = out_tensors[prev_idx].type;
                    prev_n = shape_n[prev_idx];
                }
            }
        }

        if (!nsm_dequant_tensor(fd, nt, cl_offsets[nti], out_t,
                                prev_base, prev_type, prev_n,
                                q4k_buf.data(), q4k_buf.size())) {
            ::close(fd);
            return false;
        }

        out_q4k.insert(out_q4k.end(), q4k_buf.begin(), q4k_buf.end());
        out_tensors[si] = out_t;
    }


    // Release intermediate metadata vectors (Q4_K output is already in out_q4k)
    std::vector<size_t>().swap(order);
    std::vector<size_t>().swap(shape_n);
    std::vector<int>().swap(shape_layer);
    std::unordered_map<std::string, size_t>().swap(shape_to_idx);


    if (model) model->nsm_fd = fd;
    return true;
}

} // namespace nsm
