// kv_alloc.cpp — Bare KV allocator with actual memory (no LLM, no inference)
//
// Allocates the compressed KV budget, writes/reads KV pairs with dynamic-range
// INT8 quantization (per-head scale, no clamping). No cross-layer grouping.
// Round-trip test validates accuracy.
//
// Build:  g++ -std=c++17 -O2 -o kv_alloc kv_alloc.cpp
// Run:    ./kv_alloc

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>

// ─── KVBox ─────────────────────────────────────────────────────────

struct KVBox {
    uint32_t n_layers;
    uint32_t n_groups;    // = n_layers (no cross-layer grouping)
    uint32_t n_kv_heads;
    uint32_t head_dim;

    uint64_t n_tokens;    // logical capacity
    uint64_t n_slots;     // physical slots = n_tokens
    size_t   head_bytes;  // per head: 2 floats (scale_k, scale_v) + 2*head_dim INT8
    size_t   slot_bytes;  // per slot: n_layers * n_kv_heads * head_bytes
    size_t   total_bytes; // total allocated
    uint8_t* buf;

    KVBox(uint32_t layers, uint32_t heads, uint32_t dim)
        : n_layers(layers)
        , n_groups(layers)  // no CLG
        , n_kv_heads(heads)
        , head_dim(dim)
        , n_tokens(0)
        , n_slots(0)
        , head_bytes(0)
        , slot_bytes(0)
        , total_bytes(0)
        , buf(nullptr)
    {}

    ~KVBox() {
        if (buf) free(buf);
    }

    void allocate(uint64_t tokens) {
        if (buf) { free(buf); buf = nullptr; }
        n_tokens = tokens;
        n_slots = tokens;
        if (n_slots == 0) n_slots = 1;
        // Per head: float scale_k (4B) + float scale_v (4B) + int8_t K[head_dim] + int8_t V[head_dim]
        head_bytes = 2 * sizeof(float) + (size_t)head_dim * 2;
        slot_bytes = (size_t)n_groups * n_kv_heads * head_bytes;
        total_bytes = (size_t)n_slots * slot_bytes;
        buf = (uint8_t*)malloc(total_bytes);
        assert(buf && "malloc failed");
        memset(buf, 0, total_bytes);
    }

    // Write a KV pair for (token_idx, layer, head)
    // data_fp16: pointer to head_dim * 2 fp16 values [K | V]
    void write_slot(uint64_t token_idx, uint32_t layer,
                    uint32_t head, const _Float16* data) {
        assert(buf && "not allocated");
        assert(layer < n_groups && head < n_kv_heads);
        uint64_t slot = token_idx % n_slots;
        size_t offset = slot * slot_bytes
                      + (size_t)(layer * n_kv_heads + head) * head_bytes;

        const _Float16* k_src = data;
        const _Float16* v_src = data + head_dim;

        // Compute dynamic scale
        float scale_k = 0.0f, scale_v = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) {
            float fk = fabsf((float)k_src[i]);
            float fv = fabsf((float)v_src[i]);
            if (fk > scale_k) scale_k = fk;
            if (fv > scale_v) scale_v = fv;
        }
        if (scale_k == 0.0f) scale_k = 1.0f;
        if (scale_v == 0.0f) scale_v = 1.0f;

        // Store scales
        float* scales = (float*)(buf + offset);
        scales[0] = scale_k;
        scales[1] = scale_v;

        // Quantize K
        int8_t* dst_k = (int8_t*)(buf + offset + 2 * sizeof(float));
        for (uint32_t i = 0; i < head_dim; i++) {
            float f = (float)k_src[i] / scale_k;
            int32_t q = (int32_t)lroundf(f * 127.0f);
            if (q > 127)  q = 127;
            if (q < -127) q = -127;
            dst_k[i] = (int8_t)q;
        }

        // Quantize V
        int8_t* dst_v = dst_k + head_dim;
        for (uint32_t i = 0; i < head_dim; i++) {
            float f = (float)v_src[i] / scale_v;
            int32_t q = (int32_t)lroundf(f * 127.0f);
            if (q > 127)  q = 127;
            if (q < -127) q = -127;
            dst_v[i] = (int8_t)q;
        }
    }

    // Read a KV pair for (token_idx, layer, head)
    // out_fp16: pointer to head_dim * 2 fp16 values [K | V]
    void read_slot(uint64_t token_idx, uint32_t layer,
                   uint32_t head, _Float16* out) {
        assert(buf && "not allocated");
        uint64_t slot = token_idx % n_slots;
        size_t offset = slot * slot_bytes
                      + (size_t)(layer * n_kv_heads + head) * head_bytes;

        // Read scales
        const float* scales = (const float*)(buf + offset);
        float scale_k = scales[0];
        float scale_v = scales[1];

        // Dequantize K
        const int8_t* src_k = (const int8_t*)(buf + offset + 2 * sizeof(float));
        _Float16* dst_k = out;
        for (uint32_t i = 0; i < head_dim; i++) {
            dst_k[i] = (_Float16)((float)src_k[i] * scale_k / 127.0f);
        }

        // Dequantize V
        const int8_t* src_v = src_k + head_dim;
        _Float16* dst_v = out + head_dim;
        for (uint32_t i = 0; i < head_dim; i++) {
            dst_v[i] = (_Float16)((float)src_v[i] * scale_v / 127.0f);
        }
    }
};

// ─── Round-trip test ────────────────────────────────────────────────

int main() {
    const uint32_t layers = 32;
    const uint32_t heads  = 8;
    const uint32_t dim    = 128;
    const uint64_t tokens = 10'000;

    std::printf("=== KV Alloc — Bare Allocator with Real Memory ===\n\n");

    KVBox box(layers, heads, dim);
    box.allocate(tokens);

    std::printf("Config: %u layers (no CLG), %u heads, %u dim\n",
                box.n_layers, box.n_kv_heads, box.head_dim);
    std::printf("Tokens: %llu → slots: %llu\n",
                (unsigned long long)box.n_tokens,
                (unsigned long long)box.n_slots);
    std::printf("Per head (scale_k + scale_v + K + V): %zu bytes\n", box.head_bytes);
    std::printf("Per slot: %zu bytes\n", box.slot_bytes);
    std::printf("Total allocated: %zu bytes (%.2f MiB)\n\n",
                box.total_bytes,
                (double)box.total_bytes / (1024.0 * 1024.0));

    // Round-trip test with dynamic range values (including values > 1.0)
    const int N_TEST = 1000;
    const uint32_t kv_len = box.head_dim * 2;

    _Float16* write_buf = (_Float16*)malloc(kv_len * sizeof(_Float16));
    _Float16* read_buf  = (_Float16*)malloc(kv_len * sizeof(_Float16));
    assert(write_buf && read_buf);

    srand(42);
    float max_error = 0.0f;
    float max_rel_error = 0.0f;

    for (int t = 0; t < N_TEST; t++) {
        uint64_t token_idx = (uint64_t)(rand() % (int)box.n_slots);
        uint32_t layer     = rand() % box.n_groups;
        uint32_t head      = rand() % box.n_kv_heads;

        // Generate values with wide dynamic range: [-10, 10]
        for (uint32_t i = 0; i < kv_len; i++) {
            float v = (float)(rand() / (double)RAND_MAX) * 20.0f - 10.0f;
            write_buf[i] = (_Float16)v;
        }

        box.write_slot(token_idx, layer, head, write_buf);
        box.read_slot(token_idx, layer, head, read_buf);

        for (uint32_t i = 0; i < kv_len; i++) {
            float orig = (float)write_buf[i];
            float recon = (float)read_buf[i];
            float err = fabsf(orig - recon);
            if (err > max_error) max_error = err;
            if (fabsf(orig) > 0.001f) {
                float rel = err / fabsf(orig);
                if (rel > max_rel_error) max_rel_error = rel;
            }
        }
    }

    free(write_buf);
    free(read_buf);

    // Max theoretical error: scale / 127. For max |v| = 10, error = 10/127 ≈ 0.0787
    std::printf("--- Round-trip test (dynamic range [-10, 10]) ---\n");
    std::printf("Tests: %d random KV pairs\n", N_TEST);
    std::printf("Max abs error: %f\n", max_error);
    std::printf("Max rel error: %f\n", max_rel_error);
    std::printf("Theoretical max abs error: %f (10/127)\n", 10.0f / 127.0f);

    if (max_error < (10.0f / 127.0f + 1e-6f)) {
        std::printf("Result: PASS\n");
    } else {
        std::printf("Result: FAIL\n");
    }

    assert(max_error < (10.0f / 127.0f + 1e-6f) && "Round-trip error exceeds theoretical max");

    return 0;
}
