// kv_alloc.cpp — Bare KV allocator with actual memory (no LLM, no inference)
//
// Allocates the compressed KV budget, writes/reads KV pairs with INT4
// quantization. Round-trip test validates accuracy.
//
// Build:  g++ -std=c++17 -O2 -o kv_alloc kv_alloc.cpp
// Run:    ./kv_alloc

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

// ─── Compression parameters (from kv_box.cpp) ──────────────────────
static constexpr uint32_t CLG_GROUPS            = 4;
static constexpr double   EVICT_KEEP_FRAC       = 0.4;
static constexpr double   FILLER_KEEP_FRAC      = 0.60;
static constexpr double   SEMANTIC_DEDUP_FACTOR = 2.0;

// ─── INT4 quantization (clamp + pack, no LUT) ──────────────────────
// fp16 → clamp [-1,1] → scale to [-7,7] → 4-bit signed
// Max quant error: 1/14 ≈ 0.0714 < 0.1

static int8_t quantize_int4(float v) {
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    int8_t q = (int8_t)lroundf(v * 7.0f);
    if (q > 7)  q = 7;
    if (q < -8) q = -8;
    return q;
}

static float dequantize_int4(int8_t q) {
    return (float)q / 7.0f;
}

static uint8_t pack_int4(int8_t lo, int8_t hi) {
    return (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
}

static int8_t unpack_lo_int4(uint8_t b) {
    int8_t v = (int8_t)(b & 0xF);
    if (v & 0x8) v |= 0xF0;
    return v;
}

static int8_t unpack_hi_int4(uint8_t b) {
    int8_t v = (int8_t)((b >> 4) & 0xF);
    if (v & 0x8) v |= 0xF0;
    return v;
}

// ─── KVBox ─────────────────────────────────────────────────────────

struct KVBox {
    uint32_t n_layers;
    uint32_t n_groups;    // n_layers / CLG_GROUPS
    uint32_t n_kv_heads;
    uint32_t head_dim;

    uint64_t n_tokens;    // logical capacity
    uint64_t n_slots;     // physical slots after compression
    size_t   head_bytes;  // per head: K+V in INT4 = head_dim bytes
    size_t   slot_bytes;  // per slot: n_groups * n_kv_heads * head_bytes
    size_t   total_bytes; // total allocated
    uint8_t* buf;

    KVBox(uint32_t layers, uint32_t heads, uint32_t dim)
        : n_layers(layers)
        , n_groups(CLG_GROUPS)
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
        // Physical slots after eviction + filler + dedup
        n_slots = (uint64_t)(tokens * EVICT_KEEP_FRAC
                            * FILLER_KEEP_FRAC / SEMANTIC_DEDUP_FACTOR);
        if (n_slots == 0) n_slots = 1;
        // Per head: K + V, each head_dim elements, INT4 = 0.5 bytes → head_dim bytes
        head_bytes = (size_t)head_dim;  // 2 * head_dim * 0.5
        // Per slot: all groups × all heads
        slot_bytes = (size_t)n_groups * n_kv_heads * head_bytes;
        total_bytes = (size_t)n_slots * slot_bytes;
        buf = (uint8_t*)malloc(total_bytes);
        assert(buf && "malloc failed");
        memset(buf, 0, total_bytes);
    }

    // Write a KV pair for (token_idx, layer_group, head)
    // data_fp16: pointer to head_dim * 2 fp16 values [K | V]
    void write_slot(uint64_t token_idx, uint32_t group,
                    uint32_t head, const _Float16* data) {
        assert(buf && "not allocated");
        assert(group < n_groups && head < n_kv_heads);
        uint64_t slot = token_idx % n_slots;
        size_t offset = slot * slot_bytes
                      + (size_t)(group * n_kv_heads + head) * head_bytes;
        for (uint32_t kv = 0; kv < 2; kv++) {
            const _Float16* src = data + kv * head_dim;
            uint8_t* dst = buf + offset + kv * (head_dim / 2);
            for (uint32_t i = 0; i < head_dim; i += 2) {
                int8_t q0 = quantize_int4((float)src[i]);
                int8_t q1 = quantize_int4((float)src[i + 1]);
                dst[i / 2] = pack_int4(q0, q1);
            }
        }
    }

    // Read a KV pair for (token_idx, layer_group, head)
    // out_fp16: pointer to head_dim * 2 fp16 values [K | V]
    void read_slot(uint64_t token_idx, uint32_t group,
                   uint32_t head, _Float16* out) {
        assert(buf && "not allocated");
        uint64_t slot = token_idx % n_slots;
        size_t offset = slot * slot_bytes
                      + (size_t)(group * n_kv_heads + head) * head_bytes;
        for (uint32_t kv = 0; kv < 2; kv++) {
            const uint8_t* src = buf + offset + kv * (head_dim / 2);
            _Float16* dst = out + kv * head_dim;
            for (uint32_t i = 0; i < head_dim; i += 2) {
                uint8_t b = src[i / 2];
                dst[i]     = (_Float16)dequantize_int4(unpack_lo_int4(b));
                dst[i + 1] = (_Float16)dequantize_int4(unpack_hi_int4(b));
            }
        }
    }
};

// ─── Round-trip test ────────────────────────────────────────────────

int main() {
    const uint32_t layers = 32;
    const uint32_t heads  = 8;
    const uint32_t dim    = 128;
    const uint64_t tokens = 1'000'000;

    std::printf("=== KV Alloc — Bare Allocator with Real Memory ===\n\n");

    KVBox box(layers, heads, dim);
    box.allocate(tokens);

    std::printf("Config: %u layers → %u groups, %u heads, %u dim\n",
                box.n_layers, box.n_groups, box.n_kv_heads, box.head_dim);
    std::printf("Tokens: %llu → slots: %llu\n",
                (unsigned long long)box.n_tokens,
                (unsigned long long)box.n_slots);
    std::printf("Per head (K+V, INT4): %zu bytes\n", box.head_bytes);
    std::printf("Per slot: %zu bytes\n", box.slot_bytes);
    std::printf("Total allocated: %zu bytes (%.2f MiB)\n\n",
                box.total_bytes,
                (double)box.total_bytes / (1024.0 * 1024.0));

    // Round-trip test
    const int N_TEST = 1000;
    const uint32_t kv_len = box.head_dim * 2;

    _Float16* write_buf = (_Float16*)malloc(kv_len * sizeof(_Float16));
    _Float16* read_buf  = (_Float16*)malloc(kv_len * sizeof(_Float16));
    assert(write_buf && read_buf);

    srand(42);
    float max_error = 0.0f;

    for (int t = 0; t < N_TEST; t++) {
        uint64_t token_idx = (uint64_t)(rand() % (int)box.n_slots);
        uint32_t group     = rand() % box.n_groups;
        uint32_t head      = rand() % box.n_kv_heads;

        for (uint32_t i = 0; i < kv_len; i++) {
            float v = (float)(rand() / (double)RAND_MAX) * 2.0f - 1.0f;
            write_buf[i] = (_Float16)v;
        }

        box.write_slot(token_idx, group, head, write_buf);
        box.read_slot(token_idx, group, head, read_buf);

        for (uint32_t i = 0; i < kv_len; i++) {
            float err = fabsf((float)write_buf[i] - (float)read_buf[i]);
            if (err > max_error) max_error = err;
        }
    }

    free(write_buf);
    free(read_buf);

    std::printf("--- Round-trip test ---\n");
    std::printf("Tests: %d random KV pairs\n", N_TEST);
    std::printf("Max error: %f\n", max_error);
    std::printf("Threshold: 0.1\n");

    if (max_error < 0.1f) {
        std::printf("Result: PASS\n");
    } else {
        std::printf("Result: FAIL\n");
    }

    assert(max_error < 0.1f && "Round-trip max error >= 0.1");

    return (max_error < 0.1f) ? 0 : 1;
}
