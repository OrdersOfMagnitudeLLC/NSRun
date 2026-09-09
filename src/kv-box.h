#pragma once

// KVBox — compressed KV cache mirror using INT4 quantization + cross-layer grouping.
// This is a side-channel mirror: the real KV cache is untouched. KVBox just
// receives a copy of every K/V write for statistics and future use.

#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>

struct KVBox {
    static constexpr uint32_t CLG_GROUPS            = 4;
    static constexpr double   EVICT_KEEP_FRAC       = 0.4;
    static constexpr double   FILLER_KEEP_FRAC      = 0.60;
    static constexpr double   SEMANTIC_DEDUP_FACTOR = 2.0;

    uint32_t n_layers;
    uint32_t n_groups;    // CLG_GROUPS (cross-layer grouping factor)
    uint32_t n_kv_heads;
    uint32_t head_dim;

    uint64_t n_tokens;    // logical capacity (matches ik_llama kv_size)
    uint64_t n_slots;     // physical slots after compression
    size_t   head_bytes;  // per head: K+V in INT4 = head_dim bytes
    size_t   slot_bytes;  // per slot: n_groups * n_kv_heads * head_bytes
    size_t   total_bytes; // total allocated
    uint8_t* buf;

    uint64_t writes;      // number of write_slot calls
    uint64_t abs_pos;     // absolute token position (for small working cache mode)
    std::vector<bool> slot_used;  // bitmap of written slots

    KVBox()
        : n_layers(0), n_groups(CLG_GROUPS), n_kv_heads(0), head_dim(0)
        , n_tokens(0), n_slots(0), head_bytes(0), slot_bytes(0)
        , total_bytes(0), buf(nullptr)
        , writes(0)
        , abs_pos(0)
    {}

    ~KVBox() {
        if (buf) free(buf);
    }

    void init(uint32_t layers, uint32_t heads, uint32_t dim, uint64_t tokens) {
        if (buf) { free(buf); buf = nullptr; }
        n_layers = layers;
        n_groups = CLG_GROUPS;
        n_kv_heads = heads;
        head_dim = dim;
        n_tokens = tokens;
        n_slots = (uint64_t)(tokens * EVICT_KEEP_FRAC
                            * FILLER_KEEP_FRAC / SEMANTIC_DEDUP_FACTOR);
        if (n_slots == 0) n_slots = 1;
        head_bytes = (size_t)head_dim;  // 2 * head_dim * 0.5 (INT4)
        slot_bytes = (size_t)n_groups * n_kv_heads * head_bytes;
        total_bytes = (size_t)n_slots * slot_bytes;
        buf = (uint8_t*)malloc(total_bytes);
        assert(buf && "KVBox malloc failed");
        memset(buf, 0, total_bytes);
        writes = 0;
        abs_pos = 0;
        slot_used.assign(n_slots, false);
    }

    bool initialized() const { return buf != nullptr; }

    // Write K and V for (token_idx, layer, head).
    // k_f16 / v_f16: each head_dim ggml_fp16_t values.
    void write_slot(uint64_t token_idx, uint32_t layer,
                    uint32_t head, const ggml_fp16_t* k_f16,
                    const ggml_fp16_t* v_f16) {
        assert(buf && "KVBox not allocated");
        uint32_t group = layer % n_groups;
        assert(head < n_kv_heads);
        uint64_t slot = token_idx % n_slots;
        size_t offset = slot * slot_bytes
                      + (size_t)(group * n_kv_heads + head) * head_bytes;

        const ggml_fp16_t* srcs[2] = { k_f16, v_f16 };
        for (uint32_t kv = 0; kv < 2; kv++) {
            const ggml_fp16_t* src = srcs[kv];
            uint8_t* dst = buf + offset + kv * (head_dim / 2);
            for (uint32_t i = 0; i < head_dim; i += 2) {
                float f0 = ggml_fp16_to_fp32(src[i]);
                float f1 = ggml_fp16_to_fp32(src[i + 1]);
                dst[i / 2] = pack_int4(quantize_int4(f0), quantize_int4(f1));
            }
        }
        slot_used[slot] = true;
        writes++;
    }

    uint64_t slots_used_count() const {
        return (uint64_t)std::count(slot_used.begin(), slot_used.end(), true);
    }

    // Compression ratio: ik_llama allocated bytes / KVBox allocated bytes
    double compression_ratio(size_t ik_llama_kv_bytes) const {
        if (total_bytes == 0) return 0.0;
        return (double)ik_llama_kv_bytes / (double)total_bytes;
    }

private:
    static int8_t quantize_int4(float v) {
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int8_t q = (int8_t)lroundf(v * 7.0f);
        if (q > 7)  q = 7;
        if (q < -8) q = -8;
        return q;
    }

    static uint8_t pack_int4(int8_t lo, int8_t hi) {
        return (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
    }
};
