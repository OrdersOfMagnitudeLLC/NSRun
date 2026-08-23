#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

// ── INT8 block ─────────────────────────────────────────────────────────────
// 8 floats → 1 uint16 scale (float16) + 8 bytes int8 = 10 bytes
// vs 32 bytes float32 → 3.2x reduction
struct KVBlock8 {
    uint16_t scale_bits;  // float16: max(|x|) / 127.0
    int8_t   data[8];     // 8 int8 values, range [-127, 127]
};

// ── Quantize 8 floats → KVBlock8 ─────────────────────────────────────────
inline void kv_q8_encode(const float* src, KVBlock8* dst) {
    float mx = 0.0f;
    for (int i = 0; i < 8; i++) mx = std::max(mx, std::abs(src[i]));
    float scale = (mx == 0.0f) ? 1.0f : mx / 127.0f;
    uint32_t fb; memcpy(&fb, &scale, 4);
    int exp16 = (int)((fb >> 23) & 0xff) - 127 + 15;
    if (exp16 < 0)  exp16 = 0;
    if (exp16 > 30) exp16 = 30;
    dst->scale_bits = (uint16_t)(((fb >> 16) & 0x8000) |
                                  (exp16 << 10) |
                                  ((fb >> 13) & 0x3ff));
    for (int i = 0; i < 8; i++) {
        int v = (int)std::lround(src[i] / scale);
        v = v < -127 ? -127 : v > 127 ? 127 : v;
        dst->data[i] = (int8_t)v;
    }
}

// ── Dequantize KVBlock8 → 8 floats ───────────────────────────────────────
inline void kv_q8_decode(const KVBlock8* src, float* dst) {
    uint32_t fb = (((uint32_t)(src->scale_bits & 0x8000)) << 16) |
                  ((((uint32_t)(src->scale_bits >> 10) & 0x1f) - 15 + 127) << 23) |
                  (((uint32_t)(src->scale_bits & 0x3ff)) << 13);
    float scale; memcpy(&scale, &fb, 4);
    for (int i = 0; i < 8; i++)
        dst[i] = src->data[i] * scale;
}

// ── Config ─────────────────────────────────────────────────────────────────
struct NSKVConfig {
    size_t n_layers    = 0;
    size_t n_kv_heads  = 0;
    size_t head_dim    = 0;
    size_t capacity    = 0; // max tokens before eviction triggers
    size_t latent_dim  = 0; // 0 = full KV; >0 = MLA latent dim (future, not wired yet)
    bool   debug_float32 = false;  // bypass quantization for testing
};

// ── NSKVStore ──────────────────────────────────────────────────────────────
// Paged slab. Layout per allocated page:
//   page[layer * PAGE_SLOTS * entry_bytes + (slot % PAGE_SLOTS) * entry_bytes]
//   where entry_bytes = 2 * n_blocks_per_entry * sizeof(KVBlock8)
//         (first half = K blocks, second half = V blocks)
//
// seq_pos maps to slot via: slot = seq_pos % capacity  (circular)
// info_scores_[slot]: raw L2 norm of the stored V vector for this slot
//                     (higher = more information-dense = evict last)
class NSKVStore {
public:
    explicit NSKVStore(const NSKVConfig& cfg);
    ~NSKVStore();

    // Write K and V for one (layer, seq_pos).
    // n_floats = n_kv_heads * head_dim.
    // Computes and stores the L2 V-norm as the slot's information score.
    void write(size_t layer, size_t seq_pos,
               const float* k, const float* v, size_t n_floats);

    // Read K and V for one (layer, seq_pos) into caller float buffers.
    // Dequantizes only the kv_head slice needed: 
    //   out buffers must be size head_dim.
    // kv_head: which KV head to dequant (0..n_kv_heads-1)
    // Returns false if slot was evicted — caller skips this position.
    bool read_head(size_t layer, size_t seq_pos, size_t kv_head,
                   float* k_out, float* v_out) const;

    // Same as read_head but dequantizes only K or only V — for callers
    // (e.g. attention_impl's score/accumulate loops) that only need one
    // side per pass, halving KV dequant work vs. calling read_head twice.
    bool read_k_head(size_t layer, size_t seq_pos, size_t kv_head,
                      float* k_out) const;
    bool read_v_head(size_t layer, size_t seq_pos, size_t kv_head,
                      float* v_out) const;

    // Fused K+V read: dequantizes both in one pass, avoiding duplicated
    // validity checks, page-index math, and page_last_access_ updates.
    bool read_kv_head(size_t layer, size_t seq_pos, size_t kv_head,
                      float* k_out, float* v_out) const;

    // Evict lowest-scoring 25% of stored slots.
    void evict();

    // Sort-only variant: returns all valid slots in ascending eviction-score
    // order (histogram bins), without mutating state or running the cold-page
    // pass. Used for benchmarking the sort/bin-walk against std::sort.
    std::vector<size_t> evict_sort_order() const;

    // Trigger eviction once we reach 75% of capacity.
    bool needs_eviction() const { return n_stored_ >= (cfg_.capacity * 3) / 4; }

    size_t n_stored()  const { return n_stored_; }
    size_t capacity()  const { return cfg_.capacity; }

private:
    NSKVConfig cfg_;
    size_t n_blocks_per_entry_; // ceil(n_kv_heads * head_dim / 8)
    size_t blocks_per_head_;    // ceil(head_dim / 8)
    size_t entry_bytes_;        // 2 * n_blocks_per_entry_ * sizeof(KVBlock8) (or 2*n_floats*sizeof(float) in debug mode)
    size_t n_floats_;           // n_kv_heads * head_dim
    size_t n_stored_ = 0;

    static constexpr size_t PAGE_SLOTS = 512;
    static constexpr size_t COLD_SLOTS = 512;

    std::vector<uint8_t*> pages_;       // [n_pages], lazily allocated
    mutable std::vector<size_t> page_last_access_; // [n_pages] last seq_pos touched on page
    std::vector<float>   info_scores_;  // [capacity]  raw L2 V-norm
    std::vector<size_t>  seq_pos_;      // [capacity]  token position for recency
    std::vector<bool>    valid_;        // [capacity]

    size_t max_seq_pos_ = 0;            // highest seq_pos written so far
    float  max_v_norm_  = 0.0f;         // highest V-norm seen so far (for normalization)

    void ensure_page(size_t slot);

    KVBlock8* page_blocks(size_t layer, size_t slot) {
        size_t page_idx = slot / PAGE_SLOTS;
        size_t local    = slot % PAGE_SLOTS;
        uint8_t* p = pages_[page_idx];
        if (!p) return nullptr;
        return (KVBlock8*)(p + (layer * PAGE_SLOTS + local) * entry_bytes_);
    }
    const KVBlock8* page_blocks(size_t layer, size_t slot) const {
        size_t page_idx = slot / PAGE_SLOTS;
        size_t local    = slot % PAGE_SLOTS;
        uint8_t* p = pages_[page_idx];
        if (!p) return nullptr;
        return (const KVBlock8*)(p + (layer * PAGE_SLOTS + local) * entry_bytes_);
    }
    KVBlock8* k_blocks(size_t layer, size_t slot) { return page_blocks(layer, slot); }
    KVBlock8* v_blocks(size_t layer, size_t slot) {
        KVBlock8* k = k_blocks(layer, slot);
        return k ? k + n_blocks_per_entry_ : nullptr;
    }
    const KVBlock8* k_blocks(size_t layer, size_t slot) const { return page_blocks(layer, slot); }
    const KVBlock8* v_blocks(size_t layer, size_t slot) const {
        const KVBlock8* k = k_blocks(layer, slot);
        return k ? k + n_blocks_per_entry_ : nullptr;
    }
};
