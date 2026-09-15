// NSKVCache — OOM LLC Commercial License — see /NS/LICENSING.md
// Copyright (c) 2026 Orders of Magnitude LLC. All rights reserved.
//
// KVBox — QUEST-style paged KV cold storage with content-based retrieval.
// Part of the NSRun inference stack.

#pragma once

// KVBox — QUEST-style paged KV cold storage.
// During prefill/eviction, stores per-token K+V at INT8 and maintains a
// per-page (16 tokens) scoring index: element-wise max/min of post-RoPE
// layer-0 K (FP16). Post-prefill, pages are scored with the QUEST bound
// score = sum_d max(q_d*max_k_d, q_d*min_k_d); top-24 pages (384 tokens)
// are injected into the working cache.

#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <queue>

#define KVBOX_PAGE_SIZE 16

struct KVBoxPage {
    // Element-wise max/min of post-RoPE K across tokens in page.
    // Single-layer only (retrieval_layer = n_layers/2) to minimize storage.
    // Layout: [n_kv_heads][head_dim]
    std::vector<ggml_fp16_t> max_k;
    std::vector<ggml_fp16_t> min_k;
    // Sum of pre-RoPE K per head, for content-based matching.
    // Layout: [n_kv_heads][head_dim] — single layer, stored as fp16.
    // mean_k[h] = sum_k[h] / sum_count gives per-head content vector.
    std::vector<ggml_fp16_t> sum_k;
    uint32_t sum_count;
    uint32_t n_tokens;    // tokens seen in this page
    int64_t  first_pos;   // page_id * KVBOX_PAGE_SIZE
};

struct KVBox {
    uint32_t n_layers;
    uint32_t group_size;  // unused, kept for compat
    uint32_t n_kv_heads;
    uint32_t head_dim;

    uint64_t n_tokens;    // logical capacity (full context)
    uint64_t n_slots;     // = n_tokens (for compatibility)
    size_t   per_pos_bytes; // bytes per position in buffer (INT8 K+V + scales)
    size_t   total_bytes;   // current buffer size
    bool     is_init;
    uint64_t max_positions; // max positions to store (bounds RAM usage)

    // Per-token INT8 K+V: pos → [n_layers][n_kv_heads][2][head_dim] int8
    std::unordered_map<int64_t, std::vector<int8_t>> kv_buffer;
    // Per-token scales: pos → [n_layers][n_kv_heads][2] fp16 (absmax)
    std::unordered_map<int64_t, std::vector<ggml_fp16_t>> kv_scales;
    std::queue<int64_t> write_order;  // FIFO order for eviction

    // Page scoring index: page_id → max/min/sum K (retrieval_layer only, post-RoPE)
    std::unordered_map<int64_t, KVBoxPage> pages;

    uint64_t writes;
    uint64_t retrievals;
    uint64_t prefill_retrievals;
    uint64_t abs_pos;

    std::vector<float> retrieval_q;
    int64_t retrieval_q_pos;
    uint32_t retrieval_q_ntok;
    bool decode_injected_once;
    uint32_t retrieval_layer;
    int32_t last_prefill_token;
    int64_t last_prefill_pos;

    KVBox()
        : n_layers(0), group_size(4), n_kv_heads(0), head_dim(0)
        , n_tokens(0), n_slots(0), per_pos_bytes(0), total_bytes(0), is_init(false)
        , max_positions(4096)
        , writes(0), retrievals(0), prefill_retrievals(0), abs_pos(0)
        , retrieval_q_pos(0), retrieval_q_ntok(0), decode_injected_once(false), retrieval_layer(0)
        , last_prefill_token(-1), last_prefill_pos(-1)
    {}

    ~KVBox() {}

    void init(uint32_t layers, uint32_t heads, uint32_t dim, uint64_t tokens) {
        n_layers = layers;
        group_size = 4;
        n_kv_heads = heads;
        head_dim = dim;
        n_tokens = tokens;
        n_slots = tokens;
        // INT8 K+V + fp16 scales per position
        per_pos_bytes = (size_t)n_layers * n_kv_heads * 2 * head_dim * sizeof(int8_t)
                      + (size_t)n_layers * n_kv_heads * 2 * sizeof(ggml_fp16_t);
        total_bytes = 0;
        is_init = true;
        writes = 0;
        retrievals = 0;
        prefill_retrievals = 0;
        abs_pos = 0;
        decode_injected_once = false;
        // Score pages at a middle layer — layer-0 attention is too diffuse
        // for semantic retrieval; mid layers have stronger retrieval heads.
        retrieval_layer = layers / 2;
        kv_buffer.clear();
        kv_scales.clear();
        pages.clear();
        while (!write_order.empty()) write_order.pop();
        // Bound buffer to avoid OOM: keep at most max_positions entries.
        // INT8: ~18.7KB/pos for 36-layer 2-kv-head 128-dim → 4096 pos ≈ 75MB.
        max_positions = 4096;
    }

    bool initialized() const { return is_init; }

    bool has_position(int64_t pos) const {
        return kv_buffer.find(pos) != kv_buffer.end();
    }

    std::vector<int64_t> get_stored_positions() const {
        std::vector<int64_t> positions;
        positions.reserve(kv_buffer.size());
        for (const auto & kv : kv_buffer) {
            positions.push_back(kv.first);
        }
        return positions;
    }

    // Update page scoring index with post-RoPE K (for max/min bound) and
    // pre-RoPE K (for content-based mean K matching).
    void update_page_index(int64_t pos, uint32_t layer, uint32_t head,
                           const ggml_fp16_t* k_postrope,
                           const ggml_fp16_t* k_prerope) {
        // Only build scoring index for the retrieval layer (middle layer).
        // All other layers still get K/V stored in INT8 buffer for injection.
        if (layer != retrieval_layer) return;
        assert(head < n_kv_heads);
        int64_t page_id = pos / KVBOX_PAGE_SIZE;
        auto it = pages.find(page_id);
        if (it == pages.end()) {
            KVBoxPage pg;
            pg.max_k.assign(n_kv_heads * head_dim, ggml_fp32_to_fp16(-FLT_MAX));
            pg.min_k.assign(n_kv_heads * head_dim, ggml_fp32_to_fp16(FLT_MAX));
            pg.sum_k.assign(n_kv_heads * head_dim, ggml_fp32_to_fp16(0.0f));
            pg.sum_count = 0;
            pg.n_tokens = 0;
            pg.first_pos = page_id * KVBOX_PAGE_SIZE;
            it = pages.emplace(page_id, std::move(pg)).first;
        }
        KVBoxPage & pg = it->second;
        size_t off = (size_t)head * head_dim;
        ggml_fp16_t* mx = pg.max_k.data() + off;
        ggml_fp16_t* mn = pg.min_k.data() + off;
        ggml_fp16_t* sk = pg.sum_k.data() + off;
        for (uint32_t d = 0; d < head_dim; ++d) {
            float v = ggml_fp16_to_fp32(k_postrope[d]);
            if (v > ggml_fp16_to_fp32(mx[d])) mx[d] = k_postrope[d];
            if (v < ggml_fp16_to_fp32(mn[d])) mn[d] = k_postrope[d];
            // Accumulate pre-RoPE K as fp16 sum
            sk[d] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(sk[d]) + ggml_fp16_to_fp32(k_prerope[d]));
        }
        pg.sum_count++;
        pg.n_tokens++;
    }

    // Write INT8-quantized K and V for (token_idx, layer, head).
    // k_f16 must be PRE-RoPE (unrotated); v_f16 is raw.
    void write_slot(uint64_t token_idx, uint32_t layer,
                    uint32_t head, const ggml_fp16_t* k_f16,
                    const ggml_fp16_t* v_f16) {
        assert(is_init && "KVBox not initialized");
        assert(layer < n_layers);
        assert(head < n_kv_heads);

        int64_t pos = (int64_t)token_idx;
        auto it = kv_buffer.find(pos);
        if (it == kv_buffer.end()) {
            // Evict oldest position if buffer is full
            if (kv_buffer.size() >= max_positions && !write_order.empty()) {
                int64_t evict_pos = write_order.front();
                write_order.pop();
                auto evict_it = kv_buffer.find(evict_pos);
                if (evict_it != kv_buffer.end()) {
                    kv_buffer.erase(evict_it);
                    kv_scales.erase(evict_pos);
                    total_bytes -= per_pos_bytes;
                }
            }
            kv_buffer[pos] = std::vector<int8_t>(n_layers * n_kv_heads * 2 * head_dim, 0);
            kv_scales[pos] = std::vector<ggml_fp16_t>(n_layers * n_kv_heads * 2, (ggml_fp16_t)0);
            write_order.push(pos);
            total_bytes += per_pos_bytes;
            it = kv_buffer.find(pos);
        }
        auto sit = kv_scales.find(pos);

        // Quantize K: int8 = round(fp16 / scale), scale = absmax/127
        {
            float amax = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                float a = fabsf(ggml_fp16_to_fp32(k_f16[d]));
                if (a > amax) amax = a;
            }
            float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            float inv = 1.0f / scale;
            size_t k_off = (size_t)(layer * n_kv_heads + head) * head_dim;
            int8_t* dst = it->second.data() + k_off;
            for (uint32_t d = 0; d < head_dim; ++d) {
                float q = ggml_fp16_to_fp32(k_f16[d]) * inv;
                int qi = (int)lroundf(q);
                dst[d] = (int8_t)(qi > 127 ? 127 : (qi < -127 ? -127 : qi));
            }
            sit->second[(layer * n_kv_heads + head) * 2 + 0] = ggml_fp32_to_fp16(scale);
        }

        // Quantize V
        {
            float amax = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                float a = fabsf(ggml_fp16_to_fp32(v_f16[d]));
                if (a > amax) amax = a;
            }
            float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            float inv = 1.0f / scale;
            size_t v_off = (size_t)(n_layers * n_kv_heads + layer * n_kv_heads + head) * head_dim;
            int8_t* dst = it->second.data() + v_off;
            for (uint32_t d = 0; d < head_dim; ++d) {
                float q = ggml_fp16_to_fp32(v_f16[d]) * inv;
                int qi = (int)lroundf(q);
                dst[d] = (int8_t)(qi > 127 ? 127 : (qi < -127 ? -127 : qi));
            }
            sit->second[(layer * n_kv_heads + head) * 2 + 1] = ggml_fp32_to_fp16(scale);
        }

        writes++;
    }

    // Read dequantized K and V for (token_idx, layer, head) → fp16
    bool read_slot(uint64_t token_idx, uint32_t layer,
                   uint32_t head, ggml_fp16_t* k_f16,
                   ggml_fp16_t* v_f16) {
        assert(is_init && "KVBox not initialized");
        assert(layer < n_layers);
        assert(head < n_kv_heads);

        int64_t pos = (int64_t)token_idx;
        auto it = kv_buffer.find(pos);
        if (it == kv_buffer.end()) return false;
        auto sit = kv_scales.find(pos);
        if (sit == kv_scales.end()) return false;

        size_t k_off = (size_t)(layer * n_kv_heads + head) * head_dim;
        float k_scale = ggml_fp16_to_fp32(sit->second[(layer * n_kv_heads + head) * 2 + 0]);
        const int8_t* k_src = it->second.data() + k_off;
        for (uint32_t d = 0; d < head_dim; ++d) {
            k_f16[d] = ggml_fp32_to_fp16((float)k_src[d] * k_scale);
        }

        size_t v_off = (size_t)(n_layers * n_kv_heads + layer * n_kv_heads + head) * head_dim;
        float v_scale = ggml_fp16_to_fp32(sit->second[(layer * n_kv_heads + head) * 2 + 1]);
        const int8_t* v_src = it->second.data() + v_off;
        for (uint32_t d = 0; d < head_dim; ++d) {
            v_f16[d] = ggml_fp32_to_fp16((float)v_src[d] * v_scale);
        }

        return true;
    }

    // No-op: page index is maintained incrementally during writes
    void finalize_score_buffer() {}

    // Clear the KV buffer to free memory (after injection is done)
    void clear_score_buffer() {
        kv_buffer.clear();
        kv_scales.clear();
        pages.clear();
        while (!write_order.empty()) write_order.pop();
        total_bytes = 0;
    }

    void evict_position(int64_t pos) {
        auto it = kv_buffer.find(pos);
        if (it == kv_buffer.end()) return;
        kv_buffer.erase(it);
        kv_scales.erase(pos);
        total_bytes -= per_pos_bytes;
        // Note: write_order may still contain pos; it's lazily cleaned on next evict
        // Note: page index keeps its max/min — stale entries are harmless for an upper bound
    }

    uint64_t slots_used_count() const {
        return (uint64_t)kv_buffer.size();
    }

    uint64_t page_count() const {
        return (uint64_t)pages.size();
    }

    uint64_t codebook_total_entries() const { return 0; }
    size_t codebook_bytes() const { return 0; }

    double compression_ratio(size_t ik_llama_kv_bytes) const {
        size_t total = total_bytes;
        if (total == 0) return 0.0;
        return (double)ik_llama_kv_bytes / (double)total;
    }
};
