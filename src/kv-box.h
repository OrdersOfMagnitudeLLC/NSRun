#pragma once

// KVBox — full-precision KV buffer for prefill-time retrieval.
// During prefill, stores full-precision K+V for all positions.
// After prefill, scores all positions using retrieval_q, injects top-N
// into working cache, then discards the buffer.
// No cold storage, no codebook, no compression.

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

struct KVBox {
    uint32_t n_layers;
    uint32_t group_size;  // K averaging factor for scoring (4)
    uint32_t n_kv_heads;
    uint32_t head_dim;

    uint64_t n_tokens;    // logical capacity (full context)
    uint64_t n_slots;     // = n_tokens (for compatibility)
    size_t   per_pos_bytes; // bytes per position in buffer
    size_t   total_bytes;   // current buffer size
    bool     is_init;
    uint64_t max_positions; // max positions to store (bounds RAM usage)

    // Full-precision KV buffer: position → [n_layers * n_kv_heads * 2 * head_dim] fp16
    // Layout per position: [K: layer×head×dim][V: layer×head×dim]
    std::unordered_map<int64_t, std::vector<ggml_fp16_t>> kv_buffer;
    std::queue<int64_t> write_order;  // FIFO order for eviction

    uint64_t writes;
    uint64_t retrievals;
    uint64_t prefill_retrievals;
    uint64_t abs_pos;

    std::vector<float> retrieval_q;
    int64_t retrieval_q_pos;
    bool decode_injected_once;
    uint32_t retrieval_layer;
    int32_t last_prefill_token;
    int64_t last_prefill_pos;

    KVBox()
        : n_layers(0), group_size(4), n_kv_heads(0), head_dim(0)
        , n_tokens(0), n_slots(0), per_pos_bytes(0), total_bytes(0), is_init(false)
        , max_positions(4096)
        , writes(0), retrievals(0), prefill_retrievals(0), abs_pos(0)
        , retrieval_q_pos(0), decode_injected_once(false), retrieval_layer(0)
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
        per_pos_bytes = (size_t)n_layers * n_kv_heads * 2 * head_dim * sizeof(ggml_fp16_t);
        total_bytes = 0;
        is_init = true;
        writes = 0;
        retrievals = 0;
        prefill_retrievals = 0;
        abs_pos = 0;
        decode_injected_once = false;
        retrieval_layer = 0;
        kv_buffer.clear();
        while (!write_order.empty()) write_order.pop();
        // Bound buffer to avoid OOM: keep at most max_positions entries.
        // For 36-layer 2-kv-head 128-dim model: ~36KB/pos → 4096 pos ≈ 147MB.
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

    // Write full-precision K and V for (token_idx, layer, head)
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
                    total_bytes -= per_pos_bytes;
                }
            }
            kv_buffer[pos] = std::vector<ggml_fp16_t>(n_layers * n_kv_heads * 2 * head_dim, (ggml_fp16_t)0);
            write_order.push(pos);
            total_bytes += per_pos_bytes;
            it = kv_buffer.find(pos);
        }

        size_t k_off = (size_t)(layer * n_kv_heads + head) * head_dim;
        memcpy(it->second.data() + k_off, k_f16, head_dim * sizeof(ggml_fp16_t));

        size_t v_off = (size_t)(n_layers * n_kv_heads + layer * n_kv_heads + head) * head_dim;
        memcpy(it->second.data() + v_off, v_f16, head_dim * sizeof(ggml_fp16_t));

        writes++;
    }

    // Read full-precision K and V for (token_idx, layer, head)
    bool read_slot(uint64_t token_idx, uint32_t layer,
                   uint32_t head, ggml_fp16_t* k_f16,
                   ggml_fp16_t* v_f16) {
        assert(is_init && "KVBox not initialized");
        assert(layer < n_layers);
        assert(head < n_kv_heads);

        int64_t pos = (int64_t)token_idx;
        auto it = kv_buffer.find(pos);
        if (it == kv_buffer.end()) return false;

        size_t k_off = (size_t)(layer * n_kv_heads + head) * head_dim;
        memcpy(k_f16, it->second.data() + k_off, head_dim * sizeof(ggml_fp16_t));

        size_t v_off = (size_t)(n_layers * n_kv_heads + layer * n_kv_heads + head) * head_dim;
        memcpy(v_f16, it->second.data() + v_off, head_dim * sizeof(ggml_fp16_t));

        return true;
    }

    // Read averaged K for scoring: average K across first group_size layers.
    bool get_score_k(uint64_t token_idx, uint32_t head, ggml_fp16_t* k_f16) {
        assert(head < n_kv_heads);
        int64_t pos = (int64_t)token_idx;
        auto it = kv_buffer.find(pos);
        if (it == kv_buffer.end()) return false;

        uint32_t n_avg = std::min(group_size, n_layers);
        std::vector<float> k_sum(head_dim, 0.0f);

        for (uint32_t il = 0; il < n_avg; ++il) {
            size_t k_off = (size_t)(il * n_kv_heads + head) * head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) {
                k_sum[d] += ggml_fp16_to_fp32(it->second[k_off + d]);
            }
        }

        for (uint32_t d = 0; d < head_dim; ++d) {
            k_f16[d] = ggml_fp32_to_fp16(k_sum[d] / (float)n_avg);
        }
        return true;
    }

    // No-op: K is already full precision in the buffer
    void finalize_score_buffer() {}

    // Clear the KV buffer to free memory (after injection is done)
    void clear_score_buffer() {
        kv_buffer.clear();
        while (!write_order.empty()) write_order.pop();
        total_bytes = 0;
    }

    void evict_position(int64_t pos) {
        auto it = kv_buffer.find(pos);
        if (it == kv_buffer.end()) return;
        kv_buffer.erase(it);
        total_bytes -= per_pos_bytes;
        // Note: write_order may still contain pos; it's lazily cleaned on next evict
    }

    uint64_t slots_used_count() const {
        return (uint64_t)kv_buffer.size();
    }

    uint64_t codebook_total_entries() const { return 0; }
    size_t codebook_bytes() const { return 0; }

    double compression_ratio(size_t ik_llama_kv_bytes) const {
        size_t total = total_bytes;
        if (total == 0) return 0.0;
        return (double)ik_llama_kv_bytes / (double)total;
    }
};
