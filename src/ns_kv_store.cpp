#include "ns_kv_store.h"
#include <cassert>
#include <iostream>
#include <sys/mman.h>

NSKVStore::NSKVStore(const NSKVConfig& cfg)
    : cfg_(cfg)
{
    // Compute block counts
    size_t kv_floats = cfg_.n_kv_heads * cfg_.head_dim;
    n_floats_ = kv_floats;
    n_blocks_per_entry_ = (kv_floats + 7) / 8;  // ceil
    blocks_per_head_ = (cfg_.head_dim + 7) / 8;  // ceil

    if (cfg_.debug_float32) {
        entry_bytes_ = 2 * cfg_.n_kv_heads * cfg_.head_dim * sizeof(float);
    } else {
        entry_bytes_ = 2 * n_blocks_per_entry_ * sizeof(KVBlock8);
    }

    // Assert head_dim is multiple of 8 (true for modern models)
    assert(cfg_.head_dim % 8 == 0 && "head_dim must be multiple of 8");

    // Lazy paged allocation: only reserve page pointer array.
    size_t n_pages = (cfg_.capacity + PAGE_SLOTS - 1) / PAGE_SLOTS;
    pages_.resize(n_pages, nullptr);
    page_last_access_.resize(n_pages, 0);

    // Allocate metadata
    info_scores_.resize(cfg_.capacity, 0.0f);
    seq_pos_.resize(cfg_.capacity, 0);
    valid_.resize(cfg_.capacity, false);
}

NSKVStore::~NSKVStore() {
    size_t page_bytes = cfg_.n_layers * PAGE_SLOTS * entry_bytes_;
    for (uint8_t* p : pages_) {
        if (p) ::munmap(p, page_bytes);
    }
}

void NSKVStore::ensure_page(size_t slot) {
    size_t page_idx = slot / PAGE_SLOTS;
    if (!pages_[page_idx]) {
        size_t page_bytes = cfg_.n_layers * PAGE_SLOTS * entry_bytes_;
        uint8_t* p = (uint8_t*)::mmap(nullptr, page_bytes, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(p != MAP_FAILED);
        pages_[page_idx] = p;
    }
}

void NSKVStore::write(size_t layer, size_t seq_pos,
                       const float* k, const float* v, size_t n_floats) {
    size_t slot = seq_pos % cfg_.capacity;

    ensure_page(slot);

    size_t page_idx = slot / PAGE_SLOTS;
    page_last_access_[page_idx] = seq_pos;

    // Compute L2 norm of the V vector as the information-content signal.
    float v_norm_sq = 0.0f;
    for (size_t i = 0; i < n_floats; ++i) {
        v_norm_sq += v[i] * v[i];
    }
    float v_norm = std::sqrt(v_norm_sq);
    max_v_norm_ = std::max(max_v_norm_, v_norm);
    max_seq_pos_ = std::max(max_seq_pos_, seq_pos);

    // DEBUG: bypass quantization for testing
    if (cfg_.debug_float32) {
        size_t local = slot % PAGE_SLOTS;
        float* dst = (float*)(pages_[page_idx] +
                     (layer * PAGE_SLOTS + local) * entry_bytes_);
        memcpy(dst,           k, n_floats * sizeof(float));  // K
        memcpy(dst + n_floats, v, n_floats * sizeof(float)); // V
        info_scores_[slot] = v_norm;
        seq_pos_[slot] = seq_pos;
        if (!valid_[slot]) n_stored_++;
        valid_[slot] = true;
        return;
    }

    // Quantize K blocks
    size_t n_blocks = (n_floats + 7) / 8;
    KVBlock8* k_dst = k_blocks(layer, slot);
    KVBlock8* v_dst = v_blocks(layer, slot);

    for (size_t b = 0; b < n_blocks; ++b) {
        size_t src_offset = b * 8;
        if (src_offset < n_floats) {
            // Full or partial block
            float src_buf[8] = {0};
            size_t copy_len = std::min(size_t(8), n_floats - src_offset);
            memcpy(src_buf, k + src_offset, copy_len * sizeof(float));
            kv_q8_encode(src_buf, k_dst + b);

            memcpy(src_buf, v + src_offset, copy_len * sizeof(float));
            kv_q8_encode(src_buf, v_dst + b);
        } else {
            // Zero block (shouldn't happen with proper ceil)
            memset(k_dst + b, 0, sizeof(KVBlock8));
            memset(v_dst + b, 0, sizeof(KVBlock8));
        }
    }

    info_scores_[slot] = v_norm;
    seq_pos_[slot] = seq_pos;
    if (!valid_[slot]) {
        n_stored_++;
    }
    valid_[slot] = true;
}

bool NSKVStore::read_head(size_t layer, size_t seq_pos, size_t kv_head,
                          float* k_out, float* v_out) const {
    size_t slot = seq_pos % cfg_.capacity;

    // DEBUG: bypass quantization for testing
    if (cfg_.debug_float32) {
        if (!valid_[slot]) {
            memset(k_out, 0, cfg_.head_dim * sizeof(float));
            memset(v_out, 0, cfg_.head_dim * sizeof(float));
            return false;
        }
        size_t page_idx = slot / PAGE_SLOTS;
        size_t local    = slot % PAGE_SLOTS;
        const float* src = (const float*)(pages_[page_idx] +
                           (layer * PAGE_SLOTS + local) * entry_bytes_);
        size_t head_offset = kv_head * cfg_.head_dim;
        page_last_access_[page_idx] = seq_pos;
        memcpy(k_out, src           + head_offset, cfg_.head_dim * sizeof(float));
        memcpy(v_out, src + n_floats_ + head_offset, cfg_.head_dim * sizeof(float));
        return true;
    }

    if (!valid_[slot]) {
        memset(k_out, 0, cfg_.head_dim * sizeof(float));
        memset(v_out, 0, cfg_.head_dim * sizeof(float));
        return false;
    }

    size_t head_block_start = kv_head * blocks_per_head_;
    const KVBlock8* k_src = k_blocks(layer, slot);
    if (!k_src) {
        memset(k_out, 0, cfg_.head_dim * sizeof(float));
        memset(v_out, 0, cfg_.head_dim * sizeof(float));
        return false;
    }
    k_src += head_block_start;
    const KVBlock8* v_src = k_src + n_blocks_per_entry_;

    size_t page_idx = slot / PAGE_SLOTS;
    page_last_access_[page_idx] = seq_pos;

    for (size_t b = 0; b < blocks_per_head_; ++b) {
        kv_q8_decode(k_src + b, k_out + b * 8);
        kv_q8_decode(v_src + b, v_out + b * 8);
    }

    return true;
}

bool NSKVStore::read_k_head(size_t layer, size_t seq_pos, size_t kv_head,
                             float* k_out) const {
    size_t slot = seq_pos % cfg_.capacity;

    if (cfg_.debug_float32) {
        if (!valid_[slot]) {
            memset(k_out, 0, cfg_.head_dim * sizeof(float));
            return false;
        }
        size_t page_idx = slot / PAGE_SLOTS;
        size_t local    = slot % PAGE_SLOTS;
        const float* src = (const float*)(pages_[page_idx] +
                           (layer * PAGE_SLOTS + local) * entry_bytes_);
        size_t head_offset = kv_head * cfg_.head_dim;
        page_last_access_[page_idx] = seq_pos;
        memcpy(k_out, src + head_offset, cfg_.head_dim * sizeof(float));
        return true;
    }

    if (!valid_[slot]) {
        memset(k_out, 0, cfg_.head_dim * sizeof(float));
        return false;
    }

    size_t head_block_start = kv_head * blocks_per_head_;
    const KVBlock8* k_src = k_blocks(layer, slot);
    if (!k_src) {
        memset(k_out, 0, cfg_.head_dim * sizeof(float));
        return false;
    }
    k_src += head_block_start;

    size_t page_idx_rk = slot / PAGE_SLOTS;
    page_last_access_[page_idx_rk] = seq_pos;

    for (size_t b = 0; b < blocks_per_head_; ++b) {
        kv_q8_decode(k_src + b, k_out + b * 8);
    }

    return true;
}

bool NSKVStore::read_v_head(size_t layer, size_t seq_pos, size_t kv_head,
                             float* v_out) const {
    size_t slot = seq_pos % cfg_.capacity;

    if (cfg_.debug_float32) {
        if (!valid_[slot]) {
            memset(v_out, 0, cfg_.head_dim * sizeof(float));
            return false;
        }
        size_t page_idx = slot / PAGE_SLOTS;
        size_t local    = slot % PAGE_SLOTS;
        const float* src = (const float*)(pages_[page_idx] +
                           (layer * PAGE_SLOTS + local) * entry_bytes_);
        size_t head_offset = kv_head * cfg_.head_dim;
        page_last_access_[page_idx] = seq_pos;
        memcpy(v_out, src + n_floats_ + head_offset, cfg_.head_dim * sizeof(float));
        return true;
    }

    if (!valid_[slot]) {
        memset(v_out, 0, cfg_.head_dim * sizeof(float));
        return false;
    }

    size_t head_block_start = kv_head * blocks_per_head_;
    const KVBlock8* v_src = v_blocks(layer, slot);
    if (!v_src) {
        memset(v_out, 0, cfg_.head_dim * sizeof(float));
        return false;
    }
    v_src += head_block_start;

    size_t page_idx_rv = slot / PAGE_SLOTS;
    page_last_access_[page_idx_rv] = seq_pos;

    for (size_t b = 0; b < blocks_per_head_; ++b) {
        kv_q8_decode(v_src + b, v_out + b * 8);
    }

    return true;
}

bool NSKVStore::read_kv_head(size_t layer, size_t seq_pos, size_t kv_head,
                              float* k_out, float* v_out) const {
    size_t slot = seq_pos % cfg_.capacity;

    if (cfg_.debug_float32) {
        if (!valid_[slot]) {
            memset(k_out, 0, cfg_.head_dim * sizeof(float));
            memset(v_out, 0, cfg_.head_dim * sizeof(float));
            return false;
        }
        size_t page_idx = slot / PAGE_SLOTS;
        size_t local    = slot % PAGE_SLOTS;
        const float* src = (const float*)(pages_[page_idx] +
                           (layer * PAGE_SLOTS + local) * entry_bytes_);
        size_t head_offset = kv_head * cfg_.head_dim;
        page_last_access_[page_idx] = seq_pos;
        memcpy(k_out, src           + head_offset, cfg_.head_dim * sizeof(float));
        memcpy(v_out, src + n_floats_ + head_offset, cfg_.head_dim * sizeof(float));
        return true;
    }

    if (!valid_[slot]) {
        memset(k_out, 0, cfg_.head_dim * sizeof(float));
        memset(v_out, 0, cfg_.head_dim * sizeof(float));
        return false;
    }

    size_t head_block_start = kv_head * blocks_per_head_;
    const KVBlock8* k_src = k_blocks(layer, slot);
    if (!k_src) {
        memset(k_out, 0, cfg_.head_dim * sizeof(float));
        memset(v_out, 0, cfg_.head_dim * sizeof(float));
        return false;
    }
    k_src += head_block_start;
    const KVBlock8* v_src = k_src + n_blocks_per_entry_;

    size_t page_idx = slot / PAGE_SLOTS;
    page_last_access_[page_idx] = seq_pos;

    for (size_t b = 0; b < blocks_per_head_; ++b) {
        kv_q8_decode(k_src + b, k_out + b * 8);
        kv_q8_decode(v_src + b, v_out + b * 8);
    }

    return true;
}

std::vector<size_t> NSKVStore::evict_sort_order() const {
    // Histogram sort of final eviction score (0..1) into 256 uint8 bins.
    // final_score = 0.1 * recency + 0.9 * v_norm_score.
    std::vector<uint8_t> bin(cfg_.capacity);
    int counts[256] = {};

    for (size_t slot = 0; slot < cfg_.capacity; ++slot) {
        if (valid_[slot]) {
            size_t age = max_seq_pos_ - seq_pos_[slot];
            float recency = 1.0f - std::min((float)age / (float)cfg_.capacity, 1.0f);
            float v_norm_score = info_scores_[slot] / (max_v_norm_ + 1e-9f);
            float final_score = 0.1f * recency + 0.9f * v_norm_score;
            int b = (int)(final_score * 255.0f + 0.5f);
            if (b < 0) b = 0;
            if (b > 255) b = 255;
            bin[slot] = (uint8_t)b;
            counts[b]++;
        }
    }

    // Prefix-sum offsets for each bin (counting sort, O(n)).
    int offset[256];
    int total = 0;
    for (int b = 0; b < 256; ++b) {
        offset[b] = total;
        total += counts[b];
    }

    // Place slot indices in ascending score order (low bin first).
    std::vector<size_t> order(total);
    for (size_t slot = 0; slot < cfg_.capacity; ++slot) {
        if (valid_[slot]) {
            uint8_t b = bin[slot];
            order[offset[b]++] = slot;
        }
    }

    return order;
}

void NSKVStore::evict() {
    if (n_stored_ == 0) return;

    // Histogram sort of final eviction score (0..1) into 256 uint8 bins.
    // final_score = 0.1 * recency + 0.9 * v_norm_score.
    std::vector<uint8_t> bin(cfg_.capacity);
    int counts[256] = {};

    for (size_t slot = 0; slot < cfg_.capacity; ++slot) {
        if (valid_[slot]) {
            size_t age = max_seq_pos_ - seq_pos_[slot];
            float recency = 1.0f - std::min((float)age / (float)cfg_.capacity, 1.0f);
            float v_norm_score = info_scores_[slot] / (max_v_norm_ + 1e-9f);
            float final_score = 0.1f * recency + 0.9f * v_norm_score;
            int b = (int)(final_score * 255.0f + 0.5f);
            if (b < 0) b = 0;
            if (b > 255) b = 255;
            bin[slot] = (uint8_t)b;
            counts[b]++;
        }
    }

    // Prefix-sum offsets for each bin (counting sort, O(n)).
    int offset[256];
    int total = 0;
    for (int b = 0; b < 256; ++b) {
        offset[b] = total;
        total += counts[b];
    }

    // Place slot indices in ascending score order (low bin first).
    std::vector<size_t> order(total);
    for (size_t slot = 0; slot < cfg_.capacity; ++slot) {
        if (valid_[slot]) {
            uint8_t b = bin[slot];
            order[offset[b]++] = slot;
        }
    }

    // Evict the lowest-scoring 25% of the currently stored slots.
    size_t to_evict = (n_stored_ + 3) / 4;
    for (size_t i = 0; i < to_evict; ++i) {
        size_t slot = order[i];
        valid_[slot] = false;
        n_stored_--;
    }

    // Cold-page pass: reclaim OS pages for all-invalid or cold pages.
    size_t n_pages = pages_.size();
    size_t page_bytes = cfg_.n_layers * PAGE_SLOTS * entry_bytes_;
    for (size_t p = 0; p < n_pages; ++p) {
        if (!pages_[p]) continue;

        bool any_valid = false;
        size_t start = p * PAGE_SLOTS;
        size_t end   = std::min(start + PAGE_SLOTS, cfg_.capacity);
        for (size_t slot = start; slot < end; ++slot) {
            if (valid_[slot]) { any_valid = true; break; }
        }

        bool page_cold = (max_seq_pos_ > page_last_access_[p]) &&
                         (max_seq_pos_ - page_last_access_[p] > COLD_SLOTS);

        if (!any_valid || page_cold) {
            // Invalidate any remaining valid slots on this cold/empty page.
            for (size_t slot = start; slot < end; ++slot) {
                if (valid_[slot]) {
                    valid_[slot] = false;
                    n_stored_--;
                }
            }

            // Hint OS to reclaim the physical pages, then unmap and clear.
#ifdef MADV_FREE
            ::madvise(pages_[p], page_bytes, MADV_FREE);
#else
            ::madvise(pages_[p], page_bytes, MADV_DONTNEED);
#endif
            ::munmap(pages_[p], page_bytes);
            pages_[p] = nullptr;
            page_last_access_[p] = 0;
        }
    }
}
