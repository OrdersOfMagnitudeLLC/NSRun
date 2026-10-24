// NSKVCache — OOM LLC Commercial License — see LICENSE-OOM
// KVBox GPU path: device-resident compressed KV store + write/read kernels.
#pragma once
#include <cstdint>
#include <cstddef>

// Device store layout: compressed_kv_store[layer][slot][entry_bytes]
//   entry = [k_codes n_kv_heads*head_dim int8][v_codes n_kv_heads*head_dim int8]
//           [scales n_kv_heads*2 fp16]   (k scale, v scale per head)

extern "C" {
    // Allocate/free the device store. Returns nullptr on failure.
    void * kvbox_cuda_store_alloc(size_t bytes);
    void   kvbox_cuda_store_free(void * p);
    void * kvbox_cuda_malloc(size_t bytes);
    void   kvbox_cuda_free(void * p);
    void   kvbox_cuda_upload(void * dst_dev, const void * src_host, size_t bytes);
    void   kvbox_cuda_download(void * dst_host, const void * src_dev, size_t bytes);
    void   kvbox_cuda_sync();

    // Launch the write kernel: quantize one (pos, layer) slot's K+V from
    // device tensors into compressed_kv_store. k_row/v_row are device ptrs to
    // the slot's row (k_base + cell*k_step). v_trans selects transposed V.
    void   kvbox_cuda_write_slot(const void * k_row_dev, const void * v_row_dev,
            int v_trans, long long v_rows, long long v_cell, size_t v_step,
            const float * rope_freqs_dev, int n_rot, long long pos,
            int n_kv_heads, int head_dim,
            void * store_dev, int layer, int slot, int max_slots);

    // Launch the read/inject kernel: decompress codes back to fp16, re-rotate
    // K, and write the reconstructed K/V rows into the working-cache tensors
    // on device. Symmetric to the write kernel.
    void   kvbox_cuda_read_slot(void * k_row_dev, void * v_row_dev,
            int v_trans, long long v_rows, long long v_cell, size_t v_step,
            const float * rope_freqs_dev, int n_rot, long long pos,
            int n_kv_heads, int head_dim,
            const void * store_dev, int layer, int slot, int max_slots);
}
