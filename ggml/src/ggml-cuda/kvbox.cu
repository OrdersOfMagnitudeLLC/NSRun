// NSKVCache — OOM LLC Commercial License — see LICENSE-OOM
// KVBox GPU write path: INT8-absmax quantization of one slot's K+V, on device.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include "kvbox.cuh"

extern "C" void * kvbox_cuda_store_alloc(size_t bytes) {
    void * p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    cudaMemset(p, 0, bytes);
    return p;
}
extern "C" void kvbox_cuda_store_free(void * p) { if (p) cudaFree(p); }
extern "C" void * kvbox_cuda_malloc(size_t bytes) {
    void * p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    return p;
}
extern "C" void kvbox_cuda_free(void * p) { if (p) cudaFree(p); }
extern "C" void kvbox_cuda_upload(void * dst_dev, const void * src_host, size_t bytes) {
    cudaMemcpy(dst_dev, src_host, bytes, cudaMemcpyHostToDevice);
}
extern "C" void kvbox_cuda_download(void * dst_host, const void * src_dev, size_t bytes) {
    cudaMemcpy(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost);
}
extern "C" void kvbox_cuda_sync() { cudaDeviceSynchronize(); }

// block = head_dim threads (128); grid.x = n_kv_heads; each block = one head.
// grid.y (batch kernel) = token index within the ubatch.
__device__ __forceinline__ void kvbox_write_one(
        const __half * __restrict__ k_row,
        const __half * __restrict__ v_row,
        int v_trans, long long v_rows, long long v_cell,
        const float * __restrict__ rope_freqs, int n_rot, long long pos,
        int n_kv_heads, int head_dim,
        int8_t * __restrict__ k_codes,
        int8_t * __restrict__ v_codes,
        __half * __restrict__ scales) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    __shared__ float s_val[256];
    __shared__ float s_red[8];
    __shared__ float s_scale;
    const int n_pair = n_rot >> 1;

    // ---- K head: load + un-rotate ----
    float ku = __half2float(k_row[(size_t)h * head_dim + d]);
    if (n_rot > 0 && n_rot <= head_dim && rope_freqs && d < n_rot) {
        const int j = (d < n_pair) ? d : (d - n_pair);
        const float theta = (float)(-pos) * rope_freqs[j];
        const float c = cosf(theta), s = sinf(theta);
        const float ka = __half2float(k_row[(size_t)h * head_dim + j]);
        const float kb = __half2float(k_row[(size_t)h * head_dim + j + n_pair]);
        ku = (d < n_pair) ? (ka * c - kb * s) : (ka * s + kb * c);
    }
    s_val[d] = ku;
    // ---- absmax reduce ----
    float mx = fabsf(ku);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
    if ((d & 31) == 0) s_red[d >> 5] = mx;
    __syncthreads();
    if (d == 0) {
        float m = 0.f;
        for (int w = 0; w < head_dim / 32; ++w) m = fmaxf(m, s_red[w]);
        const float sc = m > 0.f ? m / 127.f : 1.f;
        s_scale = sc;
        scales[h * 2 + 0] = __float2half(sc);
    }
    __syncthreads();
    const float inv = 1.0f / s_scale;
    int qi = (int)lroundf(s_val[d] * inv);
    qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
    k_codes[(size_t)h * head_dim + d] = (int8_t)qi;

    // ---- V head ----
    float vv = 0.f;
    if (v_row) {
        if (v_trans) {
            // transposed: v[cell + (h*hdim)*rows + d*rows]
            vv = __half2float(v_row[(size_t)(h * head_dim) * v_rows
                                    + (size_t)d * v_rows + (size_t)v_cell]);
        } else {
            vv = __half2float(v_row[(size_t)h * head_dim + d]);
        }
    }
    float mxv = fabsf(vv);
    for (int off = 16; off > 0; off >>= 1) mxv = fmaxf(mxv, __shfl_xor_sync(0xffffffff, mxv, off));
    if ((d & 31) == 0) s_red[d >> 5] = mxv;
    __syncthreads();
    if (d == 0) {
        float m = 0.f;
        for (int w = 0; w < head_dim / 32; ++w) m = fmaxf(m, s_red[w]);
        const float sc = m > 0.f ? m / 127.f : 1.f;
        s_scale = sc;
        scales[h * 2 + 1] = __float2half(sc);
    }
    __syncthreads();
    const float invv = 1.0f / s_scale;
    int qv = (int)lroundf(vv * invv);
    qv = qv > 127 ? 127 : (qv < -127 ? -127 : qv);
    v_codes[(size_t)h * head_dim + d] = (int8_t)qv;
}

__global__ void k_kvbox_write_slot(
        const __half * __restrict__ k_row,
        const __half * __restrict__ v_row,
        int v_trans, long long v_rows, long long v_cell,
        const float * __restrict__ rope_freqs, int n_rot, long long pos,
        int n_kv_heads, int head_dim,
        int8_t * __restrict__ k_codes,
        int8_t * __restrict__ v_codes,
        __half * __restrict__ scales) {
    kvbox_write_one(k_row, v_row, v_trans, v_rows, v_cell,
        rope_freqs, n_rot, pos, n_kv_heads, head_dim, k_codes, v_codes, scales);
}

extern "C" void kvbox_cuda_write_slot(
        const void * k_row_dev, const void * v_row_dev,
        int v_trans, long long v_rows, long long v_cell, size_t v_step,
        const float * rope_freqs_dev, int n_rot, long long pos,
        int n_kv_heads, int head_dim,
        void * store_dev, int layer, int slot, int max_slots) {
    const size_t entry = (size_t)n_kv_heads * head_dim * 2 + (size_t)n_kv_heads * 2 * sizeof(__half);
    uint8_t * base = (uint8_t *)store_dev;
    uint8_t * ent  = base + ((size_t)layer * max_slots + slot) * entry;
    int8_t  * kc   = (int8_t *)ent;
    int8_t  * vc   = (int8_t *)ent + (size_t)n_kv_heads * head_dim;
    __half  * sc   = (__half *)(ent + (size_t)n_kv_heads * head_dim * 2);
    k_kvbox_write_slot<<<n_kv_heads, head_dim>>>(
        (const __half *)k_row_dev, (const __half *)v_row_dev,
        v_trans, v_rows, v_cell,
        rope_freqs_dev, n_rot, pos,
        n_kv_heads, head_dim, kc, vc, sc);
}

// Batched write: one launch covers n_tokens consecutive cells/positions for a
// layer. grid = (n_kv_heads, n_tokens); block = head_dim. Positions are
// pos_base + t, cells cell_base + t (contiguous in the working cache).
__global__ void k_kvbox_write_batch(
        const char * __restrict__ k_base, const char * __restrict__ v_base,
        int v_trans, long long v_rows, long long cell_base,
        size_t k_step, size_t v_step,
        const float * __restrict__ rope_freqs, int n_rot, long long pos_base,
        int n_kv_heads, int head_dim,
        uint8_t * __restrict__ store_dev, int layer, int max_slots) {
    const int t = blockIdx.y;
    const long long pos  = pos_base + t;
    const long long cell = cell_base + t;
    const long long slot = pos % (long long)max_slots;
    const size_t entry = (size_t)n_kv_heads * head_dim * 2 + (size_t)n_kv_heads * 2 * sizeof(__half);
    uint8_t * ent = store_dev + ((size_t)layer * max_slots + slot) * entry;
    int8_t  * kc  = (int8_t *)ent;
    int8_t  * vc  = (int8_t *)ent + (size_t)n_kv_heads * head_dim;
    __half  * sc  = (__half *)(ent + (size_t)n_kv_heads * head_dim * 2);
    const __half * k_row = (const __half *)(k_base + (size_t)cell * k_step);
    const __half * v_row = v_base ? (const __half *)(v_base + (size_t)cell * v_step) : nullptr;
    kvbox_write_one(k_row, v_row, v_trans, v_rows, cell,
        rope_freqs, n_rot, pos, n_kv_heads, head_dim, kc, vc, sc);
}

extern "C" void kvbox_cuda_write_batch(
        const void * k_base_dev, const void * v_base_dev,
        int v_trans, long long v_rows, long long cell_base,
        size_t k_step, size_t v_step,
        const float * rope_freqs_dev, int n_rot, long long pos_base,
        int n_kv_heads, int head_dim, int n_tokens,
        void * store_dev, int layer, int max_slots) {
    dim3 grid((unsigned)n_kv_heads, (unsigned)n_tokens);
    k_kvbox_write_batch<<<grid, head_dim>>>(
        (const char *)k_base_dev, (const char *)v_base_dev,
        v_trans, v_rows, cell_base, k_step, v_step,
        rope_freqs_dev, n_rot, pos_base,
        n_kv_heads, head_dim, (uint8_t *)store_dev, layer, max_slots);
}

// ---- read/inject kernel: decompress one (pos,layer) slot's K+V and write the
// reconstructed fp16 rows straight into the working-cache tensors on device. ----
// block = head_dim threads; grid.x = n_kv_heads; each block = one head.
__global__ void k_kvbox_read_slot(
        const int8_t * __restrict__ k_codes,
        const int8_t * __restrict__ v_codes,
        const __half * __restrict__ scales,
        __half * __restrict__ k_row,
        __half * __restrict__ v_row,
        int v_trans, long long v_rows, long long v_cell, size_t v_step,
        const float * __restrict__ rope_freqs, int n_rot, long long pos,
        int head_dim) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const int n_pair = n_rot >> 1;
    const float k_sc = __half2float(scales[h * 2 + 0]);
    const float v_sc = __half2float(scales[h * 2 + 1]);

    // ---- K: decompress + re-rotate (opposite sign of the write un-rotate) ----
    float kf;
    if (n_rot > 0 && n_rot <= head_dim && rope_freqs && d < n_rot) {
        const int j = (d < n_pair) ? d : (d - n_pair);
        const float theta = (float)(pos) * rope_freqs[j];
        const float c = cosf(theta), s = sinf(theta);
        const float ka = (float)k_codes[(size_t)h * head_dim + j] * k_sc;
        const float kb = (float)k_codes[(size_t)h * head_dim + j + n_pair] * k_sc;
        kf = (d < n_pair) ? (ka * c - kb * s) : (ka * s + kb * c);
    } else {
        kf = (float)k_codes[(size_t)h * head_dim + d] * k_sc;
    }
    k_row[(size_t)h * head_dim + d] = __float2half(kf);

    // ---- V: decompress, no rotation ----
    float vf = (float)v_codes[(size_t)h * head_dim + d] * v_sc;
    if (v_row) {
        if (v_trans) {
            v_row[(size_t)(h * head_dim) * v_rows + (size_t)d * v_rows + (size_t)v_cell]
                = __float2half(vf);
        } else {
            v_row[(size_t)h * head_dim + d] = __float2half(vf);
        }
    }
}

extern "C" void kvbox_cuda_read_slot(
        void * k_row_dev, void * v_row_dev,
        int v_trans, long long v_rows, long long v_cell, size_t v_step,
        const float * rope_freqs_dev, int n_rot, long long pos,
        int n_kv_heads, int head_dim,
        const void * store_dev, int layer, int slot, int max_slots) {
    const size_t entry = (size_t)n_kv_heads * head_dim * 2 + (size_t)n_kv_heads * 2 * sizeof(__half);
    const uint8_t * base = (const uint8_t *)store_dev;
    const uint8_t * ent  = base + ((size_t)layer * max_slots + slot) * entry;
    const int8_t  * kc   = (const int8_t *)ent;
    const int8_t  * vc   = (const int8_t *)ent + (size_t)n_kv_heads * head_dim;
    const __half  * sc   = (const __half *)(ent + (size_t)n_kv_heads * head_dim * 2);
    k_kvbox_read_slot<<<n_kv_heads, head_dim>>>(
        kc, vc, sc,
        (__half *)k_row_dev, (__half *)v_row_dev,
        v_trans, v_rows, v_cell, v_step,
        rope_freqs_dev, n_rot, pos, head_dim);
}
