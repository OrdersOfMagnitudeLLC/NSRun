//
// NSInfer CUDA backend — sparse MLP activation masking
// MIT license
//
// Mirrors ggml_compute_forward_ns_infer (ggml.c):
//   Phase 1: mean(|activation|) per neuron across the token batch
//   Phase 2: global top-k selection (k = d_ff * keep_fraction) -> keep_mask
//   Phase 3: dst[j,t] = keep_mask[j] ? src[j,t] : 0
// The downstream mul_mat is unchanged; masked-out neurons contribute zeros.
//
#include "ns-infer.cuh"
#include <math_constants.h>

#define NS_INFER_TOPK_BLOCK 1024
#define NS_INFER_TOPK_ITERS 64

// Phase 1: one thread per neuron, strided loop over tokens.
// src is [d_ff, n_tokens] with row stride nb1 (bytes).
static __global__ void k_ns_infer_mean_abs(
        const float * __restrict__ src, int64_t nb1,
        float * __restrict__ mean_abs,
        int64_t d_ff, int64_t n_tokens) {
    const int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (j >= d_ff) return;
    const char * srow = (const char *)src;
    float sum = 0.0f;
    for (int64_t t = 0; t < n_tokens; t++) {
        sum += fabsf(*(const float *)(srow + t * nb1 + j * sizeof(float)));
    }
    mean_abs[j] = sum / (float)n_tokens;
}

// Phase 2: single block. Binary-search a threshold tau such that exactly k
// neurons have mean_abs >= tau (ties broken by atomic ticket order).
// Writes keep_mask[j] in {0,1}.
static __global__ void k_ns_infer_topk_mask(
        const float * __restrict__ mean_abs,
        int * __restrict__ keep_mask,
        int * __restrict__ tie_ctr,
        int64_t d_ff, int64_t k) {
    __shared__ float s_red[NS_INFER_TOPK_BLOCK / 32];
    __shared__ float s_lo, s_hi;
    __shared__ int   s_cnt;

    const int tid = threadIdx.x;

    // min/max reduction over mean_abs
    float lo = CUDART_INF_F, hi = -CUDART_INF_F;
    for (int64_t j = tid; j < d_ff; j += blockDim.x) {
        const float v = mean_abs[j];
        lo = fminf(lo, v);
        hi = fmaxf(hi, v);
    }
    // warp reduce
    for (int off = 16; off > 0; off >>= 1) {
        lo = fminf(lo, __shfl_xor_sync(0xffffffff, lo, off));
        hi = fmaxf(hi, __shfl_xor_sync(0xffffffff, hi, off));
    }
    if ((tid & 31) == 0) { s_red[tid >> 5] = lo; }
    __syncthreads();
    if (tid < 32) {
        float v = (tid < NS_INFER_TOPK_BLOCK / 32) ? s_red[tid] : CUDART_INF_F;
        for (int off = 16; off > 0; off >>= 1) v = fminf(v, __shfl_xor_sync(0xffffffff, v, off));
        if (tid == 0) s_lo = v;
    }
    __syncthreads();
    if ((tid & 31) == 0) { s_red[tid >> 5] = hi; }
    __syncthreads();
    if (tid < 32) {
        float v = (tid < NS_INFER_TOPK_BLOCK / 32) ? s_red[tid] : -CUDART_INF_F;
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
        if (tid == 0) { s_hi = v; *tie_ctr = 0; }
    }
    __syncthreads();

    float lo_v = s_lo, hi_v = s_hi;

    // binary search: largest tau with count(mean_abs > tau) <= k
    for (int it = 0; it < NS_INFER_TOPK_ITERS; it++) {
        const float mid = 0.5f * (lo_v + hi_v);
        int cnt = 0;
        for (int64_t j = tid; j < d_ff; j += blockDim.x) {
            cnt += (mean_abs[j] > mid) ? 1 : 0;
        }
        for (int off = 16; off > 0; off >>= 1) cnt += __shfl_xor_sync(0xffffffff, cnt, off);
        if ((tid & 31) == 0) s_red[tid >> 5] = (float)cnt;
        __syncthreads();
        if (tid < 32) {
            int c = (tid < NS_INFER_TOPK_BLOCK / 32) ? (int)s_red[tid] : 0;
            for (int off = 16; off > 0; off >>= 1) c += __shfl_xor_sync(0xffffffff, c, off);
            if (tid == 0) s_cnt = c;
        }
        __syncthreads();
        const int total = s_cnt;
        __syncthreads();
        if (total > k) lo_v = mid; else hi_v = mid;
    }

    const float tau = hi_v; // count(> tau) <= k <= count(>= tau)

    // mask = mean_abs > tau; admit (k - count_gt) ties via atomic ticket
    __shared__ int s_need;

    int cnt_gt = 0;
    for (int64_t j = tid; j < d_ff; j += blockDim.x) {
        cnt_gt += (mean_abs[j] > tau) ? 1 : 0;
    }
    for (int off = 16; off > 0; off >>= 1) cnt_gt += __shfl_xor_sync(0xffffffff, cnt_gt, off);
    if ((tid & 31) == 0) s_red[tid >> 5] = (float)cnt_gt;
    __syncthreads();
    if (tid < 32) {
        int c = (tid < NS_INFER_TOPK_BLOCK / 32) ? (int)s_red[tid] : 0;
        for (int off = 16; off > 0; off >>= 1) c += __shfl_xor_sync(0xffffffff, c, off);
        if (tid == 0) s_need = (int)k - c;
    }
    __syncthreads();
    const int need = s_need;

    for (int64_t j = tid; j < d_ff; j += blockDim.x) {
        const float v = mean_abs[j];
        int keep = 0;
        if (v > tau) {
            keep = 1;
        } else if (need > 0 && v == tau) {
            const int ticket = atomicAdd(tie_ctr, 1);
            keep = (ticket < need) ? 1 : 0;
        }
        keep_mask[j] = keep;
    }
}

// Phase 3: apply mask. 2D grid: x = neuron j, y = token t.
static __global__ void k_ns_infer_apply(
        const float * __restrict__ src, int64_t src_nb1,
        const int * __restrict__ keep_mask,
        float * __restrict__ dst, int64_t dst_nb1,
        int64_t d_ff, int64_t n_tokens) {
    const int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    if (j >= d_ff || t >= n_tokens) return;
    const float v = *(const float *)((const char *)src + t * src_nb1 + j * sizeof(float));
    *(float *)((char *)dst + t * dst_nb1 + j * sizeof(float)) = keep_mask[j] ? v : 0.0f;
}

void ggml_cuda_op_ns_infer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    float keep_fraction;
    memcpy(&keep_fraction, dst->op_params, sizeof(float));

    const int64_t d_ff     = src->ne[0];
    const int64_t n_tokens = ggml_nrows(src);

    int64_t k = (int64_t)((float)d_ff * keep_fraction);
    if (k < 1)     k = 1;
    if (k > d_ff)  k = d_ff;

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) || src->nb[0] == sizeof(float));

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    ggml_cuda_pool_alloc<float> mean_abs_alloc(pool, d_ff);
    ggml_cuda_pool_alloc<int>   keep_mask_alloc(pool, d_ff);
    ggml_cuda_pool_alloc<int>   tie_ctr_alloc(pool, 1);

    float * mean_abs  = mean_abs_alloc.get();
    int   * keep_mask = keep_mask_alloc.get();
    int   * tie_ctr   = tie_ctr_alloc.get();

    // Phase 1
    {
        const int block = 256;
        const int grid  = (int)((d_ff + block - 1) / block);
        k_ns_infer_mean_abs<<<grid, block, 0, stream>>>(
            (const float *)src->data, src->nb[1], mean_abs, d_ff, n_tokens);
    }

    // Phase 2
    k_ns_infer_topk_mask<<<1, NS_INFER_TOPK_BLOCK, 0, stream>>>(
        mean_abs, keep_mask, tie_ctr, d_ff, k);

    // Phase 3
    {
        const int block = 256;
        dim3 grid((unsigned)((d_ff + block - 1) / block), (unsigned)n_tokens, 1);
        k_ns_infer_apply<<<grid, block, 0, stream>>>(
            (const float *)src->data, src->nb[1], keep_mask,
            (float *)dst->data, dst->nb[1], d_ff, n_tokens);
    }
}
