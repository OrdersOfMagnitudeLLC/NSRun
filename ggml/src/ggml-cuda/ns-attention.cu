//
// NSAttend CUDA backend — faithful port of ggml_compute_forward_ns_attention (ggml.c)
// MIT license
//
// Semantics mirrored from the CPU implementation:
//   - head classification: when n_tokens>=16 && n_kv>=64, 8 sampled query rows are
//     scored over the full KV; Pearson correlation of attention weight vs seq-pos
//     distance decides LOCAL (|corr| avg > 0.3) vs UNIFORM heads.
//   - LOCAL heads: attend only q_pos +/- (n_kv/4) window (pos-based when kvp given).
//   - UNIFORM heads: full attention.
//   - scores = q.k * scale (+ mask); softmax; weighted V sum. Masked-out slots use
//     the -1e30f sentinel exactly like the CPU path.
//
// No score scratch is allocated: each block does a chunked online-softmax pass for
// max/sum, then re-scores and accumulates weighted V. Works for any n_kv.
//
#include "ns-attention.cuh"
#include <math_constants.h>

#define NS_ATTN_THREADS 128
#define NS_ATTN_CHUNK   512
#define NS_ATTN_MASKED  (-1e30f)

typedef int32_t ns_i32;

// ---- helpers ---------------------------------------------------------------

__device__ __forceinline__ float ns_fetch_f32(const char * p, ggml_type t) {
    return t == GGML_TYPE_F16 ? __half2float(*(const __half *)p) : *(const float *)p;
}

__device__ __forceinline__ int64_t ns_min_i64(int64_t a, int64_t b) {
    return a < b ? a : b;
}

// block-wide reduction of a (m, s) online-softmax pair via shared memory.
// s_m / s_s hold one value per warp; s_pair[0]=max, s_pair[1]=sum after call.
static __device__ void ns_block_reduce_ms(
        float m, float s, float * warp_m, float * warp_s, float * pair) {
    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int nwarp = (blockDim.x + 31) >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        const float mo = __shfl_xor_sync(0xffffffff, m, off);
        const float so = __shfl_xor_sync(0xffffffff, s, off);
        if (mo > m) {
            // mo finite here (mo > m > -inf or mo > -inf); m may be -inf -> expf(-inf)=0 ok
            s = s * expf(m - mo) + so; m = mo;
        } else if (mo != -CUDART_INF_F || m != -CUDART_INF_F) {
            // guard: -inf - -inf = NaN; both empty -> no contribution
            s = s + so * expf(mo - m);
        }
    }
    if (lane == 0) { warp_m[warp] = m; warp_s[warp] = s; }
    __syncthreads();
    if (warp == 0) {
        float bm = (lane < nwarp) ? warp_m[lane] : -CUDART_INF_F;
        float bs = (lane < nwarp) ? warp_s[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            const float mo = __shfl_xor_sync(0xffffffff, bm, off);
            const float so = __shfl_xor_sync(0xffffffff, bs, off);
            if (mo > bm) { bs = bs * expf(bm - mo) + so; bm = mo; }
            else if (mo != -CUDART_INF_F || bm != -CUDART_INF_F) {
                bs = bs + so * expf(mo - bm);
            }
        }
        if (lane == 0) { pair[0] = bm; pair[1] = bs; }
    }
    __syncthreads();
}

// block-wide max of int32 (for ns_max_pos scan)
static __device__ int ns_block_reduce_max_i32(int v, int * warp_v, int * out) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int off = 16; off > 0; off >>= 1) v = max(v, __shfl_xor_sync(0xffffffff, v, off));
    if (lane == 0) warp_v[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int bv = (lane < (int)((blockDim.x + 31) >> 5)) ? warp_v[lane] : INT_MIN;
        for (int off = 16; off > 0; off >>= 1) bv = max(bv, __shfl_xor_sync(0xffffffff, bv, off));
        if (lane == 0) *out = bv;
    }
    __syncthreads();
    return *out;
}

// dot product of one q row (in smem, Dk floats) with K row j (typed, strided)
static __device__ float ns_dot(
        const float * __restrict__ q_s,
        const char  * __restrict__ k_ptr, size_t nb0, ggml_type ktype,
        int64_t Dk) {
    float acc = 0.0f;
    for (int64_t d = 0; d < Dk; d++) {
        acc += q_s[d] * ns_fetch_f32(k_ptr + d * nb0, ktype);
    }
    return acc;
}

// ---- kernel 1: head classification -----------------------------------------
// one block per q-head; writes head_type[h] in {0,1}
static __global__ void k_ns_attn_classify(
        const char * __restrict__ q_data,
        size_t qb0, size_t qb1, size_t qb2, ggml_type qtype,
        const char * __restrict__ k_data,
        size_t kb1, size_t kb0, size_t kb2, ggml_type ktype,
        const ns_i32 * __restrict__ kvp,
        int * __restrict__ head_type, int flag_idx,
        int64_t n_tokens, int64_t n_kv, int64_t n_head, int64_t n_head_kv,
        int64_t Dk, float scale) {
    extern __shared__ char smem_raw[];
    float * q_s    = (float *)smem_raw;                        // [Dk]
    float * warp_m = q_s + Dk;                                 // [nwarp]
    float * warp_s = warp_m + 32;                              // [32]
    float * pair   = warp_s + 32;                              // [2]
    float * corr_a = pair + 2;                                 // [32*3] warp partials (num,dd,dw)
    int   * warp_v = (int *)(corr_a + 96);                     // [32]
    float * s_corr = (float *)(warp_v + 32);                   // [1]
    int   * s_maxp = (int *)(s_corr + 1);                      // [1]

    const int h  = blockIdx.x;
    const int hk = (int)((int64_t)h * n_head_kv / n_head);
    const int tid = threadIdx.x;
    const char * k_base = k_data + hk * kb2;

    // flag_idx>=0 (persistent): classify once; head_type[flag_idx]==1 => reuse.
    // flag_idx<0  (scratch):    legacy behaviour (only classify a token batch).
    if (flag_idx >= 0) {
        if (head_type[flag_idx] == 1) return;
    } else if (n_tokens < 16) {
        if (tid == 0) head_type[h] = 0;
        return;
    }
    if (n_kv < 64) {
        if (tid == 0) head_type[h] = 0;
        return;
    }

    int ns_max_pos = -1;
    if (kvp) {
        int vm = INT_MIN;
        for (int64_t j2 = tid; j2 < n_kv; j2 += blockDim.x) vm = max(vm, (int)kvp[j2]);
        ns_max_pos = ns_block_reduce_max_i32(vm, warp_v, s_maxp);
    }
    __syncthreads();

    const int sample_rows = 8;
    float total_corr = 0.0f;

    for (int s = 0; s < sample_rows; s++) {
        const int64_t i = ns_min_i64((int64_t)s * n_tokens / sample_rows, n_tokens - 1);
        // stage q row
        const char * q_ptr = q_data + h * qb2 + i * qb1;
        for (int64_t d = tid; d < Dk; d += blockDim.x) q_s[d] = ns_fetch_f32(q_ptr + d * qb0, qtype);
        __syncthreads();

        // pass 1: max + exp-sum over all j
        float m = -CUDART_INF_F, ss = 0.0f;
        for (int64_t j = tid; j < n_kv; j += blockDim.x) {
            const float sc = ns_dot(q_s, k_base + j * kb1, kb0, ktype, Dk) * scale;
            const float mn = fmaxf(m, sc);
            ss = ss * expf(m - mn) + expf(sc - mn);
            m = mn;
        }
        ns_block_reduce_ms(m, ss, warp_m, warp_s, pair);
        const float bm = pair[0], bs = pair[1];

        // pass 2: Pearson corr of w vs |q_pos - sp|
        const int q_pos = kvp ? ns_max_pos - (int)(n_tokens - 1 - i)
                              : (int)(n_kv - n_tokens + i);
        const float mean_w = bs / (float)n_kv;
        float num = 0.0f, den_d = 0.0f, den_w = 0.0f;
        for (int64_t j = tid; j < n_kv; j += blockDim.x) {
            const int sp = kvp ? (int)kvp[j] : (int)j;
            if (sp < 0) continue;
            const float sc = ns_dot(q_s, k_base + j * kb1, kb0, ktype, Dk) * scale;
            const float w = expf(sc - bm) / bs;
            const float dist = fabsf((float)(q_pos - sp));
            const float d_diff = dist - (float)(n_kv / 2.0f);
            const float w_diff = w - mean_w;
            num   += d_diff * w_diff;
            den_d += d_diff * d_diff;
            den_w += w_diff * w_diff;
        }
        // block reduce 3 sums
        for (int off = 16; off > 0; off >>= 1) {
            num   += __shfl_xor_sync(0xffffffff, num, off);
            den_d += __shfl_xor_sync(0xffffffff, den_d, off);
            den_w += __shfl_xor_sync(0xffffffff, den_w, off);
        }
        if ((tid & 31) == 0) { corr_a[tid>>5] = num; corr_a[32 + (tid>>5)] = den_d; corr_a[64 + (tid>>5)] = den_w; }
        __syncthreads();
        if (tid < 32) {
            const int nwarp = (blockDim.x + 31) >> 5;
            float a = (tid < nwarp) ? corr_a[tid] : 0.0f;
            float b = (tid < nwarp) ? corr_a[32 + tid] : 0.0f;
            float c = (tid < nwarp) ? corr_a[64 + tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) {
                a += __shfl_xor_sync(0xffffffff, a, off);
                b += __shfl_xor_sync(0xffffffff, b, off);
                c += __shfl_xor_sync(0xffffffff, c, off);
            }
            if (tid == 0) {
                const float corr = (b > 0.0f && c > 0.0f) ? a / sqrtf(b * c) : 0.0f;
                *s_corr = corr;
            }
        }
        __syncthreads();
        total_corr += fabsf(*s_corr);
        __syncthreads();
    }

    if (tid == 0) {
        head_type[h] = (total_corr / (float)sample_rows > 0.3f) ? 1 : 0;
        if (flag_idx >= 0 && h == 0) head_type[flag_idx] = 1;   // mark classified
    }
}


// ---- kernel 2: attention forward (single-pass) -------------------------------
// one block per (token, head). Warp-per-j scoring into smem chunks, then
// online (m,s) merge + weighted-V accumulate per chunk. K is read once, V once.
static __global__ void k_ns_attn_forward(
        const char * __restrict__ q_data,
        size_t qb0, size_t qb1, size_t qb2, ggml_type qtype,
        const char * __restrict__ k_data,
        size_t kb1, size_t kb0, size_t kb2, ggml_type ktype,
        const char * __restrict__ v_data,
        size_t vb1, size_t vb0, size_t vb2, ggml_type vtype,
        const char * __restrict__ mask_data,
        size_t mb0, size_t mb1, ggml_type mtype,
        const ns_i32 * __restrict__ kvp,
        const int * __restrict__ head_type,
        float * __restrict__ dst_data,
        size_t db1, size_t db2,
        float * __restrict__ scratch, int n_splits,
        int64_t n_tokens, int64_t n_kv, int64_t n_head, int64_t n_head_kv,
        int64_t Dk, int64_t Dv, float scale) {
    extern __shared__ char smem_raw[];
    float * q_s    = (float *)smem_raw;                        // [Dk]
    float * scores = q_s + Dk;                                 // [NS_ATTN_CHUNK]
    float * acc    = scores + NS_ATTN_CHUNK;                   // [Dv]
    float * warp_m = acc + Dv;                                 // [32]
    float * warp_s = warp_m + 32;                              // [32]
    float * pair   = warp_s + 32;                              // [2]
    int   * warp_v = (int *)(pair + 2);                        // [32]

    const int i  = blockIdx.x;
    const int h  = blockIdx.y;
    const int hk = (int)((int64_t)h * n_head_kv / n_head);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarp = (int)((blockDim.x + 31) >> 5);

    const char * k_base = k_data + hk * kb2;
    const char * v_base = v_data + hk * vb2;
    const int head_local = head_type[h];

    int ns_max_pos = -1;
    if (kvp) {
        int vm = INT_MIN;
        for (int64_t j2 = tid; j2 < n_kv; j2 += blockDim.x) vm = max(vm, (int)kvp[j2]);
        ns_max_pos = ns_block_reduce_max_i32(vm, warp_v, (int *)pair);
        __syncthreads();
    }

    const char * q_ptr = q_data + h * qb2 + i * qb1;
    for (int64_t d = tid; d < Dk; d += blockDim.x) q_s[d] = ns_fetch_f32(q_ptr + d * qb0, qtype);
    for (int64_t d = tid; d < Dv; d += blockDim.x) acc[d] = 0.0f;
    __syncthreads();

    const int q_seq_pos = kvp ? ns_max_pos - (int)(n_tokens - 1 - i)
                              : (int)(n_kv - n_tokens + i);
    int64_t local_window = n_kv / 4;
    if (local_window < 1) local_window = 1;

    int64_t j_lo = 0, j_hi = n_kv;
    if (head_local && !kvp) {
        j_lo = (q_seq_pos - local_window) > 0 ? (int64_t)(q_seq_pos - local_window) : 0;
        j_hi = ns_min_i64((int64_t)(q_seq_pos + local_window + 1), n_kv);
    }
    // split-KV: this block covers only its slice of [j_lo, j_hi)
    {
        const int64_t span = (j_hi - j_lo + n_splits - 1) / n_splits;
        j_lo += (int64_t)blockIdx.z * span;
        j_hi  = ns_min_i64(j_hi, j_lo + span);
        if (j_lo >= j_hi) j_lo = j_hi; // empty slice
    }

    float bm = -CUDART_INF_F, bs = 0.0f;

    for (int64_t j0 = j_lo; j0 < j_hi; j0 += NS_ATTN_CHUNK) {
        const int64_t jlen = ns_min_i64((int64_t)NS_ATTN_CHUNK, j_hi - j0);

        // --- stage scores for this chunk: warp-per-j coalesced dot ---
        for (int64_t jj = warp; jj < jlen; jj += nwarp) {
            const int64_t j = j0 + jj;
            float sc = NS_ATTN_MASKED;
            bool masked = false;
            if (head_local && kvp) {
                const int sp = (int)kvp[j];
                if (sp < 0 || sp < q_seq_pos - local_window || sp > q_seq_pos + local_window) masked = true;
            }
            if (!masked) {
                // coalesced dot: lanes split Dk, f16 path reads 4B pairs
                float part = 0.0f;
                const char * kp = k_base + j * kb1;
                if (ktype == GGML_TYPE_F16 && kb0 == 2 && (Dk & 1) == 0) {
                    for (int64_t d = lane * 2; d < Dk; d += 64) {
                        const __half2 k2 = *(const __half2 *)(kp + d * 2);
                        part += q_s[d] * __low2float(k2) + q_s[d + 1] * __high2float(k2);
                    }
                } else {
                    for (int64_t d = lane; d < Dk; d += 32) {
                        part += q_s[d] * ns_fetch_f32(kp + d * kb0, ktype);
                    }
                }
                for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffff, part, off);
                sc = part * scale;
                if (mask_data) sc += ns_fetch_f32(mask_data + i * mb1 + j * mb0, mtype);
            }
            if (lane == 0) scores[jj] = sc;
        }
        __syncthreads();

        // --- block online (m,s) over the chunk (skip masked) ---
        float cm = -CUDART_INF_F, cs = 0.0f;
        for (int64_t jl = tid; jl < jlen; jl += blockDim.x) {
            const float sc = scores[jl];
            if (!(sc > NS_ATTN_MASKED)) continue;
            const float mn = fmaxf(cm, sc);
            cs = cs * expf(cm - mn) + expf(sc - mn);
            cm = mn;
        }
        ns_block_reduce_ms(cm, cs, warp_m, warp_s, pair);
        const float m_c = pair[0], s_c = pair[1];

        if (m_c == -CUDART_INF_F) { __syncthreads(); continue; } // all masked

        const float m_new = fmaxf(bm, m_c);
        const float r1 = expf(bm - m_new);
        const float r2 = expf(m_c - m_new);
        bs = bs * r1 + s_c * r2;

        // --- write back weights w = expf(sc - m_new), 0 if masked ---
        for (int64_t jl = tid; jl < jlen; jl += blockDim.x) {
            const float sc = scores[jl];
            scores[jl] = !(sc > NS_ATTN_MASKED) ? 0.0f : expf(sc - m_new);
        }
        __syncthreads();

        // --- V accumulate: out[d] = out[d]*r1 + sum_jl w * v[j][d] ---
        for (int64_t d = tid; d < Dv; d += blockDim.x) {
            float a = acc[d] * r1;
            const char * v_row = v_base + j0 * vb1 + d * vb0;
            for (int64_t jl = 0; jl < jlen; jl++) {
                const float w = scores[jl];
                if (w == 0.0f) continue;
                a += w * ns_fetch_f32(v_row + jl * vb1, vtype);
            }
            acc[d] = a;
        }
        bm = m_new;
        __syncthreads();
    }

    // write partial result: scratch[(i,h,split)] = [acc[Dv], m, s]
    float * pout = scratch + (((int64_t)i * n_head + h) * n_splits + blockIdx.z) * (Dv + 2);
    for (int64_t d = tid; d < Dv; d += blockDim.x) pout[d] = acc[d];
    if (tid == 0) { pout[Dv] = bm; pout[Dv + 1] = bs; }
}

// combine kernel: merge per-split partials into final dst
static __global__ void k_ns_attn_combine(
        const float * __restrict__ scratch, int n_splits,
        float * __restrict__ dst_data, size_t db1, size_t db2,
        int64_t n_head, int64_t Dv) {
    const int i = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x;
    const float * base = scratch + ((int64_t)i * n_head + h) * n_splits * (Dv + 2);
    // merge all splits: m_final = max m_s (skip -inf/NaN), rescale
    float mf = -CUDART_INF_F;
    for (int x = 0; x < n_splits; x++) {
        const float m_s = base[x * (Dv + 2) + Dv];
        if (m_s > mf) mf = m_s;
    }
    if (mf == -CUDART_INF_F) {
        for (int64_t d = tid; d < Dv; d += blockDim.x)
            *(float *)((char *)dst_data + h * db1 + i * db2 + d * sizeof(float)) = 0.0f;
        return;
    }
    float sf = 0.0f;
    for (int x = 0; x < n_splits; x++) {
        const float m_s = base[x * (Dv + 2) + Dv];
        const float s_s = base[x * (Dv + 2) + Dv + 1];
        if (m_s != -CUDART_INF_F) sf += s_s * expf(m_s - mf);
    }
    for (int64_t d = tid; d < Dv; d += blockDim.x) {
        float a = 0.0f;
        for (int x = 0; x < n_splits; x++) {
            const float m_s = base[x * (Dv + 2) + Dv];
            if (m_s == -CUDART_INF_F) continue;
            a += base[x * (Dv + 2) + d] * expf(m_s - mf);
        }
        *(float *)((char *)dst_data + h * db1 + i * db2 + d * sizeof(float)) = sf > 0.0f ? a / sf : 0.0f;
    }
}

// ---- host wrapper ------------------------------------------------------------

void ggml_cuda_op_ns_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q_t   = dst->src[0];
    const ggml_tensor * k_t   = dst->src[1];
    const ggml_tensor * v_t   = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * kvpos = dst->src[4];
    const ggml_tensor * head_type_t = dst->src[5];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int64_t Dk        = q_t->ne[0];
    const int64_t n_tokens  = q_t->ne[1];
    const int64_t n_head    = q_t->ne[2];
    const int64_t n_kv      = k_t->ne[1];
    const int64_t n_head_kv = k_t->ne[2];
    const int64_t Dv        = v_t->ne[0];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!kvpos || kvpos->type == GGML_TYPE_I32);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    // Persistent head-locality buffer (src[5]): classify writes it at prefill
    // (n_tokens>=16); at decode (n_tokens<16) the classify launch is skipped and
    // the forward kernel reads the persisted flags so LOCAL heads keep their
    // n_kv/4 window. When src[5] is NULL we fall back to per-launch scratch.
    const bool have_persist = (head_type_t != nullptr) && (head_type_t->data != nullptr);
    ggml_cuda_pool_alloc<int> head_type_alloc(pool, have_persist ? 0 : n_head);
    int * head_type = have_persist ? (int *)head_type_t->data : head_type_alloc.get();

    const ns_i32 * kvp = kvpos ? (const ns_i32 *)kvpos->data : nullptr;


    // forward: one block per (token, head, split); splits used for small n_tokens
    const int n_splits = n_tokens < 16 ? 4 : 1;
    ggml_cuda_pool_alloc<float> scratch_alloc(ctx.pool(), (size_t)n_tokens * n_head * n_splits * (Dv + 2));
    float * scratch = scratch_alloc.get();
    {
        dim3 grid((unsigned)n_tokens, (unsigned)n_head, (unsigned)n_splits);
        const size_t smem = (size_t)(Dk + NS_ATTN_CHUNK + Dv + 32 + 32 + 2 + 32) * sizeof(float) + 64;
        k_ns_attn_forward<<<grid, NS_ATTN_THREADS, smem, stream>>>(
            (const char *)q_t->data, q_t->nb[0], q_t->nb[1], q_t->nb[2], q_t->type,
            (const char *)k_t->data, k_t->nb[1], k_t->nb[0], k_t->nb[2], k_t->type,
            (const char *)v_t->data, v_t->nb[1], v_t->nb[0], v_t->nb[2], v_t->type,
            mask ? (const char *)mask->data : nullptr,
            mask ? mask->nb[0] : 0, mask ? mask->nb[1] : 0,
            mask ? mask->type : GGML_TYPE_F32,
            kvp, head_type,
            (float *)dst->data, dst->nb[1], dst->nb[2],
            scratch, n_splits,
            n_tokens, n_kv, n_head, n_head_kv, Dk, Dv, scale);
    }
    k_ns_attn_combine<<<dim3((unsigned)n_tokens, (unsigned)n_head, 1), NS_ATTN_THREADS, 0, stream>>>(
        scratch, n_splits, (float *)dst->data, dst->nb[1], dst->nb[2], n_head, Dv);
}
