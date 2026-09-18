#pragma once

#include "vendor/ns_dequant.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cassert>

// Compatibility with llama.cpp source we are inlining
#define GGML_RESTRICT

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a): (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a): (b))
#endif

#define GGML_FP32_TO_FP16(x) f32_to_fp16(x)

inline uint16_t f32_to_fp16(float x) {
    union { float f; uint32_t u; } bits = { x };
    uint32_t u = bits.u;
    uint16_t s = (uint16_t)((u >> 16) & 0x8000);
    uint32_t e = (u >> 23) & 0xFF;
    uint32_t m = u & 0x7FFFFF;
    if (e == 255) return s | 0x7C00; // inf/nan
    int32_t ne = (int32_t)e - 127 + 15; // FP16 biased exponent
    if (ne >= 31) return s | 0x7C00; // overflow -> inf
    if (ne <= 0) {
        // FP16 subnormal: value = mant10 * 2^-24. Preserve small values as
 // denormals instead of flushing to zero - flushing produced d=0
        // super-block scales that flattened Q4_K blocks (the gibberish bug).
        if (ne < -10) return s; // too small -> 0
        m |= 0x800000; // restore implicit leading 1
        int shift = 14 - ne;
        uint32_t half = (m + (1u << (shift - 1))) >> shift; // round to nearest
        return s | (uint16_t)half;
    }
    uint32_t nm = (m + 0x1000) >> 13; // round mantissa to 10 bits
    if (nm == 0x800) { nm = 0; ne++; if (ne >= 31) return s | 0x7C00; }
    return s | (uint16_t)((ne << 10) | nm);
}

inline void set_scale_min_k4(int j, uint8_t* q, uint8_t d, uint8_t m) {
    if (j < 4) {
        q[j] = (q[j] & 0xC0) | (d & 0x3F);
        q[j + 4] = (q[j + 4] & 0xC0) | (m & 0x3F);
    } else {
        q[j + 4] = (q[j + 4] & 0x00) | (d & 0x0F) | ((m & 0x0F) << 4);
        q[j - 4] = (q[j - 4] & 0x3F) | ((d & 0x30) << 2); // upper 2 bits of d
        q[j - 0] = (q[j - 0] & 0x3F) | ((m & 0x30) << 2); // upper 2 bits of m
    }
}

inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

inline float make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

// Least-squares fit of the super-block scale: returns d such that x[i] ~= d*L[i]
// with L[i] in [0,nmax]. Unlike max/63 this never underflows to FP16 zero for
// small-magnitude blocks (the root cause of the Q4_K gibberish).
inline float make_qp_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L,
                            const float * GGML_RESTRICT quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) max = MAX(max, x[i]);
    if (max < 1e-16f) { for (int i = 0; i < n; ++i) L[i] = 0; return 0.f; }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) L[i] = nearest_int(iscale * x[i]);
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) { float diff = x[i] - scale*L[i]; float w = quant_weights[i]; best_mse += w*diff*diff; }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f*is + nmax)/max;
        float scale_is = 1/iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is*x[i]); l = MIN(nmax, l);
            float diff = x[i] - scale_is*l; float w = quant_weights[i]; mse += w*diff*diff;
        }
        if (mse < best_mse) { best_mse = mse; iscale = iscale_is; }
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]); l = MIN(nmax, l); L[i] = l;
        float w = quant_weights[i]; sumlx += w*x[i]*l; suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i];
            float sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx); new_l = MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l; sl2 += w*new_l*new_l;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) { L[i] = new_l; sumlx = slx; suml2 = sl2; ++n_changed; }
                }
            }
        }
        if (!n_changed) break;
    }
    return sumlx/suml2;
}

// Importance-weighted sub-block quantizer used by quantize_row_q4_K_impl.
inline float make_qkx3_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0], max = x[0];
    double sum_w = weights ? (double)weights[0] : (double)(x[0]*x[0]);
    double sum_x = sum_w * (double)x[0];
    double sum_x2 = sum_w * (double)x[0] * (double)x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w += (double)w; sum_x += (double)w * (double)x[i]; sum_x2 += (double)w * (double)x[i] * (double)x[i];
    }
    if (min > 0) min = 0;
    if (max - min < 1e-10f) { memset(L, 0, n); *the_min = -min; return 0.f; }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    double best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min)); L[i] = MAX(0, MIN(nmax, l));
        double diff = (double)scale * L[i] + (double)min - (double)x[i];
        diff = use_mad ? fabs(diff) : diff*diff;
        double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        best_mad += w * diff;
    }
    if (nstep < 1) { *the_min = -min; return scale; }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        double sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min)); l = MAX(0, MIN(nmax, l)); Laux[i] = l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l += (double)w*l; sum_l2 += (double)w*l*l; sum_xl += (double)w*l*(double)x[i];
        }
        double D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            double this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            double this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) { this_min = 0; this_scale = sum_xl / sum_l2; }
            double mad = 0;
            if (use_mad) {
                for (int i = 0; i < n; ++i) {
                    double diff = (double)this_scale * Laux[i] + (double)this_min - (double)x[i];
                    diff = fabs(diff); double w = weights ? (double)weights[i] : (double)(x[i]*x[i]); mad += w * diff;
                }
            } else {
                mad = sum_x2 - 2*this_scale*sum_xl - 2*this_min*sum_x + 2*this_scale*this_min*sum_l
                    + this_scale*this_scale*sum_l2 + this_min*this_min*sum_w;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_mad = mad; scale = (float)this_scale; min = (float)this_min;
            }
        }
    }
    if (use_mad) { *the_min = -min; return scale; }
    double sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
        int l = L[i]; double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        sum_l += w*l; sum_l2 += w*l*l; sum_xl += w*l*(double)x[i];
    }
    double best = 2*(double)scale*sum_xl + 2*(double)min*sum_x - 2*(double)scale*(double)min*sum_l
                - (double)scale*(double)scale*sum_l2 - (double)min*(double)min*sum_w;
    int last_j = -1, last_dir = 0;
    for (int itry = 0; itry < nmax*n; ++itry) {
        float gmax = 0; int best_j = -1, dir = 0;
        for (int j = 0; j < n; ++j) {
            float g = x[j] - scale*L[j] - min;
            if (g > 0 && L[j] < nmax && g > gmax) { gmax = g; best_j = j; dir = 1; }
            else if (g < 0 && L[j] > 0 && -g > gmax) { gmax = -g; best_j = j; dir = -1; }
        }
        if (best_j < 0 || (best_j == last_j && dir == -last_dir)) break;
        double w = weights ? (double)weights[best_j] : (double)(x[best_j]*x[best_j]);
        sum_l += w*dir; sum_l2 += w*(2*L[best_j]*dir + 1); sum_xl += w*(double)x[best_j]*dir;
        double D = (double)sum_w * sum_l2 - sum_l * sum_l;
        if (D <= 0) break;
        double this_scale = ((double)sum_w * sum_xl - (double)sum_x * sum_l)/D;
        double this_min   = (sum_l2 * (double)sum_x - sum_l * sum_xl)/D;
        if (this_min > 0) { this_min = 0; this_scale = sum_xl / sum_l2; }
        if (this_scale < 0) break;
        double score = 2*this_scale*sum_xl + 2*this_min*(double)sum_x - 2*this_scale*this_min*sum_l
                     - this_scale*this_scale*sum_l2 - this_min*this_min*(double)sum_w;
        if (score <= best) break;
        best = score; scale = (float)this_scale; min = (float)this_min;
        L[best_j] += dir; last_j = best_j; last_dir = dir;
    }
    *the_min = -min;
    return scale;
}

inline void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

// Importance-weighted Q4_K quantizer (what llama.cpp actually uses for model
// weights). Uses make_qp_quants to fit the super-block scale d, which avoids the
// d=0 FP16 underflow that quantize_row_q4_K_ref produces on small-magnitude
// blocks. quant_weights may be NULL -> falls back to av_x + |x| weighting.
inline void quantize_row_q4_K_impl(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y,
                                   int64_t n_per_row, const float * quant_weights) {
    assert(n_per_row % QK_K == 0);
    const int64_t nb = n_per_row / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K/32];
    uint8_t Lm[QK_K/32];
    float   weights[32];
    float   sw[QK_K/32];
    float   mins[QK_K/32];
    float   scales[QK_K/32];

    for (int i = 0; i < nb; i++) {
        float sum_x2 = 0;
        for (int l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];
        float sigma2 = 2*sum_x2/QK_K;
        float av_x = sqrtf(sigma2);

        for (int j = 0; j < QK_K/32; ++j) {
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*i + 32*j;
                for (int l = 0; l < 32; ++l) weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
            } else {
                for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            }
            float sumw = 0;
            for (int l = 0; l < 32; ++l) sumw += weights[l];
            sw[j] = sumw;
            scales[j] = make_qkx3_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
        }

        float d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
        float m_block = make_qp_quants(QK_K/32, 63, mins,   Lm, sw);
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = Ls[j];
            uint8_t lm = Lm[j];
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);
        y[i].dmin = GGML_FP32_TO_FP16(m_block);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }
        x += QK_K;
    }
}

inline void quantize_row_q4_K(const float * x, block_q4_K * y, int64_t k) {
    quantize_row_q4_K_impl(x, y, k, nullptr);
}

// Fallback block helper (kept for nsm_loader Q2/Q8 requantization of one 256-value block)
inline void quantize_block_q4_K(const float* in, block_q4_K* b, size_t n) {
    float pad[QK_K];
    std::memset(pad, 0, sizeof(pad));
    std::memcpy(pad, in, n * sizeof(float));
    quantize_row_q4_K_ref(pad, b, QK_K);
}
