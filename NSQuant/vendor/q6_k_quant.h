// Vendored Q6_K reference quantizer (6.5625 bpw), ported from ggml-quants.c
// (quantize_row_q6_K_ref + make_qx_quants, MIT, llama.cpp project).
// Wired as NSQuant's hot tier — laptop-friendly, smaller than Q8_0 while
// keeping more precision than flat Q4 on the hottest clusters.
#pragma once

#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>

#include "ns_dequant.h"       // QK_K, ggml_half, GGML_FP16_TO_FP32, block_q6_K
#include "../ns_q4k_quant.h"  // GGML_FP32_TO_FP16, nearest_int, MIN, MAX, GGML_RESTRICT

#ifndef NS_Q6K_EPS
#define NS_Q6K_EPS 1e-15f
#endif

static float make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, int rmse_type,
        const float * GGML_RESTRICT qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < NS_Q6K_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    float best_sumlx = sumlx, best_suml2 = suml2;
    for (int is = -9; is <= 9; ++is) {
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
            best_sumlx = sumlx; best_suml2 = suml2;
        }
        iscale = (nmax-1 + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
            best_sumlx = sumlx; best_suml2 = suml2;
        }
    }

    sumlx = best_sumlx; suml2 = best_suml2;
    for (int iter = 0; iter < n*(2*nmax-1); ++iter) {
        float abs_gmax = 0, gmax = 0;
        int best_j = -1;
        for (int j = 0; j < n; ++j) {
            float w = qw ? qw[j] : rmse_type == 1 ? x[j] * x[j] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[j]) : sqrtf(fabsf(x[j]));
            int l = L[j] - nmax;
            float g = scale * w * (x[j] - scale*l);
            if ((g > 0 && l < nmax-1) || (g < 0 && l > -nmax)) {
                float ag = fabsf(g);
                if (ag > abs_gmax) {
                    abs_gmax = ag; gmax = g; best_j = j;
                }
            }
        }
        if (best_j < 0) break;

        float new_sumlx = sumlx, new_suml2 = suml2;
        float w = qw ? qw[best_j] : rmse_type == 1 ? x[best_j] * x[best_j] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[best_j]) : sqrtf(fabsf(x[best_j]));
        int l = L[best_j] - nmax;
        if (gmax > 0) {
            new_sumlx += w*x[best_j];
            new_suml2 += w*(2*l + 1);
            l += 1;
        } else {
            new_sumlx -= w*x[best_j];
            new_suml2 -= w*(2*l - 1);
            l -= 1;
        }
        if (new_suml2 > 0 && new_sumlx*new_sumlx > best*new_suml2) {
            sumlx = new_sumlx; suml2 = new_suml2;
            scale = sumlx/suml2; best = scale*sumlx;
            L[best_j] = l + nmax;
            assert(L[best_j] >= 0 && L[best_j] <= 2*nmax-1);
        }
        else {
            break;
        }
    }
    return scale;
}

static void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        if (max_abs_scale < NS_Q6K_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * GGML_RESTRICT ql = y[i].ql;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

static inline void quantize_row_q6_K(const float * x, block_q6_K * y, int64_t k) {
    quantize_row_q6_K_ref(x, y, k);
}
