// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com
//
// GGUF-standard quantizers for NSQuant output: Q8_0 and Q2_K.
// Ported from ggml-quants.c (llama.cpp / ik_llama) — produces bit-exact
// ggml blocks loadable by llama-cli. Q4_K lives in ns_q4k_quant.h.

#ifndef NS_GGUF_QUANT_H
#define NS_GGUF_QUANT_H

#include "ns_q4k_quant.h"  // make_qkx2_quants, nearest_int, f32_to_fp16, block_q8_0 via ns_dequant.h
#include "vendor/iq1_s_quant.h"  // block_iq1_s, quantize_row_iq1_s
#include "vendor/iq2_xxs_quant.h"  // block_iq2_xxs, quantize_row_iq2_xxs (cold tier)

// Q2_K block: 256 values, 2-bit quants + 4-bit packed scales/mins (84 bytes)
typedef struct {
    uint8_t  scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t  qs[QK_K/4];      // quants
    uint16_t d;               // super-block scale for quantized scales (fp16)
    uint16_t dmin;            // super-block scale for quantized mins (fp16)
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(uint16_t) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

// Q8_0: 32 values/block, fp16 scale + 32 int8 quants (34 bytes)
inline void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = (int)(k / QK8_0);

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;
            y[i].qs[j] = roundf(x0);
        }
    }
}

inline void quantize_row_q8_0(const float * x, block_q8_0 * y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}

// Q2_K: 256 values/block, 2-bit quants with 4-bit scales+mins (84 bytes)
inline void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = (int)(k / QK_K);

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float   weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];

    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }

        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            y[i].d = GGML_FP32_TO_FP16(max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y[i].dmin = GGML_FP32_TO_FP16(max_min/q4scale);
        } else {
            y[i].dmin = GGML_FP32_TO_FP16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = GGML_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16*j + ii] + dm)/d);
                l = MAX(0, MIN(3, l));
                L[16*j + ii] = l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

inline void quantize_row_q2_K(const float * x, block_q2_K * y, int64_t k) {
    quantize_row_q2_K_ref(x, y, k);
}

#endif // NS_GGUF_QUANT_H
