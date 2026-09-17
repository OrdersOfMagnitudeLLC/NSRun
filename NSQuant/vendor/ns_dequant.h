#pragma once

#include <stdint.h>
#include <math.h>

// Constants from ggml-common.h
#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32

// FP16 type
typedef uint16_t ggml_half;

// Block structures from ggml-common.h
typedef struct {
    uint16_t d;    // super-block scale for quantized scales
    uint16_t dmin; // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4-bit quants
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;

typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

// block_q8_K -- copied exactly from llama.cpp ggml-common.h:372-375.
// Shared here (rather than duplicated per-TU) so AVX-512 and AVX2-only
// translation units (e.g. ns_llama_gemv.cpp) can pass pointers to it with
// matching type identity, without needing extern "C" type erasure.
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;

// FP16 to FP32 conversion from ggml-impl.h (using memcpy for type punning)
static inline float ggml_compute_fp16_to_fp32(uint16_t h) {
    union { uint32_t u; float f; } bits;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t sgn = (uint32_t)(h & 0x8000) << 16;
    if (exp == 0) {
        if (man == 0) { bits.u = sgn; return bits.f; }
        while (!(man & 0x400)) { man <<= 1; exp--; }
        man &= 0x3ff; exp++;
    } else if (exp == 31) {
        bits.u = sgn | 0x7f800000u | (man << 13);
        return bits.f;
    }
    bits.u = sgn | ((exp + 112) << 23) | (man << 13);
    return bits.f;
}

#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

// Helper from ggml-quants.c
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Dequantize functions from ggml-quants.c
static inline void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

static inline void dequantize_row_q6_K(const block_q6_K * x, float * y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * ql = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static inline void dequantize_row_q8_0(const block_q8_0 * x, float * y, int64_t k) {
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            y[i*QK8_0 + j] = x[i].qs[j]*d;
        }
    }
}
