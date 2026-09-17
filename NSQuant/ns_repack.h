#pragma once

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdexcept>
#include <vector>

#include "vendor/ns_dequant.h"

// 64-byte aligned STL allocator for repacked GEMV buffers.
template <typename T, size_t A = 64>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, A>&) {}
    T* allocate(size_t n) {
        if (n == 0) return nullptr;
        size_t bytes = n * sizeof(T);
        void* p = nullptr;
        if (posix_memalign(&p, A, bytes) != 0) {
            throw std::bad_alloc();
        }
        return reinterpret_cast<T*>(p);
    }
    void deallocate(T* p, size_t) {
        free(p);
    }
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, A>;
    };
};

template <typename T, size_t A>
inline bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<T, A>&) { return true; }

template <typename T, size_t A>
inline bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<T, A>&) { return false; }

// Convenience alias for 64-byte aligned uint8_t buffers used by the loader and parser.
using AlignedVectorU8 = std::vector<uint8_t, AlignedAllocator<uint8_t, 64>>;

// Layouts for 8-row interleaved (panel) quantized weights.
// These intentionally match the llama.cpp block_q4_Kx8 / block_q6_Kx8 layouts
// in ggml/src/ggml-cpu/repack.h.

// Q4_K panel: 8 rows, 256 values per super-block.
// Size = 1152 bytes = 8 * 144 bytes, so the repacked copy is byte-for-byte
// the same size as the original Q4_K data.
struct ns_q4_Kx8 {
    uint16_t d[8];    // super-block scales for the 8 rows
    uint16_t dmin[8]; // super-block mins for the 8 rows
    uint8_t  scales[96];
    uint8_t  qs[1024];
};
static_assert(sizeof(ns_q4_Kx8) == 1152, "ns_q4_Kx8 size mismatch");

// Q6_K panel: 8 rows, 256 values per super-block.
// Size = 1680 bytes = 8 * 210 bytes.
struct ns_q6_Kx8 {
    uint16_t d[8];
    int8_t   scales[128];  // 16 scales per row * 8 rows, transposed
    uint8_t  ql[1024];
    uint8_t  qh[512];
};
static_assert(sizeof(ns_q6_Kx8) == 1680, "ns_q6_Kx8 size mismatch");

// Helpers to unpack the 6-bit packed Q4_K scales (same as get_scale_min_k4
// but with pointer to the packed 12-byte scale/min plane).
static inline void get_scale_min_k4x8(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// Pack the scales/min plane for one panel of up to 8 Q4_K blocks.
// This mirrors make_block_q4_Kx8() in llama.cpp's repack.cpp.
static inline void pack_q4_Kx8_scales(const block_q4_K* in, int n_rows, uint8_t* out) {
    // Zero the 96 bytes first to make tail padding safe.
    memset(out, 0, 96);

    uint8_t s[8], m[8];

    // First half of the 8 scales/mins (sub-blocks 0..3 for the 8 rows).
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            if (j < n_rows) {
                s[j] = in[j].scales[i] & 63;
                m[j] = in[j].scales[i + 4] & 63;
            } else {
                s[j] = 0;
                m[j] = 0;
            }
        }

        out[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    // Second half of the 8 scales/mins (sub-blocks 4..7 for the 8 rows).
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            if (j < n_rows) {
                s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
                m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
            } else {
                s[j] = 0;
                m[j] = 0;
            }
        }

        out[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }
}

// Repack up to 8 rows of block_q4_K into one row-panel of ns_q4_Kx8.
// R is the panel width (default 8).  n_rows <= R; missing rows are zero-padded.
// dst must have space for n_blocks entries.
static inline void repack_q4_K_row_panel(const block_q4_K* src, int n_rows,
                                         int n_blocks, ns_q4_Kx8* dst, int R = 8) {
    // Build a temporary zero-padded array of 8 source blocks per super-block.
    for (int b = 0; b < n_blocks; b++) {
        ns_q4_Kx8 out;
        block_q4_K in[8];
        memset(&in, 0, sizeof(in));

        for (int i = 0; i < n_rows; i++) {
            in[i] = src[(size_t)i * n_blocks + b];
        }

        for (int i = 0; i < 8; i++) {
            if (i < n_rows) {
                out.d[i]    = in[i].d;
                out.dmin[i] = in[i].dmin;
            } else {
                out.d[i]    = 0;
                out.dmin[i] = 0;
            }
        }

        pack_q4_Kx8_scales(in, n_rows, out.scales);

        // Repack qs into 16 natural chunks per row.  Each chunk contains
        // 16 consecutive q8 values packed into 8 q4 bytes as
        //   byte i = [value 2*i] | [value 2*i + 1] << 4
        // (lower = even q8 position, upper = odd q8 position of the same
        // sub-block).  The 8 rows are interleaved in 8-byte chunks, i.e.
        // chunk c offset is c*64, row i within chunk at i*8.
        auto get_q4_value = [](const block_q4_K& x, int p) -> uint8_t {
            int super_block = p / 64;
            int offset      = p % 64;
            int q4_byte     = super_block * 32 + (offset % 32);
            return (offset < 32) ? (x.qs[q4_byte] & 0x0F)
                                 : ((x.qs[q4_byte] >> 4) & 0x0F);
        };

        memset(out.qs, 0, sizeof(out.qs));
        const int nchunks = QK_K / 2 / 8;  // 16
        for (int c = 0; c < nchunks; c++) {
            int sb   = c / 2;
            int half = c % 2;
            int base = sb * 32 + half * 16;
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < 8; j++) {
                    int p0 = base + 2 * j;
                    int p1 = p0 + 1;
                    uint8_t v0 = get_q4_value(in[i], p0);
                    uint8_t v1 = get_q4_value(in[i], p1);
                    out.qs[c * 64 + i * 8 + j] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
                }
            }
        }

        dst[b] = out;
    }
}

// Repack up to 8 rows of block_q4_K into one row-panel of ns_q4_Kx8, using
// the EXACT byte-interleave scheme from llama.cpp's make_block_q4_Kx8()
// (blck_size_interleave = 8), as required by the ggml_gemv_q4_K_8x8_q8_K
// AVX2 kernel (ns_llama_gemv.cpp). This is a raw 8-byte block copy from each
// row's qs array: NOT the same byte layout as repack_q4_K_row_panel() above
// (which reconstructs nibble values and is only compatible with the
// generic/reference kernels, vec_dot_q4k_q8k_Rx1 / vec_dot_q4k_q8k_Rx1_ref).
static inline void repack_q4_K_row_panel_llama(const block_q4_K* src, int n_rows,
                                               int n_blocks, ns_q4_Kx8* dst, int R = 8) {
    (void)R;
    for (int b = 0; b < n_blocks; b++) {
        ns_q4_Kx8 out;
        block_q4_K in[8];
        memset(&in, 0, sizeof(in));

        for (int i = 0; i < n_rows; i++) {
            in[i] = src[(size_t)i * n_blocks + b];
        }

        for (int i = 0; i < 8; i++) {
            out.d[i]    = (i < n_rows) ? in[i].d    : 0;
            out.dmin[i] = (i < n_rows) ? in[i].dmin : 0;
        }

        pack_q4_Kx8_scales(in, n_rows, out.scales);

        // Raw byte interleave, blck_size_interleave = 8: for each group of
        // 8 output bytes, cycle through rows 0..7 taking 8 raw qs bytes at
        // the same src_offset per llama.cpp's make_block_q4_Kx8.
        memset(out.qs, 0, sizeof(out.qs));
        const int blck = 8;
        const int end = QK_K * 4 / blck; // 128
        for (int i = 0; i < end; i++) {
            int src_id     = i % 8;
            int src_offset = (i / 8) * blck;
            int dst_offset = i * blck;
            if (src_id < n_rows) {
                memcpy(out.qs + dst_offset, in[src_id].qs + src_offset, blck);
            }
        }

        dst[b] = out;
    }
}

// Repack up to 8 rows of block_q6_K into one row-panel of ns_q6_Kx8, using
// the same raw byte-interleave scheme (blck_size_interleave = 8) as
// llama.cpp's make_block_q6_Kx8() in repack.cpp. ns_q6_Kx8 is already
// byte-layout-compatible with llama.cpp's block_q6_Kx8 (d, scales, ql, qh),
// so this is a straight port of that packing logic.
static inline void repack_q6_K_row_panel(const block_q6_K* src, int n_rows,
                                         int n_blocks, ns_q6_Kx8* dst, int R = 8) {
    (void)R;
    for (int b = 0; b < n_blocks; b++) {
        ns_q6_Kx8 out;
        block_q6_K in[8];
        memset(&in, 0, sizeof(in));

        for (int i = 0; i < n_rows; i++) {
            in[i] = src[(size_t)i * n_blocks + b];
        }

        for (int i = 0; i < 8; i++) {
            out.d[i] = (i < n_rows) ? in[i].d : 0;
        }

        // scales[128]: pure transpose, one int8 per (group, row).
        memset(out.scales, 0, sizeof(out.scales));
        for (int j = 0; j < QK_K / 16; j++) {
            for (int i = 0; i < n_rows; i++) {
                out.scales[j * 8 + i] = in[i].scales[j];
            }
        }

        // ql[1024]: raw 8-byte interleave across the 8 rows.
        memset(out.ql, 0, sizeof(out.ql));
        {
            const int blck = 8;
            const int end = (int)(QK_K / 2 * 8) / blck; // 128
            for (int i = 0; i < end; i++) {
                int src_id     = i % 8;
                int src_offset = (i / 8) * blck;
                int dst_offset = i * blck;
                if (src_id < n_rows) {
                    memcpy(out.ql + dst_offset, in[src_id].ql + src_offset, blck);
                }
            }
        }

        // qh[512]: same interleave pattern, half as many chunks as ql.
        memset(out.qh, 0, sizeof(out.qh));
        {
            const int blck = 8;
            const int end = (int)(QK_K / 4 * 8) / blck; // 64
            for (int i = 0; i < end; i++) {
                int src_id     = i % 8;
                int src_offset = (i / 8) * blck;
                int dst_offset = i * blck;
                if (src_id < n_rows) {
                    memcpy(out.qh + dst_offset, in[src_id].qh + src_offset, blck);
                }
            }
        }

        dst[b] = out;
    }
}
