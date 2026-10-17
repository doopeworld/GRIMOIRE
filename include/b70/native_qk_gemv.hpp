// Native GGUF decode GEMV arithmetic, shared by device kernels and host tests.
// Q4_K uses lanes 0..7 for its eight 32-value groups; Q6_K uses all
// 16 lanes for its sixteen 16-value groups.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "b70/formats.hpp"

namespace b70 {

struct NativeQKBytes16 { uint64_t lo, hi; };

inline NativeQKBytes16 native_qk_load16(const uint8_t* p) {
    NativeQKBytes16 v;
    // GGML's 210-byte Q6_K rows are not naturally aligned. A 16-byte
    // memcpy gives the compiler a wide, unaligned-safe load.
    __builtin_memcpy(&v, p, 16);
    return v;
}

inline uint8_t native_qk_byte(NativeQKBytes16 v, int i) {
    return uint8_t((i < 8 ? v.lo >> (i * 8) : v.hi >> ((i - 8) * 8)) & 255u);
}

// Add one lane's share of one block to up to four activation rows. The
// header is resolved once for all M rows; q4's dot and x sum are scaled
// only after the complete 32-value group has been accumulated.
template <Fmt F, int MB>
inline void native_qk_block_lane(const uint8_t* p, const float* const* x,
                                 int block_base, int lane, int mc, float* acc) {
    static_assert(F == Fmt::Q4_K || F == Fmt::Q6_K);
    if constexpr (F == Fmt::Q4_K) {
        if (lane >= 8) return;
        const NativeQKBytes16 header = native_qk_load16(p); // d, dmin, all 12 scale/min bytes
        int sc, mn;
        if (lane < 4) {
            sc = native_qk_byte(header, lane + 4) & 63;
            mn = native_qk_byte(header, lane + 8) & 63;
        }
        else {
            sc = (native_qk_byte(header, lane + 8) & 15) |
                 ((native_qk_byte(header, lane) >> 2) & 48);
            mn = (native_qk_byte(header, lane + 8) >> 4) |
                 ((native_qk_byte(header, lane + 4) >> 2) & 48);
        }
        const float d = f16_to_f32(uint16_t(native_qk_byte(header, 0)) |
                                     (uint16_t(native_qk_byte(header, 1)) << 8));
        const float dm = f16_to_f32(uint16_t(native_qk_byte(header, 2)) |
                                      (uint16_t(native_qk_byte(header, 3)) << 8));
        const float ds = d * float(sc), mins = dm * float(mn);
        const uint8_t* qs = p + 16 + (lane / 2) * 32;
        const NativeQKBytes16 a = native_qk_load16(qs);
        const NativeQKBytes16 b = native_qk_load16(qs + 16);
        float dot[MB] = {}, sum[MB] = {};
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q0 = (native_qk_byte(a, i) >> ((lane & 1) * 4)) & 15;
            const int q1 = (native_qk_byte(b, i) >> ((lane & 1) * 4)) & 15;
            #pragma unroll
            for (int m = 0; m < MB; ++m) {
                if (m >= mc) break;
                const float x0 = x[m][block_base + lane * 32 + i];
                const float x1 = x[m][block_base + lane * 32 + 16 + i];
                dot[m] = __builtin_fmaf(float(q0), x0, dot[m]);
                dot[m] = __builtin_fmaf(float(q1), x1, dot[m]);
                sum[m] += x0 + x1;
            }
        }
        #pragma unroll
        for (int m = 0; m < MB; ++m)
            if (m < mc) acc[m] += ds * dot[m] - mins * sum[m];
    } else {
        const int half = lane / 8, quarter = (lane % 8) / 2;
        const int offset = (lane & 1) * 16;
        const int low_shift = ((quarter / 2) * 4), high_shift = quarter * 2;
        const uint8_t* lo = p + half * 64 + (quarter % 2) * 32 + offset;
        const uint8_t* hi = p + 128 + half * 32 + offset;
        const NativeQKBytes16 l0 = native_qk_load16(lo);
        const NativeQKBytes16 h0 = native_qk_load16(hi);
        const NativeQKBytes16 scales = native_qk_load16(p + 192);
        const float d = f16_to_f32(uint16_t(p[208]) | (uint16_t(p[209]) << 8));
        const float sc = d * float(int8_t(native_qk_byte(scales, lane)));
        float dot[MB] = {};
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q = (((native_qk_byte(l0, i) >> low_shift) & 15) |
                           (((native_qk_byte(h0, i) >> high_shift) & 3) << 4)) - 32;
            #pragma unroll
            for (int m = 0; m < MB; ++m) {
                if (m >= mc) break;
                dot[m] = __builtin_fmaf(float(q), x[m][block_base + lane * 16 + i], dot[m]);
            }
        }
        #pragma unroll
        for (int m = 0; m < MB; ++m)
            if (m < mc) acc[m] += sc * dot[m];
    }
}

template <Fmt F, int MB>
inline void native_qk_row_lane(const uint8_t* row, const float* const* x,
                               int K, int lane, int mc, float* acc) {
    constexpr int BYTES = F == Fmt::Q4_K ? 144 : 210;
    for (int base = 0; base < K; base += 256)
        native_qk_block_lane<F, MB>(row + (base / 256) * BYTES, x,
                                    base, lane, mc, acc);
}

} // namespace b70
