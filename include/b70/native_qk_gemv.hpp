// Native GGUF decode GEMV arithmetic, shared by device kernels and host tests.
// Q4_K and IQ4_XS use eight 32-value lanes; Q5_K and Q6_K use sixteen
// 16-value lanes. Q8_0 and IQ4_NL use two 16-value lanes per block.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "b70/weights.hpp"
#include <cstdlib>

namespace b70 {

inline bool native_qk_fast_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("GRIMOIRE_GGUF_FAST_GEMV");
        return !(e && *e == '0');
    }();
    return enabled;
}

template <Fmt F>
inline bool native_qk_fast_weight(const QuantWeight& w) {
    static_assert(is_native_gguf(F));
    return native_qk_fast_enabled() && w.fmt == F && w.payload &&
           !w.scales && !w.zeros && w.row_scales == 0 &&
           w.K > 0 && w.K % NativeGGUFBlock<F>::elems == 0 &&
           w.row_bytes >= int64_t(w.K / NativeGGUFBlock<F>::elems) * NativeGGUFBlock<F>::bytes;
}

template <Fmt F, int EPL>
inline float native_qk_scalar_step(const uint8_t* row, const float* x, int k0) {
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < EPL; ++i)
        acc = __builtin_fmaf(decode_elem<F>(row, k0 + i, 1.0f), x[k0 + i], acc);
    return acc;
}

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
    static_assert(is_native_gguf(F));
    if constexpr (F == Fmt::Q5_K) {
        // All 16 lanes work: lane = 2*g + h owns the 16 values [g*32 + h*16,
        // +16) of sub-block g (0..7). Every qh/qs/header byte is taken from a
        // 16-byte load (native_qk_byte) and never from a raw `p[i] >> var`.
        // ROOT CAUSE of the old B70 failure: the 8-lane form read the upper
        // half of qh with unrolled uchar loads, `((p[32 + i] >> lane) & 1)`;
        // the B70 compiler (icpx/IGC) miscompiles an unrolled run of uchar
        // loads shifted by a non-zero variable amount (lane 0, shift 0, was
        // right, lanes 1..7 wrong qh bits). Host emulation cannot see it.
        // Minimal reproducer: tools/test_q5k_bits.cpp. See NATIVE_QK.md.
        const int g = lane >> 1, h = lane & 1;
        const NativeQKBytes16 header = native_qk_load16(p); // d, dmin, all 12 scale/min bytes
        int sc, mn;
        if (g < 4) {
            sc = native_qk_byte(header, g + 4) & 63;
            mn = native_qk_byte(header, g + 8) & 63;
        }
        else {
            sc = (native_qk_byte(header, g + 8) & 15) |
                 ((native_qk_byte(header, g) >> 2) & 48);
            mn = (native_qk_byte(header, g + 8) >> 4) |
                 ((native_qk_byte(header, g + 4) >> 2) & 48);
        }
        const float d = f16_to_f32(uint16_t(native_qk_byte(header, 0)) |
                                     (uint16_t(native_qk_byte(header, 1)) << 8));
        const float dm = f16_to_f32(uint16_t(native_qk_byte(header, 2)) |
                                      (uint16_t(native_qk_byte(header, 3)) << 8));
        const float ds = d * float(sc), mins = dm * float(mn);
        const NativeQKBytes16 qs = native_qk_load16(p + 48 + (g >> 1) * 32 + h * 16);
        const NativeQKBytes16 qh = native_qk_load16(p + 16 + h * 16);
        const int nib = (g & 1) * 4;
        float dot[MB] = {}, sum[MB] = {};
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q = ((native_qk_byte(qs, i) >> nib) & 15) |
                          (((native_qk_byte(qh, i) >> g) & 1) << 4);
            #pragma unroll
            for (int m = 0; m < MB; ++m) {
                if (m >= mc) break;
                const float xv = x[m][block_base + g * 32 + h * 16 + i];
                dot[m] = __builtin_fmaf(float(q), xv, dot[m]);
                sum[m] += xv;
            }
        }
        #pragma unroll
        for (int m = 0; m < MB; ++m)
            if (m < mc) acc[m] += ds * dot[m] - mins * sum[m];
    } else if constexpr (F == Fmt::Q4_K) {
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
    } else if constexpr (F == Fmt::Q6_K) {
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
    } else if constexpr (F == Fmt::Q8_0 || F == Fmt::IQ4_NL) {
        if (lane >= 2) return;
        const float sc = f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
        const NativeQKBytes16 qs = native_qk_load16(p + 2 + (F == Fmt::Q8_0 ? lane * 16 : 0));
        float dot[MB] = {};
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int q;
            if constexpr (F == Fmt::Q8_0) q = int8_t(native_qk_byte(qs, i));
            else q = iq4nl_value((native_qk_byte(qs, i) >> (lane * 4)) & 15);
            #pragma unroll
            for (int m = 0; m < MB; ++m) {
                if (m >= mc) break;
                dot[m] = __builtin_fmaf(float(q), x[m][block_base + lane * 16 + i], dot[m]);
            }
        }
        #pragma unroll
        for (int m = 0; m < MB; ++m)
            if (m < mc) acc[m] += sc * dot[m];
    } else {
        if (lane >= 8) return;
        const float d = f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
        const int h = int(uint16_t(p[2]) | (uint16_t(p[3]) << 8));
        const int low = (p[4 + lane / 2] >> (4 * (lane % 2))) & 15;
        const float sc = d * float((low | (((h >> (2 * lane)) & 3) << 4)) - 32);
        const NativeQKBytes16 qs = native_qk_load16(p + 8 + lane * 16);
        float dot[MB] = {};
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int v0 = iq4nl_value(native_qk_byte(qs, i) & 15);
            const int v1 = iq4nl_value(native_qk_byte(qs, i) >> 4);
            #pragma unroll
            for (int m = 0; m < MB; ++m) {
                if (m >= mc) break;
                dot[m] = __builtin_fmaf(float(v0), x[m][block_base + lane * 32 + i], dot[m]);
                dot[m] = __builtin_fmaf(float(v1), x[m][block_base + lane * 32 + 16 + i], dot[m]);
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
    constexpr int BYTES = NativeGGUFBlock<F>::bytes;
    constexpr int ELEMS = NativeGGUFBlock<F>::elems;
    for (int base = 0; base < K; base += ELEMS)
        native_qk_block_lane<F, MB>(row + (base / ELEMS) * BYTES, x,
                                    base, lane, mc, acc);
}

} // namespace b70
