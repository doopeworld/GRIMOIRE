// Native GGUF block decode for a 16-K slice of one GEMM B row.
// The slice stays within one block scale or sub-block scale.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "b70/formats.hpp"

namespace b70 {

template <Fmt F>
inline void native_qk_block_scales(const uint8_t* p, float* scales, float* mins) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    static_assert(is_native_gguf(F));
    if constexpr (F == Fmt::Q4_K || F == Fmt::Q5_K) {
        const uint8_t* s = p + 4;
        const float d = f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
        const float dm = f16_to_f32(uint16_t(p[2]) | (uint16_t(p[3]) << 8));
        for (int g = 0; g < 8; ++g) {
            int sc, m;
            if (g < 4) {
                sc = s[g] & 63;
                m = s[g + 4] & 63;
            } else {
                sc = (s[g + 4] & 15) | ((s[g - 4] >> 2) & 48);
                m = (s[g + 4] >> 4) | ((s[g] >> 2) & 48);
            }
            scales[g] = d * float(sc);
            mins[g] = dm * float(m);
        }
    } else if constexpr (F == Fmt::Q6_K) {
        const float d = f16_to_f32(uint16_t(p[208]) | (uint16_t(p[209]) << 8));
        for (int g = 0; g < 16; ++g)
            scales[g] = d * float(int8_t(p[192 + g]));
        (void)mins;
    } else if constexpr (F == Fmt::IQ4_XS) {
        const float d = f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
        const int h = int(uint16_t(p[2]) | (uint16_t(p[3]) << 8));
        for (int g = 0; g < 8; ++g) {
            const int low = (p[4 + g / 2] >> (4 * (g % 2))) & 15;
            scales[g] = d * float((low | (((h >> (2 * g)) & 3) << 4)) - 32);
        }
        (void)mins;
    } else {
        scales[0] = f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
        (void)mins;
    }
}

template <Fmt F>
inline void native_qk_tile_16_scaled(const uint8_t* row, int k,
                                      const float* scales, const float* mins,
                                      float* values) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    static_assert(is_native_gguf(F));
    const int pos = k % NativeGGUFBlock<F>::elems;
    const uint8_t* p = row + (k / NativeGGUFBlock<F>::elems) * NativeGGUFBlock<F>::bytes;
    if constexpr (F == Fmt::Q4_K || F == Fmt::Q5_K) {
        const int group = pos / 32, lane = pos % 32;
        for (int i = 0; i < 16; ++i) {
            int q = (p[(F == Fmt::Q5_K ? 48 : 16) + (group / 2) * 32 + lane + i] >> ((group % 2) * 4)) & 15;
            if constexpr (F == Fmt::Q5_K) q |= ((p[16 + lane + i] >> group) & 1) << 4;
            values[i] = scales[group] * float(q) - mins[group];
        }
    } else if constexpr (F == Fmt::Q6_K) {
        const int half = pos / 128, lane = (pos % 128) / 32, i0 = pos % 32;
        for (int i = 0; i < 16; ++i) {
            const int low = (p[half * 64 + (lane % 2) * 32 + i0 + i] >> ((lane / 2) * 4)) & 15;
            const int high = (p[128 + half * 32 + i0 + i] >> (lane * 2)) & 3;
            values[i] = scales[pos / 16] * float((low | (high << 4)) - 32);
        }
    } else if constexpr (F == Fmt::Q8_0) {
        for (int i = 0; i < 16; ++i)
            values[i] = scales[0] * float(int8_t(p[2 + pos + i]));
    } else if constexpr (F == Fmt::IQ4_XS) {
        const int group = pos / 32, i0 = pos % 32;
        for (int i = 0; i < 16; ++i) {
            const int ix = i0 + i;
            const int q = (p[8 + group * 16 + ix % 16] >> (4 * (ix / 16))) & 15;
            values[i] = scales[group] * float(iq4nl_value(q));
        }
    } else {
        for (int i = 0; i < 16; ++i) {
            const int ix = pos + i;
            const int q = (p[2 + ix % 16] >> (4 * (ix / 16))) & 15;
            values[i] = scales[0] * float(iq4nl_value(q));
        }
    }
}

template <Fmt F>
inline void native_qk_tile_16(const uint8_t* row, int k, float* values) {
    float scales[16], mins[8];
    native_qk_block_scales<F>(row + (k / NativeGGUFBlock<F>::elems) * NativeGGUFBlock<F>::bytes,
                               scales, mins);
    native_qk_tile_16_scaled<F>(row, k, scales, mins, values);
}

// Host view of the exact packed VNNI pair order written into the GEMM B tile.
// dst[p * leading_pairs + n] corresponds to K positions 2*p and 2*p+1.
template <Fmt F>
inline void native_qk_tile_16_vnni(const uint8_t* row, int k, uint32_t* dst,
                                   int leading_pairs, int n) {
    float values[16];
    native_qk_tile_16<F>(row, k, values);
    for (int p = 0; p < 8; ++p)
        dst[p * leading_pairs + n] =
            uint32_t(f32_to_bf16(values[2 * p]).bits) |
            (uint32_t(f32_to_bf16(values[2 * p + 1]).bits) << 16);
}

} // namespace b70
