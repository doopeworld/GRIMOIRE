// Packed group layout shared by the native ESIMD kernels and their host test.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "b70/formats.hpp"

namespace b70 {

struct NativeQ4Group { float scale, minimum; };
struct NativeQ4GroupCodes { int scale, minimum; };

inline NativeQ4GroupCodes native_q4_group_codes(const uint8_t* p, int g) {
    const uint8_t* s = p + 4;
    int sc, mn;
    if (g < 4) {
        sc = s[g] & 63;
        mn = s[g + 4] & 63;
    } else {
        sc = (s[g + 4] & 15) | ((s[g - 4] >> 2) & 48);
        mn = (s[g + 4] >> 4) | ((s[g] >> 2) & 48);
    }
    return {sc, mn};
}

inline NativeQ4Group native_q4_group(const uint8_t* p, int g) {
    const auto codes = native_q4_group_codes(p, g);
    const float d = f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
    const float dm = f16_to_f32(uint16_t(p[2]) | (uint16_t(p[3]) << 8));
    return {d * float(codes.scale), dm * float(codes.minimum)};
}

inline float native_q6_group_scale(const uint8_t* p, int g) {
    const float d = f16_to_f32(uint16_t(p[208]) | (uint16_t(p[209]) << 8));
    return d * float(int8_t(p[192 + g]));
}

inline int native_q4_group_code(const uint8_t* p, int g, int i) {
    return (p[16 + (g / 2) * 32 + i] >> ((g & 1) * 4)) & 15;
}

inline int native_q6_group_code(const uint8_t* p, int g, int i) {
    const int half = g / 8, quarter = (g % 8) / 2;
    const int offset = (g & 1) * 16;
    const int lo = p[half * 64 + (quarter % 2) * 32 + offset + i];
    const int hi = p[128 + half * 32 + offset + i];
    return (((lo >> ((quarter / 2) * 4)) & 15) |
            (((hi >> (quarter * 2)) & 3) << 4)) - 32;
}

// ---- Host mirrors of the exact byte/bit arithmetic in src/native_qk_esimd.hpp
// (the device code cannot run on the host; these functions are the same
// expressions, tested against gguf_dequantize).

// Q4_K: scale/min codes of group g exactly as the vector kernel derives them
// from header bytes 4..15 (s0 = 4..7, s1 = 8..11, s2 = 12..15).
inline NativeQ4GroupCodes native_q4_group_codes_vec(const uint8_t* hdr, int g) {
    const uint8_t* s0 = hdr + 4; const uint8_t* s1 = hdr + 8; const uint8_t* s2 = hdr + 12;
    if (g < 4) return {s0[g] & 63, s1[g] & 63};
    const int i = g - 4;
    return {(s2[i] & 15) | ((s0[i] >> 2) & 48), (s2[i] >> 4) | ((s1[i] >> 2) & 48)};
}

// Q6_K: the four 32-lane code vectors the kernel builds from ql (64 bytes of
// one half) and qh (32 bytes): v[q][l] is the unsigned 6-bit code (0..63) of
// block value half*128 + q*32 + l.
inline void native_q6_half_codes(const uint8_t* qA /*ql+0*/, const uint8_t* qB /*ql+32*/,
                                 const uint8_t* qh, uint8_t v[4][32]) {
    for (int l = 0; l < 32; ++l) {
        v[0][l] = uint8_t((qA[l] & 15) | ((qh[l] & 3) << 4));
        v[1][l] = uint8_t((qB[l] & 15) | ((qh[l] & 12) << 2));
        v[2][l] = uint8_t((qA[l] >> 4) | (qh[l] & 48));
        v[3][l] = uint8_t((qB[l] >> 4) | ((qh[l] & 192) >> 2));
    }
}

// Q6_K windowed read: block `p` is read from the 4-byte-aligned base p - S
// (S = 2 for the odd block of a pair), 212 bytes; field offsets are +S.
// Returns the window so the test can check alignment and bounds.
struct NativeQ6Window { const uint8_t* base; int len; int s; };
inline NativeQ6Window native_q6_window(const uint8_t* p, bool odd) {
    return {p - (odd ? 2 : 0), 212, odd ? 2 : 0};
}

} // namespace b70
