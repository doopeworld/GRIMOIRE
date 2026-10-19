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

} // namespace b70
