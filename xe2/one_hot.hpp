// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "layout.hpp"
#include "b70/formats.hpp"

namespace qfn_xe2 {
struct HotRow { int expert, k; };

// Known signed codes and positive scales. The oracle uses these values
// directly, never unpacking the bytes sent to either GPU kernel.
constexpr int hot_code(int e, int n, int k) {
    return ((e*3 + n*5 + k*7 + (k/32)*3) & 15) - 8;
}
constexpr uint16_t hot_scale(int e, int n, int g) {
    const int t = (e+n+g) % 4;
    if ((n % 16) < 8) return uint16_t(0x3e80 + 128*t); // power of two: exact
    return uint16_t(0x3e80 + 128*t + 1 + (e*5+n*7+g*11)%47);
}
inline float hot_product(HotRow row, int n) {
    return float(hot_code(row.expert,n,row.k)) * b70::bf16_to_f32({hot_scale(row.expert,n,row.k/group)});
}
} // namespace qfn_xe2
