// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace qfn_xe2 {
// The lane stores two signed INT4 nibbles per byte. The fold is lossless
// only when the actual checkpoint zero point (not just its config) is 8.
constexpr uint32_t fold_int4(uint32_t word) { return word ^ 0x88888888u; }

inline bool within_bf16_roundoff(float control, float candidate) {
    if (!std::isfinite(control) || !std::isfinite(candidate)) return false;
    const float magnitude = std::abs(candidate);
    const int exponent = magnitude == 0.f ? -126 : std::max(-126, std::ilogb(magnitude));
    const float half_ulp = std::ldexp(1.f, exponent - 8);
    // Allow a small FP32 accumulation difference before one BF16 rounding.
    return std::abs(control - candidate) <= half_ulp + 2.e-5f * std::max(1.f, magnitude);
}
} // namespace qfn_xe2
