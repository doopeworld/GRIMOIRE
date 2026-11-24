// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "b70/formats.hpp"

#include <cstddef>
#include <cstdint>

namespace b70::inkling2 {

enum class PackedFormat { NVFP4, MXFP8 };

// One expert of one projection. Checkpoint views use read-only mapped bytes;
// SYCL callers use the same descriptor with their staged USM addresses.
// Gate/up has 2*I rows in checkpoint order (gate0, up0, gate1, up1, ...).
struct ExpertMatrix {
    PackedFormat format;
    int n, k;
    const uint8_t* payload;
    const uint8_t* scales;
    float outer = 1.f;  // NVFP4 amax[expert] / denominator. MXFP8 must use 1.
};

constexpr size_t payload_row_bytes(PackedFormat f, int k) {
    return f == PackedFormat::NVFP4 ? k / 2 : k;
}
constexpr size_t scale_row_bytes(PackedFormat f, int k) {
    return f == PackedFormat::NVFP4 ? k / 16 : k / 32;
}
// A VNNI dword holds consecutive K values for the same output column.
constexpr size_t vnni_pair_index(int n, int col, int kpair) {
    return size_t(kpair) * n + col;
}

// Caller validates the view and bounds before dispatch. Apply the NVFP4
// outer multiplier BEFORE rounding each weight, not to the dot product.
// Do not shuffle the alternating W13 rows here; the activation reads pairs.
template<PackedFormat F>
inline uint32_t expert_bf16_pair(const ExpertMatrix& w, int col, int kpair) {
    const auto* p = w.payload + col * payload_row_bytes(F, w.k);
    const auto* s = w.scales + col * scale_row_bytes(F, w.k);
    float a, b;
    if constexpr (F == PackedFormat::NVFP4) {
        const uint8_t v = p[kpair];
        const float scale = e4m3_to_f32(s[kpair / 8]);
        a = (e2m1_to_f32(v & 15) * scale) * w.outer;
        b = (e2m1_to_f32(v >> 4) * scale) * w.outer;
    } else {
        const float scale = e8m0_to_f32(s[kpair / 16]);
        a = e4m3_to_f32(p[2 * kpair]) * scale;
        b = e4m3_to_f32(p[2 * kpair + 1]) * scale;
    }
    return uint32_t(f32_to_bf16(a).bits) | (uint32_t(f32_to_bf16(b).bits) << 16);
}

} // namespace b70::inkling2
