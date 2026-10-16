// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// =====================================================================
//  b70/weights.hpp  --  quantized tensor descriptor + host packer API
//
//  A QuantWeight is a flat description of device memory. There is no
//  ownership, no virtual dispatch, no shape inference: the kernels take
//  raw pointers and integer strides, exactly as the blueprint asks for.
//
//  Layout, all formats, weights of a linear layer y = W x  with
//  W of shape [N_out][K_in]:
//
//    payload   [N][bytes_per_row(F,K)]      packed elements, K-major
//    scales    format dependent, see below
//    zeros     INT4 only, [N][K/128] uint8
//
//    BF16          scales unused
//    FP8_*, INT8   scales [N] float          (per output channel)
//    INT4          scales [N][K/128] bf16    (per group, asymmetric)
//    MXFP8/MXFP4   scales [N][K/32]  uint8   (E8M0 shared exponent)
// =====================================================================
#ifndef B70_WEIGHTS_HPP
#define B70_WEIGHTS_HPP

#include "formats.hpp"
#include <vector>

namespace b70 {

struct QuantWeight {
    Fmt            fmt      = Fmt::BF16;
    int            N        = 0;      // output features
    int            K        = 0;      // input features
    const uint8_t* payload  = nullptr;
    const void*    scales   = nullptr;   // float* | bf16_t* | uint8_t*, per fmt
    const uint8_t* zeros    = nullptr;   // INT4 only
    int64_t        row_bytes = 0;
    int            row_scales = 0;

    // Resolve the scale that applies to element k of output row n.
    // Kept out of the hot loop by the kernels, which hoist it per block.
    inline float scale_for(int n, int k) const {
        switch (fmt) {
            case Fmt::BF16:
            case Fmt::Q4_K:
            case Fmt::Q6_K:
                return 1.0f;
            case Fmt::FP8_E4M3:
            case Fmt::FP8_E5M2:
            case Fmt::INT8:
                return static_cast<const float*>(scales)[n];
            case Fmt::INT4:
                return bf16_to_f32(static_cast<const bf16_t*>(scales)
                                   [int64_t(n) * row_scales + (k >> int4_gshift())]);
            case Fmt::MXFP8:
            case Fmt::MXFP4:
                return e8m0_to_f32(static_cast<const uint8_t*>(scales)
                                   [int64_t(n) * row_scales + k / kMXBlock]);
        }
        return 1.0f;
    }

    inline uint8_t zero_for(int n, int k) const {
        return zeros ? zeros[int64_t(n) * row_scales + (k >> int4_gshift())] : 0;
    }

    // INT4 group size as a shift: 128 (GRIMOIRE's own quantizer and most
    // checkpoints) or 64 (GPTQ / AutoRound exports with group_size 64, e.g.
    // Qwen3.8-27B-W4A16).  Derived from the shape, so no field has to be
    // threaded through every kernel.  Every INT4 scale/zero index used to
    // divide by kInt4Group = 128 unconditionally, which applied each group-64
    // scale to the wrong half-group: fluent-looking garbage.  The loader
    // refuses any other group size.
    inline int int4_gshift() const {
        const int64_t rs = row_scales, k = K;
        return rs * 32 == k ? 5 : rs * 64 == k ? 6 : rs * 256 == k ? 8 : 7;
    }

    // Reference dequantized value. Used by the tests and by the CPU
    // fallback path; the GPU kernels reimplement this fused into the
    // tile loader, but must agree with it bit for bit.
    inline float at(int n, int k) const {
        const uint8_t* row = payload + int64_t(n) * row_bytes;
        const float    s   = scale_for(n, k);
        switch (fmt) {
            case Fmt::BF16:     return decode_elem<Fmt::BF16>(row, k, s);
            case Fmt::FP8_E4M3: return decode_elem<Fmt::FP8_E4M3>(row, k, s);
            case Fmt::FP8_E5M2: return decode_elem<Fmt::FP8_E5M2>(row, k, s);
            case Fmt::INT8:     return decode_elem<Fmt::INT8>(row, k, s);
            case Fmt::MXFP8:    return decode_elem<Fmt::MXFP8>(row, k, s);
            case Fmt::MXFP4:    return decode_elem<Fmt::MXFP4>(row, k, s);
            case Fmt::Q4_K:     return decode_q4_k(row, k);
            case Fmt::Q6_K:     return decode_q6_k(row, k);
            case Fmt::INT4:     return decode_int4(row, k, s, zero_for(n, k));
        }
        return 0.0f;
    }

    inline double bytes() const {
        return double(N) * double(K) * bits_per_elem(fmt) / 8.0;
    }
};

// ---------------------------------------------------------------------
// Host-side packer. Owns its buffers; hand the pointers to USM after
// upload. Quantization happens once, offline, never at runtime.
// ---------------------------------------------------------------------
struct PackedWeight {
    Fmt                  fmt;
    int                  N = 0, K = 0;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> scales_raw;   // reinterpreted per format
    std::vector<uint8_t> zeros;
    int64_t              row_bytes = 0;
    int                  row_scales = 0;

    QuantWeight view() const {
        QuantWeight w;
        w.fmt = fmt; w.N = N; w.K = K;
        w.payload    = payload.data();
        w.scales     = scales_raw.empty() ? nullptr : scales_raw.data();
        w.zeros      = zeros.empty() ? nullptr : zeros.data();
        w.row_bytes  = row_bytes;
        w.row_scales = row_scales;
        return w;
    }
};

// Quantize a row-major [N][K] fp32 matrix into the requested format.
PackedWeight quantize(const float* src, int N, int K, Fmt fmt);

} // namespace b70
#endif // B70_WEIGHTS_HPP
