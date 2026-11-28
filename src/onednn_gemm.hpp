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
//  onednn_gemm.hpp  --  prompt-size GEMMs through Intel oneDNN (opt-in)
//
//  On by default (GRIMOIRE_ONEDNN=0 turns it off): gemm_fast's y = x W^T for
//  FP8 E4M3 (a float scale per row) and BF16 weights runs as a oneDNN matmul
//  primitive on the engine's own queue.  oneDNN decodes the stored weight inside the GEMM, so the
//  dequantize pass in front of gemm_bf16_vnni (~200-255 GB/s, as long as the
//  GEMM itself at 512 rows) goes away.  MEASURED 2026-10-25, one B70, weights
//  read cold, 512 x 34816 x 5120: FP8 4.09 -> 1.39 ms, BF16 4.23 -> 1.42 ms
//  (tools/bench/onednn/).  Built only when tools/build_grimoire_only_b70.sh
//  finds third_party/onednn (oneDNN built from source with this compiler,
//  tools/bench/onednn/build_onednn.sh); no PyTorch, no vLLM.
// =====================================================================
#ifndef GRIMOIRE_ONEDNN_GEMM_HPP
#define GRIMOIRE_ONEDNN_GEMM_HPP

#include "kernels.hpp"
#include <vector>

namespace b70 {

// oneDNN on (not GRIMOIRE_ONEDNN=0) and a weight oneDNN reads as stored.
bool onednn_gemm_ok(const QuantWeight& w, int M);

// y[M][N] (fp32) = x[M][K] (bf16) W^T, or y += x W^T when `accumulate`.
sycl::event onednn_gemm(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y,
                        int M, bool accumulate, const std::vector<sycl::event>& deps);

// INT4 prompts as int8 x int8 (gemm_fast.cpp i4a8): w = int8 [N][K] whose rows are already in the
// 128-point Hadamard domain (hadamard128.hpp), one f32 scale per row with the 1/128 folded in;
// x is rotated the same way and quantized with one int8 scale per row here.
// INT4 weights through oneDNN's weights decompression (2026-11-14): u4 nibbles [N][K/2] as given, the bf16
// scales and u8 zeros copied once per weight into oneDNN's [K/G][N] order (built on first use, ~3 bytes per 128
// weights; a weight with signed-s4 groups -- zero 0xff -- is declined).  For the 17..64-row verify GEMMs, where
// GRIMOIRE's own INT4 small-M kernel runs at 230-340 GB/s.  Weights are rounded to bf16 after the scale (as the
// INT4 prompt GEMM does), so not bit-identical to the small-M kernel.
bool onednn_int4_ok(sycl::queue& q, const QuantWeight& w, int M);
sycl::event onednn_gemm_int4(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y, int M,
                             bool accumulate, const std::vector<sycl::event>& deps);
// GRIMOIRE_I4A8G=<32|64|128> (opt-in): INT4 x int8 activations with one scale per group of that many
// values (0 = off).  onednn_gemm_int4_a8g takes the weights onednn_int4_ok accepts.  64 is the recommended
// setting: MEASURED 2026-11-23, one B70, Swift INT4, KL of the next-token distribution against the bf16 path,
// ~160 positions each through 256 / 512 / 1024-token prompts, book and code text: 128 = 0.0013-0.0035, 64 = 0.0009-0.0026,
// 32 = 0.0005-0.0017 (I4A8, the default int8 path at >= 1536 rows: 0.0013 / 0.0034); first token at 512 / 1024
// tokens 255 / 455 ms (128), 281 / 519 (64), 351 / 643 (32), against 501 / 742 on the bf16 path.
int onednn_i4a8g_group();
// Every piece primitive a call of M rows (in calls of at most `chunk` rows, 0 = one call) needs for this weight is
// a fast kernel (built and checked before the call; a refused piece turns the shape off).
bool onednn_int4_a8g_ok(sycl::queue& q, const QuantWeight& w, bool accumulate, int M, int chunk = 0);
sycl::event onednn_gemm_int4_a8g(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y, int M,
                                 bool accumulate, const std::vector<sycl::event>& deps = {});

sycl::event onednn_gemm_a8rot(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y,
                              int M, bool accumulate, const std::vector<sycl::event>& deps);

// The same for a gate|up weight [2 FI][K] (gate rows first): h[M][FI] (bf16) = silu(x Wg^T) * (x Wu^T), the
// SwiGLU as oneDNN post-ops (swish, then a multiply by the up rows), no f32 gate|up block in memory.
sycl::event onednn_gemm_a8rot_swiglu(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, sycl_bf16* h,
                                     int M, const std::vector<sycl::event>& deps);

} // namespace b70

#endif
