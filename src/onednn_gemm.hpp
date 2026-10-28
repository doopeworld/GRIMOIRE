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

} // namespace b70

#endif
