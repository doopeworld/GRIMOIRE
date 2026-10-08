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

// Speculative-verify attention on the matrix engine (src/attn_verify_dpas.cpp, built
// into bin/libgrimoire_attn.so with 256 registers per ESIMD thread).
#pragma once
#include "kernels.hpp"

namespace b70 {

// One group = the verify rows of one conversation: rows row0 .. row0+nrows-1, one KV
// slot, consecutive positions (row r has length d_lens[r]; the last row is the longest).
constexpr int kMaxVerifyGroups = 16;
struct VerifyGroups {
    int32_t n = 0;
    int32_t row0[kMaxVerifyGroups];
    int32_t nrows[kMaxVerifyGroups];
    int32_t slot[kMaxVerifyGroups];
};

// head_dim 256, FP8 E4M3 cache, no window / sparse bits, seq_cap a multiple of 16.
bool verify_dpas_ok(const AttnParams& p);
// Split count (the launch's splits_max) for these groups, len_max = the longest row's
// length over all of them: ~GRIMOIRE_VERIFY_THREADS threads, >= 32 keys per split.
int verify_dpas_splits(const AttnParams& p, const VerifyGroups& g, int len_max);
// p: row 0's q / out (row stride num_heads * head_dim), k/v_cache = the slot-0 bases,
// partials / part_m / part_l with room for rows * num_heads * splits_max.
// splits_max >= verify_dpas_splits(len_max) of every group.
sycl::event launch_verify_attn_dpas(sycl::queue& q, const AttnParams& p, const VerifyGroups& g,
                                    int64_t kv_stride, const int32_t* d_lens, int splits_max,
                                    const std::vector<sycl::event>& deps = {});
// out[row][head] = the merge of that row's splits.  Empty splits (m = -inf) are skipped.
sycl::event launch_verify_merge(sycl::queue& q, const AttnParams& p, int rows, int splits_max,
                                const std::vector<sycl::event>& deps = {});

} // namespace b70
