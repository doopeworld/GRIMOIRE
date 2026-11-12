// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// hadamard128.hpp -- the unnormalised 128-point Sylvester Hadamard transform of one 128-wide block,
// held by a 16-lane sub-group as 8 consecutive values per lane (block index = lane * 8 + t).
//
// H_128 = H_2 (x) ... (x) H_2: one butterfly per index bit, in any order.  Bits 0-2 of the index are
// t (inside a lane), bits 3-6 are the lane's bits 0-3 (a shuffle).  For the pair (i, i ^ h) with
// bit h clear in i:  out[i] = x[i] + x[i ^ h],  out[i ^ h] = x[i] - x[i ^ h].
// Entries are +-1, so H * H = 128 * I: a caller that rotates both operands of a dot product gets
// 128 times the product and folds the 1/128 into a scale.
//
// Used by the INT4-weight x INT8-activation prompt path (gemm_fast.cpp i4a8, onednn_gemm.cpp
// onednn_gemm_a8rot).  Rotating a block spreads an outlier channel over all 128 values, so one
// int8 scale per row stays accurate -- the trick exl3xpu's int8 prefill relies on.
#ifndef GRIMOIRE_HADAMARD128_HPP
#define GRIMOIRE_HADAMARD128_HPP

#include <sycl/sycl.hpp>

namespace b70 {

inline void hadamard128_sg(const sycl::sub_group& sg, int lane, float (&v)[8]) {
    #pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t & h) continue;
            const float a = v[t], b = v[t + h];
            v[t] = a + b;
            v[t + h] = a - b;
        }
    }
    #pragma unroll
    for (int m = 1; m < 16; m <<= 1) {
        const bool upper = (lane & m) != 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const float p = sycl::permute_group_by_xor(sg, v[t], m);
            v[t] = upper ? p - v[t] : v[t] + p;
        }
    }
}

} // namespace b70

#endif
