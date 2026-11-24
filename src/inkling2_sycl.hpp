// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "b70/inkling2_expert.hpp"
#include <sycl/sycl.hpp>
#include <limits>
#include <stdexcept>
#include <vector>

namespace b70::inkling2 {

using Bf16 = sycl::ext::oneapi::bfloat16;

struct ExpertScratchSizes { size_t weight_pairs, projection, activation; };
inline ExpertScratchSizes expert_scratch_sizes(int rows, int hidden, int inter) {
    // Existing Grimoire XMX GEMM consumes full 256-column work-group tiles.
    // Both 2304×2304 and 2304×1152 Mini weights meet this contract.
    if(rows<1 || rows>4096 || hidden<1 || hidden>8192 || inter<1 || inter>8192 || hidden%256 || inter%128)
        throw std::invalid_argument("unsupported Inkling2 XMX expert geometry");
    const size_t n_gu=size_t(2)*inter, n_dn=hidden;
    return {n_gu*hidden/2, size_t(rows)*(n_gu>n_dn?n_gu:n_dn), size_t(rows)*inter};
}

struct ExpertScratch {
    uint32_t* weight_pairs;  // BF16 VNNI; same buffer reused for both projections
    float* projection;      // FP32 GEMM results, reused after SwiGLU finishes
    Bf16* activation;       // post-SwiGLU BF16
};

// One selected expert, rows grouped by that expert. No routing, weighting, or
// accumulation across experts here. Caller owns non-overlapping queue-accessible
// USM (including scratch), must keep it alive until the returned event, and must
// chain that event before reusing scratch in an out-of-order queue. This does
// not allocate, wait, or alter stock dispatch.
sycl::event launch_expert_ffn(sycl::queue& q, ExpertMatrix gate_up, ExpertMatrix down,
    int rows, const Bf16* input, Bf16* out, float gate_cap, float up_cap,
    ExpertScratch scratch, const std::vector<sycl::event>& deps = {});

} // namespace b70::inkling2
