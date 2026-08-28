// =====================================================================
//  b70/moe.hpp  --  fused grouped-expert MoE
//
//  Target: Qwen3.5-MoE 35B-A3B
//    hidden 2048, 256 experts, 8 routed per token,
//    moe_intermediate 512, shared expert 512, 40 layers
//
//  WHY THIS EXISTS
//  ---------------
//  The generic MoE path dispatches one kernel per expert per matrix.
//  For this model that is 8 experts x 3 matrices = 24 launches per
//  layer, x 40 layers = 960 kernel launches per decoded token.
//
//  At ~5 us of Level Zero submit latency each, that is 4.8 ms of pure
//  launch overhead. The entire roofline budget for this model is about
//  2.9 ms per token (1.75 GB active weights / 608 GB/s). The GPU spends
//  most of the step idle, waiting to be told what to do next.
//
//  That is the 20%-of-bandwidth problem. It is not a coalescing problem:
//  each expert matrix here is 2048x512, which at 4 bits is 512 KB of
//  perfectly contiguous bytes. Big enough to saturate the bus on its
//  own. The bus is fine. The dispatch is not.
//
//  So: two launches per layer instead of 24.
//    launch 1  gate+up for all routed experts, fused SiLU-and-multiply
//    launch 2  down for all routed experts, fused weighted reduction
//  960 launches per token becomes 80.
//
//  LAYOUT
//  ------
//  Experts are stored expert-major and contiguous:
//
//    gate_up[e]  = [2 * I][H]   gate and up interleaved per expert
//    down[e]     = [H][I]
//
//  so expert e's entire working set is one contiguous run. A routed set
//  of 8 is 8 contiguous runs, not 24 scattered ones. No gather, no
//  index buffer, no copy: the kernel indexes by expert id directly.
// =====================================================================
#ifndef B70_MOE_HPP
#define B70_MOE_HPP

#include "weights.hpp"
#include <cmath>
#include <vector>
#ifdef SYCL_LANGUAGE_VERSION
#include <sycl/sycl.hpp>
#endif

namespace b70 {

struct MoeConfig {
    int hidden       = 2048;   // H
    int inter        = 512;    // I, moe_intermediate_size
    int num_experts  = 256;    // E
    int top_k        = 8;      // experts routed per token
    int shared_inter = 512;    // shared_expert_intermediate_size, 0 if none
    bool norm_topk   = true;   // renormalize router weights over the top-k
};

// One MoE layer's weights. All pointers are raw device memory.
//
// gate_up and down are single allocations spanning every expert, so
// expert e lives at a computed offset rather than in a pointer table.
// A pointer table would cost an indirection per expert in the inner
// loop and would stop the compiler proving the strides are uniform.
struct MoeLayer {
    MoeConfig cfg;

    QuantWeight gate_up;   // logical shape [E * 2I][H]
    QuantWeight down;      // logical shape [E * H][I]

    // Optional always-on shared expert (Qwen3.5 has one).
    bool        has_shared = false;
    QuantWeight shared_gate_up;   // [2 * shared_inter][H]
    QuantWeight shared_down;      // [H][shared_inter]

    const float* router = nullptr;   // [E][H] router projection, kept bf16/fp32

    // Row index of expert e's gate rows inside the packed gate_up tensor.
    // Gate occupies rows [0, I), up occupies [I, 2I) within each expert.
    inline int64_t gate_row(int e) const { return int64_t(e) * 2 * cfg.inter; }
    inline int64_t up_row(int e)   const { return int64_t(e) * 2 * cfg.inter + cfg.inter; }
    inline int64_t down_row(int e) const { return int64_t(e) * cfg.hidden; }

    // Bytes streamed for one token, given top_k routed experts. This is
    // the number that has to divide into 608 GB/s.
    double bytes_per_token() const {
        const double bpe = bits_per_elem(gate_up.fmt) / 8.0;
        double b = double(cfg.top_k) * 3.0 * double(cfg.hidden) * double(cfg.inter) * bpe;
        if (has_shared)
            b += 3.0 * double(cfg.hidden) * double(cfg.shared_inter) * bpe;
        return b;
    }
};

// Router output for one token: which experts, and with what weight.
struct Routing {
    int   expert[16];   // top_k <= 16
    float weight[16];
    int   k = 0;
};

// ---------------------------------------------------------------------
// Fused device kernels. All top_k experts in ONE launch each, with the
// expert id carried as data rather than as host-side control flow.
// Declared here (guarded) so non-SYCL translation units can include this
// header for the config types alone.
// ---------------------------------------------------------------------
#ifdef SYCL_LANGUAGE_VERSION
sycl::event launch_moe_gate_up(sycl::queue& q, const MoeLayer& L,
                               const int32_t* d_expert, const float* x, float* h,
                               const std::vector<sycl::event>& deps = {});

sycl::event launch_moe_down(sycl::queue& q, const MoeLayer& L,
                            const int32_t* d_expert, const float* d_weight,
                            const float* h, float* y,
                            const std::vector<sycl::event>& deps = {});
sycl::event launch_moe_gate_up_batched(
    sycl::queue& q, const MoeLayer& L, const int32_t* d_expert,
    const float* x, float* h, int tokens,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_moe_down_batched(
    sycl::queue& q, const MoeLayer& L, const int32_t* d_expert,
    const float* d_weight, const float* h, float* y, int tokens,
    const std::vector<sycl::event>& deps = {});
#endif

// Host reference: full MoE block for a single token.
//   y = sum_j  weight[j] * down_ej( silu(gate_ej . x) * (up_ej . x) )
// The SYCL kernels must agree with this bit for bit modulo fp order.
void moe_forward_ref(const MoeLayer& L, const Routing& r,
                     const float* x, float* y);

// Top-k selection over router logits, matching HF semantics:
// softmax over the full expert set, take top-k, optionally renormalize.
void route_ref(const float* logits, int num_experts, int top_k,
               bool norm_topk, Routing& out);

} // namespace b70
#endif
