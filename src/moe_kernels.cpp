// =====================================================================
//  moe_kernels.cpp  --  fused grouped-expert MoE decode
//
//  Two kernels per layer, total, regardless of top_k:
//
//    moe_gate_up   all routed experts' gate and up projections, with
//                  SiLU-and-multiply fused into the epilogue. Writes
//                  h[k][I]. One work-group per (expert_slot, row block).
//
//    moe_down      all routed experts' down projection, with the router
//                  weight and the cross-expert reduction fused in.
//                  Writes y[H] directly.
//
//  The routing table lives in device memory as k int32s. The kernel
//  reads it and computes its own weight offset; nothing is gathered,
//  copied, or reordered on the host between launches.
//
//  Why the expert id must come from memory rather than a kernel arg:
//  arguments are baked at submit time, so per-token routing would force
//  a resubmit per token. Reading the table inside the kernel is what
//  lets the whole layer stay at two launches while routing changes
//  every step.
// =====================================================================
#include "kernels.hpp"
#include "gemv_step.hpp"
#include "b70/moe.hpp"

namespace b70 {
namespace {

inline float silu(float v) { return v / (1.0f + sycl::exp(-v)); }

// ---------------------------------------------------------------------
// Kernel 1: gate + up, fused SiLU-and-multiply.
//
// Grid: [k * I / ROWS_PER_WG] work-groups, WG_SUBGROUPS sub-groups each.
// Sub-group s of group g computes one output row i of one expert slot.
//
// gate and up for row i live at
//   gate_up[(e*2I + i)     ][:]
//   gate_up[(e*2I + I + i) ][:]
// which are I rows apart -- two streams, both contiguous, both hitting
// the same x. x is 2048 floats (8 KB) and stays in SLM for the whole
// work-group.
// ---------------------------------------------------------------------
template <Fmt F>
// SLOTS_PER_SG: how many (gate,up) row pairs one sub-group owns.
//
// At 1 slot the work-group consumed 8 sub-groups x 2 rows x 1 KB = 16 KB of
// expert weights while staging the whole H=2048 activation (8 KB) from global
// memory -- 50% overhead -- and each sub-group had only 2 KB of weight loads
// in flight.  Measured 224 GB/s against a 602 GB/s card.  More slots amortize
// the staged activation and raise memory-level parallelism at the same time,
// which is exactly what took the dense GEMV from 61% to 86% of roofline.
static int moe_slots_per_sg() {
    static const int v = []{ const char* e = std::getenv("B70_MOE_SLOTS");
        int x = (e && *e) ? std::atoi(e) : 4;
        return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 4; }();
    return v;
}

template <Fmt F, int R>
sycl::event moe_gate_up_impl_r(sycl::queue& q, const MoeLayer& L,
                             const int32_t* d_expert,   // [k]
                             const float* x,            // [H]
                             float* h,                  // [M][k][I] out
                             int M,
                             const std::vector<sycl::event>& deps) {
    const int H = L.cfg.hidden, I = L.cfg.inter, K = L.cfg.top_k;
    const int rows_per_wg = WG_SUBGROUPS * R;
    const int groups_per_token = (K * I + rows_per_wg - 1) / rows_per_wg;
    const int n_groups = M * groups_per_token;

    return q.submit([&](sycl::handler& hc) {
        hc.depends_on(deps);
        const QuantWeight w = L.gate_up;
        const int64_t stride2I = int64_t(2) * I;

        sycl::local_accessor<float, 1> slmx(size_t(H), hc);
        // Decode tables, shared by the whole work-group. Identical to the
        // dense GEMV path -- FP8 bit-assembly and E8M0/E2M1 branches cost
        // more than the loads they decorate.
        sycl::local_accessor<float, 1> lut_slm(256, hc);
        sycl::local_accessor<float, 1> e8m0_slm(256, hc);
        sycl::local_accessor<float, 1> e2m1_slm(16, hc);


        hc.parallel_for(
            sycl::nd_range<1>(size_t(n_groups) * WG_SUBGROUPS * SG_SIZE,
                              size_t(WG_SUBGROUPS) * SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                float* lut  = lut_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* slut = e8m0_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* nlut = e2m1_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
                {
                    const int lid_ = int(it.get_local_id(0));
                    const int lsz_ = int(it.get_local_range(0));
                    if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::MXFP8)
                        for (int b_ = lid_; b_ < 256; b_ += lsz_) lut[b_] = e4m3_to_f32(uint8_t(b_));
                    else if constexpr (F == Fmt::FP8_E5M2)
                        for (int b_ = lid_; b_ < 256; b_ += lsz_) lut[b_] = e5m2_to_f32(uint8_t(b_));
                    if constexpr (Traits<F>::block == kMXBlock)
                        for (int b_ = lid_; b_ < 256; b_ += lsz_) slut[b_] = e8m0_to_f32(uint8_t(b_));
                    if constexpr (F == Fmt::MXFP4)
                        for (int b_ = lid_; b_ < 16; b_ += lsz_) nlut[b_] = e2m1_to_f32(uint8_t(b_));
                    sycl::group_barrier(it.get_group());
                }
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  lid  = int(it.get_local_id(0));

                // Stage the activation once per work-group. Every row in
                // this group reads all H of it, so SLM turns H global
                // reads per row into H per group.
                const int token = int(it.get_group(0)) / groups_per_token;
                const int local_group = int(it.get_group(0)) % groups_per_token;
                const float* xt = x + int64_t(token) * H;
                for (int c = lid; c < H; c += WG_SUBGROUPS * SG_SIZE) slmx[c] = xt[c];
                sycl::group_barrier(it.get_group());

                const int slot_row_base = local_group * rows_per_wg
                                        + int(sg.get_group_id()[0]) * R;
                float* xs = slmx.template
                    get_multi_ptr<sycl::access::decorated::no>().get();

                // All R slots share the staged activation, and their weight
                // loads are independent, so the memory system sees R x 2
                // outstanding streams instead of 2.
                float ga[R], ua[R];
                int64_t grow[R], urow[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    ga[r] = 0.0f; ua[r] = 0.0f;
                    const int sr = slot_row_base + r;
                    if (sr >= K * I) { grow[r] = -1; urow[r] = -1; continue; }
                    const int slot = sr / I;
                    const int i    = sr % I;
                    const int e = d_expert[int64_t(token) * K + slot];
                    grow[r] = int64_t(e) * stride2I + i;
                    urow[r] = grow[r] + I;
                }
                for (int base = 0; base + GEMV_STEP <= H; base += GEMV_STEP) {
                    const int k0 = base + lane * GEMV_EPL;
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (grow[r] < 0) continue;
                        const uint8_t* gp = w.payload + grow[r] * w.row_bytes;
                        const uint8_t* up = w.payload + urow[r] * w.row_bytes;
                        ga[r] += GemvStep<F, GEMV_EPL>::run(
                            w, gp, xs, lut, slut, nlut, int(grow[r]), k0);
                        ua[r] += GemvStep<F, GEMV_EPL>::run(
                            w, up, xs, lut, slut, nlut, int(urow[r]), k0);
                    }
                }
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (grow[r] < 0) continue;
                    for (int c = (H / GEMV_STEP) * GEMV_STEP + lane; c < H; c += SG_SIZE) {
                        ga[r] = sycl::fma(w.at(int(grow[r]), c), xs[c], ga[r]);
                        ua[r] = sycl::fma(w.at(int(urow[r]), c), xs[c], ua[r]);
                    }
                }
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int sr = slot_row_base + r;
                    const float g = sycl::reduce_over_group(sg, ga[r], sycl::plus<float>());
                    const float u = sycl::reduce_over_group(sg, ua[r], sycl::plus<float>());
                    if (lane == 0 && sr < K * I)
                        h[(int64_t(token) * K * I) + sr] = silu(g) * u;
                }
            });
    });
}

// ---------------------------------------------------------------------
// Kernel 2: down projection + router-weighted reduction across experts.
//
// Each sub-group owns one output element o of y. It walks all k experts,
// reading down[e][o][:] -- I contiguous values -- and accumulates
// weight[slot] * (down_row . h[slot]).
//
// Reducing across experts inside the kernel is what removes the separate
// scatter-add pass that a per-expert dispatch needs. No atomics either:
// the cross-expert sum is sequential within one sub-group.
// ---------------------------------------------------------------------
// R output rows per sub-group.  At R=1 a work-group staged slmh (K*I floats =
// 16 KB) from global memory to consume 16 KB of expert weights -- 100%
// overhead, and measured 156 GB/s against 602.  Widening amortizes the stage.
template <Fmt F, int R>
sycl::event moe_down_impl_r(sycl::queue& q, const MoeLayer& L,
                          const int32_t* d_expert,   // [k]
                          const float* d_weight,     // [k]
                          const float* h,            // [k][I]
                          float* y,                  // [M][H] out
                          int M,
                          const std::vector<sycl::event>& deps) {
    const int H = L.cfg.hidden, I = L.cfg.inter, K = L.cfg.top_k;
    const int rows_per_wg = WG_SUBGROUPS * R;
    const int groups_per_token = (H + rows_per_wg - 1) / rows_per_wg;
    const int n_groups = M * groups_per_token;

    return q.submit([&](sycl::handler& hc) {
        hc.depends_on(deps);
        const QuantWeight w = L.down;

        // h is k*I floats: 8*512 = 4096, 16 KB. Fits SLM comfortably and
        // every output row reads all of it.
        sycl::local_accessor<float, 1> slmh(size_t(K) * size_t(I), hc);
        // Decode tables, shared by the whole work-group. Identical to the
        // dense GEMV path -- FP8 bit-assembly and E8M0/E2M1 branches cost
        // more than the loads they decorate.
        sycl::local_accessor<float, 1> lut_slm(256, hc);
        sycl::local_accessor<float, 1> e8m0_slm(256, hc);
        sycl::local_accessor<float, 1> e2m1_slm(16, hc);


        hc.parallel_for(
            sycl::nd_range<1>(size_t(n_groups) * rows_per_wg * SG_SIZE,
                              size_t(rows_per_wg) * SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                float* lut  = lut_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* slut = e8m0_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* nlut = e2m1_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
                {
                    const int lid_ = int(it.get_local_id(0));
                    const int lsz_ = int(it.get_local_range(0));
                    if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::MXFP8)
                        for (int b_ = lid_; b_ < 256; b_ += lsz_) lut[b_] = e4m3_to_f32(uint8_t(b_));
                    else if constexpr (F == Fmt::FP8_E5M2)
                        for (int b_ = lid_; b_ < 256; b_ += lsz_) lut[b_] = e5m2_to_f32(uint8_t(b_));
                    if constexpr (Traits<F>::block == kMXBlock)
                        for (int b_ = lid_; b_ < 256; b_ += lsz_) slut[b_] = e8m0_to_f32(uint8_t(b_));
                    if constexpr (F == Fmt::MXFP4)
                        for (int b_ = lid_; b_ < 16; b_ += lsz_) nlut[b_] = e2m1_to_f32(uint8_t(b_));
                    sycl::group_barrier(it.get_group());
                }
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  lid  = int(it.get_local_id(0));

                const int token = int(it.get_group(0)) / groups_per_token;
                const int local_group = int(it.get_group(0)) % groups_per_token;
                const float* ht = h + int64_t(token) * K * I;
                for (int c = lid; c < K * I; c += WG_SUBGROUPS * SG_SIZE) slmh[c] = ht[c];
                sycl::group_barrier(it.get_group());

                const int o_base = local_group * rows_per_wg
                                 + int(sg.get_group_id()[0]) * R;

                float total[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) total[r] = 0.0f;

                for (int slot = 0; slot < K; ++slot) {
                    const int64_t route = int64_t(token) * K + slot;
                    const int   e  = d_expert[route];
                    const float rw = d_weight[route];

                    float acc[R];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
                    for (int base = 0; base + GEMV_STEP <= I; base += GEMV_STEP) {
                        const int k0 = base + lane * GEMV_EPL;
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            const int o = o_base + r;
                            if (o >= H) continue;
                            const int64_t d_row = int64_t(e) * H + o;
                            const uint8_t* dp = w.payload + d_row * w.row_bytes;
                            acc[r] += GemvStep<F, GEMV_EPL>::run(w, dp, &slmh[slot * I] - 0,
                                                    lut, slut, nlut, int(d_row), k0);
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const int o = o_base + r;
                        if (o >= H) continue;
                        const int64_t d_row = int64_t(e) * H + o;
                        for (int c = (I / GEMV_STEP) * GEMV_STEP + lane; c < I; c += SG_SIZE)
                            acc[r] = sycl::fma(w.at(int(d_row), c), slmh[slot * I + c], acc[r]);
                        total[r] += rw * sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                    }
                }
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int o = o_base + r;
                    if (lane == 0 && o < H) y[int64_t(token) * H + o] = total[r];
                }
            });
    });
}

template <Fmt F>
sycl::event moe_down_impl(sycl::queue& q, const MoeLayer& L,
                          const int32_t* d_expert, const float* d_weight,
                          const float* h, float* y, int M,
                          const std::vector<sycl::event>& deps) {
    // R=1 is best here: the slot loop over K experts already gives 8
    // independent streams, so widening only spills registers.  Measured
    // R=4 -> 113.4 TG against R=1 125.6.
    static const int slots = []{ const char* e = std::getenv("B70_MOE_DN_SLOTS");
        int x = (e && *e) ? std::atoi(e) : 1;
        return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 4; }();
    switch (slots) {
        case 1: return moe_down_impl_r<F, 1>(q, L, d_expert, d_weight, h, y, M, deps);
        case 2: return moe_down_impl_r<F, 2>(q, L, d_expert, d_weight, h, y, M, deps);
        case 8: return moe_down_impl_r<F, 8>(q, L, d_expert, d_weight, h, y, M, deps);
        default: return moe_down_impl_r<F, 4>(q, L, d_expert, d_weight, h, y, M, deps);
    }
}

template <Fmt F>
sycl::event moe_gate_up_impl(sycl::queue& q, const MoeLayer& L,
                             const int32_t* d_expert, const float* x, float* h,
                             int M, const std::vector<sycl::event>& deps) {
    static const int slots = []{ const char* e = std::getenv("B70_MOE_SLOTS");
        int x = (e && *e) ? std::atoi(e) : 4;
        return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 4; }();
    switch (slots) {
        case 1: return moe_gate_up_impl_r<F, 1>(q, L, d_expert, x, h, M, deps);
        case 2: return moe_gate_up_impl_r<F, 2>(q, L, d_expert, x, h, M, deps);
        case 8: return moe_gate_up_impl_r<F, 8>(q, L, d_expert, x, h, M, deps);
        default: return moe_gate_up_impl_r<F, 4>(q, L, d_expert, x, h, M, deps);
    }
}

} // namespace

// ---------------------------------------------------------------------
sycl::event launch_moe_gate_up(sycl::queue& q, const MoeLayer& L,
                               const int32_t* d_expert, const float* x, float* h,
                               const std::vector<sycl::event>& deps) {
    switch (L.gate_up.fmt) {
        case Fmt::MXFP4:    return moe_gate_up_impl<Fmt::MXFP4>(q, L, d_expert, x, h, 1, deps);
        case Fmt::INT4:     return moe_gate_up_impl<Fmt::INT4>(q, L, d_expert, x, h, 1, deps);
        case Fmt::MXFP8:    return moe_gate_up_impl<Fmt::MXFP8>(q, L, d_expert, x, h, 1, deps);
        case Fmt::INT8:     return moe_gate_up_impl<Fmt::INT8>(q, L, d_expert, x, h, 1, deps);
        case Fmt::FP8_E4M3: return moe_gate_up_impl<Fmt::FP8_E4M3>(q, L, d_expert, x, h, 1, deps);
        case Fmt::FP8_E5M2: return moe_gate_up_impl<Fmt::FP8_E5M2>(q, L, d_expert, x, h, 1, deps);
        case Fmt::BF16:     return moe_gate_up_impl<Fmt::BF16>(q, L, d_expert, x, h, 1, deps);
    }
    return {};
}

sycl::event launch_moe_down(sycl::queue& q, const MoeLayer& L,
                            const int32_t* d_expert, const float* d_weight,
                            const float* h, float* y,
                            const std::vector<sycl::event>& deps) {
    switch (L.down.fmt) {
        case Fmt::MXFP4:    return moe_down_impl<Fmt::MXFP4>(q, L, d_expert, d_weight, h, y, 1, deps);
        case Fmt::INT4:     return moe_down_impl<Fmt::INT4>(q, L, d_expert, d_weight, h, y, 1, deps);
        case Fmt::MXFP8:    return moe_down_impl<Fmt::MXFP8>(q, L, d_expert, d_weight, h, y, 1, deps);
        case Fmt::INT8:     return moe_down_impl<Fmt::INT8>(q, L, d_expert, d_weight, h, y, 1, deps);
        case Fmt::FP8_E4M3: return moe_down_impl<Fmt::FP8_E4M3>(q, L, d_expert, d_weight, h, y, 1, deps);
        case Fmt::FP8_E5M2: return moe_down_impl<Fmt::FP8_E5M2>(q, L, d_expert, d_weight, h, y, 1, deps);
        case Fmt::BF16:     return moe_down_impl<Fmt::BF16>(q, L, d_expert, d_weight, h, y, 1, deps);
    }
    return {};
}

sycl::event launch_moe_gate_up_batched(
    sycl::queue& q, const MoeLayer& L, const int32_t* d_expert,
    const float* x, float* h, int tokens,
    const std::vector<sycl::event>& deps) {
    switch (L.gate_up.fmt) {
        case Fmt::MXFP4:    return moe_gate_up_impl<Fmt::MXFP4>(q, L, d_expert, x, h, tokens, deps);
        case Fmt::INT4:     return moe_gate_up_impl<Fmt::INT4>(q, L, d_expert, x, h, tokens, deps);
        case Fmt::MXFP8:    return moe_gate_up_impl<Fmt::MXFP8>(q, L, d_expert, x, h, tokens, deps);
        case Fmt::INT8:     return moe_gate_up_impl<Fmt::INT8>(q, L, d_expert, x, h, tokens, deps);
        case Fmt::FP8_E4M3: return moe_gate_up_impl<Fmt::FP8_E4M3>(q, L, d_expert, x, h, tokens, deps);
        case Fmt::FP8_E5M2: return moe_gate_up_impl<Fmt::FP8_E5M2>(q, L, d_expert, x, h, tokens, deps);
        case Fmt::BF16:     return moe_gate_up_impl<Fmt::BF16>(q, L, d_expert, x, h, tokens, deps);
    }
    return {};
}

sycl::event launch_moe_down_batched(
    sycl::queue& q, const MoeLayer& L, const int32_t* d_expert,
    const float* d_weight, const float* h, float* y, int tokens,
    const std::vector<sycl::event>& deps) {
    switch (L.down.fmt) {
        case Fmt::MXFP4:    return moe_down_impl<Fmt::MXFP4>(q, L, d_expert, d_weight, h, y, tokens, deps);
        case Fmt::INT4:     return moe_down_impl<Fmt::INT4>(q, L, d_expert, d_weight, h, y, tokens, deps);
        case Fmt::MXFP8:    return moe_down_impl<Fmt::MXFP8>(q, L, d_expert, d_weight, h, y, tokens, deps);
        case Fmt::INT8:     return moe_down_impl<Fmt::INT8>(q, L, d_expert, d_weight, h, y, tokens, deps);
        case Fmt::FP8_E4M3: return moe_down_impl<Fmt::FP8_E4M3>(q, L, d_expert, d_weight, h, y, tokens, deps);
        case Fmt::FP8_E5M2: return moe_down_impl<Fmt::FP8_E5M2>(q, L, d_expert, d_weight, h, y, tokens, deps);
        case Fmt::BF16:     return moe_down_impl<Fmt::BF16>(q, L, d_expert, d_weight, h, y, tokens, deps);
    }
    return {};
}

} // namespace b70
