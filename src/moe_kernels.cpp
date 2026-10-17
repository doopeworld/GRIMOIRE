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
#include <sycl/ext/intel/esimd.hpp>

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

template <Fmt F, int R, bool SH = false>
sycl::event moe_gate_up_impl_r(sycl::queue& q, const MoeLayer& L,
                             const int32_t* d_expert,   // [k]
                             const float* x,            // [H]
                             float* h,                  // [M][k][I] out
                             int M,
                             const std::vector<sycl::event>& deps,
                             const QuantWeight* wsh = nullptr,   // shared expert [2I][H]
                             const uint16_t* gate_w = nullptr,   // shared_expert_gate bf16 [H]
                             float* gate_out = nullptr) {        // sigmoid(gate) per token
    const int H = L.cfg.hidden, I = L.cfg.inter, K = L.cfg.top_k;
    // With a shared expert it is slot K: h gets (K+1)*I values per token,
    // and one more work-group per token computes the shared gate.
    const bool has_sh = SH && wsh != nullptr;
    const int KS = K + (has_sh ? 1 : 0);
    const int rows_per_wg = WG_SUBGROUPS * R;
    const int row_groups = (KS * I + rows_per_wg - 1) / rows_per_wg;
    const int groups_per_token = row_groups + (gate_w ? 1 : 0);
    const int n_groups = M * groups_per_token;
    const bool fast_w = [&] {
        if constexpr (F == Fmt::Q4_K || F == Fmt::Q6_K)
            return L.gate_up.K == H && native_qk_fast_weight<F>(L.gate_up);
        return false;
    }();
    const bool fast_ws = [&] {
        if constexpr (F == Fmt::Q4_K || F == Fmt::Q6_K)
            return has_sh && wsh->K == H && native_qk_fast_weight<F>(*wsh);
        return false;
    }();

    return q.submit([&](sycl::handler& hc) {
        hc.depends_on(deps);
        const QuantWeight w = L.gate_up;
        const QuantWeight ws = has_sh ? *wsh : L.gate_up;
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

                float* xs = slmx.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                if (gate_w && local_group == row_groups) {
                    // shared_expert_gate: sigmoid(x . gate_w), the formula of
                    // launch_scale_by_sigmoid
                    if (int(sg.get_group_id()[0]) == 0) {
                        float a = 0.0f;
                        for (int c = lane; c < H; c += SG_SIZE)
                            a = sycl::fma(sycl::bit_cast<float>(uint32_t(gate_w[c]) << 16), xs[c], a);
                        const float g = sycl::reduce_over_group(sg, a, sycl::plus<float>());
                        if (lane == 0) gate_out[token] = 1.0f / (1.0f + sycl::exp(-g));
                    }
                    return;
                }
                const int slot_row_base = local_group * rows_per_wg
                                        + int(sg.get_group_id()[0]) * R;
                // the R rows of a sub-group share one slot (I % R == 0)
                // (SH is a template flag: the unfused kernel compiles as before)
                const bool shr = SH && has_sh && slot_row_base / I >= K;

                // All R slots share the staged activation, and their weight
                // loads are independent, so the memory system sees R x 2
                // outstanding streams instead of 2.
                float ga[R], ua[R];
                int64_t grow[R], urow[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    ga[r] = 0.0f; ua[r] = 0.0f;
                    const int sr = slot_row_base + r;
                    if (sr >= KS * I) { grow[r] = -1; urow[r] = -1; continue; }
                    const int slot = sr / I;
                    const int i    = sr % I;
                    if (slot < K) {
                        const int e = d_expert[int64_t(token) * K + slot];
                        grow[r] = int64_t(e) * stride2I + i;
                    } else {
                        grow[r] = i;                 // shared expert: its own [2I][H]
                    }
                    urow[r] = grow[r] + I;
                }
                // The dot loops take the weight as a parameter and are
                // inlined once per weight: a runtime pick between two
                // captured QuantWeights put the descriptor in private memory
                // and slowed every load (1.57 -> 1.81 s per 160 tokens).
                auto dots = [&](const QuantWeight& wr, bool fast) {
                    if constexpr (F == Fmt::Q4_K || F == Fmt::Q6_K) {
                        if (fast) {
                            constexpr int BYTES = F == Fmt::Q4_K ? 144 : 210;
                            const float* xp[1] = {xs};
                            for (int base = 0; base < H; base += 256) {
                                #pragma unroll
                                for (int r = 0; r < R; ++r) {
                                    if (grow[r] < 0) continue;
                                    const uint8_t* gp = wr.payload + grow[r] * wr.row_bytes + int64_t(base / 256) * BYTES;
                                    const uint8_t* up = wr.payload + urow[r] * wr.row_bytes + int64_t(base / 256) * BYTES;
                                    native_qk_block_lane<F, 1>(gp, xp, base, lane, 1, &ga[r]);
                                    native_qk_block_lane<F, 1>(up, xp, base, lane, 1, &ua[r]);
                                }
                            }
                            return;
                        }
                    }
                    for (int base = 0; base + GEMV_STEP <= H; base += GEMV_STEP) {
                        const int k0 = base + lane * GEMV_EPL;
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            if (grow[r] < 0) continue;
                            const uint8_t* gp = wr.payload + grow[r] * wr.row_bytes;
                            const uint8_t* up = wr.payload + urow[r] * wr.row_bytes;
                            ga[r] += GemvStep<F, GEMV_EPL>::run(
                                wr, gp, xs, lut, slut, nlut, int(grow[r]), k0);
                            ua[r] += GemvStep<F, GEMV_EPL>::run(
                                wr, up, xs, lut, slut, nlut, int(urow[r]), k0);
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (grow[r] < 0) continue;
                        for (int c = (H / GEMV_STEP) * GEMV_STEP + lane; c < H; c += SG_SIZE) {
                            ga[r] = sycl::fma(wr.at(int(grow[r]), c), xs[c], ga[r]);
                            ua[r] = sycl::fma(wr.at(int(urow[r]), c), xs[c], ua[r]);
                        }
                    }
                };
                if constexpr (SH) { if (shr) dots(ws, fast_ws); else dots(w, fast_w); }
                else dots(w, fast_w);
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int sr = slot_row_base + r;
                    const float g = sycl::reduce_over_group(sg, ga[r], sycl::plus<float>());
                    const float u = sycl::reduce_over_group(sg, ua[r], sycl::plus<float>());
                    if (lane == 0 && sr < KS * I)
                        h[(int64_t(token) * KS * I) + sr] = silu(g) * u;
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
template <Fmt F, int R, bool SH = false>
sycl::event moe_down_impl_r(sycl::queue& q, const MoeLayer& L,
                          const int32_t* d_expert,   // [k]
                          const float* d_weight,     // [k]
                          const float* h,            // [k][I]
                          float* y,                  // [M][H] out
                          int M,
                          const std::vector<sycl::event>& deps,
                          const QuantWeight* wdsh = nullptr,  // shared expert [H][I]
                          const float* gate_in = nullptr) {   // its weight per token
    const int H = L.cfg.hidden, I = L.cfg.inter, K = L.cfg.top_k;
    const bool has_sh = SH && wdsh != nullptr;   // shared expert = slot K of h
    const int KS = K + (has_sh ? 1 : 0);
    const int rows_per_wg = WG_SUBGROUPS * R;
    const int groups_per_token = (H + rows_per_wg - 1) / rows_per_wg;
    const int n_groups = M * groups_per_token;
    const bool fast_w = [&] {
        if constexpr (F == Fmt::Q4_K || F == Fmt::Q6_K)
            return L.down.K == I && native_qk_fast_weight<F>(L.down);
        return false;
    }();
    const bool fast_wd = [&] {
        if constexpr (F == Fmt::Q4_K || F == Fmt::Q6_K)
            return has_sh && wdsh->K == I && native_qk_fast_weight<F>(*wdsh);
        return false;
    }();

    return q.submit([&](sycl::handler& hc) {
        hc.depends_on(deps);
        const QuantWeight w = L.down;
        const QuantWeight wd = has_sh ? *wdsh : L.down;

        // h is k*I floats: 8*512 = 4096, 16 KB. Fits SLM comfortably and
        // every output row reads all of it.
        sycl::local_accessor<float, 1> slmh(size_t(KS) * size_t(I), hc);
        // Decode tables, shared by the whole work-group. Identical to the
        // dense GEMV path -- FP8 bit-assembly and E8M0/E2M1 branches cost
        // more than the loads they decorate.
        sycl::local_accessor<float, 1> lut_slm(256, hc);
        sycl::local_accessor<float, 1> e8m0_slm(256, hc);
        sycl::local_accessor<float, 1> e2m1_slm(16, hc);


        // WG_SUBGROUPS sub-groups x R rows = rows_per_wg rows per group.
        // This launched rows_per_wg SUB-GROUPS before, so at R > 1 every
        // group also recomputed its neighbour's rows (same values, written
        // twice) -- R times the work, which is what the "R=4 slower than
        // R=1" measurement below was actually seeing.
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

                const int token = int(it.get_group(0)) / groups_per_token;
                const int local_group = int(it.get_group(0)) % groups_per_token;
                const float* ht = h + int64_t(token) * KS * I;
                for (int c = lid; c < KS * I; c += WG_SUBGROUPS * SG_SIZE) slmh[c] = ht[c];
                sycl::group_barrier(it.get_group());

                const int o_base = local_group * rows_per_wg
                                 + int(sg.get_group_id()[0]) * R;

                float total[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) total[r] = 0.0f;

                // One inlined copy per weight (see moe_gate_up_impl_r: a
                // runtime pick between two QuantWeights slows every load).
                auto slot_dot = [&](const QuantWeight& wr, int64_t ebase, int slot, float rw, bool fast) {
                    float acc[R];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
                    if constexpr (F == Fmt::Q4_K || F == Fmt::Q6_K) {
                        if (fast) {
                            constexpr int BYTES = F == Fmt::Q4_K ? 144 : 210;
                            const float* xp[1] = {&slmh[slot * I]};
                            for (int base = 0; base < I; base += 256) {
                                #pragma unroll
                                for (int r = 0; r < R; ++r) {
                                    const int o = o_base + r;
                                    if (o >= H) continue;
                                    const uint8_t* dp = wr.payload + (ebase + o) * wr.row_bytes
                                                      + int64_t(base / 256) * BYTES;
                                    native_qk_block_lane<F, 1>(dp, xp, base, lane, 1, &acc[r]);
                                }
                            }
                        } else {
                            for (int base = 0; base + GEMV_STEP <= I; base += GEMV_STEP) {
                                const int k0 = base + lane * GEMV_EPL;
                                #pragma unroll
                                for (int r = 0; r < R; ++r) {
                                    const int o = o_base + r;
                                    if (o >= H) continue;
                                    const int64_t d_row = ebase + o;
                                    const uint8_t* dp = wr.payload + d_row * wr.row_bytes;
                                    acc[r] += GemvStep<F, GEMV_EPL>::run(wr, dp, &slmh[slot * I],
                                                    lut, slut, nlut, int(d_row), k0);
                                }
                            }
                        }
                    } else {
                    for (int base = 0; base + GEMV_STEP <= I; base += GEMV_STEP) {
                        const int k0 = base + lane * GEMV_EPL;
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            const int o = o_base + r;
                            if (o >= H) continue;
                            const int64_t d_row = ebase + o;
                            const uint8_t* dp = wr.payload + d_row * wr.row_bytes;
                            acc[r] += GemvStep<F, GEMV_EPL>::run(wr, dp, &slmh[slot * I] - 0,
                                                    lut, slut, nlut, int(d_row), k0);
                        }
                    }
                    }
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const int o = o_base + r;
                        if (o >= H) continue;
                        const int64_t d_row = ebase + o;
                        if (!fast)
                            for (int c = (I / GEMV_STEP) * GEMV_STEP + lane; c < I; c += SG_SIZE)
                                acc[r] = sycl::fma(wr.at(int(d_row), c), slmh[slot * I + c], acc[r]);
                        // Keep the routed sum in per-lane registers. The
                        // sub-group reduction is linear, so scaling each
                        // expert partial by its router weight and reducing
                        // ONCE at the end is identical arithmetic with K
                        // times fewer shuffles. down has only I=512 of
                        // reduction depth, so a reduce per expert slot cost
                        // more than the MACs between them.
                        total[r] = sycl::fma(rw, acc[r], total[r]);
                    }
                };
                for (int slot = 0; slot < K; ++slot) {
                    const int64_t route = int64_t(token) * K + slot;
                    const int   e  = d_expert[route];
                    const float rw = d_weight[route];
                    if (e < 0 || rw == 0.0f) continue;
                    slot_dot(w, int64_t(e) * H, slot, rw, fast_w);
                }
                if constexpr (SH) {
                    if (has_sh)                       // shared expert: slot K, its own [H][I]
                        slot_dot(wd, 0, K, gate_in ? gate_in[token] : 1.0f, fast_wd);
                }
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int o = o_base + r;
                    const float t = sycl::reduce_over_group(
                        sg, total[r], sycl::plus<float>());
                    if (lane == 0 && o < H) y[int64_t(token) * H + o] = t;
                }
            });
    });
}

// ---------------------------------------------------------------------
// K2 MoVA value projection, routed on the device (decode and small M).
//
//   y[n] = sum_j w_j * silu( V[e_j][n] . x )        e_j, w_j from rex/rwt
//
// V is the E value experts packed expert-major, one [E*N][K] weight, so
// expert e's row n is global row e*N + n.  The structure is moe_down's --
// x staged in SLM, the decode LUTs, GemvStep, the routing table read in the
// kernel -- with one difference: SiLU sits between the dot product and the
// router weight, so each slot is reduced across the sub-group before it is
// accumulated.  moe_down can defer its single reduction because its
// combine is linear; this one cannot.
//
// Replaces mova_value_packed_impl (ops.cpp) on this path, which gave each
// output row to ONE work-item walking all K serially per expert: 1,024
// work-items per layer at M=1, latency-bound, ~8 tok/s end to end on K2.
// ---------------------------------------------------------------------
template <Fmt F, int R>
sycl::event mova_decode_impl_r(sycl::queue& q, const QuantWeight& w,
                               const float* x,            // [M][K]
                               const int32_t* rex,        // [M][top_k]
                               const float* rwt,          // [M][top_k]
                               float* y,                  // [M][N] out
                               int M, int N, int E, int top_k,
                               const std::vector<sycl::event>& deps) {
    const int K = w.K;
    const int rows_per_wg = WG_SUBGROUPS * R;
    const int groups_per_token = (N + rows_per_wg - 1) / rows_per_wg;
    const int n_groups = M * groups_per_token;
    const bool fast = [&] {
        if constexpr (F == Fmt::Q4_K || F == Fmt::Q6_K)
            return native_qk_fast_weight<F>(w);
        return false;
    }();

    return q.submit([&](sycl::handler& hc) {
        hc.depends_on(deps);
        const QuantWeight wc = w;
        sycl::local_accessor<float, 1> slmx(size_t(K), hc);
        sycl::local_accessor<float, 1> lut_slm(256, hc);
        sycl::local_accessor<float, 1> e8m0_slm(256, hc);
        sycl::local_accessor<float, 1> e2m1_slm(16, hc);

        // WG_SUBGROUPS sub-groups of R rows each = rows_per_wg rows per
        // work-group.  (moe_down launches rows_per_wg sub-groups, which for
        // R > 1 makes neighbouring groups recompute each other's rows.)
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
                }
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  lid  = int(it.get_local_id(0));
                const int  lsz  = int(it.get_local_range(0));

                const int token = int(it.get_group(0)) / groups_per_token;
                const int local_group = int(it.get_group(0)) % groups_per_token;
                const float* xt = x + int64_t(token) * K;
                for (int c = lid; c < K; c += lsz) slmx[c] = xt[c];
                sycl::group_barrier(it.get_group());
                const float* xs = slmx.template
                    get_multi_ptr<sycl::access::decorated::no>().get();

                const int n_base = local_group * rows_per_wg
                                 + int(sg.get_group_id()[0]) * R;
                float total[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) total[r] = 0.0f;

                for (int slot = 0; slot < top_k; ++slot) {
                    const int64_t route = int64_t(token) * top_k + slot;
                    const int   e  = rex[route];
                    if (e < 0 || e >= E) continue;       // router produced no route
                    const float rw = rwt[route];

                    float acc[R];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
                    if constexpr (F == Fmt::Q4_K || F == Fmt::Q6_K) {
                        if (fast) {
                            constexpr int BYTES = F == Fmt::Q4_K ? 144 : 210;
                            const float* xp[1] = {xs};
                            for (int base = 0; base < K; base += 256) {
                                #pragma unroll
                                for (int r = 0; r < R; ++r) {
                                    const int n = n_base + r;
                                    if (n >= N) continue;
                                    const int64_t row = int64_t(e) * N + n;
                                    const uint8_t* rp = wc.payload + row * wc.row_bytes
                                                      + int64_t(base / 256) * BYTES;
                                    native_qk_block_lane<F, 1>(rp, xp, base, lane, 1, &acc[r]);
                                }
                            }
                        } else {
                            for (int base = 0; base + GEMV_STEP <= K; base += GEMV_STEP) {
                                const int k0 = base + lane * GEMV_EPL;
                                #pragma unroll
                                for (int r = 0; r < R; ++r) {
                                    const int n = n_base + r;
                                    if (n >= N) continue;
                                    const int64_t row = int64_t(e) * N + n;
                                    const uint8_t* rp = wc.payload + row * wc.row_bytes;
                                    acc[r] += GemvStep<F, GEMV_EPL>::run(wc, rp, xs,
                                                  lut, slut, nlut, int(row), k0);
                                }
                            }
                        }
                    } else {
                    for (int base = 0; base + GEMV_STEP <= K; base += GEMV_STEP) {
                        const int k0 = base + lane * GEMV_EPL;
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            const int n = n_base + r;
                            if (n >= N) continue;
                            const int64_t row = int64_t(e) * N + n;
                            const uint8_t* rp = wc.payload + row * wc.row_bytes;
                            acc[r] += GemvStep<F, GEMV_EPL>::run(wc, rp, xs,
                                          lut, slut, nlut, int(row), k0);
                        }
                    }
                    }
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const int n = n_base + r;
                        const int64_t row = int64_t(e) * N + (n < N ? n : 0);
                        if (n < N && !fast)
                            for (int c = (K / GEMV_STEP) * GEMV_STEP + lane; c < K; c += SG_SIZE)
                                acc[r] = sycl::fma(wc.at(int(row), c), xs[c], acc[r]);
                        // every lane gets the full dot: silu is applied
                        // redundantly per lane, and total[] stays uniform
                        const float s = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                        total[r] = sycl::fma(rw, s / (1.0f + sycl::exp(-s)), total[r]);
                    }
                }
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int n = n_base + r;
                    if (lane == 0 && n < N) y[int64_t(token) * N + n] = total[r];
                }
            });
    });
}

template <Fmt F>
sycl::event mova_decode_impl(sycl::queue& q, const QuantWeight& w, const float* x,
                             const int32_t* rex, const float* rwt, float* y,
                             int M, int N, int E, int top_k,
                             const std::vector<sycl::event>& deps) {
    // Rows per sub-group.  Each row is top_k independent streams already;
    // B70_MOVA_ROWS sweeps it without a rebuild.  Measured on K2, 256
    // tokens: 1 -> 4.375 s, 2 -> 4.504 s, 4 -> 4.776 s.
    static const int rows = []{ const char* e = std::getenv("B70_MOVA_ROWS");
        int v = (e && *e) ? std::atoi(e) : 1;
        return (v == 1 || v == 2 || v == 4) ? v : 1; }();
    switch (rows) {
        case 1:  return mova_decode_impl_r<F, 1>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
        case 4:  return mova_decode_impl_r<F, 4>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
        default: return mova_decode_impl_r<F, 2>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
    }
}

template <Fmt F>
sycl::event moe_down_impl(sycl::queue& q, const MoeLayer& L,
                          const int32_t* d_expert, const float* d_weight,
                          const float* h, float* y, int M,
                          const std::vector<sycl::event>& deps) {
    // R=1 is best here: the slot loop over K experts already gives 8
    // independent streams, so widening only spills registers.  Measured
    // R=4 -> 113.4 TG against R=1 125.6.
    // 2 rows per sub-group.  MEASURED 2026-09-30 on Ornith with the shared
    // expert fused in: moe_down 30.0 -> 23.4 us (1 -> 2), 34.1 us at 4.
    static const int slots = []{ const char* e = std::getenv("B70_MOE_DN_SLOTS");
        int x = (e && *e) ? std::atoi(e) : 2;
        return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 2; }();
    switch (slots) {
        case 1: return moe_down_impl_r<F, 1>(q, L, d_expert, d_weight, h, y, M, deps);
        case 2: return moe_down_impl_r<F, 2>(q, L, d_expert, d_weight, h, y, M, deps);
        case 8: return moe_down_impl_r<F, 8>(q, L, d_expert, d_weight, h, y, M, deps);
        default: return moe_down_impl_r<F, 4>(q, L, d_expert, d_weight, h, y, M, deps);
    }
}

// Decode with the shared expert as slot top_k (see moe_gate_up_impl_r /
// moe_down_impl_r): same slot-count choice as the unfused kernels.
template <Fmt F>
sycl::event moe_gate_up_sh(sycl::queue& q, const MoeLayer& L, const QuantWeight& ws,
                           const uint16_t* gate_w, const int32_t* d_expert,
                           const float* x, float* h, float* gate_out,
                           const std::vector<sycl::event>& deps) {
    static const int slots = []{ const char* e = std::getenv("B70_MOE_SLOTS");
        int x = (e && *e) ? std::atoi(e) : 4;
        return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 4; }();
    switch (slots) {
        case 1: return moe_gate_up_impl_r<F, 1, true>(q, L, d_expert, x, h, 1, deps, &ws, gate_w, gate_out);
        case 2: return moe_gate_up_impl_r<F, 2, true>(q, L, d_expert, x, h, 1, deps, &ws, gate_w, gate_out);
        case 8: return moe_gate_up_impl_r<F, 8, true>(q, L, d_expert, x, h, 1, deps, &ws, gate_w, gate_out);
        default: return moe_gate_up_impl_r<F, 4, true>(q, L, d_expert, x, h, 1, deps, &ws, gate_w, gate_out);
    }
}
template <Fmt F>
sycl::event moe_down_sh(sycl::queue& q, const MoeLayer& L, const QuantWeight& wd,
                        const int32_t* d_expert, const float* d_weight,
                        const float* gate_in, const float* h, float* y,
                        const std::vector<sycl::event>& deps) {
    // 2 rows per sub-group.  MEASURED 2026-09-30 on Ornith with the shared
    // expert fused in: moe_down 30.0 -> 23.4 us (1 -> 2), 34.1 us at 4.
    static const int slots = []{ const char* e = std::getenv("B70_MOE_DN_SLOTS");
        int x = (e && *e) ? std::atoi(e) : 2;
        return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 2; }();
    switch (slots) {
        case 1: return moe_down_impl_r<F, 1, true>(q, L, d_expert, d_weight, h, y, 1, deps, &wd, gate_in);
        case 2: return moe_down_impl_r<F, 2, true>(q, L, d_expert, d_weight, h, y, 1, deps, &wd, gate_in);
        case 8: return moe_down_impl_r<F, 8, true>(q, L, d_expert, d_weight, h, y, 1, deps, &wd, gate_in);
        default: return moe_down_impl_r<F, 4, true>(q, L, d_expert, d_weight, h, y, 1, deps, &wd, gate_in);
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


// ---------------------------------------------------------------------
// ESIMD decode MoE (MXFP4, one token), the llm-scaler shape.
//
// The SIMT kernels above stage the activation and three decode tables in SLM
// per work-group and walk 2-8 rows per sub-group: MEASURED 2026-10-01 with
// cold weights (tools/decode_probe.cpp, 8 random experts of 256 + shared),
// gate_up 37.7 us = 266 GB/s and down 20.2 us = 248 GB/s.  Here a thread
// streams whole rows with contiguous 64-byte block loads, keeps its slice of
// the activation in registers split into even / odd K (the low / high nibble
// of each payload byte), and decodes E2M1 in the ALU: the nibble's three
// magnitude bits at fp16 bits 9..11 plus the sign at bit 15 ARE the fp16
// value times 2^-14 (fp16 subnormals hold the E2M1 subnormal); the 2^14 folds
// into the E8M0 block scale.  No tables, no barrier before the K-split sum.
// Same probe: gate_up 21.0 us (477 GB/s), down 10.4 us (484 GB/s), max
// relative error 2e-7 against the SIMT kernels (fp32 summation order).
// GRIMOIRE_MOE_ESIMD_DECODE=0 = the SIMT kernels.
// ---------------------------------------------------------------------
namespace es = sycl::ext::intel::esimd;

template <int N>
SYCL_ESIMD_FUNCTION inline void e2m1_split(es::simd<uint8_t, N> b, es::simd<float, N>& lo,
                                           es::simd<float, N>& hi) {
    es::simd<uint16_t, N> u = b;
    es::simd<uint16_t, N> l = ((u & 0x7) << 9) | ((u & 0x8) << 12);
    es::simd<uint16_t, N> hb = ((u & 0x70) << 5) | ((u & 0x80) << 8);
    es::simd<sycl::half, N> lh = l.template bit_cast_view<sycl::half>();
    es::simd<sycl::half, N> hh = hb.template bit_cast_view<sycl::half>();
    lo = lh;
    hi = hh;
}
SYCL_ESIMD_FUNCTION inline float e8m0_x2p14(uint32_t e) {   // 2^(e-127) * 2^14
    return sycl::bit_cast<float>((e + 14u) << 23);
}
// One 128-element step of one row: acc += sum_b scale_b * (lo.xe + hi.xo).
template <int KP>
SYCL_ESIMD_FUNCTION inline void mx4_row_step(const uint8_t* prow, const uint8_t* srow, int st,
                                             es::simd<float, KP / 2>& xe,
                                             es::simd<float, KP / 2>& xo,
                                             es::simd<float, 16>& acc) {
    es::simd<uint8_t, 64> pb = es::block_load<uint8_t, 64>(prow + st * 64);
    es::simd<uint8_t, 4> sb = es::block_load<uint8_t, 4>(srow + st * 4);
    es::simd<float, 64> wl, wh;
    e2m1_split<64>(pb, wl, wh);
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        es::simd<float, 16> part =
            wl.template select<16, 1>(16 * b) * xe.template select<16, 1>(st * 64 + 16 * b) +
            wh.template select<16, 1>(16 * b) * xo.template select<16, 1>(st * 64 + 16 * b);
        acc += part * e8m0_x2p14(uint32_t(sb[b]));
    }
}

// gate_up + SiLU*up for TK routed experts plus the shared expert (slot TK),
// and the shared-expert gate in one extra work-group.  Work-group = KS
// threads splitting H; a thread owns R (gate, up) row pairs of one slot.
template <int R, int KS, int H>
sycl::event moe_gu_esimd(sycl::queue& q, const QuantWeight& w, const QuantWeight& ws,
                         const uint16_t* gate_w, const int32_t* d_expert, const float* x,
                         float* h, float* gate_out, int I, int TK,
                         const std::vector<sycl::event>& deps) {
    constexpr int KP = H / KS;
    static_assert(KP % 128 == 0, "K slice must be whole 128-element steps");
    const int rg = I / R;
    const int n_wg = (TK + 1) * rg + (gate_w ? 1 : 0);
    const uint8_t* wp = w.payload; const uint8_t* wsc = static_cast<const uint8_t*>(w.scales);
    const uint8_t* sp = ws.payload; const uint8_t* ssc = static_cast<const uint8_t*>(ws.scales);
    const int64_t wrb = w.row_bytes, wrs = w.row_scales, srb = ws.row_bytes, srs = ws.row_scales;
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::nd_range<1>(size_t(n_wg) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<(KS * R * 2 * 4 > KS * 16 ? KS * R * 2 * 4 : KS * 16)>();
            const int t = int(it.get_local_id(0));
            const int g = int(it.get_group(0));
            const int kb = t * KP;
            if (g == (TK + 1) * rg) {                           // shared-expert gate
                es::simd<float, 16> a = 0.0f;
                for (int k = kb; k < kb + KP; k += 16) {
                    es::simd<uint16_t, 16> gb = es::block_load<uint16_t, 16>(gate_w + k);
                    es::simd<uint32_t, 16> gu = es::convert<uint32_t>(gb) << 16;
                    es::simd<float, 16> gf = gu.template bit_cast_view<float>();
                    a += gf * es::block_load<float, 16>(x + k);
                }
                es::simd<float, 4> av = 0.0f;
                av[0] = es::reduce<float>(a, std::plus<>());
                es::slm_block_store<float, 4>(t * 16, av);
                es::barrier();
                if (t == 0) {
                    float sum = 0.0f;
                    es::simd<float, 4 * KS> pv = es::slm_block_load<float, 4 * KS>(0);
                    #pragma unroll
                    for (int j = 0; j < KS; ++j) sum += pv[4 * j];
                    gate_out[0] = 1.0f / (1.0f + sycl::exp(-sum));
                }
                return;
            }
            const int slot = g / rg, i0 = (g % rg) * R;
            const bool sh = slot == TK;
            const uint8_t* pay = sh ? sp : wp;
            const uint8_t* scl = sh ? ssc : wsc;
            const int64_t rb = sh ? srb : wrb, rs = sh ? srs : wrs;
            int e = sh ? 0 : d_expert[slot];
            if (e < 0) e = 0;              // unrouted slot: down weights it 0
            const int64_t row0 = sh ? int64_t(i0) : int64_t(e) * 2 * I + i0;
            es::simd<float, KP> xs = es::block_load<float, KP>(x + kb);
            es::simd<float, KP / 2> xe = xs.template select<KP / 2, 2>(0);
            es::simd<float, KP / 2> xo = xs.template select<KP / 2, 2>(1);
            es::simd<float, 16> ga[R], ua[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) { ga[r] = 0.0f; ua[r] = 0.0f; }
            #pragma unroll 1
            for (int st = 0; st < KP / 128; ++st) {
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int64_t gr = row0 + r, ur = row0 + I + r;
                    mx4_row_step<KP>(pay + gr * rb + kb / 2, scl + gr * rs + kb / 32, st, xe, xo, ga[r]);
                    mx4_row_step<KP>(pay + ur * rb + kb / 2, scl + ur * rs + kb / 32, st, xe, xo, ua[r]);
                }
            }
            es::simd<float, 2 * R> red;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                red[2 * r] = es::reduce<float>(ga[r], std::plus<>());
                red[2 * r + 1] = es::reduce<float>(ua[r], std::plus<>());
            }
            if constexpr (KS > 1) {
                es::slm_block_store<float, 2 * R>(t * 2 * R * 4, red);
                es::barrier();
                if (t != 0) return;
                red = 0.0f;
                #pragma unroll
                for (int j = 0; j < KS; ++j) red += es::slm_block_load<float, 2 * R>(j * 2 * R * 4);
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const float gv = red[2 * r], uv = red[2 * r + 1];
                h[int64_t(slot) * I + i0 + r] = gv / (1.0f + sycl::exp(-gv)) * uv;
            }
        });
    });
}

// down + routed-weight reduction over the TK experts and the shared expert.
// Work-group = TK+1 threads, thread s = slot s; a work-group owns R output
// rows; slot partials meet in SLM and thread 0 sums them in slot order.
template <int R, int I>
sycl::event moe_dn_esimd(sycl::queue& q, const QuantWeight& w, const QuantWeight& wd,
                         const int32_t* d_expert, const float* d_weight, const float* gate_in,
                         const float* h, float* y, int H, int TK,
                         const std::vector<sycl::event>& deps) {
    static_assert(I % 128 == 0, "");
    constexpr int RP = R < 4 ? 4 : R;
    const uint8_t* wp = w.payload; const uint8_t* wsc = static_cast<const uint8_t*>(w.scales);
    const uint8_t* sp = wd.payload; const uint8_t* ssc = static_cast<const uint8_t*>(wd.scales);
    const int64_t wrb = w.row_bytes, wrs = w.row_scales, srb = wd.row_bytes, srs = wd.row_scales;
    const int S = TK + 1;
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::nd_range<1>(size_t(H / R) * S, S), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<16 * RP * 4>();
            const int s = int(it.get_local_id(0));
            const int o0 = int(it.get_group(0)) * R;
            const bool sh = s == TK;
            const uint8_t* pay = sh ? sp : wp;
            const uint8_t* scl = sh ? ssc : wsc;
            const int64_t rb = sh ? srb : wrb, rs = sh ? srs : wrs;
            int e = sh ? 0 : d_expert[s];
            float wt = sh ? (gate_in ? gate_in[0] : 1.0f) : d_weight[s];
            if (e < 0) { e = 0; wt = 0.0f; }
            const int64_t row0 = sh ? int64_t(o0) : int64_t(e) * H + o0;
            es::simd<float, I> hs = es::block_load<float, I>(h + int64_t(s) * I);
            es::simd<float, I / 2> he = hs.template select<I / 2, 2>(0);
            es::simd<float, I / 2> ho = hs.template select<I / 2, 2>(1);
            es::simd<float, 16> acc[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0f;
            #pragma unroll
            for (int st = 0; st < I / 128; ++st) {
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    mx4_row_step<I>(pay + (row0 + r) * rb, scl + (row0 + r) * rs, st, he, ho, acc[r]);
            }
            es::simd<float, RP> part = 0.0f;
            #pragma unroll
            for (int r = 0; r < R; ++r) part[r] = es::reduce<float>(acc[r], std::plus<>()) * wt;
            es::slm_block_store<float, RP>(s * RP * 4, part);
            es::barrier();
            if (s != 0) return;
            es::simd<float, RP> tot = 0.0f;
            for (int j = 0; j < S; ++j) tot += es::slm_block_load<float, RP>(j * RP * 4);
            #pragma unroll
            for (int r = 0; r < R; ++r) y[o0 + r] = tot[r];
        });
    });
}

// Batched decode: moe_gu_esimd / moe_dn_esimd for M tokens in one launch --
// the decode kernels with a token index in front of every work-group.  Token
// m reads x[m], its own top-k experts d_expert[m*TK..] and writes h[m] (TK+1
// slots, the shared expert last) / y[m]; every line of arithmetic is the
// decode kernel's, so each token's MoE output is bit-identical to a decode
// step's.  Each (token, expert) pair streams that expert once: at M <= 4 two
// tokens rarely share one of 8-of-256 experts, so this reads what a grouped
// kernel would, at the decode kernels' ~380 GB/s (the grouped small-M kernels
// ran at ~220-300 GB/s, handoff 10-04).  The shared expert is re-read per
// token (1.5 MB per token per layer).
template <int R, int KS, int H>
sycl::event moe_gu_esimd_rows(sycl::queue& q, const QuantWeight& w, const QuantWeight& ws,
                              const uint16_t* gate_w, const int32_t* d_expert, const float* x,
                              float* h, float* gate_out, int I, int TK, int M,
                              const std::vector<sycl::event>& deps) {
    constexpr int KP = H / KS;
    static_assert(KP % 128 == 0, "K slice must be whole 128-element steps");
    const int rg = I / R;
    const int per_tok = (TK + 1) * rg + (gate_w ? 1 : 0);
    const uint8_t* wp = w.payload; const uint8_t* wsc = static_cast<const uint8_t*>(w.scales);
    const uint8_t* sp = ws.payload; const uint8_t* ssc = static_cast<const uint8_t*>(ws.scales);
    const int64_t wrb = w.row_bytes, wrs = w.row_scales, srb = ws.row_bytes, srs = ws.row_scales;
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::nd_range<1>(size_t(M) * per_tok * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<(KS * R * 2 * 4 > KS * 16 ? KS * R * 2 * 4 : KS * 16)>();
            const int t = int(it.get_local_id(0));
            const int m = int(it.get_group(0)) / per_tok;
            const int g = int(it.get_group(0)) % per_tok;
            const int kb = t * KP;
            const float* xm = x + int64_t(m) * H;
            if (g == (TK + 1) * rg) {                           // shared-expert gate
                es::simd<float, 16> a = 0.0f;
                for (int k = kb; k < kb + KP; k += 16) {
                    es::simd<uint16_t, 16> gb = es::block_load<uint16_t, 16>(gate_w + k);
                    es::simd<uint32_t, 16> gu = es::convert<uint32_t>(gb) << 16;
                    es::simd<float, 16> gf = gu.template bit_cast_view<float>();
                    a += gf * es::block_load<float, 16>(xm + k);
                }
                es::simd<float, 4> av = 0.0f;
                av[0] = es::reduce<float>(a, std::plus<>());
                es::slm_block_store<float, 4>(t * 16, av);
                es::barrier();
                if (t == 0) {
                    float sum = 0.0f;
                    es::simd<float, 4 * KS> pv = es::slm_block_load<float, 4 * KS>(0);
                    #pragma unroll
                    for (int j = 0; j < KS; ++j) sum += pv[4 * j];
                    gate_out[m] = 1.0f / (1.0f + sycl::exp(-sum));
                }
                return;
            }
            const int slot = g / rg, i0 = (g % rg) * R;
            const bool sh = slot == TK;
            const uint8_t* pay = sh ? sp : wp;
            const uint8_t* scl = sh ? ssc : wsc;
            const int64_t rb = sh ? srb : wrb, rs = sh ? srs : wrs;
            int e = sh ? 0 : d_expert[int64_t(m) * TK + slot];
            if (e < 0) e = 0;              // unrouted slot: down weights it 0
            const int64_t row0 = sh ? int64_t(i0) : int64_t(e) * 2 * I + i0;
            es::simd<float, KP> xs = es::block_load<float, KP>(xm + kb);
            es::simd<float, KP / 2> xe = xs.template select<KP / 2, 2>(0);
            es::simd<float, KP / 2> xo = xs.template select<KP / 2, 2>(1);
            es::simd<float, 16> ga[R], ua[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) { ga[r] = 0.0f; ua[r] = 0.0f; }
            #pragma unroll 1
            for (int st = 0; st < KP / 128; ++st) {
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int64_t gr = row0 + r, ur = row0 + I + r;
                    mx4_row_step<KP>(pay + gr * rb + kb / 2, scl + gr * rs + kb / 32, st, xe, xo, ga[r]);
                    mx4_row_step<KP>(pay + ur * rb + kb / 2, scl + ur * rs + kb / 32, st, xe, xo, ua[r]);
                }
            }
            es::simd<float, 2 * R> red;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                red[2 * r] = es::reduce<float>(ga[r], std::plus<>());
                red[2 * r + 1] = es::reduce<float>(ua[r], std::plus<>());
            }
            if constexpr (KS > 1) {
                es::slm_block_store<float, 2 * R>(t * 2 * R * 4, red);
                es::barrier();
                if (t != 0) return;
                red = 0.0f;
                #pragma unroll
                for (int j = 0; j < KS; ++j) red += es::slm_block_load<float, 2 * R>(j * 2 * R * 4);
            }
            float* hm = h + int64_t(m) * (TK + 1) * I;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const float gv = red[2 * r], uv = red[2 * r + 1];
                hm[int64_t(slot) * I + i0 + r] = gv / (1.0f + sycl::exp(-gv)) * uv;
            }
        });
    });
}

template <int R, int I>
sycl::event moe_dn_esimd_rows(sycl::queue& q, const QuantWeight& w, const QuantWeight& wd,
                              const int32_t* d_expert, const float* d_weight, const float* gate_in,
                              const float* h, float* y, int H, int TK, int M,
                              const std::vector<sycl::event>& deps) {
    static_assert(I % 128 == 0, "");
    constexpr int RP = R < 4 ? 4 : R;
    const uint8_t* wp = w.payload; const uint8_t* wsc = static_cast<const uint8_t*>(w.scales);
    const uint8_t* sp = wd.payload; const uint8_t* ssc = static_cast<const uint8_t*>(wd.scales);
    const int64_t wrb = w.row_bytes, wrs = w.row_scales, srb = wd.row_bytes, srs = wd.row_scales;
    const int S = TK + 1;
    const int per_tok = H / R;
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::nd_range<1>(size_t(M) * per_tok * S, S), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<16 * RP * 4>();
            const int s = int(it.get_local_id(0));
            const int m = int(it.get_group(0)) / per_tok;
            const int o0 = (int(it.get_group(0)) % per_tok) * R;
            const bool sh = s == TK;
            const uint8_t* pay = sh ? sp : wp;
            const uint8_t* scl = sh ? ssc : wsc;
            const int64_t rb = sh ? srb : wrb, rs = sh ? srs : wrs;
            int e = sh ? 0 : d_expert[int64_t(m) * TK + s];
            float wt = sh ? (gate_in ? gate_in[m] : 1.0f) : d_weight[int64_t(m) * TK + s];
            if (e < 0) { e = 0; wt = 0.0f; }
            const int64_t row0 = sh ? int64_t(o0) : int64_t(e) * H + o0;
            es::simd<float, I> hs = es::block_load<float, I>(h + (int64_t(m) * S + s) * I);
            es::simd<float, I / 2> he = hs.template select<I / 2, 2>(0);
            es::simd<float, I / 2> ho = hs.template select<I / 2, 2>(1);
            es::simd<float, 16> acc[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0f;
            #pragma unroll
            for (int st = 0; st < I / 128; ++st) {
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    mx4_row_step<I>(pay + (row0 + r) * rb, scl + (row0 + r) * rs, st, he, ho, acc[r]);
            }
            es::simd<float, RP> part = 0.0f;
            #pragma unroll
            for (int r = 0; r < R; ++r) part[r] = es::reduce<float>(acc[r], std::plus<>()) * wt;
            es::slm_block_store<float, RP>(s * RP * 4, part);
            es::barrier();
            if (s != 0) return;
            es::simd<float, RP> tot = 0.0f;
            for (int j = 0; j < S; ++j) tot += es::slm_block_load<float, RP>(j * RP * 4);
            float* ym = y + int64_t(m) * H;
            #pragma unroll
            for (int r = 0; r < R; ++r) ym[o0 + r] = tot[r];
        });
    });
}

bool moe_esimd_decode_on() {
    static const bool v = [] { const char* e = std::getenv("GRIMOIRE_MOE_ESIMD_DECODE");
        return !(e && *e == '0'); }();
    return v;
}
bool mx4_rows_ok(const QuantWeight& w) {
    return w.fmt == Fmt::MXFP4 && w.payload && w.scales && w.row_bytes % 64 == 0 &&
           w.row_scales % 4 == 0 && w.row_bytes >= w.K / 2 && w.row_scales >= w.K / 32;
}
// Ornith / Qwen3.5-MoE geometry: H 2048, I 512 (shared expert the same).
bool moe_esimd_gu_ok(const MoeLayer& L, const QuantWeight& ws) {
    return moe_esimd_decode_on() && L.cfg.hidden == 2048 && L.cfg.inter == 512 &&
           L.cfg.top_k >= 1 && L.cfg.top_k <= 15 && L.gate_up.K == 2048 && mx4_rows_ok(L.gate_up) &&
           mx4_rows_ok(ws) && ws.N == 2 * 512 && ws.K == 2048;
}
bool moe_esimd_dn_ok(const MoeLayer& L, const QuantWeight& wd) {
    return moe_esimd_decode_on() && L.cfg.hidden == 2048 && L.cfg.inter == 512 &&
           L.cfg.top_k >= 1 && L.cfg.top_k <= 15 && L.down.K == 512 && mx4_rows_ok(L.down) &&
           mx4_rows_ok(wd) && wd.N == 2048 && wd.K == 512;
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
        case Fmt::Q4_K: return moe_gate_up_impl<Fmt::Q4_K>(q, L, d_expert, x, h, 1, deps);
        case Fmt::Q6_K: return moe_gate_up_impl<Fmt::Q6_K>(q, L, d_expert, x, h, 1, deps);
    }
    return {};
}

sycl::event launch_moe_gate_up_shared(sycl::queue& q, const MoeLayer& L, const QuantWeight& ws,
                                      const uint16_t* gate_w, const int32_t* d_expert,
                                      const float* x, float* h, float* gate_out,
                                      const std::vector<sycl::event>& deps) {
    if (moe_esimd_gu_ok(L, ws))
        return moe_gu_esimd<2, 8, 2048>(q, L.gate_up, ws, gate_w, d_expert, x, h, gate_out,
                                        L.cfg.inter, L.cfg.top_k, deps);
    switch (L.gate_up.fmt) {
        case Fmt::MXFP4:    return moe_gate_up_sh<Fmt::MXFP4>(q, L, ws, gate_w, d_expert, x, h, gate_out, deps);
        case Fmt::INT4:     return moe_gate_up_sh<Fmt::INT4>(q, L, ws, gate_w, d_expert, x, h, gate_out, deps);
        case Fmt::MXFP8:    return moe_gate_up_sh<Fmt::MXFP8>(q, L, ws, gate_w, d_expert, x, h, gate_out, deps);
        case Fmt::INT8:     return moe_gate_up_sh<Fmt::INT8>(q, L, ws, gate_w, d_expert, x, h, gate_out, deps);
        case Fmt::FP8_E4M3: return moe_gate_up_sh<Fmt::FP8_E4M3>(q, L, ws, gate_w, d_expert, x, h, gate_out, deps);
        case Fmt::FP8_E5M2: return moe_gate_up_sh<Fmt::FP8_E5M2>(q, L, ws, gate_w, d_expert, x, h, gate_out, deps);
        case Fmt::BF16:     return moe_gate_up_sh<Fmt::BF16>(q, L, ws, gate_w, d_expert, x, h, gate_out, deps);
        case Fmt::Q4_K: return moe_gate_up_sh<Fmt::Q4_K>(q, L, ws, gate_w, d_expert, x, h, gate_out, deps);
        case Fmt::Q6_K: return moe_gate_up_sh<Fmt::Q6_K>(q, L, ws, gate_w, d_expert, x, h, gate_out, deps);
    }
    return {};
}

sycl::event launch_moe_down_shared(sycl::queue& q, const MoeLayer& L, const QuantWeight& wd,
                                   const int32_t* d_expert, const float* d_weight,
                                   const float* gate_in, const float* h, float* y,
                                   const std::vector<sycl::event>& deps) {
    if (moe_esimd_dn_ok(L, wd))
        return moe_dn_esimd<2, 512>(q, L.down, wd, d_expert, d_weight, gate_in, h, y,
                                    L.cfg.hidden, L.cfg.top_k, deps);
    switch (L.down.fmt) {
        case Fmt::MXFP4:    return moe_down_sh<Fmt::MXFP4>(q, L, wd, d_expert, d_weight, gate_in, h, y, deps);
        case Fmt::INT4:     return moe_down_sh<Fmt::INT4>(q, L, wd, d_expert, d_weight, gate_in, h, y, deps);
        case Fmt::MXFP8:    return moe_down_sh<Fmt::MXFP8>(q, L, wd, d_expert, d_weight, gate_in, h, y, deps);
        case Fmt::INT8:     return moe_down_sh<Fmt::INT8>(q, L, wd, d_expert, d_weight, gate_in, h, y, deps);
        case Fmt::FP8_E4M3: return moe_down_sh<Fmt::FP8_E4M3>(q, L, wd, d_expert, d_weight, gate_in, h, y, deps);
        case Fmt::FP8_E5M2: return moe_down_sh<Fmt::FP8_E5M2>(q, L, wd, d_expert, d_weight, gate_in, h, y, deps);
        case Fmt::BF16:     return moe_down_sh<Fmt::BF16>(q, L, wd, d_expert, d_weight, gate_in, h, y, deps);
        case Fmt::Q4_K: return moe_down_sh<Fmt::Q4_K>(q, L, wd, d_expert, d_weight, gate_in, h, y, deps);
        case Fmt::Q6_K: return moe_down_sh<Fmt::Q6_K>(q, L, wd, d_expert, d_weight, gate_in, h, y, deps);
    }
    return {};
}

// Batched decode (M tokens): the ESIMD decode MoE pair over every token in two
// launches.  rex/rwt [M][top_k], x [M][H], h scratch [M][(top_k+1)*I],
// gate_out [M] (when gate_w), y [M][H] = routed + (gated) shared expert.
bool moe_shared_rows_ok(const MoeLayer& L, const QuantWeight& ws, const QuantWeight& wd) {
    return moe_esimd_gu_ok(L, ws) && moe_esimd_dn_ok(L, wd);
}
sycl::event launch_moe_shared_rows(sycl::queue& q, const MoeLayer& L, const QuantWeight& ws,
                                   const QuantWeight& wd, const uint16_t* gate_w,
                                   const int32_t* rex, const float* rwt, const float* x,
                                   float* h, float* gate_out, float* y, int M,
                                   const std::vector<sycl::event>& deps) {
    sycl::event e = moe_gu_esimd_rows<2, 8, 2048>(q, L.gate_up, ws, gate_w, rex, x, h, gate_out,
                                                  L.cfg.inter, L.cfg.top_k, M, deps);
    return moe_dn_esimd_rows<2, 512>(q, L.down, wd, rex, rwt, gate_w ? gate_out : nullptr, h, y,
                                     L.cfg.hidden, L.cfg.top_k, M, {e});
}

sycl::event launch_mova_value_decode(sycl::queue& q, const QuantWeight& w,
                                     const float* x, const int32_t* rex,
                                     const float* rwt, float* y,
                                     int M, int N, int E, int top_k,
                                     const std::vector<sycl::event>& deps) {
    switch (w.fmt) {
        case Fmt::MXFP4:    return mova_decode_impl<Fmt::MXFP4>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
        case Fmt::INT4:     return mova_decode_impl<Fmt::INT4>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
        case Fmt::MXFP8:    return mova_decode_impl<Fmt::MXFP8>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
        case Fmt::INT8:     return mova_decode_impl<Fmt::INT8>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
        case Fmt::FP8_E4M3: return mova_decode_impl<Fmt::FP8_E4M3>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
        case Fmt::FP8_E5M2: return mova_decode_impl<Fmt::FP8_E5M2>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
        case Fmt::BF16:     return mova_decode_impl<Fmt::BF16>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
        case Fmt::Q4_K: return mova_decode_impl<Fmt::Q4_K>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
        case Fmt::Q6_K: return mova_decode_impl<Fmt::Q6_K>(q, w, x, rex, rwt, y, M, N, E, top_k, deps);
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
        case Fmt::Q4_K: return moe_down_impl<Fmt::Q4_K>(q, L, d_expert, d_weight, h, y, 1, deps);
        case Fmt::Q6_K: return moe_down_impl<Fmt::Q6_K>(q, L, d_expert, d_weight, h, y, 1, deps);
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
        case Fmt::Q4_K: return moe_gate_up_impl<Fmt::Q4_K>(q, L, d_expert, x, h, tokens, deps);
        case Fmt::Q6_K: return moe_gate_up_impl<Fmt::Q6_K>(q, L, d_expert, x, h, tokens, deps);
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
        case Fmt::Q4_K: return moe_down_impl<Fmt::Q4_K>(q, L, d_expert, d_weight, h, y, tokens, deps);
        case Fmt::Q6_K: return moe_down_impl<Fmt::Q6_K>(q, L, d_expert, d_weight, h, y, tokens, deps);
    }
    return {};
}

} // namespace b70
