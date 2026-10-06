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
//  gemv_decode.cpp  --  the token-generation datapath
//
//  During decode M == 1. A GEMM kernel here is the single most common
//  mistake in hand-written inference engines: XMX gives you nothing when
//  seven of eight rows of the A fragment are padding, and the layer is
//  bound by weight streaming anyway. At 608 GB/s the entire budget is
//  "read each weight byte once, decode it in registers, never spill".
//
//  Roofline, per decode step, 7B params:
//      mxfp4  ~3.7 GB  ->  6.1 ms  -> 164 tok/s ceiling
//      int4   ~3.6 GB  ->  6.0 ms  -> 167 tok/s
//      int8   ~7.0 GB  -> 11.5 ms  ->  87 tok/s
//      bf16  ~14.0 GB  -> 23.0 ms  ->  43 tok/s
//  Any measured number far under these means the dequant is spilling or
//  the loads are not coalescing, not that the card is slow.
// =====================================================================
#include "kernels.hpp"
#include "int4_smallm.hpp"
#include <algorithm>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemv_step.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <cstdlib>

namespace b70 {
namespace {

// ---------------------------------------------------------------------
// Per-format inner accumulation: lane consumes GEMV_EPL contiguous
// elements of row n starting at k0, multiplies by x, accumulates.
//
// The scale lookup is hoisted out of the element loop because GEMV_EPL
// divides both the MX block (32) and the INT4 group (128).
// ---------------------------------------------------------------------
// One sub-group per output row. Work-group of 8 sub-groups covers 8 rows
// and shares the activation vector through SLM.
// ---------------------------------------------------------------------
// B70_GEMV_CAP: 0/unset = one work-group per row-block (original); N > 0
// caps the launch at N groups, each striding over blocks.
static int gemv_cap() {
    static const int v = []{ const char* e = std::getenv("B70_GEMV_CAP");
        int x = (e && *e) ? std::atoi(e) : 512; return x < 0 ? 0 : x; }();
    return v;
}

// RPS = rows owned by one sub-group.  It sets the work-group count:
// n_groups = N / (WG_SUBGROUPS * RPS).  Bigger RPS means fewer x reloads
// (the hoisted activations are shared by all RPS rows) but fewer
// work-groups, and mid-size shapes run out of work-groups first --
// la_qkv (N=8192) gets 256 groups and 64% of roofline where lm_head
// (N=248320) gets 7760 and reaches 98%.
// MB > 1: mcount (<= MB) activation rows x[m*K..] -> y[m*N..] against ONE
// pass over the weights.  Every row runs exactly the same accumulation
// sequence as MB == 1, so a batched call is bit-identical to mcount single
// calls -- which is what makes a speculative verify batch exact.
template <Fmt F, int EPL_F, int UNROLL, int OPT = 0, int RPS = ROWS_PER_SG, int MB = 1>
sycl::event gemv_impl(sycl::queue& q, const QuantWeight& w,
                      const float* x, float* y,
                      const std::vector<sycl::event>& deps, int mcount = 1) {
    const int N = w.N, K = w.K;
    // Rows per sub-group: the dispatcher's R at MB == 1 (decode is
    // unchanged), 1 when batched so MB activation chunks fit in registers.
    // Which rows a sub-group owns never changes a row's accumulation order.
    constexpr int R = (MB > 1) ? 1 : RPS;
    const int rows_per_wg_sg = WG_SUBGROUPS;
    const int rows_per_wg    = WG_SUBGROUPS * R;
    const int n_blocks       = (N + rows_per_wg - 1) / rows_per_wg;
    // Wave quantization.  One work-group per row-block leaves the group count
    // at whatever N/32 happens to be: ffn gate_up gets 1088 and reaches 86%
    // of roofline, lm_head gets 7760 and reaches 98%.  If the device holds
    // ~C groups at once, 1088 is 2.18 waves and the last runs 18% full.
    // Capping the launch and striding divides the work evenly and amortizes
    // the SLM dequant-table setup over several blocks instead of per block.
    const int cap      = gemv_cap();
    const int n_groups = (cap > 0 && n_blocks > cap) ? cap : n_blocks;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;   // by value: raw pointers only, trivially copyable
        const int mc = mcount;
        sycl::local_accessor<float, 1> lut_slm(256, h);   // FP8 byte -> float
        sycl::local_accessor<float, 1> e8m0_slm(256, h);  // E8M0 byte -> 2^(x-127)
        sycl::local_accessor<float, 1> e2m1_slm(16, h);   // E2M1 nibble -> float

        const int wg_threads = rows_per_wg_sg * SG_SIZE;

        h.parallel_for(
            sycl::nd_range<1>(size_t(n_groups) * size_t(wg_threads), size_t(wg_threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  lid  = int(it.get_local_id(0));
                const int  sgid = int(sg.get_group_id()[0]);

                float* lut = lut_slm.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                float* slut = e8m0_slm.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                float* nlut = e2m1_slm.template
                    get_multi_ptr<sycl::access::decorated::no>().get();

                // MX block scales and E2M1 nibbles get tables too. E8M0
                // decode branches on 0x00 (subnormal) and 0xFF (NaN); the
                // E2M1 magnitude table is a function-local array that the
                // compiler spills to private memory. Both cost more than
                // the loads they decorate: mxfp8 ran 214 GB/s against
                // fp8_e4m3's 320 for identical element bytes, and mxfp4
                // 207 against int4's 353.
                if constexpr (Traits<F>::block == kMXBlock) {
                    for (int b = lid; b < 256; b += wg_threads)
                        slut[b] = e8m0_to_f32(uint8_t(b));
                }
                if constexpr (F == Fmt::MXFP4) {
                    for (int b = lid; b < 16; b += wg_threads)
                        nlut[b] = e2m1_to_f32(uint8_t(b));
                }
                if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::MXFP8) {
                    for (int b = lid; b < 256; b += wg_threads)
                        lut[b] = e4m3_to_f32(uint8_t(b));
                } else if constexpr (F == Fmt::FP8_E5M2) {
                    for (int b = lid; b < 256; b += wg_threads)
                        lut[b] = e5m2_to_f32(uint8_t(b));
                }
                // The table is shared across the whole work-group, so it
                // must be complete before any sub-group reads it.
                if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::FP8_E5M2 ||
                              F == Fmt::MXFP8 || F == Fmt::MXFP4)
                    sycl::group_barrier(it.get_group());

              for (int blk = int(it.get_group(0)); blk < n_blocks;
                   blk += int(it.get_group_range(0))) {
                const int n_base = blk * rows_per_wg + sgid * R;

                // Activations are read straight from global. An earlier
                // version staged them in SLM on the theory that x was
                // costing 1 GB of traffic; that was wrong. x is only
                // K*4 = 64 KB, so it lives in L2 and the repeated reads
                // never touch DRAM. Staging it bought nothing and cost 16
                // barriers per work-group -- it REGRESSED int4 320->242
                // and bf16 307->251 GB/s. Measured, then removed.
                //
                // The real limiter is memory-level parallelism: one
                // dependent 16-byte load per lane per step does not keep
                // enough requests in flight to saturate GDDR6. UNROLL
                // independent K-steps into separate accumulators so the
                // memory system sees several outstanding loads at once.
                constexpr int STEP_F = SG_SIZE * EPL_F;

                // Interleave the four rows owned by this subgroup at every
                // K step.  The former row-outer loop completed an entire row
                // before issuing the first load for the next one, exposing
                // only WG_SUBGROUPS independent DRAM streams.  Keeping one
                // accumulator per row exposes WG_SUBGROUPS*R
                // streams without changing the per-row summation order.
                float part[MB][R][UNROLL];
                #pragma unroll
                for (int m = 0; m < MB; ++m)
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    #pragma unroll
                    for (int u = 0; u < UNROLL; ++u) part[m][r][u] = 0.0f;

                const int span = STEP_F * UNROLL;
                int base = 0;
                if constexpr (OPT != 0 && F == Fmt::MXFP4 && MB > 1) {
                    // Batched rows: decode each weight chunk ONCE, then apply
                    // it to every activation row with run_xv's exact
                    // arithmetic (decode_xv).  Re-running run_xv per row
                    // repeated the two SLM table lookups per weight byte for
                    // every row, and an M=4 verify FFN cost 3.4x a decode.
                    for (; base + span <= K; base += span) {
                        const int k0 = base + lane * EPL_F;
                        #pragma unroll
                        for (int u = 0; u < UNROLL; ++u) {
                            float xa[MB][EPL_F];
                            #pragma unroll
                            for (int m = 0; m < MB; ++m) {
                                if (m >= mc) break;
                                const float* xm = x + int64_t(m) * K + k0 + u * STEP_F;
                                #pragma unroll
                                for (int j = 0; j < EPL_F; ++j) xa[m][j] = xm[j];
                            }
                            #pragma unroll
                            for (int r = 0; r < R; ++r) {
                                const int n = n_base + r;
                                if (n >= N) continue;
                                const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                                float wv[EPL_F]; float sc;
                                GemvStep<F, EPL_F>::template decode_xv<OPT>(
                                    wc, row, slut, nlut, n, k0 + u * STEP_F, wv, sc);
                                #pragma unroll
                                for (int m = 0; m < MB; ++m) {
                                    if (m >= mc) break;
                                    float a = 0.0f;
                                    #pragma unroll
                                    for (int j = 0; j < EPL_F; ++j) a = sycl::fma(wv[j], xa[m][j], a);
                                    part[m][r][u] += a * sc;
                                }
                            }
                        }
                    }
                } else if constexpr (OPT != 0 && F == Fmt::INT4) {
                    // INT4 twin of the hoisted-x path below: x loaded once
                    // per K step for all R rows, and the zero-point's
                    // sum(x) computed once per chunk instead of per row.
                    // Same per-row sequence as run() -- bit-identical.
                    for (; base + span <= K; base += span) {
                        const int k0 = base + lane * EPL_F;
                        #pragma unroll
                        for (int m = 0; m < MB; ++m) {
                        if (m >= mc) break;
                        const float* xm = x + int64_t(m) * K;
                        float xv[UNROLL][EPL_F], ax[UNROLL];
                        #pragma unroll
                        for (int u = 0; u < UNROLL; ++u) {
                            #pragma unroll
                            for (int i = 0; i < EPL_F; ++i)
                                xv[u][i] = xm[k0 + u * STEP_F + i];
                            ax[u] = GemvStep<F, EPL_F>::xsum(&xv[u][0]);
                        }
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            const int n = n_base + r;
                            if (n >= N) continue;
                            const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                            #pragma unroll
                            for (int u = 0; u < UNROLL; ++u)
                                part[m][r][u] += GemvStep<F, EPL_F>::template run_xv<OPT>(
                                    wc, row, &xv[u][0], ax[u], n, k0 + u * STEP_F);
                        }
                        }
                    }
                } else if constexpr (OPT != 0 && F == Fmt::MXFP4) {
                    // OPT bit 0: load this lane's activations ONCE per K step
                    // and reuse them across all R rows.  Without it
                    // the compiler cannot prove x and row do not alias and
                    // reloads x for every row.
                    for (; base + span <= K; base += span) {
                        const int k0 = base + lane * EPL_F;
                        #pragma unroll
                        for (int m = 0; m < MB; ++m) {
                        if (m >= mc) break;
                        const float* xm = x + int64_t(m) * K;
                        float xv[UNROLL][EPL_F];
                        #pragma unroll
                        for (int u = 0; u < UNROLL; ++u)
                            #pragma unroll
                            for (int i = 0; i < EPL_F; ++i)
                                xv[u][i] = xm[k0 + u * STEP_F + i];
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            const int n = n_base + r;
                            if (n >= N) continue;
                            const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                            #pragma unroll
                            for (int u = 0; u < UNROLL; ++u)
                                part[m][r][u] += GemvStep<F, EPL_F>::template run_xv<OPT>(
                                    wc,row,&xv[u][0],slut,nlut,n,k0+u*STEP_F);
                        }
                        }
                    }
                } else
                for (; base + span <= K; base += span) {
                    const int k0 = base + lane * EPL_F;
                    #pragma unroll
                    for (int m = 0; m < MB; ++m) {
                    if (m >= mc) break;
                    const float* xm = x + int64_t(m) * K;
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const int n = n_base + r;
                        if (n >= N) continue;
                        const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                        #pragma unroll
                        for (int u = 0; u < UNROLL; ++u)
                            part[m][r][u] += GemvStep<F, EPL_F>::run(
                                wc,row,xm,lut,slut,nlut,n,k0+u*STEP_F);
                    }
                    }
                }
                for (; base + STEP_F <= K; base += STEP_F) {
                    const int k0=base+lane*EPL_F;
                    #pragma unroll
                    for (int m=0;m<MB;++m) {
                    if (m>=mc) break;
                    const float* xm=x+int64_t(m)*K;
                    #pragma unroll
                    for (int r=0;r<R;++r) {
                        const int n=n_base+r;if(n>=N)continue;
                        const uint8_t* row=wc.payload+int64_t(n)*wc.row_bytes;
                        part[m][r][0]+=GemvStep<F,EPL_F>::run(
                            wc,row,xm,lut,slut,nlut,n,k0);
                    }
                    }
                }

                #pragma unroll
                for (int m = 0; m < MB; ++m) {
                if (m >= mc) break;
                const float* xm = x + int64_t(m) * K;
                float* ym = y + int64_t(m) * N;
                float acc[R];
                #pragma unroll
                for(int r=0;r<R;++r){
                    float sum=0.0f;
                    #pragma unroll
                    for(int u=0;u<UNROLL;++u)sum+=part[m][r][u];
                    acc[r]=sum;
                    const int n=n_base+r;if(n>=N)continue;
                    const int done=(K/STEP_F)*STEP_F;
                    for(int k=done+lane;k<K;k+=SG_SIZE)
                        acc[r]=sycl::fma(wc.at(n,k),xm[k],acc[r]);
                }

                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float total =
                        sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                    const int n = n_base + r;
                    if (lane == 0 && n < N) ym[n] = total;
                }
                }
              }
            });
    });
}


// ---------------------------------------------------------------------
//  Symmetric int4 GEMV -- decode against the SAME weights the W4A8 prefill
//  GEMM uses, so the FFN matrices exist once instead of twice.
//
//  MXFP4 g32 costs 4 + 8/32 = 4.25 bits/weight; symmetric int4 g128 with an
//  f32 scale costs 4 + 32/128 = 4.25.  Identical, so replacing the MXFP4 FFN
//  weights with these is memory-neutral and removes the ~8.5 GB the duplicate
//  copies were costing.
//
//  It should also be cheaper per byte than the MXFP4 GEMV: nibbles are
//  sign-extended arithmetically instead of going through two SLM table
//  lookups (the E2M1 magnitude table and the E8M0 scale table).
//
//  Layout: two nibbles per byte, element 2i in the LOW nibble, signed two's
//  complement -- exactly what cute's int4_t reads, verified bit-exact.
//  One f32 scale per (row, group of 128).  A lane's 16-element chunk never
//  straddles a group boundary, so the scale is looked up once per chunk.
// ---------------------------------------------------------------------
template <int RPS_, int UNROLL>
sycl::event gemv_int4sym_impl(sycl::queue& q, const uint8_t* pack,
                              const float* ws, const float* x, float* y,
                              int N, int K,
                              const std::vector<sycl::event>& deps) {
    constexpr int EPL = 16;
    constexpr int G   = 128;
    const int rows_per_wg = WG_SUBGROUPS * RPS_;
    const int n_blocks    = (N + rows_per_wg - 1) / rows_per_wg;
    const int cap         = gemv_cap();
    const int n_groups    = (cap > 0 && n_blocks > cap) ? cap : n_blocks;
    const int wg_threads  = WG_SUBGROUPS * SG_SIZE;
    const int64_t row_bytes = int64_t(K) / 2;
    const int kg = K / G;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        // Signed-nibble -> float through SLM, not arithmetic sign extension.
        // Measured 2026-08-26: for MXFP4 the ALU decode ran 80.6 ms/token
        // against 35.3 for the SLM table, and the same holds here -- the
        // arithmetic form cost ~0.9 ms/token over the table.
        sycl::local_accessor<float, 1> i4lut(16, h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(n_groups) * size_t(wg_threads),
                              size_t(wg_threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  sgid = int(sg.get_group_id()[0]);
                const int  lid  = int(it.get_local_id(0));
                constexpr int STEP = SG_SIZE * EPL;
                float* lut = i4lut.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                for (int b = lid; b < 16; b += wg_threads)
                    lut[b] = float(int(int8_t(uint8_t(b) << 4)) >> 4);
                sycl::group_barrier(it.get_group());

              for (int blk = int(it.get_group(0)); blk < n_blocks;
                   blk += int(it.get_group_range(0))) {
                const int n_base = blk * rows_per_wg + sgid * RPS_;
                float part[RPS_][UNROLL];
                #pragma unroll
                for (int r = 0; r < RPS_; ++r)
                    #pragma unroll
                    for (int u = 0; u < UNROLL; ++u) part[r][u] = 0.0f;

                const int span = STEP * UNROLL;
                int base = 0;
                for (; base + span <= K; base += span) {
                    const int k0 = base + lane * EPL;
                    // Activations loaded once and shared by all RPS_ rows --
                    // the same fix that took the MXFP4 GEMV from 61% to 86%
                    // of roofline.
                    float xv[UNROLL][EPL];
                    #pragma unroll
                    for (int u = 0; u < UNROLL; ++u)
                        #pragma unroll
                        for (int i = 0; i < EPL; ++i)
                            xv[u][i] = x[k0 + u * STEP + i];
                    #pragma unroll
                    for (int r = 0; r < RPS_; ++r) {
                        const int n = n_base + r;
                        if (n >= N) continue;
                        const uint8_t* row = pack + int64_t(n) * row_bytes;
                        #pragma unroll
                        for (int u = 0; u < UNROLL; ++u) {
                            const int kk = k0 + u * STEP;
                            const uint64_t packed =
                                *reinterpret_cast<const uint64_t*>(row + (kk >> 1));
                            float a = 0.0f;
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const uint8_t byte = uint8_t(packed >> (8 * i));
                                a = sycl::fma(lut[byte & 0x0F], xv[u][2 * i],     a);
                                a = sycl::fma(lut[byte >> 4],   xv[u][2 * i + 1], a);
                            }
                            part[r][u] = sycl::fma(a, ws[int64_t(n) * kg + kk / G],
                                                   part[r][u]);
                        }
                    }
                }
                for (; base + STEP <= K; base += STEP) {
                    const int k0 = base + lane * EPL;
                    #pragma unroll
                    for (int r = 0; r < RPS_; ++r) {
                        const int n = n_base + r;
                        if (n >= N) continue;
                        const uint8_t* row = pack + int64_t(n) * row_bytes;
                        const uint64_t packed =
                            *reinterpret_cast<const uint64_t*>(row + (k0 >> 1));
                        float a = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const uint8_t byte = uint8_t(packed >> (8 * i));
                            a = sycl::fma(lut[byte & 0x0F], x[k0 + 2 * i],     a);
                            a = sycl::fma(lut[byte >> 4],   x[k0 + 2 * i + 1], a);
                        }
                        part[r][0] = sycl::fma(a, ws[int64_t(n) * kg + k0 / G],
                                               part[r][0]);
                    }
                }

                #pragma unroll
                for (int r = 0; r < RPS_; ++r) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int u = 0; u < UNROLL; ++u) sum += part[r][u];
                    const float total =
                        sycl::reduce_over_group(sg, sum, sycl::plus<float>());
                    const int n = n_base + r;
                    if (lane == 0 && n < N) y[n] = total;
                }
              }
            });
    });
}

// Small-N symmetric-int4 variant.  The ordinary kernel assigns one subgroup
// to each output row, so N=1024/2048 launches only 8/16 workgroups with RPS=4.
// Split K across all eight subgroups instead: one workgroup per output row,
// enough independent groups to fill the B70 on routers, DeltaNet projections,
// and Ornith's shared expert.  Keep it opt-in until full-model parity and the
// crossover are measured.
sycl::event gemv_int4sym_wide(sycl::queue& q, const uint8_t* pack,
                              const float* ws, const float* x, float* y,
                              int N, int K,
                              const std::vector<sycl::event>& deps) {
    constexpr int EPL = 16;
    constexpr int G = 128;
    const int wg_threads = WG_SUBGROUPS * SG_SIZE;
    const int64_t row_bytes = int64_t(K) / 2;
    const int kg = K / G;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> part(WG_SUBGROUPS, h);
        sycl::local_accessor<float, 1> i4lut(16, h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(N) * wg_threads, wg_threads),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int sgid = int(sg.get_group_id()[0]);
                const int lid = int(it.get_local_id(0));
                float* lut = i4lut.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                for (int b = lid; b < 16; b += wg_threads)
                    lut[b] = float(int(int8_t(uint8_t(b) << 4)) >> 4);
                sycl::group_barrier(it.get_group());

                const int n = int(it.get_group(0));
                const int per_sg = K / WG_SUBGROUPS;
                const int k_beg = sgid * per_sg;
                const int k_end = k_beg + per_sg;
                const uint8_t* row = pack + int64_t(n) * row_bytes;
                float acc = 0.0f;
                for (int k = k_beg + lane * EPL; k + EPL <= k_end;
                     k += SG_SIZE * EPL) {
                    const uint64_t packed =
                        *reinterpret_cast<const uint64_t*>(row + (k >> 1));
                    float a = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const uint8_t byte = uint8_t(packed >> (8 * i));
                        a = sycl::fma(lut[byte & 0x0f], x[k + 2 * i], a);
                        a = sycl::fma(lut[byte >> 4], x[k + 2 * i + 1], a);
                    }
                    acc = sycl::fma(a, ws[int64_t(n) * kg + k / G], acc);
                }
                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                if (lane == 0) part[sgid] = acc;
                sycl::group_barrier(it.get_group());
                if (lid == 0) {
                    float sum = 0.0f;
                    for (int i = 0; i < WG_SUBGROUPS; ++i) sum += part[i];
                    y[n] = sum;
                }
            });
    });
}

// ---------------------------------------------------------------------
// Small-N variant: one work-group per output row, K split across its
// sub-groups, reduced through SLM.
//
// The main kernel maps one sub-group to a row and 32 rows to a work-group.
// That is right when N is large, and starvation when it is not: the
// deltanet a/b projection (N=64) fills TWO work-groups on a 256-EU card
// and measured 5.6 GB/s, the router (N=256) eight groups at 38 GB/s. Same
// GemvStep, so dequant is bit-identical; only the summation order and the
// launch geometry differ.
//
// Measured, int4 K=2048, us (main -> wide):
//     N=64  11.79 -> 3.54    N=1024  7.09 -> 4.31    N=4096  9.84 -> 13.02
//     N=256  6.87 -> 3.59    N=2048  7.88 -> 7.08    N=8192 16.46 -> 23.70
// so it wins up to N=2048 and loses above it, where the row-per-sub-group
// mapping already has all the parallelism it needs.
// ---------------------------------------------------------------------
template <Fmt F, int EPL_F>
sycl::event gemv_wide(sycl::queue& q, const QuantWeight& w,
                      const float* x, float* y,
                      const std::vector<sycl::event>& deps) {
    const int N = w.N, K = w.K;
    const int wg_threads = WG_SUBGROUPS * SG_SIZE;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        sycl::local_accessor<float, 1> part(WG_SUBGROUPS, h);
        sycl::local_accessor<float, 1> lut_slm(256, h);
        sycl::local_accessor<float, 1> e8m0_slm(256, h);
        sycl::local_accessor<float, 1> e2m1_slm(16, h);

        h.parallel_for(
            sycl::nd_range<1>(size_t(N) * wg_threads, wg_threads),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  sgid = int(sg.get_group_id()[0]);
                const int  lid  = int(it.get_local_id(0));

                float* lut = lut_slm.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                float* slut = e8m0_slm.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                float* nlut = e2m1_slm.template
                    get_multi_ptr<sycl::access::decorated::no>().get();

                if constexpr (Traits<F>::block == kMXBlock) {
                    for (int b = lid; b < 256; b += wg_threads)
                        slut[b] = e8m0_to_f32(uint8_t(b));
                }
                if constexpr (F == Fmt::MXFP4) {
                    for (int b = lid; b < 16; b += wg_threads)
                        nlut[b] = e2m1_to_f32(uint8_t(b));
                }
                if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::MXFP8) {
                    for (int b = lid; b < 256; b += wg_threads)
                        lut[b] = e4m3_to_f32(uint8_t(b));
                } else if constexpr (F == Fmt::FP8_E5M2) {
                    for (int b = lid; b < 256; b += wg_threads)
                        lut[b] = e5m2_to_f32(uint8_t(b));
                }
                sycl::group_barrier(it.get_group());

                // Every slice is a whole number of scale blocks AND of
                // lane-steps; the dispatch below refuses the shape
                // otherwise, so no tail handling is needed here.  With
                // scale blocks the split is block-granular: sub-group s
                // takes blocks [s*nb/8, (s+1)*nb/8).  That is exactly K/8
                // per sub-group whenever K/8 is whole blocks, and it also
                // covers K = 6656 (Muse: 52 INT4 groups), where the even
                // split was refused and k/v (N=256) fell to the row kernel
                // -- 8 work-groups, 158 us for 1.8 MB.
                const int n      = int(it.get_group(0));
                constexpr int BLK = Traits<F>::block > 0 ? Traits<F>::block : 1;
                int k_beg, k_end;
                if constexpr (BLK > 1) {
                    const int nb = K / BLK;
                    k_beg = (sgid * nb / WG_SUBGROUPS) * BLK;
                    k_end = ((sgid + 1) * nb / WG_SUBGROUPS) * BLK;
                } else {
                    const int per_sg = K / WG_SUBGROUPS;
                    k_beg = sgid * per_sg;
                    k_end = k_beg + per_sg;
                }

                const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                float acc = 0.0f;
                for (int k = k_beg + lane * EPL_F; k + EPL_F <= k_end;
                     k += SG_SIZE * EPL_F)
                    acc += GemvStep<F, EPL_F>::run(wc, row, x, lut, slut, nlut, n, k);

                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                if (lane == 0) part[sgid] = acc;
                sycl::group_barrier(it.get_group());
                if (lid == 0) {
                    float s = 0.0f;      // fixed order -> run-to-run identical
                    for (int i = 0; i < WG_SUBGROUPS; ++i) s += part[i];
                    y[n] = s;
                }
            });
    });
}
} // namespace

namespace {
int g_tune_epl = [] {
        const char* e = std::getenv("B70_EPL");
        return e ? std::atoi(e) : 16;
    }();
int g_tune_unroll = [] {
        // 0 means "use GemvGeom<F>::UNROLL_DEFAULT", the per-format value.
        // This used to default to 1, which silently overrode every format's
        // tuned unroll. On MXFP4 (UNROLL_DEFAULT 2) that cost 2.5% of decode:
        // 43.66 vs 42.68 ms/token on the 32-token full-model sweep.
        const char* e = std::getenv("B70_UNROLL");
        return e ? std::atoi(e) : 0;
    }();
int g_tune_wide = [] {
        const char* e = std::getenv("B70_WIDE");
        return e ? std::atoi(e) : -1;
    }();
}

int gemv_epl_override() { return g_tune_epl; }
int gemv_unroll_override() { return g_tune_unroll; }
void set_gemv_tuning(int epl, int unroll, int wide) {
    g_tune_epl = epl;
    g_tune_unroll = unroll;
    g_tune_wide = wide;
}

namespace {
// Dispatch to a compiled <EPL, UNROLL> variant. Only combinations that
// respect the format's scale-block bound are instantiated.
// N at or below this uses the work-group-per-row kernel; above it the
// row-per-sub-group mapping is already saturated and wins. Measured
// crossover on a B70 is N=2048-4096.
constexpr int kWideMaxN = 2048;

int gemv_wide_override() {
    return g_tune_wide;      // -1 = auto
}

// B70_RPS: 0/unset = the auto rule, otherwise force 1/2/4/8.
int rps_for(int N) {
    static const int forced = []{ const char* e = std::getenv("B70_RPS");
        int x = (e && *e) ? std::atoi(e) : 0;
        return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 0; }();
    if (forced) return forced;
    // Measured 2026-08-26: RPS=2 helps ONLY la_qkv (71.8 -> 68.8 us) and
    // q gemv (74.7 -> 69.9); it costs ffn down 93.8 -> 120.3, out 39.4 ->
    // 48.0, z 35.8 -> 42.6 and lm_head 1149 -> 1288, because halving the
    // rows per sub-group doubles how often the hoisted x is re-read.
    // A blanket N>=16384 rule measured 34.36 ms/token against 31.77.
    // Measured per-shape 2026-08-26 (model dims: la_qkv N=10240, q+gate
    // N=12288, z N=6144, down/out N=5120, gate_up N=34816, lm_head N=248320).
    // RPS=2 helps ONLY the 10240/12288 pair -- la_qkv 71.8 -> 68.8 us and
    // q gemv 74.7 -> 69.9 -- and costs down 93.8 -> 120.3, out 39.4 -> 48.0,
    // z 35.8 -> 42.6, lm_head 1149 -> 1288, because halving the rows per
    // sub-group doubles how often the hoisted x is re-read.  A blanket
    // N>=16384 rule measured 34.36 ms/token against 31.77.
    // N == 8192 (Ornith la_qkv / q+gate) is faster at 4: MEASURED 2026-09-30,
    // la_qkv 27.5 -> 24.8 us, q 24.8 -> 22.2 us; the 2 rule was tuned on
    // Qwen3.8-27B's 10240 / 12288.
    return (N > 8192 && N < 16384) ? 2 : 4;
}

bool wide_relaxed() {
    static const bool v = []{ const char* e = std::getenv("B70_WIDE_RELAX");
        return !(e && *e && std::atoi(e) == 0); }();
    return v;
}

int gemv_opt_override() {
    static const int v = []{ const char* e = std::getenv("B70_GEMV_OPT");
        int x = (e && *e) ? std::atoi(e) : 1; return (x < 0 || x > 3) ? 1 : x; }();
    return v;
}


// ---------------------------------------------------------------------
// ESIMD decode GEMV for MXFP4 (M = 1), the llm-scaler shape: a work-group of
// KS threads owns R output rows, each thread streams its K slice of those
// rows with contiguous 64-byte block loads, the activation comes from L1
// (no SLM staging), E2M1 decodes through two 1 KB SLM tables (byte -> value of
// the low / high nibble) and the E8M0 scale applies once per 32-element block.
// MEASURED IN-MODEL (GRIMOIRE_TIMELINE, Ornith decode, weights from DRAM):
//   N=2048 K=4096 (out / o)  R4 KS4  16.0 / 16.4 us  (SIMT gemv_wide 20.0 / 20.2)
//   N=1024 K=2048 (k+v)      R4 KS4   9.1 us         (10.7)
//   N=4096 K=2048 (z)        R8 KS2  21.1 us         (14.5 -- NOT used)
// tools/esgemv_probe.cpp re-runs the same weights back to back, so its
// numbers are L2-hot (13.1 / 5.1 / 13.4 us) -- do not trust them alone.
// The sum is in a different order than GemvStep's, so it is NOT
// bit-identical to the SIMT GEMV; exact speculative verify (which batches
// rows through the SIMT kernel) keeps using that one only for M > 1, and
// decode goes through here for these shapes.  GRIMOIRE_ESGEMV=0 = off.
// ---------------------------------------------------------------------
namespace {
const float* esgemv_lut(sycl::queue& q) {
    static float* lut = nullptr;
    if (!lut) {
        lut = sycl::malloc_device<float>(512, q);
        float* L = lut;                          // kernels cannot touch a static
        // built by a kernel: no host buffer, no wait -- safe under graph capture
        q.parallel_for(sycl::range<1>(512), [=](sycl::id<1> i) {
            const int b = int(i[0]) & 255, nib = int(i[0]) < 256 ? (b & 15) : (b >> 4);
            const int m = nib & 7;
            const float v = m < 2 ? 0.5f * float(m) : float(2 + (m & 1)) * float(1 << (m >> 1)) * 0.25f;
            L[i[0]] = (nib & 8) ? -v : v;
        });
    }
    return lut;
}

template <int R, int KS>
sycl::event esgemv_mxfp4(sycl::queue& q, const QuantWeight& w, const float* x, float* y,
                         const std::vector<sycl::event>& deps) {
    namespace es = sycl::ext::intel::esimd;
    const float* lut = esgemv_lut(q);
    const int N = w.N, K = w.K;
    const uint8_t* pay = w.payload;
    const uint8_t* scl = static_cast<const uint8_t*>(w.scales);
    const size_t rowB = w.row_bytes, rowS = w.row_scales;
    const int groups = (N + R - 1) / R;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<2048 + KS * R * 4>();
            const int t = int(it.get_local_id(0));
            for (int i = t; i < 16; i += KS)
                es::slm_block_store<float, 32>(i * 128, es::block_load<float, 32>(lut + i * 32));
            es::barrier();
            const int n0 = int(it.get_group(0)) * R;
            const int kp = K / KS, kb = t * kp;
            es::simd<float, 16> tot[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) tot[r] = 0.0f;
            for (int k = kb; k < kb + kp; k += 128) {
                es::simd<float, 128> xv = es::block_load<float, 128>(x + k);
                es::simd<uint8_t, 64> pb[R];
                es::simd<uint32_t, 4> sb[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int n = n0 + r < N ? n0 + r : N - 1;
                    pb[r] = es::block_load<uint8_t, 64>(pay + size_t(n) * rowB + k / 2);
                    es::simd<uint8_t, 4> s4 = es::block_load<uint8_t, 4>(scl + size_t(n) * rowS + k / 32);
                    sb[r] = es::convert<uint32_t>(s4);
                }
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    es::simd<float, 32> x32 = xv.template select<32, 1>(32 * b);
                    es::simd<float, 16> xe = x32.template select<16, 2>(0);
                    es::simd<float, 16> xo = x32.template select<16, 2>(1);
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        es::simd<uint8_t, 16> by = pb[r].template select<16, 1>(16 * b);
                        es::simd<uint32_t, 16> addr = es::convert<uint32_t>(by) << 2;
                        es::simd<float, 16> wl = es::slm_gather<float, 16>(addr);
                        es::simd<float, 16> wh = es::slm_gather<float, 16>(addr + 1024u);
                        es::simd<float, 16> part = wl * xe + wh * xo;
                        const uint32_t e = sb[r][b];
                        const float sc = sycl::bit_cast<float>(e ? e << 23 : 0x00400000u);
                        tot[r] += part * sc;
                    }
                }
            }
            float res[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) res[r] = es::reduce<float>(tot[r], std::plus<>());
            if constexpr (KS == 1) {
                #pragma unroll
                for (int r = 0; r < R; ++r) if (n0 + r < N) y[n0 + r] = res[r];
            } else {
                es::simd<float, R> rv;
                #pragma unroll
                for (int r = 0; r < R; ++r) rv[r] = res[r];
                es::slm_block_store<float, R>(2048 + t * R * 4, rv);
                es::barrier();
                if (t == 0) {
                    es::simd<float, R> acc = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < KS; ++j) acc += es::slm_block_load<float, R>(2048 + j * R * 4);
                    #pragma unroll
                    for (int r = 0; r < R; ++r) if (n0 + r < N) y[n0 + r] = acc[r];
                }
            }
        });
    });
}

// The measured winners only; everything else stays on the SIMT kernels.

// ---------------------------------------------------------------------
// ESIMD decode GEMV v2 for MXFP4 (M = 1): the MoE decode kernels' design
// (moe_kernels.cpp) on a plain matrix.  A work-group of K/512 threads owns 2
// output rows; each thread keeps its 512-element activation slice in
// registers split even / odd K, streams its slice of both rows with 64-byte
// block loads and decodes E2M1 in the ALU (nibble bits as an fp16 value times
// 2^-14, the 2^14 folded into the E8M0 scale) -- no SLM tables, which is what
// held v1 above at ~270 GB/s.  MEASURED 2026-10-01, tools/decode_probe.cpp,
// cold weights (rotating copies), engine path -> v2:
//   N=8192 K=2048 (la_qkv / q)  24.35 -> 17.35 us  (514 GB/s)
//   N=4096 K=2048 (z)           12.80 ->  9.93 us
//   N=2048 K=4096 (out / o)     16.41 -> 10.36 us  (v1 R4 KS4 before)
//   N=1024 K=2048 (k+v)          6.21 ->  3.91 us
// max relative error 1.5e-7 against the SIMT GEMV.  Not bit-identical to it
// (summation order).  GRIMOIRE_ESGEMV2=0 = v1 / SIMT as before.
// ---------------------------------------------------------------------
namespace es = sycl::ext::intel::esimd;
template <int N_>
SYCL_ESIMD_FUNCTION inline void e2m1_split_v2(es::simd<uint8_t, N_> b, es::simd<float, N_>& lo,
                                              es::simd<float, N_>& hi) {
    es::simd<uint16_t, N_> u = b;
    es::simd<uint16_t, N_> l = ((u & 0x7) << 9) | ((u & 0x8) << 12);
    es::simd<uint16_t, N_> hb = ((u & 0x70) << 5) | ((u & 0x80) << 8);
    es::simd<sycl::half, N_> lh = l.template bit_cast_view<sycl::half>();
    es::simd<sycl::half, N_> hh = hb.template bit_cast_view<sycl::half>();
    lo = lh;
    hi = hh;
}
template <int KP>
SYCL_ESIMD_FUNCTION inline void mx4_step_v2(const uint8_t* prow, const uint8_t* srow, int st,
                                            es::simd<float, KP / 2>& xe, es::simd<float, KP / 2>& xo,
                                            es::simd<float, 16>& acc) {
    es::simd<uint8_t, 64> pb = es::block_load<uint8_t, 64>(prow + st * 64);
    es::simd<uint8_t, 4> sb = es::block_load<uint8_t, 4>(srow + st * 4);
    es::simd<float, 64> wl, wh;
    e2m1_split_v2<64>(pb, wl, wh);
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        es::simd<float, 16> part =
            wl.template select<16, 1>(16 * b) * xe.template select<16, 1>(st * 64 + 16 * b) +
            wh.template select<16, 1>(16 * b) * xo.template select<16, 1>(st * 64 + 16 * b);
        acc += part * sycl::bit_cast<float>((uint32_t(sb[b]) + 14u) << 23);
    }
}
constexpr int kEs2KP = 512, kEs2MaxKS = 32;
template <int R>
sycl::event esgemv2_mxfp4(sycl::queue& q, const QuantWeight& w, const float* x, float* y,
                          const std::vector<sycl::event>& deps) {
    constexpr int KP = kEs2KP;
    constexpr int RP = R < 4 ? 4 : R;
    const int KS = w.K / KP;
    const uint8_t* pay = w.payload; const uint8_t* scl = static_cast<const uint8_t*>(w.scales);
    const int64_t rb = w.row_bytes, rs = w.row_scales;
    const int n_wg = w.N / R;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(n_wg) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<kEs2MaxKS * RP * 4>();
            const int t = int(it.get_local_id(0));
            const int64_t row0 = int64_t(it.get_group(0)) * R;
            const int kb = t * KP;
            es::simd<float, KP> xs = es::block_load<float, KP>(x + kb);
            es::simd<float, KP / 2> xe = xs.template select<KP / 2, 2>(0);
            es::simd<float, KP / 2> xo = xs.template select<KP / 2, 2>(1);
            es::simd<float, 16> acc[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0f;
            #pragma unroll 1
            for (int st = 0; st < KP / 128; ++st) {
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    mx4_step_v2<KP>(pay + (row0 + r) * rb + kb / 2, scl + (row0 + r) * rs + kb / 32,
                                    st, xe, xo, acc[r]);
            }
            es::simd<float, RP> red = 0.0f;
            #pragma unroll
            for (int r = 0; r < R; ++r) red[r] = es::reduce<float>(acc[r], std::plus<>());
            if (KS > 1) {
                es::slm_block_store<float, RP>(t * RP * 4, red);
                es::barrier();
                if (t != 0) return;
                red = 0.0f;
                for (int j = 0; j < KS; ++j) red += es::slm_block_load<float, RP>(j * RP * 4);
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) y[row0 + r] = red[r];
        });
    });
}


// BF16 decode GEMV for small matrices (the MoE router, the DeltaNet a/b
// projection): the v2 geometry -- K/512 threads, 2 rows, activation from L1,
// weights as 256-byte block loads widened to fp32 by a shift.  The SIMT path
// ran the 256x2048 router in 5.3 us per layer in-model.
// GRIMOIRE_ESGEMV2=0 also turns this off.
template <int R>
sycl::event esgemv2_bf16(sycl::queue& q, const QuantWeight& w, const float* x, float* y,
                         const std::vector<sycl::event>& deps) {
    constexpr int KP = kEs2KP;
    constexpr int RP = R < 4 ? 4 : R;
    const int K = w.K, KS = w.K / KP;
    const uint16_t* wp = reinterpret_cast<const uint16_t*>(w.payload);
    const int64_t rstride = w.row_bytes / 2;
    const int n_wg = w.N / R;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(n_wg) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<kEs2MaxKS * RP * 4>();
            const int t = int(it.get_local_id(0));
            const int64_t row0 = int64_t(it.get_group(0)) * R;
            const int kb = t * KP;
            es::simd<float, 16> acc[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0f;
            #pragma unroll
            for (int c = 0; c < KP; c += 128) {
                es::simd<float, 128> xv = es::block_load<float, 128>(x + kb + c);
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    es::simd<uint16_t, 128> wb = es::block_load<uint16_t, 128>(wp + (row0 + r) * rstride + kb + c);
                    es::simd<uint32_t, 128> wu = es::convert<uint32_t>(wb) << 16;
                    es::simd<float, 128> wf = wu.template bit_cast_view<float>();
                    es::simd<float, 128> pr = wf * xv;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[r] += pr.template select<16, 1>(16 * j);
                }
            }
            es::simd<float, RP> red = 0.0f;
            #pragma unroll
            for (int r = 0; r < R; ++r) red[r] = es::reduce<float>(acc[r], std::plus<>());
            if (KS > 1) {
                es::slm_block_store<float, RP>(t * RP * 4, red);
                es::barrier();
                if (t != 0) return;
                red = 0.0f;
                for (int j = 0; j < KS; ++j) red += es::slm_block_load<float, RP>(j * RP * 4);
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) y[row0 + r] = red[r];
        });
    });
}
bool esgemv2_bf16_try(sycl::queue& q, const QuantWeight& w, const float* x, float* y,
                      const std::vector<sycl::event>& deps, sycl::event& out) {
    static const bool v2 = [] { const char* e = std::getenv("GRIMOIRE_ESGEMV2");
        return !(e && *e == '0'); }();
    if (!v2 || w.fmt != Fmt::BF16 || !w.payload || w.K % kEs2KP != 0 || w.K / kEs2KP > kEs2MaxKS ||
        w.N % 2 != 0 || w.N > 4096 || w.row_bytes % 4 != 0 || w.row_bytes < int64_t(w.K) * 2)
        return false;
    out = esgemv2_bf16<2>(q, w, x, y, deps);
    return true;
}

// ESIMD decode GEMV for INT4 (M = 1): esgemv2_mxfp4's geometry -- K/512
// threads, 2 rows, the activation slice in registers split even / odd K --
// on GRIMOIRE's INT4: asymmetric, a bf16 scale and a u8 zero per GS = 64 or
// 128 elements, zero 0xff = signed s4 (GPTQ experts XORed with 0x88 at load).
// Every load of the slice is issued before any arithmetic (256 payload bytes
// and the scale / zero vectors per row).  (q - z) is exact in fp16: 1024 + q
// and 1024 + z are fp16 integers, so their difference is too.  The SIMT
// kernel streamed Qwen3.8-27B-W4A16 at ~280 GB/s (19.9 tok/s against 33.1
// for its MXFP4 twin).  Summation order differs from the SIMT kernel's (not
// bit-identical).  GRIMOIRE_ESGEMV_I4=0 = the SIMT kernel.
constexpr int kEs4MaxKS = 64;
template <int R, int GS>
sycl::event esgemv2_int4(sycl::queue& q, const QuantWeight& w, const float* x, float* y,
                         const std::vector<sycl::event>& deps) {
    constexpr int KP = kEs2KP;
    constexpr int RP = R < 4 ? 4 : R;
    constexpr int NG = KP / GS;           // groups per slice
    constexpr int BPG = GS / 2;           // payload bytes per group
    const int KS = w.K / KP;
    const uint8_t* pay = w.payload;
    const uint16_t* scl = static_cast<const uint16_t*>(w.scales);
    const uint8_t* zer = w.zeros;
    const int64_t rb = w.row_bytes, rs = w.row_scales;
    const int n_wg = w.N / R;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(n_wg) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<kEs4MaxKS * RP * 4>();
            const int t = int(it.get_local_id(0));
            const int64_t row0 = int64_t(it.get_group(0)) * R;
            const int kb = t * KP;
            es::simd<uint8_t, KP / 2> pb[R];
            es::simd<uint16_t, NG> sv[R];
            es::simd<uint8_t, NG> zv[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                pb[r] = es::block_load<uint8_t, KP / 2>(pay + (row0 + r) * rb + kb / 2);
                sv[r] = es::block_load<uint16_t, NG>(scl + (row0 + r) * rs + kb / GS);
                zv[r] = es::block_load<uint8_t, NG>(zer + (row0 + r) * rs + kb / GS);
            }
            es::simd<float, KP> xs = es::block_load<float, KP>(x + kb);
            es::simd<float, KP / 2> xe = xs.template select<KP / 2, 2>(0);
            es::simd<float, KP / 2> xo = xs.template select<KP / 2, 2>(1);
            es::simd<float, RP> red = 0.0f;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                es::simd<float, 16> acc = 0.0f;
                #pragma unroll
                for (int g = 0; g < NG; ++g) {
                    const uint32_t zu = uint32_t(zv[r][g]);
                    const uint32_t sg = (zu + 1u) >> 8;            // zero 0xff = signed s4
                    const uint16_t xm = uint16_t(sg * 0x88u);
                    const sycl::half zh = sycl::bit_cast<sycl::half>(uint16_t(0x6400u | (zu - sg * 247u)));
                    const float s = sycl::bit_cast<float>(uint32_t(sv[r][g]) << 16);
                    es::simd<uint16_t, BPG> u = pb[r].template select<BPG, 1>(g * BPG);
                    u ^= xm;
                    es::simd<uint16_t, BPG> lb = (u & 0xF) | 0x6400;
                    es::simd<uint16_t, BPG> hb = (u >> 4) | 0x6400;
                    es::simd<sycl::half, BPG> lh = lb.template bit_cast_view<sycl::half>().read() - zh;
                    es::simd<sycl::half, BPG> hh = hb.template bit_cast_view<sycl::half>().read() - zh;
                    es::simd<float, BPG> wl = lh, wh = hh;
                    es::simd<float, 16> part = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < BPG / 16; ++b)
                        part += wl.template select<16, 1>(16 * b) * xe.template select<16, 1>(g * BPG + 16 * b) +
                                wh.template select<16, 1>(16 * b) * xo.template select<16, 1>(g * BPG + 16 * b);
                    acc += part * s;
                }
                red[r] = es::reduce<float>(acc, std::plus<>());
            }
            if (KS > 1) {
                es::slm_block_store<float, RP>(t * RP * 4, red);
                es::barrier();
                if (t != 0) return;
                red = 0.0f;
                for (int j = 0; j < KS; ++j) red += es::slm_block_load<float, RP>(j * RP * 4);
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) y[row0 + r] = red[r];
        });
    });
}
// ESIMD decode GEMV for FP8 E4M3 (M = 1) with GRIMOIRE's per-output-channel
// fp32 scale: esgemv2_int4's geometry -- R rows per work-group, KP weights
// (= KP bytes) of each row per thread, every load issued before any
// arithmetic.  E4M3 -> fp16 is exact by moving the bits, (b & 0x7F) << 7 |
// (b & 0x80) << 8, subnormals included; the exponent biases differ by 8, so
// the 256x is folded into the row scale.  The SIMT kernel streamed
// Qwen3.8-27B-FP8 at ~250 GB/s: 9.4 tok/s on two B70s (2026-10-06).  Not
// bit-identical to it (summation order).  GRIMOIRE_ESGEMV_FP8=0 = SIMT.
template <int R, int KP>
sycl::event esgemv2_fp8(sycl::queue& q, const QuantWeight& w, const float* x, float* y,
                        const std::vector<sycl::event>& deps) {
    constexpr int RP = R < 4 ? 4 : R;
    const int KS = w.K / KP;
    const uint8_t* pay = w.payload;
    const float* scl = static_cast<const float*>(w.scales);
    const int64_t rb = w.row_bytes;
    const int n_wg = w.N / R;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(n_wg) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<kEs4MaxKS * RP * 4>();
            const int t = int(it.get_local_id(0));
            const int64_t row0 = int64_t(it.get_group(0)) * R;
            const int kb = t * KP;
            es::simd<uint8_t, KP> pb[R];
            #pragma unroll
            for (int r = 0; r < R; ++r)
                pb[r] = es::block_load<uint8_t, KP>(pay + (row0 + r) * rb + kb);
            es::simd<float, KP> xs = es::block_load<float, KP>(x + kb);
            es::simd<float, RP> red = 0.0f;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                es::simd<float, 16> acc = 0.0f;
                #pragma unroll
                for (int c = 0; c < KP / 64; ++c) {
                    es::simd<uint16_t, 64> u = pb[r].template select<64, 1>(c * 64);
                    es::simd<uint16_t, 64> hb = ((u & 0x7F) << 7) | ((u & 0x80) << 8);
                    es::simd<float, 64> wf = hb.template bit_cast_view<sycl::half>().read();
                    #pragma unroll
                    for (int b = 0; b < 4; ++b)
                        acc += wf.template select<16, 1>(16 * b) * xs.template select<16, 1>(c * 64 + 16 * b);
                }
                red[r] = es::reduce<float>(acc, std::plus<>());
            }
            if (KS > 1) {
                es::slm_block_store<float, RP>(t * RP * 4, red);
                es::barrier();
                if (t != 0) return;
                red = 0.0f;
                for (int j = 0; j < KS; ++j) red += es::slm_block_load<float, RP>(j * RP * 4);
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) y[row0 + r] = red[r] * (scl[row0 + r] * 256.0f);
        });
    });
}

bool esgemv_fp8_try(sycl::queue& q, const QuantWeight& w, const float* x, float* y,
                    const std::vector<sycl::event>& deps, sycl::event& out) {
    static const bool on = [] { const char* e = std::getenv("GRIMOIRE_ESGEMV_FP8");
        return !(e && *e == '0'); }();
    if (!on || w.fmt != Fmt::FP8_E4M3 || !w.payload || !w.scales || w.N % 2 != 0 ||
        w.row_bytes < int64_t(w.K) || w.row_bytes % 64 != 0)
        return false;
    const int K = w.K;
    if (K % 256 == 0 && K / 256 <= kEs4MaxKS) { out = esgemv2_fp8<2, 256>(q, w, x, y, deps); return true; }
    if (K % 512 == 0 && K / 512 <= kEs4MaxKS) { out = esgemv2_fp8<2, 512>(q, w, x, y, deps); return true; }
    return false;
}

bool esgemv_i4_try(sycl::queue& q, const QuantWeight& w, const float* x, float* y,
                   const std::vector<sycl::event>& deps, sycl::event& out) {
    static const bool on = [] { const char* e = std::getenv("GRIMOIRE_ESGEMV_I4");
        return !(e && *e == '0'); }();
    if (!on || w.fmt != Fmt::INT4 || !w.payload || !w.scales || !w.zeros) return false;
    const int K = w.K, gs = 1 << w.int4_gshift();
    if ((gs != 64 && gs != 128) || int64_t(w.row_scales) * gs != K) return false;
    if (K % kEs2KP != 0 || K / kEs2KP > kEs4MaxKS || w.N % 2 != 0 ||
        w.row_bytes < int64_t(K / 2) || w.row_bytes % 64 != 0 || w.row_scales % (kEs2KP / gs) != 0)
        return false;
    out = gs == 64 ? esgemv2_int4<2, 64>(q, w, x, y, deps) : esgemv2_int4<2, 128>(q, w, x, y, deps);
    return true;
}

bool esgemv_try(sycl::queue& q, const QuantWeight& w, const float* x, float* y,
                const std::vector<sycl::event>& deps, sycl::event& out) {
    static const bool on = [] { const char* e = std::getenv("GRIMOIRE_ESGEMV");
        return !(e && *e == '0'); }();
    const int N = w.N, K = w.K;
    if (!on || w.fmt != Fmt::MXFP4 || !w.payload || !w.scales || K % 128 != 0 ||
        w.row_bytes < size_t(K / 2) || w.row_scales < size_t(K / 32) || (w.row_bytes % 64) != 0)
        return false;
    static const bool v2 = [] { const char* e = std::getenv("GRIMOIRE_ESGEMV2");
        return !(e && *e == '0'); }();
    // v2 for every K that is whole 512-element slices, up to 32 of them, and
    // N up to 64K rows (lm_head, 248K rows, stays on the SIMT kernel that
    // already streams it at ~590 GB/s).
    if (v2 && K % kEs2KP == 0 && K / kEs2KP <= kEs2MaxKS && N % 2 == 0 && N <= 65536 &&
        w.row_scales % 4 == 0) {
        out = esgemv2_mxfp4<2>(q, w, x, y, deps);
        return true;
    }
    if (N <= 2048 && K == 4096 && N % 4 == 0) { out = esgemv_mxfp4<4, 4>(q, w, x, y, deps); return true; }
    if (N <= 1024 && K == 2048 && N % 4 == 0) { out = esgemv_mxfp4<4, 4>(q, w, x, y, deps); return true; }
    return false;
}
} // namespace

template <Fmt F, int MB = 1>
sycl::event dispatch(sycl::queue& q, const QuantWeight& w, const float* x,
                     float* y, const std::vector<sycl::event>& deps, int mc = 1) {
    if constexpr (F == Fmt::MXFP4 && MB == 1) {
        sycl::event ev;
        if (esgemv_try(q, w, x, y, deps, ev)) return ev;
    }
    if constexpr (F == Fmt::BF16 && MB == 1) {
        sycl::event ev;
        if (esgemv2_bf16_try(q, w, x, y, deps, ev)) return ev;
    }
    if constexpr (F == Fmt::INT4 && MB == 1) {
        sycl::event ev;
        if (esgemv_i4_try(q, w, x, y, deps, ev)) return ev;
    }
    if constexpr (F == Fmt::FP8_E4M3 && MB == 1) {
        sycl::event ev;
        if (esgemv_fp8_try(q, w, x, y, deps, ev)) return ev;
    }
    int epl = gemv_epl_override();
    if (epl != 16 && epl != 32 && epl != 64) epl = GemvGeom<F>::EPL_DEFAULT;
    if (epl > GemvGeom<F>::EPL_MAX) epl = GemvGeom<F>::EPL_MAX;

    // The wide kernel needs each sub-group's K slice to be a whole number
    // of scale blocks and of lane-steps. The per-format DEFAULT elements
    // per lane is chosen for the main kernel and is often too coarse here
    // -- bf16 defaults to 64, which needs a 1024-element slice and so
    // rejected the bf16 router and a/b projections outright, the two
    // shapes that need this kernel most. Step EPL down until the slice
    // divides; only if none does fall through to the main kernel.
    {
        const int  slice = (w.K % WG_SUBGROUPS == 0) ? w.K / WG_SUBGROUPS : 0;
        const int  blk   = Traits<F>::block > 0 ? Traits<F>::block : 1;
        const int  force = gemv_wide_override();
        const bool want  = (force == 1) || (force != 0 && w.N <= kWideMaxN);
        if (want && slice > 0 && slice % blk == 0) {
            constexpr int MX = GemvGeom<F>::EPL_MAX;
            // The old guard demanded slice % (SG_SIZE*EPL) == 0, which is
            // stricter than gemv_wide actually needs.  Its inner loop is
            // per-lane -- `k + EPL <= k_end` -- so a slice that is a whole
            // number of EPL chunks is fully covered, the last chunks simply
            // landing on fewer lanes.  The strict form rejected the deltanet
            // a/b projection (N=64, K=5120, slice 640, 640 % 256 = 128) and
            // silently sent it to the main kernel, which gives N=64 exactly
            // TWO work-groups on a 256-EU card: 0.25 MB in 32.4 us, 1% of
            // roofline.  slice % EPL == 0 is the real requirement, alongside
            // the scale-block check already done above.
            const int wdiv = wide_relaxed() ? 1 : SG_SIZE;
            if constexpr (MX >= 64)
                if (slice % (wdiv * 64) == 0)
                    { if constexpr (MB == 1) return gemv_wide<F, 64>(q, w, x, y, deps);
                  sycl::event e; for (int m = 0; m < mc; ++m)
                      e = gemv_wide<F, 64>(q, w, x + int64_t(m) * w.K, y + int64_t(m) * w.N, m ? std::vector<sycl::event>{e} : deps);
                  return e; }
            if constexpr (MX >= 32)
                if (slice % (wdiv * 32) == 0)
                    { if constexpr (MB == 1) return gemv_wide<F, 32>(q, w, x, y, deps);
                  sycl::event e; for (int m = 0; m < mc; ++m)
                      e = gemv_wide<F, 32>(q, w, x + int64_t(m) * w.K, y + int64_t(m) * w.N, m ? std::vector<sycl::event>{e} : deps);
                  return e; }
            if (slice % (wdiv * 16) == 0)
                { if constexpr (MB == 1) return gemv_wide<F, 16>(q, w, x, y, deps);
                  sycl::event e; for (int m = 0; m < mc; ++m)
                      e = gemv_wide<F, 16>(q, w, x + int64_t(m) * w.K, y + int64_t(m) * w.N, m ? std::vector<sycl::event>{e} : deps);
                  return e; }
        }
        // K/8 is not whole scale blocks (Muse K=6656 = 52 INT4 groups):
        // gemv_wide splits by blocks instead, 6 or 7 per sub-group.  A lane
        // step never straddles a block when EPL divides the block.
        if (want && blk >= 16 && w.K % blk == 0 && w.K / blk >= WG_SUBGROUPS &&
            !(slice > 0 && slice % blk == 0)) {
            constexpr int MX = GemvGeom<F>::EPL_MAX;
            auto run_w = [&](auto tag) -> sycl::event {
                constexpr int E = decltype(tag)::value;
                if constexpr (MB == 1) return gemv_wide<F, E>(q, w, x, y, deps);
                sycl::event e;
                for (int m = 0; m < mc; ++m)
                    e = gemv_wide<F, E>(q, w, x + int64_t(m) * w.K, y + int64_t(m) * w.N,
                                        m ? std::vector<sycl::event>{e} : deps);
                return e;
            };
            if constexpr (MX >= 64) if (blk % 64 == 0) return run_w(std::integral_constant<int, 64>{});
            if constexpr (MX >= 32) if (blk % 32 == 0) return run_w(std::integral_constant<int, 32>{});
            if (blk % 16 == 0) return run_w(std::integral_constant<int, 16>{});
        }
    }
    int un = gemv_unroll_override();
    if (un != 1 && un != 2 && un != 4 && un != 8) un = GemvGeom<F>::UNROLL_DEFAULT;

    constexpr int MX = GemvGeom<F>::EPL_MAX;
    // B70_GEMV_OPT (MXFP4, EPL=16 only): bit0 hoist x out of the row loop,
    // bit1 decode E2M1 in the ALU instead of the SLM table.  Both attack the
    // same measured limit -- 369 GB/s on ffn gate_up against a 602 GB/s card,
    // with 8:1 L2:DRAM traffic and two SLM gathers per weight byte.
    if constexpr (F == Fmt::MXFP4) {
        const int opt = gemv_opt_override();
        if (epl == 16 && opt > 0) {
            if (opt == 1) {
                if (un == 4) return gemv_impl<F, 16, 4, 1, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
                if (un == 8) return gemv_impl<F, 16, 8, 1, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
                const int rps = rps_for(w.N);
                if (rps == 1) return gemv_impl<F, 16, 2, 1, 1, MB>(q, w, x, y, deps, mc);
                if (rps == 2) return gemv_impl<F, 16, 2, 1, 2, MB>(q, w, x, y, deps, mc);
                if (rps == 8) return gemv_impl<F, 16, 2, 1, 8, MB>(q, w, x, y, deps, mc);
                return gemv_impl<F, 16, 2, 1, 4, MB>(q, w, x, y, deps, mc);
            }
            if (opt == 2) {
                if (un == 4) return gemv_impl<F, 16, 4, 2, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
                if (un == 8) return gemv_impl<F, 16, 8, 2, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
                return gemv_impl<F, 16, 2, 2, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
            }
            if (un == 4) return gemv_impl<F, 16, 4, 3, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
            if (un == 8) return gemv_impl<F, 16, 8, 3, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
            return gemv_impl<F, 16, 2, 3, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        }
    }
    // INT4 decode rows: the hoisted-x kernel (B70_GEMV_OPT=0 = the plain
    // one).  Same UNROLL and rows per sub-group as the plain path, so each
    // row's accumulation -- and the output -- is bit-identical.
    if constexpr (F == Fmt::INT4 && MB == 1) {
        if (epl == 16 && gemv_opt_override() > 0) {
            // Rows per sub-group: B70_RPS (1|2|4|8) sweeps it; rows owned
            // never change a row's accumulation, so any value is exact.
            // Default 2: MEASURED on Muse-Glimmer-30B INT4 (one decode token,
            // q/o_gate N=4096 K=6656, o N=6656, FFN I=24576): RPS 4 62.1 ms,
            // 2 52.0, 1 54.1, 8 90.5 -- at 4 the N=4096-6656 shapes got only
            // 128-208 work-groups.
            static const int irps = [] { const char* e = std::getenv("B70_RPS");
                const int v = (e && *e) ? std::atoi(e) : 0;
                return (v == 1 || v == 2 || v == 4 || v == 8) ? v : 2; }();
            if (un == 4) {
                if (irps == 1) return gemv_impl<F, 16, 4, 1, 1, MB>(q, w, x, y, deps, mc);
                if (irps == 2) return gemv_impl<F, 16, 4, 1, 2, MB>(q, w, x, y, deps, mc);
                if (irps == 8) return gemv_impl<F, 16, 4, 1, 8, MB>(q, w, x, y, deps, mc);
                return gemv_impl<F, 16, 4, 1, 4, MB>(q, w, x, y, deps, mc);
            }
            if (un == 1) return gemv_impl<F, 16, 1, 1, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
            if (un == 2) return gemv_impl<F, 16, 2, 1, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
            if (un == 8) return gemv_impl<F, 16, 8, 1, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
            return gemv_impl<F, 16, 4, 1, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        }
    }
    if (epl == 16) {
        if (un == 1) return gemv_impl<F, 16, 1, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        if (un == 2) return gemv_impl<F, 16, 2, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        if (un == 8) return gemv_impl<F, 16, 8, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        return gemv_impl<F, 16, 4, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
    }
    if (epl == 32 && MX >= 32) {
        if (un == 1) return gemv_impl<F, 32, 1, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        if (un == 2) return gemv_impl<F, 32, 2, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        if (un == 8) return gemv_impl<F, 32, 8, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        return gemv_impl<F, 32, 4, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
    }
    if constexpr (MX >= 64) {
        if (un == 1) return gemv_impl<F, 64, 1, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        if (un == 2) return gemv_impl<F, 64, 2, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        if (un == 8) return gemv_impl<F, 64, 8, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
        return gemv_impl<F, 64, 4, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
    }
    return gemv_impl<F, 16, 4, 0, ROWS_PER_SG, MB>(q, w, x, y, deps, mc);
}
} // namespace

// The w4a16 DPAS small-M GEMM (int4_smallm.hpp) at M <= 8: speculative
// verify (MTP k = 3 verifies 4 rows) and batched decode of <= 8 sequences.
// Built here, with 128 registers per thread and so twice the threads per
// Xe core of libgrimoire_gemm.so's 256, and with its own split-K plan: the
// MXFP4 kernel's plan (KS while tiles * KS < 4096) left the INT4 kernel
// latency-bound.  MEASURED 2026-10-05, tools/i4rows_probe.cpp, Qwen3.8-27B
// shapes at M = 4, weights streamed from DRAM (old plan in the 256-GRF
// library -> this):
//   gate_up N=34816 K=5120    225 -> 183 us  (410 -> 510 GB/s)  KS 2 -> 4
//   down    N=5120  K=17408   113 ->  94 us  (413 -> 499)       KS 16 -> 4
//   dn_qkv  N=10240 K=5120     77 ->  62 us  (357 -> 445)       KS 8 -> 20
//   attn_q  N=12288 K=5120     87 ->  71 us  (379 -> 461)       KS 8 -> 20
//   z / out N=6144 / 5120      45 / 47 -> 42 / 41 us            KS 14 / 16 -> 20 / 24
//   lm_head N=248320          1324 -> 1179 us (503 -> 564)      KS 1 -> 4
// The fp32 decode GEMV reaches ~585 GB/s at M = 1; the rest of the gap is
// the decode ALU (~10%: without it this kernel streams ~540-565 GB/s).  An
// fp32 FMA kernel over the rows (each weight decoded once, the decode
// GEMV's access pattern) was ALU-bound at ~250 GB/s for M = 4.
// GRIMOIRE_I4_SMALLM_256=1 = the library's kernel and plan for every M.
sycl::event launch_int4_smallm(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                               float* Y, int M, const std::vector<sycl::event>& deps) {
    static const bool old = [] { const char* e = std::getenv("GRIMOIRE_I4_SMALLM_256");
        return e && *e == '1'; }();
    if (old || M > 8) return launch_int4_smallm_wide(q, w, X, Y, M, deps);
    const int N = w.N, K = w.K, tiles = N / 16;
    // Wide outputs and long K: 4 K slices; otherwise 256-element slices.
    int KS = tiles >= 1024 || K >= 12288 ? 4 : std::min(32, std::max(1, K / 256));
    int kc = (K + KS - 1) / KS;
    kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= 32) TPT *= 2;
    return (1 << w.int4_gshift()) == 64
        ? int4_smallm_impl<1, 64>(q, w, X, Y, M, KS, kc, TPT, deps)
        : int4_smallm_impl<1, 128>(q, w, X, Y, M, KS, kc, TPT, deps);
}

// ---------------------------------------------------------------------
//  BATCHED symmetric-int4 GEMV: MB token rows against one weight matrix.
//
//  This is what makes speculative verification affordable.  A batched
//  forward through the PREFILL path costs 135 ms at M=4 against a 31.85 ms
//  decode step, because its GEMM tiles are built for M=128 and its GDN
//  kernel pads to 64 tokens.  But the verify batch is weight-bound exactly
//  like decode is: load each weight ONCE and do MB dot products with it, and
//  M=4 costs what M=1 costs.
//
//  UNROLL is dropped to 1 here -- the batch supplies the memory-level
//  parallelism that UNROLL supplies at M=1, and xv[MB][EPL] already
//  occupies the registers UNROLL would have wanted.
// ---------------------------------------------------------------------
template <int RPS_, int MB>
sycl::event gemv_int4sym_batch_impl(sycl::queue& q, const uint8_t* pack,
                                    const float* ws, const float* x, float* y,
                                    int N, int K,
                                    const std::vector<sycl::event>& deps) {
    constexpr int EPL = 16;
    constexpr int G   = 128;
    const int rows_per_wg = WG_SUBGROUPS * RPS_;
    const int n_blocks    = (N + rows_per_wg - 1) / rows_per_wg;
    const int cap         = gemv_cap();
    const int n_groups    = (cap > 0 && n_blocks > cap) ? cap : n_blocks;
    const int wg_threads  = WG_SUBGROUPS * SG_SIZE;
    const int64_t row_bytes = int64_t(K) / 2;
    const int kg = K / G;

    // NOTE: grf_size<256> was tried here and is a REGRESSION -- it halves the
    // threads per EU, and this kernel is memory-bound: M=1 went 19.06 -> 27.52
    // ms, M=4 35.36 -> 45.55.  Better scaling (1.65x vs 1.86x), worse at every
    // absolute M.  B70_BATCH_RPS=2 is also worse (M=4 51.42 ms).
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> i4lut(16, h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(n_groups) * size_t(wg_threads),
                              size_t(wg_threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  sgid = int(sg.get_group_id()[0]);
                const int  lid  = int(it.get_local_id(0));
                constexpr int STEP = SG_SIZE * EPL;
                float* lut = i4lut.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                for (int b = lid; b < 16; b += wg_threads)
                    lut[b] = float(int(int8_t(uint8_t(b) << 4)) >> 4);
                sycl::group_barrier(it.get_group());

              for (int blk = int(it.get_group(0)); blk < n_blocks;
                   blk += int(it.get_group_range(0))) {
                const int n_base = blk * rows_per_wg + sgid * RPS_;
                float part[RPS_][MB];
                #pragma unroll
                for (int r = 0; r < RPS_; ++r)
                    #pragma unroll
                    for (int m = 0; m < MB; ++m) part[r][m] = 0.0f;

                for (int base = 0; base + STEP <= K; base += STEP) {
                    const int k0 = base + lane * EPL;
                    float xv[MB][EPL];
                    #pragma unroll
                    for (int m = 0; m < MB; ++m)
                        #pragma unroll
                        for (int i = 0; i < EPL; ++i)
                            xv[m][i] = x[int64_t(m) * K + k0 + i];
                    #pragma unroll
                    for (int r = 0; r < RPS_; ++r) {
                        const int n = n_base + r;
                        if (n >= N) continue;
                        const uint8_t* row = pack + int64_t(n) * row_bytes;
                        const uint64_t packed =
                            *reinterpret_cast<const uint64_t*>(row + (k0 >> 1));
                        const float sc = ws[int64_t(n) * kg + k0 / G];
                        float acc[MB];
                        #pragma unroll
                        for (int m = 0; m < MB; ++m) acc[m] = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const uint8_t byte = uint8_t(packed >> (8 * i));
                            const float w0 = lut[byte & 0x0F];
                            const float w1 = lut[byte >> 4];
                            #pragma unroll
                            for (int m = 0; m < MB; ++m) {
                                acc[m] = sycl::fma(w0, xv[m][2 * i],     acc[m]);
                                acc[m] = sycl::fma(w1, xv[m][2 * i + 1], acc[m]);
                            }
                        }
                        #pragma unroll
                        for (int m = 0; m < MB; ++m)
                            part[r][m] = sycl::fma(acc[m], sc, part[r][m]);
                    }
                }

                #pragma unroll
                for (int r = 0; r < RPS_; ++r) {
                    const int n = n_base + r;
                    #pragma unroll
                    for (int m = 0; m < MB; ++m) {
                        const float total = sycl::reduce_over_group(
                            sg, part[r][m], sycl::plus<float>());
                        if (lane == 0 && n < N) y[int64_t(m) * N + n] = total;
                    }
                }
              }
            });
    });
}

sycl::event launch_gemv_int4sym_batch(sycl::queue& q, const uint8_t* pack,
                                      const float* ws, const float* x, float* y,
                                      int N, int K, int MB,
                                      const std::vector<sycl::event>& deps) {
    static const int rps = [] { const char* v = std::getenv("B70_BATCH_RPS");
        int r = v && *v ? std::atoi(v) : 4; return (r == 2 || r == 4) ? r : 4; }();
    if (rps == 2) {
        switch (MB) {
            case 1: return gemv_int4sym_batch_impl<2, 1>(q, pack, ws, x, y, N, K, deps);
            case 2: return gemv_int4sym_batch_impl<2, 2>(q, pack, ws, x, y, N, K, deps);
            case 3: return gemv_int4sym_batch_impl<2, 3>(q, pack, ws, x, y, N, K, deps);
            default: return gemv_int4sym_batch_impl<2, 4>(q, pack, ws, x, y, N, K, deps);
        }
    }
    switch (MB) {
        case 1: return gemv_int4sym_batch_impl<4, 1>(q, pack, ws, x, y, N, K, deps);
        case 2: return gemv_int4sym_batch_impl<4, 2>(q, pack, ws, x, y, N, K, deps);
        case 3: return gemv_int4sym_batch_impl<4, 3>(q, pack, ws, x, y, N, K, deps);
        default: return gemv_int4sym_batch_impl<4, 4>(q, pack, ws, x, y, N, K, deps);
    }
}

// ---------------------------------------------------------------------
// MEASURED WORSE, 2026-09-05. Kept opt-in (GRIMOIRE_GEMV_BATCH_NORED=1)
// and out of any default path as a documented dead end -- do not retry
// this exact axis change without fixing the coalescing problem below.
//
// gemv_int4sym_batch_impl above splits K across the 16 lanes of a
// sub-group and calls sycl::reduce_over_group once per (block, row,
// batch-column). Reading OpenVINO's GPU plugin kernel for this exact
// batch range (fully_connected_gpu_bf_tiled_dyn_b_core.cl, batch 2-32,
// INT4 weights) showed a different axis: each LANE owns one output row
// for the ENTIRE K-walk, no cross-lane reduction at all. The hypothesis
// was that reduce_over_group was the tax costing us the FFN (N=17408)
// -- widening GRIMOIRE_GEMV_BATCH_MAX_N to cover it on 2026-09-05 made
// verify WORSE (54.7 -> 60.4 ms/round) through the reduce-based kernel.
//
// MEASURED (llama-benchy-style MTP profile, W4A8, k=3, same binary,
// coherence PASSED both ways, 20/27 drafts accepted identically):
//     reduce-based (gemv_int4sym_batch_impl)   verify 54.5 ms/round
//     this kernel  (batch_nored)               verify 61.9 ms/round
// WORSE, not better. The hypothesis was wrong: removing the reduction
// also removes the ONE thing the reduce-based kernel gets right for
// free -- in that kernel, all 16 lanes read ADJACENT bytes of the SAME
// row (k0 = base + lane*EPL), one coalesced transaction. This kernel
// gives each lane a DIFFERENT row (row_bytes apart), so the sub-group
// now issues 16 independent, non-adjacent cache-line reads every step
// instead of one wide coalesced one. That coalescing loss costs more
// than the reduction saved. OpenVINO's kernel gets no-reduction AND
// coalesced reads together because its weight layout is physically
// interleaved (the OSV32/OSV64 layouts in its core kernel) for exactly
// this access pattern -- GRIMOIRE's row-major payload is not, so the
// axis alone does not carry the win across. A real retry needs either
// an interleaved repack of the INT4 payload (bigger, riskier change,
// touches the loader) or a block-read across lanes for DIFFERENT rows
// within the SAME weight-format constraints (unexplored). Neither is
// done. Left in as a documented negative result, not a live option.
// ---------------------------------------------------------------------
template <int MB>
sycl::event gemv_int4sym_batch_nored_impl(sycl::queue& q, const uint8_t* pack,
                                          const float* ws, const float* x, float* y,
                                          int N, int K,
                                          const std::vector<sycl::event>& deps) {
    constexpr int G = 128;
    const int rows_per_wg = WG_SUBGROUPS * SG_SIZE;  // one output row per lane
    const int n_blocks    = (N + rows_per_wg - 1) / rows_per_wg;
    const int cap         = gemv_cap();
    const int n_groups    = (cap > 0 && n_blocks > cap) ? cap : n_blocks;
    const int wg_threads  = WG_SUBGROUPS * SG_SIZE;
    const int64_t row_bytes = int64_t(K) / 2;
    const int kg = K / G;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> i4lut(16, h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(n_groups) * size_t(wg_threads),
                              size_t(wg_threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  sgid = int(sg.get_group_id()[0]);
                const int  lid  = int(it.get_local_id(0));
                float* lut = i4lut.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                for (int b = lid; b < 16; b += wg_threads)
                    lut[b] = float(int(int8_t(uint8_t(b) << 4)) >> 4);
                sycl::group_barrier(it.get_group());

                for (int blk = int(it.get_group(0)); blk < n_blocks;
                     blk += int(it.get_group_range(0))) {
                    // OpenVINO's axis: this lane owns row n for the whole
                    // K-walk. No reduce_over_group anywhere below.
                    const int n = blk * rows_per_wg + sgid * SG_SIZE + lane;
                    if (n >= N) continue;
                    const uint8_t* row = pack + int64_t(n) * row_bytes;

                    float acc[MB];
                    #pragma unroll
                    for (int m = 0; m < MB; ++m) acc[m] = 0.0f;

                    for (int k0 = 0; k0 < K; k0 += 16) {
                        const uint64_t packed =
                            *reinterpret_cast<const uint64_t*>(row + (k0 >> 1));
                        const float sc = ws[int64_t(n) * kg + k0 / G];
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const uint8_t byte = uint8_t(packed >> (8 * i));
                            const float w0 = lut[byte & 0x0F] * sc;
                            const float w1 = lut[byte >> 4]   * sc;
                            #pragma unroll
                            for (int m = 0; m < MB; ++m) {
                                acc[m] = sycl::fma(w0, x[int64_t(m) * K + k0 + 2 * i],     acc[m]);
                                acc[m] = sycl::fma(w1, x[int64_t(m) * K + k0 + 2 * i + 1], acc[m]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int m = 0; m < MB; ++m) y[int64_t(m) * N + n] = acc[m];
                }
            });
    });
}

sycl::event launch_gemv_int4sym_batch_nored(sycl::queue& q, const uint8_t* pack,
                                            const float* ws, const float* x, float* y,
                                            int N, int K, int MB,
                                            const std::vector<sycl::event>& deps) {
    switch (MB) {
        case 1: return gemv_int4sym_batch_nored_impl<1>(q, pack, ws, x, y, N, K, deps);
        case 2: return gemv_int4sym_batch_nored_impl<2>(q, pack, ws, x, y, N, K, deps);
        case 3: return gemv_int4sym_batch_nored_impl<3>(q, pack, ws, x, y, N, K, deps);
        default: return gemv_int4sym_batch_nored_impl<4>(q, pack, ws, x, y, N, K, deps);
    }
}

sycl::event launch_gemv_int4sym(sycl::queue& q, const uint8_t* pack,
                                const float* ws, const float* x, float* y,
                                int N, int K,
                                const std::vector<sycl::event>& deps) {
    static const bool wide = std::getenv("B70_I4_WIDE") != nullptr;
    if (wide && N <= 2048 && K % (WG_SUBGROUPS * 16) == 0)
        return gemv_int4sym_wide(q, pack, ws, x, y, N, K, deps);
    return gemv_int4sym_impl<4, 2>(q, pack, ws, x, y, N, K, deps);
}

// M activation rows against one weight matrix, bit-identical per row to
// launch_gemv() -- the same variant, the same accumulation order -- but the
// weights are read once per chunk of 4 rows instead of once per row.  This is
// the verify path for speculative decoding on plain (unconverted) weights.
sycl::event launch_gemv_batch(sycl::queue& q, const QuantWeight& w,
                              const float* x, float* y, int M,
                              const std::vector<sycl::event>& deps) {
    if (M <= 1) return launch_gemv(q, w, x, y, deps);
    sycl::event e;
    for (int m0 = 0; m0 < M; m0 += 4) {
        const int mc = std::min(4, M - m0);
        const float* xm = x + int64_t(m0) * w.K;
        float* ym = y + int64_t(m0) * w.N;
        const std::vector<sycl::event> d = m0 ? std::vector<sycl::event>{e} : deps;
        switch (w.fmt) {
            case Fmt::BF16:     e = dispatch<Fmt::BF16, 4>(q, w, xm, ym, d, mc); break;
            case Fmt::FP8_E4M3: e = dispatch<Fmt::FP8_E4M3, 4>(q, w, xm, ym, d, mc); break;
            case Fmt::FP8_E5M2: e = dispatch<Fmt::FP8_E5M2, 4>(q, w, xm, ym, d, mc); break;
            case Fmt::INT8:     e = dispatch<Fmt::INT8, 4>(q, w, xm, ym, d, mc); break;
            case Fmt::INT4:     e = dispatch<Fmt::INT4, 4>(q, w, xm, ym, d, mc); break;
            case Fmt::MXFP8:    e = dispatch<Fmt::MXFP8, 4>(q, w, xm, ym, d, mc); break;
            case Fmt::MXFP4:    e = dispatch<Fmt::MXFP4, 4>(q, w, xm, ym, d, mc); break;
        }
    }
    return e;
}

sycl::event launch_gemv(sycl::queue& q, const QuantWeight& w,
                        const float* x, float* y,
                        const std::vector<sycl::event>& deps) {
    switch (w.fmt) {
        case Fmt::BF16:     return dispatch<Fmt::BF16>(q, w, x, y, deps);
        case Fmt::FP8_E4M3: return dispatch<Fmt::FP8_E4M3>(q, w, x, y, deps);
        case Fmt::FP8_E5M2: return dispatch<Fmt::FP8_E5M2>(q, w, x, y, deps);
        case Fmt::INT8:     return dispatch<Fmt::INT8>(q, w, x, y, deps);
        case Fmt::INT4:     return dispatch<Fmt::INT4>(q, w, x, y, deps);
        case Fmt::MXFP8:    return dispatch<Fmt::MXFP8>(q, w, x, y, deps);
        case Fmt::MXFP4:    return dispatch<Fmt::MXFP4>(q, w, x, y, deps);
    }
    return {};
}

} // namespace b70
