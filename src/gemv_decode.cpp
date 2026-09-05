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
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemv_step.hpp"
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
template <Fmt F, int EPL_F, int UNROLL, int OPT = 0, int RPS = ROWS_PER_SG>
sycl::event gemv_impl(sycl::queue& q, const QuantWeight& w,
                      const float* x, float* y,
                      const std::vector<sycl::event>& deps) {
    const int N = w.N, K = w.K;
    const int rows_per_wg_sg = WG_SUBGROUPS;
    const int rows_per_wg    = WG_SUBGROUPS * RPS;
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
                const int n_base = blk * rows_per_wg + sgid * RPS;

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
                // accumulator per row exposes WG_SUBGROUPS*RPS
                // streams without changing the per-row summation order.
                float part[RPS][UNROLL];
                #pragma unroll
                for (int r = 0; r < RPS; ++r)
                    #pragma unroll
                    for (int u = 0; u < UNROLL; ++u) part[r][u] = 0.0f;

                const int span = STEP_F * UNROLL;
                int base = 0;
                if constexpr (OPT != 0 && F == Fmt::MXFP4) {
                    // OPT bit 0: load this lane's activations ONCE per K step
                    // and reuse them across all RPS rows.  Without it
                    // the compiler cannot prove x and row do not alias and
                    // reloads x for every row.
                    for (; base + span <= K; base += span) {
                        const int k0 = base + lane * EPL_F;
                        float xv[UNROLL][EPL_F];
                        #pragma unroll
                        for (int u = 0; u < UNROLL; ++u)
                            #pragma unroll
                            for (int i = 0; i < EPL_F; ++i)
                                xv[u][i] = x[k0 + u * STEP_F + i];
                        #pragma unroll
                        for (int r = 0; r < RPS; ++r) {
                            const int n = n_base + r;
                            if (n >= N) continue;
                            const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                            #pragma unroll
                            for (int u = 0; u < UNROLL; ++u)
                                part[r][u] += GemvStep<F, EPL_F>::template run_xv<OPT>(
                                    wc,row,&xv[u][0],slut,nlut,n,k0+u*STEP_F);
                        }
                    }
                } else
                for (; base + span <= K; base += span) {
                    const int k0 = base + lane * EPL_F;
                    #pragma unroll
                    for (int r = 0; r < RPS; ++r) {
                        const int n = n_base + r;
                        if (n >= N) continue;
                        const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                        #pragma unroll
                        for (int u = 0; u < UNROLL; ++u)
                            part[r][u] += GemvStep<F, EPL_F>::run(
                                wc,row,x,lut,slut,nlut,n,k0+u*STEP_F);
                    }
                }
                for (; base + STEP_F <= K; base += STEP_F) {
                    const int k0=base+lane*EPL_F;
                    #pragma unroll
                    for (int r=0;r<RPS;++r) {
                        const int n=n_base+r;if(n>=N)continue;
                        const uint8_t* row=wc.payload+int64_t(n)*wc.row_bytes;
                        part[r][0]+=GemvStep<F,EPL_F>::run(
                            wc,row,x,lut,slut,nlut,n,k0);
                    }
                }

                float acc[RPS];
                #pragma unroll
                for(int r=0;r<RPS;++r){
                    float sum=0.0f;
                    #pragma unroll
                    for(int u=0;u<UNROLL;++u)sum+=part[r][u];
                    acc[r]=sum;
                    const int n=n_base+r;if(n>=N)continue;
                    const int done=(K/STEP_F)*STEP_F;
                    for(int k=done+lane;k<K;k+=SG_SIZE)
                        acc[r]=sycl::fma(wc.at(n,k),x[k],acc[r]);
                }

                #pragma unroll
                for (int r = 0; r < RPS; ++r) {
                    const float total =
                        sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                    const int n = n_base + r;
                    if (lane == 0 && n < N) y[n] = total;
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
                // otherwise, so no tail handling is needed here.
                const int n      = int(it.get_group(0));
                const int per_sg = K / WG_SUBGROUPS;
                const int k_beg  = sgid * per_sg;
                const int k_end  = k_beg + per_sg;

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
    return (N >= 8192 && N < 16384) ? 2 : 4;
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

template <Fmt F>
sycl::event dispatch(sycl::queue& q, const QuantWeight& w, const float* x,
                     float* y, const std::vector<sycl::event>& deps) {
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
                    return gemv_wide<F, 64>(q, w, x, y, deps);
            if constexpr (MX >= 32)
                if (slice % (wdiv * 32) == 0)
                    return gemv_wide<F, 32>(q, w, x, y, deps);
            if (slice % (wdiv * 16) == 0)
                return gemv_wide<F, 16>(q, w, x, y, deps);
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
                if (un == 4) return gemv_impl<F, 16, 4, 1>(q, w, x, y, deps);
                if (un == 8) return gemv_impl<F, 16, 8, 1>(q, w, x, y, deps);
                const int rps = rps_for(w.N);
                if (rps == 1) return gemv_impl<F, 16, 2, 1, 1>(q, w, x, y, deps);
                if (rps == 2) return gemv_impl<F, 16, 2, 1, 2>(q, w, x, y, deps);
                if (rps == 8) return gemv_impl<F, 16, 2, 1, 8>(q, w, x, y, deps);
                return gemv_impl<F, 16, 2, 1, 4>(q, w, x, y, deps);
            }
            if (opt == 2) {
                if (un == 4) return gemv_impl<F, 16, 4, 2>(q, w, x, y, deps);
                if (un == 8) return gemv_impl<F, 16, 8, 2>(q, w, x, y, deps);
                return gemv_impl<F, 16, 2, 2>(q, w, x, y, deps);
            }
            if (un == 4) return gemv_impl<F, 16, 4, 3>(q, w, x, y, deps);
            if (un == 8) return gemv_impl<F, 16, 8, 3>(q, w, x, y, deps);
            return gemv_impl<F, 16, 2, 3>(q, w, x, y, deps);
        }
    }
    if (epl == 16) {
        if (un == 1) return gemv_impl<F, 16, 1>(q, w, x, y, deps);
        if (un == 2) return gemv_impl<F, 16, 2>(q, w, x, y, deps);
        if (un == 8) return gemv_impl<F, 16, 8>(q, w, x, y, deps);
        return gemv_impl<F, 16, 4>(q, w, x, y, deps);
    }
    if (epl == 32 && MX >= 32) {
        if (un == 1) return gemv_impl<F, 32, 1>(q, w, x, y, deps);
        if (un == 2) return gemv_impl<F, 32, 2>(q, w, x, y, deps);
        if (un == 8) return gemv_impl<F, 32, 8>(q, w, x, y, deps);
        return gemv_impl<F, 32, 4>(q, w, x, y, deps);
    }
    if constexpr (MX >= 64) {
        if (un == 1) return gemv_impl<F, 64, 1>(q, w, x, y, deps);
        if (un == 2) return gemv_impl<F, 64, 2>(q, w, x, y, deps);
        if (un == 8) return gemv_impl<F, 64, 8>(q, w, x, y, deps);
        return gemv_impl<F, 64, 4>(q, w, x, y, deps);
    }
    return gemv_impl<F, 16, 4>(q, w, x, y, deps);
}
} // namespace

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
