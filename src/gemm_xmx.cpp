// =====================================================================
//  gemm_xmx.cpp  --  the prompt-processing datapath
//
//  y[M][N] = x[M][K] * W[N][K]^T, W quantized.
//
//  Two datapaths, chosen by Traits<F>::pipe:
//
//   PIPE_FLT  bf16 DPAS, 8x16x16, fp32 accumulator.
//             Serves BF16 / FP8 / MXFP8 / MXFP4. For the MX formats the
//             E8M0 scale is a pure power of two, and bf16 carries the
//             full 8-bit fp32 exponent, so folding the scale into the
//             dequantized tile is EXACT -- no accuracy is lost versus
//             accumulating per block in fp32, and the inner loop stays
//             free of scale multiplies.
//
//   PIPE_INT  int8 DPAS, 8x16x32, int32 accumulator, 2x the bf16 rate.
//             This is the path that reaches the 367 TOPS figure. Serves
//             INT8 and INT4. Scales are applied to the int32 accumulator
//             at group boundaries, so they never enter the MAC loop.
//
//  Note on MXFP4: it CANNOT go down the int8 path. E2M1 is a floating
//  point grid {0,.5,1,1.5,2,3,4,6}; forcing it into int8 operands (as
//  the source blueprint does) collapses 0.5 and 1.5 onto integers and
//  destroys the format. MXFP4 gets 4-bit memory traffic with bf16-rate
//  math. If you want 4-bit weights AND the int8 XMX rate, use INT4.
// =====================================================================
#include "kernels.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdlib>

namespace b70 {

namespace matrix = sycl::ext::oneapi::experimental::matrix;

namespace {

// XMX prefill benefits from a wider workgroup than decode: sixteen
// subgroups reuse each decoded B tile across twice as many prompt rows.
// Keep this local so decode/MoE launch geometry remains unchanged.
constexpr int XMX_SUBGROUPS = 64;

// ---------------------------------------------------------------------
// B-operand staging. The DPAS B fragment wants VNNI (packed) order:
//   bf16: B[k][n] -> slm[(k/2)*WG_N*2 + n*2 + (k%2)]
//   int8: B[k][n] -> slm[(k/4)*WG_N*4 + n*4 + (k%4)]
// We write directly in that order while dequantizing, so no separate
// transpose pass and no second SLM round trip.
// ---------------------------------------------------------------------
inline int vnni_off_bf16(int k, int n) { return (k >> 1) * WG_N * 2 + n * 2 + (k & 1); }
template <int NPSG>
inline int vnni_off_bf16_t(int k, int n) {
    return (k >> 1) * (TN * NPSG) * 2 + n * 2 + (k & 1);
}
inline int vnni_off_int8(int k, int n) { return (k >> 2) * WG_N * 4 + n * 4 + (k & 3); }

// ---------------------------------------------------------------------
// Float pipeline
// ---------------------------------------------------------------------
template <Fmt F, int MPSG, int NPSG>
sycl::event gemm_flt(sycl::queue& q, const QuantWeight& w,
                     const sycl_bf16* x, float* y, int M,
                     const std::vector<sycl::event>& deps) {
    const int N = w.N, K = w.K;
    constexpr int WGM = TM * MPSG * XMX_SUBGROUPS;
    constexpr int WGN = TN * NPSG;
    const int gm = (M + WGM - 1) / WGM;
    const int gn = (N + WGN - 1) / WGN;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;

        sycl::local_accessor<sycl_bf16, 1> slmA(WGM * WG_K_BF16, h);
        sycl::local_accessor<sycl_bf16, 1> slmB(WG_K_BF16 * WGN, h);
        // fp32 epilogue scratch, one TM x TN tile per sub-group
        sycl::local_accessor<float, 1> slmEpi(XMX_SUBGROUPS * TM * TN, h);

        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(size_t(gm), size_t(gn) * XMX_SUBGROUPS * SG_SIZE),
                              sycl::range<2>(1, size_t(XMX_SUBGROUPS) * SG_SIZE)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg    = it.get_sub_group();
                const int  sgid  = int(sg.get_group_id()[0]);
                const int  lane  = int(sg.get_local_id()[0]);
                const int  lid   = int(it.get_local_id(1));
                const int  m_wg  = int(it.get_group(0)) * WGM;
                const int  n_wg  = int(it.get_group(1)) * WGN;

                matrix::joint_matrix<sycl::sub_group, sycl_bf16,
                                     matrix::use::a, TM, TK_BF16,
                                     matrix::layout::row_major> fragA[MPSG];
                matrix::joint_matrix<sycl::sub_group, sycl_bf16,
                                     matrix::use::b, TK_BF16, TN,
                                     matrix::layout::ext_intel_packed> fragB[NPSG];
                matrix::joint_matrix<sycl::sub_group, float,
                                     matrix::use::accumulator, TM, TN>
                                     accC[MPSG][NPSG];

                #pragma unroll
                for (int m = 0; m < MPSG; ++m) {
                    #pragma unroll
                    for (int j = 0; j < NPSG; ++j) {
                        matrix::joint_matrix_fill(sg, accC[m][j], 0.0f);
                    }
                }

                for (int k_wg = 0; k_wg < K; k_wg += WG_K_BF16) {

                    // ---- stage A: plain bf16 activations ----------------
                    for (int i = lid; i < WGM * WG_K_BF16; i += XMX_SUBGROUPS * SG_SIZE) {
                        const int mr = i / WG_K_BF16, kc = i % WG_K_BF16;
                        const int gmi = m_wg + mr, gki = k_wg + kc;
                        slmA[i] = (gmi < M && gki < K)
                                ? x[int64_t(gmi) * K + gki] : sycl_bf16(0.0f);
                    }

                    // ---- stage B: dequantize straight into VNNI order ---
                    // Each thread owns whole (n, k-block) strips so the
                    // scale lookup is hoisted out of the element loop.
                    for (int i = lid; i < WGN * WG_K_BF16; i += XMX_SUBGROUPS * SG_SIZE) {
                        const int nc = i / WG_K_BF16, kc = i % WG_K_BF16;
                        const int gni = n_wg + nc, gki = k_wg + kc;
                        // The dispatch already specialized F.  Calling
                        // QuantWeight::at() here re-ran two seven-way
                        // switches for every staged element and dominated
                        // short-K projections.  Resolve layout and scale at
                        // compile time instead.
                        float v = 0.0f;
                        if (gni < N && gki < K) {
                            const uint8_t* row = wc.payload + int64_t(gni) * wc.row_bytes;
                            float scale = 1.0f;
                            if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::FP8_E5M2 ||
                                          F == Fmt::INT8) {
                                scale = static_cast<const float*>(wc.scales)[gni];
                            } else if constexpr (F == Fmt::INT4) {
                                scale = bf16_to_f32(static_cast<const bf16_t*>(wc.scales)
                                    [int64_t(gni) * wc.row_scales + gki / kInt4Group]);
                            } else if constexpr (F == Fmt::MXFP8 || F == Fmt::MXFP4) {
                                scale = e8m0_to_f32(static_cast<const uint8_t*>(wc.scales)
                                    [int64_t(gni) * wc.row_scales + gki / kMXBlock]);
                            }
                            if constexpr (F == Fmt::INT4)
                                v = decode_int4(row, gki, scale,
                                    wc.zeros[int64_t(gni) * wc.row_scales + gki / kInt4Group]);
                            else
                                v = decode_elem<F>(row, gki, scale);
                        }
                        slmB[vnni_off_bf16_t<NPSG>(kc, nc)] = sycl_bf16(v);
                    }

                    sycl::group_barrier(it.get_group());

                    // 4 A fragments x 4 B fragments = 16 MADs from 8
                    // loads. The B fragments are reused across all M
                    // blocks and vice versa, which is where the
                    // arithmetic intensity comes from.
                    #pragma unroll
                    for (int kk = 0; kk < WG_K_BF16; kk += TK_BF16) {
                        #pragma unroll
                        for (int m = 0; m < MPSG; ++m) {
                            matrix::joint_matrix_load(
                                sg, fragA[m],
                                slmA.template get_multi_ptr<sycl::access::decorated::no>()
                                    + (sgid * MPSG + m) * TM * WG_K_BF16 + kk,
                                WG_K_BF16);
                        }
                        #pragma unroll
                        for (int j = 0; j < NPSG; ++j) {
                            matrix::joint_matrix_load(
                                sg, fragB[j],
                                slmB.template get_multi_ptr<sycl::access::decorated::no>()
                                    + vnni_off_bf16_t<NPSG>(kk, j * TN),
                                WGN * 2);
                            #pragma unroll
                            for (int m = 0; m < MPSG; ++m) {
                                matrix::joint_matrix_mad(sg, accC[m][j], fragA[m],
                                                         fragB[j], accC[m][j]);
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }

                // ---- write back -------------------------------------
                #pragma unroll
                for (int m = 0; m < MPSG; ++m) {
                #pragma unroll
                for (int j = 0; j < NPSG; ++j) {
                    const int m_base = m_wg + (sgid * MPSG + m) * TM;
                    const int n_base = n_wg + j * TN;
                    if (m_base + TM <= M && n_base + TN <= N) {
                        matrix::joint_matrix_store(
                            sg, accC[m][j],
                            sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::no>(
                                y + int64_t(m_base) * N + n_base),
                            size_t(N), matrix::layout::row_major);
                    } else {
                        // Ragged edge: spill the fragment through a float
                        // scratch tile. It must be float, not the bf16 A
                        // staging buffer -- the accumulator is fp32 and
                        // joint_matrix_store will not convert.
                        float* scratch = slmEpi.template
                            get_multi_ptr<sycl::access::decorated::no>().get()
                            + sgid * TM * TN;
                        matrix::joint_matrix_store(
                            sg, accC[m][j],
                            sycl::address_space_cast<
                                sycl::access::address_space::local_space,
                                sycl::access::decorated::no>(scratch),
                            size_t(TN), matrix::layout::row_major);
                        sycl::group_barrier(sg);
                        for (int e = lane; e < TM * TN; e += SG_SIZE) {
                            const int r = e / TN, c = e % TN;
                            if (m_base + r < M && n_base + c < N)
                                y[int64_t(m_base + r) * N + n_base + c] = scratch[e];
                        }
                        sycl::group_barrier(sg);
                    }
                }
                }
            });
    });
}

// ---------------------------------------------------------------------
// Integer pipeline. Activations are dynamically quantized to int8 with a
// per-row scale by the caller; here they arrive already as int8 in the
// bf16 buffer's storage. Accumulate int32, apply scales at the end.
// ---------------------------------------------------------------------
template <Fmt F>
sycl::event gemm_int(sycl::queue& q, const QuantWeight& w,
                     const int8_t* xq, const float* x_scale,
                     float* y, int M,
                     const std::vector<sycl::event>& deps) {
    const int N = w.N, K = w.K;
    const int gm = (M + WG_M_INT - 1) / WG_M_INT;
    const int gn = (N + WG_N - 1) / WG_N;

    // INT4 group must not straddle a work-group K tile, or the scale
    // could not be applied on the accumulator.
    const int KT = WG_K_INT8;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;

        sycl::local_accessor<int8_t, 1> slmA(WG_M_INT * KT, h);
        sycl::local_accessor<int8_t, 1> slmB(KT * WG_N, h);
        // Epilogue scratch. get_wi_data was removed in oneAPI 2026, and
        // joint_matrix_apply gives no element coordinates, so the tile is
        // spilled to SLM and scaled with an ordinary indexed loop. One
        // int32 spill tile and one fp32 running tile per sub-group.
        sycl::local_accessor<int32_t, 1> slmSpill(WG_SUBGROUPS * TM * TN, h);
        sycl::local_accessor<float, 1>   slmAcc(WG_SUBGROUPS * N_PER_SG * TM * TN, h);

        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(size_t(gm), size_t(gn) * WG_SUBGROUPS * SG_SIZE),
                              sycl::range<2>(1, size_t(WG_SUBGROUPS) * SG_SIZE)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  sgid = int(sg.get_group_id()[0]);
                const int  lid  = int(it.get_local_id(1));
                const int  m_wg = int(it.get_group(0)) * WG_M_INT;
                const int  n_wg = int(it.get_group(1)) * WG_N;

                matrix::joint_matrix<sycl::sub_group, int8_t,
                                     matrix::use::a, TM, TK_INT8,
                                     matrix::layout::row_major> fragA[M_PER_SG_INT];
                matrix::joint_matrix<sycl::sub_group, int8_t,
                                     matrix::use::b, TK_INT8, TN,
                                     matrix::layout::ext_intel_packed> fragB[N_PER_SG];
                matrix::joint_matrix<sycl::sub_group, int32_t,
                                     matrix::use::accumulator, TM, TN>
                                     accI[M_PER_SG_INT][N_PER_SG];

                const int lane = int(sg.get_local_id()[0]);
                int32_t* spill = slmSpill.template
                    get_multi_ptr<sycl::access::decorated::no>().get()
                    + sgid * TM * TN;
                float*   facc  = slmAcc.template
                    get_multi_ptr<sycl::access::decorated::no>().get()
                    + sgid * M_PER_SG_INT * N_PER_SG * TM * TN;

                // Running fp32 result across quantization groups. INT8 has
                // one group per row so this flushes once; INT4 flushes at
                // every 128-column group boundary.
                for (int e = lane; e < M_PER_SG_INT * N_PER_SG * TM * TN; e += SG_SIZE)
                    facc[e] = 0.0f;
                sycl::group_barrier(sg);

                // The INT4 weight scale is constant across a 128-column group, so
                // the int32 accumulator can run across the whole group and be
                // dequantized once.  Previously it was reset and spilled through SLM
                // every KT=64 columns -- two flushes per group, each costing a
                // joint_matrix_store plus two barriers per N block.
                const int GRP = (F == Fmt::INT4) ? kInt4Group : KT;
                for (int k_wg = 0; k_wg < K; k_wg += KT) {
                    const bool grp_start = (k_wg % GRP) == 0;
                    const bool grp_end   = ((k_wg + KT) % GRP) == 0 || (k_wg + KT) >= K;
                    if (grp_start) {
                        #pragma unroll
                        for (int mb = 0; mb < M_PER_SG_INT; ++mb)
                            #pragma unroll
                            for (int j = 0; j < N_PER_SG; ++j)
                                matrix::joint_matrix_fill(sg, accI[mb][j], 0);
                    }

                    for (int i = lid; i < WG_M_INT * KT; i += WG_SUBGROUPS * SG_SIZE) {
                        const int mr = i / KT, kc = i % KT;
                        const int gmi = m_wg + mr, gki = k_wg + kc;
                        slmA[i] = (gmi < M && gki < K) ? xq[int64_t(gmi) * K + gki] : int8_t(0);
                    }
                    for (int i = lid; i < WG_N * KT; i += WG_SUBGROUPS * SG_SIZE) {
                        const int nc = i / KT, kc = i % KT;
                        const int gni = n_wg + nc, gki = k_wg + kc;
                        int8_t v = 0;
                        if (gni < N && gki < K) {
                            const uint8_t* row = wc.payload + int64_t(gni) * wc.row_bytes;
                            if constexpr (F == Fmt::INT8) {
                                v = int8_t(row[gki]);
                            } else {   // INT4: subtract the zero point here,
                                       // keeping the DPAS operands symmetric
                                const uint8_t q4 = unpack_int4_raw(row, gki);
                                v = int8_t(int(q4) - int(wc.zero_for(gni, gki)));
                            }
                        }
                        slmB[vnni_off_int8(kc, nc)] = v;
                    }
                    sycl::group_barrier(it.get_group());

                    #pragma unroll
                    for (int kk = 0; kk < KT; kk += TK_INT8) {
                        #pragma unroll
                        for (int mb = 0; mb < M_PER_SG_INT; ++mb)
                            matrix::joint_matrix_load(
                                sg, fragA[mb],
                                slmA.template get_multi_ptr<sycl::access::decorated::no>()
                                    + (sgid * M_PER_SG_INT + mb) * TM * KT + kk,
                                KT);
                        #pragma unroll
                        for (int j = 0; j < N_PER_SG; ++j) {
                            matrix::joint_matrix_load(
                                sg, fragB[j],
                                slmB.template get_multi_ptr<sycl::access::decorated::no>()
                                    + vnni_off_int8(kk, j * TN),
                                WG_N * 4);
                            #pragma unroll
                            for (int mb = 0; mb < M_PER_SG_INT; ++mb)
                                matrix::joint_matrix_mad(sg, accI[mb][j], fragA[mb],
                                                         fragB[j], accI[mb][j]);
                        }
                    }
                    sycl::group_barrier(it.get_group());

                    // Dequantize the group's int32 partial into fp32, once per
                    // group.  Spill through SLM so each element's (row, col) is
                    // known -- the weight scale is per output channel, so the
                    // column index selects it.
                    if (!grp_end) continue;
                    #pragma unroll
                    for (int mb = 0; mb < M_PER_SG_INT; ++mb)
                    #pragma unroll
                    for (int j = 0; j < N_PER_SG; ++j) {
                        matrix::joint_matrix_store(
                            sg, accI[mb][j],
                            sycl::address_space_cast<
                                sycl::access::address_space::local_space,
                                sycl::access::decorated::no>(spill),
                            size_t(TN), matrix::layout::row_major);
                        sycl::group_barrier(sg);
                        for (int e = lane; e < TM * TN; e += SG_SIZE) {
                            const int c     = e % TN;
                            const int n_abs = n_wg + j * TN + c;
                            const float ws  = (n_abs < N) ? wc.scale_for(n_abs, k_wg) : 0.0f;
                            facc[(mb * N_PER_SG + j) * TM * TN + e] +=
                                float(spill[e]) * ws;
                        }
                        sycl::group_barrier(sg);
                    }
                }

                // Fold in the per-row activation scale and store. Writing
                // straight from SLM avoids rebuilding a joint_matrix just
                // to hand it back to the store.
                #pragma unroll
                for (int mb = 0; mb < M_PER_SG_INT; ++mb) {
                    const int m_base = m_wg + (sgid * M_PER_SG_INT + mb) * TM;
                    #pragma unroll
                    for (int j = 0; j < N_PER_SG; ++j) {
                        const int n_base = n_wg + j * TN;
                        for (int e = lane; e < TM * TN; e += SG_SIZE) {
                            const int r = e / TN, c = e % TN;
                            const int gr = m_base + r, gc = n_base + c;
                            if (gr < M && gc < N)
                                y[int64_t(gr) * N + gc] =
                                    facc[(mb * N_PER_SG + j) * TM * TN + e] * x_scale[gr];
                        }
                    }
                }
            });
    });
}

} // namespace

namespace {
// Deliberately NOT cached: the benchmark sweeps this within a single
// process, so a static would freeze the first value and every later
// measurement would silently be a duplicate of it.
int mpsg_override() {
    const char* e = std::getenv("GRIMOIRE_MPSG");
    const int n = e ? std::atoi(e) : 0;
    return (n == 1 || n == 2 || n == 4) ? n : 0;
}

int npsg_override() {
    const char* e = std::getenv("GRIMOIRE_NPSG");
    const int n = e ? std::atoi(e) : 0;
    return (n == 4 || n == 8) ? n : 0;
}

// Each work-group re-reads the whole activation matrix, so the number of
// N-groups multiplies activation traffic: at WGN=64 and N=8192 that is
// 128 passes over 16 MiB, about 2 GB of redundant reads. Widening the N
// tile halves it -- but it also doubles the accumulator count, so the
// trade is measured rather than assumed.
template <Fmt F>
sycl::event gemm_flt_dispatch(sycl::queue& q, const QuantWeight& w,
                              const sycl_bf16* x, float* y, int M,
                              const std::vector<sycl::event>& deps) {
    // At 64 subgroups, MPSG=1 already covers 512 prompt rows per WG.
    // Larger M blocking only raises register/SLM pressure and exceeded
    // the device's useful residency in the preceding sweep.
    const int mp = 1;
    const int np = npsg_override() ? npsg_override() : N_PER_SG;
    if (np == 8) {
        switch (mp) {
            case 2:  return gemm_flt<F, 2, 8>(q, w, x, y, M, deps);
            case 4:  return gemm_flt<F, 4, 8>(q, w, x, y, M, deps);
            default: return gemm_flt<F, 1, 8>(q, w, x, y, M, deps);
        }
    }
    switch (mp) {
        case 2:  return gemm_flt<F, 2, 4>(q, w, x, y, M, deps);
        case 4:  return gemm_flt<F, 4, 4>(q, w, x, y, M, deps);
        default: return gemm_flt<F, 1, 4>(q, w, x, y, M, deps);
    }
}
} // namespace

sycl::event launch_gemm_xmx(sycl::queue& q, const QuantWeight& w,
                            const sycl_bf16* x, float* y, int M,
                            const std::vector<sycl::event>& deps) {
    switch (w.fmt) {
        case Fmt::BF16:     return gemm_flt_dispatch<Fmt::BF16>(q, w, x, y, M, deps);
        case Fmt::FP8_E4M3: return gemm_flt_dispatch<Fmt::FP8_E4M3>(q, w, x, y, M, deps);
        case Fmt::FP8_E5M2: return gemm_flt_dispatch<Fmt::FP8_E5M2>(q, w, x, y, M, deps);
        case Fmt::MXFP8:    return gemm_flt_dispatch<Fmt::MXFP8>(q, w, x, y, M, deps);
        case Fmt::MXFP4:    return gemm_flt_dispatch<Fmt::MXFP4>(q, w, x, y, M, deps);
        // INT8/INT4 dequantise to bf16 in the staging loop and use the
        // same DPAS path. gemm_int() exists for the pure-integer pipeline
        // but needs pre-quantised activations; prefill does not have them.
        case Fmt::INT8:     return gemm_flt_dispatch<Fmt::INT8>(q, w, x, y, M, deps);
        case Fmt::INT4:     return gemm_flt_dispatch<Fmt::INT4>(q, w, x, y, M, deps);
    }
    return {};
}

sycl::event launch_quantize_rows_int8(sycl::queue& q, const float* x,
                                      int8_t* xq, float* scales, int M, int K,
                                      const std::vector<sycl::event>& deps) {
    constexpr int WG = 256;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> lm(WG, h);
        h.parallel_for(sycl::nd_range<1>(size_t(M) * WG, WG),
            [=](sycl::nd_item<1> it) {
                const int row = int(it.get_group(0));
                const int lane = int(it.get_local_id(0));
                float amax = 0.0f;
                for (int k = lane; k < K; k += WG)
                    amax = sycl::fmax(amax, sycl::fabs(x[int64_t(row) * K + k]));
                lm[lane] = amax;
                sycl::group_barrier(it.get_group());
                for (int stride = WG / 2; stride; stride >>= 1) {
                    if (lane < stride) lm[lane] = sycl::fmax(lm[lane], lm[lane + stride]);
                    sycl::group_barrier(it.get_group());
                }
                const float scale = lm[0] > 0.0f ? lm[0] / 127.0f : 1.0f;
                if (lane == 0) scales[row] = scale;
                for (int k = lane; k < K; k += WG) {
                    const float v = sycl::rint(x[int64_t(row) * K + k] / scale);
                    xq[int64_t(row) * K + k] = int8_t(sycl::clamp(v, -127.0f, 127.0f));
                }
            });
    });
}

sycl::event launch_gemm_xmx_int(sycl::queue& q, const QuantWeight& w,
                                const int8_t* xq, const float* scales,
                                float* y, int M,
                                const std::vector<sycl::event>& deps) {
    switch (w.fmt) {
        case Fmt::INT8: return gemm_int<Fmt::INT8>(q, w, xq, scales, y, M, deps);
        case Fmt::INT4: return gemm_int<Fmt::INT4>(q, w, xq, scales, y, M, deps);
        default: throw std::runtime_error("integer XMX requires INT8 or INT4 weights");
    }
}

sycl::event launch_gemm_b70q4(sycl::queue& q, const B70Q4View& w,
                              const int8_t* xq, const float* x_scale,
                              float* y, int M,
                              const std::vector<sycl::event>& deps) {
    const int N=w.N,K=w.K,ktiles=w.padded_k/kB70Q4K;
    const int gm=(M+WG_M_INT-1)/WG_M_INT,gn=(N+WG_N-1)/WG_N;
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        sycl::local_accessor<int8_t,1> slmA(WG_M_INT*WG_K_INT8,h);
        sycl::local_accessor<int8_t,1> slmB(WG_K_INT8*WG_N,h);
        sycl::local_accessor<int32_t,1> spill(WG_SUBGROUPS*TM*TN,h);
        sycl::local_accessor<float,1> accum(WG_SUBGROUPS*N_PER_SG*TM*TN,h);
        const B70Q4View wc=w;
        h.parallel_for(
          sycl::nd_range<2>({size_t(gm),size_t(gn)*WG_SUBGROUPS*SG_SIZE},
                            {1,size_t(WG_SUBGROUPS)*SG_SIZE}),
          [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto sg=it.get_sub_group();
            const int sgid=int(sg.get_group_id()[0]),lane=int(sg.get_local_id()[0]);
            const int lid=int(it.get_local_id(1));
            const int mb=int(it.get_group(0))*WG_M_INT;
            const int nb=int(it.get_group(1))*WG_N;
            matrix::joint_matrix<sycl::sub_group,int8_t,matrix::use::a,
                                 TM,TK_INT8,matrix::layout::row_major> fa;
            matrix::joint_matrix<sycl::sub_group,int8_t,matrix::use::b,
                                 TK_INT8,TN,matrix::layout::ext_intel_packed> fb[N_PER_SG];
            matrix::joint_matrix<sycl::sub_group,int32_t,matrix::use::accumulator,
                                 TM,TN> ci[N_PER_SG];
            int32_t* sp=spill.template get_multi_ptr<sycl::access::decorated::no>().get()
                       +sgid*TM*TN;
            float* ac=accum.template get_multi_ptr<sycl::access::decorated::no>().get()
                      +sgid*N_PER_SG*TM*TN;
            for(int e=lane;e<N_PER_SG*TM*TN;e+=SG_SIZE)ac[e]=0.0f;
            sycl::group_barrier(sg);
            for(int kb=0;kb<K;kb+=WG_K_INT8){
              #pragma unroll
              for(int j=0;j<N_PER_SG;++j)matrix::joint_matrix_fill(sg,ci[j],0);
              for(int i=lid;i<WG_M_INT*WG_K_INT8;i+=WG_SUBGROUPS*SG_SIZE){
                const int mr=i/WG_K_INT8,kc=i%WG_K_INT8;
                const int m=mb+mr,k=kb+kc;
                slmA[i]=(m<M&&k<K)?xq[int64_t(m)*K+k]:int8_t(0);
              }
              // File bytes are already grouped exactly like the VNNI B tile:
              // [k/4][n][two packed-nibble bytes]. Expand only; no transpose.
              for(int bi=lid;bi<WG_N*WG_K_INT8/2;bi+=WG_SUBGROUPS*SG_SIZE){
                const int quartet=bi/(WG_N*2);
                const int rem=bi%(WG_N*2),nc=rem/2,pair=rem&1;
                const int n=nb+nc,k=kb+quartet*4+pair*2;
                uint8_t b=0;
                if(n<N&&k<K){
                  const int nt=n/kB70Q4N,ni=n%kB70Q4N;
                  const int kt=k/kB70Q4K,ki=k%kB70Q4K;
                  const size_t tile=size_t(nt*ktiles+kt)*kB70Q4N*kB70Q4K/2;
                  b=wc.payload[tile+size_t(ki/4)*kB70Q4N*2+size_t(ni)*2+pair];
                }
                const int8_t lo=int8_t((b&8)?int(b&15)-16:int(b&15));
                const uint8_t hn=b>>4;
                const int8_t hi=int8_t((hn&8)?int(hn)-16:int(hn));
                slmB[vnni_off_int8(quartet*4+pair*2,nc)]=lo;
                slmB[vnni_off_int8(quartet*4+pair*2+1,nc)]=hi;
              }
              sycl::group_barrier(it.get_group());
              for(int kk=0;kk<WG_K_INT8;kk+=TK_INT8){
                matrix::joint_matrix_load(sg,fa,
                  slmA.template get_multi_ptr<sycl::access::decorated::no>()
                    +sgid*TM*WG_K_INT8+kk,WG_K_INT8);
                #pragma unroll
                for(int j=0;j<N_PER_SG;++j){
                  matrix::joint_matrix_load(sg,fb[j],
                    slmB.template get_multi_ptr<sycl::access::decorated::no>()
                      +vnni_off_int8(kk,j*TN),WG_N*4);
                  matrix::joint_matrix_mad(sg,ci[j],fa,fb[j],ci[j]);
                }
              }
              sycl::group_barrier(it.get_group());
              #pragma unroll
              for(int j=0;j<N_PER_SG;++j){
                matrix::joint_matrix_store(sg,ci[j],
                  sycl::address_space_cast<sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(sp),size_t(TN),matrix::layout::row_major);
                sycl::group_barrier(sg);
                for(int e=lane;e<TM*TN;e+=SG_SIZE){
                  const int n=nb+j*TN+e%TN;
                  float ws=0.0f;
                  if(n<N){
                    const int nt=n/kB70Q4N,ni=n%kB70Q4N,kt=kb/kB70Q4K;
                    ws=bf16_to_f32(wc.scales[size_t(nt*ktiles+kt)*kB70Q4N+ni]);
                  }
                  ac[j*TM*TN+e]+=float(sp[e])*ws;
                }
                sycl::group_barrier(sg);
              }
            }
            const int mbase=mb+sgid*TM;
            #pragma unroll
            for(int j=0;j<N_PER_SG;++j)for(int e=lane;e<TM*TN;e+=SG_SIZE){
              const int r=e/TN,c=e%TN,m=mbase+r,n=nb+j*TN+c;
              if(m<M&&n<N)y[int64_t(m)*N+n]=ac[j*TM*TN+e]*x_scale[m];
            }
          });
    });
}

} // namespace b70
