// =====================================================================
//  gemm_fast.cpp  --  the large-M prompt-processing GEMM
//
//  y[M][N] (fp32) = x[M][K] (bf16) * W[N][K]^T, W in any QuantWeight format.
//
//  gemm_flt (gemm_xmx.cpp) stages A and B through shared local memory one
//  element per work-item per K step and decodes every weight element inside
//  the K loop: ~13 TFLOP/s on the B70.  This path instead
//    1. dequantizes W ONCE per call into a bf16 VNNI scratch [K/2][N][2], and
//    2. runs a joint_matrix GEMM whose sub-groups each own a 32x64 output
//       tile and load A and B straight from global memory (2-D block loads)
//       -- no SLM staging, no barriers in the K loop.
//  tools/bench_gemm_bf16.cpp, M=6144 N=17408 K=5120 on the B70: 9.96 ms =
//  110 TFLOP/s, max relative error 1.6e-6 against an fp64 reference.
//
//  bin/grimoire and bin/grimoire-server link this file as
//  bin/libgrimoire_gemm.so, built with -cl-intel-256-GRF-per-thread.  The
//  flag is what makes it fast: the same kernel with the per-kernel
//  grf_size<256> property measured 70 TFLOP/s, and the flag cannot go on the
//  whole engine because gemm_flt and others launch 1024-thread work-groups,
//  which 256-register mode does not allow.  The JIT correctness gates compile
//  this file directly (slower there, same arithmetic).
// =====================================================================
#include "kernels.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>

namespace b70 {

namespace {

namespace matrix = sycl::ext::oneapi::experimental::matrix;

// Sub-group tile MC1 x NC1, K step KC1, work-group tile MC2 x NC2.
constexpr int MC1 = 32, NC1 = 64, KC1 = 32, MC2 = 256, NC2 = 256;
// Largest weight dequantized into scratch (elements).  Keeps an lm_head-sized
// matrix from allocating gigabytes; such shapes stay on gemm_flt.
constexpr size_t kMaxScratchElems = size_t(512) << 20;

// Work-groups are dispatched in linear order, so the order decides what
// shares L2.  GROUP_M row blocks at a time, rows fastest: the ~32 resident
// work-groups then share 4 A row-panels and 8 B column-panels instead of 1
// and 32 -- B (2 bytes per weight) used to be re-read once per row block.
// GRIMOIRE_GEMM_GROUP_M=1 restores the old column-fastest order.
// Prefetch A and B tiles d K steps ahead into L1.  MEASURED, Qwen3.8-27B
// 4088-token prefill, all 384 GEMMs: d=0 1579 ms, d=1 1377, d=2 1536,
// d=3 1641.  GRIMOIRE_GEMM_PREFETCH=d overrides (0 = off).
int gemm_prefetch() {
    static const int d = [] {
        const char* e = std::getenv("GRIMOIRE_GEMM_PREFETCH");
        const int v = e ? std::atoi(e) : 1;
        return v >= 0 && v <= 8 ? v : 0;
    }();
    return d;
}

int gemm_group_m() {
    static const int g = [] {
        const char* e = std::getenv("GRIMOIRE_GEMM_GROUP_M");
        const int v = e ? std::atoi(e) : 4;
        return v >= 1 ? v : 4;
    }();
    return g;
}

// decode_elem<MXFP4> for one payload byte (k even in the low nibble), as a
// VNNI pair of bf16: E2M1 magnitude built from bits, times the E8M0 block
// scale in fp32, rounded to bf16 (exact: at most 2 significant bits).
inline uint32_t mxfp4_pair_bf16(uint32_t byte, float sc) {
    auto one = [&](uint32_t nib) -> uint32_t {
        const uint32_t m = nib & 7u;
        const uint32_t mag = m >= 2u ? 0x3F00u + (m << 6) : (m == 1u ? 0x3F00u : 0u);
        const float f = sycl::bit_cast<float>(((nib & 8u) << 28) | (mag << 16)) * sc;
        return uint32_t(sycl::bit_cast<uint16_t>(sycl_bf16(f)));
    };
    return one(byte & 0x0Fu) | (one(byte >> 4) << 16);
}

template <Fmt F>
sycl::event dequant_vnni(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                         const std::vector<sycl::event>& deps, int ldn = 0) {
    const int N = w.N, K = w.K, LD = ldn ? ldn : w.N;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        // n fastest: consecutive work-items write consecutive VNNI pairs.
        h.parallel_for(sycl::range<2>(size_t(K / 2), size_t(N)), [=](sycl::id<2> id) {
            const int kp = int(id[0]), n = int(id[1]);
            const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
            float v[2];
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int k = 2 * kp + t;
                // Same decode as gemm_flt's staging loop, so both paths read
                // a weight to the same bf16 value.
                float scale = 1.0f;
                if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::FP8_E5M2 ||
                              F == Fmt::INT8) {
                    scale = static_cast<const float*>(wc.scales)[n];
                } else if constexpr (F == Fmt::INT4) {
                    scale = bf16_to_f32(static_cast<const bf16_t*>(wc.scales)
                        [int64_t(n) * wc.row_scales + k / kInt4Group]);
                } else if constexpr (F == Fmt::MXFP8 || F == Fmt::MXFP4) {
                    scale = e8m0_to_f32(static_cast<const uint8_t*>(wc.scales)
                        [int64_t(n) * wc.row_scales + k / kMXBlock]);
                }
                if constexpr (F == Fmt::INT4)
                    v[t] = decode_int4(row, k, scale, wc.zeros
                        ? wc.zeros[int64_t(n) * wc.row_scales + k / kInt4Group] : 0);
                else
                    v[t] = decode_elem<F>(row, k, scale);
            }
            sycl_bf16* o = dst + (int64_t(kp) * LD + n) * 2;
            o[0] = sycl_bf16(v[0]);
            o[1] = sycl_bf16(v[1]);
        });
    });
}

// E2M1 magnitude by arithmetic, not a table: a function-local table (as in
// e2m1_to_f32) lands in private memory on the GPU.  Same exact values.
inline float e2m1_alu_f(uint32_t nib) {
    const uint32_t m = nib & 7u, e = m >> 1, f = m & 1u;
    const float mag = e == 0 ? 0.5f * float(f) : (1.0f + 0.5f * float(f)) * float(1u << (e - 1));
    return (nib & 8u) ? -mag : mag;
}

// Tiled dequantize-transpose, W [N][K] -> bf16 VNNI [K/2][N][2].  One work-group
// owns a 64 (n) x 64 (k) tile: stage 1 reads each row's 32 bytes contiguously
// and decodes into SLM as [k][n]; stage 2 writes VNNI rows in 256-byte runs.
// The per-element dequant_vnni above moved 61 GB in 1.64 s (37 GB/s) for one
// 4088-token Qwen prefill -- as long as the GEMMs themselves.
// Requires N % 64 == 0 and K % 64 == 0.  Values are bit-identical to
// dequant_vnni (same decode, same fp32 multiply, same bf16 rounding).
template <Fmt F>
sycl::event dequant_vnni_tiled(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                               const std::vector<sycl::event>& deps) {
    constexpr int TN_ = 64, TK_ = 64, WGT = 256;
    const int N = w.N, K = w.K;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        sycl::local_accessor<sycl_bf16, 1> T(TK_ * TN_, h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(size_t(K / TK_), size_t(N / TN_) * WGT),
                              sycl::range<2>(1, WGT)),
            [=](sycl::nd_item<2> it) {
            const int lid = int(it.get_local_id(1));
            const int k0 = int(it.get_group(0)) * TK_, n0 = int(it.get_group(1)) * TN_;
            sycl_bf16* t = T.template get_multi_ptr<sycl::access::decorated::no>().get();
            {
                const int r = lid / 4, c = lid % 4;          // 64 rows x 4 chunks of 16
                const int n = n0 + r, kb = k0 + c * 16;
                const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                float v[16];
                if constexpr (F == Fmt::MXFP4) {
                    const float sc = e8m0_to_f32(static_cast<const uint8_t*>(wc.scales)
                        [int64_t(n) * wc.row_scales + kb / kMXBlock]);
                    const uint64_t packed = *reinterpret_cast<const uint64_t*>(row + (kb >> 1));
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t byte = uint32_t(packed >> (8 * i)) & 0xFFu;
                        v[2 * i]     = e2m1_alu_f(byte & 0x0Fu) * sc;
                        v[2 * i + 1] = e2m1_alu_f(byte >> 4) * sc;
                    }
                } else {
                    float sc = 1.0f;
                    if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::FP8_E5M2 || F == Fmt::INT8) {
                        sc = static_cast<const float*>(wc.scales)[n];
                    } else if constexpr (F == Fmt::INT4) {
                        sc = bf16_to_f32(static_cast<const bf16_t*>(wc.scales)
                            [int64_t(n) * wc.row_scales + kb / kInt4Group]);
                    } else if constexpr (F == Fmt::MXFP8) {
                        sc = e8m0_to_f32(static_cast<const uint8_t*>(wc.scales)
                            [int64_t(n) * wc.row_scales + kb / kMXBlock]);
                    }
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        if constexpr (F == Fmt::INT4)
                            v[j] = decode_int4(row, kb + j, sc, wc.zeros
                                ? wc.zeros[int64_t(n) * wc.row_scales + (kb + j) / kInt4Group] : 0);
                        else
                            v[j] = decode_elem<F>(row, kb + j, sc);
                    }
                }
                #pragma unroll
                for (int j = 0; j < 16; ++j) t[(c * 16 + j) * TN_ + r] = sycl_bf16(v[j]);
            }
            sycl::group_barrier(it.get_group());
            {
                const int pr = lid / 8, sgm = (lid % 8) * 8;  // 32 k-pairs x 8 segments of 8 n
                sycl_bf16* o = dst + (int64_t(k0 / 2 + pr) * N + n0 + sgm) * 2;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    o[2 * j]     = t[(2 * pr) * TN_ + sgm + j];
                    o[2 * j + 1] = t[(2 * pr + 1) * TN_ + sgm + j];
                }
            }
        });
    });
}

// MXFP4 W [N][K] -> bf16 VNNI [K/2][N][2], streaming.  One work-item per
// (row n, 128 k): one 64-byte payload line in, 64 dword stores out; the 16
// lanes of a sub-group are 16 consecutive n, so every store instruction
// writes one whole 64-byte line of the scratch.  No SLM, no barrier.
// Needs K % 128 == 0 and N % 256 == 0.  Same values as dequant_vnni.
// ilv_fi > 0 (SwiGLU): W = [gate; up] with FI = ilv_fi rows each, written
// interleaved -- scratch columns [64b, 64b+32) hold gate rows [32b, 32b+32)
// and [64b+32, 64b+64) the matching up rows -- so a sub-group's 64 contiguous
// columns are a gate tile and its up tile, and the SwiGLU GEMM loads B
// exactly like the plain one.
sycl::event dequant_mxfp4_stream(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                                 const std::vector<sycl::event>& deps, int ilv_fi = 0) {
    const int N = w.N, K = w.K;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(K / 128), size_t(N)),
                                         sycl::range<2>(1, 256)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const int kb = int(it.get_global_id(0)), n = int(it.get_global_id(1));
            const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes + kb * 64;
            const uint8_t* sr = static_cast<const uint8_t*>(wc.scales) +
                                int64_t(n) * wc.row_scales + kb * 4;
            const int col = ilv_fi == 0 ? n
                          : n < ilv_fi ? (n / 32) * 64 + n % 32
                                       : ((n - ilv_fi) / 32) * 64 + 32 + (n - ilv_fi) % 32;
            uint32_t* o = reinterpret_cast<uint32_t*>(dst) + int64_t(kb) * 64 * N + col;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float sc = e8m0_to_f32(sr[c]);
                const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(row + c * 16);
                #pragma unroll
                for (int d = 0; d < 4; ++d)
                    #pragma unroll
                    for (int b = 0; b < 4; ++b)
                        o[int64_t(c * 16 + d * 4 + b) * N] =
                            mxfp4_pair_bf16((v[d] >> (8 * b)) & 0xFFu, sc);
            }
        });
    });
}

sycl::event dequant_any(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                        const std::vector<sycl::event>& deps, int ldn = 0) {
    if (ldn && ldn != w.N) {        // padded pitch (small N): simple kernel, columns >= N untouched
        switch (w.fmt) {
            case Fmt::BF16:     return dequant_vnni<Fmt::BF16>(q, w, dst, deps, ldn);
            case Fmt::FP8_E4M3: return dequant_vnni<Fmt::FP8_E4M3>(q, w, dst, deps, ldn);
            case Fmt::FP8_E5M2: return dequant_vnni<Fmt::FP8_E5M2>(q, w, dst, deps, ldn);
            case Fmt::MXFP8:    return dequant_vnni<Fmt::MXFP8>(q, w, dst, deps, ldn);
            case Fmt::MXFP4:    return dequant_vnni<Fmt::MXFP4>(q, w, dst, deps, ldn);
            case Fmt::INT8:     return dequant_vnni<Fmt::INT8>(q, w, dst, deps, ldn);
            case Fmt::INT4:     return dequant_vnni<Fmt::INT4>(q, w, dst, deps, ldn);
        }
        return {};
    }
    static const bool no_stream = std::getenv("GRIMOIRE_DEQUANT_TILED") != nullptr;
    if (!no_stream && w.fmt == Fmt::MXFP4 && w.K % 128 == 0 && w.N % 256 == 0)
        return dequant_mxfp4_stream(q, w, dst, deps);
    if (w.N % 64 == 0 && w.K % 64 == 0) {
        switch (w.fmt) {
            case Fmt::BF16:     return dequant_vnni_tiled<Fmt::BF16>(q, w, dst, deps);
            case Fmt::FP8_E4M3: return dequant_vnni_tiled<Fmt::FP8_E4M3>(q, w, dst, deps);
            case Fmt::FP8_E5M2: return dequant_vnni_tiled<Fmt::FP8_E5M2>(q, w, dst, deps);
            case Fmt::MXFP8:    return dequant_vnni_tiled<Fmt::MXFP8>(q, w, dst, deps);
            case Fmt::MXFP4:    return dequant_vnni_tiled<Fmt::MXFP4>(q, w, dst, deps);
            case Fmt::INT8:     return dequant_vnni_tiled<Fmt::INT8>(q, w, dst, deps);
            case Fmt::INT4:     return dequant_vnni_tiled<Fmt::INT4>(q, w, dst, deps);
        }
    }
    switch (w.fmt) {
        case Fmt::BF16:     return dequant_vnni<Fmt::BF16>(q, w, dst, deps);
        case Fmt::FP8_E4M3: return dequant_vnni<Fmt::FP8_E4M3>(q, w, dst, deps);
        case Fmt::FP8_E5M2: return dequant_vnni<Fmt::FP8_E5M2>(q, w, dst, deps);
        case Fmt::MXFP8:    return dequant_vnni<Fmt::MXFP8>(q, w, dst, deps);
        case Fmt::MXFP4:    return dequant_vnni<Fmt::MXFP4>(q, w, dst, deps);
        case Fmt::INT8:     return dequant_vnni<Fmt::INT8>(q, w, dst, deps);
        case Fmt::INT4:     return dequant_vnni<Fmt::INT4>(q, w, dst, deps);
    }
    return {};
}

// C = A * B with B packed [K/2][N][2] (the dequantized scratch).
// N % NC2 == 0, K % KC1 == 0, any M >= 1: the last, partial row block runs in
// the same launch with bounds-checked A loads (rows >= M read as zero) and
// clipped stores -- no padded copy (that tail path cost 164 ms of a
// 4088-token Qwen prefill).  Work-group order: gemm_group_m().
//   EPI 0: C fp32 [M][N].
//   EPI 1: SwiGLU.  B's columns are [gate | up] (N = 2*FI); each sub-group
//          computes a gate tile and the matching up tile and writes
//          h bf16 [M][FI] = silu(gate) * up -- launch_swiglu_batched's
//          formula on the same fp32 values, rounded as launch_f32_to_bf16.
template <int EPI>
sycl::event gemm_bf16_vnni(sycl::queue& q, const sycl_bf16* A, const sycl_bf16* B,
                           void* out, int M, int N, int K,
                           const std::vector<sycl::event>& deps, int ldb = 0) {
    constexpr int SGM = MC2 / MC1, SGN = NC2 / NC1, WG = SGM * SGN * SG_SIZE;
    const int LDB = ldb ? ldb : N;               // B's column pitch: N rounded up to NC2
    const int nMB = (M + MC2 - 1) / MC2, nNB = LDB / NC2, FI = N / 2, GM = gemm_group_m();
    const int PD = gemm_prefetch();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(nMB) * nNB * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            auto sg = it.get_sub_group();
            const int L = int(it.get_group(0)), per = GM * nNB;
            const int mb0 = (L / per) * GM, gm = sycl::min(nMB - mb0, GM);
            const int mb = mb0 + (L % per) % gm, nb = (L % per) / gm;
            const int s = int(it.get_local_id(0)) / SG_SIZE, sm = s / SGN, sn = s % SGN;
            const int m0 = mb * MC2 + sm * MC1;
            if (m0 >= M) return;
            // B columns of fragment n.  For EPI 1 the scratch is gate/up
            // interleaved (dequant_mxfp4_stream ilv_fi), so fragments 0-1 are
            // gate and 2-3 the matching up columns either way.
            auto bcol = [&](int n) { return nb * NC2 + sn * NC1 + n * TN; };
            auto pA = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(A);
            auto pB = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(B);
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN>
                acc[MC1 / TM][NC1 / TN];
            #pragma unroll
            for (int m = 0; m < MC1 / TM; ++m)
                #pragma unroll
                for (int n = 0; n < NC1 / TN; ++n)
                    matrix::joint_matrix_fill(sg, acc[m][n], 0.0f);
            // A always goes through the bounds-checked 2-D block load: the
            // hardware message is the same one either way, and one loop copy
            // (not a checked + unchecked pair) keeps the code the old kernel's.
            const bool a_full = m0 + MC1 <= M;       // never prefetch past row M
            for (int k = 0; k < K; k += KC1) {
                if (PD > 0 && k + PD * KC1 < K) {
                    namespace syclex = sycl::ext::oneapi::experimental;
                    const int kp = k + PD * KC1;
                    if (a_full)
                        ix::joint_matrix_prefetch<MC1, KC1>(sg, A + size_t(m0) * K + kp, K,
                            matrix::layout::row_major,
                            syclex::properties{syclex::prefetch_hint_L1});
                    #pragma unroll
                    for (int c = 0; c < NC1 * 2; c += 32)
                        ix::joint_matrix_prefetch<KC1 / 2, 32>(sg,
                            B + size_t(kp / 2) * (size_t(LDB) * 2) + size_t(bcol(0)) * 2 + c,
                            size_t(LDB) * 2, matrix::layout::row_major,
                            syclex::properties{syclex::prefetch_hint_L1});
                }
                matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::a, TM, TK_BF16,
                                     matrix::layout::row_major> a[MC1 / TM][KC1 / TK_BF16];
                matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::b, TK_BF16, TN,
                                     matrix::layout::ext_intel_packed> b[NC1 / TN][KC1 / TK_BF16];
                #pragma unroll
                for (int kk = 0; kk < KC1 / TK_BF16; ++kk) {
                    #pragma unroll
                    for (int m = 0; m < MC1 / TM; ++m)
                        ix::joint_matrix_load_checked(sg, a[m][kk], pA, size_t(K), size_t(M),
                            size_t(K), size_t(m0 + m * TM), size_t(k + kk * TK_BF16));
                    #pragma unroll
                    for (int n = 0; n < NC1 / TN; ++n)
                        matrix::joint_matrix_load(sg, b[n][kk],
                            pB + size_t(k + kk * TK_BF16) / 2 * (size_t(LDB) * 2)
                               + size_t(bcol(n)) * 2,
                            size_t(LDB) * 2);
                }
                #pragma unroll
                for (int kk = 0; kk < KC1 / TK_BF16; ++kk)
                    #pragma unroll
                    for (int m = 0; m < MC1 / TM; ++m)
                        #pragma unroll
                        for (int n = 0; n < NC1 / TN; ++n)
                            matrix::joint_matrix_mad(sg, acc[m][n], a[m][kk], b[n][kk],
                                                     acc[m][n]);
            }
            if constexpr (EPI == 0) {
                auto pC = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(
                    static_cast<float*>(out));
                #pragma unroll
                for (int m = 0; m < MC1 / TM; ++m)
                    #pragma unroll
                    for (int n = 0; n < NC1 / TN; ++n)
                        ix::joint_matrix_store_checked(sg, acc[m][n], pC, size_t(N),
                            matrix::layout::row_major, size_t(M), size_t(N),
                            size_t(m0 + m * TM), size_t(bcol(n)));
            } else {
                // silu(gate) * up in registers, rounded to a bf16 tile, one
                // bounds-checked 2-D block store.  (Per-element coordinate
                // stores cost 225 ms of epilogue -- twice the SwiGLU kernel.)
                auto pH = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(
                    static_cast<sycl_bf16*>(out));
                #pragma unroll
                for (int m = 0; m < MC1 / TM; ++m)
                    #pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        matrix::joint_matrix_apply(sg, acc[m][n], acc[m][n + 2],
                            [](float& g, float& u) {
                                g = sycl::native::divide(g, 1.0f + sycl::native::exp(-g)) * u; });
                        matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::accumulator,
                                             TM, TN> hb;
                        matrix::joint_matrix_copy(sg, acc[m][n], hb);
                        ix::joint_matrix_store_checked(sg, hb, pH, size_t(FI),
                            matrix::layout::row_major, size_t(M), size_t(FI),
                            size_t(m0 + m * TM), size_t(nb * (NC2 / 2) + sn * (NC1 / 2) + n * TN));
                    }
            }
        });
    });
}

// One scratch set per queue.  Work on one queue is ordered, so reusing it
// call after call is safe; two queues never share one.
// GRIMOIRE_FAST_GEMM_TIMING=1: wait after each stage and total the time per
// stage; printed at exit.  Perturbs the pipeline -- diagnosis only.
struct StageTimes {
    double dq = 0, main = 0; long calls = 0;
    ~StageTimes() {
        if (calls) std::fprintf(stderr, "  fast GEMM: %ld calls  dequant %.1f ms  GEMM %.1f ms\n",
                                calls, dq, main);
    }
};
StageTimes& stage_times() { static StageTimes t; return t; }
bool stage_timing() { static const bool on = std::getenv("GRIMOIRE_FAST_GEMM_TIMING") != nullptr; return on; }

struct Scratch {
    sycl_bf16* w = nullptr; size_t w_cap = 0;   // dequantized weight, VNNI
};

template <class T>
T* grow(sycl::queue& q, T*& p, size_t& cap, size_t need) {
    if (need <= cap) return p;
    q.wait();                       // nothing in flight may still read the old one
    if (p) sycl::free(p, q);
    p = sycl::malloc_device<T>(need, q);
    cap = p ? need : 0;
    return p;
}

Scratch& scratch_for(sycl::queue& q) {
    static std::mutex mu;
    static std::map<const sycl::queue*, Scratch> all;
    std::lock_guard<std::mutex> lk(mu);
    return all[&q];
}

} // namespace

// ---------------------------------------------------------------------
// Causal prefill attention on the matrix unit.
//
// launch_flash_prefill (prefill.cpp) is scalar SIMT: one sub-group per
// query, a serial FMA loop over head_dim per key, e4m3 K/V converted into
// SLM element by element -- 329 ms per Qwen3.8-27B layer at 5987 tokens.
// Here a pre-pass converts the layer's e4m3 cache ONCE into the packed bf16
// layouts the matrix unit reads (keys >= kend zeroed):
//   Kp [KVH][D/2][Sp][2]   B operand of S = Q K^T  (k = d,   n = key)
//   Vp [KVH][Sp/2][D][2]   B operand of O = P V    (k = key, n = d)
// and every sub-group then owns 8 query rows end to end: Q in its own SLM
// slice, K/V fragments straight from global memory, the online softmax on
// its own S/P slice, no work-group barrier in the key loop.  Q and P are
// rounded to bf16 (K and V are exact: e4m3 fits in bf16).
// tools/bench_flash_prefill.cpp: 24.7 ms per Qwen layer (17.9 TFLOP/s),
// worst relative error 1.0e-2 against fp64 on ragged, resumed (start > 0)
// and full-size cases.
// ---------------------------------------------------------------------
namespace {

constexpr int FA_NSG = 8, FA_RB = TM, FA_BQ = FA_NSG * FA_RB, FA_BK = 32;

struct FlashScratch {
    sycl_bf16* lut = nullptr;                   // e4m3 byte -> bf16
    sycl_bf16* kp = nullptr; size_t kp_cap = 0;
    sycl_bf16* vp = nullptr; size_t vp_cap = 0;
};

FlashScratch& flash_scratch_for(sycl::queue& q) {
    static std::mutex mu;
    static std::map<const sycl::queue*, FlashScratch> all;
    std::lock_guard<std::mutex> lk(mu);
    FlashScratch& s = all[&q];
    if (!s.lut) {
        s.lut = sycl::malloc_device<sycl_bf16>(256, q);
        if (!s.lut) throw std::runtime_error("flash_fast: lut allocation failed");
        sycl_bf16* l = s.lut;
        q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
            l[i] = sycl_bf16(e4m3_to_f32(uint8_t(i[0])));
        }).wait();
    }
    return s;
}

template <int D>
sycl::event flash_fast_impl(sycl::queue& q, const float* qv, const uint8_t* kc,
                            const uint8_t* vc, float* out, int tokens, int start, int H,
                            int KVH, int seq_cap, float scale,
                            const std::vector<sycl::event>& deps) {
    constexpr int NF = D / TN, WG = FA_NSG * SG_SIZE;
    FlashScratch& fs = flash_scratch_for(q);
    const int kend = start + tokens, Sp = (kend + FA_BK - 1) / FA_BK * FA_BK;
    const size_t elems = size_t(KVH) * Sp * D;
    if (!grow(q, fs.kp, fs.kp_cap, elems) || !grow(q, fs.vp, fs.vp_cap, elems))
        throw std::runtime_error("flash_fast: K/V scratch allocation failed");
    sycl_bf16* Kp = fs.kp; sycl_bf16* Vp = fs.vp; const sycl_bf16* lut = fs.lut;
    sycl::event e = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<3>(size_t(KVH), size_t(D), size_t(Sp)), [=](sycl::id<3> id) {
            const int hh = int(id[0]), d = int(id[1]), s = int(id[2]);       // key fastest
            Kp[((size_t(hh) * (D / 2) + d / 2) * Sp + s) * 2 + (d & 1)] =
                s < kend ? lut[kc[(size_t(hh) * D + d) * seq_cap + s]] : sycl_bf16(0.0f);
        });
    });
    e = q.submit([&](sycl::handler& h) {
        h.depends_on(e);
        h.parallel_for(sycl::range<3>(size_t(KVH), size_t(Sp), size_t(D)), [=](sycl::id<3> id) {
            const int hh = int(id[0]), s = int(id[1]), d = int(id[2]);       // d fastest
            Vp[((size_t(hh) * (Sp / 2) + s / 2) * D + d) * 2 + (s & 1)] =
                s < kend ? lut[vc[(size_t(hh) * seq_cap + s) * D + d]] : sycl_bf16(0.0f);
        });
    });
    const int qtiles = (tokens + FA_BQ - 1) / FA_BQ;
    // Scores in log2 units: Q carries scale*log2(e), so the softmax is the
    // hardware exp2.  The accurate sycl::exp is a long instruction sequence,
    // three per score pair, and made the softmax -- not the matrix unit --
    // the bottleneck.  P is rounded to bf16 before PV anyway.
    const float qscale = scale * 1.4426950408889634f;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(e);
        sycl::local_accessor<sycl_bf16, 1> Qs(FA_NSG * FA_RB * D, h);
        sycl::local_accessor<float, 1> Ss(FA_NSG * FA_RB * FA_BK, h);
        sycl::local_accessor<sycl_bf16, 1> Ps(FA_NSG * FA_RB * FA_BK, h);
        h.parallel_for(sycl::nd_range<1>(size_t(qtiles) * H * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            constexpr auto NO = sycl::access::decorated::no;
            constexpr int RB = FA_RB, BK = FA_BK;
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]), lane = int(sg.get_local_id()[0]);
            const int g = int(it.get_group(0)), qt = g / H, hq = g % H, kvh = hq / (H / KVH);
            const int r0 = qt * FA_BQ + sgid * RB;
            if (r0 >= tokens) return;                   // uniform per sub-group; no WG barriers below
            auto qs_mp = Qs.template get_multi_ptr<NO>() + sgid * RB * D;
            auto ss_mp = Ss.template get_multi_ptr<NO>() + sgid * RB * BK;
            auto ps_mp = Ps.template get_multi_ptr<NO>() + sgid * RB * BK;
            sycl_bf16* qs = qs_mp.get(); float* ss = ss_mp.get(); sycl_bf16* ps = ps_mp.get();
            for (int i = lane; i < RB * D; i += SG_SIZE) {
                const int r = i / D, d = i % D, t = r0 + r;
                qs[i] = sycl_bf16(t < tokens ? qv[(size_t(t) * H + hq) * D + d] * qscale : 0.0f);
            }
            sycl::group_barrier(sg);
            auto kg = sycl::address_space_cast<sycl::access::address_space::global_space, NO>(
                Kp + size_t(kvh) * (D / 2) * Sp * 2);
            auto vg = sycl::address_space_cast<sycl::access::address_space::global_space, NO>(
                Vp + size_t(kvh) * (Sp / 2) * D * 2);
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN> o[NF];
            #pragma unroll
            for (int n = 0; n < NF; ++n) matrix::joint_matrix_fill(sg, o[n], 0.0f);
            float m[RB], l[RB];
            #pragma unroll
            for (int r = 0; r < RB; ++r) { m[r] = -std::numeric_limits<float>::infinity(); l[r] = 0.0f; }
            const int last_row = sycl::min(tokens - 1, r0 + RB - 1);
            const int kmax = start + last_row + 1;
            for (int s0 = 0; s0 < kmax; s0 += BK) {
                matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN>
                    sa[BK / TN];
                #pragma unroll
                for (int c = 0; c < BK / TN; ++c) matrix::joint_matrix_fill(sg, sa[c], 0.0f);
                #pragma unroll 4
                for (int k = 0; k < D; k += TK_BF16) {
                    matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::a, TM, TK_BF16,
                                         matrix::layout::row_major> a;
                    matrix::joint_matrix_load(sg, a, qs_mp + k, D);
                    #pragma unroll
                    for (int c = 0; c < BK / TN; ++c) {
                        matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::b, TK_BF16,
                                             TN, matrix::layout::ext_intel_packed> b;
                        matrix::joint_matrix_load(sg, b,
                            kg + size_t(k / 2) * Sp * 2 + size_t(s0 + c * TN) * 2, size_t(Sp) * 2);
                        matrix::joint_matrix_mad(sg, sa[c], a, b, sa[c]);
                    }
                }
                #pragma unroll
                for (int c = 0; c < BK / TN; ++c)
                    matrix::joint_matrix_store(sg, sa[c], ss_mp + c * TN, BK,
                                               matrix::layout::row_major);
                sycl::group_barrier(sg);
                float corr[RB];
                #pragma unroll
                for (int r = 0; r < RB; ++r) {
                    const int t = r0 + r, qpos = start + t;
                    const float v0 = (t < tokens && s0 + lane <= qpos) ? ss[r * BK + lane]
                                     : -std::numeric_limits<float>::infinity();
                    const float v1 = (t < tokens && s0 + lane + SG_SIZE <= qpos)
                                     ? ss[r * BK + lane + SG_SIZE]
                                     : -std::numeric_limits<float>::infinity();
                    const float bm = sycl::reduce_over_group(sg, sycl::fmax(v0, v1),
                                                             sycl::maximum<float>());
                    const float mn = sycl::fmax(m[r], bm);
                    corr[r] = sycl::isinf(m[r]) ? 0.0f : sycl::native::exp2(m[r] - mn);
                    const float p0 = sycl::isinf(v0) ? 0.0f : sycl::native::exp2(v0 - mn);
                    const float p1 = sycl::isinf(v1) ? 0.0f : sycl::native::exp2(v1 - mn);
                    ps[r * BK + lane] = sycl_bf16(p0);
                    ps[r * BK + lane + SG_SIZE] = sycl_bf16(p1);
                    l[r] = l[r] * corr[r] +
                           sycl::reduce_over_group(sg, p0 + p1, sycl::plus<float>());
                    m[r] = mn;
                }
                sycl::group_barrier(sg);
                #pragma unroll
                for (int n = 0; n < NF; ++n)
                    ix::joint_matrix_apply(sg, o[n],
                        [&](float& x, size_t row, size_t) { x *= corr[row]; });
                #pragma unroll
                for (int kk = 0; kk < BK; kk += TK_BF16) {
                    matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::a, TM, TK_BF16,
                                         matrix::layout::row_major> a;
                    matrix::joint_matrix_load(sg, a, ps_mp + kk, BK);
                    #pragma unroll
                    for (int n = 0; n < NF; ++n) {
                        matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::b, TK_BF16,
                                             TN, matrix::layout::ext_intel_packed> b;
                        matrix::joint_matrix_load(sg, b,
                            vg + size_t((s0 + kk) / 2) * D * 2 + size_t(n * TN) * 2,
                            size_t(D) * 2);
                        matrix::joint_matrix_mad(sg, o[n], a, b, o[n]);
                    }
                }
                sycl::group_barrier(sg);
            }
            float inv[RB];
            #pragma unroll
            for (int r = 0; r < RB; ++r) inv[r] = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
            #pragma unroll
            for (int n = 0; n < NF; ++n)
                ix::joint_matrix_apply(sg, o[n], [&](float& x, size_t row, size_t) { x *= inv[row]; });
            auto og = sycl::address_space_cast<sycl::access::address_space::global_space, NO>(out);
            if (r0 + RB <= tokens) {
                #pragma unroll
                for (int n = 0; n < NF; ++n)
                    matrix::joint_matrix_store(sg, o[n], og + (size_t(r0) * H + hq) * D + n * TN,
                                               size_t(H) * D, matrix::layout::row_major);
            } else {
                for (int n = 0; n < NF; ++n) {          // ragged last tile: through the S slice
                    matrix::joint_matrix_store(sg, o[n], ss_mp, TN, matrix::layout::row_major);
                    sycl::group_barrier(sg);
                    for (int x = lane; x < RB * TN; x += SG_SIZE) {
                        const int r = x / TN, c = x % TN, t = r0 + r;
                        if (t < tokens) out[(size_t(t) * H + hq) * D + n * TN + c] = ss[x];
                    }
                    sycl::group_barrier(sg);
                }
            }
        });
    });
}

} // namespace

bool flash_fast_supported(int head_dim, int num_heads, int num_kv_heads) {
    static const bool off = std::getenv("GRIMOIRE_NO_FAST_FLASH") != nullptr;
    return !off && (head_dim == 128 || head_dim == 256) && num_kv_heads > 0 &&
           num_heads % num_kv_heads == 0;
}

sycl::event launch_flash_prefill_fast(sycl::queue& q, const float* qv, const uint8_t* k_cache,
                                      const uint8_t* v_cache, float* out, int tokens,
                                      int start_pos, int num_heads, int num_kv_heads,
                                      int head_dim, int seq_cap, float softmax_scale,
                                      const std::vector<sycl::event>& deps) {
    return head_dim == 128
        ? flash_fast_impl<128>(q, qv, k_cache, v_cache, out, tokens, start_pos, num_heads,
                               num_kv_heads, seq_cap, softmax_scale, deps)
        : flash_fast_impl<256>(q, qv, k_cache, v_cache, out, tokens, start_pos, num_heads,
                               num_kv_heads, seq_cap, softmax_scale, deps);
}

namespace {

// ---------------------------------------------------------------------
// MXFP4 GEMM with the dequantize fused into the K loop.
//
// The generic path (launch_gemm_fast below) dequantizes W into a bf16
// scratch and then runs gemm_bf16_vnni: the dequant pass alone cost 327 ms
// of a 4088-token Qwen3.8-27B prefill, and the GEMM re-reads 2 bytes per
// weight per row block.  Here every work-group decodes the 256 columns x 32 k
// of W it needs for the NEXT K step from the MXFP4 payload into one half of
// a double-buffered SLM tile while its sub-groups run the matrix unit on the
// other half (one barrier per K step).  A still loads straight from global
// memory.  Work-groups run GROUP_M row blocks at a time, rows fastest, so
// the ~32 resident work-groups share 4 A row-panels and 8 weight panels.
// The last, partial row block is part of the same launch: its A loads are
// bounds-checked (rows >= M read as zero) and its stores clipped -- no
// padded copy.
//   EPI 0: y fp32 [M][N].
//   EPI 1: SwiGLU.  W = [gate; up] (N = 2*FI); each sub-group holds a gate
//          tile and the matching up tile and writes
//          h bf16 [M][FI] = silu(x W_gate^T) * (x W_up^T).
// Same bf16 weight values, same K order into the same fp32 accumulators as
// dequantize-then-GEMM, and the same SwiGLU formula as
// launch_swiglu_batched, so the results are the same.
// ---------------------------------------------------------------------

template <int EPI>
sycl::event gemm_mxfp4_fused(sycl::queue& q, const QuantWeight& w, const sycl_bf16* A,
                             void* out, int M, const std::vector<sycl::event>& deps) {
    constexpr int SGM = MC2 / MC1, SGN = NC2 / NC1, WG = SGM * SGN * SG_SIZE;
    constexpr int KP = KC1 / 2;                  // k-pairs per K step
    constexpr int BT = KP * NC2;                 // dwords (bf16 pairs) per SLM buffer
    static_assert(WG == 2 * NC2, "decode: two work-items per column (16 k each)");
    static_assert(KC1 == 2 * 16 && KC1 <= kMXBlock, "decode: one E8M0 scale per k half");
    const int N = w.N, K = w.K, FI = N / 2;
    const int nMB = (M + MC2 - 1) / MC2, nNB = N / NC2, GM = gemm_group_m();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        sycl::local_accessor<uint32_t, 1> Bs(2 * BT, h);
        h.parallel_for(sycl::nd_range<1>(size_t(nMB) * nNB * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            auto sg = it.get_sub_group();
            const int lid = int(it.get_local_id(0));
            const int L = int(it.get_group(0)), per = GM * nNB;
            const int mb0 = (L / per) * GM, gm = sycl::min(nMB - mb0, GM);
            const int mb = mb0 + (L % per) % gm, nb = (L % per) / gm;
            const int s = lid / SG_SIZE, sm = s / SGN, sn = s % SGN;
            const int m0 = mb * MC2 + sm * MC1;
            const bool active = m0 < M, full = m0 + MC1 <= M;

            // Decode role: tile column dn, k half dh (16 k = 8 payload bytes).
            const int dn = lid % NC2, dh = lid / NC2;
            const int wrow = EPI == 1
                ? (dn < NC2 / 2 ? nb * (NC2 / 2) + dn : FI + nb * (NC2 / 2) + dn - NC2 / 2)
                : nb * NC2 + dn;
            const uint8_t* prow = wc.payload + int64_t(wrow) * wc.row_bytes;
            const uint8_t* srow = static_cast<const uint8_t*>(wc.scales) +
                                  int64_t(wrow) * wc.row_scales;
            uint32_t* bs = Bs.template get_multi_ptr<sycl::access::decorated::no>().get();
            uint64_t pk = 0; uint8_t sb = 0;
            auto fetch = [&](int k) {
                const int kb = k + dh * 16;
                pk = *reinterpret_cast<const uint64_t*>(prow + (kb >> 1));
                sb = srow[kb / kMXBlock];
            };
            auto decode = [&](int buf) {
                const float sc = e8m0_to_f32(sb);
                uint32_t* d = bs + buf * BT + dh * (KP / 2) * NC2 + dn;
                #pragma unroll
                for (int p = 0; p < 8; ++p)
                    d[p * NC2] = mxfp4_pair_bf16(uint32_t(pk >> (8 * p)) & 0xFFu, sc);
            };
            // SLM column of B fragment n for this sub-group
            auto bcol = [&](int n) {
                return EPI == 1 ? (n < 2 ? sn * (NC1 / 2) + n * TN
                                         : NC2 / 2 + sn * (NC1 / 2) + (n - 2) * TN)
                                : sn * NC1 + n * TN;
            };
            auto pA = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(A);
            auto pBs = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                sycl::access::decorated::no>(
                reinterpret_cast<sycl_bf16*>(bs));
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN>
                acc[MC1 / TM][NC1 / TN];
            #pragma unroll
            for (int m = 0; m < MC1 / TM; ++m)
                #pragma unroll
                for (int n = 0; n < NC1 / TN; ++n)
                    matrix::joint_matrix_fill(sg, acc[m][n], 0.0f);
            fetch(0);
            decode(0);
            sycl::group_barrier(it.get_group());
            for (int k = 0, step = 0; k < K; k += KC1, ++step) {
                const int cur = step & 1;
                const bool more = k + KC1 < K;
                if (more) fetch(k + KC1);            // in flight during the MADs
                if (active) {
                    matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::a, TM, TK_BF16,
                                         matrix::layout::row_major> a[MC1 / TM][KC1 / TK_BF16];
                    matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::b, TK_BF16, TN,
                                         matrix::layout::ext_intel_packed> b[NC1 / TN][KC1 / TK_BF16];
                    #pragma unroll
                    for (int kk = 0; kk < KC1 / TK_BF16; ++kk) {
                        #pragma unroll
                        for (int m = 0; m < MC1 / TM; ++m) {
                            if (full)
                                matrix::joint_matrix_load(sg, a[m][kk],
                                    pA + size_t(m0 + m * TM) * K + k + kk * TK_BF16, K);
                            else
                                ix::joint_matrix_load_checked(sg, a[m][kk], pA, size_t(K),
                                    size_t(M), size_t(K), size_t(m0 + m * TM),
                                    size_t(k + kk * TK_BF16));
                        }
                        #pragma unroll
                        for (int n = 0; n < NC1 / TN; ++n)
                            matrix::joint_matrix_load(sg, b[n][kk],
                                pBs + size_t(cur * BT + kk * (TK_BF16 / 2) * NC2 + bcol(n)) * 2,
                                size_t(NC2) * 2);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < KC1 / TK_BF16; ++kk)
                        #pragma unroll
                        for (int m = 0; m < MC1 / TM; ++m)
                            #pragma unroll
                            for (int n = 0; n < NC1 / TN; ++n)
                                matrix::joint_matrix_mad(sg, acc[m][n], a[m][kk], b[n][kk],
                                                         acc[m][n]);
                }
                if (more) decode(cur ^ 1);
                sycl::group_barrier(it.get_group());
            }
            if (!active) return;
            if constexpr (EPI == 0) {
                auto pC = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(
                    static_cast<float*>(out));
                #pragma unroll
                for (int m = 0; m < MC1 / TM; ++m)
                    #pragma unroll
                    for (int n = 0; n < NC1 / TN; ++n) {
                        const int col = nb * NC2 + sn * NC1 + n * TN;
                        if (full)
                            matrix::joint_matrix_store(sg, acc[m][n],
                                pC + size_t(m0 + m * TM) * N + col, N, matrix::layout::row_major);
                        else
                            ix::joint_matrix_store_checked(sg, acc[m][n], pC, size_t(N),
                                matrix::layout::row_major, size_t(M), size_t(N),
                                size_t(m0 + m * TM), size_t(col));
                    }
            } else {
                sycl_bf16* H = static_cast<sycl_bf16*>(out);
                #pragma unroll
                for (int m = 0; m < MC1 / TM; ++m)
                    #pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        matrix::joint_matrix_apply(sg, acc[m][n], acc[m][n + 2],
                            [](float& g, float& u) {
                                g = sycl::native::divide(g, 1.0f + sycl::native::exp(-g)) * u; });
                        const int r0 = m0 + m * TM;
                        const int col = nb * (NC2 / 2) + sn * (NC1 / 2) + n * TN;
                        ix::joint_matrix_apply(sg, acc[m][n], [=](float& x, size_t r, size_t c) {
                            if (r0 + int(r) < M)
                                H[size_t(r0 + int(r)) * FI + col + c] = sycl_bf16(x);
                        });
                    }
            }
        });
    });
}

// Opt-in (GRIMOIRE_FAST_GEMM_FUSED=1).  MEASURED 2026-09-25, Qwen3.8-27B,
// 4088 tokens: correct (identical text) but 1.65x SLOWER than dequantize +
// gemm_bf16_vnni (prefill 4.49 s vs 3.14 s): B fragments from SLM plus a
// barrier per K step cost more than the dequant pass saves.  Kept for tuning
// (taller sub-group tiles would halve the SLM traffic per DPAS).
bool fused_mxfp4_ok(const QuantWeight& w) {
    static const bool on = std::getenv("GRIMOIRE_FAST_GEMM_FUSED") != nullptr;
    return on && w.fmt == Fmt::MXFP4 && w.payload && w.scales && w.N % NC2 == 0 &&
           w.K % KC1 == 0;
}

} // namespace

bool gemm_fast_supported(const QuantWeight& w, int M) {
    static const bool off = std::getenv("GRIMOIRE_NO_FAST_GEMM") != nullptr;
    const size_t np = size_t(w.N + NC2 - 1) / NC2 * NC2;
    return !off && M >= 32 && w.payload && w.N % 16 == 0 && w.K % KC1 == 0 &&
           np * size_t(w.K) <= kMaxScratchElems;
}

namespace {

// Dequantize W once into the scratch, then one GEMM launch over all M rows.
template <int EPI>
sycl::event gemm_fast_run(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, void* out,
                          int M, const std::vector<sycl::event>& deps) {
    const int N = w.N, K = w.K;
    const bool timed = stage_timing();
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto lap = [&](double& acc, std::chrono::steady_clock::time_point& t0) {
        q.wait();
        const auto t1 = now();
        acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
    };
    if (timed) q.wait();             // earlier work must not count here
    auto t0 = now();
    if (fused_mxfp4_ok(w)) {                 // N % NC2 == 0 checked there
        sycl::event e = gemm_mxfp4_fused<EPI>(q, w, x, out, M, deps);
        if (timed) { lap(stage_times().main, t0); ++stage_times().calls; }
        return e;
    }
    // N not a multiple of NC2 (DeltaNet's a|b gate projection, N = 96): the
    // scratch gets pitch Np with zeroed extra columns and the stores clip at
    // N.  It used to fall back to gemm_flt: 41 ms per Qwen prefill.
    const int Np = (N + NC2 - 1) / NC2 * NC2;
    Scratch& s = scratch_for(q);
    if (!grow(q, s.w, s.w_cap, size_t(Np) * K))
        throw std::runtime_error("gemm_fast: weight scratch allocation failed");
    sycl::event e;
    if (Np != N) {
        e = q.memset(s.w, 0, size_t(Np) * K * sizeof(sycl_bf16), deps);
        e = dequant_any(q, w, s.w, {e}, Np);
    } else if (EPI == 1) {
        e = dequant_mxfp4_stream(q, w, s.w, deps, N / 2);
    } else {
        e = dequant_any(q, w, s.w, deps);
    }
    if (timed) { lap(stage_times().dq, t0); ++stage_times().calls; }
    e = gemm_bf16_vnni<EPI>(q, x, s.w, out, M, N, K, {e}, Np);
    if (timed) lap(stage_times().main, t0);
    return e;
}

} // namespace

sycl::event launch_gemm_fast(sycl::queue& q, const QuantWeight& w,
                             const sycl_bf16* x, float* y, int M,
                             const std::vector<sycl::event>& deps) {
    return gemm_fast_run<0>(q, w, x, y, M, deps);
}

bool gemm_fast_swiglu_supported(const QuantWeight& w, int M) {
    static const bool off = std::getenv("GRIMOIRE_NO_FUSED_SWIGLU") != nullptr;
    return !off && gemm_fast_supported(w, M) && w.fmt == Fmt::MXFP4 && w.K % 128 == 0 &&
           w.N % NC2 == 0 && (w.N / 2) % 32 == 0;
}

sycl::event launch_gemm_fast_swiglu(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x,
                                    sycl_bf16* h, int M, const std::vector<sycl::event>& deps) {
    return gemm_fast_run<1>(q, w, x, h, M, deps);
}

} // namespace b70
