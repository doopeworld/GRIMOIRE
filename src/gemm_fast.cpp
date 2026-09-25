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

template <Fmt F>
sycl::event dequant_vnni(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                         const std::vector<sycl::event>& deps) {
    const int N = w.N, K = w.K;
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
                    v[t] = decode_int4(row, k, scale,
                        wc.zeros[int64_t(n) * wc.row_scales + k / kInt4Group]);
                else
                    v[t] = decode_elem<F>(row, k, scale);
            }
            sycl_bf16* o = dst + (int64_t(kp) * N + n) * 2;
            o[0] = sycl_bf16(v[0]);
            o[1] = sycl_bf16(v[1]);
        });
    });
}

sycl::event dequant_any(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                        const std::vector<sycl::event>& deps) {
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

// C[M][N] = A[M][K] * B, B packed [K/2][N][2].  M % MC2 == 0, N % NC2 == 0,
// K % KC1 == 0 -- launch_gemm_fast() guarantees it.
sycl::event gemm_bf16_vnni(sycl::queue& q, const sycl_bf16* A, const sycl_bf16* B,
                           float* C, int M, int N, int K,
                           const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const sycl::range<2> global(size_t(M / MC1), size_t(N / NC1) * SG_SIZE);
        const sycl::range<2> local(size_t(MC2 / MC1), size_t(NC2 / NC1) * SG_SIZE);
        h.parallel_for(sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto sg = it.get_sub_group();
            const int m0 = int(it.get_global_id(0)) * MC1;
            const int n0 = int(it.get_global_id(1) / SG_SIZE) * NC1;
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
            for (int k = 0; k < K; k += KC1) {
                matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::a, TM, TK_BF16,
                                     matrix::layout::row_major> a[MC1 / TM][KC1 / TK_BF16];
                matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::b, TK_BF16, TN,
                                     matrix::layout::ext_intel_packed> b[NC1 / TN][KC1 / TK_BF16];
                #pragma unroll
                for (int kk = 0; kk < KC1 / TK_BF16; ++kk) {
                    #pragma unroll
                    for (int m = 0; m < MC1 / TM; ++m)
                        matrix::joint_matrix_load(sg, a[m][kk],
                            pA + size_t(m0 + m * TM) * K + k + kk * TK_BF16, K);
                    #pragma unroll
                    for (int n = 0; n < NC1 / TN; ++n)
                        matrix::joint_matrix_load(sg, b[n][kk],
                            pB + size_t(k + kk * TK_BF16) / 2 * (size_t(N) * 2)
                               + size_t(n0 + n * TN) * 2,
                            size_t(N) * 2);
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
            auto pC = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(C);
            #pragma unroll
            for (int m = 0; m < MC1 / TM; ++m)
                #pragma unroll
                for (int n = 0; n < NC1 / TN; ++n)
                    matrix::joint_matrix_store(sg, acc[m][n],
                        pC + size_t(m0 + m * TM) * N + n0 + n * TN, N,
                        matrix::layout::row_major);
        });
    });
}

// One scratch set per queue.  Work on one queue is ordered, so reusing it
// call after call is safe; two queues never share one.
struct Scratch {
    sycl_bf16* w = nullptr; size_t w_cap = 0;   // dequantized weight, VNNI
    sycl_bf16* a = nullptr; size_t a_cap = 0;   // zero-padded tail rows of x
    float*     c = nullptr; size_t c_cap = 0;   // tail rows of y
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
                qs[i] = sycl_bf16(t < tokens ? qv[(size_t(t) * H + hq) * D + d] * scale : 0.0f);
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
                    corr[r] = sycl::isinf(m[r]) ? 0.0f : sycl::exp(m[r] - mn);
                    const float p0 = sycl::isinf(v0) ? 0.0f : sycl::exp(v0 - mn);
                    const float p1 = sycl::isinf(v1) ? 0.0f : sycl::exp(v1 - mn);
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

bool gemm_fast_supported(const QuantWeight& w, int M) {
    static const bool off = std::getenv("GRIMOIRE_NO_FAST_GEMM") != nullptr;
    return !off && M >= 32 && w.payload && w.N % NC2 == 0 && w.K % KC1 == 0 &&
           size_t(w.N) * size_t(w.K) <= kMaxScratchElems;
}

sycl::event launch_gemm_fast(sycl::queue& q, const QuantWeight& w,
                             const sycl_bf16* x, float* y, int M,
                             const std::vector<sycl::event>& deps) {
    const int N = w.N, K = w.K;
    Scratch& s = scratch_for(q);
    if (!grow(q, s.w, s.w_cap, size_t(N) * K))
        throw std::runtime_error("gemm_fast: weight scratch allocation failed");
    sycl::event e = dequant_any(q, w, s.w, deps);
    const int full = M / MC2 * MC2;
    if (full) e = gemm_bf16_vnni(q, x, s.w, y, full, N, K, {e});
    const int tail = M - full;
    if (tail) {
        // The last partial row block runs as a whole MC2-row block over a
        // zero-padded copy, and only its real rows are copied out.
        if (!grow(q, s.a, s.a_cap, size_t(MC2) * K) || !grow(q, s.c, s.c_cap, size_t(MC2) * N))
            throw std::runtime_error("gemm_fast: tail scratch allocation failed");
        e = q.memset(s.a, 0, size_t(MC2) * K * sizeof(sycl_bf16), e);
        e = q.memcpy(s.a, x + size_t(full) * K, size_t(tail) * K * sizeof(sycl_bf16), e);
        e = gemm_bf16_vnni(q, s.a, s.w, s.c, MC2, N, K, {e});
        e = q.memcpy(y + size_t(full) * N, s.c, size_t(tail) * N * sizeof(float), e);
    }
    return e;
}

} // namespace b70
