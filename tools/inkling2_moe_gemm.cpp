// SPDX-License-Identifier: GPL-3.0-or-later
// Inkling 2 Mini prompt experts: ONE grouped GEMM launch per projection for every expert of a layer visit.
// Adapted from Grimoire's gemm_mxfp4_fused_grouped (src/gemm_fast.cpp, joint_matrix path, EPI 0): each work-group
// owns a 256-row x 256-column tile of one expert; per 32-K step it decodes the 256 columns x 32 k of W it needs next
// into one half of a double-buffered SLM tile while its sub-groups run the matrix unit on the other half.
// Decode = Inkling's exact weights (as b70::inkling2::expert_bf16_pair): NVFP4 bf16((e2m1 * e4m3 scale) * outer),
// MXFP8 bf16(e4m3 * e8m0 scale).  Output fp32 rows at the expert's row offset, pitch N.
// Built as its own library with -cl-intel-256-GRF-per-thread -doubleGRF (gemm_fast.cpp's note: the flag is what
// makes this kernel shape fast).
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include "b70/formats.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {
namespace matrix = sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int SGS = 16, TM = 8, TN = 16, TKB = 16, MC1 = 32, NC1 = 64, KC1 = 32, MC2 = 256, NC2 = 256;

inline uint32_t tob(float x) { return uint32_t(b70::f32_to_bf16(x).bits); }

template <int F> class InkMoeG;

template <int F>
sycl::event run(sycl::queue& q, const uint8_t* pay, const uint8_t* scl, size_t rb, size_t sb, int N, int K,
                const bf16* A, float* out, const int* tiles, int T, const int* uinfo, const float* outer,
                const std::vector<sycl::event>& deps) {
    constexpr int SGM = MC2 / MC1, SGN = NC2 / NC1, WG = SGM * SGN * SGS;
    constexpr int KP = KC1 / 2, BT = KP * NC2;
    static_assert(WG == 2 * NC2, "decode: two work-items per column (16 k each)");
    const int nNB = N / NC2;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> Bs(2 * BT, h);
        h.parallel_for<InkMoeG<F>>(sycl::nd_range<1>(size_t(T) * nNB * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SGS)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            auto sg = it.get_sub_group();
            const int lid = int(it.get_local_id(0)), L = int(it.get_group(0));
            const int t = L / nNB, nb = L % nNB;
            const int u = tiles[2 * t], mb = tiles[2 * t + 1];
            const int el = uinfo[3 * u], M = uinfo[3 * u + 2];
            const int64_t rbase = uinfo[3 * u + 1];
            const float o = outer[u];
            const int s = lid / SGS, sm = s / SGN, sn = s % SGN;
            const int m0 = mb * MC2 + sm * MC1;
            const bool active = m0 < M, full = m0 + MC1 <= M;
            const bf16* Ae = A + rbase * K;
            const int dn = lid % NC2, dh = lid / NC2;
            const int64_t grow = int64_t(el) * N + nb * NC2 + dn;
            const uint8_t* prow = pay + grow * int64_t(rb);
            const uint8_t* srow = scl + grow * int64_t(sb);
            uint32_t* bs = Bs.template get_multi_ptr<sycl::access::decorated::no>().get();
            sycl::uint4 pk{0, 0, 0, 0};
            uint8_t sbyte = 0;
            auto fetch = [&](int k) {
                const int kb = k + dh * 16;
                if constexpr (F == 0) {
                    const sycl::uint2 v = *reinterpret_cast<const sycl::uint2*>(prow + (kb >> 1));
                    pk[0] = v[0]; pk[1] = v[1];
                    sbyte = srow[kb >> 4];
                } else {
                    pk = *reinterpret_cast<const sycl::uint4*>(prow + kb);
                    sbyte = srow[kb >> 5];
                }
            };
            auto decode = [&](int buf) {
                uint32_t* d = bs + buf * BT + dh * (KP / 2) * NC2 + dn;
                if constexpr (F == 0) {
                    // bf((e2m1 * sc) * o) exactly: |e2m1| = 2^k * {1, 1.5} and both products are exact up to the
                    // final rounding, which a power of two does not change -> 2^k * bf(sc*o) or 2^k * bf(1.5sc*o)
                    const float sc = b70::e4m3_to_f32(sbyte);
                    const uint32_t q1 = tob(sc * o), q3 = tob((1.5f * sc) * o);
                    auto val = [&](uint32_t nib) {
                        const uint32_t mag = nib & 7u;
                        const uint32_t bq = (mag & 1u) && mag > 1u ? q3 : q1;
                        const int sh = mag == 1u ? -1 : int(mag >> 1) - 1;
                        const uint32_t v = (mag == 0u || bq == 0u) ? 0u : uint32_t(int(bq) + sh * 128);
                        return v | ((nib & 8u) << 12);
                    };
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        const uint32_t byte = (pk[p >> 2] >> (8 * (p & 3))) & 0xFFu;
                        d[p * NC2] = val(byte & 15u) | (val(byte >> 4) << 16);
                    }
                } else {
                    const float sc = b70::e8m0_to_f32(sbyte);
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        const uint32_t w = pk[p >> 1] >> (16 * (p & 1));
                        d[p * NC2] = tob(b70::e4m3_to_f32(uint8_t(w & 0xFFu)) * sc) |
                                     (tob(b70::e4m3_to_f32(uint8_t((w >> 8) & 0xFFu)) * sc) << 16);
                    }
                }
            };
            auto pA = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(Ae);
            auto pBs = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                reinterpret_cast<bf16*>(bs));
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN> acc[MC1 / TM][NC1 / TN];
#pragma unroll
            for (int m = 0; m < MC1 / TM; ++m)
#pragma unroll
                for (int n = 0; n < NC1 / TN; ++n) matrix::joint_matrix_fill(sg, acc[m][n], 0.0f);
            fetch(0);
            decode(0);
            sycl::group_barrier(it.get_group());
            for (int k = 0, step = 0; k < K; k += KC1, ++step) {
                const int cur = step & 1;
                const bool more = k + KC1 < K;
                if (more) fetch(k + KC1);
                if (active) {
                    matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::a, TM, TKB, matrix::layout::row_major> a[MC1 / TM][KC1 / TKB];
                    matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::b, TKB, TN, matrix::layout::ext_intel_packed> b[NC1 / TN][KC1 / TKB];
#pragma unroll
                    for (int kk = 0; kk < KC1 / TKB; ++kk) {
#pragma unroll
                        for (int m = 0; m < MC1 / TM; ++m) {
                            if (full)
                                matrix::joint_matrix_load(sg, a[m][kk], pA + size_t(m0 + m * TM) * K + k + kk * TKB, K);
                            else
                                ix::joint_matrix_load_checked(sg, a[m][kk], pA, size_t(K), size_t(M), size_t(K),
                                                              size_t(m0 + m * TM), size_t(k + kk * TKB));
                        }
#pragma unroll
                        for (int n = 0; n < NC1 / TN; ++n)
                            matrix::joint_matrix_load(sg, b[n][kk],
                                pBs + size_t(cur * BT + kk * (TKB / 2) * NC2 + sn * NC1 + n * TN) * 2, size_t(NC2) * 2);
                    }
#pragma unroll
                    for (int kk = 0; kk < KC1 / TKB; ++kk)
#pragma unroll
                        for (int m = 0; m < MC1 / TM; ++m)
#pragma unroll
                            for (int n = 0; n < NC1 / TN; ++n)
                                matrix::joint_matrix_mad(sg, acc[m][n], a[m][kk], b[n][kk], acc[m][n]);
                }
                if (more) decode(cur ^ 1);
                sycl::group_barrier(it.get_group());
            }
            if (!active) return;
            auto pC = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                out + rbase * N);
#pragma unroll
            for (int m = 0; m < MC1 / TM; ++m)
#pragma unroll
                for (int n = 0; n < NC1 / TN; ++n) {
                    const int col = nb * NC2 + sn * NC1 + n * TN;
                    if (full)
                        matrix::joint_matrix_store(sg, acc[m][n], pC + size_t(m0 + m * TM) * N + col, N, matrix::layout::row_major);
                    else
                        ix::joint_matrix_store_checked(sg, acc[m][n], pC, size_t(N), matrix::layout::row_major, size_t(M),
                                                       size_t(N), size_t(m0 + m * TM), size_t(col));
                }
        });
    });
}
} // namespace

// fmt 0 = NVFP4 (rb = K/2, sb = K/16), 1 = MXFP8 (rb = K, sb = K/32).  tiles: T pairs (u, row block of 256);
// uinfo: per used expert u (local expert index, first row, rows); outer: per u (NVFP4 outer, 1 for MXFP8).
// N % 256 == 0, K % 32 == 0.
sycl::event ink_moe_grouped(sycl::queue& q, int fmt, const uint8_t* pay, const uint8_t* scl, size_t rb, size_t sb,
                            int N, int K, const uint16_t* A, float* out, const int* tiles, int T, const int* uinfo,
                            const float* outer, const std::vector<sycl::event>& deps) {
    const bf16* a = reinterpret_cast<const bf16*>(A);
    return fmt == 0 ? run<0>(q, pay, scl, rb, sb, N, K, a, out, tiles, T, uinfo, outer, deps)
                    : run<1>(q, pay, scl, rb, sb, N, K, a, out, tiles, T, uinfo, outer, deps);
}

// ---------------------------------------------------------------------------------------------------------------
// Prompt attention on the matrix unit.  Work-group = 64 query rows x one head, 8 sub-groups (8 rows each).
// Two passes over 64-key blocks, as the SIMT kernel: pass 1 = row max / sum, pass 2 = weights bf(exp(s-high)/sum)
// (bf16, so they feed DPAS exactly) times V.  Scores keep the reference rounding: qk = bf(bf(q.k) * scale),
// s = bf(qk + bf(bias)), bias = sum_r rel[r] * table[r][distance] for distance < E; sink in the softmax;
// out = bf(bf(acc) * sigmoid(bf(gate))).  Only the fp32 summation order of q.k and P.V differs.
// ---------------------------------------------------------------------------------------------------------------
struct InkAttnArgs {
    const float* A; int NA, oq, orel, osink, og;
    int H, groups, E;
    const uint16_t* table; const uint16_t *kc, *vc;
    int cap, K, win; float scale;
    float* aout; uint16_t* aoutb; int Q;
    const int* posd; int M;
    const uint16_t* bias;   // [M][H][E] bf16 = bf(sum_r rel[r] * table[r][d])
};

namespace {
class InkAttD;
inline float bfr_(float x) { return b70::bf16_to_f32(b70::f32_to_bf16(x)); }
inline float bfw_(uint16_t w) { return b70::bits_to_f32(uint32_t(w) << 16); }
inline float sigm_(float x) { return x >= 0.f ? 1.f / (1.f + sycl::exp(-x)) : sycl::exp(x) / (1.f + sycl::exp(x)); }
}

sycl::event ink_attn_prompt(sycl::queue& q, const InkAttnArgs& g) {
    // 16 sub-groups x 8 query rows = 128 rows of one head per work-group.  SLM: K block packed (16 KB), V block
    // packed (16 KB), each sub-group's 8 x 64 bf16 weights (16 KB); Q goes through K+V once into registers.
    // Scores stay in the DPAS accumulators: element r of a work-item = row r, column = lane (8 x 16 fp32 tile).
    constexpr int NSG = 16, BM = NSG * TM, BN = 64, D = 128, WGS = NSG * SGS;
    constexpr int OFF_V = 16384, OFF_P = 32768, SLM = 49152;
    const int nb = (g.M + BM - 1) / BM;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(SLM), h);
        const InkAttnArgs a = g;
        h.parallel_for<InkAttD>(sycl::nd_range<1>(size_t(nb) * a.H * WGS, WGS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SGS)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            const int lid = int(it.get_local_id(0)), sm = int(sg.get_group_linear_id());
            const int b = int(it.get_group(0)) / a.H, j = int(it.get_group(0)) % a.H;
            uint8_t* base8 = slm.template get_multi_ptr<sycl::access::decorated::no>().get();
            uint16_t* Ks = reinterpret_cast<uint16_t*>(base8);
            uint16_t* Vs = reinterpret_cast<uint16_t*>(base8 + OFF_V);
            uint16_t* Ps = reinterpret_cast<uint16_t*>(base8 + OFF_P) + sm * TM * BN;
            const int base = *a.posd;
            const int mf = b * BM, ml = sycl::min(a.M - 1, mf + BM - 1);
            const int posf = base + mf;
            const int lo = sycl::max(0, posf + 1 - sycl::min(posf + 1, a.win)), hi = base + ml;
            const int hd = (j / a.groups) * D;
            const int m0 = mf + sm * TM;                      // this sub-group's first row
            auto pQ = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(reinterpret_cast<bf16*>(base8));
            auto pK = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(reinterpret_cast<bf16*>(Ks));
            auto pV = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(reinterpret_cast<bf16*>(Vs));
            auto pP = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(reinterpret_cast<bf16*>(Ps));
            {   // Q block (bf16-valued floats) -> SLM [128][128] bf16 (all 32 KB) -> registers
                uint16_t* Qs = reinterpret_cast<uint16_t*>(base8);
                const int r = lid / 2, h2 = lid % 2, m = mf + r;
                const float* src = a.A + size_t(m) * a.NA + a.oq + j * D + h2 * 64;
                for (int d = 0; d < 64; ++d) Qs[r * D + h2 * 64 + d] = m < a.M ? b70::f32_to_bf16(src[d]).bits : uint16_t(0);
            }
            sycl::group_barrier(grp);
            matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::a, TM, TKB, matrix::layout::row_major> qa[D / TKB];
#pragma unroll
            for (int kk = 0; kk < D / TKB; ++kk) matrix::joint_matrix_load(sg, qa[kk], pQ + sm * TM * D + kk * TKB, D);
            auto load_kv = [&](int k0, bool with_v) {
                // 64 keys x 128 dims; 256 work-items: key lid/4, 32 dims each
                const int kk = lid / 4, q4 = lid % 4, kp = k0 + kk;
                const bool ok = kp <= hi;
                const size_t row = size_t(ok ? kp % a.cap : 0) * a.K + hd + q4 * 32;
                for (int d0 = 0; d0 < 32; d0 += 8) {
                    sycl::vec<uint16_t, 8> kv(0), vv(0);
                    if (ok) {
                        kv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(a.kc + row + d0);
                        if (with_v) vv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(a.vc + row + d0);
                    }
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int dd = q4 * 32 + d0 + e;
                        Ks[((dd >> 1) * BN + kk) * 2 + (dd & 1)] = kv[e];
                        if (with_v) Vs[((kk >> 1) * D + dd) * 2 + (kk & 1)] = vv[e];
                    }
                }
            };
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN> s[BN / TN];
            auto qk = [&]() {
#pragma unroll
                for (int n = 0; n < BN / TN; ++n) {
                    matrix::joint_matrix_fill(sg, s[n], 0.0f);
#pragma unroll
                    for (int kk = 0; kk < D / TKB; ++kk) {
                        matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::b, TKB, TN, matrix::layout::ext_intel_packed> kb;
                        matrix::joint_matrix_load(sg, kb, pK + ((kk * TKB / 2) * BN + n * TN) * 2, BN * 2);
                        matrix::joint_matrix_mad(sg, s[n], qa[kk], kb, s[n]);
                    }
                }
            };
            // score of row m (absolute), key kp, raw dot x
            auto score = [&](float x, int m, int kp) {
                const float qkv = bfr_(bfr_(x) * a.scale);
                const int distance = base + m - kp;
                const float bias = distance < a.E ? bfw_(a.bias[(size_t(m) * a.H + j) * a.E + distance]) : 0.f;
                return bfr_(qkv + bias);
            };
            float mx[TM], l[TM];
#pragma unroll
            for (int r = 0; r < TM; ++r) { mx[r] = -INFINITY; l[r] = 0.f; }
            const bool any = m0 < a.M;
            for (int k0 = lo; k0 <= hi; k0 += BN) {
                sycl::group_barrier(grp);
                load_kv(k0, false);
                sycl::group_barrier(grp);
                if (!any) continue;
                qk();
#pragma unroll
                for (int n = 0; n < BN / TN; ++n)
                    ix::joint_matrix_apply(sg, s[n], [&](float& x, size_t r, size_t c) {
                        const int m = m0 + int(r), pos = base + m, kp = k0 + n * TN + int(c);
                        const int first = pos + 1 - sycl::min(pos + 1, a.win);
                        if (m < a.M && kp >= first && kp <= pos) {
                            const float sc = score(x, m, kp);
                            if (sc > mx[r]) { l[r] = l[r] * sycl::exp(mx[r] - sc) + 1.f; mx[r] = sc; } else l[r] += sycl::exp(sc - mx[r]);
                        }
                    });
            }
            float high[TM], sum[TM];
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                const int m = m0 + r;
                const float gm = sycl::reduce_over_group(sg, mx[r], sycl::maximum<float>());
                const float ls = sycl::reduce_over_group(sg, mx[r] == -INFINITY ? 0.f : l[r] * sycl::exp(mx[r] - gm), sycl::plus<float>());
                const float sink = m < a.M ? bfr_(a.A[size_t(m) * a.NA + a.osink + j]) : 0.f;
                high[r] = sycl::fmax(sink, gm);
                sum[r] = sycl::exp(sink - high[r]) + ls * sycl::exp(gm - high[r]);
            }
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN> o[D / TN];
#pragma unroll
            for (int n = 0; n < D / TN; ++n) matrix::joint_matrix_fill(sg, o[n], 0.0f);
            for (int k0 = lo; k0 <= hi; k0 += BN) {
                sycl::group_barrier(grp);
                load_kv(k0, true);
                sycl::group_barrier(grp);
                if (!any) continue;
                qk();
#pragma unroll
                for (int n = 0; n < BN / TN; ++n)
                    ix::joint_matrix_apply(sg, s[n], [&](float& x, size_t r, size_t c) {
                        const int m = m0 + int(r), pos = base + m, kp = k0 + n * TN + int(c);
                        const int first = pos + 1 - sycl::min(pos + 1, a.win);
                        float w = 0.f;
                        if (m < a.M && kp >= first && kp <= pos) w = bfr_(sycl::exp(score(x, m, kp) - high[r]) / sum[r]);
                        Ps[int(r) * BN + n * TN + int(c)] = b70::f32_to_bf16(w).bits;
                    });
                sycl::group_barrier(sg);
#pragma unroll
                for (int kk = 0; kk < BN / TKB; ++kk) {
                    matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::a, TM, TKB, matrix::layout::row_major> pa;
                    matrix::joint_matrix_load(sg, pa, pP + kk * TKB, BN);
#pragma unroll
                    for (int n = 0; n < D / TN; ++n) {
                        matrix::joint_matrix<sycl::sub_group, bf16, matrix::use::b, TKB, TN, matrix::layout::ext_intel_packed> vb;
                        matrix::joint_matrix_load(sg, vb, pV + ((kk * TKB / 2) * D + n * TN) * 2, D * 2);
                        matrix::joint_matrix_mad(sg, o[n], pa, vb, o[n]);
                    }
                }
            }
            if (!any) return;
#pragma unroll
            for (int n = 0; n < D / TN; ++n)
                ix::joint_matrix_apply(sg, o[n], [&](float& x, size_t r, size_t c) {
                    const int m = m0 + int(r), col = n * TN + int(c);
                    if (m < a.M) {
                        const float ov = bfr_(bfr_(x) * sigm_(bfr_(a.A[size_t(m) * a.NA + a.og + j * D + col])));
                        a.aout[size_t(m) * a.Q + j * D + col] = ov;
                        a.aoutb[size_t(m) * a.Q + j * D + col] = b70::f32_to_bf16(ov).bits;
                    }
                });
        });
    });
}
