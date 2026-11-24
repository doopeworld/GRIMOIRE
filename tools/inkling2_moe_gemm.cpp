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
#include <sycl/ext/intel/esimd.hpp>
#include "b70/formats.hpp"

#include <cmath>
#include <cstdlib>
#include <functional>
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
                const std::vector<sycl::event>& deps, int Md) {
    constexpr int SGM = MC2 / MC1, SGN = NC2 / NC1, WG = SGM * SGN * SGS;
    constexpr int KP = KC1 / 2, BT = KP * NC2;
    static_assert(WG == 2 * NC2, "decode: two work-items per column (16 k each)");
    const int nNB = (N + NC2 - 1) / NC2;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> Bs(2 * BT, h);
        h.parallel_for<InkMoeG<F>>(sycl::nd_range<1>(size_t(T) * nNB * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SGS)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            auto sg = it.get_sub_group();
            const int lid = int(it.get_local_id(0)), L = int(it.get_group(0));
            const int t = L / nNB, nb = L % nNB;
            // F == 2: one dense BF16 weight, tiles implicit (row block t of Md rows)
            const int u = F == 2 ? 0 : tiles[2 * t], mb = F == 2 ? t : tiles[2 * t + 1];
            const int el = F == 2 ? 0 : uinfo[3 * u], M = F == 2 ? Md : uinfo[3 * u + 2];
            const int64_t rbase = F == 2 ? 0 : uinfo[3 * u + 1];
            const float o = F == 2 ? 1.f : outer[u];
            const int s = lid / SGS, sm = s / SGN, sn = s % SGN;
            const int m0 = mb * MC2 + sm * MC1;
            const bool active = m0 < M, full = m0 + MC1 <= M;
            const bf16* Ae = A + rbase * K;
            const int dn = lid % NC2, dh = lid / NC2;
            const bool wrow_ok = nb * NC2 + dn < N;
            const int64_t grow = int64_t(el) * N + (wrow_ok ? nb * NC2 + dn : 0);
            const uint8_t* prow = pay + grow * int64_t(rb);
            const uint8_t* srow = scl + grow * int64_t(sb);
            uint32_t* bs = Bs.template get_multi_ptr<sycl::access::decorated::no>().get();
            sycl::uint4 pk{0, 0, 0, 0}, pk2{0, 0, 0, 0};
            uint8_t sbyte = 0;
            auto fetch = [&](int k) {
                const int kb = k + dh * 16;
                if constexpr (F == 2) {
                    const sycl::uint4* src = reinterpret_cast<const sycl::uint4*>(prow + size_t(kb) * 2);
                    pk = wrow_ok ? src[0] : sycl::uint4(0u);
                    pk2 = wrow_ok ? src[1] : sycl::uint4(0u);
                } else if constexpr (F == 0) {
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
                if constexpr (F == 2) {
#pragma unroll
                    for (int p = 0; p < 8; ++p) d[p * NC2] = p < 4 ? pk[p] : pk2[p - 4];
                } else if constexpr (F == 0) {
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
                    if (F == 2 && col >= N) continue;
                    if (full && (F != 2 || col + TN <= N))
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
    return fmt == 0 ? run<0>(q, pay, scl, rb, sb, N, K, a, out, tiles, T, uinfo, outer, deps, 0)
                    : run<1>(q, pay, scl, rb, sb, N, K, a, out, tiles, T, uinfo, outer, deps, 0);
}

// Dense BF16 W [N][K] (any N, K % 32 == 0): y [M][N] fp32 = x [M][K] bf16 . W^T.  No per-M setup (oneDNN builds a
// primitive per new M), so every prompt length runs at once.
sycl::event ink_dense_bf16(sycl::queue& q, const uint16_t* W, int N, int K, const uint16_t* x, float* y, int M) {
    return run<2>(q, reinterpret_cast<const uint8_t*>(W), nullptr, size_t(K) * 2, 0, N, K, reinterpret_cast<const bf16*>(x), y,
                  nullptr, (M + MC2 - 1) / MC2, nullptr, nullptr, {}, M);
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
    int nsplit = 0, chunk = 0; float* part = nullptr;   // > 0: key-split partials (verify rows, M <= 8)
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
    constexpr int OFF_V = 16384, OFF_P = 32768, SLM = 65536;   // P region: per sub-group 8 x 64 fp32 scores, then bf16 weights
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
            uint16_t* Ps = reinterpret_cast<uint16_t*>(base8 + OFF_P + sm * TM * BN * 4);
            float* Sf = reinterpret_cast<float*>(base8 + OFF_P + sm * TM * BN * 4);
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
            auto pSf = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(Sf);
            const int lane = int(sg.get_local_linear_id());
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
            // next block's K / V fetched into registers while the current block computes, stored to SLM after
            sycl::uint4 kr[4], va[2], vb[2];
            const int kk_ld = (sm & 3) * 16 + lane, q4_ld = sm >> 2;
            auto fetch_kv = [&](int k0, bool with_v) {
                const int kp = k0 + kk_ld;
                const bool ok = kp <= hi;
                const sycl::uint4* src = reinterpret_cast<const sycl::uint4*>(a.kc + size_t(ok ? kp % a.cap : 0) * a.K + hd + q4_ld * 32);
#pragma unroll
                for (int v = 0; v < 4; ++v) kr[v] = ok ? src[v] : sycl::uint4(0u);
                if (with_v)
#pragma unroll
                    for (int t2 = 0; t2 < 2; ++t2) {
                        const int kp2 = sm * 2 + t2, d0 = lane * 8, ka = k0 + 2 * kp2, kb = ka + 1;
                        va[t2] = ka <= hi ? *reinterpret_cast<const sycl::uint4*>(a.vc + size_t(ka % a.cap) * a.K + hd + d0) : sycl::uint4(0u);
                        vb[t2] = kb <= hi ? *reinterpret_cast<const sycl::uint4*>(a.vc + size_t(kb % a.cap) * a.K + hd + d0) : sycl::uint4(0u);
                    }
            };
            auto store_kv = [&](bool with_v) {
                uint32_t* Kw = reinterpret_cast<uint32_t*>(Ks);
#pragma unroll
                for (int v = 0; v < 4; ++v)
#pragma unroll
                    for (int i = 0; i < 4; ++i) Kw[(q4_ld * 16 + v * 4 + i) * BN + kk_ld] = kr[v][i];
                if (with_v) {
                    uint32_t* Vw = reinterpret_cast<uint32_t*>(Vs);
#pragma unroll
                    for (int t2 = 0; t2 < 2; ++t2) {
                        const int kp2 = sm * 2 + t2, d0 = lane * 8;
                        sycl::uint4 w0, w1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const uint32_t lo2 = (va[t2][i] & 0xFFFFu) | (vb[t2][i] << 16), hi2 = (va[t2][i] >> 16) | (vb[t2][i] & 0xFFFF0000u);
                            if (i < 2) { w0[2 * i] = lo2; w0[2 * i + 1] = hi2; } else { w1[2 * (i - 2)] = lo2; w1[2 * (i - 2) + 1] = hi2; }
                        }
                        sycl::uint4* dst = reinterpret_cast<sycl::uint4*>(Vw + kp2 * D + d0);
                        dst[0] = w0;
                        dst[1] = w1;
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
            auto score = [&](float x, float bias) { return bfr_(bfr_(bfr_(x) * a.scale) + bias); };
            // all 32 bias values of this lane for a key block, loaded up front (independent loads, clamped index)
            float bv[TM][4];
            auto load_bias = [&](int k0) {
                // whole block at distance >= E for every row of this sub-group: no bias (full layers, old keys)
                if (base + m0 - (k0 + BN - 1) >= a.E) {
#pragma unroll
                    for (int r = 0; r < TM; ++r)
#pragma unroll
                        for (int c4 = 0; c4 < 4; ++c4) bv[r][c4] = 0.f;
                    return;
                }
#pragma unroll
                for (int r = 0; r < TM; ++r) {
                    const int m = sycl::min(m0 + r, a.M - 1);
#pragma unroll
                    for (int c4 = 0; c4 < 4; ++c4) {
                        const int distance = base + m - (k0 + c4 * 16 + lane);
                        const int dc = sycl::clamp(distance, 0, a.E - 1);
                        const float v = bfw_(a.bias[(size_t(m) * a.H + j) * a.E + dc]);
                        bv[r][c4] = distance < a.E ? v : 0.f;
                    }
                }
            };
            // single pass, online softmax (flash attention): running max m and sum l per row, the output rescaled only
            // when a row's max grows; weights p = bf16(exp(s - m)) feed the matrix unit; sink and 1 / sum at the end
            float m_[TM], l_[TM];
#pragma unroll
            for (int r = 0; r < TM; ++r) { m_[r] = -INFINITY; l_[r] = 0.f; }
            const bool any = m0 < a.M;
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN> o[D / TN];
#pragma unroll
            for (int n = 0; n < D / TN; ++n) matrix::joint_matrix_fill(sg, o[n], 0.0f);
            auto pick = [](const float (&v)[TM], int r) { float x = v[0];
#pragma unroll
                for (int k = 1; k < TM; ++k) if (r == k) x = v[k];
                return x; };
            fetch_kv(lo, true);
            for (int k0 = lo; k0 <= hi; k0 += BN) {
                sycl::group_barrier(grp);
                store_kv(true);
                sycl::group_barrier(grp);
                if (k0 + BN <= hi) fetch_kv(k0 + BN, true);
                if (!any) continue;
                qk();
#pragma unroll
                for (int n = 0; n < BN / TN; ++n) matrix::joint_matrix_store(sg, s[n], pSf + n * TN, BN, matrix::layout::row_major);
                load_bias(k0);
                sycl::group_barrier(sg);
                float alpha[TM];
                bool grew = false;
#pragma unroll
                for (int r = 0; r < TM; ++r) {
                    const int m = m0 + r, pos = base + m;
                    const int first = pos + 1 - sycl::min(pos + 1, a.win);
                    float sc4[4], bm = -INFINITY;
#pragma unroll
                    for (int c4 = 0; c4 < 4; ++c4) {
                        const int col = c4 * 16 + lane, kp = k0 + col;
                        const bool ok = m < a.M && kp >= first && kp <= pos;
                        sc4[c4] = ok ? score(Sf[r * BN + col], bv[r][c4]) : -INFINITY;
                        bm = sycl::fmax(bm, sc4[c4]);
                    }
                    bm = sycl::reduce_over_group(sg, bm, sycl::maximum<float>());
                    const float mn = sycl::fmax(m_[r], bm);
                    alpha[r] = (m_[r] == -INFINITY) ? 0.f : sycl::exp(m_[r] - mn);
                    grew = grew || (mn > m_[r] && m_[r] != -INFINITY);
                    float ps = 0.f, pw[4];
#pragma unroll
                    for (int c4 = 0; c4 < 4; ++c4) {
                        pw[c4] = sc4[c4] == -INFINITY ? 0.f : bfr_(sycl::exp(sc4[c4] - mn));
                        ps += pw[c4];
                    }
                    ps = sycl::reduce_over_group(sg, ps, sycl::plus<float>());
                    // the weights' row r overlaps score rows <= r only, all read already (rows go in order)
#pragma unroll
                    for (int c4 = 0; c4 < 4; ++c4) Ps[r * BN + c4 * 16 + lane] = b70::f32_to_bf16(pw[c4]).bits;
                    l_[r] = l_[r] * alpha[r] + ps;
                    m_[r] = mn;
                }
                if (grew) {
#pragma unroll
                    for (int n = 0; n < D / TN; ++n)
                        ix::joint_matrix_apply(sg, o[n], [&](float& x, size_t r, size_t) { x *= pick(alpha, int(r)); });
                }
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
            // the sink joins the softmax: out = o * exp(m - mf) / (l * exp(m - mf) + exp(sink - mf))
            float fin[TM];
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                const int m = m0 + r;
                const float sink = m < a.M ? bfr_(a.A[size_t(m) * a.NA + a.osink + j]) : 0.f;
                const float mf = sycl::fmax(m_[r], sink);
                const float em = m_[r] == -INFINITY ? 0.f : sycl::exp(m_[r] - mf);
                fin[r] = em / (l_[r] * em + sycl::exp(sink - mf));
            }
#pragma unroll
            for (int n = 0; n < D / TN; ++n)
                ix::joint_matrix_apply(sg, o[n], [&](float& x, size_t r, size_t c) {
                    const int m = m0 + int(r), col = n * TN + int(c);
                    if (m < a.M) {
                        const float ov = bfr_(bfr_(x * pick(fin, int(r))) * sigm_(bfr_(a.A[size_t(m) * a.NA + a.og + j * D + col])));
                        a.aout[size_t(m) * a.Q + j * D + col] = ov;
                        a.aoutb[size_t(m) * a.Q + j * D + col] = b70::f32_to_bf16(ov).bits;
                    }
                });
        });
    });
}

// Repack one projection of every expert (NVFP4 row-major [n][k/2] + scales [n][k/16]) into the xexp layout.
sycl::event ink_xexp_pack(sycl::queue& q, const uint8_t* pay, const uint8_t* scl, int E, int N, int K, uint8_t* codes, uint8_t* scales) {
    const int T = N / 16, G = K / 128;
    q.parallel_for(sycl::range<1>(size_t(E) * N * (K / 2)), [=](sycl::id<1> id) {
        const size_t i = id[0], per = size_t(N) * (K / 2), e = i / per, r = i % per;
        const size_t blk = r / 1024, within = r % 1024, gi = blk / T, t = blk % T;
        const size_t kh = within / 16, lane = within % 16;          // kh = kg / 2
        const size_t n = t * 16 + lane, kb = gi * 64 + kh;          // byte column in the row
        codes[i] = pay[e * per + n * (K / 2) + kb];
    });
    return q.parallel_for(sycl::range<1>(size_t(E) * N * (K / 16)), [=](sycl::id<1> id) {
        const size_t i = id[0], per = size_t(N) * (K / 16), e = i / per, r = i % per;
        const size_t blk = r / 128, within = r % 128, gi = blk / T, t = blk % T;
        const size_t c = within / 16, lane = within % 16;
        const size_t n = t * 16 + lane, kb = gi * 8 + c;
        scales[i] = scl[e * per + n * (K / 16) + kb];
    });
}

// ---------------------------------------------------------------------------------------------------------------
// Decode experts = GRIMOIRE's Ornith ESIMD decode MoE (src/moe_kernels.cpp moe_gu_esimd / moe_dn_esimd, the
// llm-scaler shape, MEASURED there 266/248 -> 477/484 GB/s) on Inkling's NVFP4 rows as stored (no repack):
// a thread streams whole rows with contiguous 64-byte block loads, keeps its activation slice in registers split
// even / odd K (= low / high nibble), decodes E2M1 in the ALU (nibble bits -> fp16 = value x 2^-14, the 2^14 folded
// into the scale), no tables, no barrier in the loop.  NVFP4 scale = E4M3 per 16 values x outer (amax / den);
// weights are not rounded to bf16 one by one (vLLM-style dequant), the sums are.
// ---------------------------------------------------------------------------------------------------------------
namespace {
namespace es2 = sycl::ext::intel::esimd;
template <int N>
SYCL_ESIMD_FUNCTION inline void nv_e2m1_split(es2::simd<uint8_t, N> b, es2::simd<float, N>& lo, es2::simd<float, N>& hi) {
    es2::simd<uint16_t, N> u = b;
    es2::simd<uint16_t, N> l = ((u & 0x7) << 9) | ((u & 0x8) << 12);
    es2::simd<uint16_t, N> hb = ((u & 0x70) << 5) | ((u & 0x80) << 8);
    es2::simd<sycl::half, N> lh = l.template bit_cast_view<sycl::half>();
    es2::simd<sycl::half, N> hh = hb.template bit_cast_view<sycl::half>();
    lo = lh;
    hi = hh;
}
SYCL_ESIMD_FUNCTION inline es2::simd<float, 8> nv_e4m3x8(es2::simd<uint8_t, 8> sb) {
    es2::simd<uint32_t, 8> b = sb;
    const es2::simd<uint32_t, 8> e = (b >> 3) & 15u, m = b & 7u, s = (b & 0x80u) << 24;
    es2::simd<uint32_t, 8> bits = s | ((e + 120u) << 23) | (m << 20);
    es2::simd<float, 8> v = bits.template bit_cast_view<float>().read();
    es2::simd<float, 8> sub = es2::simd<float, 8>(m) * (1.f / 512.f);
    sub.merge(-sub, s != 0u);
    v.merge(sub, e == 0u);
    return v;
}
// one 128-value step of one NVFP4 row: acc[8] += sum over the 8 scale blocks (16 values each)
template <int KP>
SYCL_ESIMD_FUNCTION inline void nv4_row_step(const uint8_t* prow, const uint8_t* srow, int st, es2::simd<float, KP / 2>& xe,
                                            es2::simd<float, KP / 2>& xo, float so, es2::simd<float, 8>& acc) {
    es2::simd<uint8_t, 64> pb = es2::block_load<uint8_t, 64>(prow + st * 64);
    es2::simd<uint8_t, 8> sb = es2::block_load<uint8_t, 8>(srow + st * 8, es2::properties{es2::alignment<4>});
    es2::simd<float, 64> wl, wh;
    nv_e2m1_split<64>(pb, wl, wh);
    es2::simd<float, 8> sc = nv_e4m3x8(sb) * so;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        es2::simd<float, 8> part = wl.template select<8, 1>(8 * b) * xe.template select<8, 1>(st * 64 + 8 * b) +
                                   wh.template select<8, 1>(8 * b) * xo.template select<8, 1>(st * 64 + 8 * b);
        acc += part * float(sc[b]);
    }
}
inline float es_bfr(float x) SYCL_ESIMD_FUNCTION {
    uint32_t u = sycl::bit_cast<uint32_t>(x);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    return sycl::bit_cast<float>(u);
}
inline float es_th(float v) SYCL_ESIMD_FUNCTION { const float t = sycl::exp(-2.f * (v < 0.f ? -v : v)); const float r = (1.f - t) / (1.f + t); return v < 0.f ? -r : r; }
template <int R, int KS, int H> class InkEsGU;
template <int R, int I> class InkEsDN;
}

struct InkEsArgs {
    const uint8_t* pay; const uint8_t* scl;     // NVFP4 rows as stored: [local expert][rows][K/2], [..][rows][K/16]
    const float* amax; const float* den;
    const int* sel; int e0, en, topk;
    const float* x; float* out;                 // gate_up: x [H] -> act [slot][I];  down: act [slot][I] -> dbuf [slot][H]
    int H, I; float gcap, ucap;
    int own = -1;                               // >= 0: every expert is on this card; slot s is this card's iff s % 2 == own
};

// gate_up + Inkling's softcapped SwiGLU.  Work-group = KS threads splitting H; a thread owns R (gate, up) row pairs
// (checkpoint rows 2i, 2i+1) of one slot.  act = bf(gcap*tanh(g/gcap) * sigmoid(g) * ucap*tanh(u/ucap)), g,u = bf(sum).
template <int R, int KS, int H>
static sycl::event es_gu(sycl::queue& q, const InkEsArgs& a) {
    constexpr int KP = H / KS;
    static_assert(KP % 128 == 0, "K slice: whole 128-value steps");
    const int rg = a.I / R, nwg = a.topk * rg;
    return q.submit([&](sycl::handler& cgh) {
        const InkEsArgs g = a;
        cgh.parallel_for<InkEsGU<R, KS, H>>(sycl::nd_range<1>(size_t(nwg) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es2::slm_init<KS * R * 2 * 4>();
            const int t = int(it.get_local_id(0)), wg = int(it.get_group(0));
            const int slot = wg / rg, i0 = (wg % rg) * R, kb = t * KP;
            const int e = g.sel[slot];
            if (g.own >= 0 ? (slot & 1) != g.own : (e < g.e0 || e >= g.e0 + g.en)) return;   // uniform per work-group
            const int64_t el = g.own >= 0 ? e : e - g.e0, rb = H / 2, rs = H / 16;
            const float so = g.amax[e] / g.den[0] * 16384.f;
            es2::simd<float, KP> xs = es2::block_load<float, KP>(g.x + kb);
            es2::simd<float, KP / 2> xe = xs.template select<KP / 2, 2>(0);
            es2::simd<float, KP / 2> xo = xs.template select<KP / 2, 2>(1);
            es2::simd<float, 8> ga[R], ua[R];
#pragma unroll
            for (int r = 0; r < R; ++r) { ga[r] = 0.f; ua[r] = 0.f; }
#pragma unroll
            for (int st = 0; st < KP / 128; ++st)
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int64_t gr = el * 2 * g.I + 2 * (i0 + r), ur = gr + 1;
                    nv4_row_step<KP>(g.pay + gr * rb + kb / 2, g.scl + gr * rs + kb / 16, st, xe, xo, so, ga[r]);
                    nv4_row_step<KP>(g.pay + ur * rb + kb / 2, g.scl + ur * rs + kb / 16, st, xe, xo, so, ua[r]);
                }
            es2::simd<float, 2 * R> red;
#pragma unroll
            for (int r = 0; r < R; ++r) { red[2 * r] = es2::reduce<float>(ga[r], std::plus<>()); red[2 * r + 1] = es2::reduce<float>(ua[r], std::plus<>()); }
            if constexpr (KS > 1) {
                es2::slm_block_store<float, 2 * R>(t * 2 * R * 4, red);
                es2::barrier();
                if (t != 0) return;
                red = 0.f;
#pragma unroll
                for (int j = 0; j < KS; ++j) red += es2::slm_block_load<float, 2 * R>(j * 2 * R * 4);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float gg = es_bfr(red[2 * r]), uu = es_bfr(red[2 * r + 1]);
                const float sg = gg >= 0.f ? 1.f / (1.f + sycl::exp(-gg)) : sycl::exp(gg) / (1.f + sycl::exp(gg));
                g.out[int64_t(slot) * g.I + i0 + r] = es_bfr((g.gcap * es_th(gg / g.gcap)) * sg * (g.ucap * es_th(uu / g.ucap)));
            }
        });
    });
}

// down: one thread per (slot, R output rows) over the whole reduction (I); the slot's activation in registers.
// dbuf[slot][row] = bf(sum); slots this card does not own write zeros (two-card partial sums).
template <int R, int I>
static sycl::event es_dn(sycl::queue& q, const InkEsArgs& a) {
    static_assert(I % 128 == 0, "");
    const int S = a.topk;
    return q.submit([&](sycl::handler& cgh) {
        const InkEsArgs g = a;
        cgh.parallel_for<InkEsDN<R, I>>(sycl::nd_range<1>(size_t(g.H / R) * S, S), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int s = int(it.get_local_id(0)), o0 = int(it.get_group(0)) * R;
            const int e = g.sel[s];
            float* y = g.out + int64_t(s) * g.H + o0;
            if (g.own >= 0 ? (s & 1) != g.own : (e < g.e0 || e >= g.e0 + g.en)) {
#pragma unroll
                for (int r = 0; r < R; ++r) y[r] = 0.f;
                return;
            }
            const int64_t el = g.own >= 0 ? e : e - g.e0, rb = I / 2, rs = I / 16;
            const float so = g.amax[e] / g.den[0] * 16384.f;
            es2::simd<float, I> hs = es2::block_load<float, I>(g.x + int64_t(s) * I);
            es2::simd<float, I / 2> he = hs.template select<I / 2, 2>(0);
            es2::simd<float, I / 2> ho = hs.template select<I / 2, 2>(1);
            es2::simd<float, 8> acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
            for (int st = 0; st < I / 128; ++st)
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int64_t row = el * g.H + o0 + r;
                    nv4_row_step<I>(g.pay + row * rb, g.scl + row * rs, st, he, ho, so, acc[r]);
                }
#pragma unroll
            for (int r = 0; r < R; ++r) y[r] = es_bfr(es2::reduce<float>(acc[r], std::plus<>()));
        });
    });
}

// Inkling 2 Mini shapes: H 2304 (18 steps of 128 -> KS 9, two steps a thread, as Ornith's KS 8 at H 2048), I 1152.
bool ink_es_moe(sycl::queue& q, const InkEsArgs& gu, const InkEsArgs& dn) {
    if (gu.H != 2304 || gu.I != 1152) return false;
    // INK_ES_GU = <R><KS> (row pairs a thread, threads splitting H): 29 (Ornith-like, default), 26, 23, 13, 43, 16, 46
    static const int cfg = std::getenv("INK_ES_GU") ? std::atoi(std::getenv("INK_ES_GU")) : 11;   // MEASURED 11-24 (us): 11 57.4, 12 59.4, 13 61.5, 29 80.1
    switch (cfg) {
        case 26: es_gu<2, 6, 2304>(q, gu); break;
        case 23: es_gu<2, 3, 2304>(q, gu); break;
        case 13: es_gu<1, 3, 2304>(q, gu); break;
        case 43: es_gu<4, 3, 2304>(q, gu); break;
        case 16: es_gu<1, 6, 2304>(q, gu); break;
        case 46: es_gu<4, 6, 2304>(q, gu); break;
        case 19: es_gu<1, 9, 2304>(q, gu); break;
        case 12: es_gu<1, 2, 2304>(q, gu); break;
        case 11: es_gu<1, 1, 2304>(q, gu); break;
        case 22: es_gu<2, 2, 2304>(q, gu); break;
        default: es_gu<2, 9, 2304>(q, gu); break;
    }
    // INK_ES_DN = R output rows a thread: 2 (default), 1, 4
    static const int rdn = std::getenv("INK_ES_DN") ? std::atoi(std::getenv("INK_ES_DN")) : 2;
    if (rdn == 1) es_dn<1, 1152>(q, dn);
    else if (rdn == 4) es_dn<4, 1152>(q, dn);
    else es_dn<2, 1152>(q, dn);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Decode int8 GEMV on the tiled layout [N/16][K/8][16 rows][8 k] (the engine's gemvt8 tiles), ESIMD: a thread owns a
// 16-row tile and a K part and streams it in 512-byte block loads (16 rows x 32 k); x in registers; KS parts summed in
// SLM in part order; y = acc * row scale (mode 0: bf-rounded, 1: raw, 2: bf + softcap).  Rows go to three outputs by
// segment (r < end0 -> y0, < end1 -> y1, else y2), as gemvt8.
// ---------------------------------------------------------------------------------------------------------------
struct InkI8Args {
    const int8_t* T; const float* sc; int N, K;
    const float *x0, *x1, *x2; float *y0, *y1, *y2; int end0, end1; int mode; float cap;
};
namespace { template <int KS> class InkEsI8; }
template <int KS>
static sycl::event es_gemv_i8(sycl::queue& q, const InkI8Args& a) {
    const int tiles = a.N / 16;
    return q.submit([&](sycl::handler& cgh) {
        const InkI8Args g = a;
        cgh.parallel_for<InkEsI8<KS>>(sycl::nd_range<1>(size_t(tiles) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es2::slm_init<KS * 16 * 4>();
            const int part = int(it.get_local_id(0)), tile = int(it.get_group(0)), r0 = tile * 16;
            const float* x = r0 < g.end0 ? g.x0 : (r0 < g.end1 ? g.x1 : g.x2);
            const int K32 = g.K / 32, c0 = part * K32 / KS, c1 = (part + 1) * K32 / KS;
            const int8_t* base = g.T + size_t(tile) * (g.K / 8) * 128;
            es2::simd<float, 16> acc = 0.f;
            for (int c = c0; c < c1; ++c) {
                es2::simd<int8_t, 512> w = es2::block_load<int8_t, 512>(base + size_t(c) * 512);   // 4 x [16 rows][8 k]
                es2::simd<float, 32> xv = es2::block_load<float, 32>(x + c * 32);
#pragma unroll
                for (int q4 = 0; q4 < 4; ++q4) {
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        es2::simd<float, 16> wk = w.template select<16, 8>(q4 * 128 + k);
                        acc += wk * float(xv[q4 * 8 + k]);
                    }
                }
            }
            if constexpr (KS > 1) {
                es2::slm_block_store<float, 16>(uint32_t(part * 64), acc);
                es2::barrier();
                if (part != 0) return;
                acc = 0.f;
#pragma unroll
                for (int p = 0; p < KS; ++p) acc += es2::slm_block_load<float, 16>(uint32_t(p * 64));
            }
            es2::simd<float, 16> s = es2::block_load<float, 16>(g.sc + r0);
            acc *= s;
            float* y = r0 < g.end0 ? g.y0 + r0 : (r0 < g.end1 ? g.y1 + (r0 - g.end0) : g.y2 + (r0 - g.end1));
            if (g.mode == 1) { es2::block_store<float, 16>(y, acc); return; }
#pragma unroll
            for (int l = 0; l < 16; ++l) {
                float v = es_bfr(acc[l]);
                if (g.mode == 2) { v = g.cap * es_th(v / g.cap); }
                acc[l] = v;
            }
            es2::block_store<float, 16>(y, acc);
        });
    });
}

// The same for M <= 8 rows (verify / drafter): the weights read once, one 16-row accumulator per input row;
// inputs x_seg + m * ldx, outputs y_seg + m * ldy_seg (per segment), mode as es_gemv_i8.
struct InkI8MArgs {
    const int8_t* T; const float* sc; int N, K, M;
    const float *x0, *x1, *x2; size_t ldx;
    float *y0, *y1, *y2; size_t ly0, ly1, ly2; int end0, end1; int mode; float cap;
};
namespace { template <int KS> class InkEsI8M; }
template <int KS>
static sycl::event es_gemv_i8m(sycl::queue& q, const InkI8MArgs& a) {
    const int tiles = a.N / 16;
    return q.submit([&](sycl::handler& cgh) {
        const InkI8MArgs g = a;
        cgh.parallel_for<InkEsI8M<KS>>(sycl::nd_range<1>(size_t(tiles) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es2::slm_init<KS * 8 * 16 * 4>();
            const int part = int(it.get_local_id(0)), tile = int(it.get_group(0)), r0 = tile * 16;
            const float* x = r0 < g.end0 ? g.x0 : (r0 < g.end1 ? g.x1 : g.x2);
            const int K32 = g.K / 32, c0 = part * K32 / KS, c1 = (part + 1) * K32 / KS;
            const int8_t* base = g.T + size_t(tile) * (g.K / 8) * 128;
            const int M = g.M;
            es2::simd<float, 8 * 16> acc = 0.f;
            for (int c = c0; c < c1; ++c) {
                es2::simd<int8_t, 512> w = es2::block_load<int8_t, 512>(base + size_t(c) * 512);   // 4 x [16 rows][8 k]
                es2::simd<float, 8 * 32> xv;
#pragma unroll
                for (int m = 0; m < 8; ++m)
                    if (m < M) xv.template select<32, 1>(m * 32) = es2::block_load<float, 32>(x + size_t(m) * g.ldx + c * 32);
#pragma unroll
                for (int q4 = 0; q4 < 4; ++q4)
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const es2::simd<float, 16> wk = w.template select<16, 8>(q4 * 128 + k);
#pragma unroll
                        for (int m = 0; m < 8; ++m)
                            if (m < M) acc.template select<16, 1>(m * 16) += wk * float(xv[m * 32 + q4 * 8 + k]);
                    }
            }
            if constexpr (KS > 1) {
                es2::slm_block_store<float, 128>(uint32_t(part * 512), acc);
                es2::barrier();
                if (part != 0) return;
                acc = 0.f;
#pragma unroll
                for (int p = 0; p < KS; ++p) acc += es2::slm_block_load<float, 128>(uint32_t(p * 512));
            }
            const es2::simd<float, 16> s = es2::block_load<float, 16>(g.sc + r0);
            float* y; size_t ly;
            if (r0 < g.end0) { y = g.y0 + r0; ly = g.ly0; }
            else if (r0 < g.end1) { y = g.y1 + (r0 - g.end0); ly = g.ly1; }
            else { y = g.y2 + (r0 - g.end1); ly = g.ly2; }
#pragma unroll
            for (int m = 0; m < 8; ++m)
                if (m < M) {
                    es2::simd<float, 16> v = acc.template select<16, 1>(m * 16);
                    v *= s;
                    if (g.mode != 1)
#pragma unroll
                        for (int l = 0; l < 16; ++l) {
                            float t = es_bfr(v[l]);
                            if (g.mode == 2) t = g.cap * es_th(t / g.cap);
                            v[l] = t;
                        }
                    es2::block_store<float, 16>(y + size_t(m) * ly, v);
                }
        });
    });
}
bool ink_es_gemv_i8m(sycl::queue& q, const InkI8MArgs& a, int ks) {
    if (a.N % 16 || a.K % 32 || a.M < 1 || a.M > 8) return false;
    if (a.end0 % 16 || a.end1 % 16) return false;
    switch (ks) {
        case 2: es_gemv_i8m<2>(q, a); break;
        case 4: es_gemv_i8m<4>(q, a); break;
        case 16: es_gemv_i8m<16>(q, a); break;
        default: es_gemv_i8m<8>(q, a); break;
    }
    return true;
}

bool ink_es_gemv_i8(sycl::queue& q, const InkI8Args& a, int ks) {
    if (a.N % 16 || a.K % 32) return false;
    if (a.end0 % 16 || a.end1 % 16) return false;   // a tile must not straddle two outputs
    switch (ks) {
        case 1: es_gemv_i8<1>(q, a); break;
        case 2: es_gemv_i8<2>(q, a); break;
        case 4: es_gemv_i8<4>(q, a); break;
        case 8: es_gemv_i8<8>(q, a); break;
        case 16: es_gemv_i8<16>(q, a); break;
        default: es_gemv_i8<4>(q, a); break;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Prompt experts = GRIMOIRE's Ornith ESIMD grouped MoE (src/gemm_fast.cpp moe_mxfp4_esimd, EPI 0; MEASURED there
// 52-68 TFLOP/s vs ~25 for the joint_matrix kernel) ported to NVFP4: per 128-K step the work-group decodes its B tiles
// ONCE into SLM (double-buffered, DPAS tile order), 8 m-threads x 2 n-threads each own 32 rows x 32 columns, two DPAS
// per 32 K from 1 KB SLM block loads.  Decode: byte -> E2M1 bf16 pair through the 1 KB table, x the E4M3 scale of its
// 16-K half (exact), x outer, RNE to bf16 = Inkling's exact weight bf16((e2m1*sc)*outer).  out fp32 [rows][Ne].
// ---------------------------------------------------------------------------------------------------------------
namespace { template <int F, int EPI> class InkMoeNvEs; }
struct InkNvEsArgs {
    const uint8_t* pay; const uint8_t* scl; int Ne, K, rows_total;   // rows_total = local experts x Ne
    const uint16_t* A; float* out;
    const int* tiles; int T; const int* uinfo; const float* outer;
    const uint32_t* lut;                                            // [256] byte -> lo | hi << 16 (E2M1 bf16)
    uint16_t* act = nullptr; float gcap = 0.f, ucap = 0.f;          // act != null: fused softcapped SwiGLU epilogue
    int fmt = 0;                                                    // 0 NVFP4 (E4M3 / 16 x outer), 1 MXFP8 (E4M3 x E8M0 / 32)
};
template <int F, int EPI>
static sycl::event ink_moe_nv_es_impl(sycl::queue& q, const InkNvEsArgs& g) {
    constexpr int MC2E = 256, MT = 32, TMT = MC2E / MT, TNT = 2, TPW = TMT * TNT;
    constexpr int CG = 2 * TNT;                 // 16-column groups per work-group
    constexpr int STEP = 4 * CG * 2 * 512;      // bytes of decoded B per 128 K
    constexpr int NC = 4 * CG / TPW;            // (32-K block, group) pairs a thread decodes
    static_assert(NC * TPW == 4 * CG, "decode pairs must split evenly");
    const int nG = g.Ne / (32 * TNT);
    return q.submit([&](sycl::handler& h) {
        const InkNvEsArgs a = g;
        h.parallel_for<InkMoeNvEs<F, EPI>>(sycl::nd_range<1>(size_t(a.T) * nG * TPW, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            using bfx = sycl::ext::oneapi::bfloat16;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            es::slm_init<1024 + 2 * STEP>();
            const int lid = int(it.get_local_id(0));
            const int grp = int(it.get_group(0));
            const int t = grp / nG, ng = grp % nG;
            const int u = a.tiles[2 * t], mb = a.tiles[2 * t + 1];
            const int el = a.uinfo[3 * u], M = a.uinfo[3 * u + 2];
            const int64_t rbase = a.uinfo[3 * u + 1];
            const float o = a.outer[u];
            if (lid < 8) es::slm_block_store<uint32_t, 32>(lid * 128, es::block_load<uint32_t, 32>(a.lut + lid * 32));
            const int ti = lid / TNT, tj = lid % TNT;
            const int m0 = mb * MC2E + ti * MT;
            const bool active = m0 < M;
            const int nrb0 = (M - m0 + 7) / 8, nrb = nrb0 < 4 ? nrb0 : 4;
            const bfx* Ae = reinterpret_cast<const bfx*>(a.A) + size_t(rbase) * a.K;
            const unsigned AW = unsigned(a.K) * 2 - 1, AH = unsigned(M) - 1;
            const uint32_t* payw = reinterpret_cast<const uint32_t*>(a.pay);
            const unsigned PW = unsigned(F == 0 ? a.K / 2 : a.K) - 1, PH = unsigned(a.rows_total) - 1, PP = PW;
            const uint32_t* sclw = reinterpret_cast<const uint32_t*>(a.scl);
            const unsigned SB = unsigned(a.K) / (F == 0 ? 16 : 32);   // scale bytes a row
            const es::simd<uint32_t, 16> iv(0, 1);
            es::barrier();
            es::simd<float, 4 * 2 * 128> acc = 0.0f;
            const int nsteps = a.K / 128;
            constexpr int TWN = F == 0 ? 64 : 128, SWN = F == 0 ? 32 : 16;
            es::simd<uint32_t, TWN * NC> tw;   // payload [4 | 8 dwords][16 rows] a pair
            es::simd<uint32_t, SWN * NC> sw;   // scale dwords: NVFP4 [2][16 rows] (8 x 16-K blocks), MXFP8 [16 rows] (4 x 32-K)
            auto fetch = [&](int st) SYCL_ESIMD_FUNCTION {
                const int k1 = st * 128;
#pragma unroll
                for (int i = 0; i < NC; ++i) {
                    const int c = lid + i * TPW, mxc = c / CG, cg = c % CG;
                    const int nbc = ng * TNT + (cg >> 1);
                    const int rh = el * a.Ne + nbc * 32 + (cg & 1) * 16;
                    if constexpr (F == 0) {
                        tw.template select<64, 1>(64 * i) = es::load_2d<uint32_t, 4, 16, 1, true, false>(payw, PW, PH, PP, (k1 + 32 * mxc) / 8, rh);
                        const es::simd<uint32_t, 16> so = (iv + unsigned(rh)) * SB + unsigned(k1 / 16);
                        sw.template select<16, 1>(32 * i) = es::gather<uint32_t, 16>(sclw, so);
                        sw.template select<16, 1>(32 * i + 16) = es::gather<uint32_t, 16>(sclw, so + 4u);
                        if (mxc == 0 && st + 2 < nsteps) es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PP, (st + 2) * 16, rh, PFH);
                    } else {
                        tw.template select<128, 1>(128 * i) = es::load_2d<uint32_t, 8, 16, 1, true, false>(payw, PW, PH, PP, (k1 + 32 * mxc) / 4, rh);
                        const es::simd<uint32_t, 16> so = (iv + unsigned(rh)) * SB + unsigned(k1 / 32);
                        sw.template select<16, 1>(16 * i) = es::gather<uint32_t, 16>(sclw, so);
                        if (mxc < 2 && st + 2 < nsteps) es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PP, (st + 2) * 32 + mxc * 16, rh, PFH);
                    }
                }
            };
            auto decode = [&](int st) SYCL_ESIMD_FUNCTION {
                const unsigned base = 1024u + unsigned(st & 1) * STEP;
#pragma unroll
                for (int i = 0; i < NC; ++i) {
                    const int cidx = lid + i * TPW, mxc = cidx / CG, cg = cidx % CG;
                    es::simd<uint32_t, 256> vb;
                    if constexpr (F == 1) {
                        // E4M3 byte -> fp16 bits (value x 2^-8, exact incl. subnormals), x 2^(x - 127 + 8), RNE to bf16 (exact)
                        const es::simd<uint32_t, 16> xb = (es::simd<uint32_t, 16>(sw.template select<16, 1>(16 * i)) >> (8 * mxc)) & 0xFFu;
                        es::simd<uint32_t, 16> sbits = (xb + 8u) << 23;
                        const es::simd<float, 16> s16 = sbits.template bit_cast_view<float>().read();
#pragma unroll
                        for (int dw = 0; dw < 8; ++dw) {
                            const es::simd<uint32_t, 16> wx = tw.template select<16, 1>(128 * i + dw * 16);
#pragma unroll
                            for (int hf = 0; hf < 2; ++hf) {
                                const es::simd<uint32_t, 16> w = hf ? es::simd<uint32_t, 16>(wx >> 16) : wx;
                                es::simd<uint32_t, 16> hp = ((w & 0x7Fu) << 7) | ((w & 0x80u) << 8) | ((w & 0x7F00u) << 15) | ((w & 0x8000u) << 16);
                                es::simd<float, 32> f = es::simd<float, 32>(hp.template bit_cast_view<sycl::half>().read());
                                f.template select<16, 2>(0) = es::simd<float, 16>(f.template select<16, 2>(0)) * s16;
                                f.template select<16, 2>(1) = es::simd<float, 16>(f.template select<16, 2>(1)) * s16;
                                es::simd<bfx, 32> bb(f);
                                vb.template select<16, 1>((2 * dw + hf) * 16) = bb.template bit_cast_view<uint32_t>().read();
                            }
                        }
                    } else
#pragma unroll
                    for (int hk = 0; hk < 2; ++hk) {
                        const int sbyte = 2 * mxc + hk;   // 16-K block within the 128-K step
                        es::simd<uint32_t, 16> ev = (es::simd<uint32_t, 16>(sw.template select<16, 1>(32 * i + 16 * (sbyte >> 2))) >> (8 * (sbyte & 3))) & 0xFFu;
                        // E4M3 -> fp32 (normal and subnormal; NaN code not produced by the checkpoint)
                        const es::simd<uint32_t, 16> ee = (ev >> 3) & 15u, mm = ev & 7u;
                        es::simd<uint32_t, 16> nb = ((ee + 120u) << 23) | (mm << 20);
                        es::simd<float, 16> s16 = nb.template bit_cast_view<float>().read();
                        es::simd<float, 16> sub = es::simd<float, 16>(mm) * (1.f / 512.f);
                        s16.merge(sub, ee == 0u);
#pragma unroll
                        for (int kp = 0; kp < 8; ++kp) {
                            const int dw = 2 * hk + (kp >> 2);
                            es::simd<uint32_t, 16> wx = tw.template select<16, 1>(64 * i + dw * 16);
                            const int j = kp & 3;
                            es::simd<uint32_t, 16> addr = j == 0 ? ((wx << 2) & 0x3FCu) : ((wx >> (8 * j - 2)) & 0x3FCu);
                            es::simd<uint32_t, 16> p = es::slm_gather<uint32_t, 16>(addr);
                            es::simd<uint32_t, 16> lo = p << 16, hi = p & 0xFFFF0000u;
                            es::simd<float, 16> lof = (lo.template bit_cast_view<float>().read() * s16) * o;
                            es::simd<float, 16> hif = (hi.template bit_cast_view<float>().read() * s16) * o;
                            es::simd<uint32_t, 16> lu = lof.template bit_cast_view<uint32_t>().read();
                            es::simd<uint32_t, 16> hu = hif.template bit_cast_view<uint32_t>().read();
                            lu = (lu + 0x7FFFu + ((lu >> 16) & 1u)) >> 16;
                            hu = (hu + 0x7FFFu + ((hu >> 16) & 1u)) & 0xFFFF0000u;
                            vb.template select<16, 1>((hk * 8 + kp) * 16) = hu | lu;
                        }
                    }
                    es::slm_block_store<uint32_t, 256>(base + unsigned((mxc * CG + cg) * 2) * 512u, vb);
                }
            };
            fetch(0);
            decode(0);
            if (nsteps > 1) fetch(1);
            es::barrier();
            for (int s = 0; s < nsteps; ++s) {
                if (active) {
                    const unsigned base = 1024u + unsigned(s & 1) * STEP;
#pragma unroll 1
                    for (int mx = 0; mx < 4; ++mx) {
                        const int kk = s * 128 + 32 * mx;
                        es::simd<bfx, 1024> av = es::load_2d<bfx, 16, 32, 2>(Ae, AW, AH, AW, kk, m0);
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            es::simd<uint32_t, 256> vb = es::slm_block_load<uint32_t, 256>(base + unsigned((mx * CG + tj * 2 + hh) * 2) * 512u);
                            const es::simd<bfx, 256> b0 = vb.template select<128, 1>(0).template bit_cast_view<bfx>().read();
                            const es::simd<bfx, 256> b1 = vb.template select<128, 1>(128).template bit_cast_view<bfx>().read();
#pragma unroll
                            for (int rb = 0; rb < 4; ++rb) {
                                if (rb < nrb) {
                                    const es::simd<bfx, 128> a0 = av.template select<128, 1>(rb * 128);
                                    const es::simd<bfx, 128> a1 = av.template select<128, 1>(512 + rb * 128);
                                    es::simd<float, 128> c = acc.template select<128, 1>((rb * 2 + hh) * 128);
                                    c = xmx::dpas<8, 8, float, float, bfx, bfx>(c, b0, a0);
                                    c = xmx::dpas<8, 8, float, float, bfx, bfx>(c, b1, a1);
                                    acc.template select<128, 1>((rb * 2 + hh) * 128) = c;
                                }
                            }
                        }
                    }
                }
                if (s + 1 < nsteps) {
                    decode(s + 1);
                    if (s + 2 < nsteps) fetch(s + 2);
                }
                es::barrier();
            }
            if (!active) return;
            const int nb = ng * TNT + tj;
            if constexpr (EPI == 1) {
                // columns 2i / 2i+1 = gate_i / up_i: act_i = bf((gc tanh(g/gc)) sigmoid(g) (uc tanh(u/uc))), g u = bf(.)
                constexpr float L2E = 1.4426950408889634f;
                uint16_t* Hb = a.act + size_t(rbase) * (a.Ne / 2);
                const unsigned HW = unsigned(a.Ne / 2) * 2 - 1;
                const float gc = a.gcap, uc = a.ucap;
                auto th = [](es::simd<float, 64> x) SYCL_ESIMD_FUNCTION {
                    const es::simd<float, 64> t = es::exp2(es::abs(x) * (-2.0f * 1.4426950408889634f));
                    es::simd<float, 64> r = (1.0f - t) / (1.0f + t);
                    r.merge(-r, x < 0.0f);
                    return r;
                };
#pragma unroll
                for (int rb = 0; rb < 4; ++rb) {
                    es::simd<uint16_t, 128> ov;
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        es::simd<float, 128> v = acc.template select<128, 1>((rb * 2 + hh) * 128);
                        es::simd<float, 64> gv = v.template select<64, 2>(0), uv = v.template select<64, 2>(1);
                        gv = es::simd<float, 64>(es::simd<bfx, 64>(gv));
                        uv = es::simd<float, 64>(es::simd<bfx, 64>(uv));
                        const es::simd<float, 64> z = es::exp2(es::abs(gv) * -L2E);
                        es::simd<float, 64> num = 1.0f;
                        num.merge(z, gv < 0.0f);
                        const es::simd<float, 64> r = (gc * th(gv / gc)) * (num / (1.0f + z)) * (uc * th(uv / uc));
                        es::simd<bfx, 64> rb16(r);
                        es::simd<uint16_t, 64> rbits = rb16.template bit_cast_view<uint16_t>().read();
#pragma unroll
                        for (int r8 = 0; r8 < 8; ++r8) ov.template select<8, 1>(r8 * 16 + hh * 8) = rbits.template select<8, 1>(r8 * 8);
                    }
                    es::store_2d<uint16_t, 16, 8>(Hb, HW, AH, HW, nb * 16, m0 + 8 * rb, ov);
                }
            } else {
                float* Oe = a.out + size_t(rbase) * a.Ne;
                const unsigned OW = unsigned(a.Ne) * 4 - 1;
#pragma unroll
                for (int rb = 0; rb < 4; ++rb)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh)
                        es::store_2d<float, 16, 8>(Oe, OW, AH, OW, nb * 32 + 16 * hh, m0 + 8 * rb,
                                                   es::simd<float, 128>(acc.template select<128, 1>((rb * 2 + hh) * 128)));
            }
        });
    });
}
sycl::event ink_moe_nv_esimd(sycl::queue& q, const InkNvEsArgs& g) {
    if (g.fmt == 1) return g.act ? ink_moe_nv_es_impl<1, 1>(q, g) : ink_moe_nv_es_impl<1, 0>(q, g);
    return g.act ? ink_moe_nv_es_impl<0, 1>(q, g) : ink_moe_nv_es_impl<0, 0>(q, g);
}

// ---------------------------------------------------------------------------------------------------------------
// Prompt attention, ESIMD flash (the design of GRIMOIRE's flash_esimd, src/gemm_fast.cpp): one hardware thread owns
// 8 query rows of one head; Q (8 x 128 bf16) stays in registers, K^T / V tiles come straight from the [cap][K] bf16
// cache with 2-D block loads (K transposed, V VNNI-transformed; no SLM, no barrier), scores and probabilities stay in
// registers (DPAS 8 x 16 accumulators), O (8 x 128 fp32) is rescaled per row (online softmax, exp2).  Inkling's score
// = bf(bf(bf(q.k) * scale) + bias[distance]) with the precomputed bf16 bias (InkBias), the sink joins the softmax at
// the end, output = bf(bf(o / sum) * sigmoid(bf(gate))) as ink_attn_prompt.  TPW threads = TPW row tiles of one head.
// ---------------------------------------------------------------------------------------------------------------
namespace { inline int imin_(int x, int y) { return x < y ? x : y; } }
namespace { template <int BK, int TPW, bool SPLIT> class InkAttE; class InkAttMerge; }
template <int BK, int TPW, bool SPLIT = false>
static sycl::event ink_attn_es_impl(sycl::queue& q, const InkAttnArgs& g) {
    constexpr int RB = 8, D = 128, DT = D / 16, NB = BK / 16;
    const int tiles = (g.M + RB - 1) / RB, nrb = (tiles + TPW - 1) / TPW;
    return q.submit([&](sycl::handler& h) {
        const InkAttnArgs a = g;
        const size_t nthr = SPLIT ? size_t(a.H) * a.nsplit : size_t(nrb) * a.H * TPW;
        h.parallel_for<InkAttE<BK, TPW, SPLIT>>(sycl::nd_range<1>(nthr, SPLIT ? 1 : TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            using bfx = sycl::ext::oneapi::bfloat16;
            constexpr float NINF = -std::numeric_limits<float>::infinity();
            constexpr float LOG2E = 1.4426950408889634f;
            const int gi = int(it.get_group(0)), lt = int(it.get_local_id(0));
            const int rb = SPLIT ? 0 : nrb - 1 - gi / a.H, j = gi % a.H;   // heaviest row blocks first
            const int sp = SPLIT ? gi / a.H : 0;                        // key slice (split mode: one thread each)
            const int r0 = SPLIT ? 0 : (rb * TPW + lt) * RB;
            if (r0 >= a.M) return;
            const int base = *a.posd;
            const int rl = imin_(a.M - 1, r0 + RB - 1);
            const int p0 = base + r0, pl = base + rl;
            const int first0 = p0 + 1 - imin_(p0 + 1, a.win);
            const int firstl = pl + 1 - imin_(pl + 1, a.win);
            const int klo0 = first0 & ~15, khi = pl;
            const int klo = SPLIT ? klo0 + sp * a.chunk : klo0;
            const int kend = SPLIT ? imin_(khi, klo + a.chunk - 1) : khi;   // blocks never cross a slice (chunk % BK == 0)
            const int hd = (j / a.groups) * D;
            const unsigned AW = unsigned(a.NA) * 4 - 1, AH = unsigned(a.M) - 1;
            es::simd<bfx, RB * D> qv;                                    // [dt][8 rows][16]
#pragma unroll
            for (int dt = 0; dt < DT; ++dt)
                qv.template select<128, 1>(dt * 128) = es::simd<bfx, 128>(es::load_2d<float, 16, 8>(a.A, AW, AH, AW, a.oq + j * D + dt * 16, r0));
            const uint32_t* kw = reinterpret_cast<const uint32_t*>(a.kc);
            const uint16_t* vw = a.vc;
            const unsigned KB = unsigned(a.K) * 2 - 1;
            es::simd<float, RB * D> o = 0.0f;                            // [dt][8][16]
            es::simd<float, RB> mrow = NINF, lrow = 0.0f;
            const es::simd<int, 16> col(0, 1);
            for (int s0 = klo; s0 <= kend; s0 += BK) {
                es::simd<float, NB * RB * 16> sc = 0.0f;                 // [b][8][16]
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    const int kc0 = s0 + b * 16;
                    if (kc0 <= khi) {
                        const int row = kc0 % a.cap;
                        const unsigned hgt = unsigned(kc0 + 15 <= khi ? a.cap : row + (khi - kc0) + 1) - 1;   // rows past khi read as 0
#pragma unroll
                        for (int dt = 0; dt < DT; ++dt) {
                            es::simd<uint32_t, 128> kt = es::load_2d<uint32_t, 8, 16, 1, true, false>(kw, KB, hgt, KB, (hd + dt * 16) / 2, row);
                            sc.template select<128, 1>(b * 128) = xmx::dpas<8, 8, float, float, bfx, bfx>(
                                es::simd<float, 128>(sc.template select<128, 1>(b * 128)), es::simd<bfx, 256>(kt.template bit_cast_view<bfx>()),
                                es::simd<bfx, 128>(qv.template select<128, 1>(dt * 128)));
                        }
                    }
                }
                const bool nobias = p0 - (s0 + BK - 1) >= a.E;
                const bool needmask = (s0 + BK - 1 > p0) || (s0 < firstl) || (r0 + RB > a.M);
#pragma unroll
                for (int b = 0; b < NB; ++b)
#pragma unroll
                    for (int r = 0; r < RB; ++r) {
                        const int kc0 = s0 + b * 16;
                        es::simd<float, 16> v = sc.template select<16, 1>((b * RB + r) * 16);
                        v = es::simd<float, 16>(es::simd<bfx, 16>(v));
                        v = es::simd<float, 16>(es::simd<bfx, 16>(v * a.scale));
                        const int m = imin_(r0 + r, a.M - 1), pos = base + m;
                        if (!nobias) {
                            const es::simd<int, 16> dist = (pos - kc0) - col;
                            const es::simd<int, 16> dc = es::max(es::min(dist, es::simd<int, 16>(a.E - 1)), es::simd<int, 16>(0));
                            const es::simd<uint32_t, 16> off = (es::simd<uint32_t, 16>(dc) + unsigned((m * a.H + j) * a.E)) * 2u;
                            es::simd<uint16_t, 16> bw = es::gather<uint16_t, 16>(a.bias, off);
                            es::simd<uint32_t, 16> bu = es::simd<uint32_t, 16>(bw) << 16;
                            es::simd<float, 16> bf = bu.template bit_cast_view<float>().read();
                            bf.merge(es::simd<float, 16>(0.0f), (dist < 0) | (dist >= a.E));
                            v = es::simd<float, 16>(es::simd<bfx, 16>(v + bf));
                        }
                        if (needmask) {
                            if (r0 + r >= a.M) v = NINF;
                            else {
                                const int first = pos + 1 - imin_(pos + 1, a.win);
                                const es::simd<int, 16> kp = col + kc0;
                                v.merge(es::simd<float, 16>(NINF), (kp > pos) | (kp < first));
                            }
                        }
                        sc.template select<16, 1>((b * RB + r) * 16) = v;
                    }
                es::simd<float, RB> bm;
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    es::simd<float, 16> v = sc.template select<16, 1>(r * 16);
#pragma unroll
                    for (int b = 1; b < NB; ++b) v = es::max(v, es::simd<float, 16>(sc.template select<16, 1>((b * RB + r) * 16)));
                    bm[r] = es::hmax<float>(v);
                }
                es::simd<float, RB> mn = es::max(mrow, bm), ms = mn;
                ms.merge(es::simd<float, RB>(0.0f), mn == NINF);
                es::simd<float, RB> corr = es::exp2((mrow - ms) * LOG2E), rs;
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    const float msr = ms[r];
                    es::simd<float, 16> sum = 0.0f;
#pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        es::simd<float, 16> p = es::exp2((es::simd<float, 16>(sc.template select<16, 1>((b * RB + r) * 16)) - msr) * LOG2E);
                        p = es::simd<float, 16>(es::simd<bfx, 16>(p));   // the weight the matrix unit sees
                        sc.template select<16, 1>((b * RB + r) * 16) = p;
                        sum += p;
                    }
                    rs[r] = es::reduce<float>(sum, std::plus<>());
                }
                lrow = lrow * corr + rs;
                mrow = mn;
                float cr[RB];
#pragma unroll
                for (int r = 0; r < RB; ++r) cr[r] = corr[r];
#pragma unroll
                for (int dt = 0; dt < DT; ++dt)
#pragma unroll
                    for (int r = 0; r < RB; ++r) o.template select<16, 1>((dt * RB + r) * 16) *= cr[r];
                es::simd<bfx, NB * RB * 16> pb = sc;
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    const int kc0 = s0 + b * 16;
                    if (kc0 <= khi) {
                        const int row = kc0 % a.cap;
                        const unsigned hgt = unsigned(kc0 + 15 <= khi ? a.cap : row + (khi - kc0) + 1) - 1;
#pragma unroll
                        for (int dt = 0; dt < DT; dt += 2) {
                            es::simd<uint16_t, 512> vv = es::load_2d<uint16_t, 16, 16, 2, false, true>(vw, KB, hgt, KB, hd + dt * 16, row);
#pragma unroll
                            for (int u = 0; u < 2; ++u)
                                o.template select<128, 1>((dt + u) * 128) = xmx::dpas<8, 8, float, float, bfx, bfx>(
                                    es::simd<float, 128>(o.template select<128, 1>((dt + u) * 128)),
                                    es::simd<bfx, 256>(vv.template select<256, 1>(u * 256).template bit_cast_view<bfx>()),
                                    es::simd<bfx, 128>(pb.template select<128, 1>(b * 128)));
                        }
                    }
                }
            }
            if constexpr (SPLIT) {   // partials: o [8][128] (fp32, unnormalized), then m and l per row
                float* po = a.part + (size_t(sp) * a.H + j) * RB * D;
                float* pml = a.part + size_t(a.nsplit) * a.H * RB * D + (size_t(sp) * a.H + j) * RB * 2;
#pragma unroll
                for (int r = 0; r < RB; ++r)
#pragma unroll
                    for (int dt = 0; dt < DT; ++dt)
                        es::block_store<float, 16>(po + r * D + dt * 16, es::simd<float, 16>(o.template select<16, 1>((dt * RB + r) * 16)));
                es::simd<float, 16> ml;
                ml.template select<8, 2>(0) = mrow;
                ml.template select<8, 2>(1) = lrow;
                es::block_store<float, 16>(pml, ml);
                return;
            }
            // sink (log2 units) joins the softmax; then the output gate
            const es::simd<int, RB> ri(0, 1);
            const es::simd<int, RB> mr = es::min(ri + r0, es::simd<int, RB>(a.M - 1));
            const es::simd<uint32_t, RB> soff = (es::simd<uint32_t, RB>(mr) * unsigned(a.NA) + unsigned(a.osink + j)) * 4u;
            es::simd<float, RB> sk = es::gather<float, RB>(a.A, soff);
            sk = es::simd<float, RB>(es::simd<bfx, RB>(sk));
            const es::simd<float, RB> mf = es::max(mrow, sk);
            const es::simd<float, RB> em = es::exp2((mrow - mf) * LOG2E);
            const es::simd<float, RB> fin = em / (lrow * em + es::exp2((sk - mf) * LOG2E));
            float fr[RB];
#pragma unroll
            for (int r = 0; r < RB; ++r) fr[r] = fin[r];
            const unsigned OW = unsigned(a.Q) * 4 - 1, OB = unsigned(a.Q) * 2 - 1;
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                es::simd<float, 128> gt = es::load_2d<float, 16, 8>(a.A, AW, AH, AW, a.og + j * D + dt * 16, r0);
                gt = es::simd<float, 128>(es::simd<bfx, 128>(gt));
                const es::simd<float, 128> sg = 1.0f / (1.0f + es::exp2(gt * -LOG2E));
                es::simd<float, 128> ov = o.template select<128, 1>(dt * 128);
#pragma unroll
                for (int r = 0; r < RB; ++r) ov.template select<16, 1>(r * 16) = ov.template select<16, 1>(r * 16) * fr[r];
                ov = es::simd<float, 128>(es::simd<bfx, 128>(ov));
                ov = es::simd<float, 128>(es::simd<bfx, 128>(ov * sg));
                es::store_2d<float, 16, 8>(a.aout, OW, AH, OW, j * D + dt * 16, r0, ov);
                es::simd<bfx, 128> ob(ov);
                es::store_2d<uint16_t, 16, 8>(a.aoutb, OB, AH, OB, j * D + dt * 16, r0, ob.template bit_cast_view<uint16_t>().read());
            }
        });
    });
}
// split mode: merge the key-slice partials, add the sink, apply the output gate (the unsplit kernel's epilogue)
static sycl::event ink_attn_merge(sycl::queue& q, const InkAttnArgs& a) {
    return q.parallel_for<InkAttMerge>(sycl::range<1>(size_t(a.M) * a.H * 128), [=](sycl::id<1> id) {
        constexpr float L2E = 1.4426950408889634f;
        const int d = int(id[0] % 128), j = int((id[0] / 128) % a.H), r = int(id[0] / (128 * size_t(a.H)));
        const float* pml = a.part + size_t(a.nsplit) * a.H * 8 * 128;
        float mx = -INFINITY;
        for (int s = 0; s < a.nsplit; ++s) mx = sycl::fmax(mx, pml[((size_t(s) * a.H + j) * 8 + r) * 2]);
        float l = 0.f, o = 0.f;
        for (int s = 0; s < a.nsplit; ++s) {
            const float ms = pml[((size_t(s) * a.H + j) * 8 + r) * 2];
            if (ms == -INFINITY) continue;
            const float w = sycl::exp2((ms - mx) * L2E);
            l += pml[((size_t(s) * a.H + j) * 8 + r) * 2 + 1] * w;
            o += a.part[((size_t(s) * a.H + j) * 8 + r) * 128 + d] * w;
        }
        const float sk = bfr_(a.A[size_t(r) * a.NA + a.osink + j]);
        const float mf = sycl::fmax(mx, sk);
        const float em = sycl::exp2((mx - mf) * L2E);
        const float fin = em / (l * em + sycl::exp2((sk - mf) * L2E));
        const float gt = bfr_(a.A[size_t(r) * a.NA + a.og + j * 128 + d]);
        const float sg = 1.0f / (1.0f + sycl::exp2(gt * -L2E));
        const float ov = bfr_(bfr_(o * fin) * sg);
        a.aout[size_t(r) * a.Q + j * 128 + d] = ov;
        a.aoutb[size_t(r) * a.Q + j * 128 + d] = b70::f32_to_bf16(ov).bits;
    });
}
sycl::event ink_attn_esimd(sycl::queue& q, const InkAttnArgs& g) {
    if (g.nsplit > 0 && g.M <= 8) {
        if (g.chunk % 64 == 0) ink_attn_es_impl<64, 8, true>(q, g);
        else ink_attn_es_impl<32, 8, true>(q, g);
        return ink_attn_merge(q, g);
    }
    static const int tpw = [] { const char* e = std::getenv("INK_ATT_TPW"); return e ? std::atoi(e) : 8; }();
    static const int bk = [] { const char* e = std::getenv("INK_ATT_BK"); return e ? std::atoi(e) : 64; }();
    if (bk == 32) return tpw == 16 ? ink_attn_es_impl<32, 16>(q, g) : ink_attn_es_impl<32, 8>(q, g);
    return tpw == 16 ? ink_attn_es_impl<64, 16>(q, g) : ink_attn_es_impl<64, 8>(q, g);
}

