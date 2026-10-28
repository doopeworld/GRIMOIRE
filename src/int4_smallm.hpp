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

// int4_smallm.hpp -- the w4a16 DPAS small-M GEMM (INT4 weights, M <= 16 bf16
// activation rows), shared by two translation units with different register
// files: gemm_fast.cpp (libgrimoire_gemm.so, 256 GRF) keeps M > 8, and
// gemv_decode.cpp (128 GRF, twice the threads per core) runs M <= 8 with
// its own split-K plan.  The caller passes the plan: KS threads split K in
// chunks of kc, TPT column tiles of 16 per work-group, TPT * KS <= 32.
#ifndef B70_INT4_SMALLM_HPP
#define B70_INT4_SMALLM_HPP

#include "kernels.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <vector>

namespace b70 {
namespace {

// INT4 weights -- GRIMOIRE's INT4: asymmetric, a bf16 scale and a u8 zero
// per GS = 64 or 128 K, zero 0xff = signed s4 (GPTQ experts, XORed with 0x88
// at load) -- with the MXFP4 kernel's small-M contract and thread design.
// Dequant on the ALU, exact: (q - z) is a small integer, made as the fp32
// difference (2^23 + q) - (2^23 + z) and truncated to bf16 (exact, |q - z|
// <= 15) in VNNI order; the group's bf16 scale multiplies the DPAS result per
// column once per 64 K.  So the only rounding is DPAS's own plus the bf16
// activation: the w4a16 engine that the verify notes in grimoire.cpp mm()
// measured as identical to the fp32 GEMV in accepted drafts and tokens.
// Each column's scale and zero come from two gathers per 64 K: the [N][K/GS]
// arrays are too narrow for 2-D block loads at K = 2048.
template <int RBN, int GS>
sycl::event int4_smallm_impl(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                             float* Y, int M, int KS, int kc, int TPT,
                             const std::vector<sycl::event>& deps) {
    static_assert(GS == 64 || GS == 128, "INT4 group of 64 or 128");
    constexpr int CT = 16;
    const int N = w.N, K = w.K;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(w.payload);
    const uint16_t* sclh = static_cast<const uint16_t*>(w.scales);
    const uint8_t* zer = w.zeros;
    const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(N) - 1, PP = unsigned(w.row_bytes) - 1;
    const unsigned RS = unsigned(w.row_scales);
    const unsigned XW = unsigned(K) * 2 - 1, XH = unsigned(M) - 1;
    const unsigned YW = unsigned(N) * 4 - 1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS),
                       [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            es::slm_init<32 * RBN * 128 * 4>();
            const int lid = int(it.get_local_id(0));
            const int tw = lid / KS, ks = lid % KS;
            const int tile = int(it.get_group(0)) * TPT + tw;
            const bool live = tile < tiles;
            const int n0 = (live ? tile : 0) * CT;
            const int kb = ks * kc;
            const int ke = kb + kc < K ? kb + kc : K;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            // element index of each column's group-0 scale / zero
            const es::simd<uint32_t, 16> gbase =
                (es::simd<uint32_t, 16>(0, 1) + uint32_t(n0)) * RS;
            es::simd<float, RBN * 128> acc = 0.0f;
            if (live) {
                // Each column's scale / zero for the next 128 K are gathered one
                // step ahead, like the payload prefetch: gathered in-step they
                // were a dependent load in front of every decode (Qwen3.8-27B
                // gate_up at M = 4 ran at ~380 GB/s against ~550 for MXFP4).
                constexpr int NGS = 128 / GS;        // groups per 128 K
                es::simd<uint32_t, 16> zn[NGS], sn[NGS];
                #pragma unroll
                for (int g = 0; g < NGS; ++g) {
                    const es::simd<uint32_t, 16> gi = gbase + uint32_t((kb + GS * g) / GS);
                    zn[g] = es::gather<uint8_t, 16>(zer, gi);
                    sn[g] = es::gather<uint16_t, 16>(sclh, gi * 2u);
                }
                for (int k = kb; k < ke; k += 128) {
                    if (k + 128 < ke)
                        es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (k + 128) / 8, n0, PFH);
                    es::simd<uint32_t, 8 * CT> twh[2];
                    twh[0] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8, n0);
                    twh[1] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8 + 8, n0);
                    es::simd<uint32_t, 16> zc[NGS], scb[NGS];
                    #pragma unroll
                    for (int g = 0; g < NGS; ++g) { zc[g] = zn[g]; scb[g] = sn[g]; }
                    if (k + 128 < ke) {
                        #pragma unroll
                        for (int g = 0; g < NGS; ++g) {
                            const es::simd<uint32_t, 16> gi = gbase + uint32_t((k + 128 + GS * g) / GS);
                            zn[g] = es::gather<uint8_t, 16>(zer, gi);
                            sn[g] = es::gather<uint16_t, 16>(sclh, gi * 2u);
                        }
                    }
                    es::simd<uint32_t, 16> xm;
                    es::simd<float, 16> zf, sc;
                    #pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        if (hf == 0 || GS == 64) {
                            es::simd<uint32_t, 16> zu = zc[GS == 64 ? hf : 0];
                            es::simd<uint32_t, 16> sb = scb[GS == 64 ? hf : 0];
                            // zero 0xff -> signed s4: flip every nibble's bit 3, zero 8
                            es::simd<uint32_t, 16> sg = (zu + 1u) >> 8;
                            xm = sg * 0x88888888u;
                            es::simd<uint32_t, 16> zb = (zu - sg * 247u) | 0x4B000000u;
                            zf = zb.template bit_cast_view<float>().read();
                            es::simd<uint32_t, 16> sw = sb << 16;
                            sc = sw.template bit_cast_view<float>().read();
                        }
                        es::simd<float, 128> tmp[RBN];
                        #pragma unroll
                        for (int rb = 0; rb < RBN; ++rb) tmp[rb] = 0.0f;
                        #pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const int kk = k + 64 * hf + 32 * b;
                            es::simd<sycl_bf16, 128> a0[RBN], a1[RBN];
                            #pragma unroll
                            for (int rb = 0; rb < RBN; ++rb) {
                                a0[rb] = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk, rb * 8);
                                a1[rb] = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk + 16, rb * 8);
                            }
                            es::simd<uint32_t, 128> vb[2];
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                es::simd<uint32_t, 16> wv = twh[hf].template select<16, 1>((4 * b + j) * CT);
                                wv ^= xm;
                                #pragma unroll
                                for (int qb = 0; qb < 4; ++qb) {
                                    es::simd<uint32_t, 16> tb = qb ? (wv >> (8 * qb)) : wv;
                                    es::simd<uint32_t, 16> lo = (tb & 0xFu) | 0x4B000000u;
                                    es::simd<uint32_t, 16> hi = ((tb >> 4) & 0xFu) | 0x4B000000u;
                                    es::simd<float, 16> lf = lo.template bit_cast_view<float>().read() - zf;
                                    es::simd<float, 16> hv = hi.template bit_cast_view<float>().read() - zf;
                                    es::simd<uint32_t, 16> pk =
                                        (lf.template bit_cast_view<uint32_t>().read() >> 16) |
                                        (hv.template bit_cast_view<uint32_t>().read() & 0xFFFF0000u);
                                    const int kp = 4 * j + qb;
                                    vb[kp >> 3].template select<16, 1>((kp & 7) * 16) = pk;
                                }
                            }
                            #pragma unroll
                            for (int rb = 0; rb < RBN; ++rb) {
                                tmp[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                    tmp[rb], vb[0].template bit_cast_view<sycl_bf16>().read(), a0[rb]);
                                tmp[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                    tmp[rb], vb[1].template bit_cast_view<sycl_bf16>().read(), a1[rb]);
                            }
                        }
                        #pragma unroll
                        for (int rb = 0; rb < RBN; ++rb)
                            #pragma unroll
                            for (int r = 0; r < 8; ++r)
                                acc.template select<16, 1>(rb * 128 + 16 * r) +=
                                    tmp[rb].template select<16, 1>(16 * r) * sc;
                    }
                }
            }
            if (KS > 1) {
                es::slm_block_store<float, RBN * 128>(lid * RBN * 128 * 4, acc);
                es::barrier();
                if (ks != 0 || !live) return;
                for (int i = 1; i < KS; ++i)
                    acc += es::slm_block_load<float, RBN * 128>((tw * KS + i) * RBN * 128 * 4);
            } else if (!live) {
                return;
            }
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb)
                es::store_2d<float, 16, 8>(Y, YW, XH, YW, n0, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>(rb * 128)));
        });
    });
}

// int4_smallm_impl with a cheaper dequant (2026-10-09).  The nibbles of a payload dword
// (K offsets 0..7) are turned into bf16 by OR-ing them into 0x4300 -- bf16 (128 + q),
// exact -- four VNNI dwords per payload dword, one shift / and / or each: the pairs come
// out as K (0,4), (1,5), (2,6), (3,7), so each 8-K block of the activation tile is
// reordered the same way in registers ([x0 x4 x1 x5 x2 x6 x3 x7]) before the DPAS.  The
// zero point leaves the inner loop: sum (q - z) x = sum (128 + q) x - (128 + z) sum x, and
// sum x per row and group comes from one more DPAS against an all-ones operand.  ~300
// ALU instructions per thread per 128 K instead of ~720.  Not bit-identical to
// int4_smallm_impl: the group sums round differently (fp32, ~1e-6 relative).
template <int RBN, int GS>
sycl::event int4_smallm_fdq_impl(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                                 float* Y, int M, int KS, int kc, int TPT,
                                 const std::vector<sycl::event>& deps) {
    static_assert(GS == 64 || GS == 128, "INT4 group of 64 or 128");
    constexpr int CT = 16;
    const int N = w.N, K = w.K;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(w.payload);
    const uint16_t* sclh = static_cast<const uint16_t*>(w.scales);
    const uint8_t* zer = w.zeros;
    const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(N) - 1, PP = unsigned(w.row_bytes) - 1;
    const unsigned RS = unsigned(w.row_scales);
    const unsigned XW = unsigned(K) * 2 - 1, XH = unsigned(M) - 1;
    const unsigned YW = unsigned(N) * 4 - 1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS),
                       [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            es::slm_init<32 * RBN * 128 * 4>();
            const int lid = int(it.get_local_id(0));
            const int tw = lid / KS, ks = lid % KS;
            const int tile = int(it.get_group(0)) * TPT + tw;
            const bool live = tile < tiles;
            const int n0 = (live ? tile : 0) * CT;
            const int kb = ks * kc;
            const int ke = kb + kc < K ? kb + kc : K;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            const es::simd<uint32_t, 16> gbase =
                (es::simd<uint32_t, 16>(0, 1) + uint32_t(n0)) * RS;
            // the all-ones B operand (any layout): DPAS against it gives sum x per row
            const es::simd<sycl_bf16, 256> onesb = sycl_bf16(1.0f);
            es::simd<float, RBN * 128> acc = 0.0f;
            if (live) {
                constexpr int NGS = 128 / GS;
                es::simd<uint32_t, 16> zn[NGS], sn[NGS];
                #pragma unroll
                for (int g = 0; g < NGS; ++g) {
                    const es::simd<uint32_t, 16> gi = gbase + uint32_t((kb + GS * g) / GS);
                    zn[g] = es::gather<uint8_t, 16>(zer, gi);
                    sn[g] = es::gather<uint16_t, 16>(sclh, gi * 2u);
                }
                for (int k = kb; k < ke; k += 128) {
                    if (k + 128 < ke)
                        es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (k + 128) / 8, n0, PFH);
                    es::simd<uint32_t, 8 * CT> twh[2];
                    twh[0] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8, n0);
                    twh[1] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8 + 8, n0);
                    es::simd<uint32_t, 16> zc[NGS], scb[NGS];
                    #pragma unroll
                    for (int g = 0; g < NGS; ++g) { zc[g] = zn[g]; scb[g] = sn[g]; }
                    if (k + 128 < ke) {
                        #pragma unroll
                        for (int g = 0; g < NGS; ++g) {
                            const es::simd<uint32_t, 16> gi = gbase + uint32_t((k + 128 + GS * g) / GS);
                            zn[g] = es::gather<uint8_t, 16>(zer, gi);
                            sn[g] = es::gather<uint16_t, 16>(sclh, gi * 2u);
                        }
                    }
                    es::simd<float, 128> tmp[RBN], tsx[RBN];
                    #pragma unroll
                    for (int rb = 0; rb < RBN; ++rb) { tmp[rb] = 0.0f; tsx[rb] = 0.0f; }
                    #pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        es::simd<uint32_t, 16> zu = zc[GS == 64 ? hf : 0];
                        es::simd<uint32_t, 16> sgn = (zu + 1u) >> 8;          // zero 0xff = signed s4
                        const es::simd<uint32_t, 16> xm = sgn * 0x88888888u;
                        #pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const int kk = k + 64 * hf + 32 * b;
                            es::simd<sycl_bf16, 128> a0[RBN], a1[RBN];
                            #pragma unroll
                            for (int rb = 0; rb < RBN; ++rb) {
                                es::simd<sycl_bf16, 128> n0t = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk, rb * 8);
                                es::simd<sycl_bf16, 128> n1t = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk + 16, rb * 8);
                                // [x0 x1 .. x7] -> [x0 x4 x1 x5 x2 x6 x3 x7] in every 8-K block
                                es::simd<uint16_t, 128> s0 = n0t.template bit_cast_view<uint16_t>();
                                es::simd<uint16_t, 128> s1 = n1t.template bit_cast_view<uint16_t>();
                                es::simd<uint16_t, 128> d0, d1;
                                auto sv0 = s0.template bit_cast_view<uint16_t, 16, 8>();
                                auto sv1 = s1.template bit_cast_view<uint16_t, 16, 8>();
                                auto dv0 = d0.template bit_cast_view<uint16_t, 16, 8>();
                                auto dv1 = d1.template bit_cast_view<uint16_t, 16, 8>();
                                dv0.template select<16, 1, 4, 2>(0, 0) = sv0.template select<16, 1, 4, 1>(0, 0);
                                dv0.template select<16, 1, 4, 2>(0, 1) = sv0.template select<16, 1, 4, 1>(0, 4);
                                dv1.template select<16, 1, 4, 2>(0, 0) = sv1.template select<16, 1, 4, 1>(0, 0);
                                dv1.template select<16, 1, 4, 2>(0, 1) = sv1.template select<16, 1, 4, 1>(0, 4);
                                a0[rb] = d0.template bit_cast_view<sycl_bf16>().read();
                                a1[rb] = d1.template bit_cast_view<sycl_bf16>().read();
                            }
                            es::simd<uint32_t, 128> vb[2];
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                es::simd<uint32_t, 16> wv = twh[hf].template select<16, 1>((4 * b + j) * CT);
                                wv ^= xm;
                                #pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    const int kp = 4 * j + i;
                                    es::simd<uint32_t, 16> t = i ? (wv >> (4 * i)) : wv;
                                    vb[kp >> 3].template select<16, 1>((kp & 7) * 16) =
                                        (t & 0x000F000Fu) | 0x43004300u;
                                }
                            }
                            #pragma unroll
                            for (int rb = 0; rb < RBN; ++rb) {
                                tmp[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                    tmp[rb], vb[0].template bit_cast_view<sycl_bf16>().read(), a0[rb]);
                                tmp[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                    tmp[rb], vb[1].template bit_cast_view<sycl_bf16>().read(), a1[rb]);
                                tsx[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                    tsx[rb], onesb, a0[rb]);
                                tsx[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                    tsx[rb], onesb, a1[rb]);
                            }
                        }
                        // end of a group: acc += s * (T - (128 + z) * sum x)
                        if (GS == 64 || hf == 1) {
                            const es::simd<uint32_t, 16> zz = zu - sgn * 247u;      // 0xff -> 8
                            const es::simd<float, 16> zp = es::simd<float, 16>(zz) + 128.0f;
                            es::simd<uint32_t, 16> sw = scb[GS == 64 ? hf : 0] << 16;
                            const es::simd<float, 16> sc = sw.template bit_cast_view<float>().read();
                            #pragma unroll
                            for (int rb = 0; rb < RBN; ++rb) {
                                #pragma unroll
                                for (int r = 0; r < 8; ++r)
                                    acc.template select<16, 1>(rb * 128 + 16 * r) +=
                                        (tmp[rb].template select<16, 1>(16 * r) -
                                         zp * tsx[rb].template select<16, 1>(16 * r)) * sc;
                                tmp[rb] = 0.0f;
                                tsx[rb] = 0.0f;
                            }
                        }
                    }
                }
            }
            if (KS > 1) {
                es::slm_block_store<float, RBN * 128>(lid * RBN * 128 * 4, acc);
                es::barrier();
                if (ks != 0 || !live) return;
                for (int i = 1; i < KS; ++i)
                    acc += es::slm_block_load<float, RBN * 128>((tw * KS + i) * RBN * 128 * 4);
            } else if (!live) {
                return;
            }
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb)
                es::store_2d<float, 16, 8>(Y, YW, XH, YW, n0, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>(rb * 128)));
        });
    });
}

// FP8 E4M3 weights with GRIMOIRE's per-output-channel fp32 scale, the same
// small-M contract and thread design as int4_smallm_impl (16 columns per
// thread, transposed 2-D payload loads, the bf16 rows of X through DPAS).
// FP8 has one byte per weight, so a 128-K step is four 8-dword transposed
// loads (32 K each).  E4M3 -> bf16 is exact: the bits move into fp16
// ((b & 0x7F) << 7 | (b & 0x80) << 8 = value / 256), fp16 -> fp32 is exact,
// and the fp32 top half is the bf16 (at most 4 significant bits).  One scale
// per row, so the DPAS results accumulate unscaled and the row scale (x 256)
// is applied once at the end.  Only rounding: DPAS's and the bf16 X.
template <int RBN>
sycl::event fp8_smallm_impl(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                            float* Y, int M, int KS, int kc, int TPT,
                            const std::vector<sycl::event>& deps) {
    constexpr int CT = 16;
    const int N = w.N, K = w.K;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(w.payload);
    const float* scl = static_cast<const float*>(w.scales);
    const unsigned PW = unsigned(K) - 1, PH = unsigned(N) - 1, PP = unsigned(w.row_bytes) - 1;
    const unsigned XW = unsigned(K) * 2 - 1, XH = unsigned(M) - 1;
    const unsigned YW = unsigned(N) * 4 - 1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS),
                       [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            es::slm_init<32 * RBN * 128 * 4>();
            const int lid = int(it.get_local_id(0));
            const int tw = lid / KS, ks = lid % KS;
            const int tile = int(it.get_group(0)) * TPT + tw;
            const bool live = tile < tiles;
            const int n0 = (live ? tile : 0) * CT;
            const int kb = ks * kc;
            const int ke = kb + kc < K ? kb + kc : K;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            es::simd<float, RBN * 128> acc = 0.0f;
            if (live) {
                for (int k = kb; k < ke; k += 128) {
                    if (k + 128 < ke) {
                        es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (k + 128) / 4, n0, PFH);
                        es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (k + 128) / 4 + 16, n0, PFH);
                    }
                    es::simd<uint32_t, 8 * CT> tq[4];
                    #pragma unroll
                    for (int s4 = 0; s4 < 4; ++s4)
                        tq[s4] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 4 + 8 * s4, n0);
                    #pragma unroll
                    for (int s4 = 0; s4 < 4; ++s4) {       // 32 K per sub-step
                        const int kk = k + 32 * s4;
                        es::simd<sycl_bf16, 128> a0[RBN], a1[RBN];
                        #pragma unroll
                        for (int rb = 0; rb < RBN; ++rb) {
                            a0[rb] = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk, rb * 8);
                            a1[rb] = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk + 16, rb * 8);
                        }
                        es::simd<uint32_t, 128> vb[2];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {          // dword j = k 4j .. 4j+3
                            const es::simd<uint32_t, 16> wv = tq[s4].template select<16, 1>(j * CT);
                            #pragma unroll
                            for (int qb = 0; qb < 2; ++qb) {   // k-pair (4j + 2qb, 4j + 2qb + 1)
                                const es::simd<uint32_t, 16> b0 = (wv >> (16 * qb)) & 0xFFu;
                                const es::simd<uint32_t, 16> b1 = (wv >> (16 * qb + 8)) & 0xFFu;
                                es::simd<uint16_t, 16> h0 = ((b0 & 0x7Fu) << 7) | ((b0 & 0x80u) << 8);
                                es::simd<uint16_t, 16> h1 = ((b1 & 0x7Fu) << 7) | ((b1 & 0x80u) << 8);
                                es::simd<float, 16> f0 = h0.template bit_cast_view<sycl::half>().read();
                                es::simd<float, 16> f1 = h1.template bit_cast_view<sycl::half>().read();
                                const es::simd<uint32_t, 16> pk =
                                    (f0.template bit_cast_view<uint32_t>().read() >> 16) |
                                    (f1.template bit_cast_view<uint32_t>().read() & 0xFFFF0000u);
                                const int kp = 2 * j + qb;
                                vb[kp >> 3].template select<16, 1>((kp & 7) * 16) = pk;
                            }
                        }
                        #pragma unroll
                        for (int rb = 0; rb < RBN; ++rb) {
                            es::simd<float, 128> c = acc.template select<128, 1>(rb * 128);
                            c = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                c, vb[0].template bit_cast_view<sycl_bf16>().read(), a0[rb]);
                            c = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                c, vb[1].template bit_cast_view<sycl_bf16>().read(), a1[rb]);
                            acc.template select<128, 1>(rb * 128) = c;
                        }
                    }
                }
            }
            if (KS > 1) {
                es::slm_block_store<float, RBN * 128>(lid * RBN * 128 * 4, acc);
                es::barrier();
                if (ks != 0 || !live) return;
                for (int i = 1; i < KS; ++i)
                    acc += es::slm_block_load<float, RBN * 128>((tw * KS + i) * RBN * 128 * 4);
            } else if (!live) {
                return;
            }
            const es::simd<float, 16> sc = es::block_load<float, 16>(scl + n0) * 256.0f;
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb) {
                #pragma unroll
                for (int r = 0; r < 8; ++r)
                    acc.template select<16, 1>(rb * 128 + 16 * r) *= sc;
                es::store_2d<float, 16, 8>(Y, YW, XH, YW, n0, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>(rb * 128)));
            }
        });
    });
}
} // namespace
} // namespace b70

#endif
