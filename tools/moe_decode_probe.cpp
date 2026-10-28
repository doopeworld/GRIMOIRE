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

// moe_decode_probe.cpp -- the decode step's ESIMD MoE kernels as batched
// decode runs them (moe_kernels.cpp moe_gu_esimd_rows / moe_dn_esimd_rows:
// every token's 8 routed experts + the shared expert, weights per (token,
// expert) pair) on Ornith's shapes, weights streamed from DRAM, a new random
// routing per launch.  DEC=1 replaces the E2M1 decode by a plain conversion
// (wrong math, same memory traffic): if that runs much faster, the kernels
// are ALU-bound and a cheaper decode pays at every batch size.
//
// Build (from the repo root, grimoire-dev):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 \
//     -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -cl-intel-256-GRF-per-thread" \
//     -O3 -std=c++20 tools/moe_decode_probe.cpp -o bin/moe_decode_probe
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>

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
SYCL_ESIMD_FUNCTION inline float e8m0_x2p14(uint32_t e) {
    return sycl::bit_cast<float>((e + 14u) << 23);
}
template <int KP, int DEC>
SYCL_ESIMD_FUNCTION inline void mx4_row_step(const uint8_t* prow, const uint8_t* srow, int st,
                                             es::simd<float, KP / 2>& xe,
                                             es::simd<float, KP / 2>& xo,
                                             es::simd<float, 16>& acc) {
    es::simd<uint8_t, 64> pb = es::block_load<uint8_t, 64>(prow + st * 64);
    es::simd<uint8_t, 4> sb = es::block_load<uint8_t, 4>(srow + st * 4);
    es::simd<float, 64> wl, wh;
    if constexpr (DEC == 0) e2m1_split<64>(pb, wl, wh);
    else { wl = pb; wh = wl; }
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        es::simd<float, 16> part =
            wl.template select<16, 1>(16 * b) * xe.template select<16, 1>(st * 64 + 16 * b) +
            wh.template select<16, 1>(16 * b) * xo.template select<16, 1>(st * 64 + 16 * b);
        acc += part * e8m0_x2p14(uint32_t(sb[b]));
    }
}

struct Mx { const uint8_t* pay; const uint8_t* scl; int64_t rb, rs; };

template <int R, int KS, int H, int DEC>
sycl::event gu_rows(sycl::queue& q, Mx w, Mx ws, const int32_t* d_expert, const float* x,
                    float* h, int I, int TK, int M) {
    constexpr int KP = H / KS;
    const int rg = I / R;
    const int per_tok = (TK + 1) * rg;
    return q.parallel_for(sycl::nd_range<1>(size_t(M) * per_tok * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<(KS * R * 2 * 4 > KS * 16 ? KS * R * 2 * 4 : KS * 16)>();
        const int t = int(it.get_local_id(0));
        const int m = int(it.get_group(0)) / per_tok;
        const int g = int(it.get_group(0)) % per_tok;
        const int kb = t * KP;
        const float* xm = x + int64_t(m) * H;
        const int slot = g / rg, i0 = (g % rg) * R;
        const bool sh = slot == TK;
        const Mx& W = sh ? ws : w;
        int e = sh ? 0 : d_expert[int64_t(m) * TK + slot];
        if (e < 0) e = 0;
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
                mx4_row_step<KP, DEC>(W.pay + gr * W.rb + kb / 2, W.scl + gr * W.rs + kb / 32, st, xe, xo, ga[r]);
                mx4_row_step<KP, DEC>(W.pay + ur * W.rb + kb / 2, W.scl + ur * W.rs + kb / 32, st, xe, xo, ua[r]);
            }
        }
        es::simd<float, 2 * R> red;
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            red[2 * r] = es::reduce<float>(ga[r], std::plus<>());
            red[2 * r + 1] = es::reduce<float>(ua[r], std::plus<>());
        }
        es::slm_block_store<float, 2 * R>(t * 2 * R * 4, red);
        es::barrier();
        if (t != 0) return;
        red = 0.0f;
        #pragma unroll
        for (int j = 0; j < KS; ++j) red += es::slm_block_load<float, 2 * R>(j * 2 * R * 4);
        float* hm = h + int64_t(m) * (TK + 1) * I;
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const float gv = red[2 * r], uv = red[2 * r + 1];
            hm[int64_t(slot) * I + i0 + r] = gv / (1.0f + sycl::exp(-gv)) * uv;
        }
    });
}

template <int R, int I, int DEC>
sycl::event dn_rows(sycl::queue& q, Mx w, Mx wd, const int32_t* d_expert, const float* d_weight,
                    const float* h, float* y, int H, int TK, int M) {
    constexpr int RP = R < 4 ? 4 : R;
    const int S = TK + 1;
    const int per_tok = H / R;
    return q.parallel_for(sycl::nd_range<1>(size_t(M) * per_tok * S, S), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<16 * RP * 4>();
        const int s = int(it.get_local_id(0));
        const int m = int(it.get_group(0)) / per_tok;
        const int o0 = (int(it.get_group(0)) % per_tok) * R;
        const bool sh = s == TK;
        const Mx& W = sh ? wd : w;
        int e = sh ? 0 : d_expert[int64_t(m) * TK + s];
        float wt = sh ? 1.0f : d_weight[int64_t(m) * TK + s];
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
                mx4_row_step<I, DEC>(W.pay + (row0 + r) * W.rb, W.scl + (row0 + r) * W.rs, st, he, ho, acc[r]);
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
}

static int run(int argc, char** argv);
int main(int argc, char** argv) {
    // never abort with device work queued: a crashed process leaves the
    // driver to clean up its jobs (xe "LR job cleanup" + engine reset)
    try { return run(argc, argv); }
    catch (const std::exception& ex) {
        std::fprintf(stderr, "probe failed: %s\n", ex.what());
        std::fflush(stderr);
        return 3;
    }
}
static int run(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
    const int E = 256, H = 2048, I = 512, TK = 8, NC = 3;
    std::mt19937 rng(5);
    auto mk = [&](size_t rows, int K, std::vector<Mx>& out) {
        const size_t pb = rows * size_t(K) / 2, sb = rows * size_t(K) / 32;
        std::vector<uint8_t> hp(pb), hs(sb);
        for (auto& v : hp) v = uint8_t(rng());
        for (auto& v : hs) v = uint8_t(118 + rng() % 12);
        for (int c = 0; c < NC; ++c) {
            uint8_t* p = sycl::malloc_device<uint8_t>(pb, q); uint8_t* s = sycl::malloc_device<uint8_t>(sb, q);
            if (!p || !s) { std::printf("alloc failed: %zu + %zu bytes\n", pb, sb); throw std::runtime_error("alloc"); }
            // in 32 MB pieces, each waited on
            for (size_t o = 0; o < pb; o += size_t(32) << 20)
                q.memcpy(p + o, hp.data() + o, std::min(size_t(32) << 20, pb - o)).wait();
            q.memcpy(s, hs.data(), sb).wait();
            out.push_back({p, s, int64_t(K / 2), int64_t(K / 32)});
            std::printf("uploaded %zu MB\n", (pb + sb) >> 20);
        }
    };
    std::vector<Mx> gu, sgu, dn, sdn;
    mk(size_t(E) * 2 * I, H, gu); mk(size_t(2 * I), H, sgu);
    mk(size_t(E) * H, I, dn); mk(size_t(H), I, sdn);
    q.wait();
    const double pair_bytes = double(2 * I) * H * (0.5 + 1.0 / 32) + double(H) * I * (0.5 + 1.0 / 32);
    for (int M : {1, 2, 4, 8}) {
        constexpr int NR = 16;
        std::vector<int32_t*> drex(NR); std::vector<float*> drwt(NR);
        for (int i = 0; i < NR; ++i) {
            std::vector<int32_t> rex; std::vector<float> rwt;
            for (int m = 0; m < M; ++m) {
                std::set<int> s; while (int(s.size()) < TK) s.insert(int(rng() % E));
                for (int e : s) { rex.push_back(e); rwt.push_back(1.0f / TK); }
            }
            drex[i] = sycl::malloc_device<int32_t>(rex.size(), q); drwt[i] = sycl::malloc_device<float>(rwt.size(), q);
            q.memcpy(drex[i], rex.data(), rex.size() * 4); q.memcpy(drwt[i], rwt.data(), rwt.size() * 4);
        }
        std::vector<float> hx(size_t(M) * H);
        std::uniform_real_distribution<float> U(-1.f, 1.f);
        for (auto& v : hx) v = U(rng);
        float* dx = sycl::malloc_device<float>(hx.size(), q);
        float* dh = sycl::malloc_device<float>(size_t(M) * (TK + 1) * I, q);
        float* dy = sycl::malloc_device<float>(size_t(M) * H, q);
        q.memcpy(dx, hx.data(), hx.size() * 4).wait();
        for (int dec = 0; dec < 2; ++dec) {
            for (int which = 0; which < 2; ++which) {
                int ci = 0;
                auto launch = [&](int ri) {
                    const int c = ci++ % NC;
                    if (which == 0)
                        return dec ? gu_rows<2, 8, 2048, 1>(q, gu[c], sgu[c], drex[ri], dx, dh, I, TK, M)
                                   : gu_rows<2, 8, 2048, 0>(q, gu[c], sgu[c], drex[ri], dx, dh, I, TK, M);
                    return dec ? dn_rows<2, 512, 1>(q, dn[c], sdn[c], drex[ri], drwt[ri], dh, dy, H, TK, M)
                               : dn_rows<2, 512, 0>(q, dn[c], sdn[c], drex[ri], drwt[ri], dh, dy, H, TK, M);
                };
                for (int i = 0; i < 3; ++i) launch(i % NR);
                q.wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i) launch(i % NR);
                q.wait();
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
                const double bytes = which == 0 ? double(M) * (TK + 1) * double(2 * I) * H * (0.5 + 1.0 / 32)
                                                : double(M) * (TK + 1) * double(H) * I * (0.5 + 1.0 / 32);
                std::printf("%s M=%d %-9s %8.1f us  %6.1f GB/s (per-pair bytes)\n", which == 0 ? "gate_up" : "down   ",
                            M, dec ? "NODECODE" : "decode", us, bytes / (us * 1e3));
            }
        }
        (void)pair_bytes;
        for (int i = 0; i < NR; ++i) { sycl::free(drex[i], q); sycl::free(drwt[i], q); }
        sycl::free(dx, q); sycl::free(dh, q); sycl::free(dy, q);
    }
    return 0;
}
