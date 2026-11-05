// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// mx4rows_probe.cpp -- MXFP4 small-M GEMM (speculative verify, batched decode) on
// Qwen3.8-27B and Ornith dense shapes (2026-11-05): the library kernel
// (launch_mxfp4_smallm: 256 registers, its own split-K plan) against the engine's
// 128-register port (launch_mxfp4_smallm_fast: the INT4 plan), and local copies of the
// 128-register kernel with other dequants:
//   DEC 0  no dequant: payload dwords straight into DPAS (wrong numbers; loads + DPAS bound)
//   DEC 1  the shipping dequant (nibble -> fp16 -> fp32 -> bf16, byte pairs (k, k+1))
//   DEC 2  nibble pairs (k, k+4) as INT4's fast dequant, X reordered to match
//   DEC 3  a 256-entry SLM table: payload byte -> its two bf16 values (same bits as DEC 1)
// Weights rotate over copies so they stream from DRAM.  Check: the decode GEMV (fp32) per
// row, with X exactly representable in bf16, so only the summation order differs.
// Argument: index of one shape (default: all).
//
// Build (grimoire-dev, from the repo root):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -fno-fast-math \
//     -ffp-contract=fast -fno-math-errno -fsycl-device-code-split=per_kernel \
//     -I include -I src tools/mx4rows_probe.cpp src/gemv_decode.cpp \
//     -Lbin -lgrimoire_gemm '-Wl,-rpath,$ORIGIN' -o bin/mx4rows_probe
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include "b70/weights.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

using namespace b70;

namespace {
struct Plan { int KS, kc, TPT; };
Plan plan_ks(int N, int K, int ks) {   // ks <= 0: the INT4 rule
    const int tiles = N / 16;
    int KS = ks > 0 ? ks : (tiles >= 1024 || K >= 12288 ? 4 : std::min(32, std::max(1, K / 256)));
    int kc = (K + KS - 1) / KS;
    kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= 32) TPT *= 2;
    return {KS, kc, TPT};
}

template <int DEC>
sycl::event mx_local(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X, float* Y, int M, int ks) {
    constexpr int CT = 16;
    constexpr int TBL = DEC == 3 ? 1024 : 0;      // SLM: the byte table first, then the K-slice sums
    const int N = w.N, K = w.K;
    const Plan p = plan_ks(N, K, ks);
    const int KS = p.KS, kc = p.kc, TPT = p.TPT;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(w.payload);
    const uint32_t* sclw = static_cast<const uint32_t*>(w.scales);
    const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(N) - 1, PP = unsigned(w.row_bytes) - 1;
    const unsigned SW = unsigned(K) / 32 - 1, SP = unsigned(w.row_scales) - 1;
    const unsigned XW = unsigned(K) * 2 - 1, XH = unsigned(M) - 1;
    const unsigned YW = unsigned(N) * 4 - 1;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS),
                       [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            es::slm_init<TBL + 32 * 128 * 4>();
            const int lid = int(it.get_local_id(0));
            if constexpr (DEC == 3) {
                if (lid == 0) {
                    // bf16 bits of code n's value * 2^-14, made exactly as DEC 1 makes them
                    es::simd<uint32_t, 16> n16(0, 1);
                    es::simd<uint32_t, 16> hb = ((n16 & 7u) << 9) | ((n16 & 8u) << 12);
                    es::simd<uint16_t, 16> h16 = hb;
                    es::simd<sycl::half, 16> hv = h16.template bit_cast_view<sycl::half>().read();
                    es::simd<float, 16> fv = hv;
                    es::simd<uint32_t, 16> bf = fv.template bit_cast_view<uint32_t>().read() >> 16;
                    #pragma unroll
                    for (int c = 0; c < 16; ++c) {           // byte = (hi << 4) | lo
                        es::simd<uint32_t, 16> e = bf | (uint32_t(bf[c]) << 16);
                        es::slm_block_store<uint32_t, 16>(c * 64, e);
                    }
                }
                es::barrier();
            }
            const int tw = lid / KS, ks2 = lid % KS;
            const int tile = int(it.get_group(0)) * TPT + tw;
            const bool live = tile < tiles;
            const int n0 = (live ? tile : 0) * CT;
            const int kb = ks2 * kc;
            const int ke = kb + kc < K ? kb + kc : K;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            es::simd<float, 128> acc = 0.0f;
            if (live) {
                for (int k = kb; k < ke; k += 128) {
                    if (k + 128 < ke)
                        es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (k + 128) / 8, n0, PFH);
                    es::simd<uint32_t, 8 * CT> tw0 =
                        es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8, n0);
                    es::simd<uint32_t, 8 * CT> tw1 =
                        es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8 + 8, n0);
                    es::simd<uint32_t, CT> sw = es::load_2d<uint32_t, 1, CT>(sclw, SW, PH, SP, k / 128, n0);
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const int kk = k + 32 * b;
                        es::simd<sycl_bf16, 128> a0, a1;
                        if constexpr (DEC == 2) {
                            es::simd<sycl_bf16, 128> n0t = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk, 0);
                            es::simd<sycl_bf16, 128> n1t = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk + 16, 0);
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
                            a0 = d0.template bit_cast_view<sycl_bf16>().read();
                            a1 = d1.template bit_cast_view<sycl_bf16>().read();
                        } else {
                            a0 = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk, 0);
                            a1 = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk + 16, 0);
                        }
                        es::simd<uint32_t, 16> ev = (sw >> (8 * b)) & 0xFFu;
                        es::simd<uint32_t, 16> sbits = (ev + 14u) << 23;
                        es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                        es::simd<uint32_t, 128> vb[2];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int dw = 4 * b + j;
                            es::simd<uint32_t, 16> wv;
                            if (dw < 8) wv = tw0.template select<16, 1>(dw * CT);
                            else        wv = tw1.template select<16, 1>((dw - 8) * CT);
                            #pragma unroll
                            for (int qb = 0; qb < 4; ++qb) {
                                const int kp = 4 * j + qb;
                                es::simd<uint32_t, 16> pk;
                                if constexpr (DEC == 0) {
                                    pk = qb ? (wv >> (8 * qb)) : wv;
                                } else if constexpr (DEC == 1) {
                                    es::simd<uint32_t, 16> tb = qb ? (wv >> (8 * qb)) : wv;
                                    es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                                    es::simd<uint32_t, 16> hb = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                                    es::simd<sycl::half, 32> hv = hb.template bit_cast_view<sycl::half>().read();
                                    es::simd<float, 32> fv = hv;
                                    es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                                    es::simd<uint16_t, 32> bb = fb >> 16;
                                    pk = bb.template bit_cast_view<uint32_t>().read();
                                } else if constexpr (DEC == 2) {
                                    es::simd<uint32_t, 16> u = (qb ? (wv >> (4 * qb)) : wv) & 0x000F000Fu;
                                    es::simd<uint32_t, 16> hb = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                                    es::simd<sycl::half, 32> hv = hb.template bit_cast_view<sycl::half>().read();
                                    es::simd<float, 32> fv = hv;
                                    es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                                    es::simd<uint16_t, 32> bb = fb >> 16;
                                    pk = bb.template bit_cast_view<uint32_t>().read();
                                } else {
                                    es::simd<uint32_t, 16> off = qb == 0 ? ((wv << 2) & 0x3FCu)
                                                                         : ((wv >> (8 * qb - 2)) & 0x3FCu);
                                    pk = es::slm_gather<uint32_t, 16>(off);
                                }
                                vb[kp >> 3].template select<16, 1>((kp & 7) * 16) = pk;
                            }
                        }
                        es::simd<float, 128> tmp = 0.0f;
                        tmp = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                            tmp, vb[0].template bit_cast_view<sycl_bf16>().read(), a0);
                        tmp = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                            tmp, vb[1].template bit_cast_view<sycl_bf16>().read(), a1);
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            acc.template select<16, 1>(16 * r) += tmp.template select<16, 1>(16 * r) * sc;
                    }
                }
            }
            if (KS > 1) {
                es::slm_block_store<float, 128>(TBL + lid * 128 * 4, acc);
                es::barrier();
                if (ks2 != 0 || !live) return;
                for (int i = 1; i < KS; ++i)
                    acc += es::slm_block_load<float, 128>(TBL + (tw * KS + i) * 128 * 4);
            } else if (!live) {
                return;
            }
            es::store_2d<float, 16, 8>(Y, YW, XH, YW, n0, 0, acc);
        });
    });
}
} // namespace

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    struct Shape { const char* nm; int N, K; };
    const std::vector<Shape> shapes = {
        {"q38 gate_up", 34816, 5120}, {"q38 down", 5120, 17408}, {"q38 dn_qkv", 10240, 5120},
        {"q38 z", 6144, 5120}, {"q38 out/o", 5120, 6144}, {"q38 attn_q", 12288, 5120},
        {"q38 kv", 1024, 5120}, {"q38 lm_head", 248320, 5120},
        {"orn la_qkv", 8192, 2048}, {"orn z", 4096, 2048}, {"orn out", 2048, 4096},
        {"orn lm_head", 248320, 2048}};
    const int only = argc > 1 ? std::atoi(argv[1]) : -1;
    std::mt19937 rng(11);
    for (int si = 0; si < int(shapes.size()); ++si) {
        if (only >= 0 && only != si) continue;
        const Shape sh = shapes[si];
        const int N = sh.N, K = sh.K, rsc = K / 32;
        const size_t pb = size_t(N) * K / 2, sb = size_t(N) * rsc;
        const size_t wb = pb + sb;
        const int copies = std::max(2, std::min(8, int((1536ull << 20) / wb)));
        std::vector<uint8_t> hp(pb), hs(sb);
        for (auto& v : hp) v = uint8_t(rng());
        for (auto& v : hs) v = uint8_t(118 + rng() % 8);
        std::vector<QuantWeight> ws(copies);
        std::vector<void*> allocs;
        for (int c = 0; c < copies; ++c) {
            uint8_t* p = sycl::malloc_device<uint8_t>(pb, q);
            uint8_t* s = sycl::malloc_device<uint8_t>(sb, q);
            q.memcpy(p, hp.data(), pb); q.memcpy(s, hs.data(), sb);
            allocs.push_back(p); allocs.push_back(s);
            QuantWeight& w = ws[c];
            w.fmt = Fmt::MXFP4; w.N = N; w.K = K; w.payload = p; w.scales = s;
            w.row_bytes = K / 2; w.row_scales = rsc;
        }
        constexpr int MR = 16;
        std::vector<float> hx(size_t(MR) * K);
        std::vector<sycl_bf16> hxb(hx.size());
        for (size_t i = 0; i < hx.size(); ++i) {
            hxb[i] = sycl_bf16(float(int(rng() % 2001) - 1000) / 1000.0f);
            hx[i] = float(hxb[i]);
        }
        float* x = sycl::malloc_device<float>(hx.size(), q);
        sycl_bf16* xb = sycl::malloc_device<sycl_bf16>(hx.size(), q);
        float* y = sycl::malloc_device<float>(size_t(MR) * N, q);
        float* y2 = sycl::malloc_device<float>(size_t(MR) * N, q);
        float* yr = sycl::malloc_device<float>(size_t(MR) * N, q);
        q.memcpy(x, hx.data(), hx.size() * 4); q.memcpy(xb, hxb.data(), hxb.size() * 2); q.wait();
        if (!mxfp4_smallm_ok(ws[0], MR, xb, y)) { std::printf("%s: mxfp4_smallm_ok refuses\n", sh.nm); continue; }
        const Plan pr = plan_ks(N, K, 0);
        std::printf("\n%-12s N=%-6d K=%-5d  %.1f MB/copy x%d  INT4 rule KS=%d kc=%d TPT=%d\n",
                    sh.nm, N, K, wb / 1e6, copies, pr.KS, pr.kc, pr.TPT);
        auto timeit = [&](const char* nm, const std::function<void(const QuantWeight&)>& f) {
            for (int i = 0; i < 2 * copies; ++i) f(ws[i % copies]);
            q.wait();
            const int it = std::max(20, 4 * copies);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i) f(ws[i % copies]);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
            std::printf("  %-34s %8.1f us  %6.1f GB/s\n", nm, us, wb / us / 1e3);
            return us;
        };
        auto check = [&](const char* nm, const float* a_d, const float* b_d, int M) {
            std::vector<float> a(size_t(M) * N), b(size_t(M) * N);
            q.memcpy(a.data(), a_d, a.size() * 4); q.memcpy(b.data(), b_d, b.size() * 4); q.wait();
            double mx = 0, md = 0; size_t bad = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                if (!std::isfinite(a[i])) ++bad;
                mx = std::max(mx, double(std::fabs(b[i]))); md = std::max(md, double(std::fabs(a[i] - b[i])));
            }
            std::printf("  check %-28s max rel %.2e%s\n", nm, mx > 0 ? md / mx : md, bad ? "  NON-FINITE" : "");
        };
        for (int m = 0; m < MR; ++m) launch_gemv(q, ws[0], x + size_t(m) * K, yr + size_t(m) * N);
        q.wait();
        timeit("gemv M=1 (engine decode)", [&](const QuantWeight& w) { launch_gemv(q, w, x, y); });
        launch_mxfp4_smallm(q, ws[0], xb, y, 8); check("library M=8 vs gemv", y, yr, 8);
        launch_mxfp4_smallm(q, ws[0], xb, y, MR); check("library M=16 vs gemv", y, yr, MR);
        launch_mxfp4_smallm_fast(q, ws[0], xb, y2, 8); check("fast M=8 vs gemv", y2, yr, 8);
        launch_mxfp4_smallm(q, ws[0], xb, y, 8); check("fast vs library M=8", y2, y, 8);
        for (int M : {1, 2, 4, 7, 8, 12, 14, 16}) {
            char nm[64];
            std::snprintf(nm, sizeof nm, "library (256 GRF) M=%d", M);
            timeit(nm, [&](const QuantWeight& w) { launch_mxfp4_smallm(q, w, xb, y, M); });
            if (M > 8) continue;
            std::snprintf(nm, sizeof nm, "fast (128 GRF, INT4 plan) M=%d", M);
            timeit(nm, [&](const QuantWeight& w) { launch_mxfp4_smallm_fast(q, w, xb, y, M); });
        }
        // y holds the last timed call: fast at M = 8 (every copy holds the same bytes); since
        // 2026-11-05 the engine's fast kernel IS DEC 2
        launch_mxfp4_smallm_fast(q, ws[0], xb, y, 8);
        mx_local<2>(q, ws[0], xb, y2, 8, 0); check("local DEC2 vs fast (expect 0)", y2, y, 8);
        mx_local<2>(q, ws[0], xb, y, 8, 0); check("local DEC2 vs gemv", y, yr, 8);
        mx_local<3>(q, ws[0], xb, y, 8, 0); check("local DEC3 vs gemv", y, yr, 8);
        for (int dec = 0; dec <= 3; ++dec) {
            char nm[64];
            std::snprintf(nm, sizeof nm, "local DEC%d M=7 rule", dec);
            auto f = [&](const QuantWeight& w) {
                if (dec == 0) mx_local<0>(q, w, xb, y, 7, 0);
                else if (dec == 1) mx_local<1>(q, w, xb, y, 7, 0);
                else if (dec == 2) mx_local<2>(q, w, xb, y, 7, 0);
                else mx_local<3>(q, w, xb, y, 7, 0);
            };
            timeit(nm, f);
        }
        if (std::getenv("SWEEP")) {
            int last = -1;
            for (int ks : {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32}) {
                const Plan pl = plan_ks(N, K, ks);
                if (pl.KS == last) continue;
                last = pl.KS;
                char nm[64];
                std::snprintf(nm, sizeof nm, "DEC2 M=7 KS=%d kc=%d TPT=%d", pl.KS, pl.kc, pl.TPT);
                timeit(nm, [&](const QuantWeight& w) { mx_local<2>(q, w, xb, y, 7, ks); });
                std::snprintf(nm, sizeof nm, "DEC2 M=4 KS=%d kc=%d TPT=%d", pl.KS, pl.kc, pl.TPT);
                timeit(nm, [&](const QuantWeight& w) { mx_local<2>(q, w, xb, y, 4, ks); });
            }
        }
        for (void* p : allocs) sycl::free(p, q);
        sycl::free(x, q); sycl::free(xb, q); sycl::free(y, q); sycl::free(y2, q); sycl::free(yr, q);
    }
    return 0;
}
