// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// fp8_probe.cpp -- the FP8 E4M3 decode GEMV (launch_gemv: esgemv2_fp8 unless GRIMOIRE_ESGEMV_FP8=0)
// against a double-precision CPU reference on Qwen3.8-27B's shapes, and its weight GB/s with the
// weights streamed from DRAM (rotating copies).
// Build (grimoire-dev, repo root):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -fno-fast-math -ffp-contract=fast \
//     -fno-math-errno -fsycl-device-code-split=per_kernel -I include -I src tools/fp8_probe.cpp \
//     src/gemv_decode.cpp -Lbin -lgrimoire_gemm '-Wl,-rpath,$ORIGIN' -o bin/fp8_probe
#include <sycl/sycl.hpp>
#include "b70/weights.hpp"
#include "kernels.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace b70;
static double e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
    return s ? -v : v;
}
int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    struct S { const char* nm; int N, K; };
    const S shapes[] = {{"gate_up", 34816, 5120}, {"down", 5120, 17408}, {"dn_qkv", 10240, 5120},
                        {"out", 5120, 6144}, {"lm_head", 248320, 5120}};
    std::mt19937 rng(11);
    for (const S& sh : shapes) {
        const int N = sh.N, K = sh.K;
        const size_t pb = size_t(N) * K;
        const int copies = std::max(1, std::min(6, int((1500ull << 20) / pb)));
        std::vector<uint8_t> hp(pb);
        for (auto& v : hp) { v = uint8_t(rng()); if ((v & 0x7F) == 0x7F) v &= 0xFE; }   // no NaN codes
        std::vector<float> hs(N), hx(K);
        for (auto& v : hs) v = 0.001f * float(1 + rng() % 64);
        for (auto& v : hx) v = float(int(rng() % 2001) - 1000) / 1000.0f;
        std::vector<QuantWeight> ws(copies); std::vector<void*> mem;
        for (int c = 0; c < copies; ++c) {
            uint8_t* p = sycl::malloc_device<uint8_t>(pb, q); float* s = sycl::malloc_device<float>(N, q);
            q.memcpy(p, hp.data(), pb); q.memcpy(s, hs.data(), size_t(N) * 4); mem.push_back(p); mem.push_back(s);
            QuantWeight& w = ws[c]; w.fmt = Fmt::FP8_E4M3; w.N = N; w.K = K; w.payload = p; w.scales = s;
            w.row_bytes = K; w.row_scales = 1;
        }
        float* x = sycl::malloc_device<float>(K, q); float* y = sycl::malloc_device<float>(N, q);
        q.memcpy(x, hx.data(), size_t(K) * 4).wait();
        launch_gemv(q, ws[0], x, y); q.wait();
        std::vector<float> hy(N); q.memcpy(hy.data(), y, size_t(N) * 4).wait();
        double mx = 0, md = 0;
        for (int n = 0; n < N; n += (N > 50000 ? 7 : 1)) {
            double acc = 0; const uint8_t* row = hp.data() + size_t(n) * K;
            for (int k = 0; k < K; ++k) acc += e4m3(row[k]) * hx[k];
            acc *= hs[n];
            mx = std::max(mx, std::fabs(acc)); md = std::max(md, std::fabs(acc - double(hy[n])));
        }
        for (int i = 0; i < 2 * copies; ++i) launch_gemv(q, ws[i % copies], x, y);
        q.wait();
        const int it = std::max(20, 4 * copies);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) launch_gemv(q, ws[i % copies], x, y);
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        std::printf("%-8s N=%-6d K=%-5d  rel err %.2e  %8.1f us  %6.1f GB/s  (M=1 GEMV)\n", sh.nm, N, K, mx > 0 ? md / mx : md, us, pb / us / 1e3);
        // M = 5 rows (an MTP-4 verify): the DPAS small-M kernel vs a double reference on the
        // bf16-rounded rows, and vs the old SIMT batched GEMV's time.
        {
            const int Mr = 5;
            std::vector<float> hxm(size_t(Mr) * K);
            for (auto& v : hxm) v = float(int(rng() % 2001) - 1000) / 1000.0f;
            std::vector<sycl_bf16> hxb(hxm.size());
            for (size_t i = 0; i < hxm.size(); ++i) hxb[i] = sycl_bf16(hxm[i]);
            float* xm = sycl::malloc_device<float>(hxm.size(), q);
            sycl_bf16* xb = sycl::malloc_device<sycl_bf16>(hxb.size(), q);
            float* ym = sycl::malloc_device<float>(size_t(Mr) * N, q);
            q.memcpy(xm, hxm.data(), hxm.size() * 4); q.memcpy(xb, hxb.data(), hxb.size() * 2).wait();
            if (!fp8_smallm_ok(ws[0], Mr, xb, ym)) { std::printf("  fp8_smallm not ok for this shape\n"); }
            else {
                launch_fp8_smallm(q, ws[0], xb, ym, Mr); q.wait();
                std::vector<float> hym(size_t(Mr) * N); q.memcpy(hym.data(), ym, hym.size() * 4).wait();
                double mx2 = 0, md2 = 0;
                for (int m = 0; m < Mr; ++m)
                    for (int n = 0; n < N; n += (N > 50000 ? 13 : 3)) {
                        double acc = 0; const uint8_t* row = hp.data() + size_t(n) * K;
                        for (int k = 0; k < K; ++k) acc += e4m3(row[k]) * double(float(hxb[size_t(m) * K + k]));
                        acc *= hs[n];
                        mx2 = std::max(mx2, std::fabs(acc)); md2 = std::max(md2, std::fabs(acc - double(hym[size_t(m) * N + n])));
                    }
                auto tm = [&](auto&& f) {
                    for (int i = 0; i < 2 * copies; ++i) f(ws[i % copies]);
                    q.wait();
                    const int it2 = std::max(10, 2 * copies);
                    auto t1 = std::chrono::steady_clock::now();
                    for (int i = 0; i < it2; ++i) f(ws[i % copies]);
                    q.wait();
                    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / it2;
                };
                const double us_new = tm([&](const QuantWeight& w) { launch_fp8_smallm(q, w, xb, ym, Mr); });
                const double us_old = tm([&](const QuantWeight& w) { launch_gemv_batch(q, w, xm, ym, Mr); });
                std::printf("         M=5 DPAS rel err %.2e  %8.1f us (%6.1f GB/s)   old batched GEMV %8.1f us (%6.1f GB/s)\n",
                            mx2 > 0 ? md2 / mx2 : md2, us_new, pb / us_new / 1e3, us_old, pb / us_old / 1e3);
            }
            sycl::free(xm, q); sycl::free(xb, q); sycl::free(ym, q);
        }
        for (void* m : mem) sycl::free(m, q); sycl::free(x, q); sycl::free(y, q);
    }
    return 0;
}
