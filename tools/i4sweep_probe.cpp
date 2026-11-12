// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// i4sweep_probe.cpp -- INT4 (GPTQ-style, group 128, u8 zeros) small-M GEMM through the ENGINE's launchers on
// Qwen3.8-27B / Ornith shapes (2026-11-05): launch_int4_smallm at M = 4 / 7 (gemv_decode.cpp, 128 registers) and
// launch_int4_smallm_wide at M = 14 / 16 (libgrimoire_gemm.so, 256 registers). The split plans are swept from outside
// with GRIMOIRE_I4_SMALLM_KS (<= 8 rows) and GRIMOIRE_I4_SMALLM16_KS (9-16 rows), one process per value, because
// MXFP4's sweep found 13-15% on its widest matrices. Check: the decode GEMV per row (fp32 X rounded to bf16).
// Weights rotate over copies so they stream from DRAM. Argument: index of one shape (default: all).
//
// Build (grimoire-dev, from the repo root):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno \
//     -fsycl-device-code-split=per_kernel -I include -I src tools/i4sweep_probe.cpp src/gemv_decode.cpp \
//     -Lbin -lgrimoire_gemm '-Wl,-rpath,$ORIGIN' -o bin/i4sweep_probe
#include <sycl/sycl.hpp>
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

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s  I4_SMALLM_KS=%s  I4_SMALLM16_KS=%s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                std::getenv("GRIMOIRE_I4_SMALLM_KS") ? std::getenv("GRIMOIRE_I4_SMALLM_KS") : "rule",
                std::getenv("GRIMOIRE_I4_SMALLM16_KS") ? std::getenv("GRIMOIRE_I4_SMALLM16_KS") : "rule");
    struct Shape { const char* nm; int N, K; };
    const std::vector<Shape> shapes = {
        {"q38 gate_up", 34816, 5120}, {"q38 down", 5120, 17408}, {"q38 dn_qkv", 10240, 5120},
        {"q38 z", 6144, 5120}, {"q38 out/o", 5120, 6144}, {"q38 attn_q", 12288, 5120},
        {"q38 kv", 1024, 5120}, {"q38 lm_head", 248320, 5120},
        // 2026-11-12: one TP2 rank's halves (Megatron: column halves of gate_up / qkv / z / q / kv,
        // K halves of down / out / o)
        {"tp gate_up", 17408, 5120}, {"tp down", 5120, 8704}, {"tp dn_qkv", 5120, 5120},
        {"tp z", 3072, 5120}, {"tp out/o", 5120, 3072}, {"tp attn_q", 6144, 5120}, {"tp kv", 512, 5120}};
    const int only = argc > 1 ? std::atoi(argv[1]) : -1;
    std::mt19937 rng(13);
    for (int si = 0; si < int(shapes.size()); ++si) {
        if (only >= 0 && only != si) continue;
        const Shape sh = shapes[si];
        const int N = sh.N, K = sh.K, GS = 128, rsc = K / GS;
        const size_t pb = size_t(N) * K / 2, sb = size_t(N) * rsc * 2, zb = size_t(N) * rsc;
        const size_t wb = pb + sb + zb;
        const int copies = std::max(2, std::min(8, int((1536ull << 20) / wb)));
        std::vector<uint8_t> hp(pb), hz(zb);
        std::vector<uint16_t> hs(size_t(N) * rsc);
        for (auto& v : hp) v = uint8_t(rng());
        for (auto& v : hz) v = uint8_t(rng() % 16);
        for (auto& v : hs) v = sycl::bit_cast<uint16_t>(sycl_bf16(0.001f * float(1 + rng() % 16)));
        std::vector<QuantWeight> ws(copies);
        std::vector<void*> allocs;
        for (int c = 0; c < copies; ++c) {
            uint8_t* p = sycl::malloc_device<uint8_t>(pb, q);
            uint16_t* s = sycl::malloc_device<uint16_t>(size_t(N) * rsc, q);
            uint8_t* z = sycl::malloc_device<uint8_t>(zb, q);
            q.memcpy(p, hp.data(), pb); q.memcpy(s, hs.data(), sb); q.memcpy(z, hz.data(), zb);
            allocs.push_back(p); allocs.push_back(s); allocs.push_back(z);
            QuantWeight& w = ws[c];
            w.fmt = Fmt::INT4; w.N = N; w.K = K; w.payload = p; w.scales = s; w.zeros = z;
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
        float* yr = sycl::malloc_device<float>(size_t(MR) * N, q);
        q.memcpy(x, hx.data(), hx.size() * 4); q.memcpy(xb, hxb.data(), hxb.size() * 2); q.wait();
        if (!int4_smallm_ok(ws[0], MR, xb, y)) { std::printf("%s: int4_smallm_ok refuses\n", sh.nm); continue; }
        std::printf("%-12s N=%-6d K=%-5d  %.1f MB/copy x%d\n", sh.nm, N, K, wb / 1e6, copies);
        auto timeit = [&](const char* nm, const std::function<void(const QuantWeight&)>& f) {
            for (int i = 0; i < 2 * copies; ++i) f(ws[i % copies]);
            q.wait();
            const int it = std::max(20, 4 * copies);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i) f(ws[i % copies]);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
            std::printf("  %-26s %8.1f us  %6.1f GB/s\n", nm, us, wb / us / 1e3);
        };
        auto check = [&](const char* nm, int M) {
            std::vector<float> a(size_t(M) * N), b(size_t(M) * N);
            q.memcpy(a.data(), y, a.size() * 4); q.memcpy(b.data(), yr, b.size() * 4); q.wait();
            double mx = 0, md = 0;
            for (size_t i = 0; i < a.size(); ++i) { mx = std::max(mx, double(std::fabs(b[i]))); md = std::max(md, double(std::fabs(a[i] - b[i]))); }
            std::printf("  check %-20s max rel %.2e\n", nm, mx > 0 ? md / mx : md);
        };
        for (int m = 0; m < MR; ++m) launch_gemv(q, ws[0], x + size_t(m) * K, yr + size_t(m) * N);
        q.wait();
        launch_int4_smallm(q, ws[0], xb, y, 8); check("engine M=8", 8);
        launch_int4_smallm(q, ws[0], xb, y, MR); check("engine M=16", MR);
        for (int M : {7, 16}) {
            char nm[64];
            std::snprintf(nm, sizeof nm, "engine M=%d", M);
            timeit(nm, [&](const QuantWeight& w) { launch_int4_smallm(q, w, xb, y, M); });
        }
        for (void* p : allocs) sycl::free(p, q);
        sycl::free(x, q); sycl::free(xb, q); sycl::free(y, q); sycl::free(yr, q);
    }
    return 0;
}
