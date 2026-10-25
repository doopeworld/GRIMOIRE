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

// gemm_fast_probe -- GRIMOIRE's prefill GEMM (bin/libgrimoire_gemm.so) timed at the shapes
// tools/bench/onednn/mm_b70.batch gives oneDNN's benchdnn, so the two are compared on equal
// terms: device time per call, weights read cold (rotated over copies spanning >= 512 MiB),
// y = x W^T with x [M][K] bf16, W [N][K] (bf16, or FP8 E4M3 with a float scale per row).
//
// Timed three ways: the whole call's LAST event (launch_gemm_fast -- that event is the GEMM, so
// "whole" excludes the dequantize pass in front of it), and the two halves -- the dequantize
// pass (launch_gemm_fast_dequant) and the GEMM over the bf16 copy (launch_gemm_fast_predq).
// The real cost of one call is dequant + predq: at 512 rows the pass is most of it.
// With GRIMOIRE_ONEDNN=1 the call is one oneDNN matmul and dequant is an empty barrier.
//
// Every case checks 64 sampled outputs against a host dot product first, so a kernel that is
// fast because it is wrong cannot pass (the BesTLA lesson).  Internal time budget per case.
//
// Usage: gemm_fast_probe <bf16|fp8> M N K [M N K ...]
#include "kernels.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace b70;
using bf16h = sycl::ext::oneapi::bfloat16;

static float e4m3_to_f(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
    float v = e == 0 ? std::ldexp(float(m) / 8.0f, -6) : std::ldexp(1.0f + float(m) / 8.0f, e - 7);
    return s ? -v : v;
}

static double ev_ms(const sycl::event& e) {
    const auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return double(t1 - t0) * 1e-6;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 5 || (argc - 2) % 3 != 0) {
        std::fprintf(stderr, "usage: %s <bf16|fp8> M N K [M N K ...]\n", argv[0]);
        return 2;
    }
    const bool fp8 = std::string(argv[1]) == "fp8";
    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{},
                                      sycl::property::queue::enable_profiling{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(1234);

    for (int a = 2; a + 2 < argc; a += 3) {
        const int M = std::atoi(argv[a]), N = std::atoi(argv[a + 1]), K = std::atoi(argv[a + 2]);
        const size_t wbytes = size_t(N) * K * (fp8 ? 1 : 2);
        const int ncopy = int(std::max<size_t>(2, (size_t(512) << 20) / wbytes + 1));

        // Host weights (one pattern, copied ncopy times on the device), activations, scales.
        std::vector<uint8_t> hw(wbytes);
        std::vector<float> hs(N);
        std::vector<bf16h> hx(size_t(M) * K);
        std::uniform_int_distribution<int> ub(0, 255);
        std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
        if (fp8) {
            for (auto& b : hw) { uint8_t v = uint8_t(ub(rng)); if ((v & 0x7F) == 0x7F) v ^= 1; b = v; }
            for (auto& s : hs) s = 0.002f + 0.002f * std::fabs(uf(rng));
        } else {
            bf16h* p = reinterpret_cast<bf16h*>(hw.data());
            for (size_t i = 0; i < size_t(N) * K; ++i) p[i] = bf16h(uf(rng) * 0.05f);
        }
        for (auto& v : hx) v = bf16h(uf(rng));

        std::vector<uint8_t*> dw(ncopy);
        for (int c = 0; c < ncopy; ++c) {
            dw[c] = sycl::malloc_device<uint8_t>(wbytes, q);
            q.memcpy(dw[c], hw.data(), wbytes);
        }
        float* ds = sycl::malloc_device<float>(N, q);
        bf16h* dx = sycl::malloc_device<bf16h>(size_t(M) * K, q);
        float* dy = sycl::malloc_device<float>(size_t(M) * N, q);
        q.memcpy(ds, hs.data(), sizeof(float) * N);
        q.memcpy(dx, hx.data(), sizeof(bf16h) * hx.size()).wait();

        auto view = [&](int c) {
            QuantWeight w;
            w.fmt = fp8 ? Fmt::FP8_E4M3 : Fmt::BF16;
            w.N = N; w.K = K; w.payload = dw[c];
            w.scales = fp8 ? ds : nullptr;
            w.row_bytes = int64_t(K) * (fp8 ? 1 : 2);
            w.row_scales = fp8 ? 1 : 0;
            return w;
        };
        const QuantWeight w0 = view(0);
        const bool ok_fast = gemm_fast_supported(w0, M);
        const bool ok_predq = gemm_fast_predq_supported(w0, M);
        std::printf("case %s M=%d N=%d K=%d copies=%d fast=%d predq=%d\n", fp8 ? "fp8" : "bf16",
                    M, N, K, ncopy, int(ok_fast), int(ok_predq));
        if (!ok_fast) goto next;
        {
            // Correctness first: 64 sampled outputs of the whole call against a host dot product.
            launch_gemm_fast(q, w0, reinterpret_cast<const sycl_bf16*>(dx), dy, M).wait();
            std::vector<float> hy(size_t(M) * N);
            q.memcpy(hy.data(), dy, sizeof(float) * hy.size()).wait();
            double worst = 0.0;
            std::uniform_int_distribution<int> um(0, M - 1), un(0, N - 1);
            for (int t = 0; t < 64; ++t) {
                const int m = um(rng), n = un(rng);
                double ref = 0.0, mag = 0.0;
                for (int k = 0; k < K; ++k) {
                    const float wv = fp8 ? e4m3_to_f(hw[size_t(n) * K + k]) * hs[n]
                                         : float(reinterpret_cast<const bf16h*>(hw.data())[size_t(n) * K + k]);
                    // The kernel multiplies bf16 operands; round the dequantized weight like it.
                    const float wr = float(bf16h(wv));
                    ref += double(float(hx[size_t(m) * K + k])) * wr;
                    mag += std::fabs(double(float(hx[size_t(m) * K + k])) * wr);
                }
                worst = std::max(worst, std::fabs(double(hy[size_t(m) * N + n]) - ref) / (mag + 1e-30));
            }
            std::printf("  check: worst |err|/sum|x*w| over 64 samples = %.3e %s\n", worst,
                        worst < 1e-3 ? "PASS" : "FAIL");
            if (!(worst < 1e-3)) goto next;

            const double flop = 2.0 * M * N * K;
            auto timed = [&](const char* what, auto&& launch) {
                std::vector<double> ts;
                const auto t_start = std::chrono::steady_clock::now();
                launch(0).wait();                       // one untimed call per variant
                for (int r = 0; r < 200; ++r) {
                    sycl::event e = launch((r + 1) % ncopy);
                    e.wait();
                    ts.push_back(ev_ms(e));
                    const double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_start).count();
                    if (r >= 4 && el > 4.0) break;      // internal budget: ~4 s per variant
                }
                const double ms = median(ts);
                std::printf("  %-8s %9.3f ms  %7.1f TFLOP/s  (n=%zu)\n", what, ms,
                            flop / (ms * 1e-3) * 1e-12, ts.size());
                return ms;
            };
            timed("whole", [&](int c) {
                return launch_gemm_fast(q, view(c), reinterpret_cast<const sycl_bf16*>(dx), dy, M);
            });
            if (ok_predq) {
                const size_t ne = gemm_fast_predq_elems(w0);
                sycl_bf16* wdq = sycl::malloc_device<sycl_bf16>(ne, q);
                timed("dequant", [&](int c) { return launch_gemm_fast_dequant(q, view(c), wdq); });
                launch_gemm_fast_dequant(q, w0, wdq).wait();
                timed("predq", [&](int c) {
                    (void)c;
                    return launch_gemm_fast_predq(q, w0, wdq,
                                                  reinterpret_cast<const sycl_bf16*>(dx), dy, M);
                });
                sycl::free(wdq, q);
            }
        }
    next:
        for (auto* p : dw) sycl::free(p, q);
        sycl::free(ds, q); sycl::free(dx, q); sycl::free(dy, q);
    }
    return 0;
}
