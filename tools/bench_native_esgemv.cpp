// bench_native_esgemv.cpp -- device check + bandwidth of the native Q4_K/Q6_K
// ESIMD decode GEMV (src/native_qk_esimd.hpp) on random raw GGUF blocks.
// Correctness: sampled rows against a double-precision host dot built from
// decode_elem<F> (the element decoder that is bit-exact vs gguf_dequantize).
// Build (repo root):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src \
//        tools/bench_native_esgemv.cpp -o bin/bench_native_esgemv
// Run only through the wrappers, under the gpu0 flock.
#include <sycl/sycl.hpp>
#include "native_qk_esimd.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace b70;

template <Fmt F>
static int run(sycl::queue& q, const char* name, int N, int K, int copies) {
    constexpr int BB = NativeGGUFBlock<F>::bytes;
    const int64_t rb = int64_t(K / 256) * BB;
    std::mt19937 rng(1234 + N);
    std::vector<uint8_t> host(size_t(N) * rb);
    for (auto& b : host) b = uint8_t(rng());
    for (int64_t r = 0; r < N; ++r)
        for (int j = 0; j < K / 256; ++j) {
            uint8_t* p = host.data() + r * rb + int64_t(j) * BB;
            if (F == Fmt::Q4_K) { p[0] = 0x00; p[1] = 0x25; p[2] = 0x00; p[3] = 0x24; }  // d~.02 dmin~.015
            else { p[208] = 0x00; p[209] = 0x25; }
        }
    std::vector<float> x(K);
    for (auto& v : x) v = (float(rng() % 2001) - 1000.0f) / 1000.0f;
    std::vector<uint8_t*> dw(copies);
    for (auto& d : dw) { d = sycl::malloc_device<uint8_t>(host.size(), q); q.memcpy(d, host.data(), host.size()); }
    float* dx = sycl::malloc_device<float>(K, q); q.memcpy(dx, x.data(), K * 4);
    float* dy = sycl::malloc_device<float>(N, q);
    q.wait();
    QuantWeight w; w.fmt = F; w.N = N; w.K = K; w.row_bytes = rb;
    // correctness
    w.payload = dw[0];
    native_qk_esgemv<F>(q, w, dx, dy, {}).wait();
    std::vector<float> y(N); q.memcpy(y.data(), dy, N * 4).wait();
    double maxrel = 0; int bad = 0;
    for (int s = 0; s < 64; ++s) {
        const int r = (s * 7919 + 13) % N;
        double ref = 0, mag = 0;
        for (int k = 0; k < K; ++k) {
            const double t = double(decode_elem<F>(host.data() + int64_t(r) * rb, k, 1.0f)) * x[k];
            ref += t; mag += std::fabs(t);
        }
        const double err = std::fabs(ref - y[r]) / (mag + 1e-30);
        maxrel = std::max(maxrel, err);
        if (!(err < 1e-5)) ++bad;
    }
    // timing: rotate over copies so the weights are cold
    const int iters = std::max(copies * 4, 20);
    for (int i = 0; i < copies; ++i) { w.payload = dw[i]; native_qk_esgemv<F>(q, w, dx, dy, {}); }
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) { w.payload = dw[i % copies]; native_qk_esgemv<F>(q, w, dx, dy, {}); }
    q.wait();
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    const double gbs = double(host.size()) / (us * 1e3);
    std::printf("%-6s %-10s N=%-7d K=%-5d  %9.2f us  %7.1f GB/s  maxrel %.2e bad %d\n",
                F == Fmt::Q4_K ? "Q4_K" : "Q6_K", name, N, K, us, gbs, maxrel, bad);
    for (auto d : dw) sycl::free(d, q);
    sycl::free(dx, q); sycl::free(dy, q);
    return bad;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int bad = 0;
    bad += run<Fmt::Q6_K>(q, "lm_head", 248320, 2048, 1);
    bad += run<Fmt::Q4_K>(q, "la_qkv", 8192, 2048, 8);
    bad += run<Fmt::Q6_K>(q, "la_qkv", 8192, 2048, 8);
    bad += run<Fmt::Q4_K>(q, "z", 4096, 2048, 16);
    bad += run<Fmt::Q4_K>(q, "out", 2048, 4096, 16);
    bad += run<Fmt::Q4_K>(q, "odd-N", 1025, 2048, 4);
    bad += run<Fmt::Q6_K>(q, "odd-N", 1025, 2048, 4);
    bad += run<Fmt::Q4_K>(q, "K512", 2048, 512, 4);
    bad += run<Fmt::Q6_K>(q, "K512", 2048, 512, 4);
    std::printf(bad ? "FAIL\n" : "OK\n");
    return bad ? 1 : 0;
}
