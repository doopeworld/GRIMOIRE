// bench_native_smallm.cpp -- check + speed of the GGUF native small-M DPAS GEMM (src/native_smallm.hpp)
// at Qwen3.8-27B's verify shapes, M = 2..8 activation rows, against a double-precision host reference
// (decode_elem<F> times the bf16 activations), next to the M = 1 ESIMD GEMV (native_qk_esgemv).
// Build (repo root, in grimoire-dev):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src \
//        tools/bench_native_smallm.cpp -o bin/bench_native_smallm
// Run only through the wrappers (tools/g0run.sh NAME /grimoire/bin/bench_native_smallm), under the
// gpu0 flock.
#include <sycl/sycl.hpp>
#include "native_smallm.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace b70;

static const char* fname(Fmt f) {
    switch (f) {
    case Fmt::Q4_K: return "Q4_K";
    case Fmt::Q5_K: return "Q5_K";
    case Fmt::Q6_K: return "Q6_K";
    case Fmt::IQ4_XS: return "IQ4_XS";
    case Fmt::IQ4_NL: return "IQ4_NL";
    case Fmt::Q8_0: return "Q8_0";
    default: return "?";
    }
}
static uint16_t to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static float from_bf16(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
// M <= 8: one DPAS row block (the engine's gemv_decode.cpp), 9..16: two (libgrimoire_gemm.so).
static sycl::event launch(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X, float* Y, int M) {
    return M <= 8 ? native_smallm_launch<1>(q, w, X, Y, M, {}) : native_smallm_launch<2>(q, w, X, Y, M, {});
}

template <Fmt F>
static int run(sycl::queue& q, const char* name, int N, int K, int copies) {
    constexpr int BS = NativeGGUFBlock<F>::bytes, E = NativeGGUFBlock<F>::elems;
    const int64_t rb = bytes_per_row(F, K);
    std::mt19937 rng(911 + N + K);
    std::vector<uint8_t> host(size_t(N) * rb);
    for (size_t i = 0; i + 4 <= host.size(); i += 4) { const uint32_t v = rng(); std::memcpy(&host[i], &v, 4); }
    // fp16 scales 2^-7 .. 2^-3, either sign
    auto scale = [&](uint8_t* p) { p[0] = uint8_t(rng()); p[1] = uint8_t(0x20 + rng() % 0x10) | (rng() & 1 ? 0x80 : 0); };
    for (int64_t r = 0; r < N; ++r)
        for (int j = 0; j < K / E; ++j) {
            uint8_t* p = host.data() + r * rb + int64_t(j) * BS;
            if (F == Fmt::Q4_K || F == Fmt::Q5_K) { scale(p); scale(p + 2); }
            else if (F == Fmt::Q6_K) scale(p + 208);
            else scale(p);
        }
    constexpr int MM = 16;
    std::vector<uint16_t> xb(size_t(MM) * K);
    std::vector<float> xf(size_t(MM) * K);
    for (size_t i = 0; i < xb.size(); ++i) {
        xb[i] = to_bf16((float(rng() % 2001) - 1000.0f) / 1000.0f);
        xf[i] = from_bf16(xb[i]);
    }
    std::vector<uint8_t*> dw(copies);
    for (auto& d : dw) { d = sycl::malloc_device<uint8_t>(host.size(), q); q.memcpy(d, host.data(), host.size()); }
    sycl_bf16* dx = sycl::malloc_device<sycl_bf16>(size_t(MM) * K, q);
    float* dxf = sycl::malloc_device<float>(size_t(K), q);
    float* dy = sycl::malloc_device<float>(size_t(MM) * N, q);
    q.memcpy(dx, xb.data(), xb.size() * 2);
    q.memcpy(dxf, xf.data(), size_t(K) * 4);
    q.wait();
    QuantWeight w; w.fmt = F; w.N = N; w.K = K; w.row_bytes = rb; w.payload = dw[0];
    if (!native_smallm_layout_ok(w, 2)) { std::printf("%-6s %-8s layout refused\n", fname(F), name); return 1; }
    const int iters = std::max(copies * 4, 20);
    // M = 1 reference speed: the decode GEMV
    double us1 = 0;
    {
        for (int i = 0; i < copies; ++i) { w.payload = dw[i]; native_qk_esgemv<F>(q, w, dxf, dy, {}); }
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) { w.payload = dw[i % copies]; native_qk_esgemv<F>(q, w, dxf, dy, {}); }
        q.wait();
        us1 = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    }
    int bad = 0;
    for (int M : {2, 5, 8, 12, 16}) {
        w.payload = dw[0];
        launch(q, w, dx, dy, M).wait();
        std::vector<float> y(size_t(M) * N);
        q.memcpy(y.data(), dy, y.size() * 4).wait();
        double maxrel = 0;
        int nb = 0;
        for (int s = 0; s < 48; ++s) {
            const int n = s < 16 ? s : (s * 7919 + 13) % N;       // the first tile + spread rows
            const uint8_t* row = host.data() + int64_t(n) * rb;
            std::vector<double> wv(K);
            for (int k = 0; k < K; ++k) wv[k] = decode_elem<F>(row, k, 1.0f);
            for (int m = 0; m < M; ++m) {
                double ref = 0, mag = 0;
                for (int k = 0; k < K; ++k) { const double t = wv[k] * xf[size_t(m) * K + k]; ref += t; mag += std::fabs(t); }
                const double err = std::fabs(ref - y[size_t(m) * N + n]) / (mag + 1e-30);
                maxrel = std::max(maxrel, err);
                if (!(err < 1e-5) && nb++ < 3)
                    std::printf("   mismatch %s M=%d n=%d m=%d ref %.6g got %.6g\n", fname(F), M, n, m, ref, y[size_t(m) * N + n]);
            }
        }
        bad += nb;
        for (int i = 0; i < copies; ++i) { w.payload = dw[i]; launch(q, w, dx, dy, M); }
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) { w.payload = dw[i % copies]; launch(q, w, dx, dy, M); }
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::printf("%-6s %-8s N=%-6d K=%-6d M=%d %9.1f us %6.1f GB/s  (M=1 GEMV %7.1f us: x%.2f)  maxrel %.1e bad %d\n",
                    fname(F), name, N, K, M, us, double(host.size()) / (us * 1e3), us1, us / us1, maxrel, nb);
    }
    for (auto d : dw) sycl::free(d, q);
    sycl::free(dx, q); sycl::free(dxf, q); sycl::free(dy, q);
    return bad;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int bad = 0;
    bad += run<Fmt::Q4_K>(q, "gate_up", 34816, 5120, 3);
    bad += run<Fmt::Q5_K>(q, "gate_up", 34816, 5120, 3);
    bad += run<Fmt::IQ4_XS>(q, "gate_up", 34816, 5120, 3);
    bad += run<Fmt::Q5_K>(q, "down", 5120, 17408, 3);
    bad += run<Fmt::Q6_K>(q, "down", 5120, 17408, 3);
    bad += run<Fmt::IQ4_NL>(q, "down", 5120, 17408, 3);
    bad += run<Fmt::Q8_0>(q, "ssm_out", 5120, 6144, 4);
    bad += run<Fmt::Q4_K>(q, "qkv", 10240, 5120, 4);
    std::printf(bad ? "FAIL\n" : "OK\n");
    return bad ? 1 : 0;
}
