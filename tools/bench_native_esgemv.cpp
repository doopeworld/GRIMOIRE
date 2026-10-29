// bench_native_esgemv.cpp -- device check + bandwidth of the native GGUF ESIMD decode GEMVs
// (src/native_qk_esimd.hpp, V0) on random raw GGUF blocks at Qwen3.8-27B's decode shapes, and
// reference variants (V > 0) of the IQ4_XS / IQ4_NL inner loops, timed the same way.
// Correctness: sampled rows against a double-precision host dot built from decode_elem<F>
// (the element decoder that is bit-exact vs gguf_dequantize).
// Build (repo root, in grimoire-dev):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src \
//        tools/bench_native_esgemv.cpp -o bin/bench_native_esgemv
// Run only through the wrappers (tools/g0run.sh NAME /grimoire/bin/bench_native_esgemv), under
// the gpu0 flock.
#include <sycl/sycl.hpp>
#include "native_qk_esimd.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace b70;
namespace es = sycl::ext::intel::esimd;

// The IQ4 table as a register vector, for the iselect reference variants (V1).
SYCL_ESIMD_FUNCTION inline es::simd<float, 16> probe_iq4_table() {
    constexpr float v[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                             1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};
    es::simd<float, 16> t;
    #pragma unroll
    for (int i = 0; i < 16; ++i) t[i] = v[i];
    return t;
}

// acc (16 lanes) += decoded 256 values at p * x[xoff .. xoff+255].
// The engine (V0) decodes the IQ4 table with nq_iq4_value (a rounded polynomial).
// IQ4_XS: V1 = the table by iselect (2026-10-29 first version: 142 vs 425 GB/s on gate_up),
//         V3 = NO table at all (timing floor, wrong values).
// IQ4_NL: V1 = the table by iselect (146 vs 431 GB/s on down).
template <Fmt F, int V>
SYCL_ESIMD_FUNCTION inline void probe_acc(const uint8_t* p, es::simd<float, 512>& x, int xoff,
                                          es::simd<float, 16>& acc) {
    if constexpr (F == Fmt::IQ4_XS) {
        es::simd<uint8_t, 8> hdr = es::block_load<uint8_t, 8>(p);
        es::simd<uint8_t, 128> qs = es::block_load<uint8_t, 128>(p + 8);
        const float d = float(sycl::bit_cast<sycl::half>(
            uint16_t(uint32_t(hdr[0]) | (uint32_t(hdr[1]) << 8))));
        es::simd<uint32_t, 8> h = uint32_t(hdr[2]) | (uint32_t(hdr[3]) << 8);
        es::simd<uint32_t, 8> sh(0, 2);
        es::simd<uint32_t, 4> s4 = hdr.template select<4, 1>(4);
        es::simd<uint32_t, 8> lo;
        lo.template select<4, 2>(0) = s4 & 15;
        lo.template select<4, 2>(1) = s4 >> 4;
        es::simd<uint32_t, 8> hi = (h >> sh) & 3;
        es::simd<int32_t, 8> si = lo | (hi << 4);
        es::simd<float, 8> sc = (es::convert<float>(si) - 32.0f) * d;
        es::simd<float, 16> tab = probe_iq4_table();
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            es::simd<uint16_t, 16> b = qs.template select<16, 1>(16 * l);
            es::simd<uint16_t, 16> i0 = b & 15;
            es::simd<uint16_t, 16> i1 = b >> 4;
            es::simd<float, 16> v0, v1;
            if constexpr (V == 1) {
                v0 = tab.iselect(i0);
                v1 = tab.iselect(i1);
            } else {
                v0 = es::convert<float>(i0);
                v1 = es::convert<float>(i1);
            }
            const float s = sc[l];
            acc += (v0 * s) * x.template select<16, 1>(xoff + 32 * l);
            acc += (v1 * s) * x.template select<16, 1>(xoff + 32 * l + 16);
        }
    } else {
        static_assert(F == Fmt::IQ4_NL);
        constexpr int BS = NativeGGUFBlock<F>::bytes, NB = nq_bytes_256<F>();
        es::simd<uint8_t, NB> raw;
        raw.template select<NB - 16, 1>(0) = es::block_load<uint8_t, NB - 16>(p);
        raw.template select<16, 1>(NB - 16) = es::block_load<uint8_t, 16>(p + NB - 16);
        es::simd<float, 16> tab = probe_iq4_table();
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const float d = float(sycl::bit_cast<sycl::half>(
                uint16_t(uint32_t(raw[b * BS]) | (uint32_t(raw[b * BS + 1]) << 8))));
            es::simd<uint16_t, 16> qv = raw.template select<16, 1>(b * BS + 2);
            es::simd<uint16_t, 16> i0 = qv & 15;
            es::simd<uint16_t, 16> i1 = qv >> 4;
            es::simd<float, 16> v0 = tab.iselect(i0);
            es::simd<float, 16> v1 = tab.iselect(i1);
            es::simd<float, 16> t = v0 * x.template select<16, 1>(xoff + 32 * b);
            t += v1 * x.template select<16, 1>(xoff + 32 * b + 16);
            acc += t * d;
        }
    }
}

// V == 0: the engine's kernel.  V > 0: the same kernel shape (KP 512, two rows a work-group,
// up to 32 threads a row pair) around probe_acc<F, V>.
template <Fmt F, int V>
sycl::event probe_gemv(sycl::queue& q, const QuantWeight& w, const float* x, float* y) {
    if constexpr (V == 0) {
        return native_qk_esgemv<F>(q, w, x, y, {});
    } else {
        constexpr int KP = 512, R = 2, RP = 4, MAXT = 32;
        constexpr int BB = nq_bytes_256<F>();
        const int chunks = w.K / KP;
        const int per = (chunks + MAXT - 1) / MAXT;
        const int ks = (chunks + per - 1) / per;
        const int groups = (w.N + R - 1) / R;
        const uint8_t* pay = w.payload;
        const int64_t rb = w.row_bytes;
        const int N = w.N;
        return q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(size_t(groups) * ks, ks),
                [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                    es::slm_init<MAXT * RP * 4>();
                    const int t = int(it.get_local_id(0));
                    const int row0 = int(it.get_group(0)) * R;
                    es::simd<float, RP> part = 0.0f;
                    for (int c = 0; c < per; ++c) {
                        const int chunk = t * per + c;
                        if (chunk >= chunks) break;
                        const int kb = chunk * KP;
                        es::simd<float, KP> xv = es::block_load<float, KP>(x + kb);
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            if (row0 + r >= N) continue;
                            const uint8_t* p = pay + int64_t(row0 + r) * rb + int64_t(kb / 256) * BB;
                            es::simd<float, 16> acc = 0.0f;
                            probe_acc<F, V>(p, xv, 0, acc);
                            probe_acc<F, V>(p + BB, xv, 256, acc);
                            part[r] += es::reduce<float>(acc, std::plus<>());
                        }
                    }
                    es::slm_block_store<float, RP>(t * RP * 4, part);
                    es::barrier();
                    if (t != 0) return;
                    es::simd<float, RP> sum = 0.0f;
                    for (int j = 0; j < ks; ++j)
                        sum += es::slm_block_load<float, RP>(j * RP * 4);
                    #pragma unroll
                    for (int r = 0; r < R; ++r)
                        if (row0 + r < N) y[row0 + r] = sum[r];
                });
        });
    }
}

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

// Random blocks with every fp16 scale set to ~0.02 (random scale bytes would make inf / NaN).
template <Fmt F>
static void sane_scales(uint8_t* row, int K) {
    constexpr int BS = NativeGGUFBlock<F>::bytes, E = NativeGGUFBlock<F>::elems;
    for (int j = 0; j < K / E; ++j) {
        uint8_t* p = row + int64_t(j) * BS;
        if (F == Fmt::Q4_K || F == Fmt::Q5_K) { p[0] = 0x00; p[1] = 0x25; p[2] = 0x00; p[3] = 0x24; }
        else if (F == Fmt::Q6_K) { p[208] = 0x00; p[209] = 0x25; }
        else { p[0] = 0x00; p[1] = 0x25; }       // IQ4_XS, IQ4_NL, Q8_0: d first
    }
}

template <Fmt F, int V>
static int run(sycl::queue& q, const char* name, int N, int K, int copies) {
    constexpr int BB = nq_bytes_256<F>();
    const int64_t rb = int64_t(K / 256) * BB;
    std::mt19937 rng(1234 + N + K);
    std::vector<uint8_t> host(size_t(N) * rb);
    for (size_t i = 0; i + 4 <= host.size(); i += 4) { const uint32_t v = rng(); std::memcpy(&host[i], &v, 4); }
    for (int64_t r = 0; r < N; ++r) sane_scales<F>(host.data() + r * rb, K);
    std::vector<float> x(K);
    for (auto& v : x) v = (float(rng() % 2001) - 1000.0f) / 1000.0f;
    std::vector<uint8_t*> dw(copies);
    for (auto& d : dw) { d = sycl::malloc_device<uint8_t>(host.size(), q); q.memcpy(d, host.data(), host.size()); }
    float* dx = sycl::malloc_device<float>(K, q); q.memcpy(dx, x.data(), K * 4);
    float* dy = sycl::malloc_device<float>(N, q);
    q.wait();
    QuantWeight w; w.fmt = F; w.N = N; w.K = K; w.row_bytes = rb;
    w.payload = dw[0];
    probe_gemv<F, V>(q, w, dx, dy).wait();
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
    for (int i = 0; i < copies; ++i) { w.payload = dw[i]; probe_gemv<F, V>(q, w, dx, dy); }
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) { w.payload = dw[i % copies]; probe_gemv<F, V>(q, w, dx, dy); }
    q.wait();
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    const double gbs = double(host.size()) / (us * 1e3);
    const bool floor = F == Fmt::IQ4_XS && V == 3;
    std::printf("%-6s V%d %-8s N=%-6d K=%-6d %9.2f us %7.1f GB/s  maxrel %.2e bad %d%s\n",
                fname(F), V, name, N, K, us, gbs, maxrel, bad, floor ? "  (floor: no table)" : "");
    for (auto d : dw) sycl::free(d, q);
    sycl::free(dx, q); sycl::free(dy, q);
    return floor ? 0 : bad;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int bad = 0;
    // Qwen3.8-27B decode shapes: ffn gate_up 34816 x 5120, ffn down 5120 x 17408,
    // DeltaNet qkv 10240 x 5120, ssm_out 5120 x 6144.
    bad += run<Fmt::IQ4_XS, 0>(q, "gate_up", 34816, 5120, 3);
    bad += run<Fmt::IQ4_XS, 1>(q, "gate_up", 34816, 5120, 3);
    bad += run<Fmt::IQ4_XS, 3>(q, "gate_up", 34816, 5120, 3);
    bad += run<Fmt::IQ4_XS, 0>(q, "down", 5120, 17408, 3);
    bad += run<Fmt::IQ4_XS, 0>(q, "o", 5120, 6144, 4);
    bad += run<Fmt::IQ4_XS, 0>(q, "qkv", 10240, 5120, 4);
    bad += run<Fmt::IQ4_NL, 0>(q, "down", 5120, 17408, 3);
    bad += run<Fmt::IQ4_NL, 1>(q, "down", 5120, 17408, 3);
    bad += run<Fmt::Q8_0, 0>(q, "ssm_out", 5120, 6144, 4);
    bad += run<Fmt::Q8_0, 0>(q, "ab", 48, 5120, 16);
    bad += run<Fmt::Q4_K, 0>(q, "gate_up", 34816, 5120, 3);
    bad += run<Fmt::Q5_K, 0>(q, "gate_up", 34816, 5120, 3);
    bad += run<Fmt::Q4_K, 0>(q, "down", 5120, 17408, 3);
    bad += run<Fmt::Q5_K, 0>(q, "down", 5120, 17408, 3);
    bad += run<Fmt::Q6_K, 0>(q, "down", 5120, 17408, 3);
    bad += run<Fmt::Q5_K, 0>(q, "qkv", 10240, 5120, 4);
    bad += run<Fmt::Q4_K, 0>(q, "qkv", 10240, 5120, 4);
    bad += run<Fmt::Q6_K, 0>(q, "odd-N", 1025, 2048, 4);
    bad += run<Fmt::Q4_K, 0>(q, "K512", 2048, 512, 4);
    bad += run<Fmt::IQ4_XS, 0>(q, "odd-N", 1025, 5120, 4);
    bad += run<Fmt::Q8_0, 0>(q, "K512", 2048, 512, 4);
    std::printf(bad ? "FAIL\n" : "OK\n");
    return bad ? 1 : 0;
}
