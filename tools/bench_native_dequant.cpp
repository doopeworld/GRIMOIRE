// bench_native_dequant.cpp -- bit-exactness and speed of the streaming GGUF -> bf16 VNNI dequant
// (src/native_dequant.hpp) for every native format at Qwen3.8-27B's prompt shapes, plain and with
// the SwiGLU interleave.  Reference: decode_elem<F> (bit-exact vs gguf_dequantize) rounded to bf16
// on the host -- what the tiled dequant writes.  Random blocks, random fp16 scales (subnormals too).
// Build (repo root, in grimoire-dev):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src \
//        tools/bench_native_dequant.cpp -o bin/bench_native_dequant
// Run only through the wrappers (tools/g0run.sh NAME /grimoire/bin/bench_native_dequant), under
// the gpu0 flock.
#include <sycl/sycl.hpp>
#include "native_dequant.hpp"
#include <chrono>
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

static uint32_t host_bf16(float f) { return nd_bf16_bits(f); }   // the host branch: RNE on bits

template <Fmt F>
static int run(sycl::queue& q, const char* name, int N, int K, bool swiglu) {
    constexpr int BS = NativeGGUFBlock<F>::bytes, E = NativeGGUFBlock<F>::elems;
    const int64_t rb = bytes_per_row(F, K);
    std::mt19937 rng(77 + N + K);
    std::vector<uint8_t> host(size_t(N) * rb);
    for (size_t i = 0; i + 4 <= host.size(); i += 4) { const uint32_t v = rng(); std::memcpy(&host[i], &v, 4); }
    // finite fp16 scales: exponent 0..30 (0 = subnormal), either sign
    auto scale = [&](uint8_t* p) { p[0] = uint8_t(rng()); p[1] = uint8_t((rng() % 0x78) | (rng() & 1 ? 0x80 : 0)); };
    for (int64_t r = 0; r < N; ++r)
        for (int j = 0; j < K / E; ++j) {
            uint8_t* p = host.data() + r * rb + int64_t(j) * BS;
            if (F == Fmt::Q4_K || F == Fmt::Q5_K) { scale(p); scale(p + 2); }
            else if (F == Fmt::Q6_K) scale(p + 208);
            else scale(p);
        }
    uint8_t* dp = sycl::malloc_device<uint8_t>(host.size(), q);
    sycl_bf16* dst = sycl::malloc_device<sycl_bf16>(size_t(N) * K, q);
    q.memcpy(dp, host.data(), host.size()).wait();
    QuantWeight w; w.fmt = F; w.N = N; w.K = K; w.row_bytes = rb; w.payload = dp;
    if (!native_stream_ok(w)) { std::printf("%-6s %-8s layout refused\n", fname(F), name); return 1; }
    const int ilv = swiglu ? N / 2 : 0;
    dequant_native_stream(q, w, dst, {}, ilv).wait();
    std::vector<uint16_t> out(size_t(N) * K);
    q.memcpy(out.data(), dst, out.size() * 2).wait();
    // every k of a third of the rows (all of the first and last 256)
    int64_t checked = 0, bad = 0;
    for (int n = 0; n < N; ++n) {
        if (!(n < 256 || n >= N - 256 || n % 3 == 0)) continue;
        const int col = ilv == 0 ? n : n < ilv ? (n / 32) * 64 + n % 32
                                               : ((n - ilv) / 32) * 64 + 32 + (n - ilv) % 32;
        const uint8_t* row = host.data() + int64_t(n) * rb;
        for (int k = 0; k < K; ++k) {
            const uint32_t ref = host_bf16(decode_elem<F>(row, k, 1.0f));
            const uint32_t got = out[(size_t(k / 2) * N + col) * 2 + (k & 1)];
            ++checked;
            if (ref != got && bad++ < 4)
                std::printf("   mismatch n=%d k=%d ref %04x got %04x\n", n, k, ref, got);
        }
    }
    const int iters = 10;
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) dequant_native_stream(q, w, dst, {}, ilv);
    q.wait();
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    const double gbs = double(host.size() + size_t(N) * K * 2) / (us * 1e3);
    std::printf("%-6s %-8s N=%-6d K=%-6d %s %9.1f us %7.1f GB/s (read+write)  checked %lld bad %lld\n",
                fname(F), name, N, K, swiglu ? "swiglu" : "plain ", us, gbs,
                (long long)checked, (long long)bad);
    sycl::free(dp, q); sycl::free(dst, q);
    return bad ? 1 : 0;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int bad = 0;
    bad += run<Fmt::Q4_K>(q, "gate_up", 34816, 5120, true);
    bad += run<Fmt::Q5_K>(q, "gate_up", 34816, 5120, true);
    bad += run<Fmt::IQ4_XS>(q, "gate_up", 34816, 5120, true);
    bad += run<Fmt::Q4_K>(q, "down", 5120, 17408, false);
    bad += run<Fmt::Q5_K>(q, "down", 5120, 17408, false);
    bad += run<Fmt::Q6_K>(q, "down", 5120, 17408, false);
    bad += run<Fmt::IQ4_XS>(q, "down", 5120, 17408, false);
    bad += run<Fmt::IQ4_NL>(q, "down", 5120, 17408, false);
    bad += run<Fmt::Q8_0>(q, "ssm_out", 5120, 6144, false);
    bad += run<Fmt::Q8_0>(q, "attn_v", 1024, 5120, false);
    bad += run<Fmt::Q6_K>(q, "attn_k", 1024, 5120, false);
    bad += run<Fmt::Q5_K>(q, "qkv", 10240, 5120, false);
    std::printf(bad ? "FAIL\n" : "OK\n");
    return bad ? 1 : 0;
}
