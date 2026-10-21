// Standalone B70 diagnostic for the packed Q5_K decode lane.
// Build with icpx -fsycl (see docs/gguf/NATIVE_QK.md), run via b70run.sh.
// Mirrors gemv_native_qk(): one 16-lane sub-group per row, lane partials
// reduced with reduce_over_group; compares against decode_q5_k in double.
// Covers aligned and misaligned row bases / padded strides, K and N shapes of
// the Qwen3.8-27B and Ornith UD-Q4_K_XL / Q4_K_M files, and MB = 1 and 4.
#include "b70/native_qk_gemv.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace b70;

template <int MB>
static int run_case(sycl::queue& q, int N, int K, int pad, int base_off, int mc, uint32_t seed) {
    const int nb = K / 256, rb = nb * 176 + pad;
    const size_t bytes = size_t(N) * rb + 64;
    auto* store = sycl::malloc_shared<uint8_t>(bytes + 64, q);
    uint8_t* p = store + base_off;                       // base_off 0 = 64B-aligned USM base
    auto* x = sycl::malloc_shared<float>(size_t(MB) * K, q);
    auto* y = sycl::malloc_shared<float>(size_t(MB) * N, q);
    uint32_t s = seed;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return s >> 8; };
    for (size_t i = 0; i < bytes; ++i) store[i] = uint8_t(rnd());
    for (int n = 0; n < N; ++n) for (int b = 0; b < nb; ++b) {
        uint8_t* blk = p + size_t(n) * rb + size_t(b) * 176;
        const uint16_t d = uint16_t(0x1c00 + (rnd() & 0x3ff)), dm = uint16_t(0x1800 + (rnd() & 0x3ff)); // ~2^-8 .. 2^-6
        blk[0] = d & 255; blk[1] = d >> 8; blk[2] = dm & 255; blk[3] = dm >> 8;
    }
    for (size_t i = 0; i < size_t(MB) * K; ++i) x[i] = float(int(rnd() % 2001) - 1000) / 500.0f;
    const uint8_t* pc = p; const float* xc = x; const int rbc = rb;
    constexpr int WGS = 8;                                // 8 sub-groups per work-group, 1 row each
    const int groups = (N + WGS - 1) / WGS;
    q.parallel_for(sycl::nd_range<1>(size_t(groups) * WGS * 16, WGS * 16), [=](sycl::nd_item<1> it)
            [[sycl::reqd_sub_group_size(16)]] {
        const auto sg = it.get_sub_group();
        const int lane = int(sg.get_local_id()[0]);
        const int n = int(it.get_group(0)) * WGS + int(sg.get_group_id()[0]);
        const float* xp[MB];
        for (int m = 0; m < MB; ++m) xp[m] = xc + size_t(m) * K;
        float acc[MB] = {};
        if (n < N)
            for (int base = 0; base < K; base += 256)
                native_qk_block_lane<Fmt::Q5_K, MB>(pc + size_t(n) * rbc + size_t(base / 256) * 176, xp, base, lane, mc, acc);
        for (int m = 0; m < MB; ++m) {
            const float v = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
            if (lane == 0 && n < N && m < mc) y[size_t(m) * N + n] = v;
        }
    }).wait_and_throw();
    int bad = 0;
    double worst = 0;
    for (int n = 0; n < N; ++n) for (int m = 0; m < mc; ++m) {
        double ref = 0, mag = 0;
        for (int k = 0; k < K; ++k) {
            const double t = double(decode_q5_k(p + size_t(n) * rb, k)) * x[size_t(m) * K + k];
            ref += t; mag += std::fabs(t);
        }
        const double err = std::fabs(double(y[size_t(m) * N + n]) - ref);
        worst = std::max(worst, err / std::max(mag, 1e-9));
        if (!std::isfinite(y[size_t(m) * N + n]) || err > 2e-4 * std::max(mag, 1.0)) {
            if (bad < 4) std::printf("   BAD n=%d m=%d dev=%g ref=%g\n", n, m, y[size_t(m) * N + n], ref);
            ++bad;
        }
    }
    std::printf("N=%-6d K=%-6d pad=%-2d off=%-2d MB=%d mc=%d  bad=%d worst_rel_to_mag=%.2e\n", N, K, pad, base_off, MB, mc, bad, worst);
    sycl::free(y, q); sycl::free(x, q); sycl::free(store, q);
    return bad;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    int bad = 0;
    // (N, K) from Qwen3.8-27B (hidden 5120, ffn 17408, attn 6144/1024...) and Ornith/Qwen3-MoE (2048, 512).
    const int shapes[][2] = {{3, 256}, {17, 512}, {64, 2048}, {1024, 5120}, {5120, 5120}, {6144, 5120},
                             {5120, 17408}, {17408, 5120}, {512, 2048}, {2048, 512}, {1030, 5120}};
    for (auto& sh : shapes) for (int pad : {0, 16, 1}) for (int off : {0, 1, 4}) {
        if (pad == 1 && off == 4) continue;
        bad += run_case<1>(q, sh[0], sh[1], pad, off, 1, 7u + sh[0]);
    }
    for (int mc : {2, 4}) for (auto& sh : {std::pair<int,int>{64, 2048}, {1024, 5120}, {5120, 17408}})
        bad += run_case<4>(q, sh.first, sh.second, 0, 0, mc, 99u + mc);
    std::printf("Q5_K device GEMV errors=%d\n", bad);
    return bad ? 1 : 0;
}
