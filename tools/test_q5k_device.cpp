// Standalone B70 diagnostic: build with icpx -fsycl and run via b70run.sh.
#include "b70/native_qk_gemv.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>

int main() {
    using namespace b70;
    constexpr int K = 2048, N = 3, RB = K / 256 * 176 + 1;
    sycl::queue q{sycl::gpu_selector_v};
    auto* p = sycl::malloc_shared<uint8_t>(N * RB + 1, q);
    auto* x = sycl::malloc_shared<float>(K, q);
    auto* out = sycl::malloc_shared<float>(N * 16, q);
    for (int k = 0; k < K; ++k) x[k] = float((k * 37) % 113 - 56) / 71.0f;
    for (int n = 0; n < N; ++n) {
        auto* row = p + 1 + n * RB;
        for (int b = 0; b < K / 256; ++b) {
            auto* block = row + b * 176;
            for (int i = 0; i < 176; ++i) block[i] = uint8_t((i * 73 + b * 41 + n * 19) & 255);
            block[0] = 0x00; block[1] = 0x28; // d = 1/32
            block[2] = 0x00; block[3] = 0x24; // dmin = 1/16
        }
    }
    q.parallel_for(sycl::nd_range<1>(N * 16, 16), [=](sycl::nd_item<1> it)
            [[sycl::reqd_sub_group_size(16)]] {
        const int n = it.get_group(0), lane = it.get_local_id(0);
        const float* xp[1] = {x};
        float acc = 0;
        native_qk_row_lane<Fmt::Q5_K, 1>(p + 1 + n * RB, xp, K, lane, 1, &acc);
        out[n * 16 + lane] = acc;
    }).wait_and_throw();
    int errors = 0;
    for (int n = 0; n < N; ++n) for (int lane = 0; lane < 16; ++lane) {
        float ref = 0;
        if (lane < 8) for (int b = 0; b < K / 256; ++b)
            for (int i = 0; i < 32; ++i) {
                const int k = b * 256 + lane * 32 + i;
                ref = std::fma(decode_q5_k(p + 1 + n * RB, k), x[k], ref);
            }
        const float tol = 2e-3f * std::max(1.0f, std::fabs(ref));
        if (!std::isfinite(out[n * 16 + lane]) || std::fabs(out[n * 16 + lane] - ref) > tol) {
            std::printf("row=%d lane=%d device=%g ref=%g\n", n, lane, out[n * 16 + lane], ref);
            ++errors;
        }
    }
    std::printf("Q5_K device lane errors=%d\n", errors);
    sycl::free(out, q); sycl::free(x, q); sycl::free(p, q);
    return errors ? 1 : 0;
}
