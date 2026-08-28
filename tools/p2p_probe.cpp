// Can the two B70s copy device-to-device, and how fast? Every TP=2 layer needs
// an all-reduce across the cards, so this number decides the design: real P2P,
// or host-staged round trips.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
#include <string>
using clk = std::chrono::high_resolution_clock;

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<sycl::device> gpus;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        const auto name = d.get_info<sycl::info::device::name>();
        if (name.find("B70") != std::string::npos) gpus.push_back(d);
    }
    std::printf("B70 devices found: %zu\n", gpus.size());
    if (gpus.size() < 2) { std::printf("need 2 B70s\n"); return 1; }

    sycl::queue q0{gpus[0]}, q1{gpus[1]};
    const size_t N = size_t(64) << 20;
    const size_t bytes = N * sizeof(float);
    float* a = sycl::malloc_device<float>(N, q0);
    float* b = sycl::malloc_device<float>(N, q1);
    q0.fill(a, 1.0f, N).wait();
    q1.fill(b, 2.0f, N).wait();

    auto timed_copy = [&](sycl::queue& dstq, float* dst, float* src, const char* tag) {
        try {
            dstq.memcpy(dst, src, bytes).wait();
            const int R = 10;
            const auto t0 = clk::now();
            for (int i = 0; i < R; ++i) dstq.memcpy(dst, src, bytes).wait();
            const double s = std::chrono::duration<double>(clk::now() - t0).count() / R;
            std::printf("  %-28s %7.1f GB/s  (%.3f ms / 256MB)\n",
                        tag, bytes / s / 1e9, s * 1e3);
        } catch (sycl::exception& e) {
            std::printf("  %-28s FAILED: %s\n", tag, e.what());
        }
    };

    std::printf("direct cross-device copy:\n");
    timed_copy(q1, b, a, "GPU0 -> GPU1 (P2P?)");
    timed_copy(q0, a, b, "GPU1 -> GPU0 (P2P?)");

    float* h = sycl::malloc_host<float>(N, q0);
    q0.memcpy(h, a, bytes).wait();
    const int R = 10;
    const auto t0 = clk::now();
    for (int i = 0; i < R; ++i) { q0.memcpy(h, a, bytes).wait(); q1.memcpy(b, h, bytes).wait(); }
    const double s = std::chrono::duration<double>(clk::now() - t0).count() / R;
    std::printf("host-staged GPU0->host->GPU1: %7.1f GB/s  (%.3f ms / 256MB)\n",
                bytes / s / 1e9, s * 1e3);

    sycl::free(a, q0); sycl::free(b, q1); sycl::free(h, q0);
    return 0;
}
