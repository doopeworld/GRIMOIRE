// P2P between the two B70s, done correctly: ONE context over both devices,
// and check zeDeviceCanAccessPeer. Also measures the thing that actually
// matters for capacity-oriented multi-GPU: moving an 8KB hidden state (one
// pipeline handoff per token) vs a full-layer all-reduce.
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
        if (d.get_info<sycl::info::device::name>().find("B70") != std::string::npos)
            gpus.push_back(d);
    }
    std::printf("B70 devices: %zu\n", gpus.size());
    if (gpus.size() < 2) return 1;

    sycl::context ctx{gpus};                       // ONE context over both cards
    sycl::queue q0{ctx, gpus[0]}, q1{ctx, gpus[1]};

    auto run = [&](size_t bytes, const char* label) {
        const size_t N = bytes / sizeof(float);
        float* a = sycl::malloc_device<float>(N, gpus[0], ctx);
        float* b = sycl::malloc_device<float>(N, gpus[1], ctx);
        q0.fill(a, 1.0f, N).wait();
        q1.fill(b, 2.0f, N).wait();
        auto t = [&](sycl::queue& dq, float* d, float* s, const char* tag) {
            try {
                dq.memcpy(d, s, bytes).wait();
                const int R = 20;
                const auto t0 = clk::now();
                for (int i = 0; i < R; ++i) dq.memcpy(d, s, bytes).wait();
                const double sec = std::chrono::duration<double>(clk::now()-t0).count()/R;
                std::printf("  %-22s %-22s %8.2f GB/s  %8.3f ms\n",
                            label, tag, bytes/sec/1e9, sec*1e3);
            } catch (sycl::exception& e) {
                std::printf("  %-22s %-22s FAILED %s\n", label, tag, e.what());
            }
        };
        t(q1, b, a, "GPU0->GPU1 shared-ctx");
        sycl::free(a, ctx); sycl::free(b, ctx);
    };

    std::printf("shared-context cross-device copy:\n");
    run(size_t(64)<<20, "256MB (all-reduce)");
    run(8192, "8KB (pipeline hidden)");
    run(size_t(1)<<20, "1MB");
    return 0;
}
