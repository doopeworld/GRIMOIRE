// tier_probe.cpp -- measure the three links an offloaded MoE lives on, so the
// VRAM / RAM / SSD split is designed from numbers instead of guesses:
//   1. host -> device copies (pinned and pageable), one big copy and
//      expert-sized (2.76 MB = one Qwen3.8-Flash-Next NVFP4 expert) chunks
//   2. a kernel reading pinned host memory directly (zero copy over PCIe),
//      against the same kernel reading VRAM
//   3. CPU memory read bandwidth at 1 / 6 / 16 threads
//   4. SSD: random expert-sized reads from a file, O_DIRECT and buffered
// Buffers are filled with a non-constant pattern (constant fills lie on this
// card -- see the roofline notes).
// Build: icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -march=native \
//          -std=c++20 tools/tier_probe.cpp -o bin/tier_probe -lpthread
// Run:   tier_probe [file-for-ssd-test]
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
static void fill(void* p, size_t n) {
    uint64_t* w = static_cast<uint64_t*>(p);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < n / 8; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; w[i] = x; }
}

int main(int argc, char** argv) {
    const size_t EXP = 2764800;              // 3*2560*640*(1/2 + 1/16) bytes
    const size_t TOT = size_t(1) << 30;      // 1 GiB per transfer test
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    char* h = sycl::malloc_host<char>(TOT, q);
    char* d = sycl::malloc_device<char>(TOT, q);
    char* p = static_cast<char*>(std::aligned_alloc(4096, TOT));
    if (!h || !d || !p) { std::printf("alloc failed\n"); return 1; }
    fill(h, TOT); std::memcpy(p, h, TOT);
    q.memcpy(d, h, TOT).wait();

    // ---- 1. host -> device copies ------------------------------------------
    auto t0 = clk::now(); q.memcpy(d, h, TOT).wait(); auto t1 = clk::now();
    std::printf("H2D pinned   1 GiB one copy        %6.2f GB/s\n", TOT / secs(t0, t1) / 1e9);
    const size_t n = TOT / EXP;
    t0 = clk::now();
    for (size_t i = 0; i < n; ++i) q.memcpy(d + i * EXP, h + i * EXP, EXP);
    q.wait(); t1 = clk::now();
    std::printf("H2D pinned   %zu x 2.76 MB chunks  %6.2f GB/s\n", n, n * EXP / secs(t0, t1) / 1e9);
    {   // one expert, round trip latency including the wait
        double best = 1e9;
        for (int r = 0; r < 20; ++r) {
            auto a = clk::now(); q.memcpy(d, h + (r % 16) * EXP, EXP).wait(); auto b = clk::now();
            best = std::min(best, secs(a, b));
        }
        std::printf("H2D pinned   one expert, waited    %6.1f us\n", best * 1e6);
    }
    t0 = clk::now(); q.memcpy(d, p, TOT).wait(); t1 = clk::now();
    std::printf("H2D pageable 1 GiB one copy        %6.2f GB/s\n", TOT / secs(t0, t1) / 1e9);
    t0 = clk::now();
    for (size_t i = 0; i < n; ++i) q.memcpy(d + i * EXP, p + i * EXP, EXP);
    q.wait(); t1 = clk::now();
    std::printf("H2D pageable %zu x 2.76 MB chunks  %6.2f GB/s\n", n, n * EXP / secs(t0, t1) / 1e9);
    {   // two queues: do two copy streams in flight beat one?
        sycl::queue q2{q.get_context(), q.get_device(), sycl::property::queue::in_order{}};
        t0 = clk::now();
        for (size_t i = 0; i < n; ++i)
            (i & 1 ? q2 : q).memcpy(d + i * EXP, h + i * EXP, EXP);
        q.wait(); q2.wait(); t1 = clk::now();
        std::printf("H2D pinned   2 queues, chunks      %6.2f GB/s\n", n * EXP / secs(t0, t1) / 1e9);
    }

    // ---- 2. kernel reads: VRAM vs pinned host (zero copy) --------------------
    uint32_t* sink = sycl::malloc_device<uint32_t>(1, q);
    auto zc = [&](const char* src, const char* tag) {
        const size_t n16 = TOT / 16;
        const sycl::uint4* s = reinterpret_cast<const sycl::uint4*>(src);
        for (int WI : {1 << 14, 1 << 16, 1 << 18}) {
            double best = 1e9;
            for (int r = 0; r < 3; ++r) {
                auto a = clk::now();
                q.parallel_for(sycl::nd_range<1>(WI, 256), [=](sycl::nd_item<1> it) {
                    const size_t g = it.get_global_id(0);
                    uint32_t acc = 0;
                    for (size_t i = g; i < n16; i += size_t(WI)) {
                        const sycl::uint4 v = s[i];
                        acc ^= v.x() ^ v.y() ^ v.z() ^ v.w();
                    }
                    if (acc == 0x12345678u) sink[0] = acc;
                }).wait();
                best = std::min(best, secs(a, clk::now()));
            }
            std::printf("kernel read  %-18s WI=%-7d %7.2f GB/s\n", tag, WI, TOT / best / 1e9);
        }
    };
    zc(d, "VRAM");
    zc(h, "pinned host (PCIe)");

    // ---- 3. CPU memory read bandwidth ---------------------------------------
    {
        const size_t CB = size_t(4) << 30;
        uint64_t* c = static_cast<uint64_t*>(std::aligned_alloc(4096, CB));
        if (c) {
            std::vector<std::thread> th;
            const int NT0 = int(std::thread::hardware_concurrency());
            for (int T = 0; T < NT0; ++T)   // parallel first touch + fill
                th.emplace_back([=] { const size_t per = CB / 8 / NT0;
                    fill(c + per * T, per * 8); });
            for (auto& t : th) t.join();
            for (int NT : {1, 6, 8, 16}) {
                std::atomic<uint64_t> tot{0};
                double best = 1e9;
                for (int r = 0; r < 3; ++r) {
                    std::vector<std::thread> w;
                    auto a = clk::now();
                    for (int T = 0; T < NT; ++T)
                        w.emplace_back([&, T] {
                            const size_t per = CB / 8 / NT;
                            const uint64_t* s = c + per * T;
                            uint64_t x0 = 0, x1 = 0, x2 = 0, x3 = 0;
                            for (size_t i = 0; i + 4 <= per; i += 4) {
                                x0 ^= s[i]; x1 ^= s[i + 1]; x2 ^= s[i + 2]; x3 ^= s[i + 3];
                            }
                            tot += x0 ^ x1 ^ x2 ^ x3;
                        });
                    for (auto& t : w) t.join();
                    best = std::min(best, secs(a, clk::now()));
                }
                std::printf("CPU read     %2d threads            %6.2f GB/s   (%llx)\n", NT,
                            CB / best / 1e9, (unsigned long long)(tot.load() & 0xf));
            }
            std::free(c);
        }
    }

    // ---- 4. SSD: random expert-sized reads -----------------------------------
    if (argc > 1) {
        struct stat st{};
        if (stat(argv[1], &st) == 0 && st.st_size > off_t(8) << 30) {
            const size_t CH = (EXP + 4095) & ~size_t(4095);
            const size_t nread = 1000;            // ~2.8 GB per mode
            for (int direct : {1, 0}) {
                for (int NT : {1, 4, 8, 16}) {
                    int fd = open(argv[1], O_RDONLY | (direct ? O_DIRECT : 0));
                    if (fd < 0) { std::printf("open failed (direct=%d)\n", direct); break; }
                    std::atomic<size_t> next{0}, bytes{0};
                    std::vector<std::thread> w;
                    const uint64_t seed = 1234 + NT + 100 * direct;
                    auto a = clk::now();
                    for (int T = 0; T < NT; ++T)
                        w.emplace_back([&, T] {
                            char* buf = static_cast<char*>(std::aligned_alloc(4096, CH));
                            std::mt19937_64 rng(seed + T);
                            const off_t span = (st.st_size - off_t(CH)) / 4096;
                            while (next++ < nread) {
                                const off_t off = off_t(rng() % span) * 4096;
                                ssize_t r = pread(fd, buf, CH, off);
                                if (r > 0) bytes += size_t(r);
                            }
                            std::free(buf);
                        });
                    for (auto& t : w) t.join();
                    const double s = secs(a, clk::now());
                    close(fd);
                    std::printf("SSD %-8s  %2d threads 2.76MB  %6.2f GB/s  %6.2f ms/expert/thread\n",
                                direct ? "O_DIRECT" : "buffered", NT, bytes / s / 1e9,
                                s * 1e3 * NT / nread);
                }
            }
        } else std::printf("SSD test skipped: %s missing or < 8 GB\n", argv[1]);
    }
    sycl::free(h, q); sycl::free(d, q); sycl::free(sink, q); std::free(p);
    std::printf("TIER PROBE DONE\n");
    return 0;
}
