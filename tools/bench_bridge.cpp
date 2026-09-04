// bench_bridge.cpp -- how wide is the road from VRAM to the EUs?
//
// No model, no layers, no attention, no framework. Allocate a decode-sized
// weight arena in VRAM and stream it, reporting achieved GB/s. This is the
// number the whole engine is judged against: the B70 is a 608 GB/s part and
// anything below 550 GB/s means the bridge is too narrow to bother building on.
//
// Four access patterns, same bytes, same arena:
//   scalar1   1 byte  per lane per step  -- what launch_flash_decode does today
//   vec4      4 bytes per lane per step
//   vec16    16 bytes per lane per step  -- 256 B per sub-group, one cache line
//   vec16x4  16 bytes x 4 independent streams -- enough in flight to hide latency
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

using namespace sycl;

constexpr int SG = 16;

static double gbps(size_t bytes, double secs) {
    return double(bytes) / secs / 1e9;
}

template <typename F>
static double timeit(queue& q, int iters, F&& launch) {
    launch(); q.wait();                      // warm
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        launch(); q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;                              // best-of, not mean: we want the
}                                             // ceiling, not the noise floor

int main(int argc, char** argv) {
    const size_t GB = size_t(argc > 1 ? atoi(argv[1]) : 8);
    const int iters = argc > 2 ? atoi(argv[2]) : 5;

    queue q{gpu_selector_v, property::queue::in_order()};
    std::printf("device : %s\n",
        q.get_device().get_info<info::device::name>().c_str());
    std::printf("arena  : %zu GB, best of %d\n\n", GB, iters);

    const size_t bytes = GB * (1ull << 30);
    uint8_t* arena = malloc_device<uint8_t>(bytes, q);
    if (!arena) { std::printf("alloc failed\n"); return 1; }
    q.memset(arena, 1, bytes).wait();

    float* sink = malloc_device<float>(1024, q);
    q.memset(sink, 0, 1024 * sizeof(float)).wait();

    // Enough work-items that each does a modest number of steps.
    const size_t threads = 1ull << 20;        // 1M work-items

    // ---- scalar: 1 byte per lane per step ------------------------------
    {
        const size_t per = bytes / threads;
        double s = timeit(q, iters, [&] {
            q.parallel_for(nd_range<1>(threads, 256), [=](nd_item<1> it) {
                const size_t gid = it.get_global_id(0);
                const uint8_t* p = arena + gid * per;
                uint32_t acc = 0;
                for (size_t i = 0; i < per; ++i) acc += p[i];
                if (acc == 0xFFFFFFFFu) sink[gid & 1023] = float(acc);
            });
        });
        std::printf("scalar1   %7.1f GB/s\n", gbps(bytes, s));
    }

    // ---- vec4 ----------------------------------------------------------
    {
        const size_t n4 = bytes / 4 / threads;
        double s = timeit(q, iters, [&] {
            q.parallel_for(nd_range<1>(threads, 256), [=](nd_item<1> it) {
                const size_t gid = it.get_global_id(0);
                const uint32_t* p = reinterpret_cast<const uint32_t*>(arena) + gid * n4;
                uint32_t acc = 0;
                for (size_t i = 0; i < n4; ++i) acc += p[i];
                if (acc == 0xFFFFFFFFu) sink[gid & 1023] = float(acc);
            });
        });
        std::printf("vec4      %7.1f GB/s\n", gbps(bytes, s));
    }

    // ---- vec16: 16 B per lane, 256 B per sub-group ---------------------
    {
        using u32x4 = vec<uint32_t, 4>;
        const size_t n16 = bytes / 16 / threads;
        double s = timeit(q, iters, [&] {
            q.parallel_for(nd_range<1>(threads, 256), [=](nd_item<1> it) {
                const size_t gid = it.get_global_id(0);
                const u32x4* p = reinterpret_cast<const u32x4*>(arena) + gid * n16;
                uint32_t acc = 0;
                for (size_t i = 0; i < n16; ++i) {
                    u32x4 v = p[i];
                    acc += v[0] + v[1] + v[2] + v[3];
                }
                if (acc == 0xFFFFFFFFu) sink[gid & 1023] = float(acc);
            });
        });
        std::printf("vec16     %7.1f GB/s\n", gbps(bytes, s));
    }

    // ---- vec16 x4 independent streams, sub-group-contiguous ------------
    // Lanes of a sub-group read ADJACENT 16 B chunks so each step retires one
    // 256 B run, and four streams keep enough loads in flight to hide latency.
    {
        using u32x4 = vec<uint32_t, 4>;
        const size_t total16 = bytes / 16;
        const size_t groups = threads / SG;
        const size_t per_group = total16 / groups;
        double s = timeit(q, iters, [&] {
            q.parallel_for(nd_range<1>(threads, 256), [=](nd_item<1> it)
                [[sycl::reqd_sub_group_size(SG)]] {
                const size_t g = it.get_global_id(0) / SG;
                const int lane = int(it.get_global_id(0) % SG);
                const u32x4* base = reinterpret_cast<const u32x4*>(arena) + g * per_group;
                uint32_t a0=0,a1=0,a2=0,a3=0;
                const size_t step = SG * 4;
                for (size_t i = 0; i + step <= per_group; i += step) {
                    u32x4 v0 = base[i + lane];
                    u32x4 v1 = base[i + SG + lane];
                    u32x4 v2 = base[i + 2*SG + lane];
                    u32x4 v3 = base[i + 3*SG + lane];
                    a0 += v0[0]+v0[1]+v0[2]+v0[3];
                    a1 += v1[0]+v1[1]+v1[2]+v1[3];
                    a2 += v2[0]+v2[1]+v2[2]+v2[3];
                    a3 += v3[0]+v3[1]+v3[2]+v3[3];
                }
                const uint32_t acc = a0+a1+a2+a3;
                if (acc == 0xFFFFFFFFu) sink[g & 1023] = float(acc);
            });
        });
        std::printf("vec16x4   %7.1f GB/s   <-- the bridge\n", gbps(bytes, s));
    }

    std::printf("\nB70 spec 608 GB/s. Bar for building on: 550 GB/s.\n");
    free(arena, q); free(sink, q);
    return 0;
}
