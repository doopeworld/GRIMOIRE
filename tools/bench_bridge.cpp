// bench_bridge.cpp -- how wide is the road from VRAM to the EUs?
//
// No model, no layers, no framework. Stream a decode-sized arena out of VRAM
// and report achieved GB/s. The B70 is a 608 GB/s part; the bar for building
// on is 550 GB/s.
//
// TWO WAYS THIS BENCHMARK LIES IF YOU LET IT, both hit on 2026-09-05:
//
//  1. CONSTANT-FILLED BUFFERS. An arena of memset(1) is compressed by the
//     hardware, so wide reads never touch DRAM and report 1600+ GB/s on a
//     608 GB/s part. The arena is now filled with word[i] = i * PHI, which
//     does not compress.
//  2. DEAD-CODE ELIMINATION. Guarding the store behind a condition the
//     compiler can fold lets it delete the load loop entirely. Every pattern
//     now stores unconditionally and the host verifies an exact checksum.
//     A pattern that fails verification prints FAIL and its timing is void.
//
// Everything is read at uint32 granularity so the only variable is WIDTH and
// how many loads are in flight -- which is the thing being measured.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace sycl;
constexpr int SG = 16;
constexpr uint32_t PHI = 2654435761u;         // Knuth; makes lines incompressible

static double gbps(size_t bytes, double s) { return double(bytes) / s / 1e9; }

// sum of PHI*i for i in [a, a+n), in wrapping uint32 -- closed form, no loop
static uint32_t expect_run(uint64_t a, uint64_t n) {
    uint32_t sa = uint32_t(a), sn = uint32_t(n);
    return PHI * (sn * sa + uint32_t(uint64_t(sn) * uint64_t(sn - 1) / 2));
}

template <typename F>
static double timeit(queue& q, int iters, F&& f) {
    f(); q.wait();
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f(); q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

int main(int argc, char** argv) {
    const size_t GB    = size_t(argc > 1 ? atoi(argv[1]) : 8);
    const int    iters = argc > 2 ? atoi(argv[2]) : 3;

    queue q{gpu_selector_v, property::queue::in_order()};
    std::printf("device : %s\n", q.get_device().get_info<info::device::name>().c_str());
    std::printf("arena  : %zu GB, best of %d, incompressible fill\n\n", GB, iters);

    const size_t bytes = GB * (1ull << 30);
    const size_t words = bytes / 4;
    uint32_t* arena = malloc_device<uint32_t>(words, q);
    if (!arena) { std::printf("alloc failed\n"); return 1; }
    q.parallel_for(range<1>(words), [=](id<1> i) {
        arena[i] = PHI * uint32_t(i);
    }).wait();

    const size_t threads = 1ull << 20;
    const size_t groups  = threads / SG;
    uint32_t* out = malloc_device<uint32_t>(threads, q);
    std::vector<uint32_t> host(threads);

    auto verify = [&](const char* name, double secs,
                      const std::vector<uint32_t>& want) {
        q.memcpy(host.data(), out, threads * sizeof(uint32_t)).wait();
        size_t bad = 0;
        for (size_t i = 0; i < threads; ++i) if (host[i] != want[i]) ++bad;
        std::printf("%-9s %7.1f GB/s   %s\n", name, gbps(bytes, secs),
                    bad == 0 ? "verified" : "FAIL -- checksum, timing void");
    };

    // ---- narrow: 1 word (4 B) per lane per step, contiguous per thread ----
    {
        const size_t n = words / threads;
        std::vector<uint32_t> want(threads);
        for (size_t t = 0; t < threads; ++t) want[t] = expect_run(t * n, n);
        double s = timeit(q, iters, [&] {
            q.parallel_for(nd_range<1>(threads, 256), [=](nd_item<1> it) {
                const size_t g = it.get_global_id(0);
                const uint32_t* p = arena + g * n;
                uint32_t a = 0;
                for (size_t i = 0; i < n; ++i) a += p[i];
                out[g] = a;
            });
        });
        verify("w4", s, want);
    }

    // ---- wide: 4 words (16 B) per lane per step --------------------------
    {
        using u32x4 = vec<uint32_t, 4>;
        const size_t nc = words / 4 / threads;          // chunks per thread
        std::vector<uint32_t> want(threads);
        for (size_t t = 0; t < threads; ++t) want[t] = expect_run(t * nc * 4, nc * 4);
        double s = timeit(q, iters, [&] {
            q.parallel_for(nd_range<1>(threads, 256), [=](nd_item<1> it) {
                const size_t g = it.get_global_id(0);
                const u32x4* p = reinterpret_cast<const u32x4*>(arena) + g * nc;
                uint32_t a = 0;
                for (size_t i = 0; i < nc; ++i) {
                    u32x4 v = p[i]; a += v[0] + v[1] + v[2] + v[3];
                }
                out[g] = a;
            });
        });
        verify("w16", s, want);
    }

    // ---- wide + sub-group contiguous + 4 streams in flight ---------------
    // Lanes of a sub-group read ADJACENT 16 B chunks so each step retires one
    // 256 B run, and four independent streams keep the load pipe busy.
    {
        using u32x4 = vec<uint32_t, 4>;
        const size_t chunks = words / 4;
        const size_t per_g  = chunks / groups;          // chunks per sub-group
        const size_t step   = SG * 4;
        const size_t iters_ = per_g / step;
        std::vector<uint32_t> want(threads);
        for (size_t t = 0; t < threads; ++t) {
            const size_t g = t / SG, lane = t % SG;
            uint32_t acc = 0;
            for (size_t i = 0; i < iters_; ++i)
                for (int k = 0; k < 4; ++k) {
                    const uint64_t c = (g * per_g) + i * step + k * SG + lane;
                    acc += expect_run(c * 4, 4);
                }
            want[t] = acc;
        }
        double s = timeit(q, iters, [&] {
            q.parallel_for(nd_range<1>(threads, 256), [=](nd_item<1> it)
                [[sycl::reqd_sub_group_size(SG)]] {
                const size_t gid = it.get_global_id(0);
                const size_t g = gid / SG; const int lane = int(gid % SG);
                const u32x4* base = reinterpret_cast<const u32x4*>(arena) + g * per_g;
                uint32_t a0=0,a1=0,a2=0,a3=0;
                for (size_t i = 0; i + step <= per_g; i += step) {
                    u32x4 v0 = base[i + lane];
                    u32x4 v1 = base[i + SG + lane];
                    u32x4 v2 = base[i + 2*SG + lane];
                    u32x4 v3 = base[i + 3*SG + lane];
                    a0 += v0[0]+v0[1]+v0[2]+v0[3];
                    a1 += v1[0]+v1[1]+v1[2]+v1[3];
                    a2 += v2[0]+v2[1]+v2[2]+v2[3];
                    a3 += v3[0]+v3[1]+v3[2]+v3[3];
                }
                out[gid] = a0+a1+a2+a3;
            });
        });
        verify("w16x4", s, want);
    }

    // ---- how many roads? sweep independent streams per lane -------------
    // Little's Law: bandwidth = bytes-in-flight / latency. Width is one way to
    // raise bytes-in-flight; INDEPENDENT STREAMS are another, and they also
    // overlap ALU with memory -- which is the only lever that helps a dequant
    // -heavy kernel like the MXFP4 GEMV, where the load cannot get wider.
    {
        using u32x4 = vec<uint32_t, 4>;
        const size_t chunks = words / 4;
        std::printf("\nstreams per lane (16 B each, sub-group contiguous):\n");
        auto run_streams = [&](int S) {
            const size_t per_g = chunks / groups;
            const size_t step  = size_t(SG) * S;
            if (per_g < step) return;
            const size_t nit = per_g / step;
            std::vector<uint32_t> want(threads);
            for (size_t t = 0; t < threads; ++t) {
                const size_t g = t / SG, lane = t % SG;
                uint32_t acc = 0;
                for (size_t i = 0; i < nit; ++i)
                    for (int k = 0; k < S; ++k) {
                        const uint64_t c = (g * per_g) + i * step + size_t(k) * SG + lane;
                        acc += expect_run(c * 4, 4);
                    }
                want[t] = acc;
            }
            double secs = timeit(q, iters, [&] {
                q.submit([&](handler& h) {
                    h.parallel_for(nd_range<1>(threads, 256), [=](nd_item<1> it)
                        [[sycl::reqd_sub_group_size(SG)]] {
                        const size_t gid = it.get_global_id(0);
                        const size_t g = gid / SG; const int lane = int(gid % SG);
                        const u32x4* base =
                            reinterpret_cast<const u32x4*>(arena) + g * per_g;
                        uint32_t a[16];
                        for (int k = 0; k < 16; ++k) a[k] = 0;
                        for (size_t i = 0; i + step <= per_g; i += step)
                            for (int k = 0; k < S; ++k) {
                                u32x4 v = base[i + size_t(k) * SG + lane];
                                a[k] += v[0] + v[1] + v[2] + v[3];
                            }
                        uint32_t acc = 0;
                        for (int k = 0; k < S; ++k) acc += a[k];
                        out[gid] = acc;
                    });
                });
            });
            q.memcpy(host.data(), out, threads * sizeof(uint32_t)).wait();
            size_t bad = 0;
            for (size_t t = 0; t < threads; ++t) if (host[t] != want[t]) ++bad;
            std::printf("  x%-3d    %7.1f GB/s   %s\n", S, gbps(bytes, secs),
                        bad == 0 ? "verified" : "FAIL");
        };
        for (int S : {1, 2, 4, 8, 16}) run_streams(S);
    }

    std::printf("\nB70 spec 608 GB/s. Bar for building on: 550 GB/s.\n");
    free(arena, q); free(out, q);
    return 0;
}
