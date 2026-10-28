// tp_devar_probe.cpp -- latency of a device-side all-reduce for Megatron TP decode, on ONE card.
//
// Today's tp_shm_allreduce blocks the host on every call: q.memcpy(D2H).wait(), CPU exchange,
// q.memcpy(H2D).  The GPU drains and idles 2x per layer (128 times per Qwen3.8 token).
// Candidate: the GPU writes its partial + a sequence flag into its OWN pinned host buffer; a
// CPU proxy thread (standing in for the exchange with the peer rank through CPU-only shared
// memory) sums in rank order and writes the result + flag into a second pinned host buffer of
// the same context; the GPU polls that flag with a bounded spin and copies the sum back.  The
// host never waits, so a whole token's kernels can be queued at once.  No buffer is ever mapped
// by two GPUs (Ian's 10-07 rule).
//
//   tp_devar_probe [n_floats=5120] [iters=2000]
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }
static float peer_val(size_t i, uint64_t s) { return float((i % 97) * 1e-3 + double(s % 13) * 0.5); }

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? size_t(std::atoll(argv[1])) : 5120;
    const uint64_t iters = argc > 2 ? uint64_t(std::atoll(argv[2])) : 2000;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s  n=%zu floats (%.1f KB)  iters=%llu\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), n, n * 4 / 1024.0,
                (unsigned long long)iters);

    float* dev = sycl::malloc_device<float>(n, q);
    float* hout = sycl::malloc_host<float>(n, q);          // GPU -> proxy
    float* hin = sycl::malloc_host<float>(n, q);           // proxy -> GPU
    uint64_t* flags = sycl::malloc_host<uint64_t>(32, q);  // [0] out flag, [8] in flag, [16] error
    std::vector<float> init(n), cpu(n);
    for (size_t i = 0; i < n; ++i) init[i] = float(i % 31) * 0.25f;
    auto reset = [&] {
        q.memcpy(dev, init.data(), n * 4).wait();
        for (int i = 0; i < 32; ++i) flags[i] = 0;
    };
    auto expect = [&](uint64_t upto) {  // dev after `upto` all-reduces, same float order as the proxy
        for (size_t i = 0; i < n; ++i) { float v = init[i]; for (uint64_t s = 1; s <= upto; ++s) v = v + peer_val(i, s); cpu[i] = v; }
    };
    auto check = [&](const char* tag, uint64_t upto) {
        std::vector<float> got(n);
        q.memcpy(got.data(), dev, n * 4).wait();
        expect(upto);
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) if (got[i] != cpu[i]) ++bad;
        std::printf("  %-34s %s (%zu/%zu wrong)\n", tag, bad ? "WRONG" : "bit-exact", bad, n);
        return bad == 0;
    };

    // ---- A: today's path (blocking D2H, CPU add, async H2D) ------------------------------------
    {
        reset();
        std::vector<float> res(n);
        float* stage = hout;
        auto run = [&](uint64_t cnt) {
            for (uint64_t s = 1; s <= cnt; ++s) {
                q.memcpy(stage, dev, n * 4).wait();
                for (size_t i = 0; i < n; ++i) hin[i] = stage[i] + peer_val(i, s);
                q.memcpy(dev, hin, n * 4);
            }
            q.wait();
        };
        run(50); reset();
        auto t0 = clk::now(); run(iters); double t = ms_since(t0);
        std::printf("A blocking host round trip:      %8.2f us per all-reduce\n", t * 1000 / double(iters));
        check("A result", iters);
    }

    // ---- B: device-side, GPU polls its own pinned host buffer ------------------------------------
    for (int wg : {256, 1024}) {
        reset();
        std::atomic<bool> stop{false};
        auto proxy = [&](uint64_t cnt) {
            auto* of = reinterpret_cast<std::atomic<uint64_t>*>(&flags[0]);
            auto* inf = reinterpret_cast<std::atomic<uint64_t>*>(&flags[8]);
            for (uint64_t s = 1; s <= cnt && !stop; ++s) {
                auto t = clk::now();
                while (of->load(std::memory_order_acquire) < s) {
                    __builtin_ia32_pause();
                    if (ms_since(t) > 5000) { std::printf("  proxy: no partial %llu\n", (unsigned long long)s); stop = true; return; }
                }
                for (size_t i = 0; i < n; ++i) hin[i] = hout[i] + peer_val(i, s);
                inf->store(s, std::memory_order_release);
            }
        };
        auto enqueue = [&](uint64_t cnt) {
            for (uint64_t s = 1; s <= cnt; ++s) {
                q.submit([&](sycl::handler& h) {
                    const size_t N = n; const uint64_t seq = s; const int WG = wg;
                    float* d = dev; float* o = hout; float* in = hin; uint64_t* f = flags;
                    h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
                        const int lid = int(it.get_local_id(0));
                        for (size_t i = lid; i < N; i += WG) o[i] = d[i];
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        it.barrier(sycl::access::fence_space::global_space);
                        if (lid == 0) {
                            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                             sycl::access::address_space::global_space> out_flag(f[0]);
                            out_flag.store(seq, sycl::memory_order::release);
                            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                             sycl::access::address_space::global_space> in_flag(f[8]);
                            uint64_t spins = 0;
                            while (in_flag.load(sycl::memory_order::acquire) < seq) {
                                if (++spins > (uint64_t(1) << 22)) {   // ~seconds: give up, never hang the GPU
                                    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                                     sycl::access::address_space::global_space> err(f[16]);
                                    err.store(seq);
                                    break;
                                }
                            }
                        }
                        it.barrier(sycl::access::fence_space::global_space);
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        for (size_t i = lid; i < N; i += WG) d[i] = in[i];
                    });
                });
            }
        };
        { std::thread p(proxy, uint64_t(50)); enqueue(50); q.wait(); p.join(); }
        reset();
        std::thread p(proxy, iters);
        auto t0 = clk::now();
        enqueue(iters);
        const double tq = ms_since(t0);
        q.wait();
        const double t = ms_since(t0);
        p.join();
        std::printf("B device-side poll, WG %4d:      %8.2f us per all-reduce (host queued all in %.1f ms)%s\n",
                    wg, t * 1000 / double(iters), tq, flags[16] ? "  TIMEOUT SEEN" : "");
        check(wg == 256 ? "B result (WG 256)" : "B result (WG 1024)", iters);
    }
    sycl::free(dev, q); sycl::free(hout, q); sycl::free(hin, q); sycl::free(flags, q);
    return 0;
}
