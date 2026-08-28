// Memory roofline probe for Arc Pro B70. Sweeps buffer sizes so cache-resident
// results are distinguishable from true DRAM streaming bandwidth.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>

using u32 = uint32_t;
using vec4 = sycl::vec<u32, 4>;

int main() {
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  auto d = q.get_device();
  std::printf("device : %s | %u EUs | %.2f GiB\n",
              d.get_info<sycl::info::device::name>().c_str(),
              d.get_info<sycl::ext::intel::info::device::gpu_eu_count>(),
              double(d.get_info<sycl::info::device::global_mem_size>()) / (1<<30));

  const size_t wg      = 256;
  const size_t threads = size_t(1) << 20;
  u32* out = sycl::malloc_device<u32>(threads, q);   // always written: keeps loads live

  for (size_t mib : {64ull, 256ull, 1024ull, 4096ull, 8192ull}) {
    const size_t bytes = mib * 1024 * 1024;
    const size_t n4    = bytes / sizeof(vec4);
    vec4* buf = sycl::malloc_device<vec4>(n4, q);
    if (!buf) { std::printf("%5zu MiB: alloc failed\n", mib); continue; }
    // Fill with an incompressible pseudo-random pattern: a constant buffer is
    // transparently compressed by the memory controller and reports fake bandwidth.
    q.parallel_for(sycl::range<1>{n4}, [=](sycl::id<1> i) {
      u32 x = u32(i[0]) * 2654435761u + 1013904223u;
      auto mix = [](u32 v) { v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; return v; };
      buf[i[0]] = vec4{mix(x), mix(x+1u), mix(x+2u), mix(x+3u)};
    }).wait();

    auto run = [&]() {
      return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>{threads, wg}, [=](sycl::nd_item<1> it) {
          size_t gid = it.get_global_id(0), stride = it.get_global_range(0);
          vec4 acc{0,0,0,0};
          for (size_t i = gid; i < n4; i += stride) acc += buf[i];
          out[gid] = acc[0] + acc[1] + acc[2] + acc[3];   // unconditional write
        });
      });
    };
    run().wait();

    double best = 0;
    for (int i = 0; i < 5; ++i) {
      auto t0 = std::chrono::high_resolution_clock::now();
      run().wait();
      auto t1 = std::chrono::high_resolution_clock::now();
      double sec = std::chrono::duration<double>(t1 - t0).count();
      best = std::max(best, double(bytes) / sec / 1e9);
    }
    std::printf("%5zu MiB: %.1f GB/s\n", mib, best);
    sycl::free(buf, q);
  }
  sycl::free(out, q);
  return 0;
}
