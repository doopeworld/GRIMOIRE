// How much can a single process actually allocate on each B70?
// If the USB4-attached card caps at ~256MB, that's the Thunderbolt BAR limit
// and single-process pipeline is blocked until OCuLink.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<sycl::device> g;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_info<sycl::info::device::name>().find("B70") != std::string::npos) g.push_back(d);
    printf("B70s: %zu\n", g.size());
    sycl::context ctx{g};
    for (size_t i = 0; i < g.size(); ++i) {
        const double total = double(g[i].get_info<sycl::info::device::global_mem_size>())/(1<<30);
        const double maxalloc = double(g[i].get_info<sycl::info::device::max_mem_alloc_size>())/(1<<30);
        printf("device %zu: total=%.1f GiB  max_alloc=%.1f GiB\n", i, total, maxalloc);
        sycl::queue q{ctx, g[i]};
        double got = 0; std::vector<void*> ps;
        for (int mb = 0; mb < 40000; mb += 256) {   // step 256MB up to 40GB
            void* p = sycl::malloc_device(size_t(256)<<20, q);
            if (!p) { printf("  malloc returned null at %.2f GiB\n", got/1024); break; }
            try { q.memset(p, 0, 1<<20).wait(); }
            catch (sycl::exception& e) { printf("  FAILED at %.2f GiB allocated: %s\n", got/1024, e.what()); sycl::free(p,q); break; }
            ps.push_back(p); got += 256;
        }
        printf("  reached %.2f GiB before failure\n", got/1024);
        for (void* p : ps) sycl::free(p, q);
    }
    return 0;
}
