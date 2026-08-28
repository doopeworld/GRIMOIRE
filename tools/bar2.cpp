// Does device-1 allocation fail AFTER device 0 is heavily loaded?
// That is the USB4 peer/BAR-mapping limit signature (single process).
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<sycl::device> g;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_info<sycl::info::device::name>().find("B70") != std::string::npos) g.push_back(d);
    if (g.size()<2){printf("need 2\n");return 1;}
    sycl::context shared{g};
    sycl::queue q0{shared, g[0]}, q1{shared, g[1]};

    // Fill device 0 to ~10 GB with real (backed) memory, like the model load.
    printf("filling device 0...\n");
    std::vector<void*> d0;
    for (int i=0;i<400;i++){ void* p=sycl::malloc_device(size_t(25)<<20,q0);
        q0.memset(p,1,size_t(25)<<20).wait(); d0.push_back(p); }
    printf("device 0 now holds ~10 GB via 400 small allocs (fully backed)\n");

    // Now try to allocate + back memory on device 1.
    printf("now allocating on device 1:\n");
    double got=0;
    for (int i=0;i<40;i++){
        try { void* p=sycl::malloc_device(size_t(256)<<20,q1);
              q1.memset(p,2,size_t(256)<<20).wait(); got+=0.25; }
        catch(sycl::exception& e){ printf("  device1 FAILED at %.2f GiB: %s\n",got,e.what()); break; }
    }
    printf("device 1 reached %.2f GiB after device 0 was full\n", got);
    return 0;
}
