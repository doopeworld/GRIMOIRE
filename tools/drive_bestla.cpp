// Thin SYCL driver: allocates a USM buffer and drives the BesTLA bridge.
#include <sycl/sycl.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <cstdlib>
typedef std::chrono::steady_clock clk;
typedef int (*InitFn)(int*,int*,int*);
typedef int (*LinFn)(sycl::queue*,const void*,int);
int main(int argc,char**argv){
    const int M=argc>1?std::atoi(argv[1]):4096;
    sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n",q.get_device().get_info<sycl::info::device::name>().c_str());
    void* h=dlopen("/grimoire/src/libgrimoire_bestla.so",RTLD_NOW|RTLD_GLOBAL);
    if(!h){ std::fprintf(stderr,"dlopen: %s\n",dlerror()); return 2; }
    auto init=(InitFn)dlsym(h,"grimoire_bestla_init");
    auto lin =(LinFn )dlsym(h,"grimoire_bestla_linear");
    if(!init||!lin){ std::fprintf(stderr,"dlsym failed\n"); return 3; }
    int N=0,K=0,G=0;
    int rc=init(&N,&K,&G);
    if(rc){ std::fprintf(stderr,"init failed rc=%d\n",rc); return 4; }
    std::printf("  packed real checkpoint weight: N=%d K=%d G=%d\n",N,K,G);
    auto* x=sycl::malloc_device<sycl::ext::oneapi::bfloat16>(size_t(M)*K,q);
    { std::vector<sycl::ext::oneapi::bfloat16> hb(size_t(M)*K);
      std::mt19937 rng(3); std::uniform_real_distribution<float> d(-1.f,1.f);
      for(auto&v:hb) v=sycl::ext::oneapi::bfloat16(d(rng));
      q.memcpy(x,hb.data(),hb.size()*2).wait(); }
    std::printf("  GRIMOIRE USM buffer %d x %d handed to BesTLA\n",M,K);
    if(lin(&q,x,M)){ std::fprintf(stderr,"first call failed\n"); return 5; }
    q.wait();
    double best=1e18;
    for(int i=0;i<6;++i){ auto t0=clk::now(); lin(&q,x,M); q.wait();
        double ms=std::chrono::duration<double,std::milli>(clk::now()-t0).count();
        if(i&&ms<best)best=ms; }
    const double fl=2.0*M*N*K;
    // weight traffic dominates at small M: int4 payload + bf16 scales per group
    const double wbytes=double(N)*K*0.5 + double(N)*(K/G)*2.0;
    std::printf("\n  M=%d  BesTLA: %8.3f ms  %7.1f TFLOP/s  %7.1f GB/s (weights)\n",
                M,best,fl/best*1e-9,wbytes/best*1e-6);
    return 0;
}
