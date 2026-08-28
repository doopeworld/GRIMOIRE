// Validate the FA2 bridge against GRIMOIRE's existing chunk-prefill path:
// same inputs, compare outputs, time both.  Qwen3.8-27B full-attention shape.
#include <sycl/sycl.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
using bf16 = sycl::ext::oneapi::bfloat16;
typedef void (*PrefillFn)(sycl::queue*, const void*, const void*, const void*, void*,
                          int,int,int,int,int, const int*, const int*, float, bool);
typedef int  (*AvailFn)();
typedef std::chrono::steady_clock clk;

static void* open_or_die(const char* p){
    void* h=dlopen(p,RTLD_NOW|RTLD_LOCAL);
    if(!h){ std::fprintf(stderr,"dlopen %s: %s\n",p,dlerror()); std::exit(2); }
    return h;
}
int main(int argc,char**argv){
    const int M   = argc>1?std::atoi(argv[1]):4096;   // tokens
    const int QH  = 24, KH = 4, D = 256;              // Qwen3.8-27B
    sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
    std::printf("device: %s   M=%d QH=%d KH=%d D=%d\n",
        q.get_device().get_info<sycl::info::device::name>().c_str(),M,QH,KH,D);

    void* hf=open_or_die("/grimoire/src/libgrimoire_xe2_fa2.so");
    auto avail=(AvailFn)dlsym(hf,"grimoire_xe2_fa2_available");
    auto fa2  =(PrefillFn)dlsym(hf,"grimoire_xe2_fa2_prefill_bf16");
    std::printf("FA2 op registered: %s\n", (avail&&avail())?"YES":"NO");
    if(!avail||!avail()||!fa2){ std::fprintf(stderr,"FA2 unavailable\n"); return 3; }

    void* ho=open_or_die("/grimoire/src/libgrimoire_xe2_attention_raw.so");
    auto old=(PrefillFn)dlsym(ho,"grimoire_xe2_chunk_prefill_bf16");
    std::printf("baseline (attention_raw) loaded: %s\n", old?"YES":"NO");

    const size_t qe=size_t(M)*QH*D, ke=size_t(M)*KH*D;
    bf16 *Q=sycl::malloc_device<bf16>(qe,q), *K=sycl::malloc_device<bf16>(ke,q),
         *V=sycl::malloc_device<bf16>(ke,q), *O1=sycl::malloc_device<bf16>(qe,q),
         *O2=sycl::malloc_device<bf16>(qe,q);
    int *cq=sycl::malloc_device<int>(2,q), *ck=sycl::malloc_device<int>(2,q);
    { std::mt19937 rng(5); std::uniform_real_distribution<float> d(-1.f,1.f);
      std::vector<bf16> h(qe); for(auto&x:h)x=bf16(d(rng));
      q.memcpy(Q,h.data(),qe*sizeof(bf16)).wait();
      h.resize(ke); for(auto&x:h)x=bf16(d(rng));
      q.memcpy(K,h.data(),ke*sizeof(bf16)).wait();
      for(auto&x:h)x=bf16(d(rng));
      q.memcpy(V,h.data(),ke*sizeof(bf16)).wait();
      int hcu[2]={0,M}; q.memcpy(cq,hcu,8).wait(); q.memcpy(ck,hcu,8).wait();
      q.memset(O1,0,qe*sizeof(bf16)).wait(); q.memset(O2,0,qe*sizeof(bf16)).wait(); }

    const float scale=1.0f/std::sqrt(float(D));
    auto bench=[&](PrefillFn f,bf16*O,const char*name){
        f(&q,Q,K,V,O,M,M,QH,KH,D,cq,ck,scale,true); q.wait();
        double best=1e18;
        for(int i=0;i<5;++i){ auto t=clk::now(); f(&q,Q,K,V,O,M,M,QH,KH,D,cq,ck,scale,true);
            q.wait(); double ms=std::chrono::duration<double,std::milli>(clk::now()-t).count();
            if(ms<best)best=ms; }
        double fl=2.0*(double(M)*M/2*QH*D*2);
        std::printf("  %-22s %9.3f ms   %7.1f TFLOP/s\n",name,best,fl/best*1e-9);
        return best;
    };
    std::printf("\n");
    double tb = old?bench(old,O2,"attention_raw (ours)"):0;
    double tf = bench(fa2,O1,"FA2 (vLLM)");

    if(old){
        std::vector<bf16> a(qe),b(qe);
        q.memcpy(a.data(),O1,qe*sizeof(bf16)).wait();
        q.memcpy(b.data(),O2,qe*sizeof(bf16)).wait();
        double mx=0,sum=0; size_t nz=0;
        for(size_t i=0;i<qe;++i){ double x=float(a[i]),y=float(b[i]);
            double e=std::fabs(x-y); if(e>mx)mx=e; sum+=e; if(x!=0.0)++nz; }
        std::printf("\n  max|FA2-ours| = %.4f   mean = %.5f   nonzero FA2 outputs = %zu/%zu\n",
                    mx,sum/qe,nz,qe);
        std::printf("  speedup: %.2fx\n", tb/tf);
    }
    return 0;
}
