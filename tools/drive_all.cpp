// Pack all 64 FFN layers from the int4 checkpoint, then time one gate/up/down call.
#include <sycl/sycl.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>
typedef std::chrono::steady_clock clk;
typedef int (*InitAll)(const char*,int,int*,int*,int*,int*);
typedef int (*Ffn)(sycl::queue*,int,int,const void*,void*,int);
int main(int argc,char**argv){
    const int M=argc>1?std::atoi(argv[1]):4096;
    const int NL=argc>2?std::atoi(argv[2]):64;
    sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
    std::printf("device: %s   M=%d layers=%d\n",
        q.get_device().get_info<sycl::info::device::name>().c_str(),M,NL);
    void* h=dlopen("/grimoire/src/libgrimoire_bestla.so",RTLD_NOW|RTLD_GLOBAL);
    if(!h){ std::fprintf(stderr,"dlopen: %s\n",dlerror()); return 2; }
    auto init=(InitAll)dlsym(h,"grimoire_bestla_init_all");
    auto ffn =(Ffn    )dlsym(h,"grimoire_bestla_ffn");
    if(!init||!ffn){ std::fprintf(stderr,"dlsym failed\n"); return 3; }
    int Ng=0,Kg=0,Nd=0,Kd=0;
    auto t0=clk::now();
    int rc=init("/models/Qwen3.8-27B-int4-AutoRound",NL,&Ng,&Kg,&Nd,&Kd);
    double packms=std::chrono::duration<double,std::milli>(clk::now()-t0).count();
    if(rc){ std::fprintf(stderr,"init_all rc=%d\n",rc); return 4; }
    std::printf("  gate/up N=%d K=%d   down N=%d K=%d   pack time %.1f s\n",
                Ng,Kg,Nd,Kd,packms/1000.0);
    auto* x=sycl::malloc_device<sycl::ext::oneapi::bfloat16>(size_t(M)*Kg,q);
    auto* xd=sycl::malloc_device<sycl::ext::oneapi::bfloat16>(size_t(M)*Kd,q);
    auto* o=sycl::malloc_device<sycl::ext::oneapi::bfloat16>(size_t(M)*(Ng>Nd?Ng:Nd),q);
    if(!x||!xd||!o){ std::fprintf(stderr,"USM alloc failed (out of VRAM)\n"); return 5; }
    { std::vector<sycl::ext::oneapi::bfloat16> hb(size_t(M)*Kg);
      std::mt19937 rng(3); std::uniform_real_distribution<float> d(-1.f,1.f);
      for(auto&v:hb) v=sycl::ext::oneapi::bfloat16(d(rng));
      q.memcpy(x,hb.data(),hb.size()*2).wait();
      q.memset(xd,0,size_t(M)*Kd*2).wait(); }
    auto bench=[&](int which,const void* in,int N,const char* nm){
        if(ffn(&q,0,which,in,o,M)){ std::fprintf(stderr,"%s call failed\n",nm); return; }
        q.wait(); double best=1e18;
        for(int i=0;i<5;++i){ auto t=clk::now(); ffn(&q,i%NL,which,in,o,M); q.wait();
            double ms=std::chrono::duration<double,std::milli>(clk::now()-t).count();
            if(i&&ms<best)best=ms; }
        std::printf("  %-10s %8.3f ms   %6.1f TFLOP/s\n",nm,best,2.0*M*N*Kg/best*1e-9);
    };
    bench(0,x,Ng,"gate_proj"); bench(1,x,Ng,"up_proj"); bench(2,xd,Nd,"down_proj");
    std::printf("\n  ALL-LAYER PACK: PASS\n");
    return 0;
}
