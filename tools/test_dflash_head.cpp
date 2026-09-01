#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using HeadFn = void (*)(sycl::queue*, const void*, const unsigned char*,
    const float*, const float*, void*, int, int, int);

static float f16_to_f32(uint16_t h) {
    const uint32_t s=uint32_t(h&0x8000u)<<16;
    uint32_t e=(h>>10)&0x1fu,m=h&0x3ffu,out;
    if(e==0){
        if(m==0)out=s;
        else{e=1;while((m&0x400u)==0){m<<=1;--e;}m&=0x3ffu;
            out=s|((e+112u)<<23)|(m<<13);}
    }else if(e==31)out=s|0x7f800000u|(m<<13);
    else out=s|((e+112u)<<23)|(m<<13);
    float v;std::memcpy(&v,&out,sizeof(v));return v;
}

template<class T> static bool read_exact(const char* path,std::vector<T>& v) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||size_t(f.tellg())!=v.size()*sizeof(T))return false;
    f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T));
    return bool(f);
}

int main(int argc,char** argv) {
    if(argc!=4){
        std::fprintf(stderr,"usage: %s HEAD.q4g64 IDS.i32 BRIDGE.so\n",argv[0]);
        return 2;
    }
    constexpr int M=7,N=131072,K=2048,G=64,KG=K/G;
    constexpr size_t CB=size_t(N)*K/2,SC=size_t(N)*KG;
    std::vector<uint8_t> payload(CB+SC*sizeof(uint16_t));
    std::vector<int32_t> ids(N);
    if(!read_exact(argv[1],payload)||!read_exact(argv[2],ids)){
        std::fprintf(stderr,"bad head or token-id extract\n");return 2;
    }
    std::vector<float> ws(SC);
    const auto* hs=reinterpret_cast<const uint16_t*>(payload.data()+CB);
    for(size_t i=0;i<SC;++i)ws[i]=f16_to_f32(hs[i]);
    std::vector<int8_t> a(size_t(M)*K);
    std::vector<float> as(M),out(size_t(M)*N);
    std::mt19937 rng(1);std::uniform_int_distribution<int> d(-100,100);
    for(auto& x:a)x=int8_t(d(rng));
    for(int r=0;r<M;++r)as[r]=0.005f+0.001f*r;

    void* lib=dlopen(argv[3],RTLD_NOW|RTLD_LOCAL);
    auto fn=lib?reinterpret_cast<HeadFn>(
        dlsym(lib,"grimoire_xe2_dense_w4a8_f32_m8g64")):nullptr;
    if(!fn){std::fprintf(stderr,"missing Q4G64 bridge symbol: %s\n",dlerror());return 2;}
    sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::in_order{}};
    auto* da=sycl::malloc_device<int8_t>(a.size(),q);
    auto* dw=sycl::malloc_device<uint8_t>(CB,q);
    auto* ds=sycl::malloc_device<float>(SC,q);
    auto* das=sycl::malloc_device<float>(M,q);
    auto* dout=sycl::malloc_device<float>(out.size(),q);
    if(!da||!dw||!ds||!das||!dout)return 2;
    q.memcpy(da,a.data(),a.size());q.memcpy(dw,payload.data(),CB);
    q.memcpy(ds,ws.data(),SC*sizeof(float));q.memcpy(das,as.data(),M*sizeof(float)).wait();
    for(int i=0;i<3;++i)fn(&q,da,dw,ds,das,dout,M,N,K);q.wait();
    constexpr int ITERS=20;
    const auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<ITERS;++i)fn(&q,da,dw,ds,das,dout,M,N,K);
    q.wait();
    const double ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-t0).count()/ITERS;
    q.memcpy(out.data(),dout,out.size()*sizeof(float)).wait();

    const int checks[]={0,1,17,65535,131071};
    float max_abs=0,max_rel=0;
    for(int r=0;r<M;++r)for(int n:checks){
        float ref=0;
        for(int g=0;g<KG;++g){
            int32_t acc=0;
            const uint8_t* w=payload.data()+size_t(n)*(K/2)+g*(G/2);
            for(int j=0;j<G;++j){
                const uint8_t u=(w[j/2]>>((j&1)*4))&15u;
                const int8_t s4=int8_t(u&8u?int(u)-16:int(u));
                acc+=int32_t(a[size_t(r)*K+g*G+j])*int32_t(s4);
            }
            ref+=float(acc)*ws[size_t(n)*KG+g]*as[r];
        }
        const float got=out[size_t(r)*N+n],ae=std::abs(got-ref);
        max_abs=std::max(max_abs,ae);
        max_rel=std::max(max_rel,ae/(std::abs(ref)+1e-5f));
    }
    for(int r=0;r<M;++r){
        const auto b=out.begin()+size_t(r)*N;
        const int idx=int(std::max_element(b,b+N)-b);
        std::printf("row %d proposal=%d token=%d value=%.6f\n",
                    r,idx,ids[idx],out[size_t(r)*N+idx]);
    }
    std::printf("Q4G64 head: %.3f ms, max_abs %.6g, max_rel %.6g %s\n",
                ms,max_abs,max_rel,(max_rel<2e-4f||max_abs<2e-3f)?"PASS":"FAIL");
    sycl::free(da,q);sycl::free(dw,q);sycl::free(ds,q);sycl::free(das,q);
    sycl::free(dout,q);dlclose(lib);
    return (max_rel<2e-4f||max_abs<2e-3f)?0:1;
}
