// Native grouped small-M GEMM: independent FP64 oracle, ragged/shuffled experts,
// padding, output canaries and the existing dense path. GPU0 guarded launcher only.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "native_smallm.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace b70;
static uint16_t bf(float x) {
    uint32_t u; std::memcpy(&u,&x,4);
    return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
static float fl(uint16_t x) {
    uint32_t u=uint32_t(x)<<16; float f; std::memcpy(&f,&u,4); return f;
}

template<Fmt F>
static int check(sycl::queue& q, int Ne, int K, int M) {
    constexpr int E=7;
    const int64_t rb=bytes_per_row(F,K);
    std::mt19937 rng(1937+int(F)*31+M+K+Ne);
    std::vector<uint8_t> hw(size_t(E)*Ne*rb);
    for(size_t i=0;i<hw.size();++i) hw[i]=uint8_t(rng());
    auto scale=[&](uint8_t* p){const uint16_t v=uint16_t(0x2000+rng()%0x1000);
        std::memcpy(p,&v,2);};
    for(int n=0;n<E*Ne;++n)
        for(int k=0;k<K/NativeGGUFBlock<F>::elems;++k){
            uint8_t* p=hw.data()+size_t(n)*rb+size_t(k)*NativeGGUFBlock<F>::bytes;
            if constexpr(F==Fmt::Q6_K) scale(p+208);
            else {scale(p); if constexpr(F==Fmt::Q4_K||F==Fmt::Q5_K) scale(p+2);}
        }
    std::vector<int32_t> count={M,0,1,std::max(1,M-1),std::min(2,M),M/2+1,1};
    std::vector<int32_t> off(E);
    int rows=1;
    for(int e=E-1;e>=0;--e){off[e]=rows; rows+=count[e]+1;}
    ++rows;
    const std::vector<int32_t> table={-1,5,2,0,6,3,1,4,-1};
    std::vector<uint16_t> hx(size_t(rows)*K);
    for(auto& x:hx) x=bf((float(rng()%2001)-1000.f)/1000.f);
    constexpr float sentinel=0x1.345678p+45f;
    std::vector<float> hy(size_t(rows)*Ne,sentinel);
    auto* dw=sycl::malloc_device<uint8_t>(hw.size(),q);
    auto* dx=sycl::malloc_device<sycl_bf16>(hx.size(),q);
    auto* dy=sycl::malloc_device<float>(hy.size(),q);
    auto* dt=sycl::malloc_device<int32_t>(table.size(),q);
    auto* dc=sycl::malloc_device<int32_t>(E,q);
    auto* doff=sycl::malloc_device<int32_t>(E,q);
    if(!dw||!dx||!dy||!dt||!dc||!doff) throw std::runtime_error("probe allocation");
    q.memcpy(dw,hw.data(),hw.size());q.memcpy(dx,hx.data(),hx.size()*2);
    q.memcpy(dy,hy.data(),hy.size()*4);q.memcpy(dt,table.data(),table.size()*4);
    q.memcpy(dc,count.data(),E*4);q.memcpy(doff,off.data(),E*4);
    q.wait_and_throw();
    QuantWeight w;w.fmt=F;w.N=E*Ne;w.K=K;w.row_bytes=rb;w.payload=dw;
    if(!native_moe_smallm_layout_ok(w,Ne,M)) throw std::runtime_error("valid shape refused");
    auto launch=[&]{return M<=8
        ?native_moe_smallm_launch<1>(q,w,Ne,dx,dy,M,dt,doff,dc,int(table.size()),{})
        :native_moe_smallm_launch<2>(q,w,Ne,dx,dy,M,dt,doff,dc,int(table.size()),{});};
    launch().wait_and_throw();
    q.memcpy(hy.data(),dy,hy.size()*4).wait_and_throw();
    std::vector<unsigned char> live(static_cast<size_t>(rows),0);
    int bad=0;double maxrel=0;
    auto compare=[&](int outrow,int n,int weightrow,float got){
        const uint8_t* p=hw.data()+size_t(weightrow)*rb;
        double ref=0,mag=0;
        for(int k=0;k<K;++k){const double v=double(decode_elem<F>(p,k,1.f))*fl(hx[size_t(outrow)*K+k]);
            ref+=v;mag+=std::fabs(v);}
        const double rel=std::fabs(ref-double(got))/(mag+1e-30);
        maxrel=std::max(maxrel,rel);
        if(!(rel<1e-5)&&bad++<3) std::printf("mismatch fmt=%d row=%d n=%d ref=%.7g got=%.7g\n",int(F),outrow,n,ref,got);
    };
    for(int e=0;e<E;++e) for(int m=0;m<count[e];++m){
        const int r=off[e]+m;live[r]=1;
        const int samples=std::min(Ne,64);
        for(int s=0;s<samples;++s){const int n=Ne<=64?s:(s<16?s:(s*7919+13)%Ne);
            compare(r,n,e*Ne+n,hy[size_t(r)*Ne+n]);}
    }
    for(int r=0;r<rows;++r) if(!live[r])
        for(int n=0;n<Ne;++n) if(hy[size_t(r)*Ne+n]!=sentinel) ++bad;
    // Timing is internally bounded; never rely on a container timeout to stop kernels.
    const auto start=std::chrono::steady_clock::now();
    int reps=0;
    do{launch().wait_and_throw();++reps;}
    while(reps<20&&std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<0.25);
    const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/reps;
    std::printf("grouped fmt=%d Ne=%d K=%d M=%d: %.1f us maxrel=%.2e bad=%d\n",int(F),Ne,K,M,us,maxrel,bad);
    // The shared implementation's dense instantiation must retain its original math.
    if(Ne==48){
        std::vector<float> dense(size_t(M)*w.N);
        auto* dd=sycl::malloc_device<float>(dense.size(),q);
        if(M<=8)native_smallm_launch<1>(q,w,dx,dd,M,{}).wait_and_throw();
        else native_smallm_launch<2>(q,w,dx,dd,M,{}).wait_and_throw();
        q.memcpy(dense.data(),dd,dense.size()*4).wait_and_throw();
        for(int m=0;m<M;++m) for(int s=0;s<48;++s){
            const int n=(s*7919+13)%w.N;compare(m,n,n,dense[size_t(m)*w.N+n]);}
        sycl::free(dd,q);
    }
    // Shape refusals are checked before any launch, including partial Q6 blocks.
    if(native_moe_smallm_layout_ok(w,Ne-1,M)||native_moe_smallm_layout_ok(w,Ne,17))++bad;
    if constexpr(F==Fmt::Q6_K){auto odd=w;odd.K=256;odd.row_bytes=210;
        if(native_moe_smallm_layout_ok(odd,Ne,M))++bad;}
    sycl::free(dw,q);sycl::free(dx,q);sycl::free(dy,q);sycl::free(dt,q);
    sycl::free(dc,q);sycl::free(doff,q);
    return bad;
}

int main(){
    std::setvbuf(stdout,nullptr,_IONBF,0);
    try{
        sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::in_order{}};
        std::printf("device: %s\n",q.get_device().get_info<sycl::info::device::name>().c_str());
        int bad=0;
        for(int M:{2,3,8,9,16}){
            bad+=check<Fmt::Q4_K>(q,48,512,M);bad+=check<Fmt::Q5_K>(q,48,512,M);
            bad+=check<Fmt::Q6_K>(q,48,512,M);bad+=check<Fmt::Q8_0>(q,48,512,M);
            bad+=check<Fmt::IQ4_XS>(q,48,512,M);bad+=check<Fmt::IQ4_NL>(q,48,512,M);
        }
        bad+=check<Fmt::Q4_K>(q,1024,2048,8);
        bad+=check<Fmt::Q6_K>(q,2048,512,8);
        std::printf("NATIVE_MOE_SMALLM_%s bad=%d\n",bad?"FAIL":"PASS",bad);
        return bad?1:0;
    }catch(const std::exception& e){std::fprintf(stderr,"probe failed: %s\n",e.what());return 1;}
}
