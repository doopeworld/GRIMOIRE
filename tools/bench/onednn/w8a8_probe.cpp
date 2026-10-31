// Exercise the integrated oneDNN entry point, including activation quantization.
// Checks its integer math against FP64 and reports the separate error vs BF16 input.
// Weight copies span 512 MiB; timing is the whole call, not just its final GEMM event.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "onednn_gemm.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace b70;

static int run(sycl::queue& q,int M,int N,int K,bool integer_input){
    std::mt19937 rng(19237+M+N+K);
    const size_t wb=size_t(N)*K;
    const int copies=wb<(size_t(8)<<20)?2:int(std::max<size_t>(2,(size_t(512)<<20)/wb+1));
    std::vector<int8_t> hw(wb);
    for(auto& v:hw)v=int8_t(int(rng()%255)-127);
    std::vector<float> ws(static_cast<size_t>(N));
    for(int n=0;n<N;++n)ws[n]=0.0002f+float(rng()%100)*0.00001f;
    std::vector<sycl_bf16> hx(size_t(M)*K);
    std::vector<int8_t> hq(hx.size());
    std::vector<float> xs(static_cast<size_t>(M));
    for(int m=0;m<M;++m){
        const float factor=m==1?0.0001f:1.f;
        float maximum=0.f;
        for(int k=0;k<K;++k){
            float f=(float(rng()%2001)-1000.f)*0.001f*factor;
            if(m==0)f=0.f;
            if(m==2&&k==0)f=17.f;
            hx[size_t(m)*K+k]=sycl_bf16(f);
            maximum=std::max(maximum,std::fabs(float(hx[size_t(m)*K+k])));
        }
        xs[m]=maximum>0?maximum/127.f:1.f;
        const float inverse=1.f/xs[m];
        for(int k=0;k<K;++k)hq[size_t(m)*K+k]=int8_t(std::clamp(
            std::nearbyint(float(hx[size_t(m)*K+k])*inverse),-127.f,127.f));
    }
    std::vector<uint8_t*> dw(static_cast<size_t>(copies));
    for(auto& p:dw){p=sycl::malloc_device<uint8_t>(wb,q);if(!p)throw std::bad_alloc();q.memcpy(p,hw.data(),wb);}
    auto* dws=sycl::malloc_device<float>(N,q);
    auto* dx=sycl::malloc_device<sycl_bf16>(hx.size(),q);
    auto* dy=sycl::malloc_device<float>(size_t(M)*N,q);
    if(!dws||!dx||!dy)throw std::bad_alloc();
    q.memcpy(dws,ws.data(),size_t(N)*4);q.memcpy(dx,hx.data(),hx.size()*2);q.wait_and_throw();
    QuantWeight w;w.fmt=Fmt::INT8;w.N=N;w.K=K;w.payload=dw[0];w.scales=dws;
    w.row_bytes=K;w.row_scales=1;
    auto launch=[&](int c,bool acc){w.payload=dw[size_t(c)];return onednn_gemm(q,w,dx,dy,M,acc,{});};
    launch(0,false).wait_and_throw();
    std::vector<float> out(size_t(M)*N);q.memcpy(out.data(),dy,out.size()*4).wait_and_throw();
    int bad=0;double worst=0,err_bf=0,norm_bf=0;
    for(int s=0;s<96;++s){
        const int m=s==0?0:s==1?1:s==2?2:s==3?M-1:(s*1171)%M;
        const int n=s<16?s:(s*7919+13)%N;
        double ref=0,original=0,mag=0;
        for(int k=0;k<K;++k){
            const double weight=double(hw[size_t(n)*K+k])*ws[n];
            const double x=integer_input?double(hq[size_t(m)*K+k])*xs[m]:float(hx[size_t(m)*K+k]);
            ref+=x*weight;mag+=std::fabs(x*weight);
            original+=double(float(hx[size_t(m)*K+k]))*weight;
        }
        const double got=out[size_t(m)*N+n],rel=std::fabs(got-ref)/(mag+1e-30);
        worst=std::max(worst,rel);
        if(!(rel<1e-5)&&bad++<4)std::printf("mismatch %d,%d: ref %.9g got %.9g rel %.3g\n",m,n,ref,got,rel);
        err_bf+=(got-original)*(got-original);norm_bf+=original*original;
    }
    for(int n=0;n<N;++n)if(out[n]!=0.f)++bad;
    // Sum post-op must retain the old destination, including a zero source row.
    if(N==256){
        std::fill(out.begin(),out.end(),0.5f);q.memcpy(dy,out.data(),out.size()*4).wait();
        launch(0,true).wait_and_throw();q.memcpy(out.data(),dy,out.size()*4).wait();
        for(int n=0;n<N;++n)if(out[n]!=0.5f)++bad;
    }
    for(int c=0;c<copies;++c)launch(c,false).wait_and_throw();
    std::vector<double> times;
    const auto budget=std::chrono::steady_clock::now();
    for(int i=0;i<40;++i){
        const auto t=std::chrono::steady_clock::now();
        launch(i%copies,false).wait_and_throw();
        times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
        if(times.size()>=size_t(copies)&&std::chrono::duration<double>(std::chrono::steady_clock::now()-budget).count()>1.)break;
    }
    std::sort(times.begin(),times.end());const double ms=times[times.size()/2];
    std::printf("mode=%s M=%d N=%d K=%d copies=%d whole=%.3fms effective=%.1fTOPS oracle_rel=%.2e activation_rel_rms=%.3g bad=%d\n",
        integer_input?"W8A8":"W8A16",M,N,K,copies,ms,2.*M*N*K/(ms*1.e9),worst,
        std::sqrt(err_bf/(norm_bf+1e-30)),bad);
    for(auto p:dw)sycl::free(p,q);sycl::free(dws,q);sycl::free(dx,q);sycl::free(dy,q);
    return bad;
}
int main(int argc,char** argv){
    std::setvbuf(stdout,nullptr,_IONBF,0);
    try{
        const bool a8=argc>1&&std::atoi(argv[1])!=0;
        setenv("GRIMOIRE_ONEDNN","1",1);setenv("GRIMOIRE_ONEDNN_VERBOSE","1",1);
        setenv("GRIMOIRE_ONEDNN_W8A8",a8?"1":"0",1);
        sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::in_order{}};
        int bad=0;
        bad+=run(q,17,256,512,false); // feature must leave small verify batches unchanged
        bad+=run(q,33,256,512,a8);
        bad+=run(q,513,32768,512,a8); // wide row pieces and a padded one-row tail
        bad+=run(q,512,34816,5120,a8);
        bad+=run(q,2048,5120,17408,a8);
        std::printf("ONEDNN_W8A8_%s bad=%d\n",bad?"FAIL":"PASS",bad);
        return bad?1:0;
    }catch(const std::exception& e){std::fprintf(stderr,"probe failed: %s\n",e.what());return 1;}
}
