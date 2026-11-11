// SPDX-License-Identifier: GPL-3.0-or-later
// Short GPU0 microbenchmark for the TP2 matrix shapes absent from the old full-shape sweep.
#include "int4_smallm.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
using namespace b70;
int main() {
    sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::in_order{}};
    struct Shape{const char* name;int N,K;};
    const Shape shapes[]={{"tp_gate_up",17408,5120},{"tp_down",5120,8704},
                          {"tp_dn_qkv",5120,5120},{"tp_z",3072,5120},
                          {"tp_attn_out",5120,3072},{"tp_attn_q",6144,5120}};
    constexpr int M=7, GS=128;
    std::mt19937 rng(1111);
    for(auto sh:shapes) {
        const int N=sh.N,K=sh.K,G=K/GS;
        const size_t pb=size_t(N)*K/2,sb=size_t(N)*G*2,zb=size_t(N)*G,wb=pb+sb+zb;
        const int copies=std::clamp(int(((128ull<<20)+wb-1)/wb),4,64);
        std::vector<uint8_t> pay(pb),zero(zb,8);
        std::vector<sycl_bf16> scales(size_t(N)*G);
        for(auto& x:pay)x=uint8_t(rng());
        for(auto& x:scales)x=sycl_bf16(.001f*float(1+rng()%16));
        std::vector<QuantWeight> ws;
        for(int c=0;c<copies;++c) {
            QuantWeight w{};w.fmt=Fmt::INT4;w.N=N;w.K=K;w.row_bytes=K/2;w.row_scales=G;
            auto* p=sycl::malloc_device<uint8_t>(pb,q);
            auto* s=sycl::malloc_device<sycl_bf16>(size_t(N)*G,q);
            auto* z=sycl::malloc_device<uint8_t>(zb,q);
            if(!p||!s||!z)throw std::runtime_error("weight allocation failed");
            q.memcpy(p,pay.data(),pb);q.memcpy(s,scales.data(),sb);q.memcpy(z,zero.data(),zb);
            w.payload=p;w.scales=s;w.zeros=z;ws.push_back(w);
        }
        std::vector<sycl_bf16> hx(size_t(8)*K);
        for(auto& v:hx)v=sycl_bf16(float(int(rng()%2001)-1000)/1000.f);
        auto* x=sycl::malloc_device<sycl_bf16>(hx.size(),q);
        auto* y=sycl::malloc_device<float>(size_t(8)*N,q);
        if(!x||!y)throw std::runtime_error("activation allocation failed");
        q.memcpy(x,hx.data(),hx.size()*2);q.wait_and_throw();
        printf("SHAPE %s N=%d K=%d copies=%d bytes=%zu M=%d\n",sh.name,N,K,copies,wb,M);
        const int tiles=N/16;
        const int rule=K>=12288?4:tiles>=256&&tiles<600&&K>=4096?2:std::min(32,std::max(1,K/256));
        std::vector<int> seen;
        for(int requested:{0,2,4,8,12,16,20,24,32,-1}) {
            int ks=requested>0?requested:rule,kc=(K+ks-1)/ks;
            kc=(kc+127)/128*128;ks=(K+kc-1)/kc;
            if(requested>=0 && std::find(seen.begin(),seen.end(),ks)!=seen.end())continue;
            seen.push_back(ks);
            int tpt=1;while(tpt*ks<8&&tpt*2*ks<=32)tpt*=2;
            auto launch=[&](const QuantWeight& w){int4_smallm_fdq_impl<1,128>(q,w,x,y,M,ks,kc,tpt,{});};
            q.fill(y,-123456.f,size_t(8)*N);launch(ws[0]);q.wait_and_throw();
            std::vector<float> got(size_t(8)*N);q.memcpy(got.data(),y,got.size()*4).wait();
            for(size_t i=0;i<size_t(7)*N;++i)if(!std::isfinite(got[i]))throw std::runtime_error("nonfinite output");
            double md=0,mr=0;
            for(int r:{0,3,6})for(int j=0;j<32;++j) {
                const int n=(j*503+11)%N;double ref=0;
                for(int k=0;k<K;++k) {
                    const uint8_t b=pay[size_t(n)*K/2+k/2];
                    const int v=int((b>>(4*(k%2)))&15)-8;
                    ref+=double(float(hx[size_t(r)*K+k]))*v*float(scales[size_t(n)*G+k/GS]);
                }
                md=std::max(md,std::abs(double(got[size_t(r)*N+n])-ref));mr=std::max(mr,std::abs(ref));
            }
            for(int n=0;n<N;++n)if(got[size_t(7)*N+n]!=-123456.f)throw std::runtime_error("output row overrun");
            if(!std::isfinite(md)||md>1e-3+mr*1e-3)throw std::runtime_error("CPU reference mismatch");
            for(int i=0;i<128;++i)launch(ws[size_t(i)%ws.size()]);
            q.wait_and_throw();
            constexpr int iters=48;std::vector<double> trials;
            for(int trial=0;trial<3;++trial) {
                auto start=std::chrono::steady_clock::now();
                for(int i=0;i<iters;++i)launch(ws[size_t(i)%ws.size()]);
                q.wait_and_throw();
                trials.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/iters);
            }
            auto sorted=trials;std::sort(sorted.begin(),sorted.end());const double us=sorted[1];
            printf("PLAN %s requested=%d KS=%d kc=%d TPT=%d us=%.3f GBs=%.2f rel=%.3g trials=%.3f,%.3f,%.3f\n",sh.name,requested,ks,kc,tpt,us,double(wb)/us/1e3,mr>0?md/mr:md,trials[0],trials[1],trials[2]);fflush(stdout);
        }
        q.wait_and_throw();
        for(auto w:ws){sycl::free(const_cast<uint8_t*>(w.payload),q);sycl::free(const_cast<void*>(w.scales),q);sycl::free(const_cast<uint8_t*>(w.zeros),q);}
        sycl::free(x,q);sycl::free(y,q);
    }
    puts("TP_INT4_PROBE_PASS");
}
