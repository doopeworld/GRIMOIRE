#include "xe2_grouped_bridge.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

static uint8_t sm4(int v) {
    return uint8_t(v) & 15;
}

int main() {
    constexpr int E=2, M=32, N=256, K=256, GS=128;
    sycl::queue q{sycl::gpu_selector_v};
    auto* a=sycl::malloc_shared<bf16>(size_t(M)*K,q);
    auto* b=sycl::malloc_shared<uint8_t>(size_t(E)*N*K/2,q);
    auto* s=sycl::malloc_shared<bf16>(size_t(E)*N*(K/GS),q);
    auto* d=sycl::malloc_shared<bf16>(size_t(M)*N,q);
    auto* rows=sycl::malloc_shared<int>(E,q);
    auto* atomic=sycl::malloc_shared<int>(1,q);
    rows[0]=13; rows[1]=19; atomic[0]=0;
    for(int m=0;m<M;++m) for(int k=0;k<K;++k)
        a[size_t(m)*K+k]=bf16(float(((m*7+k*3)%19)-9)/16.0f);
    for(int e=0;e<E;++e) for(int n=0;n<N;++n) {
        for(int g=0;g<K/GS;++g) s[(size_t(e)*N+n)*(K/GS)+g]=bf16(0.03125f*(1+g));
        for(int kb=0;kb<K/2;++kb) {
            int lo=((e*5+n*3+2*kb)%15)-7;
            int hi=((e*7+n+2*kb+1)%15)-7;
            b[(size_t(e)*N+n)*(K/2)+kb]=uint8_t(sm4(hi)<<4)|sm4(lo);
        }
    }
    std::fill(d,d+size_t(M)*N,bf16(0));
    grimoire_xe2_grouped_w4a16(&q,a,b,s,d,N,K,rows,nullptr,E,GS,atomic);
    q.wait_and_throw();
    double max_abs=0, max_rel=0, sum_abs=0, ref2=0, got2=0;
    for(int m=0;m<M;++m) for(int n=0;n<N;++n) {
        const int e=m<rows[0]?0:1;
        float ref=0;
        for(int k=0;k<K;++k) {
            uint8_t nib=(b[(size_t(e)*N+n)*(K/2)+k/2] >> ((k&1)*4))&15;
            int w=(nib&8) ? int(nib)-16 : int(nib);
            ref += float(a[size_t(m)*K+k])*w*float(s[(size_t(e)*N+n)*(K/GS)+k/GS]);
        }
        float got=float(d[size_t(m)*N+n]);
        if(m<2 && n<8) std::printf("m%d n%d ref=% .6f got=% .6f\n",m,n,ref,got);
        double ae=std::abs(double(got)-ref);
        ref2+=double(ref)*ref; got2+=double(got)*got;
        max_abs=std::max(max_abs,ae); sum_abs+=ae;
        max_rel=std::max(max_rel,ae/(std::abs(double(ref))+1e-3));
    }
    std::printf("max_abs=%.6f mean_abs=%.6f max_rel=%.6f ref_rms=%.6f got_rms=%.6f\n",
                max_abs,sum_abs/(M*N),max_rel,std::sqrt(ref2/(M*N)),std::sqrt(got2/(M*N)));
    sycl::free(a,q); sycl::free(b,q); sycl::free(s,q); sycl::free(d,q);
    sycl::free(rows,q); sycl::free(atomic,q);
    return max_abs < 0.2 ? 0 : 2;
}
