#include "xe2_attention_bridge.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using bf16=sycl::ext::oneapi::bfloat16;

int main(){
    constexpr int M=128,QH=16,KH=2,D=256;
    sycl::queue q{sycl::gpu_selector_v};
    auto* Q=sycl::malloc_device<bf16>(size_t(M)*QH*D,q);
    auto* K=sycl::malloc_device<bf16>(size_t(M)*KH*D,q);
    auto* V=sycl::malloc_device<bf16>(size_t(M)*KH*D,q);
    auto* O=sycl::malloc_device<bf16>(size_t(M)*QH*D,q);
    auto* cuq=sycl::malloc_device<int>(2,q); auto* cuk=sycl::malloc_device<int>(2,q);
    std::vector<bf16> hQ(size_t(M)*QH*D),hK(size_t(M)*KH*D),hV(hK.size()),hO(hQ.size());
    int hcu[2]={0,M};q.memcpy(cuq,hcu,sizeof(hcu));q.memcpy(cuk,hcu,sizeof(hcu));
    for(size_t i=0;i<hQ.size();++i)hQ[i]=bf16(float(int(i*13%31)-15)/32);
    for(size_t i=0;i<hK.size();++i){hK[i]=bf16(float(int(i*7%29)-14)/32);hV[i]=bf16(float(int(i*11%37)-18)/32);}
    q.memcpy(Q,hQ.data(),hQ.size()*sizeof(bf16));q.memcpy(K,hK.data(),hK.size()*sizeof(bf16));q.memcpy(V,hV.data(),hV.size()*sizeof(bf16));q.wait();
    grimoire_xe2_chunk_prefill_bf16(&q,Q,K,V,O,M,M,QH,KH,D,cuq,cuk,1/std::sqrt(float(D)),true);
    q.wait_and_throw();
    q.memcpy(hO.data(),O,hO.size()*sizeof(bf16)).wait();
    double se=0,sr=0,maxe=0;
    for(int t=0;t<M;t+=17)for(int h=0;h<QH;h+=3){
        int kh=h/(QH/KH);std::vector<float> score(t+1);float mx=-INFINITY;
        for(int s=0;s<=t;++s){float x=0;for(int d=0;d<D;++d)x+=float(hQ[(size_t(t)*QH+h)*D+d])*float(hK[(size_t(s)*KH+kh)*D+d]);score[s]=x/std::sqrt(float(D));mx=std::max(mx,score[s]);}
        float den=0;for(float x:score)den+=std::exp(x-mx);
        for(int d=0;d<D;d+=19){float ref=0;for(int s=0;s<=t;++s)ref+=std::exp(score[s]-mx)/den*float(hV[(size_t(s)*KH+kh)*D+d]);float got=float(hO[(size_t(t)*QH+h)*D+d]);double e=got-ref;se+=e*e;sr+=double(ref)*ref;maxe=std::max(maxe,std::abs(e));}
    }
    std::printf("rel_l2=%.6e max_abs=%.6e\n",std::sqrt(se/(sr+1e-30)),maxe);
    return std::sqrt(se/(sr+1e-30))<0.02?0:2;
}
