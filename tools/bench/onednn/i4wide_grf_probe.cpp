// Compare the production 256-GRF INT4 verify kernel with the same DPAS
// arithmetic compiled at the engine's default 128 GRF. No engine dispatch changes.
// FP64 reference, full output guards, GS64/128, partial second row blocks,
// and rotating cold weights. Run only through the guarded GPU0 launcher.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "int4_smallm.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace b70;

struct Plan { int ks,kc,tpt; };
static Plan plan(int N,int K,bool narrow) {
    int ks=1;
    if(narrow)ks=N/16>=1024||K>=12288?4:std::min(32,std::max(1,K/256));
    else while(ks<16&&(N/16)*ks<4096&&K/(ks*2)>=128)ks*=2;
    const int kc=((K+ks-1)/ks+127)/128*128;
    ks=(K+kc-1)/kc;
    int tpt=1;while(tpt*ks<8&&tpt*2*ks<=32)tpt*=2;
    return {ks,kc,tpt};
}
static float from_bf(uint16_t v) {
    uint32_t bits=uint32_t(v)<<16;float out;std::memcpy(&out,&bits,4);return out;
}
static sycl::event local128(sycl::queue& q,const QuantWeight& w,
                           const sycl_bf16* x,float* y,int M,bool narrow) {
    const auto p=plan(w.N,w.K,narrow);
    return w.row_scales*64==w.K
        ?int4_smallm_fdq_impl<2,64>(q,w,x,y,M,p.ks,p.kc,p.tpt,{})
        :int4_smallm_fdq_impl<2,128>(q,w,x,y,M,p.ks,p.kc,p.tpt,{});
}
static int run(sycl::queue& q,int N,int K,int gs) {
    const size_t pb=size_t(N)*K/2,groups=size_t(N)*(K/gs);
    const size_t bytes=pb+groups*3;
    const int copies=bytes<(1u<<20)?2:int(std::clamp<size_t>((512ull<<20)/bytes+1,2,12));
    std::mt19937 rng(1101+N+K+gs);
    std::vector<uint8_t> hp(pb),hz(groups);
    std::vector<uint16_t> hs(groups);
    for(auto& v:hp)v=uint8_t(rng());
    for(size_t i=0;i<groups;++i){
        hz[i]=i%67==0?255:uint8_t(rng()%16);
        hs[i]=sycl::bit_cast<uint16_t>(sycl_bf16(i<size_t(K/gs)?0.f:0.001f*float(1+rng()%64)));
    }
    std::vector<void*> allocated;
    auto alloc=[&](size_t n){void* p=sycl::aligned_alloc_device(64,n,q);
        if(!p)throw std::bad_alloc();allocated.push_back(p);return p;};
    std::vector<QuantWeight> weights;
    for(int c=0;c<copies;++c){
        auto* p=static_cast<uint8_t*>(alloc(pb));
        auto* s=static_cast<uint16_t*>(alloc(groups*2));
        auto* z=static_cast<uint8_t*>(alloc(groups));
        q.memcpy(p,hp.data(),pb);q.memcpy(s,hs.data(),groups*2);q.memcpy(z,hz.data(),groups);
        QuantWeight w;w.fmt=Fmt::INT4;w.N=N;w.K=K;w.payload=p;w.scales=s;w.zeros=z;
        w.row_bytes=K/2;w.row_scales=K/gs;weights.push_back(w);
    }
    int bad=0;
    for(int M:{9,14,16}){
        std::vector<sycl_bf16> hx(size_t(M)*K);
        for(int m=0;m<M;++m)for(int k=0;k<K;++k)
            hx[size_t(m)*K+k]=sycl_bf16(m==1?0.f:(float(rng()%2001)-1000.f)*0.001f);
        auto* dx=static_cast<sycl_bf16*>(alloc(hx.size()*2));
        auto* base=static_cast<float*>(alloc((size_t(M)*N+128)*4));
        float* dy=base+64;
        q.memcpy(dx,hx.data(),hx.size()*2).wait_and_throw();
        auto launch=[&](int variant,int c){
            if(variant==0)return launch_int4_smallm_wide(q,weights[c],dx,dy,M,{});
            return local128(q,weights[c],dx,dy,M,variant==2);
        };
        constexpr float sentinel=0x1.345678p+45f;
        double worst[3]={},micros[3]={};
        for(int v=0;v<3;++v){
            std::vector<float> out(size_t(M)*N+128,sentinel);
            q.memcpy(base,out.data(),out.size()*4).wait_and_throw();
            launch(v,0).wait_and_throw();q.memcpy(out.data(),base,out.size()*4).wait_and_throw();
            for(int i=0;i<64;++i)if(out[i]!=sentinel||out[out.size()-1-i]!=sentinel)++bad;
            for(size_t i=64;i<out.size()-64;++i)if(!std::isfinite(out[i]))++bad;
            for(int m=0;m<M;++m)for(int sample=0;sample<32;++sample){
                const int n=sample<4?sample:(sample*7919+13)%N;
                double ref=0,mag=0;
                for(int k=0;k<K;++k){
                    const int nib=(hp[size_t(n)*K/2+k/2]>>(4*(k&1)))&15;
                    const size_t si=size_t(n)*(K/gs)+k/gs;
                    const int value=hz[si]==255?(nib&8?nib-16:nib):nib-int(hz[si]);
                    const double product=double(value)*from_bf(hs[si])*float(hx[size_t(m)*K+k]);
                    ref+=product;mag+=std::fabs(product);
                }
                const double rel=std::fabs(double(out[64+size_t(m)*N+n])-ref)/(mag+1e-30);
                worst[v]=std::max(worst[v],rel);
                if(!(rel<3e-5)&&bad++<4)std::printf("mismatch v=%d M=%d n=%d rel=%g\n",v,M,n,rel);
            }
            for(int c=0;c<copies;++c)launch(v,c).wait_and_throw();
        }
        std::vector<double> times[3];
        const auto started=std::chrono::steady_clock::now();
        for(int round=0;round<48;++round){
            for(int turn=0;turn<3;++turn){
                const int v=(round+turn)%3;
                const auto t=std::chrono::steady_clock::now();
                // Use different weight allocations for adjacent variants;
                // sharing one allocation would give later variants a hot cache.
                launch(v,(3*round+turn)%copies).wait_and_throw();
                times[v].push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count());
            }
            if(round>=copies&&std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>2.)break;
        }
        for(int v=0;v<3;++v){std::sort(times[v].begin(),times[v].end());micros[v]=times[v][times[v].size()/2];}
        std::printf("GS=%d N=%d K=%d M=%d copies=%d 256=%.2fus 128same=%.2fus 128plan=%.2fus maxrel=%.2e/%.2e/%.2e bad=%d\n",
            gs,N,K,M,copies,micros[0],micros[1],micros[2],worst[0],worst[1],worst[2],bad);
        std::fflush(stdout);
        // Leave only the cold weight copies resident between row counts.
        q.wait_and_throw();sycl::free(dx,q);sycl::free(base,q);
        allocated.resize(allocated.size()-2);
    }
    q.wait_and_throw();for(void* p:allocated)sycl::free(p,q);
    return bad;
}
int main(int argc,char** argv){
    try{
        setenv("GRIMOIRE_I4_WIDE_FDQ","1",1);
        sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
        std::printf("device=%s default-GRF TU, production 256-GRF library\n",q.get_device().get_info<sycl::info::device::name>().c_str());
        const std::vector<std::pair<int,int>> shapes={{48,1024},{272,512},{34816,5120},
            {5120,17408},{10240,5120},{12288,5120},{6144,5120},{5120,6144},{248320,5120}};
        const int only=argc>1?std::atoi(argv[1]):-1;
        if(only<-1||only>=int(shapes.size()))throw std::runtime_error("invalid shape index");
        int bad=0,cases=0;
        for(int gs:{64,128})for(size_t s=0;s<shapes.size();++s){
            if(only>=0&&int(s)!=only)continue;
            bad+=run(q,shapes[s].first,shapes[s].second,gs);cases+=3;
        }
        std::printf("I4_WIDE_GRF_CASES=%d failures=%d %s\n",cases,bad,bad?"FAIL":"PASS");
        return bad?1:0;
    }catch(const std::exception& e){std::fprintf(stderr,"probe: %s\n",e.what());return 1;}
}
