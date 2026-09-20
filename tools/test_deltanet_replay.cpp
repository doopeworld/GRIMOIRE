// Execute the production recurrence and compare compact replay against
// full state snapshots at EVERY accepted prefix. Not a host mock.
#include "kernels.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
using namespace b70;

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    std::printf("Replay device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int failures = 0, prefixes = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { if (failures < 12) std::printf("FAIL %s\n", what); ++failures; }
    };
    for (auto shape : {std::array<int,4>{2,1,64,64}, {6,2,128,128},
                       {48,16,128,128}, {4,1,256,32}, {3,1,17,32}}) {
        const auto [H, HK, KD, VD] = shape;
        const size_t ns = size_t(H)*KD*VD, nr = size_t(H)*(KD+1+VD);
        const size_t offset = 13, stride = nr+29;
        std::vector<float*> allocated;
        auto alloc = [&](size_t n) {
            float* p = sycl::malloc_shared<float>(n, q);
            if (!p) throw std::bad_alloc();
            allocated.push_back(p); return p;
        };
        float *s=alloc(ns), *ref=alloc(ns), *base=alloc(ns), *restored=alloc(ns);
        float *log=alloc(offset+16*stride), *qs=alloc(size_t(HK)*KD);
        float *ks=alloc(size_t(HK)*KD), *vs=alloc(size_t(H)*VD);
        float *a=alloc(H), *b=alloc(H), *out=alloc(size_t(H)*VD), *outref=alloc(size_t(H)*VD);
        std::mt19937 rng(704+KD+H);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        auto inputs = [&] {
            for (int h=0; h<HK; ++h) {
                float nq=0, nk=0;
                for (int k=0; k<KD; ++k) {
                    const int i=h*KD+k; qs[i]=dist(rng); ks[i]=dist(rng);
                    nq+=qs[i]*qs[i]; nk+=ks[i]*ks[i];
                }
                for (int k=0; k<KD; ++k) { qs[h*KD+k]/=std::sqrt(nq); ks[h*KD+k]/=std::sqrt(nk); }
            }
            for (int h=0; h<H; ++h) { a[h]=0.7f+0.25f*(dist(rng)+0.5f); b[h]=0.2f+0.6f*(dist(rng)+0.5f); }
            for (size_t i=0; i<size_t(H)*VD; ++i) vs[i]=dist(rng);
        };
        DeltaNetParams p{qs,ks,vs,a,b,s,out,H,KD,VD,HK};
        for (size_t i=0; i<ns; ++i) s[i]=dist(rng);
        for (int t=0; t<41; ++t) { inputs(); launch_deltanet_step(q,p).wait_and_throw(); }
        std::copy_n(s,ns,base);
        auto equal = [&](const float* x, const float* y, size_t n) {
            return std::all_of(x,x+n,[](float v){return std::isfinite(v);}) &&
                   std::memcmp(x,y,n*sizeof(float))==0;
        };
        for (int width : {1,2,4,8,16}) {
            std::fill_n(log,offset+16*stride,-9876.0f);
            std::copy_n(base,ns,s); std::copy_n(base,ns,ref);
            std::vector<std::vector<float>> snapshots{std::vector<float>(base,base+ns)};
            for (int t=0; t<width; ++t) {
                inputs(); p.state=s; p.out=out; p.replay=log+offset+size_t(t)*stride;
                launch_deltanet_step(q,p).wait_and_throw();
                auto oracle=p; oracle.state=ref; oracle.out=outref; oracle.replay=nullptr;
                launch_deltanet_step(q,oracle).wait_and_throw();
                check(equal(s,ref,ns), "recording changed recurrence");
                check(equal(out,outref,size_t(H)*VD), "recording changed output");
                snapshots.emplace_back(ref,ref+ns);
            }
            const std::vector<float> logs(log,log+offset+16*stride);
            for (int accepted=0; accepted<=width; ++accepted) {
                ++prefixes;
                std::copy(logs.begin(),logs.end(),log);
                // A replay that reads even one rejected update must fail.
                for (int t=accepted; t<width; ++t)
                    std::fill_n(log+offset+size_t(t)*stride,nr,std::numeric_limits<float>::quiet_NaN());
                std::fill_n(restored,ns,12345.0f);
                DeltaNetReplayParams rp{base,log+offset,restored,H,KD,VD,accepted,stride};
                launch_deltanet_replay(q,rp).wait_and_throw();
                check(equal(restored,snapshots[accepted].data(),ns), "accepted prefix differs from snapshot");
                // In-place replay is used when the destination already holds
                // the committed checkpoint; cover that alias explicitly.
                std::copy_n(base,ns,s); rp.checkpoint=s; rp.state=s;
                launch_deltanet_replay(q,rp).wait_and_throw();
                check(equal(s,restored,ns), "in-place replay differs");
                inputs(); std::copy(snapshots[accepted].begin(),snapshots[accepted].end(),ref);
                p.replay=nullptr; p.state=restored; p.out=out;
                launch_deltanet_step(q,p).wait_and_throw();
                p.state=ref; p.out=outref;
                launch_deltanet_step(q,p).wait_and_throw();
                check(equal(restored,ref,ns) && equal(out,outref,size_t(H)*VD), "decode after rollback differs");
            }
            for (size_t i=0; i<offset; ++i) check(logs[i]==-9876.0f,"leading guard overwritten");
            for (int t=0; t<16; ++t)
                for (size_t i=nr; i<stride; ++i)
                    check(logs[offset+size_t(t)*stride+i]==-9876.0f,"layer stride guard overwritten");
        }
        for (int bad : {-1,17}) {
            bool refused=false;
            try { launch_deltanet_replay(q,{base,log,restored,H,KD,VD,bad,stride}); }
            catch (const std::invalid_argument&) { refused=true; }
            check(refused,"invalid prefix accepted");
        }
        bool refused=false;
        try { launch_deltanet_replay(q,{base,log,restored,H,KD,VD,1,nr-1}); }
        catch (const std::invalid_argument&) { refused=true; }
        check(refused,"invalid stride accepted");
        q.wait_and_throw();
        for (float* ptr : allocated) sycl::free(ptr,q);
    }
    std::printf("DeltaNet replay: %d prefixes, %d failures\n", prefixes, failures);
    return failures ? 1 : 0;
}
