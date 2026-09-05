// Numerical diff: launch_flash_decode(+merge) vs launch_flash_decode_batched
// with tokens=1, on an identical synthetic KV cache, swept over context depth.
// GRIMOIRE_DECODE_BATCHED_ATTN produces correct text at ~6 tokens and garbage
// at 2000. This finds the depth where they diverge and by how much, instead of
// reading the kernel again.
#include "kernels.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
#include <cmath>
using namespace b70;

int main() {
    sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const int HEADS = 24, KVH = 4, HD = 256, CAP = 12288;
    const float scale = 1.0f / std::sqrt(float(HD));

    auto* kc = sycl::malloc_device<uint8_t>(size_t(KVH)*HD*CAP, q);
    auto* vc = sycl::malloc_device<uint8_t>(size_t(KVH)*CAP*HD, q);
    auto* qv = sycl::malloc_device<float>(size_t(HEADS)*HD, q);
    auto* o1 = sycl::malloc_device<float>(size_t(HEADS)*HD, q);
    auto* o2 = sycl::malloc_device<float>(size_t(HEADS)*HD, q);
    auto* part = sycl::malloc_device<float>(size_t(16)*HEADS*64*HD, q);
    auto* pm   = sycl::malloc_device<float>(size_t(16)*HEADS*64, q);
    auto* pl   = sycl::malloc_device<float>(size_t(16)*HEADS*64, q);
    auto* dsl  = sycl::malloc_device<int32_t>(1, q);

    { std::vector<uint8_t> h(size_t(KVH)*HD*CAP); uint32_t r=7u;
      for(size_t i=0;i<h.size();++i){ r=r*1664525u+1013904223u; h[i]=uint8_t((r>>24)&0x7F); }
      q.memcpy(kc,h.data(),h.size()).wait();
      for(size_t i=0;i<h.size();++i){ r=r*1664525u+1013904223u; h[i]=uint8_t((r>>24)&0x7F); }
      q.memcpy(vc,h.data(),h.size()).wait();
      std::vector<float> hq(size_t(HEADS)*HD);
      for(size_t i=0;i<hq.size();++i) hq[i]=0.02f*float((i%11)-5);
      q.memcpy(qv,hq.data(),hq.size()*4).wait(); }

    std::printf("%8s %12s %12s %10s\n","pos","maxdiff","rel","verdict");
    for (int pos : {5, 31, 63, 127, 255, 511, 1023, 2047, 4095}) {
        const int32_t seq = pos + 1;
        q.memcpy(dsl,&seq,4).wait();
        q.memset(o1,0,size_t(HEADS)*HD*4).wait();
        q.memset(o2,0,size_t(HEADS)*HD*4).wait();

        AttnParams ap{};
        ap.q=qv; ap.k_cache=kc; ap.v_cache=vc; ap.out=o1;
        ap.seq_len=pos+1; ap.seq_cap=CAP; ap.head_dim=HD;
        ap.num_heads=HEADS; ap.num_kv_heads=KVH; ap.softmax_scale=scale;
        ap.partials=part; ap.part_m=pm; ap.part_l=pl;
        ap.splits=8; ap.d_seq_len=dsl;
        launch_flash_decode(q,ap,{}); launch_flash_merge(q,ap,{}); q.wait();

        launch_flash_decode_batched(q,qv,kc,vc,o2,1,pos+1,HEADS,KVH,HD,CAP,
                                    scale,part,pm,pl,8,{});
        q.wait();

        std::vector<float> a(size_t(HEADS)*HD), b(size_t(HEADS)*HD);
        q.memcpy(a.data(),o1,a.size()*4).wait();
        q.memcpy(b.data(),o2,b.size()*4).wait();
        double md=0, mag=0;
        for(size_t i=0;i<a.size();++i){ md=std::max(md,double(std::fabs(a[i]-b[i])));
                                        mag=std::max(mag,double(std::fabs(a[i]))); }
        const double rel = mag>0 ? md/mag : 0;
        std::printf("%8d %12.6f %12.6f %10s\n", pos, md, rel,
                    rel < 1e-3 ? "match" : "DIVERGE");
    }
    return 0;
}
