// Decode-path check: can GRIMOIRE's INT4 g128 GEMV match its MXFP4 GEMV?
// If yes, one int4 artifact serves BOTH BesTLA prefill and GRIMOIRE decode.
#include "kernels.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
using namespace b70;
typedef std::chrono::steady_clock clk;
struct Shape{int n,k;const char*name;};
static const Shape SH[]={{34816,5120,"ffn-gate-up"},{5120,17408,"ffn-down"},
                         {10240,5120,"la-qkv"},{248320,5120,"lm_head"}};
int main(){
  sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
  std::printf("device: %s   M=1 decode GEMV\n",
      q.get_device().get_info<sycl::info::device::name>().c_str());
  std::printf("%-13s %7s %7s | %10s %9s | %10s %9s | %s\n",
      "shape","N","K","MXFP4 ms","GB/s","INT4 ms","GB/s","ratio");
  std::mt19937 rng(9);
  for(const auto& z:SH){
    float* x=sycl::malloc_device<float>(z.k,q);
    float* y=sycl::malloc_device<float>(z.n,q);
    { std::vector<float> h(z.k); std::uniform_real_distribution<float> d(-1.f,1.f);
      for(auto&v:h)v=d(rng); q.memcpy(x,h.data(),h.size()*4).wait(); }
    auto run=[&](Fmt f)->double{
      QuantWeight w{}; w.fmt=f; w.N=z.n; w.K=z.k;
      const int grp = (f==Fmt::MXFP4)?32:128;
      w.row_bytes = int64_t(z.k)/2;
      w.row_scales = z.k/grp;
      const size_t pb=size_t(z.n)*w.row_bytes;
      const size_t sb=size_t(z.n)*w.row_scales*((f==Fmt::MXFP4)?1:sizeof(bf16_t));
      auto* pay=sycl::malloc_device<uint8_t>(pb,q);
      auto* sc =sycl::malloc_device<uint8_t>(sb,q);
      { std::vector<uint8_t> h(pb); for(auto&v:h)v=(uint8_t)(rng()&0xff);
        q.memcpy(pay,h.data(),pb).wait();
        std::vector<uint8_t> s(sb,(f==Fmt::MXFP4)?127:0);
        if(f!=Fmt::MXFP4){ std::vector<bf16_t> sb16(sb/sizeof(bf16_t));
          for(auto&v:sb16)v=f32_to_bf16(0.02f);
          q.memcpy(sc,sb16.data(),sb).wait(); }
        else q.memcpy(sc,s.data(),sb).wait(); }
      w.payload=pay; w.scales=sc; w.zeros=nullptr;
      double best=1e18;
      try{
        launch_gemv(q,w,x,y,{}); q.wait();
        for(int i=0;i<8;++i){ auto t=clk::now(); launch_gemv(q,w,x,y,{}); q.wait();
          double ms=std::chrono::duration<double,std::milli>(clk::now()-t).count();
          if(i&&ms<best)best=ms; }
      }catch(const std::exception&e){ std::printf("   [%s failed: %s]\n",
          f==Fmt::MXFP4?"MXFP4":"INT4",e.what()); best=-1; }
      sycl::free(pay,q); sycl::free(sc,q);
      return best;
    };
    double mx=run(Fmt::MXFP4), i4=run(Fmt::INT4);
    auto gbs=[&](double ms,int grp,size_t sw){ if(ms<=0)return 0.0;
      double b=double(z.n)*z.k*0.5 + double(z.n)*(z.k/grp)*sw; return b/ms*1e-6; };
    std::printf("%-13s %7d %7d | %10.4f %9.1f | %10.4f %9.1f | %s\n",
      z.name,z.n,z.k,mx,gbs(mx,32,1),i4,gbs(i4,128,2),
      (i4>0&&mx>0)?(i4<=mx*1.1?"INT4 OK":"INT4 SLOWER"):"n/a");
    sycl::free(x,q); sycl::free(y,q);
  }
  return 0;
}
