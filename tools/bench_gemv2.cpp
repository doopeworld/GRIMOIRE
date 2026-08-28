// Isolate: is the decode GEMV limited by DRAM bandwidth or by dequant ALU work?
// Same shape, three formats.  BF16 does no dequant; MXFP4 unpacks a nibble and
// applies an E8M0 scale per element.  If BF16 reaches far higher GB/s, the limiter
// is dequant, and memory-parallelism tuning cannot help.
#include "kernels.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
using namespace b70;
typedef std::chrono::steady_clock clk;
struct S{int n,k;const char*name;};
static const S SH[]={{34816,5120,"ffn-gate-up"},{10240,5120,"la-qkv"},{5120,17408,"ffn-down"}};
static double bench(sycl::queue&q,Fmt f,int N,int K,double& gbs){
  QuantWeight w{}; w.fmt=f; w.N=N; w.K=K; w.row_bytes=bytes_per_row(f,K);
  const int grp=(f==Fmt::MXFP4)?kMXBlock:(f==Fmt::INT4?kInt4Group:0);
  w.row_scales=grp?K/grp:0;
  const size_t pb=size_t(N)*w.row_bytes;
  size_t sb=0;
  if(f==Fmt::MXFP4) sb=size_t(N)*w.row_scales;                 // uint8 E8M0
  else if(f==Fmt::INT4) sb=size_t(N)*w.row_scales*sizeof(bf16_t);
  auto* pay=sycl::malloc_device<uint8_t>(pb,q);
  uint8_t* sc=sb?sycl::malloc_device<uint8_t>(sb,q):nullptr;
  auto* x=sycl::malloc_device<float>(K,q);
  auto* y=sycl::malloc_device<float>(N,q);
  if(!pay||!x||!y||(sb&&!sc)){ std::printf("   alloc failed\n"); return -1; }
  { std::vector<uint8_t> h(pb,0x44); q.memcpy(pay,h.data(),pb).wait();
    if(sb){ std::vector<uint8_t> s(sb,127); q.memcpy(sc,s.data(),sb).wait(); }
    std::vector<float> hx(K,0.01f); q.memcpy(x,hx.data(),K*4).wait(); }
  w.payload=pay; w.scales=sc; w.zeros=nullptr;
  double best=1e18;
  try{
    launch_gemv(q,w,x,y,{}); q.wait();
    for(int i=0;i<10;++i){ auto t=clk::now(); launch_gemv(q,w,x,y,{}); q.wait();
      double ms=std::chrono::duration<double,std::milli>(clk::now()-t).count();
      if(i&&ms<best)best=ms; }
  }catch(const std::exception&e){ std::printf("   %s\n",e.what()); best=-1; }
  gbs = best>0 ? double(pb+sb)/best*1e-6 : 0.0;
  sycl::free(pay,q); if(sc) sycl::free(sc,q); sycl::free(x,q); sycl::free(y,q);
  return best;
}
int main(){
  sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
  std::printf("device: %s   roofline 602 GB/s\n",
     q.get_device().get_info<sycl::info::device::name>().c_str());
  std::printf("%-13s %-8s %10s %10s %8s\n","shape","fmt","ms","GB/s","%roof");
  for(const auto&z:SH){
    for(auto f:{Fmt::MXFP4}){
      double g=0, ms=bench(q,f,z.n,z.k,g);
      const char* fn = f==Fmt::MXFP4?"MXFP4":(f==Fmt::INT4?"INT4":"BF16");
      if(ms>0) std::printf("%-13s %-8s %10.4f %10.1f %7.0f%%\n",z.name,fn,ms,g,100*g/602.0);
    }
  }
  return 0;
}
