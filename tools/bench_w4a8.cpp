// GRIMOIRE's OWN W4A8 path (int4 weights x int8 activations, int XMX) on Qwen's FFN
// shapes.  This is vLLM's recipe implemented in pure C++/SYCL and already present in
// gemm_xmx.cpp -- but nothing in the engine calls it.  Target: BesTLA's 148.5 TFLOP/s.
#include "kernels.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
using namespace b70;
typedef std::chrono::steady_clock clk;

struct Shape{int n,k;const char*name;double grim_ms;};
static const Shape SH[]={{34816,5120,"ffn-gate-up",14.63},{5120,17408,"ffn-down",6.98}};
int main(){
  sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
  std::printf("device: %s   M=4096  W4A8 (int4 weights x int8 activations)\n",
      q.get_device().get_info<sycl::info::device::name>().c_str());
  const int M=4096;
  std::mt19937 rng(17);
  std::printf("%-13s %7s %7s | %10s %9s | %9s %s\n",
      "shape","N","K","W4A8 ms","TFLOP/s","grim ms","vs grimoire");
  for(const auto& z:SH){
    float* x=sycl::malloc_device<float>(size_t(M)*z.k,q);
    int8_t* xq=sycl::malloc_device<int8_t>(size_t(M)*z.k,q);
    float* xs=sycl::malloc_device<float>(M,q);
    float* y=sycl::malloc_device<float>(size_t(M)*z.n,q);
    { std::vector<float> h(size_t(M)*z.k); std::uniform_real_distribution<float> d(-1.f,1.f);
      for(auto&v:h)v=d(rng); q.memcpy(x,h.data(),h.size()*4).wait(); }
    QuantWeight w{}; w.fmt=Fmt::INT4; w.N=z.n; w.K=z.k;
    w.row_bytes=int64_t(z.k)/2; w.row_scales=z.k/128;
    const size_t pb=size_t(z.n)*w.row_bytes, sb=size_t(z.n)*w.row_scales;
    auto* pay=sycl::malloc_device<uint8_t>(pb,q);
    auto* sc =sycl::malloc_device<bf16_t>(sb,q);
    { std::vector<uint8_t> h(pb); for(auto&v:h)v=(uint8_t)(rng()&0xff);
      q.memcpy(pay,h.data(),pb).wait();
      std::vector<bf16_t> s(sb); for(auto&v:s)v=f32_to_bf16(0.02f);
      q.memcpy(sc,s.data(),sb*sizeof(bf16_t)).wait(); }
    w.payload=pay; w.scales=sc; w.zeros=nullptr;
    double best=1e18;
    try{
      launch_quantize_rows_int8(q,x,xq,xs,M,z.k); q.wait();
      launch_gemm_xmx_int(q,w,xq,xs,y,M); q.wait();
      for(int i=0;i<5;++i){
        auto t=clk::now();
        launch_quantize_rows_int8(q,x,xq,xs,M,z.k);
        launch_gemm_xmx_int(q,w,xq,xs,y,M);
        q.wait();
        double ms=std::chrono::duration<double,std::milli>(clk::now()-t).count();
        if(i&&ms<best)best=ms;
      }
      const double fl=2.0*M*z.n*z.k;
      std::printf("%-13s %7d %7d | %10.3f %9.1f | %9.3f %.2fx\n",
        z.name,z.n,z.k,best,fl/best*1e-9,z.grim_ms,z.grim_ms/best);
    }catch(const std::exception& e){
      std::printf("%-13s %7d %7d | FAILED: %s\n",z.name,z.n,z.k,e.what());
    }
    sycl::free(x,q);sycl::free(xq,q);sycl::free(xs,q);sycl::free(y,q);
    sycl::free(pay,q);sycl::free(sc,q);
  }
  return 0;
}
