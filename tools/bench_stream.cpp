// Does the FFN GEMM slow down because 64 DISTINCT weight matrices stream through
// (inherent), or because of pipeline overhead (recoverable)?
//   pass A: one weight matrix, reused  -> cache-hot, the optimistic microbench
//   pass B: 64 distinct matrices, cycled -> production memory conditions
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>
#include "xe2_grouped_raw_launcher.hpp"
using bf16 = cutlass::bfloat16_t;
class bp128x256 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_128,cute::_256,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_4,cute::_8,cute::_1>,
   cute::Stride<cute::_8,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};
typedef std::chrono::steady_clock clk;
struct Shape { int n, k; const char* name; int layers; };
int main(){
  sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
  printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  const int M=4096;
  const Shape shapes[2] = { {34816,5120,"ffn-gate-up",64}, {5120,17408,"ffn-down",64} };
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> ux(-1.f,1.f);
  printf("%-13s | %11s | %11s | %s\n","shape","1 matrix","64 matrices","delta/layer");
  for(int si=0;si<2;++si){
    const Shape z=shapes[si];
    const size_t bw=size_t(z.n)*z.k/2, sw=size_t(z.n)*z.k/32;
    bf16* x=sycl::malloc_device<bf16>(size_t(M)*z.k,q);
    bf16* y=sycl::malloc_device<bf16>(size_t(M)*z.n,q);
    { std::vector<bf16> hx(size_t(M)*z.k);
      for(size_t i=0;i<hx.size();++i) hx[i]=bf16(ux(rng));
      q.memcpy(x,hx.data(),hx.size()*sizeof(bf16)).wait(); }
    // how many distinct weight matrices fit?
    int L=z.layers;
    std::vector<unsigned char*> W; std::vector<unsigned char*> S;
    std::vector<unsigned char> hw(bw), hs(sw,127);
    for(size_t i=0;i<hw.size();++i) hw[i]=(unsigned char)(rng()&0xff);
    for(int i=0;i<L;++i){
      unsigned char* w=sycl::malloc_device<unsigned char>(bw,q);
      unsigned char* s=sycl::malloc_device<unsigned char>(sw,q);
      if(!w||!s){ L=i; break; }
      hw[i%hw.size()]^=0x5a;                    // make each matrix distinct
      q.memcpy(w,hw.data(),bw).wait(); q.memcpy(s,hs.data(),sw).wait();
      W.push_back(w); S.push_back(s);
    }
    // pass A: one matrix reused
    double a=1e18;
    for(int r=0;r<6;++r){ clk::time_point t0=clk::now();
      grimoire_moe_raw::launch_dense_mxfp4<bp128x256>(q,x,W[0],S[0],y,M,z.n,z.k);
      q.wait_and_throw();
      double ms=std::chrono::duration<double,std::milli>(clk::now()-t0).count();
      if(r&&ms<a) a=ms; }
    // pass B: cycle all L distinct matrices back to back, time the whole sweep
    double b=1e18;
    for(int r=0;r<3;++r){ clk::time_point t0=clk::now();
      for(int i=0;i<L;++i)
        grimoire_moe_raw::launch_dense_mxfp4<bp128x256>(q,x,W[i],S[i],y,M,z.n,z.k);
      q.wait_and_throw();
      double ms=std::chrono::duration<double,std::milli>(clk::now()-t0).count()/L;
      if(r&&ms<b) b=ms; }
    printf("%-13s | %8.3f ms | %8.3f ms | %+7.3f ms  (L=%d)\n",z.name,a,b,b-a,L);
    for(int i=0;i<L;++i){ sycl::free(W[i],q); sycl::free(S[i],q); }
    sycl::free(x,q); sycl::free(y,q);
  }
  return 0;
}
