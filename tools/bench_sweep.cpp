// Policy sweep under PRODUCTION memory conditions (64 distinct weight matrices).
// All previous tuning used one cache-hot matrix, which is the wrong regime.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>
#include "xe2_grouped_raw_launcher.hpp"
using bf16 = cutlass::bfloat16_t;
#define POL(NAME,M_,N_,SM,SN) \
 class NAME : public MoE::xe_gemm_policy_base { public: \
  using WGTile=cute::Shape<cute::_##M_,cute::_##N_,cute::_32>; \
  using SGLayout=cute::Layout<cute::Shape<cute::_##SM,cute::_##SN,cute::_1>, \
    cute::Stride<cute::_##SN,cute::_1,cute::_0>>; \
  using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>; };
POL(q128x256,128,256,4,8)
POL(q128x128,128,128,4,4)
POL(q128x64 ,128,64 ,4,2)
POL(q64x128 ,64 ,128,2,4)
POL(q64x64  ,64 ,64 ,2,2)
POL(q256x128,256,128,8,4)
typedef std::chrono::steady_clock clk;
struct Shape { int n,k; const char* name; };
static bf16 *g_x, *g_y; static int g_M;
static std::vector<unsigned char*> g_W, g_S;
static std::vector<bf16*> g_SB;   // bf16 scales for the int4 path
template<class P,int G> static double sweep_i4(sycl::queue&q,const Shape&z,int L){
  double b=1e18;
  for(int r=0;r<3;++r){
    clk::time_point t0=clk::now();
    for(int i=0;i<L;++i)
      grimoire_moe_raw::launch_dense_int4<P,G>(q,g_x,g_W[i],g_SB[i],g_y,g_M,z.n,z.k);
    q.wait_and_throw();
    double ms=std::chrono::duration<double,std::milli>(clk::now()-t0).count()/L;
    if(r&&ms<b) b=ms;
  }
  return b;
}
template<class P> static double sweep(sycl::queue&q,const Shape&z,int L){
  double b=1e18;
  for(int r=0;r<3;++r){
    clk::time_point t0=clk::now();
    for(int i=0;i<L;++i)
      grimoire_moe_raw::launch_dense_mxfp4<P>(q,g_x,g_W[i],g_S[i],g_y,g_M,z.n,z.k);
    q.wait_and_throw();
    double ms=std::chrono::duration<double,std::milli>(clk::now()-t0).count()/L;
    if(r&&ms<b) b=ms;
  }
  return b;
}
int main(){
  sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
  printf("device: %s   (64 distinct weight matrices = production conditions)\n",
         q.get_device().get_info<sycl::info::device::name>().c_str());
  g_M=4096;
  // QKVZ fusion candidate (16384 = qkv 10240 + z 6144) plus the two parts it
  // would replace, so the fused shape can be compared against the sum.
  const Shape shapes[3]={{16384,5120,"QKVZ-fused"},{10240,5120,"dn-qkv"},
                         {6144,5120,"dn-z"}};
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> ux(-1.f,1.f);
  for(int si=0;si<3;++si){
    const Shape z=shapes[si];
    const size_t bw=size_t(z.n)*z.k/2, sw=size_t(z.n)*z.k/32;
    g_x=sycl::malloc_device<bf16>(size_t(g_M)*z.k,q);
    g_y=sycl::malloc_device<bf16>(size_t(g_M)*z.n,q);
    { std::vector<bf16> hx(size_t(g_M)*z.k);
      for(size_t i=0;i<hx.size();++i) hx[i]=bf16(ux(rng));
      q.memcpy(g_x,hx.data(),hx.size()*sizeof(bf16)).wait(); }
    std::vector<unsigned char> hw(bw), hs(sw,127);
    for(size_t i=0;i<hw.size();++i) hw[i]=(unsigned char)(rng()&0xff);
    g_W.clear(); g_S.clear(); g_SB.clear();
    int L=64;
    for(int i=0;i<64;++i){
      unsigned char* w=sycl::malloc_device<unsigned char>(bw,q);
      unsigned char* s=sycl::malloc_device<unsigned char>(sw,q);
      if(!w||!s){ L=i; break; }
      hw[i%hw.size()]^=0x5a;
      q.memcpy(w,hw.data(),bw).wait(); q.memcpy(s,hs.data(),sw).wait();
      g_W.push_back(w); g_S.push_back(s);
      bf16* sb=sycl::malloc_device<bf16>(size_t(z.n)*(z.k/32),q);   // sized for g32, the largest scale count
      if(sb){ std::vector<bf16> hb(size_t(z.n)*(z.k/32), bf16(0.02f));
              q.memcpy(sb,hb.data(),hb.size()*sizeof(bf16)).wait(); }
      g_SB.push_back(sb);
    }
    const double fl=2.0*g_M*z.n*z.k;
    printf("\n== %s  N=%d K=%d  (L=%d)\n",z.name,z.n,z.k,L);
    printf("   %-10s %10s %9s\n","policy","ms/layer","TFLOP/s");
    struct R{const char*n;double ms;} rs[8];
    rs[0]={"128x256",sweep<q128x256>(q,z,L)};
    rs[1]={"128x128",sweep<q128x128>(q,z,L)};
    rs[2]={"128x64" ,sweep<q128x64 >(q,z,L)};
    rs[3]={"64x128" ,sweep<q64x128 >(q,z,L)};
    rs[4]={"64x64"  ,sweep<q64x64  >(q,z,L)};
    rs[5]={"256x128",sweep<q256x128>(q,z,L)};
    rs[6]={"INT4 g128",sweep_i4<q128x256,128>(q,z,L)};
    rs[7]={"INT4 g32" ,sweep_i4<q128x256,32 >(q,z,L)};
    int best=0; for(int i=1;i<8;++i) if(rs[i].ms<rs[best].ms) best=i;
    for(int i=0;i<8;++i)
      printf("   %-10s %10.3f %9.1f %s\n",rs[i].n,rs[i].ms,fl/rs[i].ms*1e-9,
             i==best?"  <-- BEST":"");
    for(int i=0;i<L;++i){ sycl::free(g_W[i],q); sycl::free(g_S[i],q); if(g_SB[i]) sycl::free(g_SB[i],q); }
    sycl::free(g_x,q); sycl::free(g_y,q);
  }
  return 0;
}
