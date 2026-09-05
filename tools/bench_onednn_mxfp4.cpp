#include "xe2_onednn_bridge.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using bf16=sycl::ext::oneapi::bfloat16;
static float e2m1(unsigned x){
  static constexpr float mag[8]={0,0.5f,1,1.5f,2,3,4,6};
  return (x&8)?-mag[x&7]:mag[x&7];
}
static float e8m0(unsigned x){
  if(x==255)return NAN;
  return x==0?std::ldexp(1.0f,-127):std::ldexp(1.0f,int(x)-127);
}

static void one(int N,int K,const char* nm);
int main(){
  one(34816,5120,"ffn-gate-up");
  one(10240,5120,"la-qkv");
  one(5120,17408,"ffn-down");
  return 0;
}
static void one(int N,int K,const char* nm){
  constexpr int M=1;
  sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::in_order{}};
  std::vector<bf16> ah(K);std::vector<unsigned char> bh(size_t(N)*K/2);
  std::vector<unsigned char> sh(size_t(N)*K/32); for(size_t i=0;i<sh.size();++i) sh[i]=(unsigned char)(119+(i%7));
  for(int k=0;k<K;++k)ah[k]=bf16(float(int(k*17%31)-15)/32.0f);
  // INCOMPRESSIBLE fill: the old pattern ((i*5)%16 | ((i*11)%16)<<4) repeats
  // and is hardware-compressed, which inflates GB/s. 2026-09-05.
  { uint32_t r=2463534242u;
    for(size_t i=0;i<bh.size();++i){ r^=r<<13; r^=r>>17; r^=r<<5; bh[i]=(unsigned char)(r>>24); } }
  auto*a=sycl::malloc_device<bf16>(ah.size(),q);
  auto*b=sycl::malloc_device<unsigned char>(bh.size(),q);
  auto*s=sycl::malloc_device<unsigned char>(sh.size(),q);
  auto*o=sycl::malloc_device<bf16>(N,q);
  q.memcpy(a,ah.data(),ah.size()*2);q.memcpy(b,bh.data(),bh.size());
  q.memcpy(s,sh.data(),sh.size()).wait();
  void*p=grimoire_onednn_mxfp4_w4a16_create(&q,M,N,K);
  if(!p){std::printf("%-13s plan creation FAILED\n",nm);return;}
  size_t sb=grimoire_onednn_mxfp4_w4a16_scratch_size(p);
  auto*scratch=sb?sycl::malloc_device<unsigned char>(sb,q):nullptr;
  for(int i=0;i<10;++i){grimoire_onednn_mxfp4_w4a16_execute(p,a,b,s,o,scratch);q.wait();}
  std::vector<double>ts;
  for(int i=0;i<20;++i){auto t0=std::chrono::steady_clock::now();
    grimoire_onednn_mxfp4_w4a16_execute(p,a,b,s,o,scratch);q.wait();
    ts.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count());}
  std::sort(ts.begin(),ts.end());double us=0;for(int i=5;i<15;++i)us+=ts[i]/10;
  std::vector<bf16>oh(N);q.memcpy(oh.data(),o,N*2).wait();
  double me=0,mr=0;
  for(int n=0;n<N;n+=257){double ref=0;for(int k=0;k<K;++k){
    unsigned z=bh[size_t(n)*K/2+k/2];unsigned q4=(z>>((k&1)*4))&15;
    ref+=float(ah[k])*e2m1(q4)*e8m0(sh[size_t(n)*K/32+k/32]);}
    me=std::max(me,std::abs(double(float(oh[n]))-ref));mr=std::max(mr,std::abs(ref));}
  std::printf("%-13s %6dx%-6d %8.1f us %8.1f GB/s  scratch %.1f MiB  err %.2e\n",
    nm,N,K,us,(double(bh.size()+sh.size())/1e9)/(us*1e-6),
    double(sb)/(1<<20),me/(mr+1e-30));
}
