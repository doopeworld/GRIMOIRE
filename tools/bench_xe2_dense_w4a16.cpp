#include "xe2_grouped_bridge.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
using bf16=sycl::ext::oneapi::bfloat16;
int main(){
  constexpr int M=4096,N=8192,K=2048,GS=128;
  sycl::queue q{sycl::gpu_selector_v,
    sycl::property_list{sycl::property::queue::enable_profiling{}}};
  std::vector<bf16> ah(size_t(M)*K),sh(size_t(N)*(K/GS));
  std::vector<uint8_t> bh(size_t(N)*K/2);
  for(size_t i=0;i<ah.size();++i)ah[i]=bf16(float(int(i*17%31)-15)/32.0f);
  for(size_t i=0;i<sh.size();++i)sh[i]=bf16(0.00390625f);
  for(size_t i=0;i<bh.size();++i){int lo=int(i*5%15)-7,hi=int(i*11%15)-7;
    bh[i]=uint8_t((uint8_t(hi)&15)<<4)|(uint8_t(lo)&15);}
  auto* a=sycl::malloc_device<bf16>(ah.size(),q);
  auto* b=sycl::malloc_device<uint8_t>(bh.size(),q);
  auto* s=sycl::malloc_device<bf16>(sh.size(),q);
  auto* d=sycl::malloc_device<bf16>(size_t(M)*N,q);
  auto* rows=sycl::malloc_shared<int>(1,q);auto* atomic=sycl::malloc_device<int>(1,q);
  rows[0]=M;q.memcpy(a,ah.data(),ah.size()*sizeof(bf16));q.memcpy(b,bh.data(),bh.size());
  q.memcpy(s,sh.data(),sh.size()*sizeof(bf16));q.memset(atomic,0,sizeof(int)).wait();
  grimoire_xe2_grouped_w4a16(&q,a,b,s,d,N,K,rows,nullptr,1,GS,atomic);q.wait();
  q.memset(atomic,0,sizeof(int)).wait();
  const auto start=std::chrono::steady_clock::now();
  grimoire_xe2_grouped_w4a16(&q,a,b,s,d,N,K,rows,nullptr,1,GS,atomic);q.wait();
  const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  std::vector<bf16> out(N);q.memcpy(out.data(),d,N*sizeof(bf16)).wait();
  double maxe=0,maxr=0;
  for(int n=0;n<N;n+=257){double ref=0;for(int k=0;k<K;++k){uint8_t z=bh[size_t(n)*K/2+k/2];
    int w=(z>>((k&1)*4))&15;if(w&8)w-=16;ref+=float(ah[k])*w*float(sh[size_t(n)*(K/GS)+k/GS]);}
    maxe=std::max(maxe,std::fabs(double(float(out[n]))-ref));maxr=std::max(maxr,std::fabs(ref));}
  std::printf("raw Xe2 W4A16 dense %dx%dx%d: %.3f ms, %.1f TOPS, sampled rel-max %.3e\n",
    M,N,K,ms,2.0*double(M)*N*K/(ms*1e9),maxe/(maxr+1e-30));
  sycl::free(a,q);sycl::free(b,q);sycl::free(s,q);sycl::free(d,q);sycl::free(rows,q);sycl::free(atomic,q);
  return maxe/(maxr+1e-30)>0.02;
}
