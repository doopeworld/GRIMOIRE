#include "xe2_onednn_bridge.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main(){
  constexpr int M=1,N=10240,K=5120,GS=128;
  sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::in_order{}};
  using act=sycl::ext::oneapi::bfloat16;
  std::vector<act> ah(size_t(M)*K),sh(size_t(K/GS)*N);
  std::vector<uint32_t> bh(size_t(N)*(K/8));
  for(size_t i=0;i<ah.size();++i)ah[i]=act(float(int(i*17%31)-15)/32.0f);
  for(size_t i=0;i<sh.size();++i)sh[i]=act(0.00390625f);
  for(size_t i=0;i<bh.size();++i){uint32_t x=0;for(int j=0;j<8;++j)x|=uint32_t((i*5+j*3)%15)<<4*j;bh[i]=x;}
  auto* a=sycl::malloc_device<act>(ah.size(),q);
  auto* b=sycl::malloc_device<uint32_t>(bh.size(),q);
  auto* s=sycl::malloc_device<act>(sh.size(),q);
  auto* z=sycl::malloc_device<int8_t>(1,q);
  auto* o=sycl::malloc_device<act>(size_t(M)*N,q);
  const int8_t hz=8;
  q.memcpy(a,ah.data(),ah.size()*2);q.memcpy(b,bh.data(),bh.size()*4);
  q.memcpy(s,sh.data(),sh.size()*2);q.memcpy(z,&hz,1).wait();
  void* p=grimoire_onednn_w4a16_create(&q,M,N,K,GS,1);
  if(!p){std::puts("plan creation failed");return 2;}
  const size_t sb=grimoire_onednn_w4a16_scratch_size(p);
  auto* scratch=sycl::malloc_device<uint8_t>(sb,q);
  for(int i=0;i<15;++i){grimoire_onednn_w4a16_execute(p,a,b,s,z,o,scratch);q.wait();}
  std::vector<double> ts;
  for(int i=0;i<10;++i){auto t0=std::chrono::steady_clock::now();
    grimoire_onednn_w4a16_execute(p,a,b,s,z,o,scratch);q.wait();
    ts.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());}
  std::sort(ts.begin(),ts.end());double ms=0;for(int i=2;i<8;++i)ms+=ts[i]/6;
  std::vector<act> oh(N);q.memcpy(oh.data(),o,N*2).wait();
  double me=0,mr=0;
  for(int n=0;n<N;n+=257){double ref=0;for(int k=0;k<K;++k){
    uint32_t x=bh[size_t(n)*(K/8)+k/8];int w=int((x>>(4*(k&7)))&15)-8;
    ref+=float(ah[k])*w*float(sh[size_t(k/GS)*N+n]);}
    me=std::max(me,std::fabs(double(float(oh[n]))-ref));mr=std::max(mr,std::fabs(ref));}
  std::printf("raw oneDNN W4A16 %dx%dx%d: %.3f ms, %.1f TOPS, scratch %.1f MiB, rel %.3e\n",
    M,N,K,ms,2.0*double(M)*N*K/(ms*1e9),double(sb)/1048576.0,me/(mr+1e-30));
  grimoire_onednn_w4a16_destroy(p);sycl::free(a,q);sycl::free(b,q);sycl::free(s,q);
  sycl::free(z,q);sycl::free(o,q);sycl::free(scratch,q);return me/(mr+1e-30)>0.02;
}
