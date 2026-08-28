#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>

extern "C" void grimoire_xe2_chunk_gdn_raw_bf16(
    sycl::queue*,void*,const void*,const void*,const void*,void*,void*,void*,
    const float*,float*,const float*,const void*,float*,int,const int*,
    const int*,const bool*,int,int,int,int,int);

int main(){
  constexpr int M=4096,KH=16,KD=128,VH=32,VD=128;
  using bf16=sycl::ext::oneapi::bfloat16;
  sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{},
    sycl::property::queue::enable_profiling{}}};
  auto bd=[&](size_t n){return sycl::malloc_device<bf16>(n,q);};
  auto fd=[&](size_t n){return sycl::malloc_device<float>(n,q);};
  auto *qq=bd(size_t(M)*KH*KD),*kk=bd(size_t(M)*KH*KD);
  auto *vv=bd(size_t(M)*VH*VD),*oo=bd(size_t(M)*VH*VD),*db=bd(VH);
  const size_t pitch=M+63;
  auto *aa=bd(size_t(VH)*pitch*64),*ww=bd(size_t(VH)*pitch*KD),
       *uu=bd(size_t(VH)*pitch*VD);
  auto *beta=fd(size_t(VH)*M),*gate=fd(size_t(VH)*M),*alog=fd(VH),
       *state=fd(size_t(VH)*VD*KD);
  auto *cu=sycl::malloc_device<int>(2,q),*ci=sycl::malloc_device<int>(1,q);
  auto *hs=sycl::malloc_device<bool>(1,q);int hcu[2]={0,M},z=0;bool no=false;
  q.memset(qq,0,size_t(M)*KH*KD*2);q.memset(kk,0,size_t(M)*KH*KD*2);
  q.memset(vv,0,size_t(M)*VH*VD*2);q.memset(beta,0,size_t(VH)*M*4);
  q.memset(gate,0,size_t(VH)*M*4);q.memset(alog,0,VH*4);q.memset(db,0,VH*2);
  q.memset(state,0,size_t(VH)*VD*KD*4);q.memcpy(cu,hcu,8);
  q.memcpy(ci,&z,4);q.memcpy(hs,&no,1);q.wait();
  auto run=[&]{grimoire_xe2_chunk_gdn_raw_bf16(&q,oo,qq,kk,vv,aa,ww,uu,
    beta,gate,alog,db,state,VD*KD,cu,ci,hs,M,KH,KD,VH,VD);q.wait();};
  run();double best=1e9;
  for(int i=0;i<3;++i){auto t=std::chrono::steady_clock::now();run();best=std::min(best,
    std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
  std::printf("GDN raw M=%d %.3f ms\n",M,best);
}
