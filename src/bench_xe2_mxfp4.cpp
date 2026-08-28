#include "b70/weights.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

extern "C" void grimoire_xe2_grouped_mxfp4_bf16(
    sycl::queue*,const void*,const unsigned char*,const unsigned char*,void*,
    int,int,const int*,const int*,int,int*);

int main(){
 constexpr int M=4096,N=8192,K=2048;
 using bf16=sycl::ext::oneapi::bfloat16;
 sycl::queue q{sycl::gpu_selector_v,sycl::property_list{
   sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}};
 std::vector<float>w(size_t(N)*K);for(size_t i=0;i<w.size();++i)w[i]=.02f*std::sin(float(i%10007)*.013f);
 auto pw=b70::quantize(w.data(),N,K,b70::Fmt::MXFP4);
 std::vector<bf16>x(size_t(M)*K,bf16(.125f));
 auto*dx=sycl::malloc_device<bf16>(x.size(),q);auto*dw=sycl::malloc_device<uint8_t>(pw.payload.size(),q);
 auto*ds=sycl::malloc_device<uint8_t>(pw.scales_raw.size(),q);auto*dy=sycl::malloc_device<bf16>(size_t(M)*N,q);
 auto*dr=sycl::malloc_device<int>(1,q);auto*de=sycl::malloc_device<int>(1,q);auto*da=sycl::malloc_device<int>(1,q);
 int rows=M,expert=0;q.memcpy(dx,x.data(),x.size()*2);q.memcpy(dw,pw.payload.data(),pw.payload.size());
 q.memcpy(ds,pw.scales_raw.data(),pw.scales_raw.size());q.memcpy(dr,&rows,4);q.memcpy(de,&expert,4);q.memset(da,0,4).wait();
 for(int i=0;i<2;++i){grimoire_xe2_grouped_mxfp4_bf16(&q,dx,dw,ds,dy,N,K,dr,de,1,da);q.wait();}
 uint64_t best=~0ull;for(int i=0;i<5;++i){auto s=std::chrono::steady_clock::now();
  grimoire_xe2_grouped_mxfp4_bf16(&q,dx,dw,ds,dy,N,K,dr,de,1,da);q.wait();
  auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-s).count();best=std::min(best,uint64_t(ns));}
 double ms=best*1e-6,tops=2.0*M*N*K/(ms*1e9);std::printf("Xe2 MXFP4 %dx%dx%d %.3f ms %.1f TOPS\n",M,N,K,ms,tops);
 return 0;
}
