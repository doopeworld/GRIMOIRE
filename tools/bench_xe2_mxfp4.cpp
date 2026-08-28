#include "b70/weights.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void grimoire_xe2_grouped_mxfp4_bf16(
    sycl::queue*,const void*,const unsigned char*,const unsigned char*,void*,
    int,int,const int*,const int*,int,int*);
extern "C" void grimoire_xe2_dense_mxfp4_bf16(
    sycl::queue*,const void*,const unsigned char*,const unsigned char*,void*,int,int,int);

int main(int argc,char**argv){
 const int M=argc>1?std::atoi(argv[1]):4096;
 const int N=argc>2?std::atoi(argv[2]):8192;
 const int K=argc>3?std::atoi(argv[3]):2048;
 using bf16=sycl::ext::oneapi::bfloat16;
 sycl::queue q{sycl::gpu_selector_v,sycl::property_list{
   sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}};
 std::fprintf(stderr,"queue ok\n");
 std::vector<float>w(size_t(N)*K);for(size_t i=0;i<w.size();++i)w[i]=.02f*std::sin(float(i%10007)*.013f);
 auto pw=b70::quantize(w.data(),N,K,b70::Fmt::MXFP4);
 std::fprintf(stderr,"quant ok\n");
 std::vector<bf16>x(size_t(M)*K,bf16(.125f));
 auto*dx=sycl::malloc_device<bf16>(x.size(),q);auto*dw=sycl::malloc_device<uint8_t>(pw.payload.size(),q);
 auto*ds=sycl::malloc_device<uint8_t>(pw.scales_raw.size(),q);auto*dy=sycl::malloc_device<bf16>(size_t(M)*N,q);
 auto*dr=sycl::malloc_device<int>(1,q);auto*de=sycl::malloc_device<int>(1,q);auto*da=sycl::malloc_device<int>(1,q);
 int rows=M,expert=0;q.memcpy(dx,x.data(),x.size()*2);q.memcpy(dw,pw.payload.data(),pw.payload.size());
 q.memcpy(ds,pw.scales_raw.data(),pw.scales_raw.size());q.memcpy(dr,&rows,4);q.memcpy(de,&expert,4);q.memset(da,0,4).wait();
 std::fprintf(stderr,"upload ok\n");
 for(int i=0;i<2;++i){grimoire_xe2_dense_mxfp4_bf16(&q,dx,dw,ds,dy,M,N,K);q.wait();}
 uint64_t best=~0ull;for(int i=0;i<5;++i){auto s=std::chrono::steady_clock::now();
  grimoire_xe2_dense_mxfp4_bf16(&q,dx,dw,ds,dy,M,N,K);q.wait();
  auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-s).count();best=std::min(best,uint64_t(ns));}
 double ms=best*1e-6,tops=2.0*M*N*K/(ms*1e9);std::printf("Xe2 MXFP4 %dx%dx%d %.3f ms %.1f TOPS\n",M,N,K,ms,tops);
 std::vector<bf16> out(N);q.memcpy(out.data(),dy,N*2).wait();double se=0,sr=0;
 auto qv=pw.view();
 for(int n=0;n<N;n+=std::max(1,N/127)){double ref=0;for(int k=0;k<K;++k)ref+=.125*qv.at(n,k);
   double e=double(float(out[n]))-ref;se+=e*e;sr+=ref*ref;}
 double rel=std::sqrt(se/(sr+1e-30));std::printf("sampled rel_l2 %.3e\n",rel);
 return rel<0.02?0:2;
}
