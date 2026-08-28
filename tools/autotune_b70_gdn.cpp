#include <sycl/sycl.hpp>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using bf16=sycl::ext::oneapi::bfloat16;
using Fn=void(*)(sycl::queue*,void*,const void*,const void*,const void*,void*,
 void*,void*,const float*,float*,const float*,const void*,float*,int,
 const int*,const int*,const bool*,int,int,int,int,int);

int main(){
 constexpr int M=4096,KH=16,KD=128,VH=32,VD=128;
 const char*names[]={"d1_s2x1","d1_s2x2","d1_s4x1","d1_s4x2",
                     "d2_s2x1","d2_s2x2","d2_s4x1","d2_s4x2"};
 sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
 auto bd=[&](size_t n){return sycl::malloc_device<bf16>(n,q);};
 auto fd=[&](size_t n){return sycl::malloc_device<float>(n,q);};
 auto *qq=bd(size_t(M)*KH*KD),*kk=bd(size_t(M)*KH*KD);
 auto *vv=bd(size_t(M)*VH*VD),*oo=bd(size_t(M)*VH*VD),*db=bd(VH);
 const size_t pitch=M+63;auto*aa=bd(size_t(VH)*pitch*64);
 auto*ww=bd(size_t(VH)*pitch*KD);auto*uu=bd(size_t(VH)*pitch*VD);
 auto*beta=fd(size_t(VH)*M);auto*gate=fd(size_t(VH)*M);
 auto*alog=fd(VH);auto*state=fd(size_t(VH)*VD*KD);
 auto*cu=sycl::malloc_device<int>(2,q);auto*ci=sycl::malloc_device<int>(1,q);
 auto*hs=sycl::malloc_device<bool>(1,q);int hcu[2]={0,M},z=0;bool no=false;
 q.fill(qq,bf16(.01f),size_t(M)*KH*KD);q.fill(kk,bf16(.01f),size_t(M)*KH*KD);
 q.fill(vv,bf16(.02f),size_t(M)*VH*VD);q.fill(beta,.5f,size_t(VH)*M);
 q.fill(alog,-2.f,VH);q.fill(db,bf16(-.2f),VH);q.memcpy(cu,hcu,8);
 q.memcpy(ci,&z,4);q.memcpy(hs,&no,1);q.wait();
 std::vector<float>ref;double winner=INFINITY;int wi=-1;
 std::puts("config,time_ms,max_abs,status");
 // The established coherent 1-way 4x2 kernel is evaluated first as control.
 const int order[]={3,0,1,2,4,5,6,7};
 for(int oi=0;oi<8;++oi){int i=order[oi];char path[128];
  std::snprintf(path,sizeof path,"/tmp/libgdn_%s.so",names[i]);void*h=dlopen(path,RTLD_NOW|RTLD_LOCAL);
  if(!h){std::printf("%s,nan,nan,REJECT_LOAD\n",names[i]);continue;}
  auto fn=reinterpret_cast<Fn>(dlsym(h,"grimoire_xe2_chunk_gdn_raw_bf16"));
  auto run=[&]{q.fill(gate,-1.f,size_t(VH)*M);q.memset(state,0,size_t(VH)*VD*KD*4);
   fn(&q,oo,qq,kk,vv,aa,ww,uu,beta,gate,alog,db,state,VD*KD,cu,ci,hs,
      M,KH,KD,VH,VD);q.wait_and_throw();};
  try{run();double best=INFINITY;for(int r=0;r<3;++r){auto t=std::chrono::steady_clock::now();
    run();best=std::min(best,std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-t).count());}
    std::vector<bf16>out(8192);q.memcpy(out.data(),oo,out.size()*2).wait();
    if(ref.empty()){ref.resize(out.size());for(size_t x=0;x<out.size();++x)ref[x]=float(out[x]);}
    double err=0;for(size_t x=0;x<out.size();++x)err=std::max(err,
      std::abs(double(float(out[x]))-ref[x]));bool ok=err<=.01;
    std::printf("%s,%.3f,%.6g,%s\n",names[i],best,err,ok?"PASS":"REJECT_NUMERIC");
    if(ok&&best<winner){winner=best;wi=i;}
  }catch(const std::exception&e){std::printf("%s,nan,nan,REJECT:%s\n",names[i],e.what());}
  dlclose(h);
 }
 if(wi<0){std::puts("NO VALID CONFIGURATION");return 2;}
 std::printf("WINNER config=%s time=%.3f ms candidates=8\n",names[wi],winner);return 0;
}
