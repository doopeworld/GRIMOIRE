#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
using bf16=sycl::ext::oneapi::bfloat16;
extern "C" int grimoire_xe2_dense_mxfp4_autotune(sycl::queue*,const void*,
 const unsigned char*,const unsigned char*,void*,int,int,int,int);
struct Shape{int n,k,freq;const char*name;};
static const Shape shapes[]={
 {34816,5120,64,"ffn-gate-up"},{5120,17408,64,"ffn-down"},
 {10240,5120,48,"dn-qkv"},{6144,5120,48,"dn-z"},
 {5120,6144,48,"dn-out"},{12288,5120,16,"fa-q"},
 {2048,5120,16,"fa-kv"},{5120,6144,16,"fa-o"}};
static double run(sycl::queue&q,int p,const Shape&z,std::vector<float>&ref,
 double&err,unsigned char*cold,size_t cold_bytes){constexpr int M=4096;
 auto*x=sycl::malloc_device<bf16>(size_t(M)*z.k,q);
 auto*w=sycl::malloc_device<unsigned char>(size_t(z.n)*z.k/2,q);
 auto*s=sycl::malloc_device<unsigned char>(size_t(z.n)*z.k/32,q);
 auto*y=sycl::malloc_device<bf16>(size_t(M)*z.n,q);
 q.fill(x,bf16(.125f),size_t(M)*z.k);q.memset(w,0x22,size_t(z.n)*z.k/2);
 q.memset(s,127,size_t(z.n)*z.k/32);q.wait();
 auto launch=[&]{grimoire_xe2_dense_mxfp4_autotune(&q,x,w,s,y,M,z.n,z.k,p);q.wait_and_throw();};
 launch();double best=INFINITY;
 for(int i=0;i<3;++i){q.fill(cold,(unsigned char)(i+1),cold_bytes).wait();
  auto a=std::chrono::steady_clock::now();launch();best=std::min(best,
   std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count());}
 size_t ns=std::min<size_t>(4096,size_t(M)*z.n);std::vector<bf16>o(ns);
 q.memcpy(o.data(),y,ns*sizeof(bf16)).wait();if(ref.empty()){ref.resize(ns);
  for(size_t i=0;i<ns;++i)ref[i]=float(o[i]);}err=0;
 for(size_t i=0;i<ns;++i)err=std::max(err,std::abs(double(float(o[i]))-ref[i]));
 sycl::free(x,q);sycl::free(w,q);sycl::free(s,q);sycl::free(y,q);return best;}
int main(){sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
 const char*pn[]={"32x64","32x128","32x256","64x64","64x128","64x256","128x64","128x128","128x256"};
 constexpr size_t cb=size_t(768)<<20;auto*cold=sycl::malloc_device<unsigned char>(cb,q);
 std::vector<float>refs[8];double totals[9]={};bool ok[9];std::fill(ok,ok+9,true);
 // Establish one correctness reference per exact production shape.
 for(int j=0;j<8;++j){double e;run(q,5,shapes[j],refs[j],e,cold,cb);}
 std::puts("policy,shape,N,K,freq,cold_ms,weighted_ms,max_abs,status");
 for(int p=0;p<9;++p)for(int j=0;j<8;++j){try{double e;double ms=run(q,p,shapes[j],refs[j],e,cold,cb);
   bool pass=e<=.01;ok[p]&=pass;totals[p]+=ms*shapes[j].freq;
   std::printf("%s,%s,%d,%d,%d,%.3f,%.3f,%.6g,%s\n",pn[p],shapes[j].name,
    shapes[j].n,shapes[j].k,shapes[j].freq,ms,ms*shapes[j].freq,e,pass?"PASS":"REJECT_NUMERIC");
  }catch(const std::exception&e){ok[p]=false;std::printf("%s,%s,%d,%d,%d,nan,nan,nan,REJECT:%s\n",
   pn[p],shapes[j].name,shapes[j].n,shapes[j].k,shapes[j].freq,e.what());}}
 int win=-1;double best=INFINITY;for(int p=0;p<9;++p){std::printf("TOTAL,%s,%.3f,%s\n",pn[p],totals[p],ok[p]?"PASS":"REJECT");
  if(ok[p]&&totals[p]<best){best=totals[p];win=p;}}
 if(win<0)return 2;std::printf("WINNER policy=%s weighted_total=%.3f ms candidates=72\n",pn[win],best);
 sycl::free(cold,q);return 0;}
