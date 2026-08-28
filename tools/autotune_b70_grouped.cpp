#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

extern "C" int grimoire_xe2_grouped_mxfp4_autotune(
 sycl::queue*,const void*,const unsigned char*,const unsigned char*,void*,
 int,int,const int*,int,int*,int,int);

using bf16=sycl::ext::oneapi::bfloat16;
struct Result{double gu=INFINITY,dn=INFINITY,err=INFINITY;bool ok=false;};

static double run_shape(sycl::queue&q,int policy,int mode,int M,int N,int K,
                        std::vector<float>*reference,double&maxerr,
                        unsigned char* cold,size_t cold_bytes){
 constexpr int E=256;auto*rows=sycl::malloc_shared<int>(E,q);
 for(int e=0;e<E;++e)rows[e]=M/E;
 auto*x=sycl::malloc_device<bf16>(size_t(M)*K,q);
 auto*w=sycl::malloc_device<unsigned char>(size_t(E)*N*K/2,q);
 auto*s=sycl::malloc_device<unsigned char>(size_t(E)*N*K/32,q);
 auto*y=sycl::malloc_device<bf16>(size_t(M)*N,q);
 auto*atomic=sycl::malloc_device<int>(1,q);
 q.fill(x,bf16(.125f),size_t(M)*K);q.memset(w,0x22,size_t(E)*N*K/2);
 q.memset(s,127,size_t(E)*N*K/32);q.wait();
 auto launch=[&]{q.memset(atomic,0,4);grimoire_xe2_grouped_mxfp4_autotune(
   &q,x,w,s,y,N,K,rows,E,atomic,policy,mode);q.wait_and_throw();};
 launch();double best=INFINITY;
 for(int i=0;i<3;++i){
  // Grimoire streams a different expert block every layer. Evict the prior
  // block so the tuner measures that production condition, not L2 reuse.
  q.fill(cold,static_cast<unsigned char>(i+1),cold_bytes).wait();
  auto t=std::chrono::steady_clock::now();launch();
  best=std::min(best,std::chrono::duration<double,std::milli>(
    std::chrono::steady_clock::now()-t).count());}
 const size_t sample=std::min<size_t>(4096,size_t(M)*N);
 std::vector<bf16> out(sample);q.memcpy(out.data(),y,sample*2).wait();
 if(reference->empty()){reference->resize(sample);for(size_t i=0;i<sample;++i)(*reference)[i]=float(out[i]);}
 maxerr=0;for(size_t i=0;i<sample;++i)maxerr=std::max(maxerr,
   std::abs(double(float(out[i]))-(*reference)[i]));
 sycl::free(x,q);sycl::free(w,q);sycl::free(s,q);sycl::free(y,q);
 sycl::free(atomic,q);sycl::free(rows,q);return best;
}

int main(){
 sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
 const char*policies[]={"32x64","32x128","32x256","64x64","64x128",
   "64x256","128x64","128x128","128x256"};
 const char*modes[]={"1xEU","2xEU","4xEU","8xEU","exact"};
 std::vector<float> ref_gu,ref_dn;Result all[9][5];
 constexpr size_t cold_bytes=size_t(768)<<20;
 auto*cold=sycl::malloc_device<unsigned char>(cold_bytes,q);
 if(!cold){std::puts("cold-cache allocation failed");return 3;}
 double ignored=0;
 (void)run_shape(q,5,0,32768,1024,2048,&ref_gu,ignored,cold,cold_bytes);
 (void)run_shape(q,5,0,32768,2048,512,&ref_dn,ignored,cold,cold_bytes);
 std::puts("policy,groups,gate_up_ms,down_ms,total_ms,max_abs,status");
 double winner=INFINITY;int wp=-1,wm=-1;
 for(int p=0;p<9;++p)for(int m=0;m<5;++m){
  try{double e1=0,e2=0;auto gu=run_shape(q,p,m,32768,1024,2048,&ref_gu,e1,cold,cold_bytes);
      auto dn=run_shape(q,p,m,32768,2048,512,&ref_dn,e2,cold,cold_bytes);
      auto&e=all[p][m];e={gu,dn,std::max(e1,e2),std::max(e1,e2)<=0.01};
      std::printf("%s,%s,%.3f,%.3f,%.3f,%.6g,%s\n",policies[p],modes[m],
        gu,dn,gu+dn,e.err,e.ok?"PASS":"REJECT_NUMERIC");
      if(e.ok&&gu+dn<winner){winner=gu+dn;wp=p;wm=m;}
  }catch(const std::exception&e){std::printf("%s,%s,nan,nan,nan,nan,REJECT:%s\n",
      policies[p],modes[m],e.what());}
 }
 if(wp<0){std::puts("NO VALID CONFIGURATION");return 2;}
 std::printf("WINNER policy=%s groups=%s total=%.3f ms candidates=45\n",
   policies[wp],modes[wm],winner);sycl::free(cold,q);return 0;
}
