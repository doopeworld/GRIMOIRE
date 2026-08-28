#include "xe2_attention_bridge.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using bf16=sycl::ext::oneapi::bfloat16;
int main(){
 constexpr int M=64,KH=16,KD=128,VH=32,VD=128;
 sycl::queue z{sycl::gpu_selector_v};
 auto bd=[&](size_t n){return sycl::malloc_device<bf16>(n,z);};
 auto fd=[&](size_t n){return sycl::malloc_device<float>(n,z);};
 auto *q=bd(size_t(M)*KH*KD),*k=bd(size_t(M)*KH*KD),*v=bd(size_t(M)*VH*VD),*o=bd(size_t(M)*VH*VD),*db=bd(VH);
 auto *b=fd(size_t(VH)*M),*a=fd(size_t(VH)*M),*al=fd(VH),*st=fd(size_t(VH)*VD*KD);
 auto *cu=sycl::malloc_device<int>(2,z),*ci=sycl::malloc_device<int>(1,z);auto* hs=sycl::malloc_device<bool>(1,z);
 std::vector<bf16> hq(size_t(M)*KH*KD),hk(hq.size()),hv(size_t(M)*VH*VD),hdb(VH),ho(hv.size());
 std::vector<float> hb(size_t(VH)*M),ha(hb.size()),hal(VH),hst(size_t(VH)*VD*KD,0),ref(hv.size());
 for(size_t i=0;i<hq.size();++i){hq[i]=bf16(float(int(i*5%23)-11)/32);hk[i]=bf16(float(int(i*7%29)-14)/32);}
 for(int t=0;t<M;++t)for(int h=0;h<KH;++h){float qn=0,kn=0;for(int d=0;d<KD;++d){float qv=float(hq[(size_t(t)*KH+h)*KD+d]),kv=float(hk[(size_t(t)*KH+h)*KD+d]);qn+=qv*qv;kn+=kv*kv;}qn=std::sqrt(qn)*std::sqrt(float(KD));kn=std::sqrt(kn);for(int d=0;d<KD;++d){hq[(size_t(t)*KH+h)*KD+d]=bf16(float(hq[(size_t(t)*KH+h)*KD+d])/qn);hk[(size_t(t)*KH+h)*KD+d]=bf16(float(hk[(size_t(t)*KH+h)*KD+d])/kn);}}
 for(size_t i=0;i<hv.size();++i)hv[i]=bf16(float(int(i*11%31)-15)/32);
 for(int h=0;h<VH;++h){hal[h]=-2.0f+0.01f*h;hdb[h]=bf16(-0.2f+0.01f*h);for(int t=0;t<M;++t){ha[size_t(h)*M+t]=-1.0f+float((h+t)%17)/16;hb[size_t(h)*M+t]=0.1f+0.8f*float((h*3+t)%19)/18;}}
 int hcu[2]={0,M},hci=0;bool hhs=false;
 z.memcpy(q,hq.data(),hq.size()*2);z.memcpy(k,hk.data(),hk.size()*2);z.memcpy(v,hv.data(),hv.size()*2);z.memcpy(b,hb.data(),hb.size()*4);z.memcpy(a,ha.data(),ha.size()*4);z.memcpy(al,hal.data(),hal.size()*4);z.memcpy(db,hdb.data(),hdb.size()*2);z.memset(st,0,hst.size()*4);z.memcpy(cu,hcu,8);z.memcpy(ci,&hci,4);z.memcpy(hs,&hhs,1);z.wait();
 grimoire_xe2_chunk_gdn_bf16(&z,o,q,k,v,b,a,al,db,st,M,KH,KD,VH,VD,cu,ci,hs);z.wait_and_throw();z.memcpy(ho.data(),o,ho.size()*2).wait();
 for(int h=0;h<VH;++h)for(int t=0;t<M;++t){int khh=h/(VH/KH);float alpha=std::exp(-std::exp(hal[h])*std::log1p(std::exp(ha[size_t(h)*M+t]+float(hdb[h]))));for(int r=0;r<VD;++r){float w=0;for(int d=0;d<KD;++d)w+=hst[(size_t(h)*VD+r)*KD+d]*float(hk[(size_t(t)*KH+khh)*KD+d]);float corr=hb[size_t(h)*M+t]*(float(hv[(size_t(t)*VH+h)*VD+r])-alpha*w);float y=0;for(int d=0;d<KD;++d){float& s=hst[(size_t(h)*VD+r)*KD+d];s=alpha*s+corr*float(hk[(size_t(t)*KH+khh)*KD+d]);y+=s*float(hq[(size_t(t)*KH+khh)*KD+d]);}ref[(size_t(t)*VH+h)*VD+r]=y;}}
 double se=0,sr=0,mx=0;for(size_t i=0;i<ref.size();i+=97){double e=float(ho[i])-ref[i];se+=e*e;sr+=double(ref[i])*ref[i];mx=std::max(mx,std::abs(e));}double rel=std::sqrt(se/(sr+1e-30));std::printf("rel_l2=%.6e max_abs=%.6e\n",rel,mx);return rel<0.03?0:2;
}
