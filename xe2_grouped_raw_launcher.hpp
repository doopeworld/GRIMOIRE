#pragma once
#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace grimoire_moe_raw {
using namespace cute;
template<typename A,typename B,typename S,typename D,char LA,char LB,
         class P,MoE::A_DTYPE AD,MoE::B_DTYPE BD> class Kernel;
template<typename A,typename D,class P> class DenseMXFP4Kernel;

template<char LA,char LB,class Policy,MoE::A_DTYPE AD,MoE::B_DTYPE BD,
         typename A,typename B,typename S,typename Bias,typename D>
void launch(sycl::queue&q,const A*a,const B*b,const S*s,const Bias*bias,D*d,
            int n,int k,const int*rows,const int*experts,int ne,int gs,int*atomic,
            int group_mode=0){
 using Op=XE_DPAS_TT<8,float,A>;
 using MMA=typename TiledMMAHelper<MMA_Atom<Op>,Layout<typename Policy::WGTile>,
     typename Policy::SGLayout>::TiledMMA;
 MMA mma{};const int eus=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
 const int threads=size(mma);sycl::range<3> local(1,1,threads);
 int groups=eus*512/threads;
 if(group_mode==1)groups*=2;
 else if(group_mode==2)groups*=4;
 else if(group_mode==3)groups*=8;
 else if(group_mode==4){
   const auto ht=mma.tile_mnk();const int tm=get<0>(ht),tn=get<1>(ht);
   groups=0;for(int e=0;e<ne;++e)
     groups+=((rows[e]+tm-1)/tm)*((n+tn-1)/tn);
   if(groups<1)groups=1;
 }
 sycl::range<3> global(1,groups,1);
 namespace sx=sycl::ext::oneapi::experimental;
 namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 using CA=typename Policy::GmemTiledCopyA;using CB=typename Policy::GmemTiledCopyB;
 using CD=typename Policy::GmemTiledCopyD;
 q.submit([&](sycl::handler&h){sycl::local_accessor<int32_t,1> lm(sycl::range<1>(1),h);
  h.parallel_for<Kernel<A,B,S,D,LA,LB,Policy,AD,BD>>(
   sycl::nd_range<3>{global*local,local},props,[=](auto){
    MoE::MoEGEMM<AD,BD,CA,CB,CD,LA,LB,'R'>(a,b,s,bias,d,mma,rows,experts,
      ne,gs,n,k,atomic,lm);});});
}

template<class Policy,typename D=cutlass::bfloat16_t>
void launch_dense_mxfp4(sycl::queue&q,const cutlass::bfloat16_t*a,
                        const unsigned char*b,const unsigned char*s,
                        D*d,int m,int n,int k){
 using A=cutlass::bfloat16_t;using Op=XE_DPAS_TT<8,float,A>;
 using MMA=typename TiledMMAHelper<MMA_Atom<Op>,Layout<typename Policy::WGTile>,
     typename Policy::SGLayout>::TiledMMA;
 MMA mma{};const int threads=size(mma);sycl::range<3> local(1,1,threads);
 const auto host_tile=mma.tile_mnk();
 const int tile_m=get<0>(host_tile),tile_n=get<1>(host_tile);
 const int groups=((m+tile_m-1)/tile_m)*((n+tile_n-1)/tile_n);
 sycl::range<3> global(1,groups,1);
 namespace sx=sycl::ext::oneapi::experimental;namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 using CA=typename Policy::GmemTiledCopyA;using CB=typename Policy::GmemTiledCopyB;
 using CD=typename Policy::GmemTiledCopyD;
 q.submit([&](sycl::handler&h){h.parallel_for<DenseMXFP4Kernel<A,D,Policy>>(
  sycl::nd_range<3>{global*local,local},props,[=](auto item){
   MMA mm{};auto wt=mm.tile_mnk();const int TM=get<0>(wt),TN=get<1>(wt);
   const int mt=(m+TM-1)/TM,nt=(n+TN-1)/TN,total=mt*nt;
   auto at=MoE::make_moe_tensor<A,'R'>(const_cast<A*>(a),m,k);
   auto bt=MoE::make_moe_tensor<cutlass::float_e2m1_t,'R'>(
       reinterpret_cast<cutlass::float_e2m1_t*>(const_cast<unsigned char*>(b)),n,k);
   auto dt=MoE::make_moe_tensor<D,'R'>(d,m,n);
   for(int t=item.get_group_linear_id();t<total;t+=item.get_group_range(1)){
    auto coord=make_coord(t/nt,t%nt,_,0);
    MoE::xe_gemm_4bits<CA,CB,CD,MoE::B_DTYPE::MXFP4,32>(
        at,bt,s,static_cast<const unsigned char*>(nullptr),dt,coord,mm);
   }});});
}
} // namespace grimoire_moe_raw
