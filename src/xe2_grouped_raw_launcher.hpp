#pragma once
#include <cstdlib>
#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace grimoire_moe_raw {
using namespace cute;
template<typename A,typename B,typename S,typename D,char LA,char LB,
         class P,MoE::A_DTYPE AD,MoE::B_DTYPE BD> class Kernel;
template<typename A,typename D,class P> class DenseMXFP4Kernel;
template<typename A,typename D,class P,int G> class DenseINT4Kernel;

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

// =====================================================================
//  W4A8: int8 activations x SIGNED int4 weights on the NATIVE s8xs4 DPAS.
//
//  The MXFP4/INT4 paths above dequantize B into bf16 fragments and run a
//  bf16 DPAS; that dequant is what pins them near 100 TFLOP/s while the
//  bf16 XMX peak is ~179.  This path feeds the packed 4-bit weights to the
//  DPAS DIRECTLY -- cutlass declares XE_DPAS_TT<8,int32,int8,int4,int32>
//  and `reorder` bridges the 2-D block-load fragment to the atom's B
//  operand with no conversion at all.
//
//  MEASURED 2026-08-26, 64 distinct matrices (production streaming), with
//  the group rescale and per-row activation scale below included:
//      ffn-gate-up 168.2 TFLOP/s (1.69x MXFP4)
//      ffn-down    186.1          (1.87x)
//      dn-qkv      174.4          (1.75x)
//  Correctness is BIT-EXACT against an integer host reference
//  (tools/test_w4a8_atom.cpp) -- int32 math, so there is no tolerance to
//  hide behind.
//
//  Weights must be SYMMETRIC signed int4, group size G along K, packed two
//  nibbles per byte with element 2i in the LOW nibble (cute's int4_t is
//  signed two's complement, low-nibble-first -- verified, not assumed).
//  An int32 sum is only valid while the weight scale is constant, so the
//  accumulator is folded into a float accumulator at every G boundary.
// =====================================================================
template<typename A,typename D,class P,int G> class DenseW4A8Kernel;

template<class Policy,int G=128,typename DT=float>
void launch_dense_w4a8(sycl::queue&q,const int8_t*a,const unsigned char*b,
                       const float*wscale,const float*ascale,DT*d,
                       int m,int n,int k){
 using Op=XE_DPAS_TT<8,int32_t,int8_t,cute::int4_t,int32_t>;
 using MMA=typename TiledMMAHelper<MMA_Atom<Op>,Layout<typename Policy::WGTile>,
     typename Policy::SGLayout>::TiledMMA;
 MMA mma{};const int threads=size(mma);sycl::range<3> local(1,1,threads);
 const auto host_tile=mma.tile_mnk();
 const int tile_m=get<0>(host_tile),tile_n=get<1>(host_tile);
 const int blocks=((m+tile_m-1)/tile_m)*((n+tile_n-1)/tile_n);
 const int eus=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
 int groups=eus*512/threads; if(groups<1)groups=1;
 if(groups>blocks)groups=blocks;
 sycl::range<3> global(1,groups,1);
 namespace sx=sycl::ext::oneapi::experimental;namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 using CA=typename Policy::GmemTiledCopyA;using CB=typename Policy::GmemTiledCopyB;
 q.submit([&](sycl::handler&h){h.parallel_for<DenseW4A8Kernel<int8_t,DT,Policy,G>>(
  sycl::nd_range<3>{global*local,local},props,[=](auto item){
   MMA mm{};
   auto At=MoE::make_moe_tensor<int8_t,'R'>(const_cast<int8_t*>(a),m,k);
   auto Bt=MoE::make_moe_tensor<cute::int4_t,'R'>(
       reinterpret_cast<cute::int4_t*>(const_cast<unsigned char*>(b)),n,k);
   auto Dt=MoE::make_moe_tensor<DT,'R'>(d,m,n);
   auto wg_tile=mm.tile_mnk();
   const int TM=int(get<0>(wg_tile)),TN=int(get<1>(wg_tile)),KT=int(get<2>(wg_tile));
   const int mt=(m+TM-1)/TM,nt=(n+TN-1)/TN,total=mt*nt;
   const int lid=int(item.get_local_linear_id());
   const int ktpg=G/KT;                      // k-tiles per quantization group
   const int kg=k/G;
   constexpr int ATOM_N=get<2>(typename MMA::ThrLayoutVMNK{}.shape());
   constexpr int ATOM_M=get<1>(typename MMA::ThrLayoutVMNK{}.shape());
   const int SG_M=TM/ATOM_M,SG_N=TN/ATOM_N;
   const int sgid=int(cutlass::get_sub_group_id());
   const int lane=int(cutlass::get_sub_group_local_id());
   for(int t=int(item.get_group_linear_id());t<total;t+=int(item.get_group_range(1))){
    const int mi=t/nt,ni=t%nt;
    Tensor cA=make_identity_tensor(At.shape());
    Tensor cB=make_identity_tensor(Bt.shape());
    Tensor cC=make_identity_tensor(Dt.shape());
    Tensor gA=local_tile(cA,select<0,2>(wg_tile),make_coord(mi,_));
    Tensor gB=local_tile(cB,select<1,2>(wg_tile),make_coord(ni,_));
    Tensor gC=local_tile(cC,wg_tile,make_coord(mi,ni,0),Step<_1,_1,X>{});
    auto copy_a=MoE::get_block_2d_copy_A<CA>(mm,At);
    auto copy_b=MoE::get_block_2d_copy_B<CB>(mm,Bt);
    auto thr_mma=mm.get_slice(lid);
    auto tCrA=thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB=thr_mma.partition_sg_fragment_B(gB(_,_,0));
    auto tArA=copy_a.get_slice(lid).partition_sg_fragment_D(gA(_,_,0));
    auto tBrB=copy_b.get_slice(lid).partition_sg_fragment_D(gB(_,_,0));
    Tensor tAgA=copy_a.get_slice(lid).partition_S(gA);
    Tensor tBgB=copy_b.get_slice(lid).partition_S(gB);
    auto tCrC=thr_mma.partition_sg_fragment_C(gC);
    auto pfa=make_block_2d_prefetch(copy_a);
    auto pfb=make_block_2d_prefetch(copy_b);
    auto pA=pfa.get_slice(lid).partition_S(gA);
    auto pB=pfb.get_slice(lid).partition_S(gB);
    const int m0=mi*TM+(sgid/ATOM_N)*SG_M;
    const int n0=ni*TN+(sgid%ATOM_N)*SG_N;
    constexpr int NACC=64;
    float facc[NACC];
    #pragma unroll
    for(int i=0;i<NACC;++i)facc[i]=0.0f;
    const int k_tiles=k/KT;
    int kp=0;
    #pragma unroll
    for(;kp<3;++kp){prefetch(pfa,pA(_,_,_,kp));prefetch(pfb,pB(_,_,_,kp));}
    clear(tCrC);
    for(int kt=0;kt<k_tiles;++kt,++kp){
     barrier_arrive(2);
     copy(copy_a,tAgA(_,_,_,kt),tArA);
     copy(copy_b,tBgB(_,_,_,kt),tBrB);
     if(kp<k_tiles){prefetch(pfa,pA(_,_,_,kp));prefetch(pfb,pB(_,_,_,kp));}
     reorder(tArA,tCrA);
     reorder(tBrB,tCrB);
     cute::gemm(mm,tCrA,tCrB,tCrC);
     barrier_wait(2);
     if((kt+1)%ktpg==0){
      const int g=(kt+1)/ktpg-1;
      for(int sn=0;sn<SG_N/16;++sn){
       const int nn=n0+sn*16+lane;
       const float ws=(nn<n)?wscale[size_t(nn)*kg+g]:0.0f;
       #pragma unroll
       for(int sm=0;sm<SG_M;++sm)
        facc[sn*SG_M+sm]=sycl::fma(float(tCrC(sn*SG_M+sm)),ws,facc[sn*SG_M+sm]);
      }
      clear(tCrC);
     }
    }
    for(int sn=0;sn<SG_N/16;++sn)
     for(int sm=0;sm<SG_M;++sm){
      const int mm_=m0+sm,nn=n0+sn*16+lane;
      if(mm_<m&&nn<n) d[size_t(mm_)*n+nn]=DT(facc[sn*SG_M+sm]*ascale[mm_]);
     }
   }});});
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
 // L2 swizzle.  The default row-major walk (t/nt, t%nt) sweeps every n-tile for
 // one m-tile, so the whole B matrix is re-streamed once per m-tile: for gate_up
 // (n=34816,k=5120) that is 89 MB x 32 m-tiles = 2.85 GB per layer.  Grouping
 // GM consecutive m-tiles against one n-tile keeps that B tile resident across
 // GM workgroups and cuts B traffic by ~GM.  GM=1 reproduces the old order.
 static const int kSwizzleGM=[]{const char*e=std::getenv("GRIMOIRE_GEMM_SWIZZLE_GM");
   int v=e&&*e?std::atoi(e):4;return v<1?1:v;}();
 // 2-D L2 blocking.  Grouping only M caps B reuse at GM, and the A working set
 // (GM x A_tile) evicts B past GM=4.  Blocking BOTH m and n keeps a BM x BN tile
 // pair resident: traffic = (mt/BM)*B_total + (nt/BN)*A_total.  BN=0 disables.
 static const int kSwizzleBN=[]{const char*e=std::getenv("GRIMOIRE_GEMM_SWIZZLE_BN");
   int v=e&&*e?std::atoi(e):0;return v<0?0:v;}();
 const int swz_gm=kSwizzleGM, swz_bn=kSwizzleBN;
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
    int mi,ni;
    if(swz_bn>0){
      const int bm=swz_gm<1?1:swz_gm, bn=swz_bn;
      const int nblk=(nt+bn-1)/bn, per=bm*bn;
      const int blk=t/per, r=t-blk*per;
      const int bi=blk/nblk, bj=blk-bi*nblk;
      const int m0=bi*bm, n0=bj*bn;
      int curm=mt-m0; if(curm>bm) curm=bm;
      int curn=nt-n0; if(curn>bn) curn=bn;
      if(curm<1||curn<1){mi=t/nt;ni=t%nt;}
      else{ const int rr=r%(curm*curn); mi=m0+(rr%curm); ni=n0+(rr/curm); }
    }
    else if(swz_gm<=1){mi=t/nt;ni=t%nt;}
    else{const int tpg=swz_gm*nt;const int g=t/tpg,r=t-g*tpg;
         const int m0=g*swz_gm;int cur=mt-m0;if(cur>swz_gm)cur=swz_gm;
         mi=m0+(r%cur);ni=r/cur;}
    auto coord=make_coord(mi,ni,_,0);
    MoE::xe_gemm_4bits<CA,CB,CD,MoE::B_DTYPE::MXFP4,32>(
        at,bt,s,static_cast<const unsigned char*>(nullptr),dt,coord,mm);
   }});});
}

// INT4 W4A16 twin of launch_dense_mxfp4.  Same cutlass template, same policy;
// only B_DTYPE, the B element type and the scale type differ.  INT4 scales are
// BF16 (MXFP4 scales are uint8 E8M0), and the group size is a template arg so
// 32/64/128 can be swept.  Measured 2026-08-25: int4 g128 beats MXFP4 by 1.28x
// on ffn-gate-up and 1.62x on ffn-down.
template<class Policy,int GroupSize=128,typename D=cutlass::bfloat16_t>
void launch_dense_int4(sycl::queue&q,const cutlass::bfloat16_t*a,
                        const unsigned char*b,const cutlass::bfloat16_t*s,
                        D*d,int m,int n,int k){
 using A=cutlass::bfloat16_t;using Op=XE_DPAS_TT<8,float,A>;
 using MMA=typename TiledMMAHelper<MMA_Atom<Op>,Layout<typename Policy::WGTile>,
     typename Policy::SGLayout>::TiledMMA;
 MMA mma{};const int threads=size(mma);sycl::range<3> local(1,1,threads);
 const auto host_tile=mma.tile_mnk();
 const int tile_m=get<0>(host_tile),tile_n=get<1>(host_tile);
 const int groups=((m+tile_m-1)/tile_m)*((n+tile_n-1)/tile_n);
 // L2 swizzle.  The default row-major walk (t/nt, t%nt) sweeps every n-tile for
 // one m-tile, so the whole B matrix is re-streamed once per m-tile: for gate_up
 // (n=34816,k=5120) that is 89 MB x 32 m-tiles = 2.85 GB per layer.  Grouping
 // GM consecutive m-tiles against one n-tile keeps that B tile resident across
 // GM workgroups and cuts B traffic by ~GM.  GM=1 reproduces the old order.
 static const int kSwizzleGM=[]{const char*e=std::getenv("GRIMOIRE_GEMM_SWIZZLE_GM");
   int v=e&&*e?std::atoi(e):4;return v<1?1:v;}();
 // 2-D L2 blocking.  Grouping only M caps B reuse at GM, and the A working set
 // (GM x A_tile) evicts B past GM=4.  Blocking BOTH m and n keeps a BM x BN tile
 // pair resident: traffic = (mt/BM)*B_total + (nt/BN)*A_total.  BN=0 disables.
 static const int kSwizzleBN=[]{const char*e=std::getenv("GRIMOIRE_GEMM_SWIZZLE_BN");
   int v=e&&*e?std::atoi(e):0;return v<0?0:v;}();
 const int swz_gm=kSwizzleGM, swz_bn=kSwizzleBN;
 sycl::range<3> global(1,groups,1);
 namespace sx=sycl::ext::oneapi::experimental;namespace ix=sycl::ext::intel::experimental;
 sx::properties props{sx::sub_group_size<16>,ix::grf_size<256>};
 using CA=typename Policy::GmemTiledCopyA;using CB=typename Policy::GmemTiledCopyB;
 using CD=typename Policy::GmemTiledCopyD;
 q.submit([&](sycl::handler&h){h.parallel_for<DenseINT4Kernel<A,D,Policy,GroupSize>>(
  sycl::nd_range<3>{global*local,local},props,[=](auto item){
   MMA mm{};auto wt=mm.tile_mnk();const int TM=get<0>(wt),TN=get<1>(wt);
   const int mt=(m+TM-1)/TM,nt=(n+TN-1)/TN,total=mt*nt;
   auto at=MoE::make_moe_tensor<A,'R'>(const_cast<A*>(a),m,k);
   auto bt=MoE::make_moe_tensor<cute::int4_t,'R'>(
       reinterpret_cast<cute::int4_t*>(const_cast<unsigned char*>(b)),n,k);
   auto dt=MoE::make_moe_tensor<D,'R'>(d,m,n);
   for(int t=item.get_group_linear_id();t<total;t+=item.get_group_range(1)){
    int mi,ni;
    if(swz_bn>0){
      const int bm=swz_gm<1?1:swz_gm, bn=swz_bn;
      const int nblk=(nt+bn-1)/bn, per=bm*bn;
      const int blk=t/per, r=t-blk*per;
      const int bi=blk/nblk, bj=blk-bi*nblk;
      const int m0=bi*bm, n0=bj*bn;
      int curm=mt-m0; if(curm>bm) curm=bm;
      int curn=nt-n0; if(curn>bn) curn=bn;
      if(curm<1||curn<1){mi=t/nt;ni=t%nt;}
      else{ const int rr=r%(curm*curn); mi=m0+(rr%curm); ni=n0+(rr/curm); }
    }
    else if(swz_gm<=1){mi=t/nt;ni=t%nt;}
    else{const int tpg=swz_gm*nt;const int g=t/tpg,r=t-g*tpg;
         const int m0=g*swz_gm;int cur=mt-m0;if(cur>swz_gm)cur=swz_gm;
         mi=m0+(r%cur);ni=r/cur;}
    auto coord=make_coord(mi,ni,_,0);
    MoE::xe_gemm_4bits<CA,CB,CD,MoE::B_DTYPE::INT4,GroupSize>(
        at,bt,s,static_cast<const cutlass::bfloat16_t*>(nullptr),dt,coord,mm);
   }});});
}
} // namespace grimoire_moe_raw
