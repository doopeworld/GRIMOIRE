#include "xe2_grouped_bridge.hpp"
#include "gemm_xe2_policy.hpp"
#include "xe2_grouped_raw_launcher.hpp"

#include <cstdlib>
namespace {
class mxfp4_64x256_policy : public MoE::xe_gemm_policy_base {
 public:
  using WGTile = cute::Shape<cute::_64,cute::_256,cute::_32>;
  using SGLayout = cute::Layout<cute::Shape<cute::_2,cute::_8,cute::_1>,
                                cute::Stride<cute::_8,cute::_1,cute::_0>>;
  using GmemTiledCopyD = cute::XE_STORE_2D<16,8,32>;
};
#define GRIMOIRE_P128X128_BODY \
 public: using WGTile=cute::Shape<cute::_128,cute::_128,cute::_32>; \
 using SGLayout=cute::Layout<cute::Shape<cute::_4,cute::_4,cute::_1>, \
   cute::Stride<cute::_4,cute::_1,cute::_0>>; \
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
class p128x128 : public MoE::xe_gemm_policy_base { GRIMOIRE_P128X128_BODY };
#undef GRIMOIRE_P128X128_BODY
class p128x128_f32 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_128,cute::_128,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_4,cute::_4,cute::_1>,
   cute::Stride<cute::_4,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<32,8,16>;
};
class p128x64 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_128,cute::_64,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_4,cute::_2,cute::_1>,
   cute::Stride<cute::_2,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};
class p128x256 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_128,cute::_256,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_4,cute::_8,cute::_1>,
   cute::Stride<cute::_8,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};
// Larger-tile candidates beyond the original 9-policy set.  gate_up is
// 34816x5120 at M=4096, so there is ample N and K to feed a bigger tile.
class p256x128 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_256,cute::_128,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_8,cute::_4,cute::_1>,
   cute::Stride<cute::_4,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};
class p256x256 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_256,cute::_256,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_8,cute::_4,cute::_1>,
   cute::Stride<cute::_4,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};
class p128x256_k64 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_128,cute::_256,cute::_64>;
 using SGLayout=cute::Layout<cute::Shape<cute::_4,cute::_8,cute::_1>,
   cute::Stride<cute::_8,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};
class p128x512 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_128,cute::_512,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_4,cute::_8,cute::_1>,
   cute::Stride<cute::_8,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};
class p64x128_prod : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_64,cute::_128,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_2,cute::_4,cute::_1>,
   cute::Stride<cute::_4,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};
#ifdef GRIMOIRE_ENABLE_AUTOTUNE
#define GRIMOIRE_POLICY(NAME,M,N,SM,SN) \
class NAME : public MoE::xe_gemm_policy_base { public: \
 using WGTile=cute::Shape<cute::_##M,cute::_##N,cute::_32>; \
 using SGLayout=cute::Layout<cute::Shape<cute::_##SM,cute::_##SN,cute::_1>, \
   cute::Stride<cute::_##SN,cute::_1,cute::_0>>; \
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>; };
GRIMOIRE_POLICY(p32x64,32,64,1,2)
GRIMOIRE_POLICY(p32x128,32,128,1,4)
GRIMOIRE_POLICY(p32x256,32,256,1,8)
GRIMOIRE_POLICY(p64x64,64,64,2,2)
GRIMOIRE_POLICY(p64x128,64,128,2,4)
#undef GRIMOIRE_POLICY
#define GRIMOIRE_POLICY_F32(NAME,M,N,SM,SN) \
class NAME : public MoE::xe_gemm_policy_base { public: \
 using WGTile=cute::Shape<cute::_##M,cute::_##N,cute::_32>; \
 using SGLayout=cute::Layout<cute::Shape<cute::_##SM,cute::_##SN,cute::_1>, \
   cute::Stride<cute::_##SN,cute::_1,cute::_0>>; \
 using GmemTiledCopyD=cute::XE_STORE_2D<32,8,16>; };
GRIMOIRE_POLICY_F32(p32x64_f32,32,64,1,2)
GRIMOIRE_POLICY_F32(p32x128_f32,32,128,1,4)
GRIMOIRE_POLICY_F32(p32x256_f32,32,256,1,8)
GRIMOIRE_POLICY_F32(p64x64_f32,64,64,2,2)
GRIMOIRE_POLICY_F32(p64x128_f32,64,128,2,4)
GRIMOIRE_POLICY_F32(p64x256_f32,64,256,2,8)
GRIMOIRE_POLICY_F32(p128x64_f32,128,64,4,2)
GRIMOIRE_POLICY_F32(p128x256_f32,128,256,4,8)
#undef GRIMOIRE_POLICY_F32
#endif
template<class Policy>
void launch_mxfp4(sycl::queue& q, const void* a, const unsigned char* b,
                  const unsigned char* scales, void* d, int n, int k,
                  const int* rows, const int* experts, int ne, int* atomic) {
    grimoire_moe_raw::launch<'R','C',Policy,MoE::A_DTYPE::BITS16,
        MoE::B_DTYPE::MXFP4,cutlass::bfloat16_t,unsigned char,unsigned char,
        unsigned char,cutlass::bfloat16_t>(
        q, static_cast<const cutlass::bfloat16_t*>(a), b, scales,
        static_cast<const unsigned char*>(nullptr),
        static_cast<cutlass::bfloat16_t*>(d), n, k, rows, experts, ne, 32,
        atomic);
}
template<class Policy>
void tune_mxfp4(sycl::queue& q,const void*a,const unsigned char*b,
 const unsigned char*s,void*d,int n,int k,const int*rows,int ne,int*atomic,int gm){
 grimoire_moe_raw::launch<'R','C',Policy,MoE::A_DTYPE::BITS16,MoE::B_DTYPE::MXFP4,
  cutlass::bfloat16_t,unsigned char,unsigned char,unsigned char,cutlass::bfloat16_t>(
  q,static_cast<const cutlass::bfloat16_t*>(a),b,s,
  static_cast<const unsigned char*>(nullptr),static_cast<cutlass::bfloat16_t*>(d),
  n,k,rows,nullptr,ne,32,atomic,gm);
}
}

class m8x128_g : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_8,cute::_128,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_1,cute::_4,cute::_1>,
   cute::Stride<cute::_4,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};
class m16x128_g : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_16,cute::_128,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_2,cute::_4,cute::_1>,
   cute::Stride<cute::_4,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};

extern "C" void grimoire_xe2_grouped_mxfp4_bf16(
    sycl::queue* q, const void* a, const unsigned char* b,
    const unsigned char* scales, void* d, int n, int k, const int* rows,
    const int* experts, int ne, int* atomic) {
    launch_mxfp4<p64x128_prod>(
        *q,a,b,scales,d,n,k,rows,experts,ne,atomic);
}

// Small-M grouped MoE.  The production grouped kernel is hardwired to
// p64x128_prod -- a 64-ROW tile.  In a speculative verifier the 4 rows spread
// over ~22 distinct experts per layer (measured 1.43x reuse), so each expert
// receives one or two rows of a 64-row tile and the rest is padding.  Ornith's
// routed MoE cost 7.08 ms at M=4 against a ~2.8 ms weight-traffic roofline.
// These are the same tile heights that fixed the dense path.
extern "C" void grimoire_xe2_grouped_mxfp4_bf16_m8(
    sycl::queue* q, const void* a, const unsigned char* b,
    const unsigned char* scales, void* d, int n, int k, const int* rows,
    const int* experts, int ne, int* atomic) {
    launch_mxfp4<m8x128_g>(*q,a,b,scales,d,n,k,rows,experts,ne,atomic);
}

extern "C" void grimoire_xe2_grouped_mxfp4_bf16_m16(
    sycl::queue* q, const void* a, const unsigned char* b,
    const unsigned char* scales, void* d, int n, int k, const int* rows,
    const int* experts, int ne, int* atomic) {
    launch_mxfp4<m16x128_g>(*q,a,b,scales,d,n,k,rows,experts,ne,atomic);
}

extern "C" int grimoire_xe2_dense_mxfp4_autotune(sycl::queue*,const void*,
 const unsigned char*,const unsigned char*,void*,int,int,int,int);
extern "C" int grimoire_xe2_dense_mxfp4_f32_autotune(sycl::queue*,const void*,
 const unsigned char*,const unsigned char*,void*,int,int,int,int);
// Runtime policy override so the dense tile can be swept in PRODUCTION (real
// activations) instead of the autotune harness, whose constant-filled buffers
// compress and therefore report times that do not transfer.
static int grimoire_dense_pol(const char* v){
  const char* e=std::getenv(v); return e&&*e?std::atoi(e):-1; }

extern "C" void grimoire_xe2_dense_mxfp4_bf16(
    sycl::queue* q,const void* a,const unsigned char* b,
    const unsigned char* scales,void* d,int m,int n,int k){
  static const int pol=grimoire_dense_pol("GRIMOIRE_DENSE_POLICY");
  if(pol>=0){grimoire_xe2_dense_mxfp4_autotune(q,a,b,scales,d,m,n,k,pol);return;}
  grimoire_moe_raw::launch_dense_mxfp4<p128x256>(
      *q,static_cast<const cutlass::bfloat16_t*>(a),b,scales,
      static_cast<cutlass::bfloat16_t*>(d),m,n,k);
}

// Small-M W4A8 tiles.  The 128-row tile fetches 128 rows of A per tile no
// matter how few tokens are given, which is ~89 MB/layer of padding at M=4.
// The DPAS atom is natively M=8, so a 16-row tile is weight-bound instead:
// measured on the FFN of 64 layers, 20.47 ms at M=16 against the 128-row
// tile's 46.09, and flat from M=1.  This is what makes a speculative verify
// batch cost about what a single decode step costs.
class m8x256 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_8,cute::_256,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_1,cute::_8,cute::_1>,
   cute::Stride<cute::_8,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<32,8,16>;
};
class m8x128 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_8,cute::_128,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_1,cute::_4,cute::_1>,
   cute::Stride<cute::_4,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<32,8,16>;
};
class m16x256 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_16,cute::_256,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_2,cute::_8,cute::_1>,
   cute::Stride<cute::_8,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<32,8,16>;
};
class m16x128 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_16,cute::_128,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_2,cute::_4,cute::_1>,
   cute::Stride<cute::_4,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<32,8,16>;
};

extern "C" void grimoire_xe2_dense_w4a8_f32_m8n128(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<m8x128,128,float>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<float*>(d),m,n,k);
}

extern "C" void grimoire_xe2_dense_w4a8_bf16_m8n128(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<m8x128,128,cutlass::bfloat16_t>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<cutlass::bfloat16_t*>(d),m,n,k);
}

extern "C" void grimoire_xe2_dense_w4a8_f32_m16n128(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<m16x128,128,float>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<float*>(d),m,n,k);
}

extern "C" void grimoire_xe2_dense_w4a8_bf16_m16n128(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<m16x128,128,cutlass::bfloat16_t>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<cutlass::bfloat16_t*>(d),m,n,k);
}

extern "C" void grimoire_xe2_dense_w4a8_f32_m8(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<m8x256,128,float>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<float*>(d),m,n,k);
}

// NInfer Build-2's dedicated 131072-row proposal head is symmetric signed
// Q4 with one FP16 scale per 64 K values.  Its byte plane is already the
// native low-nibble-first s4 layout consumed by the B70 DPAS kernel; only the
// scale plane is widened to float when the artifact is loaded.
extern "C" void grimoire_xe2_dense_w4a8_f32_m8g64(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<m8x256,64,float>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<float*>(d),m,n,k);
}

extern "C" void grimoire_xe2_dense_w4a8_bf16_m8(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<m8x256,128,cutlass::bfloat16_t>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<cutlass::bfloat16_t*>(d),m,n,k);
}

extern "C" void grimoire_xe2_dense_w4a8_f32_m16(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<m16x256,128,float>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<float*>(d),m,n,k);
}

extern "C" void grimoire_xe2_dense_w4a8_bf16_m16(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<m16x256,128,cutlass::bfloat16_t>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<cutlass::bfloat16_t*>(d),m,n,k);
}

// W4A8: int8 activations x symmetric signed int4 weights on the native
// s8xs4 DPAS.  1.69-1.87x the MXFP4 path on production shapes, bit-exact
// against an integer reference.  See tools/test_w4a8_atom.cpp.
extern "C" void grimoire_xe2_dense_w4a8_f32(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<p128x256_f32,128>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<float*>(d),m,n,k);
}

extern "C" void grimoire_xe2_dense_w4a8_bf16(
    sycl::queue* q,const void* a,const unsigned char* b,
    const float* wscale,const float* ascale,void* d,int m,int n,int k){
  grimoire_moe_raw::launch_dense_w4a8<p128x256,128,cutlass::bfloat16_t>(
      *q,static_cast<const int8_t*>(a),b,wscale,ascale,
      static_cast<cutlass::bfloat16_t*>(d),m,n,k);
}

extern "C" void grimoire_xe2_dense_mxfp4_f32(
    sycl::queue* q,const void* a,const unsigned char* b,
    const unsigned char* scales,void* d,int m,int n,int k){
  static const int pol=grimoire_dense_pol("GRIMOIRE_DENSE_POLICY_F32");
  if(pol>=0){grimoire_xe2_dense_mxfp4_f32_autotune(q,a,b,scales,d,m,n,k,pol);return;}
  grimoire_moe_raw::launch_dense_mxfp4<p128x256_f32,float>(
      *q,static_cast<const cutlass::bfloat16_t*>(a),b,scales,
      static_cast<float*>(d),m,n,k);
}

#ifdef GRIMOIRE_ENABLE_AUTOTUNE
extern "C" int grimoire_xe2_dense_mxfp4_autotune(
 sycl::queue*q,const void*a,const unsigned char*b,const unsigned char*s,void*d,
 int m,int n,int k,int policy){
 switch(policy){
  case 0:grimoire_moe_raw::launch_dense_mxfp4<p32x64>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 1:grimoire_moe_raw::launch_dense_mxfp4<p32x128>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 2:grimoire_moe_raw::launch_dense_mxfp4<p32x256>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 3:grimoire_moe_raw::launch_dense_mxfp4<p64x64>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 4:grimoire_moe_raw::launch_dense_mxfp4<p64x128>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 5:grimoire_moe_raw::launch_dense_mxfp4<mxfp4_64x256_policy>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 6:grimoire_moe_raw::launch_dense_mxfp4<p128x64>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 7:grimoire_moe_raw::launch_dense_mxfp4<p128x128>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 8:grimoire_moe_raw::launch_dense_mxfp4<MoE::w4a16_policy>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 9:grimoire_moe_raw::launch_dense_mxfp4<p256x128>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 10:grimoire_moe_raw::launch_dense_mxfp4<p256x256>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 11:grimoire_moe_raw::launch_dense_mxfp4<p128x256_k64>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  case 12:grimoire_moe_raw::launch_dense_mxfp4<p128x512>(*q,(const cutlass::bfloat16_t*)a,b,s,(cutlass::bfloat16_t*)d,m,n,k);break;
  default:return -1;
 }return 0;
}

extern "C" int grimoire_xe2_dense_mxfp4_f32_autotune(
 sycl::queue*q,const void*a,const unsigned char*b,const unsigned char*s,void*d,
 int m,int n,int k,int policy){
 switch(policy){
  case 0:grimoire_moe_raw::launch_dense_mxfp4<p32x64_f32,float>(*q,(const cutlass::bfloat16_t*)a,b,s,(float*)d,m,n,k);break;
  case 1:grimoire_moe_raw::launch_dense_mxfp4<p32x128_f32,float>(*q,(const cutlass::bfloat16_t*)a,b,s,(float*)d,m,n,k);break;
  case 2:grimoire_moe_raw::launch_dense_mxfp4<p32x256_f32,float>(*q,(const cutlass::bfloat16_t*)a,b,s,(float*)d,m,n,k);break;
  case 3:grimoire_moe_raw::launch_dense_mxfp4<p64x64_f32,float>(*q,(const cutlass::bfloat16_t*)a,b,s,(float*)d,m,n,k);break;
  case 4:grimoire_moe_raw::launch_dense_mxfp4<p64x128_f32,float>(*q,(const cutlass::bfloat16_t*)a,b,s,(float*)d,m,n,k);break;
  case 5:grimoire_moe_raw::launch_dense_mxfp4<p64x256_f32,float>(*q,(const cutlass::bfloat16_t*)a,b,s,(float*)d,m,n,k);break;
  case 6:grimoire_moe_raw::launch_dense_mxfp4<p128x64_f32,float>(*q,(const cutlass::bfloat16_t*)a,b,s,(float*)d,m,n,k);break;
  case 7:grimoire_moe_raw::launch_dense_mxfp4<p128x128_f32,float>(*q,(const cutlass::bfloat16_t*)a,b,s,(float*)d,m,n,k);break;
  case 8:grimoire_moe_raw::launch_dense_mxfp4<p128x256_f32,float>(*q,(const cutlass::bfloat16_t*)a,b,s,(float*)d,m,n,k);break;
  default:return -1;
 }return 0;
}

extern "C" int grimoire_xe2_grouped_mxfp4_autotune(
 sycl::queue*q,const void*a,const unsigned char*b,const unsigned char*s,void*d,
 int n,int k,const int*rows,int ne,int*atomic,int policy,int group_mode){
 switch(policy){
  case 0:tune_mxfp4<p32x64>(*q,a,b,s,d,n,k,rows,ne,atomic,group_mode);break;
  case 1:tune_mxfp4<p32x128>(*q,a,b,s,d,n,k,rows,ne,atomic,group_mode);break;
  case 2:tune_mxfp4<p32x256>(*q,a,b,s,d,n,k,rows,ne,atomic,group_mode);break;
  case 3:tune_mxfp4<p64x64>(*q,a,b,s,d,n,k,rows,ne,atomic,group_mode);break;
  case 4:tune_mxfp4<p64x128>(*q,a,b,s,d,n,k,rows,ne,atomic,group_mode);break;
  case 5:tune_mxfp4<mxfp4_64x256_policy>(*q,a,b,s,d,n,k,rows,ne,atomic,group_mode);break;
  case 6:tune_mxfp4<p128x64>(*q,a,b,s,d,n,k,rows,ne,atomic,group_mode);break;
  case 7:tune_mxfp4<p128x128>(*q,a,b,s,d,n,k,rows,ne,atomic,group_mode);break;
  case 8:tune_mxfp4<MoE::w4a16_policy>(*q,a,b,s,d,n,k,rows,ne,atomic,group_mode);break;
  default:return -1;
 }return 0;
}
#endif
