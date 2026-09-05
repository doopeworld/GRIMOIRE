#include "xe2_onednn_bridge.hpp"
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <unordered_map>

namespace {
using namespace dnnl;

struct Plan {
  engine eng;
  stream strm;
  matmul::primitive_desc pd;
  matmul prim;
  memory::desc scale_md;
  memory::desc zp_md;
  size_t scratch_bytes;
  memory ma,mw,ms,mz,mo,mx;
  bool bound=false;

  Plan(sycl::queue& q,int m,int n,int k,int gs,bool bf16)
    : eng(sycl_interop::make_engine(q.get_device(),q.get_context())),
      strm(sycl_interop::make_stream(eng,q)),
      pd(make_pd(eng,m,n,k,gs,bf16)),prim(pd),
      scale_md({k/gs,n},bf16?memory::data_type::bf16:memory::data_type::f16,
               memory::dims{n,1}),
      zp_md({1},memory::data_type::s8,memory::dims{1}),
      scratch_bytes(pd.scratchpad_desc().get_size()) {}

  static matmul::primitive_desc make_pd(const engine& e,int m,int n,int k,
                                         int gs,bool bf16){
    const auto f16=bf16?memory::data_type::bf16:memory::data_type::f16;
    memory::desc src({m,k},f16,memory::dims{k,1});
    memory::desc wei({k,n},memory::data_type::u4,memory::dims{1,k});
    memory::desc dst({m,n},f16,memory::dims{n,1});
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    attr.set_scales(DNNL_ARG_WEIGHTS,3,{gs,1},f16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS,0,{},memory::data_type::s8);
    attr.set_fpmath_mode(bf16?fpmath_mode::bf16:fpmath_mode::f16,true);
    return matmul::primitive_desc(e,src,wei,dst,attr);
  }

  memory wrap(const memory::desc& md,void* p){
    return sycl_interop::make_memory(md,eng,sycl_interop::memory_kind::usm,p);
  }
  void run(void* a,void* b,void* s,void* zp,void* out,void* scratch){
    if(!bound){
      ma=wrap(pd.src_desc(),a);mw=wrap(pd.weights_desc(),b);mo=wrap(pd.dst_desc(),out);
      ms=wrap(scale_md,s);mz=wrap(zp_md,zp);mx=wrap(pd.scratchpad_desc(),scratch);
      bound=true;
    }else{
      ma.set_data_handle(a);mw.set_data_handle(b);ms.set_data_handle(s);
      mz.set_data_handle(zp);mo.set_data_handle(out);mx.set_data_handle(scratch);
    }
    prim.execute(strm,{{DNNL_ARG_SRC,ma},{DNNL_ARG_WEIGHTS,mw},{DNNL_ARG_DST,mo},
      {DNNL_ARG_ATTR_SCALES|DNNL_ARG_WEIGHTS,ms},
      {DNNL_ARG_ATTR_ZERO_POINTS|DNNL_ARG_WEIGHTS,mz},{DNNL_ARG_SCRATCHPAD,mx}});
  }
};

struct MXPlan {
  engine eng;
  stream strm;
  matmul::primitive_desc pd;
  matmul prim;
  memory::desc scale_md;
  size_t scratch_bytes;
  memory ma,mw,ms,mo,mx;
  bool bound=false;

  MXPlan(sycl::queue& q,int m,int n,int k)
    : eng(sycl_interop::make_engine(q.get_device(),q.get_context())),
      strm(sycl_interop::make_stream(eng,q)),
      pd(make_pd(eng,m,n,k)),prim(pd),
      // oneDNN's MX scale argument is a flat physical buffer. Native
      // Grimoire scales are already contiguous in the required order:
      // one E8M0 byte after each physical 32-value block of [N,K].
      scale_md({int64_t(k/32)*n},memory::data_type::e8m0,memory::dims{1}),
      scratch_bytes(pd.scratchpad_desc().get_size()) {}

  static matmul::primitive_desc make_pd(const engine& e,int m,int n,int k){
    memory::desc src({m,k},memory::data_type::bf16,memory::dims{k,1});
    // Grimoire payload is [N,K] row-major packed E2M1.  The transposed
    // logical oneDNN view [K,N] is therefore stride {1,K}.
    memory::desc wei({k,n},memory::data_type::f4_e2m1,memory::dims{1,k});
    memory::desc dst({m,n},memory::data_type::bf16,memory::dims{n,1});
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    attr.set_scales(DNNL_ARG_WEIGHTS,3,{32,1},memory::data_type::e8m0);
    attr.set_fpmath_mode(fpmath_mode::bf16,true);
    return matmul::primitive_desc(e,src,wei,dst,attr);
  }
  memory wrap(const memory::desc& md,void* p){
    return sycl_interop::make_memory(md,eng,sycl_interop::memory_kind::usm,p);
  }
  void run(void* a,void* b,void* s,void* out,void* scratch){
    if(!bound){
      ma=wrap(pd.src_desc(),a);mw=wrap(pd.weights_desc(),b);
      ms=wrap(scale_md,s);mo=wrap(pd.dst_desc(),out);
      mx=wrap(pd.scratchpad_desc(),scratch);bound=true;
    }else{
      ma.set_data_handle(a);mw.set_data_handle(b);ms.set_data_handle(s);
      mo.set_data_handle(out);mx.set_data_handle(scratch);
    }
    prim.execute(strm,{{DNNL_ARG_SRC,ma},{DNNL_ARG_WEIGHTS,mw},
      {DNNL_ARG_DST,mo},{DNNL_ARG_ATTR_SCALES|DNNL_ARG_WEIGHTS,ms},
      {DNNL_ARG_SCRATCHPAD,mx}});
  }
};

struct BFPlan {
  engine eng;
  stream strm;
  matmul::primitive_desc pd;
  matmul prim;
  size_t scratch_bytes;
  memory ma,mw,mo,mx;
  bool bound=false;

  BFPlan(sycl::queue& q,int m,int n,int k)
    : eng(sycl_interop::make_engine(q.get_device(),q.get_context())),
      strm(sycl_interop::make_stream(eng,q)),
      pd(make_pd(eng,m,n,k)),prim(pd),
      scratch_bytes(pd.scratchpad_desc().get_size()) {}

  static matmul::primitive_desc make_pd(const engine& e,int m,int n,int k){
    memory::desc src({m,k},memory::data_type::bf16,memory::dims{k,1});
    memory::desc wei({k,n},memory::data_type::bf16,memory::dims{1,k});
    memory::desc dst({m,n},memory::data_type::f32,memory::dims{n,1});
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    return matmul::primitive_desc(e,src,wei,dst,attr);
  }
  memory wrap(const memory::desc& md,void* p){
    return sycl_interop::make_memory(md,eng,sycl_interop::memory_kind::usm,p);
  }
  void run(void* a,void* b,void* out,void* scratch){
    if(!bound){
      ma=wrap(pd.src_desc(),a);mw=wrap(pd.weights_desc(),b);
      mo=wrap(pd.dst_desc(),out);mx=wrap(pd.scratchpad_desc(),scratch);
      bound=true;
    }else{
      ma.set_data_handle(a);mw.set_data_handle(b);
      mo.set_data_handle(out);mx.set_data_handle(scratch);
    }
    prim.execute(strm,{{DNNL_ARG_SRC,ma},{DNNL_ARG_WEIGHTS,mw},
      {DNNL_ARG_DST,mo},{DNNL_ARG_SCRATCHPAD,mx}});
  }
};

struct F16Plan {
  engine eng;
  stream strm;
  matmul::primitive_desc pd;
  matmul prim;
  size_t scratch_bytes;
  memory ma,mw,mo,mx;
  bool bound=false;

  F16Plan(sycl::queue& q,int m,int n,int k)
    : eng(sycl_interop::make_engine(q.get_device(),q.get_context())),
      strm(sycl_interop::make_stream(eng,q)),pd(make_pd(eng,m,n,k)),prim(pd),
      scratch_bytes(pd.scratchpad_desc().get_size()) {}
  static matmul::primitive_desc make_pd(const engine& e,int m,int n,int k){
    memory::desc src({m,k},memory::data_type::f16,memory::dims{k,1});
    // Checkpoint/Grimoire storage is [N,K] row-major. This logical [K,N]
    // view is the exact transpose consumed by torch F.linear.
    memory::desc wei({k,n},memory::data_type::f16,memory::dims{1,k});
    memory::desc dst({m,n},memory::data_type::f16,memory::dims{n,1});
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    return matmul::primitive_desc(e,src,wei,dst,attr);
  }
  memory wrap(const memory::desc& md,void* p){
    return sycl_interop::make_memory(md,eng,sycl_interop::memory_kind::usm,p);
  }
  void run(void* a,void* b,void* out,void* scratch){
    if(!bound){
      ma=wrap(pd.src_desc(),a);mw=wrap(pd.weights_desc(),b);
      mo=wrap(pd.dst_desc(),out);mx=wrap(pd.scratchpad_desc(),scratch);
      bound=true;
    }else{
      ma.set_data_handle(a);mw.set_data_handle(b);
      mo.set_data_handle(out);mx.set_data_handle(scratch);
    }
    prim.execute(strm,{{DNNL_ARG_SRC,ma},{DNNL_ARG_WEIGHTS,mw},
      {DNNL_ARG_DST,mo},{DNNL_ARG_SCRATCHPAD,mx}});
  }
};
}

// ---------------------------------------------------------------------
// SIGNED int4 twin of Plan.
//
// GRIMOIRE's W4A8 payload stores every weight as a SIGN-EXTENDED nibble:
// launch_gemv_int4sym decodes it as int8_t(b << 4) >> 4, so 0..7 map to 0..7
// and 8..15 map to -8..-1.  Plan above declares the same bytes as u4 with a
// zero point of 8, which decodes nibble b as b - 8.  Those are the same value
// set ROTATED BY EIGHT -- nibble 0 is 0 to GRIMOIRE and -8 to oneDNN -- so
// every weight in the model comes out wrong.  That is why GRIMOIRE_ONEDNN_I4
// produced fluent nonsense ("lykebeebeebe") instead of noise, and why the
// speedup it reported on 2026-09-05 was measured on a broken kernel.
//
// s4 sign-extends exactly like the GEMV's LUT, so the payload maps straight in
// and the zero point disappears entirely.
struct SPlan {
  engine eng;
  stream strm;
  matmul::primitive_desc pd;
  matmul prim;
  memory::desc scale_md;
  size_t scratch_bytes;
  memory ma,mw,ms,mo,mx;
  bool bound=false;

  SPlan(sycl::queue& q,int m,int n,int k,int gs,bool bf16)
    : eng(sycl_interop::make_engine(q.get_device(),q.get_context())),
      strm(sycl_interop::make_stream(eng,q)),
      pd(make_pd(eng,m,n,k,gs,bf16)),prim(pd),
      scale_md({k/gs,n},bf16?memory::data_type::bf16:memory::data_type::f16,
               memory::dims{n,1}),
      scratch_bytes(pd.scratchpad_desc().get_size()) {}

  static matmul::primitive_desc make_pd(const engine& e,int m,int n,int k,
                                         int gs,bool bf16){
    const auto ft=bf16?memory::data_type::bf16:memory::data_type::f16;
    // wei {k,n} with stride {1,k} reads GRIMOIRE's [N][K] row-major payload
    // in place -- the same transposed-view trick the MXFP4 plan uses.
    memory::desc src({m,k},ft,memory::dims{k,1});
    memory::desc wei({k,n},memory::data_type::s4,memory::dims{1,k});
    memory::desc dst({m,n},ft,memory::dims{n,1});
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    attr.set_scales(DNNL_ARG_WEIGHTS,3,{gs,1},ft);
    attr.set_fpmath_mode(bf16?fpmath_mode::bf16:fpmath_mode::f16,true);
    return matmul::primitive_desc(e,src,wei,dst,attr);
  }

  memory wrap(const memory::desc& md,void* p){
    return sycl_interop::make_memory(md,eng,sycl_interop::memory_kind::usm,p);
  }
  void run(void* a,void* b,void* s,void* out,void* scratch){
    if(!bound){
      ma=wrap(pd.src_desc(),a);mw=wrap(pd.weights_desc(),b);mo=wrap(pd.dst_desc(),out);
      ms=wrap(scale_md,s);mx=wrap(pd.scratchpad_desc(),scratch);
      bound=true;
    }else{
      ma.set_data_handle(a);mw.set_data_handle(b);ms.set_data_handle(s);
      mo.set_data_handle(out);mx.set_data_handle(scratch);
    }
    prim.execute(strm,{{DNNL_ARG_SRC,ma},{DNNL_ARG_WEIGHTS,mw},{DNNL_ARG_DST,mo},
      {DNNL_ARG_ATTR_SCALES|DNNL_ARG_WEIGHTS,ms},{DNNL_ARG_SCRATCHPAD,mx}});
  }
};

extern "C" void* grimoire_onednn_s4a16_create(
    sycl::queue* q,int m,int n,int k,int gs,int bf16){
  try{return new SPlan(*q,m,n,k,gs,bf16!=0);}catch(...){return nullptr;}
}
extern "C" size_t grimoire_onednn_s4a16_scratch_size(void* p){
  return p?static_cast<SPlan*>(p)->scratch_bytes:0;
}
extern "C" void grimoire_onednn_s4a16_execute(
    void* p,const void* a,const void* b,const void* s,void* out,void* scratch){
  static_cast<SPlan*>(p)->run(const_cast<void*>(a),const_cast<void*>(b),
    const_cast<void*>(s),out,scratch);
}
extern "C" void grimoire_onednn_s4a16_destroy(void* p){delete static_cast<SPlan*>(p);}

// ---------------------------------------------------------------------
// W4A8: int8 activations x int4 weights, ported from vLLM XPU's
// csrc/xpu/onednn/int4_gemm_w4a8.h (dnnl_matmul_w4a8_int4).
//
// vLLM routes EVERY linear layer through oneDNN at every batch size.
// GRIMOIRE instead sends M=1 to its own GEMV and M<=16 verify batches to
// grimoire_xe2_dense_w4a8_f32 in libgrimoire_xe2_grouped.so, a prefill-tuned
// tile. MEASURED 2026-09-05 against vLLM 0.28 live on the same box, same
// model, 21-token context, ~2.5 committed tokens/round on BOTH engines:
//     GRIMOIRE round 73 ms  (verify 63.9, draft 9.4)  -> ~227 GB/s
//     vLLM     round 29 ms                            -> ~500 GB/s
// Acceptance is equal; the entire gap is the verify forward.
//
// The oneDNN attribute setup below is copied from that file unchanged:
//   src scales      mask (1<<0)+(1<<1), groups {1, k}
//   src zero points mask (1<<0)+(1<<1), groups {1, k}, s32
//   weight scales   mask (1<<0)+(1<<1), groups {group_size, 1}
//   scratchpad      user mode
//
// ONE DELIBERATE DEVIATION, and it is required: vLLM stores int4 as u4 and
// subtracts a zero point of 8. GRIMOIRE's W4A8 payload is SIGN-EXTENDED
// (launch_gemv_int4sym decodes int8_t(b<<4)>>4), so declaring it u4 with
// zp=8 rotates every weight by 8 -- the exact bug that made
// GRIMOIRE_ONEDNN_I4 emit fluent nonsense before commit 160c1c0. It is
// therefore declared s4 with no weight zero point, which is the same
// arithmetic on GRIMOIRE's encoding.
//
// wei {k,n} with stride {1,k} reads GRIMOIRE's native [N][K] row-major
// payload in place, so no weight repacking is needed.
struct W4A8Plan {
  engine eng;
  stream strm;
  matmul::primitive_desc pd;
  matmul prim;
  memory::desc a_sc_md;   // src scales  [m]
  memory::desc w_sc_md;   // weight scales [k/gs, n]
  size_t scratch_bytes;
  memory ma, mw, masc, mwsc, mo, mx;
  bool bound=false;

  W4A8Plan(sycl::queue& q,int m,int n,int k,int gs)
    : eng(sycl_interop::make_engine(q.get_device(),q.get_context())),
      strm(sycl_interop::make_stream(eng,q)),
      pd(make_pd(eng,m,n,k,gs)),prim(pd),
      a_sc_md({m,1},memory::data_type::f32,memory::dims{1,1}),
      w_sc_md({k/gs,n},memory::data_type::f32,memory::dims{n,1}),
      scratch_bytes(pd.scratchpad_desc().get_size()) {}

  static matmul::primitive_desc make_pd(const engine& e,int m,int n,int k,int gs){
    memory::desc src({m,k},memory::data_type::s8,memory::dims{k,1});
    memory::desc wei({k,n},memory::data_type::s4,memory::dims{1,k});
    memory::desc dst({m,n},memory::data_type::f32,memory::dims{n,1});
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    // src scales: per-row, one scale spanning all of k -- vLLM's
    // m1_sc.numel() == m branch.
    attr.set_scales(DNNL_ARG_SRC,(1<<0)+(1<<1),{1,k},memory::data_type::f32);
    // weight scales: grouped along k, exactly vLLM's {group_size, 1}.
    attr.set_scales(DNNL_ARG_WEIGHTS,(1<<0)+(1<<1),{gs,1},memory::data_type::f32);
    return matmul::primitive_desc(e,src,wei,dst,attr);
  }

  memory wrap(const memory::desc& md,void* p){
    return sycl_interop::make_memory(md,eng,sycl_interop::memory_kind::usm,p);
  }
  void run(void* a,void* w,void* a_sc,void* w_sc,void* out,void* scratch){
    if(!bound){
      ma=wrap(pd.src_desc(),a); mw=wrap(pd.weights_desc(),w);
      mo=wrap(pd.dst_desc(),out); masc=wrap(a_sc_md,a_sc);
      mwsc=wrap(w_sc_md,w_sc); mx=wrap(pd.scratchpad_desc(),scratch);
      bound=true;
    }else{
      ma.set_data_handle(a); mw.set_data_handle(w); mo.set_data_handle(out);
      masc.set_data_handle(a_sc); mwsc.set_data_handle(w_sc);
      mx.set_data_handle(scratch);
    }
    prim.execute(strm,{{DNNL_ARG_SRC,ma},{DNNL_ARG_WEIGHTS,mw},{DNNL_ARG_DST,mo},
      {DNNL_ARG_ATTR_SCALES|DNNL_ARG_SRC,masc},
      {DNNL_ARG_ATTR_SCALES|DNNL_ARG_WEIGHTS,mwsc},
      {DNNL_ARG_SCRATCHPAD,mx}});
  }
};

extern "C" void* grimoire_onednn_w4a8_create(
    sycl::queue* q,int m,int n,int k,int gs){
  try{return new W4A8Plan(*q,m,n,k,gs);}catch(...){return nullptr;}
}
extern "C" size_t grimoire_onednn_w4a8_scratch_size(void* p){
  return p?static_cast<W4A8Plan*>(p)->scratch_bytes:0;
}
extern "C" void grimoire_onednn_w4a8_execute(
    void* p,const void* a,const void* w,const void* a_sc,const void* w_sc,
    void* out,void* scratch){
  static_cast<W4A8Plan*>(p)->run(const_cast<void*>(a),const_cast<void*>(w),
    const_cast<void*>(a_sc),const_cast<void*>(w_sc),out,scratch);
}
extern "C" void grimoire_onednn_w4a8_destroy(void* p){delete static_cast<W4A8Plan*>(p);}

extern "C" void* grimoire_onednn_w4a16_create(
    sycl::queue* q,int m,int n,int k,int gs,int bf16){
  try{return new Plan(*q,m,n,k,gs,bf16!=0);}catch(...){return nullptr;}
}
extern "C" size_t grimoire_onednn_w4a16_scratch_size(void* p){
  return p?static_cast<Plan*>(p)->scratch_bytes:0;
}
extern "C" void grimoire_onednn_w4a16_execute(
    void* p,const void* a,const void* b,const void* s,const int8_t* zp,
    void* out,void* scratch){
  static_cast<Plan*>(p)->run(const_cast<void*>(a),const_cast<void*>(b),
    const_cast<void*>(s),const_cast<int8_t*>(zp),out,scratch);
}
extern "C" void grimoire_onednn_w4a16_destroy(void* p){delete static_cast<Plan*>(p);}
extern "C" void* grimoire_onednn_mxfp4_w4a16_create(
    sycl::queue* q,int m,int n,int k){
  try{return new MXPlan(*q,m,n,k);}catch(...){return nullptr;}
}
extern "C" size_t grimoire_onednn_mxfp4_w4a16_scratch_size(void* p){
  return p?static_cast<MXPlan*>(p)->scratch_bytes:0;
}
extern "C" void grimoire_onednn_mxfp4_w4a16_execute(
    void* p,const void* a,const void* b,const void* s,void* out,void* scratch){
  static_cast<MXPlan*>(p)->run(const_cast<void*>(a),const_cast<void*>(b),
    const_cast<void*>(s),out,scratch);
}
extern "C" void grimoire_onednn_mxfp4_w4a16_destroy(void* p){
  delete static_cast<MXPlan*>(p);
}
extern "C" void* grimoire_onednn_bf16_f32_create(
    sycl::queue* q,int m,int n,int k){
  try{return new BFPlan(*q,m,n,k);}catch(...){return nullptr;}
}
extern "C" size_t grimoire_onednn_bf16_f32_scratch_size(void* p){
  return p?static_cast<BFPlan*>(p)->scratch_bytes:0;
}
extern "C" void grimoire_onednn_bf16_f32_execute(
    void* p,const void* a,const void* b,void* out,void* scratch){
  static_cast<BFPlan*>(p)->run(const_cast<void*>(a),const_cast<void*>(b),
    out,scratch);
}
extern "C" void grimoire_onednn_bf16_f32_destroy(void* p){
  delete static_cast<BFPlan*>(p);
}
extern "C" void* grimoire_onednn_f16_create(
    sycl::queue* q,int m,int n,int k){
  try{return new F16Plan(*q,m,n,k);}catch(...){return nullptr;}
}
extern "C" size_t grimoire_onednn_f16_scratch_size(void* p){
  return p?static_cast<F16Plan*>(p)->scratch_bytes:0;
}
extern "C" void grimoire_onednn_f16_execute(
    void* p,const void* a,const void* b,void* out,void* scratch){
  static_cast<F16Plan*>(p)->run(const_cast<void*>(a),const_cast<void*>(b),
    out,scratch);
}
extern "C" void grimoire_onednn_f16_destroy(void* p){
  delete static_cast<F16Plan*>(p);
}
