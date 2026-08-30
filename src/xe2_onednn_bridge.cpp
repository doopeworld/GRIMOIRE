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
    attr.set_fpmath_mode(fpmath_mode::f16,true);
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
