#pragma once
#include "b70/qwen35.hpp"
#include <array>

namespace b70 {
inline TensorRef fused_expert_slice(const TensorRef& src,uint64_t elem0,int n,int k,const char* suffix) {
    if(!src.ok() || n<=0 || k<=0 || src.t.shape.size()!=3 || src.t.shape.back()!=k ||
       elem0%uint64_t(k) || src.t.numel()<=0 || elem0>uint64_t(src.t.numel()) ||
       uint64_t(n)*k>uint64_t(src.t.numel())-elem0)return {};
    TensorRef r=src;r.t.name+=suffix;
    if(src.native) {
        NativeLayout l;std::string err;
        if(!native_layout(*src.native,l,err)||l.K!=k)return {};
        const uint64_t row=elem0/uint64_t(k);
        r.native_payload_offset+=row*l.row_payload_bytes;
        r.native_scale_offset+=row*l.row_scale_bytes;
        r.t.begin=0;r.t.end=uint64_t(n)*l.row_payload_bytes;
    } else {
        if(src.t.dtype!=STDtype::BF16 && src.t.dtype!=STDtype::F16 && src.t.dtype!=STDtype::F32)return {};
        const uint64_t bytes=uint64_t(st_dtype_size(src.t.dtype));
        if(elem0>UINT64_MAX/bytes || src.t.begin>UINT64_MAX-elem0*bytes)return {};
        r.t.begin=src.t.begin+elem0*bytes;
        if(uint64_t(n)*k>UINT64_MAX/bytes)return {};
        const uint64_t count=uint64_t(n)*k*bytes;
        if(r.t.begin>src.t.end || count>src.t.end-r.t.begin)return {};
        r.t.end=r.t.begin+count;
    }
    r.t.shape={n,k};return r;
}

inline TensorRef tensor_ref(const Qwen35Model& model,const std::string& name) {
    auto it=model.index.find(name);return it==model.index.end()?TensorRef{}:it->second;
}
// Used by the MTP loader; the same slice function serves the target loader.
inline std::array<TensorRef,3> resolve_expert(const Qwen35Model& m,const std::string& prefix,
                                           int e,int experts,int inter,int hidden) {
    std::array<TensorRef,3> r;
    if(e<0||e>=experts||inter<=0||hidden<=0)return r;
    const std::string individual=prefix+std::to_string(e)+".";
    r[0]=tensor_ref(m,individual+"gate_proj.weight");
    r[1]=tensor_ref(m,individual+"up_proj.weight");
    r[2]=tensor_ref(m,individual+"down_proj.weight");
    auto gu=tensor_ref(m,prefix+"gate_up_proj");if(!gu.ok())gu=tensor_ref(m,prefix+"gate_up_proj.weight");
    auto dn=tensor_ref(m,prefix+"down_proj");if(!dn.ok())dn=tensor_ref(m,prefix+"down_proj.weight");
    if(gu.ok()&&gu.t.shape!=std::vector<int64_t>{experts,2LL*inter,hidden})return {};
    if(dn.ok()&&dn.t.shape!=std::vector<int64_t>{experts,hidden,inter})return {};
    if(!r[0].ok())r[0]=fused_expert_slice(gu,uint64_t(e)*2*inter*hidden,inter,hidden,".gate");
    if(!r[1].ok())r[1]=fused_expert_slice(gu,(uint64_t(e)*2+1)*inter*hidden,inter,hidden,".up");
    if(!r[2].ok())r[2]=fused_expert_slice(dn,uint64_t(e)*hidden*inter,hidden,inter,".down");
    for(int i=0;i<3;++i)
        if(!r[i].ok() || r[i].t.shape!=std::vector<int64_t>{i==2?hidden:inter,i==2?inter:hidden})return {};
    return r;
}
} // namespace b70
