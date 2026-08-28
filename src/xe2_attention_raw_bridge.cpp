#include "xe2_attention_bridge.hpp"
#include "chunk_prefill.hpp"

extern "C" void grimoire_xe2_chunk_prefill_bf16(
    sycl::queue* q,const void* query,const void* key,const void* value,
    void* output,int qt,int kt,int qh,int kh,int hd,const int* cuq,
    const int* cuk,float scale,bool causal){
  chunk_prefill_args_t a{};
  a.query=const_cast<void*>(query);a.key=const_cast<void*>(key);
  a.value=const_cast<void*>(value);a.out=output;
  a.cu_seqlens_q=const_cast<int*>(cuq);a.cu_seqlens_k=const_cast<int*>(cuk);
  a.max_queries=qt;a.max_keys=kt;a.total_seqlen_q=qt;a.total_seqlen_k=kt;
  a.sm_scale=scale;a.batch_size=1;a.num_heads_q=qh;a.num_heads_k=kh;
  a.head_size=hd;a.is_varlen=true;a.is_causal=causal;
  a.q_stride_seq=qh*hd;a.q_stride_heads=hd;
  a.k_stride_seq=kh*hd;a.k_stride_heads=hd;
  a.v_stride_seq=kh*hd;a.v_stride_heads=hd;
  a.o_stride_seq=qh*hd;a.o_stride_heads=hd;
  CutlassQKType types{CutlassDType::bfloat16};
  // Ornith-1.5 and Qwen3.8-27B both use head_dim=128.  Compile exactly the
  // one causal kernel they need instead of carrying a generic FMHA matrix.
  if(hd==128)
    policy_dispatch_impl<chunk_policy_head128,false,true,false,false,false>(*q,types,a);
}
