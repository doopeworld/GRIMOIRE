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

extern "C" int grimoire_xe2_dflash_paged_f16(
    sycl::queue* q, const void* query, const void* key_cache,
    const void* value_cache, void* output, int q_tokens, int kv_tokens,
    int q_heads, int kv_heads, int head_dim, int block_size, int num_blocks,
    const int* block_table, const int* cu_q, const int* dummy_cu_k,
    const int* seqused_k, float softmax_scale, int window_left,
    int window_right) {
  if(head_dim!=128||block_size!=64||!q||!query||!key_cache||!value_cache||
     !output||!block_table||!cu_q||!dummy_cu_k||!seqused_k)return 1;
  chunk_prefill_args_t a{};
  a.query=const_cast<void*>(query);
  a.key=const_cast<void*>(key_cache);
  a.value=const_cast<void*>(value_cache);
  a.out=output;
  a.block_table=const_cast<int*>(block_table);
  a.cu_seqlens_q=const_cast<int*>(cu_q);
  // flash_api.cpp replaces the dummy cumulative-K tensor with seqused_k for
  // paged attention before it enters cutlass_chunk_prefill_interface.
  a.cu_seqlens_k=const_cast<int*>(seqused_k);
  a.max_queries=q_tokens;
  a.max_keys=kv_tokens;
  a.total_seqlen_q=q_tokens;
  a.total_seqlen_k=num_blocks*block_size;
  a.sm_scale=softmax_scale;
  a.batch_size=1;
  a.num_heads_q=q_heads;
  a.num_heads_k=kv_heads;
  a.head_size=head_dim;
  a.max_blocks_per_seq=num_blocks;
  a.block_size=block_size;
  a.window_size_left=window_left;
  a.window_size_right=window_right;
  a.is_varlen=true;
  a.is_paged=true;
  a.is_causal=false;
  a.is_local=window_left>=0||window_right>=0;
  a.q_stride_seq=q_heads*head_dim;
  a.q_stride_heads=head_dim;
  a.k_stride_seq=kv_heads*head_dim;
  a.k_stride_heads=head_dim;
  a.v_stride_seq=kv_heads*head_dim;
  a.v_stride_heads=head_dim;
  a.o_stride_seq=q_heads*head_dim;
  a.o_stride_heads=head_dim;
  a.page_stride_elements=block_size*kv_heads*head_dim;
  CutlassQKType types{CutlassDType::half};
  policy_dispatch_impl<chunk_policy_head128,true,false,true,false,false>(
      *q,types,a);
  return 0;
}
