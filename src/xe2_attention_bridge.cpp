#include "xe2_attention_bridge.hpp"
#include <torch/all.h>
#include <c10/xpu/XPUStream.h>
#include <optional>

void cutlass_chunk_prefill_xe2(
    sycl::queue&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int, int, std::optional<const at::Tensor>&, std::optional<const at::Tensor>&,
    double, std::optional<const at::Tensor>&, int, int, bool, bool, bool, bool,
    bool, std::optional<at::Tensor>&, std::optional<const at::Tensor>&);

void chunk_gated_delta_rule_xe2(
    sycl::queue&, torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&,
    const std::optional<torch::Tensor>&, int, int, const int*);

extern "C" void grimoire_xe2_chunk_prefill_bf16(
    sycl::queue* queue, const void* query, const void* key, const void* value,
    void* output, int q_tokens, int kv_tokens, int q_heads, int kv_heads,
    int head_dim, const int* cu_q, const int* cu_k, float softmax_scale,
    bool causal) {
    auto bf=at::TensorOptions().dtype(at::kBFloat16).device(at::kXPU);
    auto ii=at::TensorOptions().dtype(at::kInt).device(at::kXPU);
    auto q=at::from_blob(const_cast<void*>(query),{q_tokens,q_heads,head_dim},bf);
    auto k=at::from_blob(const_cast<void*>(key),{kv_tokens,kv_heads,head_dim},bf);
    auto v=at::from_blob(const_cast<void*>(value),{kv_tokens,kv_heads,head_dim},bf);
    auto o=at::from_blob(output,{q_tokens,q_heads,head_dim},bf);
    auto cq=at::from_blob(const_cast<int*>(cu_q),{2},ii);
    auto ck=at::from_blob(const_cast<int*>(cu_k),{2},ii);
    auto empty=at::empty({0},ii);
    std::optional<const at::Tensor> none_const;
    std::optional<at::Tensor> none_out;
    cutlass_chunk_prefill_xe2(*queue,q,k,v,o,empty,cq,ck,q_tokens,kv_tokens,
        none_const,none_const,double(softmax_scale),none_const,-1,-1,true,false,
        causal,false,false,none_out,none_const);
}

extern "C" void grimoire_xe2_chunk_gdn_bf16(
    sycl::queue* queue, void* output, const void* query, const void* key,
    const void* value, const float* beta, const float* gate_a,
    const float* a_log, const void* dt_bias, float* state, int tokens,
    int k_heads, int k_dim, int v_heads, int v_dim, const int* cu_seqlens,
    const int* cache_index, const bool* has_initial_state) {
    auto bf=at::TensorOptions().dtype(at::kBFloat16).device(at::kXPU);
    auto ff=at::TensorOptions().dtype(at::kFloat).device(at::kXPU);
    auto ii=at::TensorOptions().dtype(at::kInt).device(at::kXPU);
    auto bb=at::TensorOptions().dtype(at::kBool).device(at::kXPU);
    auto o=at::from_blob(output,{tokens,v_heads,v_dim},bf);
    auto q=at::from_blob(const_cast<void*>(query),{tokens,k_heads,k_dim},bf);
    auto k=at::from_blob(const_cast<void*>(key),{tokens,k_heads,k_dim},bf);
    auto v=at::from_blob(const_cast<void*>(value),{tokens,v_heads,v_dim},bf);
    auto b=at::from_blob(const_cast<float*>(beta),{v_heads,tokens},ff);
    auto a=at::from_blob(const_cast<float*>(gate_a),{v_heads,tokens},ff);
    auto al=at::from_blob(const_cast<float*>(a_log),{v_heads},ff);
    auto db=at::from_blob(const_cast<void*>(dt_bias),{v_heads},bf);
    auto st=at::from_blob(state,{1,v_heads,v_dim,k_dim},ff);
    auto cu=at::from_blob(const_cast<int*>(cu_seqlens),{2},ii);
    auto ci=at::from_blob(const_cast<int*>(cache_index),{1},ii);
    auto hs=at::from_blob(const_cast<bool*>(has_initial_state),{1},bb);
    std::optional<torch::Tensor> has{hs};
    // The native implementation allocates and zeroes three scratch tensors
    // through PyTorch's current XPU stream, then launches compute on the raw
    // queue supplied here. Make those initializations visible first.
    queue->wait();
    auto& torch_queue=c10::xpu::getCurrentXPUStream().queue();
    chunk_gated_delta_rule_xe2(torch_queue,o,q,k,v,b,a,al,db,st,cu,ci,has,1,0,nullptr);
    torch_queue.wait();
}
