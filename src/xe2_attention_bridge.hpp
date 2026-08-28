#pragma once
#include <sycl/sycl.hpp>

extern "C" void grimoire_xe2_chunk_prefill_bf16(
    sycl::queue* queue, const void* query, const void* key, const void* value,
    void* output, int q_tokens, int kv_tokens, int q_heads, int kv_heads,
    int head_dim, const int* cu_q, const int* cu_k, float softmax_scale,
    bool causal);

extern "C" void grimoire_xe2_chunk_gdn_bf16(
    sycl::queue* queue, void* output, const void* query, const void* key,
    const void* value, const float* beta, const float* gate_a,
    const float* a_log, const void* dt_bias, float* state, int tokens,
    int k_heads, int k_dim, int v_heads, int v_dim, const int* cu_seqlens,
    const int* cache_index, const bool* has_initial_state);
