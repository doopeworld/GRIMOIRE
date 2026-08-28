#pragma once
#include <sycl/sycl.hpp>

extern "C" void grimoire_xe2_grouped_w4a16(
    sycl::queue* queue, const void* activations_bf16,
    const unsigned char* weights_int4, const void* scales_bf16,
    void* outputs_bf16, int n, int k, const int* rows_per_expert,
    const int* expert_ids, int num_experts, int group_size,
    int* atomic_buffer);
