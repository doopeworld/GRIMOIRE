#include <sycl/sycl.hpp>
#include "fused_moe/common/fused_moe_gate_up_launcher.hpp"
#include "fused_moe/xe2/fused_moe_xe2_policy.h"

namespace {
class grimoire_moe_64x128 : public FusedMOE::xe_gemm_policy_base {
 public:
  using WGTile = cute::Shape<cute::_64, cute::_128, cute::_32>;
  using SGLayout = cute::Layout<cute::Shape<cute::_2, cute::_8, cute::_1>,
                                cute::Stride<cute::_8, cute::_1, cute::_0>>;
};
}

extern "C" void grimoire_xe2_fused_moe_gate_up_mxfp4_silu_bf16(
    sycl::queue* q, const void* activations, const unsigned char* weights,
    const unsigned char* scales, void* outputs, int n, int k,
    const int* rows_per_expert, int num_experts, int32_t* atomic_buffer) {
  using namespace FusedMOE;
  FusedMOEGateUpLauncher<false, ActivationType::SILU, 'R', 'C',
      grimoire_moe_64x128>(*q,
      static_cast<const cutlass::bfloat16_t*>(activations), weights, scales,
      static_cast<const cutlass::bfloat16_t*>(nullptr),
      static_cast<cutlass::bfloat16_t*>(outputs), n, k, rows_per_expert,
      num_experts, 32, atomic_buffer, 0.0);
}
