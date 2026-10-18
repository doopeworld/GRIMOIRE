// Shared scalar pieces of the batched Q/K norm and RoPE path.  Kept free of
// SYCL so the M>1 head layout and arithmetic can be checked on a host.
#ifndef B70_PREFILL_QK_MATH_HPP
#define B70_PREFILL_QK_MATH_HPP

#include <cstdint>

namespace b70 {
extern float g_norm_weight_offset;
inline float prefill_default_weight_offset() { return g_norm_weight_offset; }
inline int64_t prefill_head_offset(int token, int head, int heads, int dim) {
    return (int64_t(token) * heads + head) * dim;
}
inline float prefill_norm_weight(float stored_weight, float weight_offset) {
    return weight_offset + stored_weight;
}
inline void prefill_rope_pair(float a, float b, float cs, float sn,
                             float& first, float& second) {
    first = a * cs - b * sn;
    second = a * sn + b * cs;
}
} // namespace b70

#endif
