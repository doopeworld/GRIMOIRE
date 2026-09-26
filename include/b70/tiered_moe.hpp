// =====================================================================
//  tiered_moe.hpp -- routed experts that live in VRAM OR in system RAM.
//
//  A model whose experts do not fit in VRAM (Qwen3.8-Flash-Next: 68 GB of
//  NVFP4 experts against a 32 GB card) keeps the hottest experts in VRAM
//  and the rest in pinned host memory.  Every expert is one self-contained
//  block, and a per-layer table of block pointers is all a kernel needs:
//  a pointer into VRAM or a USM host pointer read over PCIe look the same
//  to the kernel, so one set of kernels serves every tier and moving an
//  expert between tiers is one block copy plus one pointer store.
//
//  The block is NVFP4 exactly as the checkpoint stores it -- E2M1 nibbles,
//  one E4M3 scale per 16 elements, one F32 scale per projection -- so no
//  weight is ever re-quantized.  Measured links (tools/tier_probe.cpp):
//  VRAM 600 GB/s, PCIe 12.4-12.8 GB/s, CPU DRAM 67 GB/s.  Design and the
//  numbers behind it: FLASH-NEXT-TIERED.md.
// =====================================================================
#ifndef B70_TIERED_MOE_HPP
#define B70_TIERED_MOE_HPP

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace b70 {

// Byte offsets inside one expert block.  gate rows [0, I) and up rows
// [I, 2I) share one [2I][H] payload so gate_up is a single matrix, as in
// the VRAM-only MoE.  The last 16 bytes before padding hold the three
// per-projection scales {gate, up, down, 0} as MULTIPLIERS (a
// compressed-tensors global scale is stored inverted).
struct NvExpertLayout {
    int    H = 0, I = 0;
    size_t gu_p = 0, gu_s = 0, dn_p = 0, dn_s = 0, gsc = 0, bytes = 0;
    static NvExpertLayout make(int H, int I) {
        NvExpertLayout L; L.H = H; L.I = I;
        size_t o = 0;
        L.gu_p = o; o += size_t(2) * I * (H / 2);
        L.gu_s = o; o += size_t(2) * I * (H / 16);
        L.dn_p = o; o += size_t(H) * (I / 2);
        L.dn_s = o; o += size_t(H) * (I / 16);
        L.gsc  = o; o += 16;
        L.bytes = (o + 4095) & ~size_t(4095);     // page aligned: O_DIRECT-able
        return L;
    }
    // Every offset the kernels load from with 16-byte (payload) and 2-byte
    // (scale pair) vectors must stay aligned; H and I multiples of 128 give
    // that and are what the prefill dequant needs anyway.
    bool supported() const { return H % 128 == 0 && I % 128 == 0; }
};

// What the kernels take for one layer.
struct TieredMoeView {
    NvExpertLayout lay;
    const uint8_t* const* eptr = nullptr;   // device array [E] of block pointers
    int E = 0, top_k = 0;
};

// Decode (and small-M) path: two launches per layer, like launch_moe_*.
//   h[M][k][I] = silu(x W_gate^T) * (x W_up^T)          for the routed k
//   y[M][H]    = sum_slot weight[slot] * (h[slot] W_down^T)
sycl::event launch_tmoe_gate_up(sycl::queue& q, const TieredMoeView& v,
                                const int32_t* d_expert, const float* x, float* h,
                                int M, const std::vector<sycl::event>& deps);
sycl::event launch_tmoe_down(sycl::queue& q, const TieredMoeView& v,
                             const int32_t* d_expert, const float* d_weight,
                             const float* h, float* y, int M,
                             const std::vector<sycl::event>& deps);

// Prefill path, one expert: out[M][N] fp32 = A[M][K] (bf16 rows) x W^T,
// W = the expert's gate_up (N = 2I, K = H; columns [0,I) scaled by s0 and
// [I,2I) by s1) or down (N = H, K = I; scaled by s0).  The NVFP4 block is
// decoded into `scratch` (>= N*K bf16) first; `block` may be VRAM or host.
// Defined in gemm_fast.cpp (256-GRF library).
sycl::event launch_nvfp4_expert_gemm(sycl::queue& q, const uint8_t* block,
                                     const NvExpertLayout& L, bool gate_up,
                                     float s0, float s1,
                                     const sycl::ext::oneapi::bfloat16* A, float* out,
                                     int M, sycl::ext::oneapi::bfloat16* scratch,
                                     const std::vector<sycl::event>& deps);

} // namespace b70
#endif
