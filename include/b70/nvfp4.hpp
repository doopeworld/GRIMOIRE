// =====================================================================
//  nvfp4.hpp -- host reference for reading an NVFP4 checkpoint.
//
//  NVFP4 is NVIDIA's Blackwell 4-bit format.  It is NOT MXFP4, and the
//  two are close enough that mistaking one for the other is silent:
//
//                  element   block   scale dtype     extra
//    MXFP4         E2M1      32      E8M0 (pow2)     --
//    NVFP4         E2M1      16      E4M3            one FP32 per TENSOR
//
//  Same nibble payload, same [N][K/2] packing.  What differs is the
//  block WIDTH and the scale's DTYPE, plus a global scale that has no
//  counterpart in MXFP4 at all.  A .weight_packed tensor therefore tells
//  you nothing about which format you are holding -- only the presence
//  of .weight_global_scale does.
//
//  This engine reads NVFP4 by DEQUANTIZING it and letting the existing
//  quantizer re-pack to whatever --proj asks for.  There is no NVFP4
//  GEMM here and there does not need to be: the step is the same one
//  AMD's ROCm blog does for CDNA4 ("NVFP4 to MXFP4 online
//  requantization") and GGZ14/vllm-mxfp4 does for RDNA4, except that
//  going through f32 means the destination can be mxfp4, int4, fp8 or
//  anything else this engine packs, not one hardcoded pair.
//
//  IT IS LOSSY, and in a direction worth stating: 32-wide power-of-two
//  blocks are coarser than 16-wide E4M3 ones, so requantizing to MXFP4
//  gives up both resolution (E8M0 can only hold a power of two) and
//  locality (one scale covers twice as many elements).  The reference
//  implementations gate this on an end task -- GSM8K 97.40 requantized
//  against 97.8 for a natively-MXFP4 checkpoint -- NOT on kernel parity,
//  because a kernel-parity check against a reference that requantizes
//  the same way cannot see the loss at all.
//
//  THE TRAP, and it is the expensive one: A MERGED LINEAR HAS ONE GLOBAL
//  SCALE PER PARTITION.  `gate_up_proj` is two tensors stored as one and
//  carries TWO global scales; vLLM's stock path collapses them with
//  .max(), which the vllm-mxfp4 authors call out as a real accuracy
//  loss.  This engine concatenates gate|up into one weight at load
//  (concat_upload_t -> sh_gu), so it would make exactly that mistake --
//  unless each side is dequantized with ITS OWN global scale BEFORE the
//  concatenation.  Dequantizing per TensorRef, which is what the reader
//  below does, is what makes that automatic rather than careful.
// =====================================================================
#ifndef B70_NVFP4_HPP
#define B70_NVFP4_HPP

#include "b70/formats.hpp"

#include <cstdint>
#include <cstddef>

namespace b70 {

// NVFP4's block width.  Half MXFP4's, which is the whole reason a
// requantization loses locality.
constexpr int kNVFP4Block = 16;

// One row of an NVFP4 weight -> f32.
//
//   packed  [K/2]        E2M1 nibbles, LOW nibble is element 2i
//   scales  [K/16]       E4M3, one per 16 consecutive elements
//   gscale               the tensor's single FP32 scale, AS STORED in the
//                        checkpoint -- read raw, no inversion applied
//                        before it reaches here
//
// The low-nibble-first order is compressed-tensors' own
// (break_fp4_bytes); getting it backwards transposes every pair of
// weights inside a block and still produces a model that runs.
//
// DIVIDE by gscale, not multiply (fixed 2026-09-21, external audit).
// compressed-tensors' own dequantizer -- the format this reader targets
// -- computes `scale = local_scale / global_scale` and passes the stored
// weight_global_scale straight through with no inversion first
// (compressors/nvfp4/base.py -> quantization/lifecycle/forward_helpers.py,
// revision 525a7a7b84ebccdbc0db07956338b6f8c851d58f). Verified against
// that source directly, not inferred. Multiplying instead is not a
// nearby approximation: the error is `global_scale^2` on every weight,
// silent, and the checkpoint still loads and generates fluent text.
//
// It survived because the "independent" host reference this file's own
// gate compared against used the SAME wrong direction to build its
// expected value -- an equality test between two copies of one mistake
// proves the two copies agree, nothing about the checkpoint's actual
// convention. See rule 18: a format that shares another's on-disk shape
// still needs the discriminator checked against an OUTSIDE source, not
// a reference derived from the same misreading.
//
// mul: modelopt's convention (weight_scale_2), which MULTIPLIES --
// NVIDIA ModelOpt / TensorRT-LLM / vLLM modelopt all compute
// w = e2m1 * weight_scale * weight_scale_2.  Same nibbles, same E4M3
// block scales, opposite use of the per-tensor scalar.
inline void nvfp4_dequant_row(const uint8_t* packed, const uint8_t* scales,
                              float gscale, int K, float* out, bool mul = false) {
    for (int k = 0; k < K; ++k) {
        const uint8_t byte = packed[k >> 1];
        const uint8_t nib  = (k & 1) ? uint8_t(byte >> 4) : uint8_t(byte & 0x0F);
        const float   blk  = e4m3_to_f32(scales[k / kNVFP4Block]);
        out[k] = mul ? e2m1_to_f32(nib) * blk * gscale
                     : e2m1_to_f32(nib) * blk / gscale;
    }
}

// The whole matrix, row-major [N][K].
inline void nvfp4_dequant(const uint8_t* packed, const uint8_t* scales,
                          float gscale, int N, int K, float* out, bool mul = false) {
    for (int n = 0; n < N; ++n)
        nvfp4_dequant_row(packed + size_t(n) * (K / 2),
                          scales + size_t(n) * (K / kNVFP4Block),
                          gscale, K, out + size_t(n) * K, mul);
}

} // namespace b70
#endif
