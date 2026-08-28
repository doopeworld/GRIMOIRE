// =====================================================================
//  b70/gptq.hpp  --  GPTQ / AutoGPTQ 4-bit checkpoint decode
//
//  GPTQ does not ship a weight tensor. Each projection is a triple:
//
//    .qweight   I32  [in/8][out]      8 nibbles packed along IN
//    .qzeros    I32  [in/group][out/8]  8 nibbles packed along OUT
//    .scales    F16  [in/group][out]
//    .g_idx     I32  [in]             input->group map; identity when
//                                     desc_act=false, then ignorable
//
//  Note the two packing axes differ. qweight packs along the INPUT
//  dimension, qzeros along the OUTPUT dimension. Getting this backwards
//  produces values that look plausible for output channel 0 and are
//  wrong for every channel after it -- which is exactly the shape of bug
//  that survives a spot check.
//
//  Dequantization, AutoGPTQ convention:
//
//      w[o][k] = (nibble - (zero_nibble + 1)) * scale
//
//  The +1 on the packed zero-point is the single most common porting
//  error. It is not a rounding detail: omitting it shifts every weight
//  by one quantization step.
//
//  Shapes, worked for a real routed expert (gate_proj, out=512 in=2048,
//  group_size=128) from Qwen3.6-35B-A3B-GPTQ-Int4:
//
//      qweight I32 [256, 512]     = [in/8, out]
//      qzeros  I32 [16,  64]      = [in/group, out/8]
//      scales  F16 [16,  512]     = [in/group, out]
//
//  down_proj has the dims reversed (out=2048, in=512) -- always derive
//  in/out from the qweight shape, never from a config field.
// =====================================================================
#ifndef B70_GPTQ_HPP
#define B70_GPTQ_HPP

#include "weights.hpp"
#include <cstdint>

namespace b70 {

constexpr int kGptqGroup = 128;   // group_size; AutoGPTQ default

// A GPTQ 4-bit projection, pointing straight at mmapped checkpoint bytes.
struct GptqTensor {
    const int32_t*  qweight = nullptr;
    const int32_t*  qzeros  = nullptr;
    const uint16_t* scales  = nullptr;   // IEEE binary16
    int in    = 0;                       // input features  (K)
    int out   = 0;                       // output features (N)
    int group = kGptqGroup;

    bool ok() const {
        return qweight && qzeros && scales && in > 0 && out > 0 &&
               group > 0 && (in % group) == 0 && (in % 8) == 0 && (out % 8) == 0;
    }
};

// Single element, for tests and spot checks. w[o][k].
inline float gptq_at(const GptqTensor& t, int o, int k) {
    const int grp = k / t.group;

    // qweight [in/8][out]: 8 nibbles along k, one output channel per column
    const int32_t  wword = t.qweight[int64_t(k >> 3) * t.out + o];
    const uint32_t nib   = (uint32_t(wword) >> ((k & 7) * 4)) & 0xFu;

    // qzeros [in/group][out/8]: 8 nibbles along o
    const int32_t  zword = t.qzeros[int64_t(grp) * (t.out >> 3) + (o >> 3)];
    const uint32_t znib  = (uint32_t(zword) >> ((o & 7) * 4)) & 0xFu;

    const float scale = f16_to_f32(t.scales[int64_t(grp) * t.out + o]);
    return (float(nib) - float(znib + 1u)) * scale;
}

// Dequantize the whole projection into row-major [out][in] fp32.
// That is the layout quantize(src, N=out, K=in, fmt) expects, so the
// result feeds the existing packer unchanged -- GPTQ never needs its own
// kernel, it only needs to land in VRAM as MXFP4/INT4 like everything else.
void gptq_dequant_4bit(const GptqTensor& t, float* dst);
PackedWeight gptq_repack_int4(const GptqTensor& t);

} // namespace b70
#endif // B70_GPTQ_HPP
