// =====================================================================
//  gptq.cpp  --  GPTQ 4-bit -> fp32, host side, once at load
// =====================================================================
#include "b70/gptq.hpp"

namespace b70 {

void gptq_dequant_4bit(const GptqTensor& t, float* dst) {
    if (!t.ok() || !dst) return;

    const int   out   = t.out, in = t.in, group = t.group;
    const int   zcols = out >> 3;
    const int   ngrp  = in / group;

    // Walk group-major so the scale and zero for a whole [group] run of k
    // are resolved once, outside the inner loop, instead of per element.
    for (int o = 0; o < out; ++o) {
        const int      zsh  = (o & 7) * 4;
        const int      zcol = o >> 3;
        float* const   row  = dst + int64_t(o) * in;

        for (int g = 0; g < ngrp; ++g) {
            const uint32_t znib =
                (uint32_t(t.qzeros[int64_t(g) * zcols + zcol]) >> zsh) & 0xFu;
            const float scale = f16_to_f32(t.scales[int64_t(g) * out + o]);
            const float bias  = -float(znib + 1u) * scale;

            const int k0 = g * group, k1 = k0 + group;
            for (int k = k0; k < k1; ++k) {
                const int32_t  wword = t.qweight[int64_t(k >> 3) * out + o];
                const uint32_t nib   = (uint32_t(wword) >> ((k & 7) * 4)) & 0xFu;
                row[k] = float(nib) * scale + bias;
            }
        }
    }
}

PackedWeight gptq_repack_int4(const GptqTensor& t) {
    PackedWeight p;
    if (!t.ok()) return p;
    p.fmt = Fmt::INT4; p.N = t.out; p.K = t.in;
    p.row_bytes = bytes_per_row(Fmt::INT4, t.in);
    p.row_scales = t.in / t.group;
    p.payload.assign(size_t(t.out) * p.row_bytes, 0);
    p.scales_raw.resize(size_t(t.out) * p.row_scales * sizeof(bf16_t));
    p.zeros.resize(size_t(t.out) * p.row_scales);
    bf16_t* ds = reinterpret_cast<bf16_t*>(p.scales_raw.data());

    for (int o = 0; o < t.out; ++o) {
        uint8_t* row = p.payload.data() + int64_t(o) * p.row_bytes;
        for (int g = 0; g < p.row_scales; ++g) {
            const int zword = t.qzeros[int64_t(g) * (t.out >> 3) + (o >> 3)];
            const uint8_t z = uint8_t((uint32_t(zword) >> ((o & 7) * 4)) & 0xFu);
            // AutoGPTQ stores zero-1; runtime INT4 stores the actual zero.
            p.zeros[int64_t(o) * p.row_scales + g] = uint8_t(z + 1u);
            ds[int64_t(o) * p.row_scales + g] =
                f32_to_bf16(f16_to_f32(t.scales[int64_t(g) * t.out + o]));
        }
        for (int k = 0; k < t.in; ++k) {
            const uint32_t word = uint32_t(t.qweight[int64_t(k >> 3) * t.out + o]);
            const uint8_t q = uint8_t((word >> ((k & 7) * 4)) & 0xFu);
            if (k & 1) row[k >> 1] = uint8_t(row[k >> 1] | (q << 4));
            else       row[k >> 1] = q;
        }
    }
    return p;
}

} // namespace b70
