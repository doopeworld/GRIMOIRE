// =====================================================================
//  quantize.cpp  --  offline fp32 -> {bf16,fp8,int8,int4,mxfp8,mxfp4}
//
//  Runs on the host, once, when the model is converted. Nothing here is
//  on the inference critical path.
// =====================================================================
#include "b70/weights.hpp"
#include <cmath>
#include <algorithm>

namespace b70 {
namespace {

inline float absmax(const float* p, int n, int stride = 1) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(p[i * stride]));
    return m;
}

// OCP MX scale selection: place the shared exponent so the block maximum
// lands at the top of the element format's range.
//   X = clamp(floor(log2(amax)) - emax_elem + 127, 0, 254)
uint8_t pick_e8m0(float amax, int emax_elem) {
    if (!(amax > 0.0f) || !std::isfinite(amax)) return 127;   // 2^0
    int e = int(std::floor(std::log2(amax))) - emax_elem;
    e += 127;
    if (e < 0)   e = 0;
    if (e > 254) e = 254;
    return uint8_t(e);
}

} // namespace

PackedWeight quantize(const float* src, int N, int K, Fmt fmt) {
    PackedWeight w;
    w.fmt = fmt; w.N = N; w.K = K;
    w.row_bytes  = bytes_per_row(fmt, K);
    w.row_scales = scales_per_row(fmt, K);
    w.payload.assign(size_t(N) * size_t(w.row_bytes), 0);

    switch (fmt) {

    // -----------------------------------------------------------------
    case Fmt::BF16: {
        for (int n = 0; n < N; ++n) {
            uint8_t* row = w.payload.data() + int64_t(n) * w.row_bytes;
            for (int k = 0; k < K; ++k) {
                bf16_t b = f32_to_bf16(src[int64_t(n) * K + k]);
                row[2 * k]     = uint8_t(b.bits & 0xFF);
                row[2 * k + 1] = uint8_t(b.bits >> 8);
            }
        }
        break;
    }

    // -----------------------------------------------------------------
    // Per-output-channel symmetric scaling. One fp32 per row.
    case Fmt::INT8:
    case Fmt::FP8_E4M3:
    case Fmt::FP8_E5M2: {
        w.scales_raw.assign(size_t(N) * sizeof(float), 0);
        float* sc = reinterpret_cast<float*>(w.scales_raw.data());
        const float target = (fmt == Fmt::INT8)     ? 127.0f
                           : (fmt == Fmt::FP8_E4M3) ? 448.0f
                                                    : 57344.0f;
        for (int n = 0; n < N; ++n) {
            const float* in  = src + int64_t(n) * K;
            uint8_t*     row = w.payload.data() + int64_t(n) * w.row_bytes;
            const float  am  = absmax(in, K);
            const float  s   = (am > 0.0f) ? am / target : 1.0f;
            sc[n] = s;
            const float inv = 1.0f / s;
            for (int k = 0; k < K; ++k) {
                const float v = in[k] * inv;
                if (fmt == Fmt::INT8) {
                    int q = int(std::nearbyint(v));
                    q = std::max(-127, std::min(127, q));
                    row[k] = uint8_t(int8_t(q));
                } else if (fmt == Fmt::FP8_E4M3) {
                    row[k] = f32_to_e4m3(v);
                } else {
                    row[k] = f32_to_e5m2(v);
                }
            }
        }
        break;
    }

    // -----------------------------------------------------------------
    // Asymmetric 4-bit, group of 128 along K. bf16 scale + uint8 zero.
    // This is the GPTQ/AWQ convention, so converted checkpoints from the
    // usual toolchains drop straight in.
    case Fmt::INT4: {
        const int G = w.row_scales;
        w.scales_raw.assign(size_t(N) * G * sizeof(bf16_t), 0);
        w.zeros.assign(size_t(N) * G, 0);
        bf16_t* sc = reinterpret_cast<bf16_t*>(w.scales_raw.data());
        for (int n = 0; n < N; ++n) {
            const float* in  = src + int64_t(n) * K;
            uint8_t*     row = w.payload.data() + int64_t(n) * w.row_bytes;
            for (int g = 0; g < G; ++g) {
                const int base = g * kInt4Group;
                float lo = in[base], hi = in[base];
                for (int i = 1; i < kInt4Group; ++i) {
                    lo = std::min(lo, in[base + i]);
                    hi = std::max(hi, in[base + i]);
                }
                float s = (hi - lo) / 15.0f;
                if (!(s > 0.0f)) s = 1.0f;
                // Round the scale to bf16 *before* choosing the codes, so
                // the packer quantizes against the value the kernel will
                // actually see. Skipping this is a classic silent 2x
                // error amplifier at group boundaries.
                const bf16_t sb = f32_to_bf16(s);
                s = bf16_to_f32(sb);
                int z = int(std::nearbyint(-lo / s));
                z = std::max(0, std::min(15, z));
                sc[int64_t(n) * G + g] = sb;
                w.zeros[int64_t(n) * G + g] = uint8_t(z);
                for (int i = 0; i < kInt4Group; ++i) {
                    int q = int(std::nearbyint(in[base + i] / s)) + z;
                    q = std::max(0, std::min(15, q));
                    const int k = base + i;
                    if (k & 1) row[k >> 1] = uint8_t((row[k >> 1] & 0x0F) | (q << 4));
                    else       row[k >> 1] = uint8_t((row[k >> 1] & 0xF0) | q);
                }
            }
        }
        break;
    }

    // -----------------------------------------------------------------
    // OCP Microscaling. Block of 32 along K, one E8M0 byte per block.
    case Fmt::MXFP8:
    case Fmt::MXFP4: {
        const int B      = w.row_scales;
        const int emax   = (fmt == Fmt::MXFP8) ? kEmaxE4M3 : kEmaxE2M1;
        w.scales_raw.assign(size_t(N) * B, 0);
        for (int n = 0; n < N; ++n) {
            const float* in  = src + int64_t(n) * K;
            uint8_t*     row = w.payload.data() + int64_t(n) * w.row_bytes;
            for (int b = 0; b < B; ++b) {
                const int   base = b * kMXBlock;
                const float am   = absmax(in + base, kMXBlock);
                const uint8_t X  = pick_e8m0(am, emax);
                w.scales_raw[int64_t(n) * B + b] = X;
                const float inv = 1.0f / e8m0_to_f32(X);
                for (int i = 0; i < kMXBlock; ++i) {
                    const float v = in[base + i] * inv;
                    const int   k = base + i;
                    if (fmt == Fmt::MXFP8) {
                        row[k] = f32_to_e4m3(v);
                    } else {
                        const uint8_t q = f32_to_e2m1(v);
                        if (k & 1) row[k >> 1] = uint8_t((row[k >> 1] & 0x0F) | (q << 4));
                        else       row[k >> 1] = uint8_t((row[k >> 1] & 0xF0) | q);
                    }
                }
            }
        }
        break;
    }
    }

    return w;
}

} // namespace b70
