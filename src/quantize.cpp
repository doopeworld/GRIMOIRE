// =====================================================================
//  quantize.cpp  --  offline fp32 -> {bf16,fp8,int8,int4,mxfp8,mxfp4}
//
//  Runs on the host, once, when the model is converted. Nothing here is
//  on the inference critical path.
// =====================================================================
#include "b70/weights.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

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
    int e = std::ilogb(amax) - emax_elem;
    e += 127;
    if (e < 0)   e = 0;
    if (e > 254) e = 254;
    return uint8_t(e);
}

} // namespace

PackedWeight quantize(const float* src, int N, int K, Fmt fmt) {
    if (!src || N <= 0 || K <= 0 || K > std::numeric_limits<int>::max() / 2)
        throw std::invalid_argument("quantize: invalid matrix dimensions or null input");
    switch (fmt) {
    case Fmt::BF16: case Fmt::FP8_E4M3: case Fmt::FP8_E5M2:
    case Fmt::INT8: case Fmt::INT4: case Fmt::MXFP8: case Fmt::MXFP4: break;
    default: throw std::invalid_argument("quantize: unknown format");
    }
    const int block = fmt == Fmt::INT4 ? kInt4Group :
        (fmt == Fmt::MXFP4 || fmt == Fmt::MXFP8) ? kMXBlock : 1;
    if (K % block)
        throw std::invalid_argument("quantize: row width is not a whole number of groups");
    if (size_t(N) > std::numeric_limits<size_t>::max() / size_t(K) / sizeof(float))
        throw std::overflow_error("quantize: matrix size overflow");
    const size_t count = size_t(N) * size_t(K);
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(src[i]))
            throw std::invalid_argument("quantize: non-finite input weight");
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
            const float  s   = am > 0.0f ? float(std::max(double(am) / target,
                double(std::numeric_limits<float>::min()))) : 1.0f;
            sc[n] = s;
            for (int k = 0; k < K; ++k) {
                const float v = float(double(in[k]) / s);
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
                // The stored zero point is an integer in [0,15], so zero
                // must lie inside the represented range. Clamping only the
                // zero point after using [min,max] collapses one-sided groups.
                float lo = 0.0f, hi = 0.0f;
                for (int i = 0; i < kInt4Group; ++i) {
                    lo = std::min(lo, in[base + i]);
                    hi = std::max(hi, in[base + i]);
                }
                float s = float((double(hi) - double(lo)) / 15.0);
                if (!(s > 0.0f)) s = 1.0f;
                // Round the scale to bf16 *before* choosing the codes, so
                // the packer quantizes against the value the kernel will
                // actually see. Skipping this is a classic silent 2x
                // error amplifier at group boundaries.
                bf16_t sb = f32_to_bf16(s);
                if (sb.bits == 0) sb.bits = 1; // smallest positive BF16 scale
                s = bf16_to_f32(sb);
                if (!std::isfinite(s) || !(s > 0.0f))
                    throw std::overflow_error("quantize: INT4 scale is not representable");
                int z = int(std::nearbyint(-double(lo) / s));
                z = std::max(0, std::min(15, z));
                sc[int64_t(n) * G + g] = sb;
                w.zeros[int64_t(n) * G + g] = uint8_t(z);
                for (int i = 0; i < kInt4Group; ++i) {
                    int q = int(std::nearbyint(double(in[base + i]) / s)) + z;
                    q = std::max(0, std::min(15, q));
                    if (!std::isfinite(float((double(q) - z) * s)))
                        throw std::overflow_error("quantize: INT4 reconstructed value overflows");
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
