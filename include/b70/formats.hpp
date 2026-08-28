// =====================================================================
//  b70/formats.hpp  --  numeric format zoo for the bare-metal engine
//
//  Pure C++17. NO SYCL include on purpose: this header is compiled both
//  by icpx into device code and by a plain host compiler for the unit
//  tests, so the exact same decode path is what gets validated.
//
//  Supported weight formats
//  ------------------------
//   BF16       raw bfloat16, no scale                      16 bit/elem
//   FP8_E4M3   OCP FP8, per-output-channel fp32 scale        8 bit/elem
//   FP8_E5M2   OCP FP8, per-output-channel fp32 scale        8 bit/elem
//   INT8       symmetric, per-output-channel fp32 scale      8 bit/elem
//   INT4       asymmetric, per-group(128) bf16 scale+zero    4 bit/elem
//   MXFP8      OCP Microscaling: E4M3 + E8M0 per 32          8.25 bit/elem
//   MXFP4      OCP Microscaling: E2M1 + E8M0 per 32          4.25 bit/elem
//
//  Every format decodes through one of two compute pipelines:
//    PIPE_INT : INT8 / INT4  -> XMX int8 DPAS, int32 accumulator (2x rate)
//    PIPE_FLT : everything else -> XMX bf16 DPAS, fp32 accumulator
// =====================================================================
#ifndef B70_FORMATS_HPP
#define B70_FORMATS_HPP

#include <cstdint>
#include <cstring>

namespace b70 {

// ---------------------------------------------------------------------
// bit_cast that is legal in SYCL device code (no <bit>, no unions)
// ---------------------------------------------------------------------
inline float bits_to_f32(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline uint32_t f32_to_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

// ---------------------------------------------------------------------
// bfloat16 storage type. We keep our own POD so the header stays free of
// SYCL; at the kernel boundary it is bit-cast to
// sycl::ext::oneapi::bfloat16, which has identical layout.
// ---------------------------------------------------------------------
struct bf16_t { uint16_t bits; };

inline float bf16_to_f32(bf16_t x) { return bits_to_f32(uint32_t(x.bits) << 16); }

// IEEE-754 binary16 -> fp32. GPTQ stores its group scales as F16 (not bf16),
// so this is not interchangeable with bf16_to_f32: different exponent width,
// different bias. Handles subnormals, inf and NaN.
inline float f16_to_f32(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1F;
    const uint32_t m = h & 0x3FF;
    if (e == 0) {
        if (m == 0) return bits_to_f32(s);                     // +/- zero
        // subnormal: normalise by shifting the mantissa up
        uint32_t mm = m, ex = 127 - 15 + 1;
        while (!(mm & 0x400)) { mm <<= 1; --ex; }
        return bits_to_f32(s | (ex << 23) | ((mm & 0x3FF) << 13));
    }
    if (e == 0x1F) return bits_to_f32(s | 0x7F800000u | (m << 13));   // inf / NaN
    return bits_to_f32(s | ((e - 15 + 127) << 23) | (m << 13));
}

// round-to-nearest-even fp32 -> bf16
inline bf16_t f32_to_bf16(float f) {
    uint32_t u = f32_to_bits(f);
    if (((u >> 23) & 0xFF) == 0xFF) return bf16_t{uint16_t(u >> 16)};   // inf/nan
    uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
    return bf16_t{uint16_t((u + rounding) >> 16)};
}

// ---------------------------------------------------------------------
// E8M0 -- the OCP Microscaling shared exponent. A raw power of two.
//   X == 0x00 -> 2^-127   (subnormal in fp32, must not be built by shift)
//   X == 0xFF -> NaN
//   else      -> 2^(X-127), exactly representable, exponent field = X
// ---------------------------------------------------------------------
inline float e8m0_to_f32(uint8_t x) {
    if (x == 0xFF) return bits_to_f32(0x7FC00000u);          // NaN
    if (x == 0x00) return bits_to_f32(0x00400000u);          // 2^-127 subnormal
    return bits_to_f32(uint32_t(x) << 23);
}

// exponent of the largest finite magnitude of each element format,
// i.e. floor(log2(max_value)). Used to place the MX shared scale.
constexpr int kEmaxE2M1 = 2;   // max 6.0   = 1.5  * 2^2
constexpr int kEmaxE4M3 = 8;   // max 448.0 = 1.75 * 2^8
constexpr int kEmaxE5M2 = 15;  // max 57344 = 1.75 * 2^15

// ---------------------------------------------------------------------
// E2M1 -- the 4-bit element of MXFP4. Eight magnitudes plus a sign bit.
// A table beats bit-twiddling here: it lives in registers and the whole
// unpack folds into a shift + and + indexed move.
// ---------------------------------------------------------------------
inline float e2m1_to_f32(uint8_t nib) {
    constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = kMag[nib & 0x7];
    return (nib & 0x8) ? -v : v;
}

// round-to-nearest-even onto the E2M1 grid
inline uint8_t f32_to_e2m1(float v) {
    uint8_t sign = (f32_to_bits(v) >> 31) ? 0x8 : 0x0;
    float a = v < 0.0f ? -v : v;
    if (!(a == a)) return sign | 0x7;                 // NaN -> saturate, E2M1 has no NaN
    uint8_t code;
    // midpoints, ties resolved to the even code
    if      (a <  0.25f) code = 0;
    else if (a == 0.25f) code = 0;
    else if (a <  0.75f) code = 1;
    else if (a <  1.25f) code = 2;
    else if (a == 1.25f) code = 2;
    else if (a <  1.75f) code = 3;
    else if (a <  2.50f) code = 4;
    else if (a == 2.50f) code = 4;
    else if (a <  3.50f) code = 5;
    else if (a <  5.00f) code = 6;
    else if (a == 5.00f) code = 6;
    else                 code = 7;
    // the two remaining exact ties, 0.75 and 1.75, land on even codes 2 and 4
    if (a == 0.75f) code = 2;
    if (a == 1.75f) code = 4;
    return sign | code;
}

// ---------------------------------------------------------------------
// FP8 E4M3 (OCP binary8p4). bias 7, no infinities, 0x7F/0xFF are NaN,
// max finite magnitude 448.
// ---------------------------------------------------------------------
inline float e4m3_to_f32(uint8_t b) {
    const uint32_t s = uint32_t(b & 0x80) << 24;
    const int      e = (b >> 3) & 0x0F;
    const int      m = b & 0x07;
    if (e == 0x0F && m == 0x07) return bits_to_f32(0x7FC00000u | s);   // NaN
    if (e == 0) {                                                      // subnormal
        float v = float(m) * (1.0f / 8.0f) * 0.015625f;                // 2^-6
        return s ? -v : v;
    }
    // normal: (1 + m/8) * 2^(e-7)  built directly in the fp32 exponent field
    const uint32_t bits = s | (uint32_t(e - 7 + 127) << 23) | (uint32_t(m) << 20);
    return bits_to_f32(bits);
}

inline uint8_t f32_to_e4m3(float v) {
    const uint32_t u = f32_to_bits(v);
    const uint8_t  s = uint8_t((u >> 24) & 0x80);
    float a = v < 0.0f ? -v : v;
    if (!(a == a)) return uint8_t(s | 0x7F);                 // NaN
    if (a >= 464.0f) return uint8_t(s | 0x7E);               // saturate to 448
    if (a < 0.001953125f) {                                  // below 2^-9, half of min subnormal
        return s;                                            // -> signed zero
    }
    if (a < 0.015625f) {                                     // subnormal range, 2^-6
        int m = int(a / (0.015625f / 8.0f) + 0.5f);
        if (m > 7) m = 7;
        return uint8_t(s | m);
    }
    int e; float mant;
    {   // frexp without <cmath>: pull the exponent straight out of the bits
        const uint32_t ua = f32_to_bits(a);
        e = int((ua >> 23) & 0xFF) - 127;
        mant = bits_to_f32((ua & 0x807FFFFFu) | (127u << 23));   // in [1,2)
    }
    int m = int((mant - 1.0f) * 8.0f + 0.5f);
    if (m == 8) { m = 0; ++e; }
    if (e > 8) return uint8_t(s | 0x7E);
    return uint8_t(s | (uint8_t(e + 7) << 3) | uint8_t(m));
}

// ---------------------------------------------------------------------
// FP8 E5M2 (OCP binary8p5). bias 15, IEEE-like, has inf and NaN.
// ---------------------------------------------------------------------
inline float e5m2_to_f32(uint8_t b) {
    const uint32_t s = uint32_t(b & 0x80) << 24;
    const int      e = (b >> 2) & 0x1F;
    const int      m = b & 0x03;
    if (e == 0x1F) return bits_to_f32(s | (m ? 0x7FC00000u : 0x7F800000u));
    if (e == 0) {
        float v = float(m) * (1.0f / 4.0f) * 0.00006103515625f;  // 2^-14
        return s ? -v : v;
    }
    const uint32_t bits = s | (uint32_t(e - 15 + 127) << 23) | (uint32_t(m) << 21);
    return bits_to_f32(bits);
}

inline uint8_t f32_to_e5m2(float v) {
    const uint32_t u = f32_to_bits(v);
    const uint8_t  s = uint8_t((u >> 24) & 0x80);
    float a = v < 0.0f ? -v : v;
    if (!(a == a)) return uint8_t(s | 0x7F);
    if (a >= 61440.0f) return uint8_t(s | 0x7B);             // saturate to 57344
    if (a < 0.000007629394531f) return s;                    // half of min subnormal
    if (a < 0.00006103515625f) {
        int m = int(a / (0.00006103515625f / 4.0f) + 0.5f);
        if (m > 3) m = 3;
        return uint8_t(s | m);
    }
    const uint32_t ua = f32_to_bits(a);
    int   e    = int((ua >> 23) & 0xFF) - 127;
    float mant = bits_to_f32((ua & 0x807FFFFFu) | (127u << 23));
    int   m    = int((mant - 1.0f) * 4.0f + 0.5f);
    if (m == 4) { m = 0; ++e; }
    if (e > 15) return uint8_t(s | 0x7B);
    return uint8_t(s | (uint8_t(e + 15) << 2) | uint8_t(m));
}

// ---------------------------------------------------------------------
// Format enumeration and static traits
// ---------------------------------------------------------------------
enum class Fmt : uint8_t {
    BF16 = 0, FP8_E4M3, FP8_E5M2, INT8, INT4, MXFP8, MXFP4
};

enum class Pipe : uint8_t { FLT, INT };   // which XMX datapath consumes it

// MX_BLOCK is fixed by the OCP spec. INT4 uses a coarser group because
// its scale is a full bf16 rather than a single byte exponent.
constexpr int kMXBlock   = 32;
constexpr int kInt4Group = 128;

template <Fmt F> struct Traits;

template <> struct Traits<Fmt::BF16> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int  block = 0;          // 0 == no blocking, tensor-wide
    static constexpr int  bits_per_elem = 16;
    static constexpr const char* name = "bf16";
};
template <> struct Traits<Fmt::FP8_E4M3> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int  block = 0;          // per-output-channel scale
    static constexpr int  bits_per_elem = 8;
    static constexpr const char* name = "fp8_e4m3";
};
template <> struct Traits<Fmt::FP8_E5M2> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int  block = 0;
    static constexpr int  bits_per_elem = 8;
    static constexpr const char* name = "fp8_e5m2";
};
template <> struct Traits<Fmt::INT8> {
    static constexpr Pipe pipe = Pipe::INT;
    static constexpr int  block = 0;
    static constexpr int  bits_per_elem = 8;
    static constexpr const char* name = "int8";
};
template <> struct Traits<Fmt::INT4> {
    static constexpr Pipe pipe = Pipe::INT;
    static constexpr int  block = kInt4Group;
    static constexpr int  bits_per_elem = 4;
    static constexpr const char* name = "int4";
};
template <> struct Traits<Fmt::MXFP8> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int  block = kMXBlock;
    static constexpr int  bits_per_elem = 8;
    static constexpr const char* name = "mxfp8";
};
template <> struct Traits<Fmt::MXFP4> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int  block = kMXBlock;
    static constexpr int  bits_per_elem = 4;
    static constexpr const char* name = "mxfp4";
};

// ---------------------------------------------------------------------
// Element decode. `payload` points at the start of the row's packed
// bytes, `i` is the element index inside the row, `scale` is the already
// resolved block or channel scale.
//
// These are the functions that run in the innermost loop of every
// kernel, so they must be branch-free and inline.
// ---------------------------------------------------------------------
template <Fmt F>
inline float decode_elem(const uint8_t* payload, int i, float scale);

template <> inline float decode_elem<Fmt::BF16>(const uint8_t* p, int i, float) {
    return bf16_to_f32(bf16_t{uint16_t(uint16_t(p[2*i]) | (uint16_t(p[2*i+1]) << 8))});
}
template <> inline float decode_elem<Fmt::FP8_E4M3>(const uint8_t* p, int i, float s) {
    return e4m3_to_f32(p[i]) * s;
}
template <> inline float decode_elem<Fmt::FP8_E5M2>(const uint8_t* p, int i, float s) {
    return e5m2_to_f32(p[i]) * s;
}
template <> inline float decode_elem<Fmt::INT8>(const uint8_t* p, int i, float s) {
    return float(int8_t(p[i])) * s;
}
template <> inline float decode_elem<Fmt::MXFP8>(const uint8_t* p, int i, float s) {
    return e4m3_to_f32(p[i]) * s;
}
template <> inline float decode_elem<Fmt::MXFP4>(const uint8_t* p, int i, float s) {
    const uint8_t byte = p[i >> 1];
    const uint8_t nib  = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    return e2m1_to_f32(nib) * s;
}
// INT4 is asymmetric so it needs the zero point too; handled separately.
inline float decode_int4(const uint8_t* p, int i, float scale, uint8_t zero) {
    const uint8_t byte = p[i >> 1];
    const uint8_t q    = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    // 0xff marks the two's-complement s4 layout consumed directly by
    // Intel's persistent Xe2 grouped GEMM. Routed expert weights use this
    // representation so decode and prefill can share one payload.
    if (zero == 0xff) return float(q & 8 ? int(q) - 16 : int(q)) * scale;
    return (float(q) - float(zero)) * scale;
}

// Raw integer unpack, for the INT8 XMX datapath (no float conversion).
inline int8_t unpack_int4_raw(const uint8_t* p, int i) {
    const uint8_t byte = p[i >> 1];
    return int8_t((i & 1) ? (byte >> 4) : (byte & 0x0F));
}

// ---------------------------------------------------------------------
// Storage geometry helpers. Weights are row-major [N_out][K_in]; blocks
// run along K so a GEMV reduction walks contiguous bytes.
// ---------------------------------------------------------------------
inline int bytes_per_row(Fmt f, int K) {
    switch (f) {
        case Fmt::BF16:     return K * 2;
        case Fmt::FP8_E4M3:
        case Fmt::FP8_E5M2:
        case Fmt::INT8:     return K;
        case Fmt::INT4:     return K / 2;
        case Fmt::MXFP8:    return K;            // scales in a side array
        case Fmt::MXFP4:    return K / 2;        // scales in a side array
    }
    return 0;
}

inline int scales_per_row(Fmt f, int K) {
    switch (f) {
        case Fmt::BF16:     return 0;
        case Fmt::FP8_E4M3:
        case Fmt::FP8_E5M2:
        case Fmt::INT8:     return 1;            // one fp32 per output channel
        case Fmt::INT4:     return K / kInt4Group;
        case Fmt::MXFP8:
        case Fmt::MXFP4:    return K / kMXBlock; // one E8M0 byte each
    }
    return 0;
}

inline double bits_per_elem(Fmt f) {
    switch (f) {
        case Fmt::BF16:     return 16.0;
        case Fmt::FP8_E4M3:
        case Fmt::FP8_E5M2:
        case Fmt::INT8:     return 8.0;
        case Fmt::INT4:     return 4.0 + 24.0 / kInt4Group;   // bf16 scale + u8 zero
        case Fmt::MXFP8:    return 8.0 + 8.0 / kMXBlock;
        case Fmt::MXFP4:    return 4.0 + 8.0 / kMXBlock;
    }
    return 0.0;
}

inline const char* fmt_name(Fmt f) {
    switch (f) {
        case Fmt::BF16:     return "bf16";
        case Fmt::FP8_E4M3: return "fp8_e4m3";
        case Fmt::FP8_E5M2: return "fp8_e5m2";
        case Fmt::INT8:     return "int8";
        case Fmt::INT4:     return "int4";
        case Fmt::MXFP8:    return "mxfp8";
        case Fmt::MXFP4:    return "mxfp4";
    }
    return "?";
}

} // namespace b70
#endif // B70_FORMATS_HPP
