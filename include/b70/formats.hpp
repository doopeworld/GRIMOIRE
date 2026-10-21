// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

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
//   Q4_K       GGUF/ggml 256-element super-block             4.50 bit/elem
//   Q5_K       GGUF/ggml 256-element super-block             5.50 bit/elem
//   Q6_K       GGUF/ggml 256-element super-block             6.5625 bit/elem
//   Q8_0       GGUF/ggml 32-element block                    8.50 bit/elem
//   IQ4_XS     GGUF/ggml 256-element super-block             4.25 bit/elem
//   IQ4_NL     GGUF/ggml 32-element block                    4.50 bit/elem
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
    if (((u >> 23) & 0xFF) == 0xFF) {
        // A NaN whose payload lives only in the discarded bits must not
        // become infinity when narrowed.
        return bf16_t{uint16_t((u >> 16) | ((u & 0x7FFFFFu) ? 0x40u : 0u))};
    }
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
    BF16 = 0, FP8_E4M3, FP8_E5M2, INT8, INT4, MXFP8, MXFP4, Q4_K, Q6_K, Q5_K, Q8_0, IQ4_XS, IQ4_NL
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
template <> struct Traits<Fmt::Q4_K> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int block = 256, bits_per_elem = 4;
    static constexpr const char* name = "q4_k";
};
template <> struct Traits<Fmt::Q6_K> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int block = 256, bits_per_elem = 6;
    static constexpr const char* name = "q6_k";
};

template <> struct Traits<Fmt::Q5_K> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int block = 256, bits_per_elem = 5;
    static constexpr const char* name = "q5_k";
};
template <> struct Traits<Fmt::Q8_0> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int block = 32, bits_per_elem = 8;
    static constexpr const char* name = "q8_0";
};
template <> struct Traits<Fmt::IQ4_XS> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int block = 256, bits_per_elem = 4;
    static constexpr const char* name = "iq4_xs";
};
template <> struct Traits<Fmt::IQ4_NL> {
    static constexpr Pipe pipe = Pipe::FLT;
    static constexpr int block = 32, bits_per_elem = 4;
    static constexpr const char* name = "iq4_nl";
};

template <Fmt F> struct NativeGGUFBlock;
#define B70_NATIVE_GGUF(F, E, B, T, L) template <> struct NativeGGUFBlock<Fmt::F> { \
    static constexpr int elems = E, bytes = B, type = T, lanes = L; \
};
B70_NATIVE_GGUF(Q4_K, 256, 144, 12, 8)
B70_NATIVE_GGUF(Q5_K, 256, 176, 13, 16)
B70_NATIVE_GGUF(Q6_K, 256, 210, 14, 16)
B70_NATIVE_GGUF(Q8_0, 32, 34, 8, 2)
B70_NATIVE_GGUF(IQ4_XS, 256, 136, 23, 8)
B70_NATIVE_GGUF(IQ4_NL, 32, 18, 20, 2)
#undef B70_NATIVE_GGUF
constexpr bool is_native_gguf(Fmt f) {
    return f == Fmt::Q4_K || f == Fmt::Q5_K || f == Fmt::Q6_K ||
           f == Fmt::Q8_0 || f == Fmt::IQ4_XS || f == Fmt::IQ4_NL;
}
inline int native_gguf_block_elems(Fmt f) {
    switch (f) {
    case Fmt::Q4_K: case Fmt::Q5_K: case Fmt::Q6_K: case Fmt::IQ4_XS: return 256;
    case Fmt::Q8_0: case Fmt::IQ4_NL: return 32;
    default: return 0;
    }
}
inline Fmt native_gguf_fmt(uint32_t type) {
    switch (type) {
    case 12: return Fmt::Q4_K;
    case 13: return Fmt::Q5_K;
    case 14: return Fmt::Q6_K;
    case 8: return Fmt::Q8_0;
    case 23: return Fmt::IQ4_XS;
    case 20: return Fmt::IQ4_NL;
    default: return Fmt::BF16;
    }
}
inline bool is_native_gguf_type(uint32_t type) { return is_native_gguf(native_gguf_fmt(type)); }
inline uint32_t native_gguf_type(Fmt f) {
    switch (f) {
    case Fmt::Q4_K: return 12;
    case Fmt::Q5_K: return 13;
    case Fmt::Q6_K: return 14;
    case Fmt::Q8_0: return 8;
    case Fmt::IQ4_XS: return 23;
    case Fmt::IQ4_NL: return 20;
    default: return UINT32_MAX;
    }
}

// GGML super-blocks carry their own scales. These functions are shared by
// host tests and device kernels; no side scale array is involved.
inline float decode_q4_k(const uint8_t* row, int k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const uint8_t* p = row + (k / 256) * 144;
    const int group = (k % 256) / 32, i = k % 32;
    const uint8_t* s = p + 4;
    int sc, m;
    if (group < 4) { sc = s[group] & 63; m = s[group + 4] & 63; }
    else {
        sc = (s[group + 4] & 15) | ((s[group - 4] >> 2) & 48);
        m = (s[group + 4] >> 4) | ((s[group] >> 2) & 48);
    }
    const float d = f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
    const float dm = f16_to_f32(uint16_t(p[2]) | (uint16_t(p[3]) << 8));
    const int q = (p[16 + (group / 2) * 32 + i] >> ((group % 2) * 4)) & 15;
    const float ds = d * float(sc), mins = dm * float(m);
    return ds * float(q) - mins;
}
inline float decode_q6_k(const uint8_t* row, int k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const uint8_t* p = row + (k / 256) * 210;
    const int idx = k % 256, half = idx / 128, lane = (idx % 128) / 32, i = idx % 32;
    const int low = (p[half * 64 + (lane % 2) * 32 + i] >> ((lane / 2) * 4)) & 15;
    const int high = (p[128 + half * 32 + i] >> (lane * 2)) & 3;
    const int q = (low | (high << 4)) - 32;
    const float d = f16_to_f32(uint16_t(p[208]) | (uint16_t(p[209]) << 8));
    return (d * float(int8_t(p[192 + idx / 16]))) * float(q);
}

inline int iq4nl_value(int q) {
    constexpr int values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    return values[q];
}
inline float decode_q5_k(const uint8_t* row, int k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const uint8_t* p = row + (k / 256) * 176;
    const int g = (k % 256) / 32, i = k % 32;
    const uint8_t* s = p + 4;
    int sc, m;
    if (g < 4) { sc = s[g] & 63; m = s[g + 4] & 63; }
    else { sc = (s[g + 4] & 15) | ((s[g - 4] >> 2) & 48);
           m = (s[g + 4] >> 4) | ((s[g] >> 2) & 48); }
    const float d = f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
    const float dm = f16_to_f32(uint16_t(p[2]) | (uint16_t(p[3]) << 8));
    const int q = ((p[48 + (g / 2) * 32 + i] >> ((g % 2) * 4)) & 15) |
                  (((p[16 + i] >> g) & 1) << 4);
    return (d * float(sc)) * float(q) - dm * float(m);
}
inline float decode_q8_0(const uint8_t* row, int k) {
    const uint8_t* p = row + (k / 32) * 34;
    return f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8)) * float(int8_t(p[2 + k % 32]));
}
inline float decode_iq4_xs(const uint8_t* row, int k) {
    const uint8_t* p = row + (k / 256) * 136;
    const int g = (k % 256) / 32, i = k % 32;
    const int h = int(uint16_t(p[2]) | (uint16_t(p[3]) << 8));
    const int low = (p[4 + g / 2] >> (4 * (g % 2))) & 15;
    const int sc = (low | (((h >> (2 * g)) & 3) << 4)) - 32;
    const int q = (p[8 + g * 16 + i % 16] >> (4 * (i / 16))) & 15;
    return (f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8)) * float(sc)) * float(iq4nl_value(q));
}
inline float decode_iq4_nl(const uint8_t* row, int k) {
    const uint8_t* p = row + (k / 32) * 18;
    const int i = k % 32;
    const int q = (p[2 + i % 16] >> (4 * (i / 16))) & 15;
    return f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8)) * float(iq4nl_value(q));
}

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
template <> inline float decode_elem<Fmt::Q4_K>(const uint8_t* p, int i, float) { return decode_q4_k(p, i); }
template <> inline float decode_elem<Fmt::Q6_K>(const uint8_t* p, int i, float) { return decode_q6_k(p, i); }
template <> inline float decode_elem<Fmt::Q5_K>(const uint8_t* p, int i, float) { return decode_q5_k(p, i); }
template <> inline float decode_elem<Fmt::Q8_0>(const uint8_t* p, int i, float) { return decode_q8_0(p, i); }
template <> inline float decode_elem<Fmt::IQ4_XS>(const uint8_t* p, int i, float) { return decode_iq4_xs(p, i); }
template <> inline float decode_elem<Fmt::IQ4_NL>(const uint8_t* p, int i, float) { return decode_iq4_nl(p, i); }
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
        case Fmt::Q4_K:     return K % 256 ? 0 : K / 256 * 144;
        case Fmt::Q6_K:     return K % 256 ? 0 : K / 256 * 210;
        case Fmt::Q5_K:     return K % 256 ? 0 : K / 256 * 176;
        case Fmt::IQ4_XS:   return K % 256 ? 0 : K / 256 * 136;
        case Fmt::Q8_0:     return K % 32 ? 0 : K / 32 * 34;
        case Fmt::IQ4_NL:   return K % 32 ? 0 : K / 32 * 18;
    }
    return 0;
}

inline int scales_per_row(Fmt f, int K) {
    switch (f) {
        case Fmt::BF16:     return 0;
        case Fmt::Q4_K:
        case Fmt::Q6_K:
        case Fmt::Q5_K:
        case Fmt::Q8_0:
        case Fmt::IQ4_XS:
        case Fmt::IQ4_NL:   return 0;
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
        case Fmt::Q4_K:     return 4.5;
        case Fmt::Q6_K:     return 6.5625;
        case Fmt::Q5_K:     return 5.5;
        case Fmt::Q8_0:     return 8.5;
        case Fmt::IQ4_XS:   return 4.25;
        case Fmt::IQ4_NL:   return 4.5;
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
        case Fmt::Q4_K:     return "q4_k";
        case Fmt::Q6_K:     return "q6_k";
        case Fmt::Q5_K:     return "q5_k";
        case Fmt::Q8_0:     return "q8_0";
        case Fmt::IQ4_XS:   return "iq4_xs";
        case Fmt::IQ4_NL:   return "iq4_nl";
    }
    return "?";
}

} // namespace b70
#endif // B70_FORMATS_HPP
