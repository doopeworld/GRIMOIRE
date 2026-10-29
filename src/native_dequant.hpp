// GGUF native weights -> bf16 VNNI for the prompt GEMM, streaming.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// W [N][K] in Q4_K / Q5_K / Q6_K / Q8_0 / IQ4_XS / IQ4_NL blocks -> bf16 VNNI [K/2][N][2], the
// walk of dequant_mxfp4_stream (gemm_fast.cpp): one work-item per (row n, 128 k), the 16 lanes of
// a sub-group are 16 consecutive n, so every dword store writes one whole 64-byte line of the
// scratch; row blocks fastest.  Each work-item decodes its block header once and then 128 values
// with decode_elem<F>'s operations in decode_elem<F>'s order (no contraction), rounded to bf16 the
// same way, so the scratch is bit-identical to dequant_vnni's.  ilv_fi > 0 (SwiGLU): W = [gate;
// up] with ilv_fi rows each, written in dequant_mxfp4_stream's interleave.
// 2026-10-29: the generic tiled dequant (decode_elem per value, header re-read per value) made a
// Qwen3.8-27B UD-Q4_K_XL prompt 1,703 tok/s against 2,097 for the same file as MXFP4.
// GRIMOIRE_GGUF_DQ_STREAM=0 keeps the tiled dequant.
#pragma once
#include "kernels.hpp"
#include "b70/formats.hpp"
#include <cstdlib>
#include <cstring>

namespace b70 {

// float -> bf16 bits, round to nearest even (gemm_fast.cpp bf16_rne: the conversion instruction on
// the device; the same result for every finite value on the host).
inline uint32_t nd_bf16_bits(float f) {
#ifdef __SYCL_DEVICE_ONLY__
    return uint32_t(__spirv_ConvertFToBF16INTEL(f));
#else
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
#endif
}
inline uint32_t nd_pair(float lo, float hi) { return nd_bf16_bits(lo) | (nd_bf16_bits(hi) << 16); }

// iq4nl_value(q) as a float without its table (a function-local table lands in private memory
// on the GPU): native_qk_esimd.hpp's nq_iq4_value polynomial, rounded -- exactly the 16 values.
inline float nd_iq4_value(uint32_t q) {
    const float i = float(q);
    float v = i * 0.00158604281f + 0.0310864393f;
    v = v * i - 1.37192702f;
    v = v * i + 24.17383f;
    v = v * i - 126.567009f;
    return sycl::rint(v);
}

inline uint32_t nd_byte(const sycl::uint4* b, int i) { return (b[i >> 4][(i >> 2) & 3] >> (8 * (i & 3))) & 0xFFu; }
inline uint32_t nd_u16(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8); }
// Byte i of n bytes read as 16-bit words from a 2-byte aligned address (Q6_K / Q8_0 / IQ4_NL
// blocks are 2-byte aligned only).
template <int W> struct nd_words {
    uint16_t w[W];
    inline void load(const uint8_t* p) {
        const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
        #pragma unroll
        for (int j = 0; j < W; ++j) w[j] = q[j];
    }
    inline uint32_t operator[](int i) const { return (uint32_t(w[i >> 1]) >> (8 * (i & 1))) & 0xFFu; }
};

// The K-quant 6-bit scale and min of 32-value group g (decode_q4_k / decode_q5_k).
inline void nd_kq_scale(const uint8_t* s, int g, int& sc, int& m) {
    if (g < 4) { sc = s[g] & 63; m = s[g + 4] & 63; }
    else {
        sc = (s[g + 4] & 15) | ((s[g - 4] >> 2) & 48);
        m = (s[g + 4] >> 4) | ((s[g] >> 2) & 48);
    }
}

// The 128 values k = 128 kb .. 128 kb + 127 of one row as 64 bf16 pairs, pair j at o[j * N].
template <Fmt F>
inline void nd_chunk(const uint8_t* row, int kb, uint32_t* o, int64_t N) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if constexpr (F == Fmt::Q4_K || F == Fmt::Q5_K) {
        constexpr int BS = F == Fmt::Q4_K ? 144 : 176, QS = F == Fmt::Q4_K ? 16 : 48;
        const uint8_t* p = row + int64_t(kb / 2) * BS;
        const int half = kb & 1;
        const float d = f16_to_f32(uint16_t(nd_u16(p)));
        const float dm = f16_to_f32(uint16_t(nd_u16(p + 2)));
        sycl::uint4 qh[2];
        if constexpr (F == Fmt::Q5_K) {
            qh[0] = *reinterpret_cast<const sycl::uint4*>(p + 16);
            qh[1] = *reinterpret_cast<const sycl::uint4*>(p + 32);
        }
        #pragma unroll
        for (int gp = 0; gp < 2; ++gp) {
            const int g0 = 4 * half + 2 * gp;          // groups g0 (low nibbles), g0+1 (high)
            const uint8_t* qb = p + QS + (g0 / 2) * 32;
            const sycl::uint4 b[2] = {*reinterpret_cast<const sycl::uint4*>(qb),
                                      *reinterpret_cast<const sycl::uint4*>(qb + 16)};
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int g = g0 + t;
                int sc, m;
                nd_kq_scale(p + 4, g, sc, m);
                const float ds = d * float(sc), mins = dm * float(m);
                #pragma unroll
                for (int i = 0; i < 32; i += 2) {
                    uint32_t q0 = (nd_byte(b, i) >> (4 * t)) & 15u;
                    uint32_t q1 = (nd_byte(b, i + 1) >> (4 * t)) & 15u;
                    if constexpr (F == Fmt::Q5_K) {
                        q0 |= ((nd_byte(qh, i) >> g) & 1u) << 4;
                        q1 |= ((nd_byte(qh, i + 1) >> g) & 1u) << 4;
                    }
                    o[int64_t((2 * gp + t) * 16 + i / 2) * N] =
                        nd_pair(ds * float(q0) - mins, ds * float(q1) - mins);
                }
            }
        }
    } else if constexpr (F == Fmt::Q6_K) {
        // 210-byte blocks, 2-byte aligned.  128 values = one half of a block: its 64 low-bit bytes
        // (ql), 32 high-bit bytes (qh), 8 scales.
        const uint8_t* p = row + int64_t(kb / 2) * 210;
        const int half = kb & 1;
        const float d = f16_to_f32(uint16_t(nd_u16(p + 208)));
        nd_words<32> lo;
        nd_words<16> hi;
        nd_words<4> scb;
        lo.load(p + half * 64);
        hi.load(p + 128 + half * 32);
        scb.load(p + 192 + half * 8);
        #pragma unroll
        for (int L = 0; L < 4; ++L) {
            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                const float sc = d * float(int8_t(uint8_t(scb[L * 2 + s])));
                #pragma unroll
                for (int i = 16 * s; i < 16 * s + 16; i += 2) {
                    const int q0 = int(((lo[(L % 2) * 32 + i] >> ((L / 2) * 4)) & 15u) |
                                       (((hi[i] >> (L * 2)) & 3u) << 4)) - 32;
                    const int q1 = int(((lo[(L % 2) * 32 + i + 1] >> ((L / 2) * 4)) & 15u) |
                                       (((hi[i + 1] >> (L * 2)) & 3u) << 4)) - 32;
                    o[int64_t(L * 16 + i / 2) * N] = nd_pair(sc * float(q0), sc * float(q1));
                }
            }
        }
    } else if constexpr (F == Fmt::IQ4_XS) {
        // 136-byte blocks, 8-byte aligned: the 16 index bytes of a group as two 8-byte loads.
        const uint8_t* p = row + int64_t(kb / 2) * 136;
        const int half = kb & 1;
        const float d = f16_to_f32(uint16_t(nd_u16(p)));
        const uint32_t h = nd_u16(p + 2);
        #pragma unroll
        for (int gl = 0; gl < 4; ++gl) {
            const int g = 4 * half + gl;
            const int low = (p[4 + g / 2] >> (4 * (g % 2))) & 15;
            const float ds = d * float(int(low | ((int(h >> (2 * g)) & 3) << 4)) - 32);
            const sycl::uint2 w0 = *reinterpret_cast<const sycl::uint2*>(p + 8 + g * 16);
            const sycl::uint2 w1 = *reinterpret_cast<const sycl::uint2*>(p + 16 + g * 16);
            const sycl::uint4 b[1] = {sycl::uint4(w0[0], w0[1], w1[0], w1[1])};
            #pragma unroll
            for (int i = 0; i < 32; i += 2) {
                const uint32_t q0 = (nd_byte(b, i % 16) >> (4 * (i / 16))) & 15u;
                const uint32_t q1 = (nd_byte(b, (i + 1) % 16) >> (4 * (i / 16))) & 15u;
                o[int64_t(gl * 16 + i / 2) * N] = nd_pair(ds * nd_iq4_value(q0), ds * nd_iq4_value(q1));
            }
        }
    } else {
        // Q8_0 (34 B) / IQ4_NL (18 B): blocks of 32, 2-byte aligned; 4 blocks here.
        static_assert(F == Fmt::Q8_0 || F == Fmt::IQ4_NL);
        constexpr int BS = F == Fmt::Q8_0 ? 34 : 18;
        #pragma unroll
        for (int bl = 0; bl < 4; ++bl) {
            const uint8_t* p = row + int64_t(kb * 4 + bl) * BS;
            nd_words<BS / 2> b;
            b.load(p);
            const float d = f16_to_f32(uint16_t(b.w[0]));
            #pragma unroll
            for (int i = 0; i < 32; i += 2) {
                float v0, v1;
                if constexpr (F == Fmt::Q8_0) {
                    v0 = d * float(int8_t(uint8_t(b[2 + i])));
                    v1 = d * float(int8_t(uint8_t(b[2 + i + 1])));
                } else {
                    v0 = d * nd_iq4_value((b[2 + i % 16] >> (4 * (i / 16))) & 15u);
                    v1 = d * nd_iq4_value((b[2 + (i + 1) % 16] >> (4 * (i / 16))) & 15u);
                }
                o[int64_t(bl * 16 + i / 2) * N] = nd_pair(v0, v1);
            }
        }
    }
}

// The layout the stream kernel needs: whole 256-value rows, packed, N a multiple of 256 (one
// work-group = 256 rows), rows and payload aligned for its loads (Q4_K / Q5_K 16 bytes, IQ4_XS 8,
// the others 2).
inline bool native_stream_ok(const QuantWeight& w) {
    static const bool on = [] { const char* e = std::getenv("GRIMOIRE_GGUF_DQ_STREAM");
        return !(e && *e == '0'); }();
    const int64_t al = w.fmt == Fmt::Q4_K || w.fmt == Fmt::Q5_K ? 16 : w.fmt == Fmt::IQ4_XS ? 8 : 2;
    return on && is_native_gguf(w.fmt) && w.payload && !w.scales && !w.zeros &&
           w.K % 256 == 0 && w.N % 256 == 0 && w.row_bytes == bytes_per_row(w.fmt, w.K) &&
           w.row_bytes % al == 0 && int64_t(reinterpret_cast<uintptr_t>(w.payload) % uintptr_t(al)) == 0;
}

template <Fmt F>
sycl::event dequant_native_stream_t(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                                    const std::vector<sycl::event>& deps, int ilv_fi) {
    const int N = w.N, K = w.K;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const uint8_t* pay = w.payload;
        const int64_t rb = w.row_bytes;
        const int KB = K / 128, NB = N / 256;
        // SwiGLU with ilv_fi % 128 == 0: a work-group takes 128 gate rows and the matching 128 up
        // rows, i.e. 256 CONSECUTIVE scratch columns (gate rows n..n+255 would scatter its stores
        // over 512 columns, the other half written 68 groups later on Qwen3.8: gate_up 34816 x
        // 5120 1.84-1.97 ms, ~245 GB/s, vs ~435 for a plain matrix).
        const bool pair = ilv_fi > 0 && ilv_fi % 128 == 0;
        h.parallel_for(sycl::nd_range<1>(size_t(KB) * NB * 256, 256),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const int L = int(it.get_group(0));
            const int j = int(it.get_local_id(0));
            const int kb = L / NB;
            int n, col;
            if (pair) {
                col = (L % NB) * 256 + j;
                const int g = (L % NB) * 128 + (j / 64) * 32 + j % 32;
                n = j % 64 < 32 ? g : ilv_fi + g;
            } else {
                n = (L % NB) * 256 + j;
                col = ilv_fi == 0 ? n
                    : n < ilv_fi ? (n / 32) * 64 + n % 32
                                 : ((n - ilv_fi) / 32) * 64 + 32 + (n - ilv_fi) % 32;
            }
            uint32_t* o = reinterpret_cast<uint32_t*>(dst) + int64_t(kb) * 64 * N + col;
            nd_chunk<F>(pay + int64_t(n) * rb, kb, o, N);
        });
    });
}

inline sycl::event dequant_native_stream(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                                         const std::vector<sycl::event>& deps, int ilv_fi = 0) {
    switch (w.fmt) {
        case Fmt::Q4_K:   return dequant_native_stream_t<Fmt::Q4_K>(q, w, dst, deps, ilv_fi);
        case Fmt::Q5_K:   return dequant_native_stream_t<Fmt::Q5_K>(q, w, dst, deps, ilv_fi);
        case Fmt::Q6_K:   return dequant_native_stream_t<Fmt::Q6_K>(q, w, dst, deps, ilv_fi);
        case Fmt::Q8_0:   return dequant_native_stream_t<Fmt::Q8_0>(q, w, dst, deps, ilv_fi);
        case Fmt::IQ4_XS: return dequant_native_stream_t<Fmt::IQ4_XS>(q, w, dst, deps, ilv_fi);
        case Fmt::IQ4_NL: return dequant_native_stream_t<Fmt::IQ4_NL>(q, w, dst, deps, ilv_fi);
        default: break;
    }
    throw std::runtime_error("dequant_native_stream: not a native GGUF format");
}

} // namespace b70
