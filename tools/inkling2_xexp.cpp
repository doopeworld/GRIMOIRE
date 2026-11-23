// SPDX-License-Identifier: GPL-3.0-or-later
// Inkling 2 Mini decode experts on the matrix unit, built WITHOUT the 256-GRF option (8 threads per EU):
// one thread per (slot, 16-column tile) over the whole reduction, so the 1152-thread grid is one wave
// (no tail), no SLM, no barrier; the next 1 KB weight block is loaded while the current one computes.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include "b70/formats.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
struct InkXArgs {
    const uint8_t* codes; const uint8_t* scales;   // packed, this card's experts
    const float* amax; const float* den;            // NVFP4 outer = amax[e] / den[0]
    const int* sel; int e0, en, topk;
    int N, K;                                        // projection rows / reduction
    const float* x; size_t ldx;                      // input per slot: x + slot * ldx (ldx 0 = shared row)
    float* out;                                      // gate_up: act [slot][N/2]; down: dbuf [slot][N]
    int gate_up; float gcap, ucap;
    const uint32_t* lut;   // [2][256]: byte -> bf16 pair of the 1x / 1.5x magnitude class (0 for the other class)
};

namespace {
template <int KS> class InkXExp;
namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

template <int N> inline es::simd<float, N> v_bfr(es::simd<float, N> v) SYCL_ESIMD_FUNCTION {
    es::simd<uint32_t, N> u = v.template bit_cast_view<uint32_t>().read();
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    return u.template bit_cast_view<float>().read();
}
template <int N> inline es::simd<float, N> v_e4m3(es::simd<uint32_t, N> b) SYCL_ESIMD_FUNCTION {
    const es::simd<uint32_t, N> e = (b >> 3) & 15u, m = b & 7u, s = (b & 0x80u) << 24;
    es::simd<uint32_t, N> bits = s | ((e + 120u) << 23) | (m << 20);
    es::simd<float, N> v = bits.template bit_cast_view<float>().read();
    es::simd<float, N> sub = es::simd<float, N>(m) * (1.f / 512.f);
    sub.merge(-sub, s != 0u);
    v.merge(sub, e == 0u);
    return v;
}
inline float th_(float v) SYCL_ESIMD_FUNCTION { const float t = sycl::exp(-2.f * (v < 0.f ? -v : v)); const float r = (1.f - t) / (1.f + t); return v < 0.f ? -r : r; }
// bf16 bits of the 1x (b1) and 1.5x (b3) magnitude classes of 128 nibbles (0 where the nibble is the other class)
inline void widen(es::simd<uint16_t, 128> n, es::simd<uint16_t, 128>& b1, es::simd<uint16_t, 128>& b3) SYCL_ESIMD_FUNCTION {
    const es::simd<uint16_t, 128> mag = n & 7, sign = (n & 8) << 12;
    // 2^j: mag 1 -> 0.5, 2|3 -> 1, 4|5 -> 2, 6|7 -> 4
    es::simd<uint16_t, 128> bits = (es::simd<uint16_t, 128>(0x3F00) + ((mag >> 1) << 7)) | sign;
    const auto odd3 = ((mag & 1) != 0) & (mag > 1);
    const auto zero = mag == 0;
    b1 = bits; b1.merge(es::simd<uint16_t, 128>(0), odd3 | zero);
    b3 = bits; b3.merge(es::simd<uint16_t, 128>(0), !odd3);
}
}

static sycl::event xexp_run1(sycl::queue& q, const InkXArgs& g) {
    const int T = g.N / 16, G = g.K / 128;
    return q.submit([&](sycl::handler& h) {
        const InkXArgs a = g;
        h.parallel_for<InkXExp<1>>(sycl::nd_range<1>(size_t(a.topk) * T, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int u = int(it.get_global_id(0)), s = u / T, t = u % T;
            const int e = a.sel[s];
            const bool own = e >= a.e0 && e < a.e0 + a.en;
            es::simd<float, 16> acc = 0.f;
            if (own) {
                const size_t el = size_t(e - a.e0);
                const float o = a.amax[e] / a.den[0];
                const uint8_t* cb = a.codes + el * size_t(a.N) * (a.K / 2) + size_t(t) * 1024;
                const uint8_t* sb = a.scales + el * size_t(a.N) * (a.K / 16) + size_t(t) * 128;
                const float* x = a.x + size_t(s) * a.ldx;
                constexpr auto kStream = es::properties{es::cache_hint_L1<es::cache_hint::streaming>, es::cache_hint_L2<es::cache_hint::uncached>, es::alignment<64>};
                const size_t gstride = size_t(T) * 1024, sstride = size_t(T) * 128;
                es::simd<uint8_t, 1024> nxt;
                nxt.template select<512, 1>(0) = es::block_load<uint8_t, 512>(cb, kStream);
                nxt.template select<512, 1>(512) = es::block_load<uint8_t, 512>(cb + 512, kStream);
                es::simd<uint8_t, 128> nsc = es::block_load<uint8_t, 128>(sb, kStream);
                for (int gi = 0; gi < G; ++gi) {
                    es::simd<uint8_t, 1024> raw = nxt;
                    es::simd<uint8_t, 128> sc8 = nsc;
                    if (gi + 1 < G) {
                        nxt.template select<512, 1>(0) = es::block_load<uint8_t, 512>(cb + (gi + 1) * gstride, kStream);
                        nxt.template select<512, 1>(512) = es::block_load<uint8_t, 512>(cb + (gi + 1) * gstride + 512, kStream);
                        nsc = es::block_load<uint8_t, 128>(sb + (gi + 1) * sstride, kStream);
                    }
                    es::simd<float, 128> xf = es::block_load<float, 128>(x + gi * 128);
                    es::simd<sycl::ext::oneapi::bfloat16, 128> xb = xf;
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        es::simd<uint8_t, 128> by = raw.template select<128, 1>(c * 128);
                        es::simd<uint16_t, 128> l1, l3, h1, h3;
                        widen(es::simd<uint16_t, 128>(by & 15), l1, l3);
                        widen(es::simd<uint16_t, 128>(by >> 4), h1, h3);
                        es::simd<uint16_t, 256> w1, w3;
                        w1.template select<128, 2>(0) = l1; w1.template select<128, 2>(1) = h1;
                        w3.template select<128, 2>(0) = l3; w3.template select<128, 2>(1) = h3;
                        es::simd<sycl::ext::oneapi::bfloat16, 256> B1 = w1.template bit_cast_view<sycl::ext::oneapi::bfloat16>().read();
                        es::simd<sycl::ext::oneapi::bfloat16, 256> B3 = w3.template bit_cast_view<sycl::ext::oneapi::bfloat16>().read();
                        es::simd<sycl::ext::oneapi::bfloat16, 16> A = xb.template select<16, 1>(c * 16);
                        es::simd<float, 16> z = 0.f;
                        es::simd<float, 16> r1 = xmx::dpas<8, 1, float, float, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(z, B1, A);
                        es::simd<float, 16> r3 = xmx::dpas<8, 1, float, float, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(z, B3, A);
                        es::simd<uint8_t, 16> s16 = sc8.template select<16, 1>(c * 16);
                        es::simd<float, 16> scf = v_e4m3<16>(es::simd<uint32_t, 16>(s16));
                        es::simd<float, 16> q1 = v_bfr<16>(scf * o), q3 = v_bfr<16>((scf * 1.5f) * o);
                        acc += q1 * r1 + q3 * r3;
                    }
                }
            }
            if (a.gate_up) {
                if (!own) return;
                es::simd<float, 16> r = v_bfr<16>(acc);
                es::simd<float, 8> gg = r.template select<8, 2>(0), uu = r.template select<8, 2>(1);
                es::simd<float, 8> out;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float g0 = gg[i], u0 = uu[i];
                    const float sg = g0 >= 0.f ? 1.f / (1.f + sycl::exp(-g0)) : sycl::exp(g0) / (1.f + sycl::exp(g0));
                    out[i] = (a.gcap * th_(g0 / a.gcap)) * sg * (a.ucap * th_(u0 / a.ucap));
                }
                out = v_bfr<8>(out);
                es::block_store<float, 8>(a.out + size_t(s) * (a.N / 2) + t * 8, out);
            } else {
                es::simd<float, 16> r = own ? v_bfr<16>(acc) : es::simd<float, 16>(0.f);
                es::block_store<float, 16>(a.out + size_t(s) * a.N + t * 16, r);
            }
        });
    });
}

// SLM-table version (Grimoire's Ornith ESIMD MoE trick): one gather per weight byte per class gives the VNNI dword
// of both nibbles -- no per-nibble arithmetic.  KS threads per (slot, tile) split K; next block loaded ahead.
template <int KS> class InkXExpL;
template <int KS>
static sycl::event xexp_lut(sycl::queue& q, const InkXArgs& g) {
    const int T = g.N / 16, G = g.K / 128;
    return q.submit([&](sycl::handler& h) {
        const InkXArgs a = g;
        h.parallel_for<InkXExpL<KS>>(sycl::nd_range<1>(size_t(a.topk) * T * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<2048 + KS * 16 * sizeof(float)>();
            const int part = int(it.get_local_id(0));
            if (part == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) es::slm_block_store<uint32_t, 128>(uint32_t(i * 512), es::block_load<uint32_t, 128>(a.lut + i * 128));
            const int grp = int(it.get_group(0)), s = grp / T, t = grp % T;
            const int e = a.sel[s];
            const bool own = e >= a.e0 && e < a.e0 + a.en;
            es::barrier();
            es::simd<float, 16> acc = 0.f;
            if (own) {
                const size_t el = size_t(e - a.e0);
                const float o = a.amax[e] / a.den[0];
                const uint8_t* cb = a.codes + el * size_t(a.N) * (a.K / 2) + size_t(t) * 1024;
                const uint8_t* sb = a.scales + el * size_t(a.N) * (a.K / 16) + size_t(t) * 128;
                const float* x = a.x + size_t(s) * a.ldx;
                constexpr auto kStream = es::properties{es::cache_hint_L1<es::cache_hint::streaming>, es::cache_hint_L2<es::cache_hint::uncached>, es::alignment<64>};
                const size_t gstride = size_t(T) * 1024, sstride = size_t(T) * 128;
                const int g0 = part * G / KS, g1 = (part + 1) * G / KS;
                es::simd<uint8_t, 1024> nxt;
                es::simd<uint8_t, 128> nsc;
                nxt.template select<512, 1>(0) = es::block_load<uint8_t, 512>(cb + g0 * gstride, kStream);
                nxt.template select<512, 1>(512) = es::block_load<uint8_t, 512>(cb + g0 * gstride + 512, kStream);
                nsc = es::block_load<uint8_t, 128>(sb + g0 * sstride, kStream);
                for (int gi = g0; gi < g1; ++gi) {
                    es::simd<uint8_t, 1024> raw = nxt;
                    es::simd<uint8_t, 128> sc8 = nsc;
                    if (gi + 1 < g1) {
                        nxt.template select<512, 1>(0) = es::block_load<uint8_t, 512>(cb + (gi + 1) * gstride, kStream);
                        nxt.template select<512, 1>(512) = es::block_load<uint8_t, 512>(cb + (gi + 1) * gstride + 512, kStream);
                        nsc = es::block_load<uint8_t, 128>(sb + (gi + 1) * sstride, kStream);
                    }
                    es::simd<float, 128> xf = es::block_load<float, 128>(x + gi * 128);
                    es::simd<sycl::ext::oneapi::bfloat16, 128> xb = xf;
                    es::simd<float, 128> scf = v_e4m3<128>(es::simd<uint32_t, 128>(sc8));
                    es::simd<float, 128> q1 = v_bfr<128>(scf * o), q3 = v_bfr<128>((scf * 1.5f) * o);
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        es::simd<uint8_t, 128> by = raw.template select<128, 1>(c * 128);
                        es::simd<uint32_t, 128> off = es::simd<uint32_t, 128>(by) << 2;
                        es::simd<uint32_t, 128> d1, d3;
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            es::simd<uint32_t, 16> o16 = off.template select<16, 1>(v * 16);
                            d1.template select<16, 1>(v * 16) = es::slm_gather<uint32_t, 16>(o16);
                            d3.template select<16, 1>(v * 16) = es::slm_gather<uint32_t, 16>(o16 + 1024u);
                        }
                        es::simd<sycl::ext::oneapi::bfloat16, 256> B1 = d1.template bit_cast_view<sycl::ext::oneapi::bfloat16>().read();
                        es::simd<sycl::ext::oneapi::bfloat16, 256> B3 = d3.template bit_cast_view<sycl::ext::oneapi::bfloat16>().read();
                        es::simd<sycl::ext::oneapi::bfloat16, 16> A = xb.template select<16, 1>(c * 16);
                        es::simd<float, 16> z = 0.f;
                        es::simd<float, 16> r1 = xmx::dpas<8, 1, float, float, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(z, B1, A);
                        es::simd<float, 16> r3 = xmx::dpas<8, 1, float, float, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(z, B3, A);
                        acc += q1.template select<16, 1>(c * 16) * r1 + q3.template select<16, 1>(c * 16) * r3;
                    }
                }
            }
            if constexpr (KS > 1) {
                es::slm_block_store<float, 16>(uint32_t(2048 + part * 16 * sizeof(float)), acc);
                es::barrier();
                if (part != 0) return;
                acc = 0.f;
#pragma unroll
                for (int p = 0; p < KS; ++p) acc += es::slm_block_load<float, 16>(uint32_t(2048 + p * 16 * sizeof(float)));
            }
            if (a.gate_up) {
                if (!own) return;
                es::simd<float, 16> r = v_bfr<16>(acc);
                es::simd<float, 8> gg = r.template select<8, 2>(0), uu = r.template select<8, 2>(1);
                es::simd<float, 8> out;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float g0 = gg[i], u0 = uu[i];
                    const float sg = g0 >= 0.f ? 1.f / (1.f + sycl::exp(-g0)) : sycl::exp(g0) / (1.f + sycl::exp(g0));
                    out[i] = (a.gcap * th_(g0 / a.gcap)) * sg * (a.ucap * th_(u0 / a.ucap));
                }
                out = v_bfr<8>(out);
                es::block_store<float, 8>(a.out + size_t(s) * (a.N / 2) + t * 8, out);
            } else {
                es::simd<float, 16> r = own ? v_bfr<16>(acc) : es::simd<float, 16>(0.f);
                es::block_store<float, 16>(a.out + size_t(s) * a.N + t * 16, r);
            }
        });
    });
}

sycl::event ink_xexp(sycl::queue& q, const InkXArgs& g) {
    static const int mode = std::getenv("INK_XEXP_MODE") ? std::atoi(std::getenv("INK_XEXP_MODE")) : 0;
    if (mode == 1) return xexp_run1(q, g);
    const int G = g.K / 128;
    if (G % 6 == 0) return xexp_lut<6>(q, g);
    if (G % 3 == 0) return xexp_lut<3>(q, g);
    return xexp_lut<1>(q, g);
}
