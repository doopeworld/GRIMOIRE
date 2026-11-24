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

// moved from inkling2_moe_gemm.cpp 11-24: this library runs 8 threads per EU (128 GRF) -- twice the loads in flight
// ---------------------------------------------------------------------------------------------------------------
// Verify experts (M <= 8 rows a step) = GRIMOIRE's moe_mxfp4_esimd_small (src/gemm_fast.cpp) ported to NVFP4 / MXFP8:
// a thread owns 32 weight rows (two 16-row halves) of one used expert and streams them once, decoding each 32-K block
// into DPAS B tiles in registers (no SLM sharing, no barrier); the expert's rows of A (bf16, slot order, <= 8) come
// from L1 with one 2-D load per 32 K.  Groups from InkGroup: gi[0] = used experts, gi[1 + u] = expert id,
// gi[129 + u] .. gi[130 + u] = its slots.  EPI 1: softcapped SwiGLU -> bf16 act [slot][Ne/2]; EPI 2: bf(out) as fp32.
// ---------------------------------------------------------------------------------------------------------------
namespace { template <int F, int EPI, int KS> class InkMoeSmE; }
struct InkSmArgs {
    const uint8_t* pay; const uint8_t* scl; int Ne, K, rows_total;
    const uint16_t* A; int slots;
    void* out;
    const int* gi; int e0, en;
    const float* amax; const float* den;
    const uint32_t* lut;
    float gcap, ucap;
    int U;
    int alu = 1;   // NVFP4 nibble decode: 1 ALU bit trick, 0 SLM table
    int own = -1;  // >= 0: every expert on this card; used expert u is this card's iff u % 2 == own
};
template <int F, int EPI, int KS>
static sycl::event ink_moe_small_impl(sycl::queue& q, const InkSmArgs& g) {
    // 11-24 v2: weights decode to fp16 with NO per-weight arithmetic (NVFP4: byte -> fp16 pair from a 1 KB SLM table;
    // MXFP8: E4M3 byte -> fp16 bits = value x 2^-8), fp16 DPAS per 16-K (NVFP4) / 32-K (MXFP8) block, the block's
    // scale (E4M3 x outer, or 2^(x - 119)) applied once to the 8 x 16 partial.  The bf16 version spent ~1 ALU
    // instruction per weight on the decode and ran ALU-bound (360 us / visit for gate_up).  A rows are fp16 (bf16
    // values, exact); act out (EPI 1) is fp16 for the down kernel; dbuf (EPI 2) = bf(out) as fp32.
    constexpr int TNT = 8;
    const int nG = g.Ne / (32 * TNT);
    return q.submit([&](sycl::handler& h) {
        const InkSmArgs a = g;
        h.parallel_for<InkMoeSmE<F, EPI, KS>>(sycl::nd_range<1>(size_t(a.U) * nG * TNT * KS, TNT * KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            using hf = sycl::half;
            using bfx = sycl::ext::oneapi::bfloat16;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            es::slm_init<1024 + (KS - 1) * TNT * 1024>();   // LUT, then the K-split partial sums
            const int lid0 = int(it.get_local_id(0)), grp = int(it.get_group(0));
            const int lid = lid0 % TNT, ks = lid0 / TNT;     // ks: this thread's K slice (KS threads per 32 columns)
            const int u = grp / nG, ng = grp % nG;
            if (F == 0 && lid0 < 8) es::slm_block_store<uint32_t, 32>(lid0 * 128, es::block_load<uint32_t, 32>(a.lut + lid0 * 32));
            if (F == 0) es::barrier();
            if (u >= a.gi[0]) return;
            const int e = a.gi[1 + u];
            if (a.own >= 0 ? (u & 1) != a.own : (e < a.e0 || e >= a.e0 + a.en)) return;
            const int el = a.own >= 0 ? e : e - a.e0, p0 = a.gi[129 + u], np = a.gi[130 + u] - p0;
            const float o = F == 0 ? a.amax[e] / a.den[0] : 1.f;
            const int nb = ng * TNT + lid;
            const int rh0 = el * a.Ne + nb * 32, rh1 = rh0 + 16;
            const hf* Ab = reinterpret_cast<const hf*>(a.A);
            const unsigned AW = unsigned(a.K) * 2 - 1, AH = unsigned(a.slots) - 1;
            const uint32_t* payw = reinterpret_cast<const uint32_t*>(a.pay);
            const unsigned PW = unsigned(F == 0 ? a.K / 2 : a.K) - 1, PH = unsigned(a.rows_total) - 1, PP = PW;
            const uint32_t* sclw = reinterpret_cast<const uint32_t*>(a.scl);
            const unsigned SB = unsigned(a.K) / (F == 0 ? 16 : 32);
            const es::simd<uint32_t, 16> iv(0, 1);
            const es::simd<float, 128> zero = 0.0f;
            es::simd<float, 2 * 128> acc = 0.0f;
            const bool perm = F == 0 && (a.alu & 4) != 0;
            const bool alu = (a.alu & 1) != 0 || perm, whole = (a.alu & 2) != 0, dbuf = (a.alu & 16) != 0;
            // NVFP4 whole-step mode: register double buffer -- step k + 128's payload and scales are loaded before step k
            // computes (the per-step load -> use chain left the EUs at ~0.2 IPC)
            es::simd<uint32_t, 256> nx0, nx1;
            es::simd<uint32_t, 16> n0a, n0b, n1a, n1b;
            auto load_step = [&](int kn) SYCL_ESIMD_FUNCTION {
                nx0 = es::load_2d<uint32_t, 16, 16, 1, false, false>(payw, PW, PH, PP, kn / 8, rh0);
                nx1 = es::load_2d<uint32_t, 16, 16, 1, false, false>(payw, PW, PH, PP, kn / 8, rh1);
                const es::simd<uint32_t, 16> so0 = (iv + unsigned(rh0)) * SB + unsigned(kn / 16), so1 = (iv + unsigned(rh1)) * SB + unsigned(kn / 16);
                n0a = es::gather<uint32_t, 16>(sclw, so0); n0b = es::gather<uint32_t, 16>(sclw, so0 + 4u);
                n1a = es::gather<uint32_t, 16>(sclw, so1); n1b = es::gather<uint32_t, 16>(sclw, so1 + 4u);
            };
            // K slices: contiguous (ks * K/KS ..) or interleaved 128-K steps (ks, ks + KS, ..: the KS threads of a group
            // read neighbouring 64-byte pieces of the same rows at the same time)
            const bool ilv = (a.alu & 32) || a.own >= 0;   // two cards: interleaved K slices (GRIMOIRE int4_smallm ilv)
            const int KK = a.K / KS, kbeg = ilv ? ks * 128 : ks * KK, kend = ilv ? a.K : kbeg + KK, kstep = ilv ? 128 * KS : 128;
            if (F == 0 && whole && dbuf) load_step(kbeg);
            for (int k = kbeg; k < kend; k += kstep) {
                if (F == 0 && whole && dbuf) {
                    // fall through: the buffers are consumed below, then refilled after the copy
                } else if (k + kstep < kend) {
                    if constexpr (F == 0) {
                        es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PP, (k + 128) / 8, rh0, PFH);
                        es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PP, (k + 128) / 8, rh1, PFH);
                    } else {
#pragma unroll
                        for (int pf = 0; pf < 2; ++pf) {
                            es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PP, (k + 128) / 4 + 16 * pf, rh0, PFH);
                            es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PP, (k + 128) / 4 + 16 * pf, rh1, PFH);
                        }
                    }
                }
                es::simd<uint32_t, 256> st0, st1;   // NVFP4: this step's payload, [16 rows][16 dwords] per half
                es::simd<uint32_t, 16> s0a, s0b, s1a, s1b;
                if (F == 0 && whole && dbuf) {
                    st0 = nx0; st1 = nx1; s0a = n0a; s0b = n0b; s1a = n1a; s1b = n1b;
                    if (k + kstep < kend) load_step(k + kstep);
                } else {
                if constexpr (F == 0) {
                    if (whole) {
                        st0 = es::load_2d<uint32_t, 16, 16, 1, false, false>(payw, PW, PH, PP, k / 8, rh0);
                        st1 = es::load_2d<uint32_t, 16, 16, 1, false, false>(payw, PW, PH, PP, k / 8, rh1);
                    }
                }
                if constexpr (F == 0) {
                    const es::simd<uint32_t, 16> so0 = (iv + unsigned(rh0)) * SB + unsigned(k / 16), so1 = (iv + unsigned(rh1)) * SB + unsigned(k / 16);
                    s0a = es::gather<uint32_t, 16>(sclw, so0); s0b = es::gather<uint32_t, 16>(sclw, so0 + 4u);
                    s1a = es::gather<uint32_t, 16>(sclw, so1); s1b = es::gather<uint32_t, 16>(sclw, so1 + 4u);
                } else {
                    s0a = es::gather<uint32_t, 16>(sclw, (iv + unsigned(rh0)) * SB + unsigned(k / 32));
                    s1a = es::gather<uint32_t, 16>(sclw, (iv + unsigned(rh1)) * SB + unsigned(k / 32));
                }
                }
#pragma unroll 1
                for (int b = 0; b < 4; ++b) {
                    const int kk = k + 32 * b;
                    es::simd<hf, 256> av = es::load_2d<hf, 16, 8, 2>(Ab, AW, AH, AW, kk, p0);
                    const es::simd<hf, 128> a0 = av.template select<128, 1>(0), a1 = av.template select<128, 1>(128);
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        const int rh = hh ? rh1 : rh0;
                        if constexpr (F == 0) {
                            es::simd<uint32_t, 64> tw;
                            if (whole) {
#pragma unroll
                                for (int d4 = 0; d4 < 4; ++d4)
                                    tw.template select<16, 1>(d4 * 16) = hh ? es::simd<uint32_t, 16>(st1.template select<16, 16>(4 * b + d4))
                                                                            : es::simd<uint32_t, 16>(st0.template select<16, 16>(4 * b + d4));
                            } else
                                tw = es::load_2d<uint32_t, 4, 16, 1, true, false>(payw, PW, PH, PP, kk / 8, rh);
#pragma unroll
                            for (int hk = 0; hk < 2; ++hk) {
                                const int sbyte = 2 * b + hk;
                                const es::simd<uint32_t, 16> sw = hh ? (sbyte < 4 ? s1a : s1b) : (sbyte < 4 ? s0a : s0b);
                                const es::simd<uint32_t, 16> ev = (sw >> (8 * (sbyte & 3))) & 0xFFu;
                                const es::simd<uint32_t, 16> ee = (ev >> 3) & 15u, mm = ev & 7u;
                                es::simd<uint32_t, 16> nbits = ((ee + 120u) << 23) | (mm << 20);
                                es::simd<float, 16> s16 = nbits.template bit_cast_view<float>().read();
                                s16.merge(es::simd<float, 16>(mm) * (1.f / 512.f), ee == 0u);
                                s16 = s16 * (alu ? o * 16384.0f : o);   // the ALU nibble decode below is value x 2^-14
                                es::simd<uint32_t, 128> vb;
                                if (perm) {
                                    // K order within each 8: 0 4 1 5 2 6 3 7 (A rows are written in the same order): a dword
                                    // of nibbles k0..k7 gives the fp16 pairs (k0,k4) (k1,k5) (k2,k6) (k3,k7) with 2 shifts,
                                    // 2 ANDs and 1 OR each (magnitude at bits 9..11 / 25..27, sign at 15 / 31 = value x 2^-14)
#pragma unroll
                                    for (int d2 = 0; d2 < 2; ++d2) {
                                        const es::simd<uint32_t, 16> wx = tw.template select<16, 1>((2 * hk + d2) * 16);
                                        vb.template select<16, 1>((4 * d2 + 0) * 16) = ((wx << 9) & 0x0E000E00u) | ((wx << 12) & 0x80008000u);
                                        vb.template select<16, 1>((4 * d2 + 1) * 16) = ((wx << 5) & 0x0E000E00u) | ((wx << 8) & 0x80008000u);
                                        vb.template select<16, 1>((4 * d2 + 2) * 16) = ((wx << 1) & 0x0E000E00u) | ((wx << 4) & 0x80008000u);
                                        vb.template select<16, 1>((4 * d2 + 3) * 16) = ((wx >> 3) & 0x0E000E00u) | (wx & 0x80008000u);
                                    }
                                } else
#pragma unroll
                                for (int kp = 0; kp < 8; ++kp) {
                                    const int dw = 2 * hk + (kp >> 2), j = kp & 3;
                                    const es::simd<uint32_t, 16> wx = tw.template select<16, 1>(dw * 16);
                                    if (alu) {
                                        // E2M1 nibble -> fp16 bits: magnitude bits at 11..9, sign at 15 = value x 2^-14 (exact)
                                        const es::simd<uint32_t, 16> w = j == 0 ? wx : es::simd<uint32_t, 16>(wx >> (8 * j));
                                        vb.template select<16, 1>(kp * 16) = ((w << 9) & 0x0E00u) | ((w << 12) & 0x8000u) | ((w << 21) & 0x0E000000u) | ((w << 24) & 0x80000000u);
                                    } else {
                                        const es::simd<uint32_t, 16> addr = j == 0 ? es::simd<uint32_t, 16>((wx << 2) & 0x3FCu) : es::simd<uint32_t, 16>((wx >> (8 * j - 2)) & 0x3FCu);
                                        es::simd<uint32_t, 16> pr = es::slm_gather<uint32_t, 16>(addr);
                                        // table pairs are plain fp16 values: scale them like the ALU path (x 2^-14) -> fold back
                                        vb.template select<16, 1>(kp * 16) = pr;
                                    }
                                }
                                es::simd<float, 128> c = xmx::dpas<8, 8, float, float, hf, hf>(zero, es::simd<hf, 256>(vb.template bit_cast_view<hf>()),
                                                                                                    hk ? a1 : a0);
#pragma unroll
                                for (int r = 0; r < 8; ++r)
                                    acc.template select<16, 1>(hh * 128 + r * 16) = es::simd<float, 16>(acc.template select<16, 1>(hh * 128 + r * 16)) +
                                                                                    es::simd<float, 16>(c.template select<16, 1>(r * 16)) * s16;
                            }
                        } else {
                            es::simd<uint32_t, 128> tw = es::load_2d<uint32_t, 8, 16, 1, true, false>(payw, PW, PH, PP, kk / 4, rh);
                            const es::simd<uint32_t, 16> sw = hh ? s1a : s0a;
                            const es::simd<uint32_t, 16> xb = (sw >> (8 * b)) & 0xFFu;
                            es::simd<uint32_t, 16> sbits = (xb + 8u) << 23;
                            const es::simd<float, 16> s16 = sbits.template bit_cast_view<float>().read();
                            es::simd<uint32_t, 256> vb;
#pragma unroll
                            for (int dw = 0; dw < 8; ++dw) {
                                const es::simd<uint32_t, 16> wx = tw.template select<16, 1>(dw * 16);
#pragma unroll
                                for (int hf2 = 0; hf2 < 2; ++hf2) {
                                    const es::simd<uint32_t, 16> w = hf2 ? es::simd<uint32_t, 16>(wx >> 16) : wx;
                                    vb.template select<16, 1>((2 * dw + hf2) * 16) = ((w & 0x7Fu) << 7) | ((w & 0x80u) << 8) | ((w & 0x7F00u) << 15) | ((w & 0x8000u) << 16);
                                }
                            }
                            es::simd<float, 128> c = xmx::dpas<8, 8, float, float, hf, hf>(zero, es::simd<hf, 256>(vb.template select<128, 1>(0).template bit_cast_view<hf>()), a0);
                            c = xmx::dpas<8, 8, float, float, hf, hf>(c, es::simd<hf, 256>(vb.template select<128, 1>(128).template bit_cast_view<hf>()), a1);
#pragma unroll
                            for (int r = 0; r < 8; ++r)
                                acc.template select<16, 1>(hh * 128 + r * 16) = es::simd<float, 16>(acc.template select<16, 1>(hh * 128 + r * 16)) +
                                                                                es::simd<float, 16>(c.template select<16, 1>(r * 16)) * s16;
                        }
                    }
                }
            }
            if constexpr (KS > 1) {   // K-split partials meet in SLM, summed in slice order by the slice-0 thread
                if (ks > 0) es::slm_block_store<float, 256>(1024u + unsigned((ks - 1) * TNT + lid) * 1024u, acc);
                es::barrier();
                if (ks > 0) return;
#pragma unroll
                for (int j = 1; j < KS; ++j) acc += es::slm_block_load<float, 256>(1024u + unsigned((j - 1) * TNT + lid) * 1024u);
            }
            const unsigned OH = unsigned(p0 + np) - 1;   // rows past this expert's slots are clipped by the 2-D store
            if constexpr (EPI == 1) {
                constexpr float L2E = 1.4426950408889634f;
                const float gc = a.gcap, uc = a.ucap;
                auto th = [](es::simd<float, 64> x) SYCL_ESIMD_FUNCTION {
                    const es::simd<float, 64> t = es::exp2(es::abs(x) * (-2.0f * 1.4426950408889634f));
                    es::simd<float, 64> r = (1.0f - t) / (1.0f + t);
                    r.merge(-r, x < 0.0f);
                    return r;
                };
                es::simd<uint16_t, 128> ov;
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    es::simd<float, 128> v = acc.template select<128, 1>(hh * 128);
                    es::simd<float, 64> gv = v.template select<64, 2>(0), uv = v.template select<64, 2>(1);
                    gv = es::simd<float, 64>(es::simd<bfx, 64>(gv));
                    uv = es::simd<float, 64>(es::simd<bfx, 64>(uv));
                    const es::simd<float, 64> z = es::exp2(es::abs(gv) * -L2E);
                    es::simd<float, 64> num = 1.0f;
                    num.merge(z, gv < 0.0f);
                    const es::simd<float, 64> r = (gc * th(gv / gc)) * (num / (1.0f + z)) * (uc * th(uv / uc));
                    const es::simd<float, 64> rbf = es::simd<bfx, 64>(r);
                    es::simd<hf, 64> rh16 = rbf;   // bf16 value (reference), stored as fp16
                    es::simd<uint16_t, 64> rbits = rh16.template bit_cast_view<uint16_t>().read();
                    if (a.alu & 8) {   // act in the 0 4 1 5 2 6 3 7 order the NVFP4 down kernel reads
                        es::simd<uint16_t, 64> pr;
#pragma unroll
                        for (int r8 = 0; r8 < 8; ++r8) {
                            pr.template select<4, 2>(r8 * 8) = rbits.template select<4, 1>(r8 * 8);
                            pr.template select<4, 2>(r8 * 8 + 1) = rbits.template select<4, 1>(r8 * 8 + 4);
                        }
                        rbits = pr;
                    }
#pragma unroll
                    for (int r8 = 0; r8 < 8; ++r8) ov.template select<8, 1>(r8 * 16 + hh * 8) = rbits.template select<8, 1>(r8 * 8);
                }
                const unsigned HW = unsigned(a.Ne / 2) * 2 - 1;
                es::store_2d<uint16_t, 16, 8>(static_cast<uint16_t*>(a.out), HW, OH, HW, nb * 16, p0, ov);
            } else {
                float* Of = static_cast<float*>(a.out);
                const unsigned OW = unsigned(a.Ne) * 4 - 1;
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    es::simd<float, 128> v = acc.template select<128, 1>(hh * 128);
                    if constexpr (EPI == 2) v = es::simd<float, 128>(es::simd<bfx, 128>(v));
                    es::store_2d<float, 16, 8>(Of, OW, OH, OW, nb * 32 + 16 * hh, p0, v);
                }
            }
        });
    });
}
sycl::event ink_moe_small_esimd(sycl::queue& q, const InkSmArgs& g, int fmt, int epi) {
    static const int kenv = [] { const char* e = std::getenv("INK_SM_KS"); return e ? std::atoi(e) : 0; }();
    const int ksel = kenv > 0 ? kenv : 6;   // MEASURED 11-24: one card KS 3/6 = 38.4/38.3 ms (KS 1 48.0); two cards KS 6 ilv 24.8 vs KS 3 25.2 ms/step (down K 1152 -> 3)
    int ks = 1;
    for (int c : {6, 3, 2}) if (c <= ksel && g.K % (128 * c) == 0) { ks = c; break; }
    if (ks == 6) {
        if (fmt == 0) return epi == 1 ? ink_moe_small_impl<0, 1, 6>(q, g) : ink_moe_small_impl<0, 2, 6>(q, g);
        return epi == 1 ? ink_moe_small_impl<1, 1, 6>(q, g) : ink_moe_small_impl<1, 2, 6>(q, g);
    }
    if (ks == 3) {
        if (fmt == 0) return epi == 1 ? ink_moe_small_impl<0, 1, 3>(q, g) : ink_moe_small_impl<0, 2, 3>(q, g);
        return epi == 1 ? ink_moe_small_impl<1, 1, 3>(q, g) : ink_moe_small_impl<1, 2, 3>(q, g);
    }
    if (ks == 2) {
        if (fmt == 0) return epi == 1 ? ink_moe_small_impl<0, 1, 2>(q, g) : ink_moe_small_impl<0, 2, 2>(q, g);
        return epi == 1 ? ink_moe_small_impl<1, 1, 2>(q, g) : ink_moe_small_impl<1, 2, 2>(q, g);
    }
    if (fmt == 0) return epi == 1 ? ink_moe_small_impl<0, 1, 1>(q, g) : ink_moe_small_impl<0, 2, 1>(q, g);
    return epi == 1 ? ink_moe_small_impl<1, 1, 1>(q, g) : ink_moe_small_impl<1, 2, 1>(q, g);
}

// ---------------------------------------------------------------------------------------------------------------
// Cross-pass message gate blend for 2..8 rows on the matrix unit (the GRIMOIRE small-M pattern: a thread owns 32
// output columns, the weight rows come in as transposed 2-D tiles = DPAS B operands, A = the BF16-rounded low rows):
//   G = low . gain^T, F = low . forget^T  (K = rank = 64),  out = (1 - tanh F) cur + tanh G msg   (fp32)
// ---------------------------------------------------------------------------------------------------------------
namespace { class InkGBlendE; }
struct InkGBArgs {
    const float* low; int ldlow;              // [job][MM][64] fp32 (BF16-rounded on use)
    const uint16_t* gain[3]; const uint16_t* forget[3];   // [h][64] bf16 per job
    float* pool; size_t RS; int cur[3], msg[3], out[3];
    int count, M, h;
};
sycl::event ink_gate_blend_esimd(sycl::queue& q, const InkGBArgs& g) {
    const int nb = g.h / 32;
    return q.submit([&](sycl::handler& h) {
        const InkGBArgs a = g;
        h.parallel_for<InkGBlendE>(sycl::nd_range<1>(size_t(a.count) * nb, 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            using bfx = sycl::ext::oneapi::bfloat16;
            const int gi = int(it.get_group(0)), jb = gi / nb, n0 = (gi % nb) * 32;
            const uint16_t* gw = jb == 0 ? a.gain[0] : (jb == 1 ? a.gain[1] : a.gain[2]);
            const uint16_t* fw = jb == 0 ? a.forget[0] : (jb == 1 ? a.forget[1] : a.forget[2]);
            const int cur = jb == 0 ? a.cur[0] : (jb == 1 ? a.cur[1] : a.cur[2]);
            const int msg = jb == 0 ? a.msg[0] : (jb == 1 ? a.msg[1] : a.msg[2]);
            const int out = jb == 0 ? a.out[0] : (jb == 1 ? a.out[1] : a.out[2]);
            // A: 8 rows x 64 (rows >= M read as 0 by the 2-D load), BF16-rounded
            const float* lw = a.low + size_t(jb) * a.ldlow;
            const unsigned LW = 64u * 4 - 1, LH = unsigned(a.M) - 1;
            es::simd<bfx, 128> av[4];
#pragma unroll
            for (int kc = 0; kc < 4; ++kc) av[kc] = es::simd<bfx, 128>(es::load_2d<float, 16, 8>(lw, LW, LH, LW, kc * 16, 0));
            // B: weight rows n0 .. n0 + 31, K = 64: transposed u32 tiles [8 K-pairs][16 rows] per 16 K
            const uint32_t* g32 = reinterpret_cast<const uint32_t*>(gw);
            const uint32_t* f32 = reinterpret_cast<const uint32_t*>(fw);
            const unsigned BW = 64u * 2 - 1, BH = unsigned(a.h) - 1;
            es::simd<uint32_t, 128> bg[2][4], bf[2][4];
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
                for (int kc = 0; kc < 4; ++kc) {
                    bg[hh][kc] = es::load_2d<uint32_t, 8, 16, 1, true, false>(g32, BW, BH, BW, kc * 8, n0 + 16 * hh);
                    bf[hh][kc] = es::load_2d<uint32_t, 8, 16, 1, true, false>(f32, BW, BH, BW, kc * 8, n0 + 16 * hh);
                }
            const unsigned PWd = unsigned(a.h) * 4 - 1, PHd = unsigned(a.M) - 1;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                es::simd<float, 128> G = 0.0f, F = 0.0f;
#pragma unroll
                for (int kc = 0; kc < 4; ++kc) {
                    G = xmx::dpas<8, 8, float, float, bfx, bfx>(G, es::simd<bfx, 256>(bg[hh][kc].template bit_cast_view<bfx>()), av[kc]);
                    F = xmx::dpas<8, 8, float, float, bfx, bfx>(F, es::simd<bfx, 256>(bf[hh][kc].template bit_cast_view<bfx>()), av[kc]);
                }
                const float* cp = a.pool + size_t(cur) * a.RS;
                const float* mp = a.pool + size_t(msg) * a.RS;
                float* op = a.pool + size_t(out) * a.RS;
                es::simd<float, 128> cv = es::load_2d<float, 16, 8>(cp, PWd, PHd, PWd, n0 + 16 * hh, 0);
                es::simd<float, 128> mv = es::load_2d<float, 16, 8>(mp, PWd, PHd, PWd, n0 + 16 * hh, 0);
                auto th = [](es::simd<float, 128> x) SYCL_ESIMD_FUNCTION {
                    const es::simd<float, 128> t = es::exp2(es::abs(x) * (-2.0f * 1.4426950408889634f));
                    es::simd<float, 128> r = (1.0f - t) / (1.0f + t);
                    r.merge(-r, x < 0.0f);
                    return r;
                };
                const es::simd<float, 128> o = (1.0f - th(F)) * cv + th(G) * mv;
                es::store_2d<float, 16, 8>(op, PWd, PHd, PWd, n0 + 16 * hh, 0, o);
            }
        });
    });
}
