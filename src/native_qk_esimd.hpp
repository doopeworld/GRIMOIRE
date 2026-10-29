// Vector native Q4_K/Q5_K/Q6_K decode dots and the dense GEMV using them.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "kernels.hpp"
#include "b70/native_qk_gemv.hpp"
#include "b70/native_qk_vector_math.hpp"
#include <sycl/ext/intel/esimd.hpp>

namespace b70 {
namespace nqes = sycl::ext::intel::esimd;

// Accumulator lanes: Q4_K has 32-value groups, Q6_K 16-value groups. A row
// keeps ONE vector accumulator over all its blocks and reduces it once.
template <Fmt F> constexpr int nq_acc_lanes() { return F == Fmt::Q4_K || F == Fmt::Q5_K ? 32 : 16; }
// The IQ4_NL / IQ4_XS value table (formats.hpp iq4nl_value) as a register vector, for iselect.
SYCL_ESIMD_FUNCTION inline nqes::simd<float, 16> nq_iq4_table() {
    constexpr float v[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                             1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};
    nqes::simd<float, 16> t;
    #pragma unroll
    for (int i = 0; i < 16; ++i) t[i] = v[i];
    return t;
}
template <Fmt F> using nq_acc_t = nqes::simd<float, nq_acc_lanes<F>()>;

// acc += (decoded 256 values of the block at p) * x[xoff .. xoff+255].
// Every byte of a Q4_K block is loaded up front (16 B header + 128 quant
// bytes: 144-byte blocks are 16-byte aligned so plain block loads are legal)
// and the scales are decoded as vectors, so the loads of several rows/blocks
// overlap. Q6_K blocks are 210 bytes (2-byte aligned only): byte gathers.
template <Fmt F, bool ODD, int NX>
SYCL_ESIMD_FUNCTION inline void native_qk_esacc_256(
        const uint8_t* p, nqes::simd<float, NX>& x, int xoff, nq_acc_t<F>& acc) {
    static_assert(F == Fmt::Q4_K || F == Fmt::Q5_K || F == Fmt::Q6_K || F == Fmt::IQ4_XS);
    if constexpr (F == Fmt::IQ4_XS) {
        // IQ4_XS (2026-10-29): 136-byte blocks of 256 -- d (fp16), scales_h (2 bits a sub-block),
        // scales_l (4 bits a sub-block), then 8 sub-blocks of 32 as 16 bytes each: low nibbles =
        // values 0..15, high nibbles = 16..31, each an index into the IQ4_NL table (iselect).
        // Blocks are 8-byte aligned; every load here starts on one.
        nqes::simd<uint8_t, 8> hdr = nqes::block_load<uint8_t, 8>(p);
        nqes::simd<uint8_t, 128> qs = nqes::block_load<uint8_t, 128>(p + 8);
        const float d = float(sycl::bit_cast<sycl::half>(
            uint16_t(uint32_t(hdr[0]) | (uint32_t(hdr[1]) << 8))));
        const uint32_t h = uint32_t(hdr[2]) | (uint32_t(hdr[3]) << 8);
        nqes::simd<float, 16> tab = nq_iq4_table();
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const uint32_t low = (uint32_t(hdr[4 + l / 2]) >> (4 * (l % 2))) & 15;
            const float sc = d * float(int(low | (((h >> (2 * l)) & 3) << 4)) - 32);
            nqes::simd<uint16_t, 16> b = qs.template select<16, 1>(16 * l);
            nqes::simd<uint16_t, 16> i0 = b & 15;
            nqes::simd<uint16_t, 16> i1 = b >> 4;
            nqes::simd<float, 16> v0 = tab.iselect(i0);
            nqes::simd<float, 16> v1 = tab.iselect(i1);
            acc += (v0 * sc) * x.template select<16, 1>(xoff + 32 * l);
            acc += (v1 * sc) * x.template select<16, 1>(xoff + 32 * l + 16);
        }
    } else if constexpr (F == Fmt::Q5_K) {
        // Q5_K (2026-10-29): Q4_K's header (d, dmin, 12 scale/min bytes), then qh (32 B: bit g of
        // byte l = the fifth bit of value l of sub-block g) and qs (128 B, sub-block 2j in the low
        // nibbles of bytes 32j..32j+31, 2j+1 in the high ones) -- 176-byte blocks, 16-byte aligned.
        // qh / qs are widened to 16 bits before any shift: the B70 compiler has miscompiled byte
        // shifts (native_qk_gemv.hpp, tools/test_q5k_bits.cpp); every shift here is a constant.
        nqes::simd<uint8_t, 16> hdr = nqes::block_load<uint8_t, 16>(p);
        nqes::simd<uint8_t, 32> qh8 = nqes::block_load<uint8_t, 32>(p + 16);
        nqes::simd<uint8_t, 128> qs = nqes::block_load<uint8_t, 128>(p + 48);
        const float d = float(sycl::bit_cast<sycl::half>(
            uint16_t(uint32_t(hdr[0]) | (uint32_t(hdr[1]) << 8))));
        const float dm = float(sycl::bit_cast<sycl::half>(
            uint16_t(uint32_t(hdr[2]) | (uint32_t(hdr[3]) << 8))));
        nqes::simd<uint8_t, 4> s0 = hdr.template select<4, 1>(4);
        nqes::simd<uint8_t, 4> s1 = hdr.template select<4, 1>(8);
        nqes::simd<uint8_t, 4> s2 = hdr.template select<4, 1>(12);
        nqes::simd<uint8_t, 8> sc, mn;
        sc.template select<4, 1>(0) = s0 & 63;
        sc.template select<4, 1>(4) = (s2 & 15) | ((s0 >> 2) & 48);
        mn.template select<4, 1>(0) = s1 & 63;
        mn.template select<4, 1>(4) = (s2 >> 4) | ((s1 >> 2) & 48);
        nqes::simd<float, 8> scf = nqes::convert<float>(sc) * d;
        nqes::simd<float, 8> mnf = nqes::convert<float>(mn) * dm;
        nqes::simd<uint16_t, 32> qh = qh8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            nqes::simd<uint16_t, 32> b = qs.template select<32, 1>(32 * j);
            nqes::simd<uint16_t, 32> lo = (b & 15) | (((qh >> (2 * j)) & 1) << 4);
            nqes::simd<uint16_t, 32> hi = (b >> 4) | (((qh >> (2 * j + 1)) & 1) << 4);
            nqes::simd<float, 32> flo = nqes::convert<float>(lo);
            nqes::simd<float, 32> fhi = nqes::convert<float>(hi);
            const float sl = scf[2 * j], ml = mnf[2 * j];
            const float sh = scf[2 * j + 1], mh = mnf[2 * j + 1];
            acc += (flo * sl - ml) * x.template select<32, 1>(xoff + 64 * j);
            acc += (fhi * sh - mh) * x.template select<32, 1>(xoff + 64 * j + 32);
        }
    } else if constexpr (F == Fmt::Q4_K) {
        nqes::simd<uint8_t, 16> hdr = nqes::block_load<uint8_t, 16>(p);
        nqes::simd<uint8_t, 128> qs = nqes::block_load<uint8_t, 128>(p + 16);
        const float d = float(sycl::bit_cast<sycl::half>(
            uint16_t(uint32_t(hdr[0]) | (uint32_t(hdr[1]) << 8))));
        const float dm = float(sycl::bit_cast<sycl::half>(
            uint16_t(uint32_t(hdr[2]) | (uint32_t(hdr[3]) << 8))));
        nqes::simd<uint8_t, 4> s0 = hdr.template select<4, 1>(4);
        nqes::simd<uint8_t, 4> s1 = hdr.template select<4, 1>(8);
        nqes::simd<uint8_t, 4> s2 = hdr.template select<4, 1>(12);
        nqes::simd<uint8_t, 8> sc, mn;
        sc.template select<4, 1>(0) = s0 & 63;
        sc.template select<4, 1>(4) = (s2 & 15) | ((s0 >> 2) & 48);
        mn.template select<4, 1>(0) = s1 & 63;
        mn.template select<4, 1>(4) = (s2 >> 4) | ((s1 >> 2) & 48);
        nqes::simd<float, 8> scf = nqes::convert<float>(sc) * d;
        nqes::simd<float, 8> mnf = nqes::convert<float>(mn) * dm;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            nqes::simd<uint8_t, 32> b = qs.template select<32, 1>(32 * j);
            nqes::simd<uint8_t, 32> lo = b & 15;
            nqes::simd<uint8_t, 32> hi = b >> 4;
            nqes::simd<float, 32> flo = nqes::convert<float>(lo);
            nqes::simd<float, 32> fhi = nqes::convert<float>(hi);
            const float sl = scf[2 * j], ml = mnf[2 * j];
            const float sh = scf[2 * j + 1], mh = mnf[2 * j + 1];
            acc += (flo * sl - ml) * x.template select<32, 1>(xoff + 64 * j);
            acc += (fhi * sh - mh) * x.template select<32, 1>(xoff + 64 * j + 32);
        }
    } else {
        // Q6_K: 210-byte blocks. Block 2k of a row starts 4-byte aligned, block
        // 2k+1 two bytes later (ODD). Every load starts on a 4-byte boundary:
        // an odd block reads from p-2 and addresses its bytes at +2. 212 bytes
        // cover ql(128) qh(64) scales(16) d(2) for both parities, and never
        // cross the end of the row (rows hold an even number of blocks here).
        constexpr int S = ODD ? 2 : 0;
        const uint8_t* b = p - S;
        nqes::simd<uint8_t, 212> raw;
        raw.template select<128, 1>(0) = nqes::block_load<uint8_t, 128>(b);
        raw.template select<64, 1>(128) = nqes::block_load<uint8_t, 64>(b + 128);
        raw.template select<16, 1>(192) = nqes::block_load<uint8_t, 16>(b + 192);
        raw.template select<4, 1>(208) = nqes::block_load<uint8_t, 4>(b + 208);
        const float d = float(sycl::bit_cast<sycl::half>(
            uint16_t(uint32_t(raw[S + 208]) | (uint32_t(raw[S + 209]) << 8))));
        nqes::simd<uint8_t, 16> sc8 = raw.template select<16, 1>(S + 192);
        nqes::simd<int8_t, 16> sci = sc8.template bit_cast_view<int8_t>();
        nqes::simd<float, 16> scf = nqes::convert<float>(sci) * d;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            nqes::simd<uint8_t, 32> qA = raw.template select<32, 1>(S + 64 * h);
            nqes::simd<uint8_t, 32> qB = raw.template select<32, 1>(S + 64 * h + 32);
            nqes::simd<uint8_t, 32> qh = raw.template select<32, 1>(S + 128 + 32 * h);
            nqes::simd<uint8_t, 32> v[4];
            v[0] = (qA & 15) | ((qh & 3) << 4);
            v[1] = (qB & 15) | ((qh & 12) << 2);
            v[2] = (qA >> 4) | (qh & 48);
            v[3] = (qB >> 4) | ((qh & 192) >> 2);
            #pragma unroll
            for (int qd = 0; qd < 4; ++qd) {
                #pragma unroll
                for (int sb = 0; sb < 2; ++sb) {
                    const float sc = scf[h * 8 + qd * 2 + sb];
                    nqes::simd<uint8_t, 16> vb = v[qd].template select<16, 1>(16 * sb);
                    nqes::simd<float, 16> f = nqes::convert<float>(vb);
                    acc += (f * sc - 32.0f * sc) *
                           x.template select<16, 1>(xoff + h * 128 + qd * 32 + sb * 16);
                }
            }
        }
    }
}

// One block's dot, reduced (for kernels that want a scalar).
template <Fmt F, bool ODD, int NX>
SYCL_ESIMD_FUNCTION inline float native_qk_esdot_256(
        const uint8_t* p, nqes::simd<float, NX>& x, int xoff) {
    nq_acc_t<F> acc = 0.0f;
    native_qk_esacc_256<F, ODD>(p, x, xoff, acc);
    return nqes::reduce<float>(acc, std::plus<>());
}

template <Fmt F>
sycl::event native_qk_esgemv(sycl::queue& q, const QuantWeight& w,
                             const float* x, float* y,
                             const std::vector<sycl::event>& deps) {
    constexpr int KP = 512, R = 2, RP = 4, MAXT = 32;
    constexpr int BB = NativeGGUFBlock<F>::bytes;
    // At most MAXT threads a row pair; past MAXT slices of 512 each thread takes `per`
    // consecutive ones (2026-10-29: Qwen3.8-27B's ffn_down, K = 17408 = 34 slices, was refused
    // and ran the SIMT GEMV at ~85 GB/s).  per == 1 is the original kernel exactly.
    const int chunks = w.K / KP;
    const int per = (chunks + MAXT - 1) / MAXT;
    const int ks = (chunks + per - 1) / per;
    const int groups = (w.N + R - 1) / R;
    const uint8_t* pay = w.payload;
    const int64_t rb = w.row_bytes;
    const int N = w.N;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * ks, ks),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                nqes::slm_init<MAXT * RP * 4>();
                const int t = int(it.get_local_id(0));
                const int row0 = int(it.get_group(0)) * R;
                nqes::simd<float, RP> part = 0.0f;
                for (int c = 0; c < per; ++c) {
                    const int chunk = t * per + c;
                    if (chunk >= chunks) break;
                    const int kb = chunk * KP;
                    nqes::simd<float, KP> xv = nqes::block_load<float, KP>(x + kb);
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (row0 + r >= N) continue;
                        const uint8_t* p = pay + int64_t(row0 + r) * rb +
                            int64_t(kb / 256) * BB;
                        nq_acc_t<F> acc = 0.0f;
                        native_qk_esacc_256<F, false>(p, xv, 0, acc);
                        native_qk_esacc_256<F, true>(p + BB, xv, 256, acc);
                        part[r] += nqes::reduce<float>(acc, std::plus<>());
                    }
                }
                nqes::slm_block_store<float, RP>(t * RP * 4, part);
                nqes::barrier();
                if (t != 0) return;
                nqes::simd<float, RP> sum = 0.0f;
                for (int j = 0; j < ks; ++j)
                    sum += nqes::slm_block_load<float, RP>(j * RP * 4);
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    if (row0 + r < N) y[row0 + r] = sum[r];
            });
    });
}

// Layout the vector kernels rely on: whole rows of an EVEN number of blocks
// (K % 512 == 0), packed row stride, and a payload aligned for the loads they
// issue (Q4_K / Q5_K: 16 B, 144- / 176-byte blocks; Q6_K: 4 B, because an odd 210-byte block
// is read from two bytes before its start). Anything else keeps the checked
// SIMT GEMV (GRIMOIRE_GGUF_ESGEMV / GRIMOIRE_MOE_NATIVE_ESIMD select it too).
template <Fmt F>
inline bool native_qk_es_layout_ok(const QuantWeight& w) {
    constexpr int BB = NativeGGUFBlock<F>::bytes;
    constexpr uintptr_t AL = F == Fmt::Q4_K || F == Fmt::Q5_K ? 16 : F == Fmt::IQ4_XS ? 8 : 4;
    return native_qk_fast_weight<F>(w) && w.K % 512 == 0 &&
           w.row_bytes == int64_t(w.K / 256) * BB && w.row_bytes % int64_t(AL) == 0 &&
           reinterpret_cast<uintptr_t>(w.payload) % AL == 0;
}

inline bool native_qk_esgemv_try(sycl::queue& q, const QuantWeight& w,
                                  const float* x, float* y,
                                  const std::vector<sycl::event>& deps,
                                  sycl::event& out) {
    static const bool enabled = [] {
        const char* e = std::getenv("GRIMOIRE_GGUF_ESGEMV");
        return !(e && *e == '0');
    }();
    if (!enabled || !native_qk_fast_enabled() || !w.payload || w.scales || w.zeros ||
        w.row_scales || w.K < 512 || w.K % 512 || w.K / 512 > 128)
        return false;
    if (w.fmt == Fmt::Q4_K && native_qk_es_layout_ok<Fmt::Q4_K>(w)) {
        out = native_qk_esgemv<Fmt::Q4_K>(q, w, x, y, deps);
        return true;
    }
    if (w.fmt == Fmt::Q6_K && native_qk_es_layout_ok<Fmt::Q6_K>(w)) {
        out = native_qk_esgemv<Fmt::Q6_K>(q, w, x, y, deps);
        return true;
    }
    static const bool q5 = [] { const char* e = std::getenv("GRIMOIRE_GGUF_ESGEMV_Q5K");
        return !(e && *e == '0'); }();
    if (q5 && w.fmt == Fmt::Q5_K && native_qk_es_layout_ok<Fmt::Q5_K>(w)) {
        out = native_qk_esgemv<Fmt::Q5_K>(q, w, x, y, deps);
        return true;
    }
    static const bool iq4xs = [] { const char* e = std::getenv("GRIMOIRE_GGUF_ESGEMV_IQ4XS");
        return !(e && *e == '0'); }();
    if (iq4xs && w.fmt == Fmt::IQ4_XS && native_qk_es_layout_ok<Fmt::IQ4_XS>(w)) {
        out = native_qk_esgemv<Fmt::IQ4_XS>(q, w, x, y, deps);
        return true;
    }
    return false;
}

} // namespace b70
