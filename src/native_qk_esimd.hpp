// Vector native Q4_K/Q6_K decode dots and the dense GEMV using them.
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
template <Fmt F> constexpr int nq_acc_lanes() { return F == Fmt::Q4_K ? 32 : 16; }
template <Fmt F> using nq_acc_t = nqes::simd<float, nq_acc_lanes<F>()>;

// acc += (decoded 256 values of the block at p) * x[xoff .. xoff+255].
// Every byte of a Q4_K block is loaded up front (16 B header + 128 quant
// bytes: 144-byte blocks are 16-byte aligned so plain block loads are legal)
// and the scales are decoded as vectors, so the loads of several rows/blocks
// overlap. Q6_K blocks are 210 bytes (2-byte aligned only): byte gathers.
template <Fmt F, int NX>
SYCL_ESIMD_FUNCTION inline void native_qk_esacc_256(
        const uint8_t* p, nqes::simd<float, NX>& x, int xoff, nq_acc_t<F>& acc) {
    static_assert(F == Fmt::Q4_K || F == Fmt::Q6_K);
    if constexpr (F == Fmt::Q4_K) {
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
        const float d = f16_to_f32(uint16_t(p[208]) | (uint16_t(p[209]) << 8));
        const nqes::simd<uint32_t, 16> ix(0, 1);
        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            const int half = g / 8, quarter = (g % 8) / 2;
            const int off = (g & 1) * 16;
            const nqes::simd<uint8_t, 16> lo = nqes::gather<uint8_t, 16>(
                p + half * 64 + (quarter % 2) * 32 + off, ix);
            const nqes::simd<uint8_t, 16> hi = nqes::gather<uint8_t, 16>(
                p + 128 + half * 32 + off, ix);
            nqes::simd<uint16_t, 16> q = (nqes::simd<uint16_t, 16>(lo) >>
                ((quarter / 2) * 4)) & 15;
            q |= ((nqes::simd<uint16_t, 16>(hi) >> (quarter * 2)) & 3) << 4;
            const nqes::simd<float, 16> w =
                (nqes::convert<float>(q) - 32.0f) *
                (d * float(int8_t(p[192 + g])));
            acc += w * x.template select<16, 1>(xoff + g * 16);
        }
    }
}

// One block's dot, reduced (for kernels that want a scalar).
template <Fmt F, int NX>
SYCL_ESIMD_FUNCTION inline float native_qk_esdot_256(
        const uint8_t* p, nqes::simd<float, NX>& x, int xoff) {
    nq_acc_t<F> acc = 0.0f;
    native_qk_esacc_256<F>(p, x, xoff, acc);
    return nqes::reduce<float>(acc, std::plus<>());
}

template <Fmt F>
sycl::event native_qk_esgemv(sycl::queue& q, const QuantWeight& w,
                             const float* x, float* y,
                             const std::vector<sycl::event>& deps) {
    constexpr int KP = 512, R = 2, RP = 4;
    constexpr int BB = NativeGGUFBlock<F>::bytes;
    const int ks = w.K / KP;
    const int groups = (w.N + R - 1) / R;
    const uint8_t* pay = w.payload;
    const int64_t rb = w.row_bytes;
    const int N = w.N;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * ks, ks),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                nqes::slm_init<32 * RP * 4>();
                const int t = int(it.get_local_id(0));
                const int row0 = int(it.get_group(0)) * R;
                const int kb = t * KP;
                nqes::simd<float, KP> xv = nqes::block_load<float, KP>(x + kb);
                nqes::simd<float, RP> part = 0.0f;
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (row0 + r >= N) continue;
                    const uint8_t* p = pay + int64_t(row0 + r) * rb +
                        int64_t(kb / 256) * BB;
                    nq_acc_t<F> acc = 0.0f;
                    native_qk_esacc_256<F>(p, xv, 0, acc);
                    native_qk_esacc_256<F>(p + BB, xv, 256, acc);
                    part[r] = nqes::reduce<float>(acc, std::plus<>());
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

inline bool native_qk_esgemv_try(sycl::queue& q, const QuantWeight& w,
                                  const float* x, float* y,
                                  const std::vector<sycl::event>& deps,
                                  sycl::event& out) {
    static const bool enabled = [] {
        const char* e = std::getenv("GRIMOIRE_GGUF_ESGEMV");
        return !(e && *e == '0');
    }();
    if (!enabled || !native_qk_fast_enabled() || !w.payload || w.scales || w.zeros ||
        w.row_scales || w.K < 512 || w.K % 512 || w.K / 512 > 32)
        return false;
    if (w.fmt == Fmt::Q4_K && native_qk_fast_weight<Fmt::Q4_K>(w)) {
        out = native_qk_esgemv<Fmt::Q4_K>(q, w, x, y, deps);
        return true;
    }
    if (w.fmt == Fmt::Q6_K && native_qk_fast_weight<Fmt::Q6_K>(w)) {
        out = native_qk_esgemv<Fmt::Q6_K>(q, w, x, y, deps);
        return true;
    }
    return false;
}

} // namespace b70
