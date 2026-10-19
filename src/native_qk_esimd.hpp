// Vector native Q4_K/Q6_K decode dots and the dense GEMV using them.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "kernels.hpp"
#include "b70/native_qk_gemv.hpp"
#include "b70/native_qk_vector_math.hpp"
#include <sycl/ext/intel/esimd.hpp>

namespace b70 {
namespace nqes = sycl::ext::intel::esimd;

template <Fmt F>
SYCL_ESIMD_FUNCTION inline float native_qk_esdot_256(
        const uint8_t* p, const nqes::simd<float, 256>& x) {
    static_assert(F == Fmt::Q4_K || F == Fmt::Q6_K);
    if constexpr (F == Fmt::Q4_K) {
        const float d = f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
        const float dm = f16_to_f32(uint16_t(p[2]) | (uint16_t(p[3]) << 8));
        nqes::simd<float, 32> acc = 0.0f;
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            const auto sc = native_q4_group_codes(p, g);
            nqes::simd<uint8_t, 32> b =
                nqes::block_load<uint8_t, 32>(p + 16 + (g / 2) * 32);
            nqes::simd<uint16_t, 32> q = b;
            q = (q >> ((g & 1) * 4)) & 15;
            const nqes::simd<float, 32> w = nqes::convert<float>(q) *
                (d * float(sc.scale)) - dm * float(sc.minimum);
            acc += w * x.template select<32, 1>(g * 32);
        }
        return nqes::reduce<float>(acc, std::plus<>());
    } else {
        const float d = f16_to_f32(uint16_t(p[208]) | (uint16_t(p[209]) << 8));
        nqes::simd<float, 16> acc = 0.0f;
        const nqes::simd<uint32_t, 16> ix(0, 1);
        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            const int half = g / 8, quarter = (g % 8) / 2;
            const int off = (g & 1) * 16;
            // Q6_K rows are 210 bytes. Byte gathers avoid an alignment
            // assumption on the packed low/high planes.
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
            acc += w * x.template select<16, 1>(g * 16);
        }
        return nqes::reduce<float>(acc, std::plus<>());
    }
}

template <Fmt F>
sycl::event native_qk_esgemv(sycl::queue& q, const QuantWeight& w,
                             const float* x, float* y,
                             const std::vector<sycl::event>& deps) {
    constexpr int KP = 512, R = 2, RP = 4;
    const int ks = w.K / KP;
    const int groups = (w.N + R - 1) / R;
    const uint8_t* pay = w.payload;
    const int64_t rb = w.row_bytes;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * ks, ks),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                nqes::slm_init<32 * RP * 4>();
                const int t = int(it.get_local_id(0));
                const int row0 = int(it.get_group(0)) * R;
                const int kb = t * KP;
                const nqes::simd<float, KP> xv = nqes::block_load<float, KP>(x + kb);
                nqes::simd<float, RP> part = 0.0f;
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (row0 + r >= w.N) continue;
                    const uint8_t* p = pay + int64_t(row0 + r) * rb +
                        int64_t(kb / 256) * NativeGGUFBlock<F>::bytes;
                    part[r] = native_qk_esdot_256<F>(p, xv.template select<256, 1>(0)) +
                              native_qk_esdot_256<F>(p + NativeGGUFBlock<F>::bytes,
                                                      xv.template select<256, 1>(256));
                }
                nqes::slm_block_store<float, RP>(t * RP * 4, part);
                nqes::barrier();
                if (t != 0) return;
                nqes::simd<float, RP> sum = 0.0f;
                for (int j = 0; j < ks; ++j)
                    sum += nqes::slm_block_load<float, RP>(j * RP * 4);
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    if (row0 + r < w.N) y[row0 + r] = sum[r];
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
