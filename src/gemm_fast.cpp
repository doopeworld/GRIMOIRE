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
//  gemm_fast.cpp  --  the large-M prompt-processing GEMM
//
//  y[M][N] (fp32) = x[M][K] (bf16) * W[N][K]^T, W in any QuantWeight format.
//
//  gemm_flt (gemm_xmx.cpp) stages A and B through shared local memory one
//  element per work-item per K step and decodes every weight element inside
//  the K loop: ~13 TFLOP/s on the B70.  This path instead
//    1. dequantizes W ONCE per call into a bf16 VNNI scratch [K/2][N][2], and
//    2. runs a joint_matrix GEMM whose sub-groups each own a 32x64 output
//       tile and load A and B straight from global memory (2-D block loads)
//       -- no SLM staging, no barriers in the K loop.
//  tools/bench_gemm_bf16.cpp, M=6144 N=17408 K=5120 on the B70: 9.96 ms =
//  110 TFLOP/s, max relative error 1.6e-6 against an fp64 reference.
//
//  bin/grimoire and bin/grimoire-server link this file as
//  bin/libgrimoire_gemm.so, built with -cl-intel-256-GRF-per-thread and
//  -doubleGRF (also needed by ESIMD flash prefill). The
//  flag is what makes it fast: the same kernel with the per-kernel
//  grf_size<256> property measured 70 TFLOP/s, and the flag cannot go on the
//  whole engine because gemm_flt and others launch 1024-thread work-groups,
//  which 256-register mode does not allow.  The JIT correctness gates compile
//  this file directly (slower there, same arithmetic).
// =====================================================================
#include "kernels.hpp"
#include "int4_smallm.hpp"
#include "b70/tiered_moe.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>

namespace b70 {

namespace {

namespace matrix = sycl::ext::oneapi::experimental::matrix;

// Sub-group tile MC1 x NC1, K step KC1, work-group tile MC2 x NC2.
constexpr int MC1 = 32, NC1 = 64, KC1 = 32, MC2 = 256, NC2 = 256;
// Largest weight dequantized into scratch (elements).  Keeps an lm_head-sized
// matrix from allocating gigabytes; such shapes stay on gemm_flt.
constexpr size_t kMaxScratchElems = size_t(512) << 20;

// float -> bf16, round to nearest even, on the conversion hardware.
// sycl::ext::oneapi::bfloat16's constructor links the SOFTWARE devicelib
// fallback for this AOT target: integer RNE plus a NaN branch per value --
// 130 divergent branches in the SwiGLU GEMM epilogue, 22 in the dequant, 16
// per key block in flash (IGC ISA dump, 2026-09-26).  Identical result for
// every non-NaN input.
inline sycl_bf16 bf16_rne(float f) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::bit_cast<sycl_bf16>(__spirv_ConvertFToBF16INTEL(f));
#else
    return sycl_bf16(f);
#endif
}

// Work-groups are dispatched in linear order, so the order decides what
// shares L2.  GROUP_M row blocks at a time, rows fastest: the ~32 resident
// work-groups then share 4 A row-panels and 8 B column-panels instead of 1
// and 32 -- B (2 bytes per weight) used to be re-read once per row block.
// GRIMOIRE_GEMM_GROUP_M=1 restores the old column-fastest order.
// Prefetch A and B tiles d K steps ahead into L1.  MEASURED, Qwen3.8-27B
// 4088-token prefill, all 384 GEMMs: d=0 1579 ms, d=1 1377, d=2 1536,
// d=3 1641.  GRIMOIRE_GEMM_PREFETCH=d overrides (0 = off).
int gemm_prefetch() {
    static const int d = [] {
        const char* e = std::getenv("GRIMOIRE_GEMM_PREFETCH");
        const int v = e ? std::atoi(e) : 1;
        return v >= 0 && v <= 8 ? v : 0;
    }();
    return d;
}

int gemm_group_m() {
    static const int g = [] {
        const char* e = std::getenv("GRIMOIRE_GEMM_GROUP_M");
        const int v = e ? std::atoi(e) : 4;
        return v >= 1 ? v : 4;
    }();
    return g;
}

// decode_elem<MXFP4> for one payload byte (k even in the low nibble), as a
// VNNI pair of bf16: E2M1 magnitude built from bits, times the E8M0 block
// scale in fp32, rounded to bf16 (exact: at most 2 significant bits).
inline uint32_t mxfp4_pair_bf16(uint32_t byte, float sc) {
    auto one = [&](uint32_t nib) -> uint32_t {
        const uint32_t m = nib & 7u;
        const uint32_t mag = m >= 2u ? 0x3F00u + (m << 6) : (m == 1u ? 0x3F00u : 0u);
        const float f = sycl::bit_cast<float>(((nib & 8u) << 28) | (mag << 16)) * sc;
        return uint32_t(sycl::bit_cast<uint16_t>(bf16_rne(f)));
    };
    return one(byte & 0x0Fu) | (one(byte >> 4) << 16);
}

template <Fmt F>
sycl::event dequant_vnni(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                         const std::vector<sycl::event>& deps, int ldn = 0) {
    const int N = w.N, K = w.K, LD = ldn ? ldn : w.N;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        // n fastest: consecutive work-items write consecutive VNNI pairs.
        h.parallel_for(sycl::range<2>(size_t(K / 2), size_t(N)), [=](sycl::id<2> id) {
            const int kp = int(id[0]), n = int(id[1]);
            const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
            float v[2];
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int k = 2 * kp + t;
                // Same decode as gemm_flt's staging loop, so both paths read
                // a weight to the same bf16 value.
                float scale = 1.0f;
                if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::FP8_E5M2 ||
                              F == Fmt::INT8) {
                    scale = static_cast<const float*>(wc.scales)[n];
                } else if constexpr (F == Fmt::INT4) {
                    scale = bf16_to_f32(static_cast<const bf16_t*>(wc.scales)
                        [int64_t(n) * wc.row_scales + (k >> wc.int4_gshift())]);
                } else if constexpr (F == Fmt::MXFP8 || F == Fmt::MXFP4) {
                    scale = e8m0_to_f32(static_cast<const uint8_t*>(wc.scales)
                        [int64_t(n) * wc.row_scales + k / kMXBlock]);
                }
                if constexpr (F == Fmt::INT4)
                    v[t] = decode_int4(row, k, scale, wc.zeros
                        ? wc.zeros[int64_t(n) * wc.row_scales + (k >> wc.int4_gshift())] : 0);
                else
                    v[t] = decode_elem<F>(row, k, scale);
            }
            sycl_bf16* o = dst + (int64_t(kp) * LD + n) * 2;
            o[0] = bf16_rne(v[0]);
            o[1] = bf16_rne(v[1]);
        });
    });
}

// E2M1 magnitude by arithmetic, not a table: a function-local table (as in
// e2m1_to_f32) lands in private memory on the GPU.  Same exact values.
inline float e2m1_alu_f(uint32_t nib) {
    const uint32_t m = nib & 7u, e = m >> 1, f = m & 1u;
    const float mag = e == 0 ? 0.5f * float(f) : (1.0f + 0.5f * float(f)) * float(1u << (e - 1));
    return (nib & 8u) ? -mag : mag;
}

// Tiled dequantize-transpose, W [N][K] -> bf16 VNNI [K/2][N][2].  One work-group
// owns a 64 (n) x 64 (k) tile: stage 1 reads each row's 32 bytes contiguously
// and decodes into SLM as [k][n]; stage 2 writes VNNI rows in 256-byte runs.
// The per-element dequant_vnni above moved 61 GB in 1.64 s (37 GB/s) for one
// 4088-token Qwen prefill -- as long as the GEMMs themselves.
// Requires N % 64 == 0 and K % 64 == 0.  Values are bit-identical to
// dequant_vnni (same decode, same fp32 multiply, same bf16 rounding).
template <Fmt F>
sycl::event dequant_vnni_tiled(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                               const std::vector<sycl::event>& deps) {
    constexpr int TN_ = 64, TK_ = 64, WGT = 256;
    const int N = w.N, K = w.K;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        sycl::local_accessor<sycl_bf16, 1> T(TK_ * TN_, h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(size_t(K / TK_), size_t(N / TN_) * WGT),
                              sycl::range<2>(1, WGT)),
            [=](sycl::nd_item<2> it) {
            const int lid = int(it.get_local_id(1));
            const int k0 = int(it.get_group(0)) * TK_, n0 = int(it.get_group(1)) * TN_;
            sycl_bf16* t = T.template get_multi_ptr<sycl::access::decorated::no>().get();
            {
                const int r = lid / 4, c = lid % 4;          // 64 rows x 4 chunks of 16
                const int n = n0 + r, kb = k0 + c * 16;
                const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                float v[16];
                if constexpr (F == Fmt::MXFP4) {
                    const float sc = e8m0_to_f32(static_cast<const uint8_t*>(wc.scales)
                        [int64_t(n) * wc.row_scales + kb / kMXBlock]);
                    const uint64_t packed = *reinterpret_cast<const uint64_t*>(row + (kb >> 1));
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t byte = uint32_t(packed >> (8 * i)) & 0xFFu;
                        v[2 * i]     = e2m1_alu_f(byte & 0x0Fu) * sc;
                        v[2 * i + 1] = e2m1_alu_f(byte >> 4) * sc;
                    }
                } else {
                    float sc = 1.0f;
                    if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::FP8_E5M2 || F == Fmt::INT8) {
                        sc = static_cast<const float*>(wc.scales)[n];
                    } else if constexpr (F == Fmt::INT4) {
                        sc = bf16_to_f32(static_cast<const bf16_t*>(wc.scales)
                            [int64_t(n) * wc.row_scales + (kb >> wc.int4_gshift())]);
                    } else if constexpr (F == Fmt::MXFP8) {
                        sc = e8m0_to_f32(static_cast<const uint8_t*>(wc.scales)
                            [int64_t(n) * wc.row_scales + kb / kMXBlock]);
                    }
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        if constexpr (F == Fmt::INT4)
                            v[j] = decode_int4(row, kb + j, sc, wc.zeros
                                ? wc.zeros[int64_t(n) * wc.row_scales + ((kb + j) >> wc.int4_gshift())] : 0);
                        else
                            v[j] = decode_elem<F>(row, kb + j, sc);
                    }
                }
                #pragma unroll
                for (int j = 0; j < 16; ++j) t[(c * 16 + j) * TN_ + r] = bf16_rne(v[j]);
            }
            sycl::group_barrier(it.get_group());
            {
                const int pr = lid / 8, sgm = (lid % 8) * 8;  // 32 k-pairs x 8 segments of 8 n
                sycl_bf16* o = dst + (int64_t(k0 / 2 + pr) * N + n0 + sgm) * 2;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    o[2 * j]     = t[(2 * pr) * TN_ + sgm + j];
                    o[2 * j + 1] = t[(2 * pr + 1) * TN_ + sgm + j];
                }
            }
        });
    });
}


// MXFP4 W [N][K] -> bf16 VNNI [K/2][N][2], streaming.  One work-item per
// (row n, 128 k): one 64-byte payload line in, 64 dword stores out; the 16
// lanes of a sub-group are 16 consecutive n, so every store instruction
// writes one whole 64-byte line of the scratch.  No SLM, no barrier.
// Needs K % 128 == 0 and N % 256 == 0.  Same values as dequant_vnni.
// ilv_fi > 0 (SwiGLU): W = [gate; up] with FI = ilv_fi rows each, written
// interleaved -- scratch columns [64b, 64b+32) hold gate rows [32b, 32b+32)
// and [64b+32, 64b+64) the matching up rows -- so a sub-group's 64 contiguous
// columns are a gate tile and its up tile, and the SwiGLU GEMM loads B
// exactly like the plain one.
sycl::event dequant_mxfp4_stream(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                                 const std::vector<sycl::event>& deps, int ilv_fi = 0) {
    const int N = w.N, K = w.K;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        // Row blocks fastest: the work-groups in flight write adjacent 1 KB
        // runs of the same k-pair rows.  MEASURED (all 384 dequants of a
        // 4088-token Qwen prefill): this order 159 ms; k chunks fastest
        // 293 ms; 2 or 4 chunks per work-item 192 / 242 ms -- write locality
        // dominates.
        const int KB = K / 128, NB = N / 256;
        h.parallel_for(sycl::nd_range<1>(size_t(KB) * NB * 256, 256),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const int L = int(it.get_group(0));
            const int n = (L % NB) * 256 + int(it.get_local_id(0));
            const int kb = L / NB;
            const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes + kb * 64;
            const uint8_t* sr = static_cast<const uint8_t*>(wc.scales) +
                                int64_t(n) * wc.row_scales + kb * 4;
            const int col = ilv_fi == 0 ? n
                          : n < ilv_fi ? (n / 32) * 64 + n % 32
                                       : ((n - ilv_fi) / 32) * 64 + 32 + (n - ilv_fi) % 32;
            uint32_t* o = reinterpret_cast<uint32_t*>(dst) + int64_t(kb) * 64 * N + col;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float sc = e8m0_to_f32(sr[c]);
                const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(row + c * 16);
                #pragma unroll
                for (int d = 0; d < 4; ++d)
                    #pragma unroll
                    for (int b = 0; b < 4; ++b)
                        o[int64_t(c * 16 + d * 4 + b) * N] =
                            mxfp4_pair_bf16((v[d] >> (8 * b)) & 0xFFu, sc);
            }
        });
    });
}

// INT4 W [N][K] -> bf16 VNNI [K/2][N][2]: dequant_mxfp4_stream's walk and
// SwiGLU interleave for GRIMOIRE's INT4 (bf16 scale + u8 zero per 64 or 128 k;
// zero 0xff = signed s4).  decode_int4's arithmetic and the same bf16
// rounding, so the values are dequant_vnni's.  INT4 used to take the generic
// tiled dequant and, because the fused SwiGLU path needs this interleave, the
// three-step FFN: Qwen3.8-27B GPTQ prefilled 4,096 tokens at ~1,730 tok/s
// against ~2,050 for the same model in MXFP4 (2026-10-05).
// Needs K % 128 == 0 and N % 256 == 0.
inline uint32_t int4_pair_bf16(uint32_t byte, float sc, uint32_t zero) {
    auto one = [&](uint32_t q) -> uint32_t {
        const float v = zero == 0xffu ? float(q & 8u ? int(q) - 16 : int(q)) * sc
                                      : (float(q) - float(zero)) * sc;
        return uint32_t(sycl::bit_cast<uint16_t>(bf16_rne(v)));
    };
    return one(byte & 0x0Fu) | (one(byte >> 4) << 16);
}
sycl::event dequant_int4_stream(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                                const std::vector<sycl::event>& deps, int ilv_fi = 0) {
    const int N = w.N, K = w.K, gsh = w.int4_gshift();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        const int KB = K / 128, NB = N / 256;
        h.parallel_for(sycl::nd_range<1>(size_t(KB) * NB * 256, 256),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const int L = int(it.get_group(0));
            const int n = (L % NB) * 256 + int(it.get_local_id(0));
            const int kb = L / NB;
            const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes + kb * 64;
            const bf16_t* sr = static_cast<const bf16_t*>(wc.scales) + int64_t(n) * wc.row_scales;
            const uint8_t* zr = wc.zeros ? wc.zeros + int64_t(n) * wc.row_scales : nullptr;
            const int col = ilv_fi == 0 ? n
                          : n < ilv_fi ? (n / 32) * 64 + n % 32
                                       : ((n - ilv_fi) / 32) * 64 + 32 + (n - ilv_fi) % 32;
            uint32_t* o = reinterpret_cast<uint32_t*>(dst) + int64_t(kb) * 64 * N + col;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int g = (kb * 128 + c * 32) >> gsh;
                const float sc = bf16_to_f32(sr[g]);
                const uint32_t zero = zr ? uint32_t(zr[g]) : 0u;
                const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(row + c * 16);
                #pragma unroll
                for (int d = 0; d < 4; ++d)
                    #pragma unroll
                    for (int b = 0; b < 4; ++b)
                        o[int64_t(c * 16 + d * 4 + b) * N] =
                            int4_pair_bf16((v[d] >> (8 * b)) & 0xFFu, sc, zero);
            }
        });
    });
}
bool int4_stream_ok(const QuantWeight& w) {
    if (w.fmt != Fmt::INT4 || !w.payload || !w.scales) return false;
    const int gs = 1 << w.int4_gshift();
    return (gs == 64 || gs == 128) && int64_t(w.row_scales) * gs == w.K && w.K % 128 == 0 &&
           w.N % 256 == 0 && w.row_bytes >= int64_t(w.K / 2) && w.row_bytes % 16 == 0;
}

// NVFP4 W [N][K] (E2M1 nibbles + one E4M3 scale per 16) -> bf16 VNNI
// [K/2][N][2], the same streaming walk as dequant_mxfp4_stream: a 16-byte
// payload chunk covers two 16-wide blocks, so its first eight bytes take
// scale 2c and the last eight scale 2c+1.  e2m1 x e4m3 has at most 6
// significant bits, so the bf16 values are EXACT; the tensor's F32 scale
// is applied in fp32 by the GEMM epilogue, not folded in here.
// Needs K % 128 == 0 and N % 256 == 0.  `pay`/`scl` may be host memory.
sycl::event dequant_nvfp4_stream(sycl::queue& q, const uint8_t* pay, const uint8_t* scl,
                                 int N, int K, sycl_bf16* dst,
                                 const std::vector<sycl::event>& deps, int ilv_fi = 0) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const int KB = K / 128, NB = N / 256;
        const int64_t rb = K / 2, rs = K / 16;
        h.parallel_for(sycl::nd_range<1>(size_t(KB) * NB * 256, 256),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const int L = int(it.get_group(0));
            const int n = (L % NB) * 256 + int(it.get_local_id(0));
            const int kb = L / NB;
            const uint8_t* row = pay + int64_t(n) * rb + kb * 64;
            const uint8_t* sr = scl + int64_t(n) * rs + kb * 8;
            // ilv_fi: the SwiGLU interleave of dequant_mxfp4_stream
            const int col = ilv_fi == 0 ? n
                          : n < ilv_fi ? (n / 32) * 64 + n % 32
                                       : ((n - ilv_fi) / 32) * 64 + 32 + (n - ilv_fi) % 32;
            uint32_t* o = reinterpret_cast<uint32_t*>(dst) + int64_t(kb) * 64 * N + col;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float s0 = e4m3_to_f32(sr[2 * c]), s1 = e4m3_to_f32(sr[2 * c + 1]);
                const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(row + c * 16);
                #pragma unroll
                for (int d = 0; d < 4; ++d)
                    #pragma unroll
                    for (int b = 0; b < 4; ++b)
                        o[int64_t(c * 16 + d * 4 + b) * N] =
                            mxfp4_pair_bf16((v[d] >> (8 * b)) & 0xFFu, d < 2 ? s0 : s1);
            }
        });
    });
}

sycl::event dequant_any(sycl::queue& q, const QuantWeight& w, sycl_bf16* dst,
                        const std::vector<sycl::event>& deps, int ldn = 0) {
    if (ldn && ldn != w.N) {        // padded pitch (small N): simple kernel, columns >= N untouched
        switch (w.fmt) {
            case Fmt::BF16:     return dequant_vnni<Fmt::BF16>(q, w, dst, deps, ldn);
            case Fmt::FP8_E4M3: return dequant_vnni<Fmt::FP8_E4M3>(q, w, dst, deps, ldn);
            case Fmt::FP8_E5M2: return dequant_vnni<Fmt::FP8_E5M2>(q, w, dst, deps, ldn);
            case Fmt::MXFP8:    return dequant_vnni<Fmt::MXFP8>(q, w, dst, deps, ldn);
            case Fmt::MXFP4:    return dequant_vnni<Fmt::MXFP4>(q, w, dst, deps, ldn);
            case Fmt::INT8:     return dequant_vnni<Fmt::INT8>(q, w, dst, deps, ldn);
            case Fmt::INT4:     return dequant_vnni<Fmt::INT4>(q, w, dst, deps, ldn);
        }
        return {};
    }
    static const bool no_stream = std::getenv("GRIMOIRE_DEQUANT_TILED") != nullptr;
    if (!no_stream && w.fmt == Fmt::MXFP4 && w.K % 128 == 0 && w.N % 256 == 0)
        return dequant_mxfp4_stream(q, w, dst, deps);
    if (!no_stream && int4_stream_ok(w))
        return dequant_int4_stream(q, w, dst, deps);
    if (w.N % 64 == 0 && w.K % 64 == 0) {
        switch (w.fmt) {
            case Fmt::BF16:     return dequant_vnni_tiled<Fmt::BF16>(q, w, dst, deps);
            case Fmt::FP8_E4M3: return dequant_vnni_tiled<Fmt::FP8_E4M3>(q, w, dst, deps);
            case Fmt::FP8_E5M2: return dequant_vnni_tiled<Fmt::FP8_E5M2>(q, w, dst, deps);
            case Fmt::MXFP8:    return dequant_vnni_tiled<Fmt::MXFP8>(q, w, dst, deps);
            case Fmt::MXFP4:    return dequant_vnni_tiled<Fmt::MXFP4>(q, w, dst, deps);
            case Fmt::INT8:     return dequant_vnni_tiled<Fmt::INT8>(q, w, dst, deps);
            case Fmt::INT4:     return dequant_vnni_tiled<Fmt::INT4>(q, w, dst, deps);
        }
    }
    switch (w.fmt) {
        case Fmt::BF16:     return dequant_vnni<Fmt::BF16>(q, w, dst, deps);
        case Fmt::FP8_E4M3: return dequant_vnni<Fmt::FP8_E4M3>(q, w, dst, deps);
        case Fmt::FP8_E5M2: return dequant_vnni<Fmt::FP8_E5M2>(q, w, dst, deps);
        case Fmt::MXFP8:    return dequant_vnni<Fmt::MXFP8>(q, w, dst, deps);
        case Fmt::MXFP4:    return dequant_vnni<Fmt::MXFP4>(q, w, dst, deps);
        case Fmt::INT8:     return dequant_vnni<Fmt::INT8>(q, w, dst, deps);
        case Fmt::INT4:     return dequant_vnni<Fmt::INT4>(q, w, dst, deps);
    }
    return {};
}

// C = A * B with B packed [K/2][N][2] (the dequantized scratch).
// N % NC2 == 0, K % KC1 == 0, any M >= 1: the last, partial row block runs in
// the same launch with bounds-checked A loads (rows >= M read as zero) and
// clipped stores -- no padded copy (that tail path cost 164 ms of a
// 4088-token Qwen prefill).  Work-group order: gemm_group_m().
//   EPI 0: C fp32 [M][N].
//   EPI 1: SwiGLU.  B's columns are [gate | up] (N = 2*FI); each sub-group
//          computes a gate tile and the matching up tile and writes
//          h bf16 [M][FI] = silu(gate) * up -- launch_swiglu_batched's
//          formula on the same fp32 values, rounded as launch_f32_to_bf16.
//   SC: the NVFP4 per-projection F32 scales, applied in fp32.  EPI 0:
//          C *= s0 for columns < split, s1 from split on (split is a
//          multiple of TN, so one fragment never straddles it).  EPI 1:
//          gate *= s0 and up *= s1 before silu(gate) * up.
template <int EPI, bool SC = false>
sycl::event gemm_bf16_vnni(sycl::queue& q, const sycl_bf16* A, const sycl_bf16* B,
                           void* out, int M, int N, int K,
                           const std::vector<sycl::event>& deps, int ldb = 0,
                           float s0 = 1.0f, float s1 = 1.0f, int split = 0) {
    constexpr int SGM = MC2 / MC1, SGN = NC2 / NC1, WG = SGM * SGN * SG_SIZE;
    const int LDB = ldb ? ldb : N;               // B's column pitch: N rounded up to NC2
    const int nMB = (M + MC2 - 1) / MC2, nNB = LDB / NC2, FI = N / 2, GM = gemm_group_m();
    const int PD = gemm_prefetch();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(nMB) * nNB * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            auto sg = it.get_sub_group();
            const int L = int(it.get_group(0)), per = GM * nNB;
            const int mb0 = (L / per) * GM, gm = sycl::min(nMB - mb0, GM);
            const int mb = mb0 + (L % per) % gm, nb = (L % per) / gm;
            const int s = int(it.get_local_id(0)) / SG_SIZE, sm = s / SGN, sn = s % SGN;
            const int m0 = mb * MC2 + sm * MC1;
            if (m0 >= M) return;
            // B columns of fragment n.  For EPI 1 the scratch is gate/up
            // interleaved (dequant_mxfp4_stream ilv_fi), so fragments 0-1 are
            // gate and 2-3 the matching up columns either way.
            auto bcol = [&](int n) { return nb * NC2 + sn * NC1 + n * TN; };
            auto pA = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(A);
            auto pB = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(B);
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN>
                acc[MC1 / TM][NC1 / TN];
            #pragma unroll
            for (int m = 0; m < MC1 / TM; ++m)
                #pragma unroll
                for (int n = 0; n < NC1 / TN; ++n)
                    matrix::joint_matrix_fill(sg, acc[m][n], 0.0f);
            // A always goes through the bounds-checked 2-D block load: the
            // hardware message is the same one either way, and one loop copy
            // (not a checked + unchecked pair) keeps the code the old kernel's.
            const bool a_full = m0 + MC1 <= M;       // never prefetch past row M
            for (int k = 0; k < K; k += KC1) {
                if (PD > 0 && k + PD * KC1 < K) {
                    namespace syclex = sycl::ext::oneapi::experimental;
                    const int kp = k + PD * KC1;
                    if (a_full)
                        ix::joint_matrix_prefetch<MC1, KC1>(sg, A + size_t(m0) * K + kp, K,
                            matrix::layout::row_major,
                            syclex::properties{syclex::prefetch_hint_L1});
                    #pragma unroll
                    for (int c = 0; c < NC1 * 2; c += 32)
                        ix::joint_matrix_prefetch<KC1 / 2, 32>(sg,
                            B + size_t(kp / 2) * (size_t(LDB) * 2) + size_t(bcol(0)) * 2 + c,
                            size_t(LDB) * 2, matrix::layout::row_major,
                            syclex::properties{syclex::prefetch_hint_L1});
                }
                matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::a, TM, TK_BF16,
                                     matrix::layout::row_major> a[MC1 / TM][KC1 / TK_BF16];
                matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::b, TK_BF16, TN,
                                     matrix::layout::ext_intel_packed> b[NC1 / TN][KC1 / TK_BF16];
                #pragma unroll
                for (int kk = 0; kk < KC1 / TK_BF16; ++kk) {
                    #pragma unroll
                    for (int m = 0; m < MC1 / TM; ++m)
                        ix::joint_matrix_load_checked(sg, a[m][kk], pA, size_t(K), size_t(M),
                            size_t(K), size_t(m0 + m * TM), size_t(k + kk * TK_BF16));
                    #pragma unroll
                    for (int n = 0; n < NC1 / TN; ++n)
                        matrix::joint_matrix_load(sg, b[n][kk],
                            pB + size_t(k + kk * TK_BF16) / 2 * (size_t(LDB) * 2)
                               + size_t(bcol(n)) * 2,
                            size_t(LDB) * 2);
                }
                #pragma unroll
                for (int kk = 0; kk < KC1 / TK_BF16; ++kk)
                    #pragma unroll
                    for (int m = 0; m < MC1 / TM; ++m)
                        #pragma unroll
                        for (int n = 0; n < NC1 / TN; ++n)
                            matrix::joint_matrix_mad(sg, acc[m][n], a[m][kk], b[n][kk],
                                                     acc[m][n]);
            }
            if constexpr (EPI == 0 || EPI == 2) {
                auto pC = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(
                    static_cast<float*>(out));
                #pragma unroll
                for (int m = 0; m < MC1 / TM; ++m)
                    #pragma unroll
                    for (int n = 0; n < NC1 / TN; ++n) {
                        if constexpr (EPI == 2) {
                            // residual: out = out + x W^T -- the same fp32
                            // addition the next norm would do (h + r0)
                            matrix::joint_matrix<sycl::sub_group, float,
                                                 matrix::use::accumulator, TM, TN> cin;
                            ix::joint_matrix_load_checked(sg, cin, pC, size_t(N),
                                matrix::layout::row_major, size_t(M), size_t(N),
                                size_t(m0 + m * TM), size_t(bcol(n)));
                            matrix::joint_matrix_apply(sg, acc[m][n], cin,
                                [](float& a, float& c) { a = c + a; });
                        }
                        if constexpr (SC) {
                            const float sc = bcol(n) < split ? s0 : s1;
                            matrix::joint_matrix_apply(sg, acc[m][n],
                                [=](float& a) { a *= sc; });
                        }
                        ix::joint_matrix_store_checked(sg, acc[m][n], pC, size_t(N),
                            matrix::layout::row_major, size_t(M), size_t(N),
                            size_t(m0 + m * TM), size_t(bcol(n)));
                    }
            } else {
                // silu(gate) * up in registers, rounded to a bf16 tile, one
                // bounds-checked 2-D block store.  (Per-element coordinate
                // stores cost 225 ms of epilogue -- twice the SwiGLU kernel.)
                auto pH = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(
                    static_cast<sycl_bf16*>(out));
                #pragma unroll
                for (int m = 0; m < MC1 / TM; ++m)
                    #pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        if constexpr (SC)
                            matrix::joint_matrix_apply(sg, acc[m][n], acc[m][n + 2],
                                [=](float& g, float& u) {
                                    const float gs = g * s0, us = u * s1;
                                    g = sycl::native::divide(gs, 1.0f + sycl::native::exp(-gs)) * us; });
                        else
                        matrix::joint_matrix_apply(sg, acc[m][n], acc[m][n + 2],
                            [](float& g, float& u) {
                                g = sycl::native::divide(g, 1.0f + sycl::native::exp(-g)) * u; });
                        matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::accumulator,
                                             TM, TN> hb;
                        // joint_matrix_copy converts through the software
                        // bf16 path (a NaN branch per element); convert on
                        // the hardware instead, element for element.
                        {
                            auto src = sycl::ext::oneapi::detail::get_wi_data(sg, acc[m][n]);
                            auto dst = sycl::ext::oneapi::detail::get_wi_data(sg, hb);
                            #pragma unroll
                            for (int i = 0; i < TM; ++i) dst[i] = bf16_rne(float(src[i]));
                        }
                        ix::joint_matrix_store_checked(sg, hb, pH, size_t(FI),
                            matrix::layout::row_major, size_t(M), size_t(FI),
                            size_t(m0 + m * TM), size_t(nb * (NC2 / 2) + sn * (NC1 / 2) + n * TN));
                    }
            }
        });
    });
}

// One scratch set per queue.  Work on one queue is ordered, so reusing it
// call after call is safe; two queues never share one.
// GRIMOIRE_FAST_GEMM_TIMING=1: wait after each stage and total the time per
// stage; printed at exit.  Perturbs the pipeline -- diagnosis only.
struct StageTimes {
    double dq = 0, main = 0; long calls = 0;
    ~StageTimes() {
        if (calls) std::fprintf(stderr, "  fast GEMM: %ld calls  dequant %.1f ms  GEMM %.1f ms\n",
                                calls, dq, main);
    }
};
StageTimes& stage_times() { static StageTimes t; return t; }
bool stage_timing() { static const bool on = std::getenv("GRIMOIRE_FAST_GEMM_TIMING") != nullptr; return on; }

struct Scratch {
    sycl_bf16* w = nullptr; size_t w_cap = 0;   // dequantized weight, VNNI
};

template <class T>
T* grow(sycl::queue& q, T*& p, size_t& cap, size_t need) {
    if (need <= cap) return p;
    q.wait();                       // nothing in flight may still read the old one
    if (p) sycl::free(p, q);
    p = sycl::malloc_device<T>(need, q);
    cap = p ? need : 0;
    return p;
}

Scratch& scratch_for(sycl::queue& q) {
    static std::mutex mu;
    static std::map<const sycl::queue*, Scratch> all;
    std::lock_guard<std::mutex> lk(mu);
    return all[&q];
}

} // namespace

// ---------------------------------------------------------------------
// Causal prefill attention on the matrix unit.
//
// launch_flash_prefill (prefill.cpp) is scalar SIMT: one sub-group per
// query, a serial FMA loop over head_dim per key, e4m3 K/V converted into
// SLM element by element -- 329 ms per Qwen3.8-27B layer at 5987 tokens.
// Here a pre-pass converts the layer's e4m3 cache ONCE into the packed bf16
// layouts the matrix unit reads (keys >= kend zeroed):
//   Kp [KVH][D/2][Sp][2]   B operand of S = Q K^T  (k = d,   n = key)
//   Vp [KVH][Sp/2][D][2]   B operand of O = P V    (k = key, n = d)
// and every sub-group then owns 8 query rows end to end: Q in its own SLM
// slice, K/V fragments straight from global memory, the online softmax on
// its own S/P slice, no work-group barrier in the key loop.  Q and P are
// rounded to bf16 (K and V are exact: e4m3 fits in bf16).
// tools/bench_flash_prefill.cpp: 24.7 ms per Qwen layer (17.9 TFLOP/s),
// worst relative error 1.0e-2 against fp64 on ragged, resumed (start > 0)
// and full-size cases.
// ---------------------------------------------------------------------
namespace {

constexpr int FA_NSG = 8, FA_RB = TM, FA_BQ = FA_NSG * FA_RB, FA_BK = 32;

struct FlashScratch {
    sycl_bf16* lut = nullptr;                   // e4m3 byte -> bf16
    sycl_bf16* kp = nullptr; size_t kp_cap = 0;
    sycl_bf16* vp = nullptr; size_t vp_cap = 0;
    sycl_bf16* qb = nullptr; size_t qb_cap = 0; // Q * scale*log2(e), [H][Tp][D]
};

FlashScratch& flash_scratch_for(sycl::queue& q) {
    static std::mutex mu;
    static std::map<const sycl::queue*, FlashScratch> all;
    std::lock_guard<std::mutex> lk(mu);
    FlashScratch& s = all[&q];
    if (!s.lut) {
        s.lut = sycl::malloc_device<sycl_bf16>(256, q);
        if (!s.lut) throw std::runtime_error("flash_fast: lut allocation failed");
        sycl_bf16* l = s.lut;
        q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
            l[i] = sycl_bf16(e4m3_to_f32(uint8_t(i[0])));
        }).wait();
    }
    return s;
}

// ESIMD defaults on for the validated Qwen3.8 geometry (24 query heads,
// 4 KV heads, width 256). Other shapes, QSA and sliding-window attention
// keep joint_matrix. GRIMOIRE_FLASH_ESIMD=1|0 overrides the shape choice;
// GRIMOIRE_FLASH_BLOCK=32|64|128 tunes its keys.
int& flash_esimd_flag() {
    static int f = [] { const char* e = std::getenv("GRIMOIRE_FLASH_ESIMD"); return e ? std::atoi(e) : -1; }();
    return f;
}
bool flash_use_esimd(int width, int heads, int kv_heads) {
    const int mode = flash_esimd_flag();
    return mode < 0 ? width == 256 && heads == 24 && kv_heads == 4 : mode != 0;
}

// ---------------------------------------------------------------------
// Flash prefill, ESIMD: one hardware thread owns 8 query rows of a head.
// Same inputs as the joint_matrix kernel below (Q packed bf16 * scale*log2e
// [H][Tp][D], K packed [D/2][Sp][2], V packed [Sp/2][D][2]) and the same math.
// Explicit registers: each DPAS result (8 rows x 16 keys, fp32, row-major) is
// masked, max-reduced per row, exponentiated and converted to bf16 IN PLACE
// as the A operand of P.V; O (8 x D fp32) stays in registers and is rescaled
// per row with compile-time register indices.  The joint_matrix version
// round-tripped S and P through SLM and kept corr[] in private memory (IGC
// ISA dump).  All building blocks checked on the B70 by tools/esimd_probe.cpp.
// ---------------------------------------------------------------------
template <int D, int BK>
sycl::event flash_esimd(sycl::queue& q, sycl::event dep, const sycl_bf16* Qb, int Tp,
                        const sycl_bf16* Kp, const sycl_bf16* Vp, int Sp, float* out,
                        int tokens, int start, int H, int KVH) {
    namespace es = sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    constexpr int RB = 8, DT = D / 16, TPW = 8, NB = BK / 16;
    const int qtiles = (tokens + RB * TPW - 1) / (RB * TPW);
    const size_t nthreads = size_t(qtiles) * H * TPW;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for(sycl::nd_range<1>(nthreads, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            constexpr float NINF = -std::numeric_limits<float>::infinity();
            const int g = int(it.get_group(0)), lt = int(it.get_local_id(0));
            const int hq = g % H, qt = qtiles - 1 - g / H, kvh = hq / (H / KVH);
            const int r0 = (qt * TPW + lt) * RB;
            if (r0 >= tokens) return;
            const int lastr = tokens - 1 < r0 + RB - 1 ? tokens - 1 : r0 + RB - 1;
            const int kmax = start + lastr + 1;
            const sycl_bf16* qh = Qb + size_t(hq) * Tp * D;
            const uint32_t* kh = reinterpret_cast<const uint32_t*>(Kp + size_t(kvh) * (D / 2) * Sp * 2);
            const uint32_t* vh = reinterpret_cast<const uint32_t*>(Vp + size_t(kvh) * (Sp / 2) * D * 2);
            es::simd<float, RB * D> o = 0.0f;
            es::simd<float, RB> m = NINF, l = 0.0f;
            const es::simd<int, 16> col(0, 1);
            for (int s0 = 0; s0 < kmax; s0 += BK) {
                // Key tiles stay block-major: each 16-key chunk is an 8x16
                // DPAS accumulator, so every register offset is constant.
                es::simd<float, NB * RB * 16> scores = 0.0f;
                #pragma unroll 2
                for (int dt = 0; dt < DT; ++dt) {
                    es::simd<sycl_bf16, RB * 16> a = es::load_2d<sycl_bf16, 16, RB>(
                        qh, D * 2 - 1, Tp - 1, D * 2 - 1, dt * 16, r0);
                    #pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        es::simd<uint32_t, 128> keys = es::load_2d<uint32_t, 16, 8>(
                            kh, Sp * 4 - 1, D / 2 - 1, Sp * 4 - 1, s0 + b * 16, dt * 8);
                        scores.template select<RB * 16, 1>(b * RB * 16) =
                            xmx::dpas<8, RB, float, float, sycl_bf16, sycl_bf16>(
                                es::simd<float, RB * 16>(scores.template select<RB * 16, 1>(b * RB * 16)),
                                keys.template bit_cast_view<sycl_bf16>().read(), a);
                    }
                }
                if (s0 + BK - 1 > start + r0) {
                    #pragma unroll
                    for (int b = 0; b < NB; ++b)
                        #pragma unroll
                        for (int r = 0; r < RB; ++r) {
                            const int lim = r0 + r < tokens ? start + r0 + r : -1;
                            es::simd<float, 16> v = scores.template select<16, 1>((b * RB + r) * 16);
                            v.merge(es::simd<float, 16>(NINF), col + s0 + b * 16 > lim);
                            scores.template select<16, 1>((b * RB + r) * 16) = v;
                        }
                }
                es::simd<float, RB> bm;
                #pragma unroll
                for (int r = 0; r < RB; ++r) {
                    es::simd<float, 16> v = scores.template select<16, 1>(r * 16);
                    #pragma unroll
                    for (int b = 1; b < NB; ++b)
                        v = es::max(v, es::simd<float, 16>(scores.template select<16, 1>((b * RB + r) * 16)));
                    bm[r] = es::hmax<float>(v);
                }
                es::simd<float, RB> mn = es::max(m, bm), ms = mn;
                ms.merge(es::simd<float, RB>(0.0f), mn == NINF);
                es::simd<float, RB> corr = es::exp2(m - ms), rs;
                #pragma unroll
                for (int r = 0; r < RB; ++r) {
                    const float msr = ms[r];
                    es::simd<float, 16> sum = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        es::simd<float, 16> prob = es::exp2(
                            es::simd<float, 16>(scores.template select<16, 1>((b * RB + r) * 16)) - msr);
                        scores.template select<16, 1>((b * RB + r) * 16) = prob;
                        sum += prob;
                    }
                    rs[r] = es::reduce<float>(sum, std::plus<>());
                }
                l = l * corr + rs;
                m = mn;
                float cr[RB];
                #pragma unroll
                for (int r = 0; r < RB; ++r) cr[r] = corr[r];
                #pragma unroll
                for (int dt = 0; dt < DT; ++dt)
                    #pragma unroll
                    for (int r = 0; r < RB; ++r)
                        o.template select<16, 1>((dt * RB + r) * 16) *= cr[r];
                es::simd<sycl_bf16, NB * RB * 16> probabilities = scores;
                #pragma unroll
                for (int dt = 0; dt < DT; ++dt) {
                    #pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        es::simd<uint32_t, 128> values = es::load_2d<uint32_t, 16, 8>(
                            vh, D * 4 - 1, Sp / 2 - 1, D * 4 - 1, dt * 16, (s0 + b * 16) / 2);
                        o.template select<RB * 16, 1>(dt * RB * 16) =
                            xmx::dpas<8, RB, float, float, sycl_bf16, sycl_bf16>(
                                es::simd<float, RB * 16>(o.template select<RB * 16, 1>(dt * RB * 16)),
                                values.template bit_cast_view<sycl_bf16>().read(),
                                es::simd<sycl_bf16, RB * 16>(probabilities.template select<RB * 16, 1>(b * RB * 16)));
                    }
                }
            }
            es::simd<float, RB> inv = 1.0f / l;
            inv.merge(es::simd<float, RB>(0.0f), l == 0.0f);
            float iv[RB];
            #pragma unroll
            for (int r = 0; r < RB; ++r) iv[r] = inv[r];
            #pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                #pragma unroll
                for (int r = 0; r < RB; ++r)
                    o.template select<16, 1>((dt * RB + r) * 16) *= iv[r];
                es::store_2d<float, 16, RB>(out, H * D * 4 - 1, tokens - 1, H * D * 4 - 1,
                    hq * D + dt * 16, r0, es::simd<float, RB * 16>(o.template select<RB * 16, 1>(dt * RB * 16)));
            }
        });
    });
}

int flash_key_block() {
    static const int block = [] {
        const char* e = std::getenv("GRIMOIRE_FLASH_BLOCK");
        const int n = e ? std::atoi(e) : 128;
        return n == 64 || n == 128 ? n : 32;
    }();
    return block;
}

template <int D>
// qbits (QSA, optional): per query row, a bitmap over key blocks of qrat
// tokens -- a key is attended iff it is causal AND (its block's bit is set
// OR it lies in the row's own incomplete tail block).  That is exactly the
// token set Qwen4-Exp's indexer selects (top-k complete blocks + the tail),
// so a sparse row is computed by the XMX kernel, masked, instead of by the
// scalar gather kernel.
sycl::event flash_fast_impl(sycl::queue& q, const float* qv, const uint8_t* kc,
                            const uint8_t* vc, float* out, int tokens, int start, int H,
                            int KVH, int seq_cap, float scale,
                            const std::vector<sycl::event>& deps,
                            const uint32_t* qbits = nullptr, int qwords = 0, int qrat = 4,
                            int window = 0) {
    constexpr int NF = D / TN, WG = FA_NSG * SG_SIZE;
    FlashScratch& fs = flash_scratch_for(q);
    const int kend = start + tokens;
    const bool use_esimd = flash_use_esimd(D, H, KVH) && !qbits && window <= 0;
    const int block = use_esimd ? flash_key_block() : FA_BK;
    const int Sp = (kend + block - 1) / block * block;
    const size_t elems = size_t(KVH) * Sp * D;
    if (!grow(q, fs.kp, fs.kp_cap, elems) || !grow(q, fs.vp, fs.vp_cap, elems))
        throw std::runtime_error("flash_fast: K/V scratch allocation failed");
    sycl_bf16* Kp = fs.kp; sycl_bf16* Vp = fs.vp; const sycl_bf16* lut = fs.lut;
    sycl::event e = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<3>(size_t(KVH), size_t(D), size_t(Sp)), [=](sycl::id<3> id) {
            const int hh = int(id[0]), d = int(id[1]), s = int(id[2]);       // key fastest
            Kp[((size_t(hh) * (D / 2) + d / 2) * Sp + s) * 2 + (d & 1)] =
                s < kend ? lut[kc[(size_t(hh) * D + d) * seq_cap + s]] : sycl_bf16(0.0f);
        });
    });
    e = q.submit([&](sycl::handler& h) {
        h.depends_on(e);
        h.parallel_for(sycl::range<3>(size_t(KVH), size_t(Sp), size_t(D)), [=](sycl::id<3> id) {
            const int hh = int(id[0]), s = int(id[1]), d = int(id[2]);       // d fastest
            Vp[((size_t(hh) * (Sp / 2) + s / 2) * D + d) * 2 + (s & 1)] =
                s < kend ? lut[vc[(size_t(hh) * seq_cap + s) * D + d]] : sycl_bf16(0.0f);
        });
    });
    const int qtiles = (tokens + FA_BQ - 1) / FA_BQ, Tp = qtiles * FA_BQ;
    // Scores in log2 units: Q carries scale*log2(e), so the softmax is the
    // hardware exp2.  The accurate sycl::exp is a long instruction sequence,
    // three per score pair, and made the softmax -- not the matrix unit --
    // the bottleneck.  P is rounded to bf16 before PV anyway.
    const float qscale = scale * 1.4426950408889634f;
    // Q packed once, head-major and zero-padded to whole query tiles, so each
    // sub-group loads its A fragments from global memory.  (Q used to sit in
    // SLM -- 32 of the work-group's 44 KB -- which allowed only 2 work-groups
    // per Xe core.)
    if (!grow(q, fs.qb, fs.qb_cap, size_t(H) * Tp * D))
        throw std::runtime_error("flash_fast: Q scratch allocation failed");
    sycl_bf16* Qb = fs.qb;
    e = q.submit([&](sycl::handler& h) {
        h.depends_on(e);
        h.parallel_for(sycl::range<3>(size_t(H), size_t(Tp), size_t(D)), [=](sycl::id<3> id) {
            const int hh = int(id[0]), t = int(id[1]), d = int(id[2]);
            Qb[(size_t(hh) * Tp + t) * D + d] =
                bf16_rne(t < tokens ? qv[(size_t(t) * H + hh) * D + d] * qscale : 0.0f);
        });
    });
    if (use_esimd) {
        if (block == 128) return flash_esimd<D, 128>(q, e, Qb, Tp, Kp, Vp, Sp, out, tokens, start, H, KVH);
        if (block == 64) return flash_esimd<D, 64>(q, e, Qb, Tp, Kp, Vp, Sp, out, tokens, start, H, KVH);
        return flash_esimd<D, 32>(q, e, Qb, Tp, Kp, Vp, Sp, out, tokens, start, H, KVH);
    }
    return q.submit([&](sycl::handler& h) {
        h.depends_on(e);
        sycl::local_accessor<float, 1> Ss(FA_NSG * FA_RB * FA_BK, h);
        sycl::local_accessor<sycl_bf16, 1> Ps(FA_NSG * FA_RB * FA_BK, h);
        h.parallel_for(sycl::nd_range<1>(size_t(qtiles) * H * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            constexpr auto NO = sycl::access::decorated::no;
            constexpr int RB = FA_RB, BK = FA_BK;
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]), lane = int(sg.get_local_id()[0]);
            // Longest (latest) query tiles first: causal work grows with qt,
            // and the last wave should be the short tiles.
            const int g = int(it.get_group(0)), qt = qtiles - 1 - g / H, hq = g % H,
                      kvh = hq / (H / KVH);
            const int r0 = qt * FA_BQ + sgid * RB;
            if (r0 >= tokens) return;                   // uniform per sub-group; no WG barriers below
            auto ss_mp = Ss.template get_multi_ptr<NO>() + sgid * RB * BK;
            auto ps_mp = Ps.template get_multi_ptr<NO>() + sgid * RB * BK;
            float* ss = ss_mp.get(); sycl_bf16* ps = ps_mp.get();
            auto qg = sycl::address_space_cast<sycl::access::address_space::global_space, NO>(
                Qb + (size_t(hq) * Tp + r0) * D);
            auto kg = sycl::address_space_cast<sycl::access::address_space::global_space, NO>(
                Kp + size_t(kvh) * (D / 2) * Sp * 2);
            auto vg = sycl::address_space_cast<sycl::access::address_space::global_space, NO>(
                Vp + size_t(kvh) * (Sp / 2) * D * 2);
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN> o[NF];
            #pragma unroll
            for (int n = 0; n < NF; ++n) matrix::joint_matrix_fill(sg, o[n], 0.0f);
            float m[RB], l[RB];
            #pragma unroll
            for (int r = 0; r < RB; ++r) { m[r] = -std::numeric_limits<float>::infinity(); l[r] = 0.0f; }
            const int last_row = sycl::min(tokens - 1, r0 + RB - 1);
            const int kmax = start + last_row + 1;
            // Sliding window: row t attends keys (qpos - window, qpos].  The
            // tile's FIRST row has the earliest window start, so no key block
            // before it can be live for any row of the tile -- skip them.
            const int s_first = window > 0
                ? sycl::max(0, start + r0 - window + 1) / BK * BK : 0;
            for (int s0 = s_first; s0 < kmax; s0 += BK) {
                matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN>
                    sa[BK / TN];
                #pragma unroll
                for (int c = 0; c < BK / TN; ++c) matrix::joint_matrix_fill(sg, sa[c], 0.0f);
                #pragma unroll 4
                for (int k = 0; k < D; k += TK_BF16) {
                    matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::a, TM, TK_BF16,
                                         matrix::layout::row_major> a;
                    matrix::joint_matrix_load(sg, a, qg + k, D);
                    #pragma unroll
                    for (int c = 0; c < BK / TN; ++c) {
                        matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::b, TK_BF16,
                                             TN, matrix::layout::ext_intel_packed> b;
                        matrix::joint_matrix_load(sg, b,
                            kg + size_t(k / 2) * Sp * 2 + size_t(s0 + c * TN) * 2, size_t(Sp) * 2);
                        matrix::joint_matrix_mad(sg, sa[c], a, b, sa[c]);
                    }
                }
                #pragma unroll
                for (int c = 0; c < BK / TN; ++c)
                    matrix::joint_matrix_store(sg, sa[c], ss_mp + c * TN, BK,
                                               matrix::layout::row_major);
                sycl::group_barrier(sg);
                // Branch-free: the S slice is always valid memory, so load it
                // unconditionally and mask with selects; exp2(-inf) = 0 covers
                // masked keys and an empty history, and a fully masked row
                // (mn = -inf) subtracts 0 instead of -inf.  (The conditional
                // loads and isinf branches compiled to ~40 divergent branches
                // per key block.)  Same values as before in every case.
                constexpr float NINF = -std::numeric_limits<float>::infinity();
                float corr[RB];
                #pragma unroll
                for (int r = 0; r < RB; ++r) {
                    const int t = r0 + r, qpos = start + t;
                    const float sv0 = ss[r * BK + lane], sv1 = ss[r * BK + lane + SG_SIZE];
                    bool a0 = t < tokens && s0 + lane <= qpos;
                    bool a1 = t < tokens && s0 + lane + SG_SIZE <= qpos;
                    if (window > 0) {
                        a0 = a0 && s0 + lane > qpos - window;
                        a1 = a1 && s0 + lane + SG_SIZE > qpos - window;
                    }
                    if (qbits && t < tokens) {
                        const int tail = (qpos + 1) / qrat * qrat;
                        const uint32_t* rb = qbits + size_t(t) * qwords;
                        const int k0 = s0 + lane, k1 = k0 + SG_SIZE;
                        const int b0 = k0 / qrat, b1 = k1 / qrat;
                        a0 = a0 && (k0 >= tail || ((rb[b0 >> 5] >> (b0 & 31)) & 1u));
                        a1 = a1 && (k1 >= tail || ((rb[b1 >> 5] >> (b1 & 31)) & 1u));
                    }
                    const float v0 = a0 ? sv0 : NINF;
                    const float v1 = a1 ? sv1 : NINF;
                    const float bm = sycl::reduce_over_group(sg, sycl::fmax(v0, v1),
                                                             sycl::maximum<float>());
                    const float mn = sycl::fmax(m[r], bm);
                    const float ms = sycl::isinf(mn) ? 0.0f : mn;
                    corr[r] = sycl::native::exp2(m[r] - ms);
                    const float p0 = sycl::native::exp2(v0 - ms);
                    const float p1 = sycl::native::exp2(v1 - ms);
                    ps[r * BK + lane] = bf16_rne(p0);
                    ps[r * BK + lane + SG_SIZE] = bf16_rne(p1);
                    l[r] = l[r] * corr[r] +
                           sycl::reduce_over_group(sg, p0 + p1, sycl::plus<float>());
                    m[r] = mn;
                }
                sycl::group_barrier(sg);
                // corr[] indexed by the runtime row lives in private memory
                // (512 B/work-item).  Tried: constant-indexed get_wi_data
                // (spilled 5.7 KB), a row-scale tile + element-wise apply
                // (compiled to the same code).  The D=256 kernel is at the
                // 256-register limit.
                // (Lazy rescale -- skip unless a row's max grew by > 2^8 --
                // measured SLOWER again on 2026-09-26: 181 vs 137 ms.)
                #pragma unroll
                for (int n = 0; n < NF; ++n)
                    ix::joint_matrix_apply(sg, o[n],
                        [&](float& x, size_t row, size_t) { x *= corr[row]; });
                #pragma unroll
                for (int kk = 0; kk < BK; kk += TK_BF16) {
                    matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::a, TM, TK_BF16,
                                         matrix::layout::row_major> a;
                    matrix::joint_matrix_load(sg, a, ps_mp + kk, BK);
                    #pragma unroll
                    for (int n = 0; n < NF; ++n) {
                        matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::b, TK_BF16,
                                             TN, matrix::layout::ext_intel_packed> b;
                        matrix::joint_matrix_load(sg, b,
                            vg + size_t((s0 + kk) / 2) * D * 2 + size_t(n * TN) * 2,
                            size_t(D) * 2);
                        matrix::joint_matrix_mad(sg, o[n], a, b, o[n]);
                    }
                }
                sycl::group_barrier(sg);
            }
            float inv[RB];
            #pragma unroll
            for (int r = 0; r < RB; ++r) inv[r] = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
            #pragma unroll
            for (int n = 0; n < NF; ++n)
                ix::joint_matrix_apply(sg, o[n], [&](float& x, size_t row, size_t) { x *= inv[row]; });
            auto og = sycl::address_space_cast<sycl::access::address_space::global_space, NO>(out);
            if (r0 + RB <= tokens) {
                #pragma unroll
                for (int n = 0; n < NF; ++n)
                    matrix::joint_matrix_store(sg, o[n], og + (size_t(r0) * H + hq) * D + n * TN,
                                               size_t(H) * D, matrix::layout::row_major);
            } else {
                for (int n = 0; n < NF; ++n) {          // ragged last tile: through the S slice
                    matrix::joint_matrix_store(sg, o[n], ss_mp, TN, matrix::layout::row_major);
                    sycl::group_barrier(sg);
                    for (int x = lane; x < RB * TN; x += SG_SIZE) {
                        const int r = x / TN, c = x % TN, t = r0 + r;
                        if (t < tokens) out[(size_t(t) * H + hq) * D + n * TN + c] = ss[x];
                    }
                    sycl::group_barrier(sg);
                }
            }
        });
    });
}

} // namespace

sycl::event launch_flash_prefill_qsa(sycl::queue& q, const float* qv, const uint8_t* k_cache,
                                     const uint8_t* v_cache, float* out, int tokens,
                                     int start_pos, int num_heads, int num_kv_heads,
                                     int head_dim, int seq_cap, float softmax_scale,
                                     const uint32_t* qbits, int qwords, int qrat,
                                     const std::vector<sycl::event>& deps) {
    return head_dim == 128
        ? flash_fast_impl<128>(q, qv, k_cache, v_cache, out, tokens, start_pos, num_heads,
                               num_kv_heads, seq_cap, softmax_scale, deps, qbits, qwords, qrat)
        : flash_fast_impl<256>(q, qv, k_cache, v_cache, out, tokens, start_pos, num_heads,
                               num_kv_heads, seq_cap, softmax_scale, deps, qbits, qwords, qrat);
}

// Causal flash prefill with an optional sliding window (keys in
// (qpos - window, qpos]; window <= 0 = full causal).  Muse's sliding layers.
sycl::event launch_flash_prefill_window(sycl::queue& q, const float* qv, const uint8_t* k_cache,
                                        const uint8_t* v_cache, float* out, int tokens,
                                        int start_pos, int num_heads, int num_kv_heads,
                                        int head_dim, int seq_cap, float softmax_scale,
                                        int window, const std::vector<sycl::event>& deps) {
    return head_dim == 128
        ? flash_fast_impl<128>(q, qv, k_cache, v_cache, out, tokens, start_pos, num_heads,
                               num_kv_heads, seq_cap, softmax_scale, deps, nullptr, 0, 4, window)
        : flash_fast_impl<256>(q, qv, k_cache, v_cache, out, tokens, start_pos, num_heads,
                               num_kv_heads, seq_cap, softmax_scale, deps, nullptr, 0, 4, window);
}

bool flash_fast_supported(int head_dim, int num_heads, int num_kv_heads) {
    static const bool off = std::getenv("GRIMOIRE_NO_FAST_FLASH") != nullptr;
    return !off && (head_dim == 128 || head_dim == 256) && num_kv_heads > 0 &&
           num_heads % num_kv_heads == 0;
}

sycl::event launch_flash_prefill_fast(sycl::queue& q, const float* qv, const uint8_t* k_cache,
                                      const uint8_t* v_cache, float* out, int tokens,
                                      int start_pos, int num_heads, int num_kv_heads,
                                      int head_dim, int seq_cap, float softmax_scale,
                                      const std::vector<sycl::event>& deps) {
    auto run = [&](float* o, const std::vector<sycl::event>& dd) {
        return head_dim == 128
            ? flash_fast_impl<128>(q, qv, k_cache, v_cache, o, tokens, start_pos, num_heads,
                                   num_kv_heads, seq_cap, softmax_scale, dd)
            : flash_fast_impl<256>(q, qv, k_cache, v_cache, o, tokens, start_pos, num_heads,
                                   num_kv_heads, seq_cap, softmax_scale, dd);
    };
    // GRIMOIRE_FLASH_VERIFY=n: for the first n calls run the ESIMD kernel into
    // `out` and the joint_matrix kernel into a scratch copy, and print how far
    // apart they are.
    static int verify_left = [] { const char* e = std::getenv("GRIMOIRE_FLASH_VERIFY"); return e ? std::atoi(e) : 0; }();
    if (verify_left > 0) {
        --verify_left;
        const size_t n = size_t(tokens) * num_heads * head_dim;
        float* ref = sycl::malloc_device<float>(n, q);
        const int saved = flash_esimd_flag();
        flash_esimd_flag() = 0;
        run(ref, deps).wait();
        flash_esimd_flag() = 1;
        run(out, {}).wait();
        flash_esimd_flag() = saved;
        std::vector<float> a(n), b(n);
        q.memcpy(a.data(), out, n * sizeof(float));
        q.memcpy(b.data(), ref, n * sizeof(float)).wait();
        double dmax = 0.0, rmax = 0.0;
        for (size_t i = 0; i < n; ++i) {
            dmax = std::max(dmax, std::fabs(double(a[i]) - double(b[i])));
            rmax = std::max(rmax, std::fabs(double(b[i])));
        }
        std::fprintf(stderr, "  flash verify (esimd vs joint_matrix, %d tokens, start %d): "
                     "max|diff| %.3e (max|ref| %.3e)\n", tokens, start_pos, dmax, rmax);
        sycl::free(ref, q);
        return sycl::event{};
    }
    return run(out, deps);
}

namespace {

// ---------------------------------------------------------------------
// MXFP4 GEMM with the dequantize fused into the K loop.
//
// The generic path (launch_gemm_fast below) dequantizes W into a bf16
// scratch and then runs gemm_bf16_vnni: the dequant pass alone cost 327 ms
// of a 4088-token Qwen3.8-27B prefill, and the GEMM re-reads 2 bytes per
// weight per row block.  Here every work-group decodes the 256 columns x 32 k
// of W it needs for the NEXT K step from the MXFP4 payload into one half of
// a double-buffered SLM tile while its sub-groups run the matrix unit on the
// other half (one barrier per K step).  A still loads straight from global
// memory.  Work-groups run GROUP_M row blocks at a time, rows fastest, so
// the ~32 resident work-groups share 4 A row-panels and 8 weight panels.
// The last, partial row block is part of the same launch: its A loads are
// bounds-checked (rows >= M read as zero) and its stores clipped -- no
// padded copy.
//   EPI 0: y fp32 [M][N].
//   EPI 1: SwiGLU.  W = [gate; up] (N = 2*FI); each sub-group holds a gate
//          tile and the matching up tile and writes
//          h bf16 [M][FI] = silu(x W_gate^T) * (x W_up^T).
// Same bf16 weight values, same K order into the same fp32 accumulators as
// dequantize-then-GEMM, and the same SwiGLU formula as
// launch_swiglu_batched, so the results are the same.
// ---------------------------------------------------------------------

template <int EPI>
sycl::event gemm_mxfp4_fused(sycl::queue& q, const QuantWeight& w, const sycl_bf16* A,
                             void* out, int M, const std::vector<sycl::event>& deps) {
    constexpr int SGM = MC2 / MC1, SGN = NC2 / NC1, WG = SGM * SGN * SG_SIZE;
    constexpr int KP = KC1 / 2;                  // k-pairs per K step
    constexpr int BT = KP * NC2;                 // dwords (bf16 pairs) per SLM buffer
    static_assert(WG == 2 * NC2, "decode: two work-items per column (16 k each)");
    static_assert(KC1 == 2 * 16 && KC1 <= kMXBlock, "decode: one E8M0 scale per k half");
    const int N = w.N, K = w.K, FI = N / 2;
    const int nMB = (M + MC2 - 1) / MC2, nNB = N / NC2, GM = gemm_group_m();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        sycl::local_accessor<uint32_t, 1> Bs(2 * BT, h);
        h.parallel_for(sycl::nd_range<1>(size_t(nMB) * nNB * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            auto sg = it.get_sub_group();
            const int lid = int(it.get_local_id(0));
            const int L = int(it.get_group(0)), per = GM * nNB;
            const int mb0 = (L / per) * GM, gm = sycl::min(nMB - mb0, GM);
            const int mb = mb0 + (L % per) % gm, nb = (L % per) / gm;
            const int s = lid / SG_SIZE, sm = s / SGN, sn = s % SGN;
            const int m0 = mb * MC2 + sm * MC1;
            const bool active = m0 < M, full = m0 + MC1 <= M;

            // Decode role: tile column dn, k half dh (16 k = 8 payload bytes).
            const int dn = lid % NC2, dh = lid / NC2;
            const int wrow = EPI == 1
                ? (dn < NC2 / 2 ? nb * (NC2 / 2) + dn : FI + nb * (NC2 / 2) + dn - NC2 / 2)
                : nb * NC2 + dn;
            const uint8_t* prow = wc.payload + int64_t(wrow) * wc.row_bytes;
            const uint8_t* srow = static_cast<const uint8_t*>(wc.scales) +
                                  int64_t(wrow) * wc.row_scales;
            uint32_t* bs = Bs.template get_multi_ptr<sycl::access::decorated::no>().get();
            uint64_t pk = 0; uint8_t sb = 0;
            auto fetch = [&](int k) {
                const int kb = k + dh * 16;
                pk = *reinterpret_cast<const uint64_t*>(prow + (kb >> 1));
                sb = srow[kb / kMXBlock];
            };
            auto decode = [&](int buf) {
                const float sc = e8m0_to_f32(sb);
                uint32_t* d = bs + buf * BT + dh * (KP / 2) * NC2 + dn;
                #pragma unroll
                for (int p = 0; p < 8; ++p)
                    d[p * NC2] = mxfp4_pair_bf16(uint32_t(pk >> (8 * p)) & 0xFFu, sc);
            };
            // SLM column of B fragment n for this sub-group
            auto bcol = [&](int n) {
                return EPI == 1 ? (n < 2 ? sn * (NC1 / 2) + n * TN
                                         : NC2 / 2 + sn * (NC1 / 2) + (n - 2) * TN)
                                : sn * NC1 + n * TN;
            };
            auto pA = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(A);
            auto pBs = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                sycl::access::decorated::no>(
                reinterpret_cast<sycl_bf16*>(bs));
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN>
                acc[MC1 / TM][NC1 / TN];
            #pragma unroll
            for (int m = 0; m < MC1 / TM; ++m)
                #pragma unroll
                for (int n = 0; n < NC1 / TN; ++n)
                    matrix::joint_matrix_fill(sg, acc[m][n], 0.0f);
            fetch(0);
            decode(0);
            sycl::group_barrier(it.get_group());
            for (int k = 0, step = 0; k < K; k += KC1, ++step) {
                const int cur = step & 1;
                const bool more = k + KC1 < K;
                if (more) fetch(k + KC1);            // in flight during the MADs
                if (active) {
                    matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::a, TM, TK_BF16,
                                         matrix::layout::row_major> a[MC1 / TM][KC1 / TK_BF16];
                    matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::b, TK_BF16, TN,
                                         matrix::layout::ext_intel_packed> b[NC1 / TN][KC1 / TK_BF16];
                    #pragma unroll
                    for (int kk = 0; kk < KC1 / TK_BF16; ++kk) {
                        #pragma unroll
                        for (int m = 0; m < MC1 / TM; ++m) {
                            if (full)
                                matrix::joint_matrix_load(sg, a[m][kk],
                                    pA + size_t(m0 + m * TM) * K + k + kk * TK_BF16, K);
                            else
                                ix::joint_matrix_load_checked(sg, a[m][kk], pA, size_t(K),
                                    size_t(M), size_t(K), size_t(m0 + m * TM),
                                    size_t(k + kk * TK_BF16));
                        }
                        #pragma unroll
                        for (int n = 0; n < NC1 / TN; ++n)
                            matrix::joint_matrix_load(sg, b[n][kk],
                                pBs + size_t(cur * BT + kk * (TK_BF16 / 2) * NC2 + bcol(n)) * 2,
                                size_t(NC2) * 2);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < KC1 / TK_BF16; ++kk)
                        #pragma unroll
                        for (int m = 0; m < MC1 / TM; ++m)
                            #pragma unroll
                            for (int n = 0; n < NC1 / TN; ++n)
                                matrix::joint_matrix_mad(sg, acc[m][n], a[m][kk], b[n][kk],
                                                         acc[m][n]);
                }
                if (more) decode(cur ^ 1);
                sycl::group_barrier(it.get_group());
            }
            if (!active) return;
            if constexpr (EPI == 0) {
                auto pC = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(
                    static_cast<float*>(out));
                #pragma unroll
                for (int m = 0; m < MC1 / TM; ++m)
                    #pragma unroll
                    for (int n = 0; n < NC1 / TN; ++n) {
                        const int col = nb * NC2 + sn * NC1 + n * TN;
                        if (full)
                            matrix::joint_matrix_store(sg, acc[m][n],
                                pC + size_t(m0 + m * TM) * N + col, N, matrix::layout::row_major);
                        else
                            ix::joint_matrix_store_checked(sg, acc[m][n], pC, size_t(N),
                                matrix::layout::row_major, size_t(M), size_t(N),
                                size_t(m0 + m * TM), size_t(col));
                    }
            } else {
                sycl_bf16* H = static_cast<sycl_bf16*>(out);
                #pragma unroll
                for (int m = 0; m < MC1 / TM; ++m)
                    #pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        matrix::joint_matrix_apply(sg, acc[m][n], acc[m][n + 2],
                            [](float& g, float& u) {
                                g = sycl::native::divide(g, 1.0f + sycl::native::exp(-g)) * u; });
                        const int r0 = m0 + m * TM;
                        const int col = nb * (NC2 / 2) + sn * (NC1 / 2) + n * TN;
                        ix::joint_matrix_apply(sg, acc[m][n], [=](float& x, size_t r, size_t c) {
                            if (r0 + int(r) < M)
                                H[size_t(r0 + int(r)) * FI + col + c] = bf16_rne(x);
                        });
                    }
            }
        });
    });
}

// ---------------------------------------------------------------------
// GROUPED MoE MXFP4 GEMM: gemm_mxfp4_fused over every expert of a layer in
// ONE launch.  The pure-mode MoE prefill ran one dequant + one GEMM per
// expert: ~190 rows each on a 5987-token Ornith prompt, so a gate_up GEMM
// (N = 1024) was 1 M-block x 4 N-blocks = FOUR work-groups on a 32-core
// card, 512 serialized launches per layer, and every expert's weights went
// through a bf16 scratch copy -- 1.39 s of a 1.90 s prefill.  Here work-group
// L is (m-tile tile_e/tile_mb[L / nNB], n-block L % nNB); a tile reads its
// expert's rows [off[e] + mb*MC2, off[e] + cnt[e]) of A (bounds-checked, as
// the dense kernel does for rows >= M), its expert's weight rows
// e*Ne + ..., and writes the same rows of the output.  The per-tile math
// is gemm_mxfp4_fused's, unchanged.
// ---------------------------------------------------------------------
template <int EPI>
sycl::event gemm_mxfp4_fused_grouped(sycl::queue& q, const QuantWeight& w, int Ne,
                                     const sycl_bf16* A, void* out,
                                     const int32_t* tile_e, const int32_t* tile_mb,
                                     const int32_t* off, const int32_t* cnt, int T,
                                     const std::vector<sycl::event>& deps) {
    constexpr int SGM = MC2 / MC1, SGN = NC2 / NC1, WG = SGM * SGN * SG_SIZE;
    constexpr int KP = KC1 / 2;
    constexpr int BT = KP * NC2;
    const int N = Ne, K = w.K, FI = N / 2;
    const int nNB = N / NC2;
    // A tiles prefetched PD K steps ahead into L1 (GRIMOIRE_MOE_PREFETCH=d,
    // 0 = off): every K step ends in a work-group barrier, so an A load that
    // misses L1 stalls all 32 sub-groups of the core.
    static const int pd_env = [] {
        const char* v = std::getenv("GRIMOIRE_MOE_PREFETCH");
        const int d = v ? std::atoi(v) : 1;
        return d >= 0 && d <= 8 ? d : 1;
    }();
    const int PD = pd_env;                       // kernels cannot read a dynamic static
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        sycl::local_accessor<uint32_t, 1> Bs(2 * BT, h);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * nNB * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            namespace syclex = sycl::ext::oneapi::experimental;
            auto sg = it.get_sub_group();
            const int lid = int(it.get_local_id(0));
            const int L = int(it.get_group(0));
            const int t = L / nNB, nb = L % nNB;
            const int e = tile_e[t], mb = tile_mb[t];
            const int M = cnt[e];
            const int64_t rbase = off[e];
            const int s = lid / SG_SIZE, sm = s / SGN, sn = s % SGN;
            const int m0 = mb * MC2 + sm * MC1;
            const bool active = m0 < M, full = m0 + MC1 <= M;
            const sycl_bf16* Ae = A + rbase * K;

            const int dn = lid % NC2, dh = lid / NC2;
            const int wrow = EPI == 1
                ? (dn < NC2 / 2 ? nb * (NC2 / 2) + dn : FI + nb * (NC2 / 2) + dn - NC2 / 2)
                : nb * NC2 + dn;
            const int64_t grow = int64_t(e) * Ne + wrow;
            const uint8_t* prow = wc.payload + grow * wc.row_bytes;
            const uint8_t* srow = static_cast<const uint8_t*>(wc.scales) + grow * wc.row_scales;
            uint32_t* bs = Bs.template get_multi_ptr<sycl::access::decorated::no>().get();
            uint64_t pk = 0; uint8_t sb = 0;
            auto fetch = [&](int k) {
                const int kb = k + dh * 16;
                pk = *reinterpret_cast<const uint64_t*>(prow + (kb >> 1));
                sb = srow[kb / kMXBlock];
            };
            auto decode = [&](int buf) {
                const float sc = e8m0_to_f32(sb);
                uint32_t* d = bs + buf * BT + dh * (KP / 2) * NC2 + dn;
                #pragma unroll
                for (int p = 0; p < 8; ++p)
                    d[p * NC2] = mxfp4_pair_bf16(uint32_t(pk >> (8 * p)) & 0xFFu, sc);
            };
            auto bcol = [&](int n) {
                return EPI == 1 ? (n < 2 ? sn * (NC1 / 2) + n * TN
                                         : NC2 / 2 + sn * (NC1 / 2) + (n - 2) * TN)
                                : sn * NC1 + n * TN;
            };
            auto pA = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(Ae);
            auto pBs = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                sycl::access::decorated::no>(
                reinterpret_cast<sycl_bf16*>(bs));
            matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator, TM, TN>
                acc[MC1 / TM][NC1 / TN];
            #pragma unroll
            for (int m = 0; m < MC1 / TM; ++m)
                #pragma unroll
                for (int n = 0; n < NC1 / TN; ++n)
                    matrix::joint_matrix_fill(sg, acc[m][n], 0.0f);
            fetch(0);
            decode(0);
            if (full)
                for (int d = 0; d < PD && d * KC1 < K; ++d)
                    ix::joint_matrix_prefetch<MC1, KC1>(sg, Ae + size_t(m0) * K + d * KC1, K,
                        matrix::layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
            sycl::group_barrier(it.get_group());
            for (int k = 0, step = 0; k < K; k += KC1, ++step) {
                const int cur = step & 1;
                const bool more = k + KC1 < K;
                if (more) fetch(k + KC1);
                if (full && PD > 0 && k + PD * KC1 < K)
                    ix::joint_matrix_prefetch<MC1, KC1>(sg, Ae + size_t(m0) * K + k + PD * KC1, K,
                        matrix::layout::row_major, syclex::properties{syclex::prefetch_hint_L1});
                if (active) {
                    matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::a, TM, TK_BF16,
                                         matrix::layout::row_major> a[MC1 / TM][KC1 / TK_BF16];
                    matrix::joint_matrix<sycl::sub_group, sycl_bf16, matrix::use::b, TK_BF16, TN,
                                         matrix::layout::ext_intel_packed> b[NC1 / TN][KC1 / TK_BF16];
                    #pragma unroll
                    for (int kk = 0; kk < KC1 / TK_BF16; ++kk) {
                        #pragma unroll
                        for (int m = 0; m < MC1 / TM; ++m) {
                            if (full)
                                matrix::joint_matrix_load(sg, a[m][kk],
                                    pA + size_t(m0 + m * TM) * K + k + kk * TK_BF16, K);
                            else
                                ix::joint_matrix_load_checked(sg, a[m][kk], pA, size_t(K),
                                    size_t(M), size_t(K), size_t(m0 + m * TM),
                                    size_t(k + kk * TK_BF16));
                        }
                        #pragma unroll
                        for (int n = 0; n < NC1 / TN; ++n)
                            matrix::joint_matrix_load(sg, b[n][kk],
                                pBs + size_t(cur * BT + kk * (TK_BF16 / 2) * NC2 + bcol(n)) * 2,
                                size_t(NC2) * 2);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < KC1 / TK_BF16; ++kk)
                        #pragma unroll
                        for (int m = 0; m < MC1 / TM; ++m)
                            #pragma unroll
                            for (int n = 0; n < NC1 / TN; ++n)
                                matrix::joint_matrix_mad(sg, acc[m][n], a[m][kk], b[n][kk],
                                                         acc[m][n]);
                }
                if (more) decode(cur ^ 1);
                sycl::group_barrier(it.get_group());
            }
            if (!active) return;
            if constexpr (EPI == 0) {
                auto pC = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(
                    static_cast<float*>(out) + rbase * N);
                #pragma unroll
                for (int m = 0; m < MC1 / TM; ++m)
                    #pragma unroll
                    for (int n = 0; n < NC1 / TN; ++n) {
                        const int col = nb * NC2 + sn * NC1 + n * TN;
                        if (full)
                            matrix::joint_matrix_store(sg, acc[m][n],
                                pC + size_t(m0 + m * TM) * N + col, N, matrix::layout::row_major);
                        else
                            ix::joint_matrix_store_checked(sg, acc[m][n], pC, size_t(N),
                                matrix::layout::row_major, size_t(M), size_t(N),
                                size_t(m0 + m * TM), size_t(col));
                    }
            } else {
                sycl_bf16* H = static_cast<sycl_bf16*>(out) + rbase * FI;
                #pragma unroll
                for (int m = 0; m < MC1 / TM; ++m)
                    #pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        matrix::joint_matrix_apply(sg, acc[m][n], acc[m][n + 2],
                            [](float& g, float& u) {
                                g = sycl::native::divide(g, 1.0f + sycl::native::exp(-g)) * u; });
                        const int r0 = m0 + m * TM;
                        const int col = nb * (NC2 / 2) + sn * (NC1 / 2) + n * TN;
                        ix::joint_matrix_apply(sg, acc[m][n], [=](float& x, size_t r, size_t c) {
                            if (r0 + int(r) < M)
                                H[size_t(r0 + int(r)) * FI + col + c] = bf16_rne(x);
                        });
                    }
            }
        });
    });
}

// ---------------------------------------------------------------------
// GROUPED MoE MXFP4 GEMM, ESIMD (default; GRIMOIRE_MOE_ESIMD=0 selects the
// joint_matrix kernel above).  Same tile table, same output.  Work-group:
// 8 m-threads x 2 n-threads over one 256-row tile; each thread owns 32 rows x
// 32 columns (EPI 0: 32 output columns; EPI 1: 16 gate rows + the 16
// matching up rows -> 16 SwiGLU outputs).  Per 128-K step the work-group
// dequantizes its B tiles ONCE into SLM (double-buffered, DPAS tile order
// [MX block][column group][K half][8 k-pairs][16 cols]): transposed 2-D
// payload loads, byte -> bf16 pair through a 1 KB SLM table, times the E8M0
// scale in fp32 (exact).  Every thread then reads its B tiles with 1 KB SLM
// block loads and runs two DPAS per 32 K straight into its accumulator; one
// barrier per 128 K.  Same bf16 B values in the same K order as the
// joint_matrix kernel, so the results are bit-identical.
// tools/moe_esimd_probe.cpp, Ornith shapes (47.8K rows, 256 experts):
//   gate_up+SwiGLU  joint_matrix ~25 TFLOP/s in-model; per-thread dequant
//                   5.2 ms (38.5); SLM-staged 2.95 ms (68.0)
//   down            per-thread 3.49 ms (28.7); SLM-staged 1.92 ms (52.2)
// 64-row thread tiles spill 7 KB and run 4-5x slower -- keep 32 rows.
// ---------------------------------------------------------------------
const uint32_t* mxfp4_pair_lut(sycl::queue& q) {
    static uint32_t* lut = nullptr;
    if (!lut) {
        static const uint32_t mag[8] = {0x0000u, 0x3F00u, 0x3F80u, 0x3FC0u,
                                        0x4000u, 0x4040u, 0x4080u, 0x40C0u};
        std::vector<uint32_t> h(256);
        for (int b = 0; b < 256; ++b) {
            const uint32_t lo = mag[b & 7] | ((b & 8) ? 0x8000u : 0u);
            const uint32_t hi = mag[(b >> 4) & 7] | ((b & 0x80) ? 0x8000u : 0u);
            h[b] = lo | (hi << 16);
        }
        lut = sycl::malloc_device<uint32_t>(256, q);
        q.memcpy(lut, h.data(), 1024).wait();
    }
    return lut;
}

template <int EPI>
sycl::event moe_mxfp4_esimd(sycl::queue& q, const QuantWeight& w, int Ne, const sycl_bf16* A,
                            void* out, const int32_t* tile_e, const int32_t* tile_mb,
                            const int32_t* off, const int32_t* cnt, int T,
                            const std::vector<sycl::event>& deps,
                            const float* rowscale = nullptr) {
    constexpr int MT = 32, TMT = MC2 / MT, TNT = 2, TPW = TMT * TNT;
    constexpr int CG = 2 * TNT;                   // 16-column groups per work-group
    constexpr int STEP = 4 * CG * 2 * 512;        // bytes of dequantized B per 128 K
    constexpr int NC = 4 * CG / TPW;              // dequant (MX block, group) pairs per thread
    static_assert(NC * TPW == 4 * CG, "dequant pairs must split evenly");
    const uint32_t* lut = mxfp4_pair_lut(q);
    const int K = w.K, FI = Ne / 2, Wrows = w.N;
    const int nG = EPI == 1 ? FI / (16 * TNT) : Ne / (32 * TNT);
    const uint8_t* pay = w.payload;
    const uint8_t* scl = static_cast<const uint8_t*>(w.scales);
    const unsigned PP = unsigned(w.row_bytes) - 1, SB = unsigned(w.row_scales);
    // Payload lines prefetched PFD 128-K steps ahead (GRIMOIRE_MOE_PFD, 0 = off):
    // at 1-3 rows per expert (a verify batch) the kernel only streams weights,
    // and one step of loads in flight per work-group left it at ~300 GB/s.
    static const int pfd_env = [] {
        const char* v = std::getenv("GRIMOIRE_MOE_PFD");
        const int d = v ? std::atoi(v) : 2;
        return d >= 0 && d <= 8 ? d : 2;
    }();
    const int PFD = pfd_env;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * nG * TPW, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            es::slm_init<1024 + 2 * STEP>();
            const int lid = int(it.get_local_id(0));
            const int g = int(it.get_group(0));
            const int t = g / nG, ng = g % nG;
            const int e = tile_e[t];
            if (e < 0) return;                        // padding of a device-built table
            if (lid < 8)
                es::slm_block_store<uint32_t, 32>(lid * 128, es::block_load<uint32_t, 32>(lut + lid * 32));
            const int ti = lid / TNT, tj = lid % TNT;
            const int M = cnt[e];
            const int m0 = tile_mb[t] * MC2 + ti * MT;
            const bool active = m0 < M;               // idle threads still dequantize
            // live 8-row blocks of this thread: a verify batch leaves 1-3 rows
            // per expert, and DPAS on the empty blocks was 4x the needed work
            // on the one thread that does all of it
            const int nrb0 = (M - m0 + 7) / 8, nrb = nrb0 < 4 ? nrb0 : 4;
            const sycl_bf16* Ae = A + size_t(off[e]) * K;
            const unsigned AW = unsigned(K) * 2 - 1, AH = unsigned(M) - 1;
            const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
            const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(Wrows) - 1;
            const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
            const es::simd<uint32_t, 16> iv(0, 1);
            es::barrier();                            // table staged
            es::simd<float, 4 * 2 * 128> acc = 0.0f;
            const int nsteps = K / 128;
            es::simd<uint32_t, 64 * NC> tw;           // payload [4 dwords][16 rows] per pair
            es::simd<uint32_t, 16 * NC> sw;           // scale dwords (4 MX blocks) per pair
            // (lambdas are not allowed in ESIMD kernels)
            #define MOE_ESIMD_FETCH(st)                                                        \
            {                                                                              \
                const int k1 = (st) * 128;                                                 \
                _Pragma("unroll")                                                          \
                for (int i = 0; i < NC; ++i) {                                             \
                    const int c = lid + i * TPW, mxc = c / CG, cg = c % CG;                \
                    const int nbc = ng * TNT + (cg >> 1);                                  \
                    const int rh = e * Ne + (EPI == 1 ? ((cg & 1) ? FI : 0) + nbc * 16     \
                                                      : nbc * 32 + (cg & 1) * 16);         \
                    tw.template select<64, 1>(64 * i) = es::load_2d<uint32_t, 4, 16, 1, true, false>( \
                        payw, PW, PH, PP, (k1 + 32 * mxc) / 8, rh);                        \
                    sw.template select<16, 1>(16 * i) = es::gather<uint32_t, 16>(sclw,     \
                        (iv + unsigned(rh)) * SB + unsigned(k1 / 32));                     \
                    if (PFD > 0 && mxc == 0 && (st) + PFD < nsteps)                        \
                        es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PP,                \
                            ((st) + PFD) * 16, rh, PFH);                                   \
                }                                                                          \
            }
            MOE_ESIMD_FETCH(0)
            for (int s = -1; s < nsteps; ++s) {
                if (s >= 0 && active) {
                    const unsigned base = 1024u + unsigned(s & 1) * STEP;
                    #pragma unroll 1
                    for (int mx = 0; mx < 4; ++mx) {
                        const int kk = s * 128 + 32 * mx;
                        // [2 K halves][32 rows][16 K] of A
                        es::simd<sycl_bf16, 1024> a = es::load_2d<sycl_bf16, 16, 32, 2>(Ae, AW, AH, AW, kk, m0);
                        #pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            es::simd<uint32_t, 256> vb = es::slm_block_load<uint32_t, 256>(
                                base + unsigned((mx * CG + tj * 2 + hh) * 2) * 512u);
                            const es::simd<sycl_bf16, 256> b0 = vb.template select<128, 1>(0).template bit_cast_view<sycl_bf16>().read();
                            const es::simd<sycl_bf16, 256> b1 = vb.template select<128, 1>(128).template bit_cast_view<sycl_bf16>().read();
                            #pragma unroll
                            for (int rb = 0; rb < 4; ++rb) {
                                if (rb < nrb) {
                                    const es::simd<sycl_bf16, 128> a0 = a.template select<128, 1>(rb * 128);
                                    const es::simd<sycl_bf16, 128> a1 = a.template select<128, 1>(512 + rb * 128);
                                    es::simd<float, 128> c = acc.template select<128, 1>((rb * 2 + hh) * 128);
                                    c = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(c, b0, a0);
                                    c = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(c, b1, a1);
                                    acc.template select<128, 1>((rb * 2 + hh) * 128) = c;
                                }
                            }
                        }
                    }
                }
                if (s + 1 < nsteps) {
                    const unsigned base = 1024u + unsigned((s + 1) & 1) * STEP;
                    #pragma unroll
                    for (int i = 0; i < NC; ++i) {
                        const int cidx = lid + i * TPW, mxc = cidx / CG, cg = cidx % CG;
                        es::simd<uint32_t, 16> ev =
                            (es::simd<uint32_t, 16>(sw.template select<16, 1>(16 * i)) >> (8 * mxc)) & 0xFFu;
                        es::simd<uint32_t, 16> sb = ev << 23;
                        sb.merge(es::simd<uint32_t, 16>(0x00400000u), ev == 0u);
                        const es::simd<float, 16> s16 = sb.template bit_cast_view<float>().read();
                        es::simd<uint32_t, 256> vb;           // [K half][8 k-pairs][16 cols]
                        #pragma unroll
                        for (int hk = 0; hk < 2; ++hk) {
                            #pragma unroll
                            for (int kp = 0; kp < 8; ++kp) {
                                const int dw = 2 * hk + (kp >> 2);
                                es::simd<uint32_t, 16> wx = tw.template select<16, 1>(64 * i + dw * 16);
                                const int j = kp & 3;
                                es::simd<uint32_t, 16> addr;
                                if (j == 0) addr = (wx << 2) & 0x3FCu;
                                else        addr = (wx >> (8 * j - 2)) & 0x3FCu;
                                es::simd<uint32_t, 16> p = es::slm_gather<uint32_t, 16>(addr);
                                es::simd<uint32_t, 16> lo = p << 16, hi = p & 0xFFFF0000u;
                                es::simd<float, 16> lof = lo.template bit_cast_view<float>().read() * s16;
                                es::simd<float, 16> hif = hi.template bit_cast_view<float>().read() * s16;
                                vb.template select<16, 1>((hk * 8 + kp) * 16) =
                                    hif.template bit_cast_view<uint32_t>().read() |
                                    (lof.template bit_cast_view<uint32_t>().read() >> 16);
                            }
                        }
                        es::slm_block_store<uint32_t, 256>(base + unsigned((mxc * CG + cg) * 2) * 512u, vb);
                    }
                    if (s + 2 < nsteps) MOE_ESIMD_FETCH(s + 2)
                }
                es::barrier();
            }
            #undef MOE_ESIMD_FETCH
            if (!active) return;
            const int nb = ng * TNT + tj;
            if constexpr (EPI == 2) {
                // bf16 output [rows][Ne] (RNE), dword pairs
                uint32_t* Ob = reinterpret_cast<uint32_t*>(static_cast<sycl_bf16*>(out) + size_t(off[e]) * Ne);
                const unsigned OW = unsigned(Ne) * 2 - 1;
                #pragma unroll
                for (int rb = 0; rb < 4; ++rb)
                    if (rb < nrb) {
                        #pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            es::simd<float, 128> v = acc.template select<128, 1>((rb * 2 + hh) * 128);
                            es::simd<uint32_t, 128> u = v.template bit_cast_view<uint32_t>().read();
                            es::simd<uint32_t, 128> r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                            es::simd<uint32_t, 64> lo = r.template select<64, 2>(0), hi = r.template select<64, 2>(1);
                            es::simd<uint32_t, 64> pk = lo | (hi << 16);
                            es::store_2d<uint32_t, 8, 8>(Ob, OW, AH, OW, (nb * 32 + 16 * hh) / 2, m0 + 8 * rb, pk);
                        }
                    }
            } else if constexpr (EPI == 0) {
                float* Oe = static_cast<float*>(out) + size_t(off[e]) * Ne;
                const unsigned OW = unsigned(Ne) * 4 - 1;
                if (rowscale) {                       // out row r *= rowscale[off + r]
                    const float* rs = rowscale + size_t(off[e]);
                    #pragma unroll
                    for (int rb = 0; rb < 4; ++rb) {
                        const es::simd<uint32_t, 8> ri(0, 1);
                        es::simd<uint32_t, 8> rows = ri + unsigned(m0 + 8 * rb);
                        es::simd_mask<8> ok = rows < unsigned(M);
                        es::simd<float, 8> sv = es::gather<float, 8>(rs, rows * 4u, ok);
                        #pragma unroll
                        for (int hh = 0; hh < 2; ++hh)
                            #pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float sr = sv[r];
                                const int o = (rb * 2 + hh) * 128 + 16 * r;
                                es::simd<float, 16> v = acc.template select<16, 1>(o);
                                acc.template select<16, 1>(o) = v * sr;
                            }
                    }
                }
                #pragma unroll
                for (int rb = 0; rb < 4; ++rb)
                    #pragma unroll
                    for (int hh = 0; hh < 2; ++hh)
                        es::store_2d<float, 16, 8>(Oe, OW, AH, OW, nb * 32 + 16 * hh, m0 + 8 * rb,
                            es::simd<float, 128>(acc.template select<128, 1>((rb * 2 + hh) * 128)));
            } else {
                // h = silu(gate) * up, as the joint_matrix kernel: g * 1/(1 + exp(-g)) * u,
                // rounded to bf16 (RNE) and stored as dword pairs
                uint32_t* He = reinterpret_cast<uint32_t*>(static_cast<sycl_bf16*>(out) + size_t(off[e]) * FI);
                const unsigned OW = unsigned(FI) * 2 - 1;
                #pragma unroll
                for (int rb = 0; rb < 4; ++rb) {
                    es::simd<float, 128> gt = acc.template select<128, 1>((rb * 2 + 0) * 128);
                    es::simd<float, 128> up = acc.template select<128, 1>((rb * 2 + 1) * 128);
                    es::simd<float, 128> hv = gt * es::inv(1.0f + es::exp2(gt * -1.4426950408889634f)) * up;
                    es::simd<uint32_t, 128> u = hv.template bit_cast_view<uint32_t>().read();
                    es::simd<uint32_t, 128> r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                    es::simd<uint32_t, 64> lo = r.template select<64, 2>(0), hi = r.template select<64, 2>(1);
                    es::simd<uint32_t, 64> pk = lo | (hi << 16);
                    es::store_2d<uint32_t, 8, 8>(He, OW, AH, OW, nb * 8, m0 + 8 * rb, pk);
                }
            }
        });
    });
}

// Small-batch variant (every tile has <= 32 rows: speculative verify / draft
// batches).  moe_mxfp4_esimd shares a dequantized tile among 8 m-threads,
// which at 1-16 rows per expert leaves ONE thread doing all the DPAS work
// behind a barrier per 128 K.  Here a work-group is TNT n-threads over rows
// [tile_mb*MC2, +32) of one expert; each thread dequantizes its own 32
// columns in registers (every weight exactly once -- there is no second
// m-thread to share with) and runs the DPAS for them; no barrier after the
// table is staged.  Same bf16 B values and K order as moe_mxfp4_esimd.
template <int EPI, int TNT>
sycl::event moe_mxfp4_esimd_small(sycl::queue& q, const QuantWeight& w, int Ne,
                                  const sycl_bf16* A, void* out, const int32_t* tile_e,
                                  const int32_t* tile_mb, const int32_t* off,
                                  const int32_t* cnt, int T,
                                  const std::vector<sycl::event>& deps) {
    constexpr int TPW = TNT;
    const uint32_t* lut = mxfp4_pair_lut(q);
    const int K = w.K, FI = Ne / 2, Wrows = w.N;
    const int nG = EPI == 1 ? FI / (16 * TNT) : Ne / (32 * TNT);
    const uint8_t* pay = w.payload;
    const uint8_t* scl = static_cast<const uint8_t*>(w.scales);
    const unsigned PP = unsigned(w.row_bytes) - 1, SB = unsigned(w.row_scales);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * nG * TPW, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            es::slm_init<1024>();
            const int lid = int(it.get_local_id(0));
            const int g = int(it.get_group(0));
            const int t = g / nG, ng = g % nG;
            const int e = tile_e[t];
            if (e < 0) return;                        // padding of a device-built table
            for (int i = lid; i < 8; i += TPW)
                es::slm_block_store<uint32_t, 32>(i * 128, es::block_load<uint32_t, 32>(lut + i * 32));
            es::barrier();
            const int M = cnt[e];
            const int m0 = tile_mb[t] * MC2;
            const int nrb0 = (M - m0 + 7) / 8, nrb = nrb0 < 4 ? nrb0 : 4;   // live 8-row blocks
            const int nb = ng * TNT + lid;
            const int rh0 = e * Ne + (EPI == 1 ? nb * 16 : nb * 32);
            const int rh1 = e * Ne + (EPI == 1 ? FI + nb * 16 : nb * 32 + 16);
            const sycl_bf16* Ae = A + size_t(off[e]) * K;
            const unsigned AW = unsigned(K) * 2 - 1, AH = unsigned(M) - 1;
            const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
            const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(Wrows) - 1;
            const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            const es::simd<uint32_t, 16> iv(0, 1);
            es::simd<float, 4 * 2 * 128> acc = 0.0f;
            for (int k = 0; k < K; k += 128) {
                if (k + 128 < K) {
                    es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PP, (k + 128) / 8, rh0, PFH);
                    es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PP, (k + 128) / 8, rh1, PFH);
                }
                es::simd<uint32_t, 16> sw0 = es::gather<uint32_t, 16>(sclw, (iv + unsigned(rh0)) * SB + unsigned(k / 32));
                es::simd<uint32_t, 16> sw1 = es::gather<uint32_t, 16>(sclw, (iv + unsigned(rh1)) * SB + unsigned(k / 32));
                #pragma unroll 1
                for (int b = 0; b < 4; ++b) {
                    const int kk = k + 32 * b;
                    es::simd<uint32_t, 64> tw0 = es::load_2d<uint32_t, 4, 16, 1, true, false>(payw, PW, PH, PP, kk / 8, rh0);
                    es::simd<uint32_t, 64> tw1 = es::load_2d<uint32_t, 4, 16, 1, true, false>(payw, PW, PH, PP, kk / 8, rh1);
                    es::simd<sycl_bf16, 1024> a = es::load_2d<sycl_bf16, 16, 32, 2>(Ae, AW, AH, AW, kk, m0);
                    #pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        es::simd<uint32_t, 16> ev;
                        if (hh == 0) ev = (sw0 >> (8 * b)) & 0xFFu;
                        else         ev = (sw1 >> (8 * b)) & 0xFFu;
                        es::simd<uint32_t, 16> sb = ev << 23;
                        sb.merge(es::simd<uint32_t, 16>(0x00400000u), ev == 0u);
                        const es::simd<float, 16> s16 = sb.template bit_cast_view<float>().read();
                        es::simd<uint32_t, 256> vb;          // [K half][8 k-pairs][16 cols]
                        #pragma unroll
                        for (int hk = 0; hk < 2; ++hk) {
                            #pragma unroll
                            for (int kp = 0; kp < 8; ++kp) {
                                const int dw = 2 * hk + (kp >> 2);
                                es::simd<uint32_t, 16> wx;
                                if (hh == 0) wx = tw0.template select<16, 1>(dw * 16);
                                else         wx = tw1.template select<16, 1>(dw * 16);
                                const int j = kp & 3;
                                es::simd<uint32_t, 16> addr;
                                if (j == 0) addr = (wx << 2) & 0x3FCu;
                                else        addr = (wx >> (8 * j - 2)) & 0x3FCu;
                                es::simd<uint32_t, 16> p = es::slm_gather<uint32_t, 16>(addr);
                                es::simd<uint32_t, 16> lo = p << 16, hi = p & 0xFFFF0000u;
                                es::simd<float, 16> lof = lo.template bit_cast_view<float>().read() * s16;
                                es::simd<float, 16> hif = hi.template bit_cast_view<float>().read() * s16;
                                vb.template select<16, 1>((hk * 8 + kp) * 16) =
                                    hif.template bit_cast_view<uint32_t>().read() |
                                    (lof.template bit_cast_view<uint32_t>().read() >> 16);
                            }
                        }
                        const es::simd<sycl_bf16, 256> b0 = vb.template select<128, 1>(0).template bit_cast_view<sycl_bf16>().read();
                        const es::simd<sycl_bf16, 256> b1 = vb.template select<128, 1>(128).template bit_cast_view<sycl_bf16>().read();
                        #pragma unroll
                        for (int rb = 0; rb < 4; ++rb) {
                            if (rb < nrb) {
                                const es::simd<sycl_bf16, 128> a0 = a.template select<128, 1>(rb * 128);
                                const es::simd<sycl_bf16, 128> a1 = a.template select<128, 1>(512 + rb * 128);
                                es::simd<float, 128> c = acc.template select<128, 1>((rb * 2 + hh) * 128);
                                c = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(c, b0, a0);
                                c = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(c, b1, a1);
                                acc.template select<128, 1>((rb * 2 + hh) * 128) = c;
                            }
                        }
                    }
                }
            }
            if constexpr (EPI == 0) {
                float* Oe = static_cast<float*>(out) + size_t(off[e]) * Ne;
                const unsigned OW = unsigned(Ne) * 4 - 1;
                #pragma unroll
                for (int rb = 0; rb < 4; ++rb)
                    if (rb < nrb) {
                        #pragma unroll
                        for (int hh = 0; hh < 2; ++hh)
                            es::store_2d<float, 16, 8>(Oe, OW, AH, OW, nb * 32 + 16 * hh, m0 + 8 * rb,
                                es::simd<float, 128>(acc.template select<128, 1>((rb * 2 + hh) * 128)));
                    }
            } else {
                uint32_t* He = reinterpret_cast<uint32_t*>(static_cast<sycl_bf16*>(out) + size_t(off[e]) * FI);
                const unsigned OW = unsigned(FI) * 2 - 1;
                #pragma unroll
                for (int rb = 0; rb < 4; ++rb)
                    if (rb < nrb) {
                        es::simd<float, 128> gt = acc.template select<128, 1>((rb * 2 + 0) * 128);
                        es::simd<float, 128> up = acc.template select<128, 1>((rb * 2 + 1) * 128);
                        es::simd<float, 128> hv = gt * es::inv(1.0f + es::exp2(gt * -1.4426950408889634f)) * up;
                        es::simd<uint32_t, 128> u = hv.template bit_cast_view<uint32_t>().read();
                        es::simd<uint32_t, 128> r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                        es::simd<uint32_t, 64> lo = r.template select<64, 2>(0), hi = r.template select<64, 2>(1);
                        es::simd<uint32_t, 64> pk = lo | (hi << 16);
                        es::store_2d<uint32_t, 8, 8>(He, OW, AH, OW, nb * 8, m0 + 8 * rb, pk);
                    }
            }
        });
    });
}

bool moe_esimd_ok(const QuantWeight& w, int Ne, bool swiglu) {
    static const int on = [] { const char* v = std::getenv("GRIMOIRE_MOE_ESIMD"); return v ? std::atoi(v) : 1; }();
    if (!on) return false;
    const int K = w.K;
    return K % 128 == 0 && w.row_bytes >= size_t(K / 2) && w.row_bytes % 16 == 0 &&
           w.row_bytes >= 64 && w.row_scales * 32 == size_t(K) && w.row_scales % 4 == 0 &&
           (swiglu ? (Ne / 2) % 32 == 0 : Ne % 64 == 0);
}

// Opt-in (GRIMOIRE_FAST_GEMM_FUSED=1).  MEASURED 2026-09-25, Qwen3.8-27B,
// 4088 tokens: correct (identical text) but 1.65x SLOWER than dequantize +
// gemm_bf16_vnni (prefill 4.49 s vs 3.14 s): B fragments from SLM plus a
// barrier per K step cost more than the dequant pass saves.  Kept for tuning
// (taller sub-group tiles would halve the SLM traffic per DPAS).
bool fused_mxfp4_ok(const QuantWeight& w) {
    static const bool on = std::getenv("GRIMOIRE_FAST_GEMM_FUSED") != nullptr;
    return on && w.fmt == Fmt::MXFP4 && w.payload && w.scales && w.N % NC2 == 0 &&
           w.K % KC1 == 0;
}

} // namespace

// Grouped MoE entry points (see gemm_mxfp4_fused_grouped).  The host builds
// the m-tile table in units of moe_grouped_rows() rows per tile.
int moe_grouped_rows() { return MC2; }
// Upload the ESIMD MoE kernels' byte -> bf16 table now: its first use waits
// on the queue, which a queue recording a command graph refuses.
void moe_esimd_warmup(sycl::queue& q) { (void)mxfp4_pair_lut(q); }
bool moe_mxfp4_grouped_esimd(const QuantWeight& w, int Ne, bool swiglu) {
    return moe_mxfp4_grouped_supported(w, Ne) && moe_esimd_ok(w, Ne, swiglu);
}
bool moe_mxfp4_grouped_supported(const QuantWeight& w, int Ne) {
    return w.fmt == Fmt::MXFP4 && w.payload && w.scales && Ne > 0 &&
           Ne % NC2 == 0 && w.K % KC1 == 0 && (w.N % Ne) == 0;
}
sycl::event launch_moe_mxfp4_grouped(sycl::queue& q, const QuantWeight& w, int Ne,
                                     bool swiglu, const sycl_bf16* A, void* out,
                                     const int32_t* tile_e, const int32_t* tile_mb,
                                     const int32_t* off, const int32_t* cnt, int T,
                                     const std::vector<sycl::event>& deps,
                                     const float* rowscale, bool out_bf16) {
    if (out_bf16 && (swiglu || rowscale || !moe_esimd_ok(w, Ne, false)))
        throw std::runtime_error("launch_moe_mxfp4_grouped: bf16 output needs the ESIMD down kernel");
    if (out_bf16)
        return moe_mxfp4_esimd<2>(q, w, Ne, A, out, tile_e, tile_mb, off, cnt, T, deps);
    if (moe_esimd_ok(w, Ne, swiglu))
        return swiglu ? moe_mxfp4_esimd<1>(q, w, Ne, A, out, tile_e, tile_mb, off, cnt, T, deps)
                      : moe_mxfp4_esimd<0>(q, w, Ne, A, out, tile_e, tile_mb, off, cnt, T, deps,
                                           rowscale);
    if (rowscale)
        throw std::runtime_error("launch_moe_mxfp4_grouped: rowscale needs the ESIMD kernel");
    return swiglu
        ? gemm_mxfp4_fused_grouped<1>(q, w, Ne, A, out, tile_e, tile_mb, off, cnt, T, deps)
        : gemm_mxfp4_fused_grouped<0>(q, w, Ne, A, out, tile_e, tile_mb, off, cnt, T, deps);
}

// ---------------------------------------------------------------------
// GEMV-style grouped MoE for verify-sized batches (M <= 16 tokens, so 1-2
// rows per touched expert).  MEASURED 2026-10-01, Ornith DFlash2 verify
// (M = 8): the routed MoE took 14.4 of 29.2 ms -- ~360 us per layer for ~50
// touched experts, ~2x their weight-streaming time -- because the SLM-staged
// DPAS kernel dequantizes a 64-row block into SLM for 1-2 rows of A.  Here a
// work-group of K/512 threads owns R weight rows (EPI 1: R gate/up pairs) of
// one expert; each thread decodes a 128-K step of its rows ONCE in the ALU
// (E2M1 nibble -> fp16 = value * 2^-14, 2^14 folded into the E8M0 scale) and
// applies it to each of the expert's rows (<= 8 per pass) read from the
// permuted bf16 A.  Same tables, inputs and outputs as the grouped GEMM.
// GRIMOIRE_MOE_GEMV_SMALL=0 = the SLM-staged kernel.
// ---------------------------------------------------------------------
template <int N_>
SYCL_ESIMD_FUNCTION inline void gv_e2m1(sycl::ext::intel::esimd::simd<uint8_t, N_> b,
                                        sycl::ext::intel::esimd::simd<float, N_>& lo,
                                        sycl::ext::intel::esimd::simd<float, N_>& hi) {
    namespace es = sycl::ext::intel::esimd;
    es::simd<uint16_t, N_> u = b;
    es::simd<uint16_t, N_> l = ((u & 0x7) << 9) | ((u & 0x8) << 12);
    es::simd<uint16_t, N_> hb = ((u & 0x70) << 5) | ((u & 0x80) << 8);
    es::simd<sycl::half, N_> lh = l.template bit_cast_view<sycl::half>();
    es::simd<sycl::half, N_> hh = hb.template bit_cast_view<sycl::half>();
    lo = lh;
    hi = hh;
}
template <int EPI, int R>
sycl::event moe_mxfp4_gemv_grouped(sycl::queue& q, const QuantWeight& w, int Ne, const sycl_bf16* A,
                                   void* out, const int32_t* tile_e, const int32_t* off,
                                   const int32_t* cnt, int T, const std::vector<sycl::event>& deps) {
    constexpr int KP = 512, TB = 8, NW = (EPI == 1) ? 2 * R : R;   // weight rows per work-group
    constexpr int RED = NW * TB;                                   // partials per thread
    const int K = w.K, KS = K / KP;
    const int FI = Ne / 2;
    const int units = (EPI == 1) ? FI / R : Ne / R;               // row groups per expert
    const uint8_t* pay = w.payload;
    const uint8_t* scl = static_cast<const uint8_t*>(w.scales);
    const int64_t rb = w.row_bytes, rs = w.row_scales;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * units * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            es::slm_init<16 * RED * 4>();
            const int t = int(it.get_local_id(0));
            const int g = int(it.get_group(0));
            const int tile = g / units, u0 = (g % units) * R;
            const int e = tile_e[tile];
            if (e < 0) return;                        // padding of a device-built table
            const int M = cnt[e];
            if (M <= 0) return;
            const int kb = t * KP;
            // weight rows of this work-group
            int64_t wr[NW];
            #pragma unroll
            for (int r = 0; r < NW; ++r) {
                if constexpr (EPI == 1) wr[r] = int64_t(e) * Ne + (r < R ? u0 + r : FI + u0 + (r - R));
                else                    wr[r] = int64_t(e) * Ne + u0 + r;
            }
            const sycl_bf16* Ae = A + size_t(off[e]) * K;
            for (int m0 = 0; m0 < M; m0 += TB) {
                const int mb = M - m0 < TB ? M - m0 : TB;
                es::simd<float, 16> acc[NW][TB];
                #pragma unroll
                for (int r = 0; r < NW; ++r)
                    #pragma unroll
                    for (int j = 0; j < TB; ++j) acc[r][j] = 0.0f;
                #pragma unroll 1
                for (int st = 0; st < KP / 128; ++st) {
                    const int k = kb + st * 128;
                    es::simd<float, 64> wl[NW], wh[NW];
                    es::simd<float, 4> sc[NW];
                    #pragma unroll
                    for (int r = 0; r < NW; ++r) {
                        es::simd<uint8_t, 64> pb = es::block_load<uint8_t, 64>(pay + wr[r] * rb + k / 2);
                        es::simd<uint8_t, 4> sb = es::block_load<uint8_t, 4>(scl + wr[r] * rs + k / 32);
                        gv_e2m1<64>(pb, wl[r], wh[r]);
                        es::simd<uint32_t, 4> sv = (es::convert<uint32_t>(sb) + 14u) << 23;
                        sc[r] = sv.template bit_cast_view<float>();
                    }
                    #pragma unroll
                    for (int j = 0; j < TB; ++j) {
                        if (j < mb) {
                            es::simd<uint16_t, 128> ab = es::block_load<uint16_t, 128>(
                                reinterpret_cast<const uint16_t*>(Ae + size_t(m0 + j) * K + k));
                            es::simd<uint32_t, 128> au = es::convert<uint32_t>(ab) << 16;
                            es::simd<float, 128> af = au.template bit_cast_view<float>();
                            es::simd<float, 64> xe = af.template select<64, 2>(0);
                            es::simd<float, 64> xo = af.template select<64, 2>(1);
                            #pragma unroll
                            for (int r = 0; r < NW; ++r) {
                                #pragma unroll
                                for (int b = 0; b < 4; ++b) {
                                    es::simd<float, 16> part =
                                        wl[r].template select<16, 1>(16 * b) * xe.template select<16, 1>(16 * b) +
                                        wh[r].template select<16, 1>(16 * b) * xo.template select<16, 1>(16 * b);
                                    acc[r][j] += part * float(sc[r][b]);
                                }
                            }
                        }
                    }
                }
                es::simd<float, RED> red = 0.0f;
                #pragma unroll
                for (int r = 0; r < NW; ++r)
                    #pragma unroll
                    for (int j = 0; j < TB; ++j) red[r * TB + j] = es::reduce<float>(acc[r][j], std::plus<>());
                if (KS > 1) {
                    es::barrier();                    // previous pass's reads done
                    es::slm_block_store<float, RED>(t * RED * 4, red);
                    es::barrier();
                    if (t == 0) {
                        red = 0.0f;
                        for (int i = 0; i < KS; ++i) red += es::slm_block_load<float, RED>(i * RED * 4);
                    }
                }
                if (t == 0) {
                    for (int j = 0; j < mb; ++j) {
                        const size_t row = size_t(off[e]) + m0 + j;
                        if constexpr (EPI == 1) {
                            sycl_bf16* hrow = static_cast<sycl_bf16*>(out) + row * FI;
                            #pragma unroll
                            for (int r = 0; r < R; ++r) {
                                const float gv = red[r * TB + j], uv = red[(R + r) * TB + j];
                                const float v = gv / (1.0f + sycl::exp(-gv)) * uv;
                                const uint32_t ub = sycl::bit_cast<uint32_t>(v);
                                const uint32_t rr = (ub + 0x7FFFu + ((ub >> 16) & 1u)) >> 16;
                                reinterpret_cast<uint16_t*>(hrow)[u0 + r] = uint16_t(rr);
                            }
                        } else {
                            float* orow = static_cast<float*>(out) + row * Ne;
                            #pragma unroll
                            for (int r = 0; r < R; ++r) orow[u0 + r] = red[r * TB + j];
                        }
                    }
                }
            }
        });
    });
}
bool moe_gemv_small_on() {
    // Default OFF.  MEASURED 2026-10-01, Ornith DFlash2 math prompt (M=8
    // verify, both WITH the tile cap above): old SLM-staged kernel 190.3
    // tok/s (5.56 accepted/step, 217/270 drafted); this GEMV kernel 169.8
    // tok/s (4.45 accepted/step, 209/321 drafted) -- a net LOSS.  The two
    // kernels are not bit-identical (different summation order, as noted
    // where moe_mxfp4_gemv_grouped is defined), and DFlash2's compounding
    // 7-deep verify is sensitive enough to that reordering that the lower
    // acceptance rate costs more than the faster per-call MoE time saves.
    // Native MTP K=1 moved the OTHER way in one run (76.1 -> 95.0 tok/s) --
    // a single data point, not trusted as a trend.  GRIMOIRE_MOE_GEMV_SMALL=1
    // opts back in for further tuning; the tile cap in grimoire.cpp is the
    // part of this change that is a proven, bit-exact win (reproduces the
    // old kernel's own accept pattern) and stays on unconditionally.
    static const bool v = [] { const char* e = std::getenv("GRIMOIRE_MOE_GEMV_SMALL");
        return e && *e == '1'; }();
    return v;
}

// Every tile holds at most 32 rows (the caller guarantees it: verify/draft
// batches).  moe_mxfp4_esimd_small when the shapes allow, else the general path.
sycl::event launch_moe_mxfp4_grouped_small(sycl::queue& q, const QuantWeight& w, int Ne,
                                           bool swiglu, const sycl_bf16* A, void* out,
                                           const int32_t* tile_e, const int32_t* tile_mb,
                                           const int32_t* off, const int32_t* cnt, int T,
                                           const std::vector<sycl::event>& deps) {
    constexpr int TNT = 8;
    if (moe_gemv_small_on() && w.fmt == Fmt::MXFP4 && w.payload && w.scales && w.K % 512 == 0 &&
        w.K / 512 <= 16 && w.row_bytes % 64 == 0 && w.row_scales % 4 == 0 &&
        (swiglu ? (Ne / 2) % 2 == 0 : Ne % 2 == 0))
        return swiglu ? moe_mxfp4_gemv_grouped<1, 2>(q, w, Ne, A, out, tile_e, off, cnt, T, deps)
                      : moe_mxfp4_gemv_grouped<0, 2>(q, w, Ne, A, out, tile_e, off, cnt, T, deps);
    // MEASURED 2026-09-30, Ornith DFlash2 verify: 47.8 ms vs 44.9 for the
    // SLM-staged kernel (per-thread dequant is a latency-bound chain, spills).
    static const bool on_env = std::getenv("GRIMOIRE_SMALL_KERNEL") != nullptr;
    if (on_env && moe_esimd_ok(w, Ne, swiglu) &&
        (swiglu ? (Ne / 2) % (16 * TNT) == 0 : Ne % (32 * TNT) == 0))
        return swiglu ? moe_mxfp4_esimd_small<1, TNT>(q, w, Ne, A, out, tile_e, tile_mb, off, cnt, T, deps)
                      : moe_mxfp4_esimd_small<0, TNT>(q, w, Ne, A, out, tile_e, tile_mb, off, cnt, T, deps);
    return launch_moe_mxfp4_grouped(q, w, Ne, swiglu, A, out, tile_e, tile_mb, off, cnt, T, deps);
}

// ---------------------------------------------------------------------
// DeltaNet prefill recurrence, 4 lanes per state row.
//
// Per token, row i of a head's state does  w = S_i . k,  corr = beta*(v_i -
// a*w),  S_i = a*S_i + corr*k,  o_i = S_i . q  -- two k_dim-long dot
// products whose results every lane needs before the next token.
// prefill.cpp's kernels spread a row over all 16 lanes (4-step sub-group
// reductions, twice per row per token): latency-bound at ~4 TFLOP/s, 246 ms
// of a 4088-token Qwen3.8-27B prefill.  Here a row lives on 4 lanes (k_dim/4
// elements each, 4 partial sums), so each reduction is two xor-shuffles and
// every lane has 4x the independent work; each lane holds 2 rows.  The
// token stream goes through SLM T tokens at a time, the next chunk loading
// while this one runs.  k_dim floats of state per lane-quad row: needs the
// 256-register mode of this library.  Same formula; the k_dim summation
// order differs from prefill.cpp's kernels.
// ---------------------------------------------------------------------
namespace {

template <int KD>
sycl::event deltanet_q4_impl(sycl::queue& q, const DeltaNetPrefillParams& p,
                             const std::vector<sycl::event>& deps) {
    constexpr int LPR = 4, GPS = SG_SIZE / LPR, KPL = KD / LPR, RR = 2;
    constexpr int SGS = 8, WG = SGS * SG_SIZE, ROWS = SGS * GPS * RR;   // 64 rows
    constexpr int T = 16;          // tokens per SLM chunk; 32 spills (541 vs 218 ms)
    constexpr int KQ4 = T * KD / (WG * 4);
    constexpr int VPW = T * ROWS / WG;
    static_assert(KPL % 4 == 0 && KQ4 >= 1 && T * KD == KQ4 * WG * 4, "k split");
    static_assert(VPW >= 1 && T * ROWS == VPW * WG && 2 * T <= WG, "chunk split");
    const int VD = p.v_dim, wg_per_head = VD / ROWS;
    const size_t n_wg = size_t(p.n_heads) * size_t(wg_per_head);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetPrefillParams pp = p;
        sycl::local_accessor<sycl::float4, 1> sk(T * KD / 4, h), sq(T * KD / 4, h);
        sycl::local_accessor<float, 1> sv(T * ROWS, h), sab(2 * T, h);
        h.parallel_for(sycl::nd_range<1>(n_wg * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const auto sg = it.get_sub_group();
            const int lane = int(sg.get_local_id()[0]);
            const int lid = int(it.get_local_id(0)), sgi = lid / SG_SIZE;
            const int grp = lane / LPR, j = lane % LPR;
            const int wg = int(it.get_group(0));
            const int head = wg / wg_per_head, rbase = (wg % wg_per_head) * ROWS;
            const int nk = pp.n_k_heads ? pp.n_k_heads : pp.n_heads;
            const int khead = (nk == pp.n_heads) ? head : head / (pp.n_heads / nk);
            const int NT = pp.n_tokens, NH = pp.n_heads;
            auto rloc = [&](int r) { return (sgi * GPS + grp) * RR + r; };
            float* Sg = pp.state + int64_t(head) * VD * KD;
            float S[RR][KPL];
            #pragma unroll
            for (int r = 0; r < RR; ++r)
                #pragma unroll
                for (int i = 0; i < KPL; i += 4) {
                    const sycl::float4 v4 = *reinterpret_cast<const sycl::float4*>(
                        Sg + int64_t(rbase + rloc(r)) * KD + j * KPL + i);
                    S[r][i] = v4[0]; S[r][i + 1] = v4[1]; S[r][i + 2] = v4[2]; S[r][i + 3] = v4[3];
                }
            const float scale = sycl::rsqrt(float(KD));
            sycl::float4 rk[KQ4], rq[KQ4];
            float rv[VPW], rab = 0.0f;
            auto fetch = [&](int t0) {
                #pragma unroll
                for (int i = 0; i < KQ4; ++i) {
                    const int e = (i * WG + lid) * 4;
                    const int t = sycl::min(t0 + e / KD, NT - 1);
                    const int64_t off = (int64_t(t) * nk + khead) * KD + e % KD;
                    rk[i] = *reinterpret_cast<const sycl::float4*>(pp.k + off);
                    rq[i] = *reinterpret_cast<const sycl::float4*>(pp.q + off);
                }
                #pragma unroll
                for (int i = 0; i < VPW; ++i) {
                    const int e = i * WG + lid;
                    const int t = sycl::min(t0 + e / ROWS, NT - 1);
                    rv[i] = pp.v[(int64_t(t) * NH + head) * VD + rbase + e % ROWS];
                }
                if (lid < 2 * T) {
                    const int t = sycl::min(t0 + lid % T, NT - 1);
                    rab = (lid < T ? pp.a : pp.beta)[int64_t(t) * NH + head];
                }
            };
            fetch(0);
            for (int t0 = 0; t0 < NT; t0 += T) {
                #pragma unroll
                for (int i = 0; i < KQ4; ++i) { sk[i * WG + lid] = rk[i]; sq[i * WG + lid] = rq[i]; }
                #pragma unroll
                for (int i = 0; i < VPW; ++i) sv[i * WG + lid] = rv[i];
                if (lid < 2 * T) sab[lid] = rab;
                sycl::group_barrier(it.get_group());
                if (t0 + T < NT) fetch(t0 + T);          // in flight while this chunk runs
                const int tn = sycl::min(T, NT - t0);
                for (int tt = 0; tt < tn; ++tt) {
                    float kr[KPL], qr[KPL];
                    #pragma unroll
                    for (int i = 0; i < KPL; i += 4) {
                        const sycl::float4 k4 = sk[(tt * KD + j * KPL + i) / 4];
                        const sycl::float4 q4 = sq[(tt * KD + j * KPL + i) / 4];
                        kr[i] = k4[0]; kr[i + 1] = k4[1]; kr[i + 2] = k4[2]; kr[i + 3] = k4[3];
                        qr[i] = q4[0]; qr[i + 1] = q4[1]; qr[i + 2] = q4[2]; qr[i + 3] = q4[3];
                    }
                    const float av = sab[tt], bv = sab[T + tt];
                    #pragma unroll
                    for (int r = 0; r < RR; ++r) {
                        float p4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        #pragma unroll
                        for (int i = 0; i < KPL; ++i) p4[i % 4] = sycl::fma(S[r][i], kr[i], p4[i % 4]);
                        float w = (p4[0] + p4[1]) + (p4[2] + p4[3]);
                        w += sycl::permute_group_by_xor(sg, w, 1);
                        w += sycl::permute_group_by_xor(sg, w, 2);
                        const float corr = bv * (sv[tt * ROWS + rloc(r)] - av * w);
                        float o4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        #pragma unroll
                        for (int i = 0; i < KPL; ++i) {
                            S[r][i] = sycl::fma(av, S[r][i], corr * kr[i]);
                            o4[i % 4] = sycl::fma(S[r][i], qr[i], o4[i % 4]);
                        }
                        float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
                        o += sycl::permute_group_by_xor(sg, o, 1);
                        o += sycl::permute_group_by_xor(sg, o, 2);
                        if (j == 0)
                            pp.out[(int64_t(t0 + tt) * NH + head) * VD + rbase + rloc(r)] = o * scale;
                    }
                }
                sycl::group_barrier(it.get_group());     // chunk fully read before it is overwritten
            }
            #pragma unroll
            for (int r = 0; r < RR; ++r)
                #pragma unroll
                for (int i = 0; i < KPL; i += 4)
                    *reinterpret_cast<sycl::float4*>(Sg + int64_t(rbase + rloc(r)) * KD + j * KPL + i) =
                        sycl::float4(S[r][i], S[r][i + 1], S[r][i + 2], S[r][i + 3]);
        });
    });
}

} // namespace

bool deltanet_q4_supported(const DeltaNetPrefillParams& p) {
    static const bool off = std::getenv("GRIMOIRE_DN_PREFILL_NOQ4") != nullptr;
    return !off && (p.k_dim == 64 || p.k_dim == 128) && p.v_dim % 64 == 0 && p.n_tokens > 0 &&
           (p.n_k_heads == 0 || p.n_heads % p.n_k_heads == 0);
}

sycl::event launch_deltanet_prefill_q4(sycl::queue& q, const DeltaNetPrefillParams& p,
                                       const std::vector<sycl::event>& deps) {
    return p.k_dim == 64 ? deltanet_q4_impl<64>(q, p, deps) : deltanet_q4_impl<128>(q, p, deps);
}

// ---------------------------------------------------------------------
// DeltaNet prefill, chunked: 16 tokens at a time, fp32 on the vector units.
//
// Per state row s (a k_dim vector) the recurrence is
//   s_t = a_t s_{t-1} + u_t k_t,   u_t = beta_t (v_t - a_t s_{t-1}.k_t)
// and over a chunk starting from s0, with gamma_j = prod_{i<=j} a_i:
//   (I + L) u = beta . (v - gamma . (K s0)),  L[j][i] = beta_j (gamma_j/gamma_i) k_j.k_i  (i < j)
//   o_j = gamma_j s0.q_j + sum_{i<=j} (gamma_j/gamma_i) (q_j.k_i) u_i
//   s_C = gamma_last s0 + sum_i (gamma_last/gamma_i) u_i k_i
// L, the output mixing matrix Mq and the decays depend only on (head,
// chunk): dn_chunk_prep builds them once and all 128 rows of the head reuse
// them.  Per row a chunk is then 2 x 16 INDEPENDENT dot products with s0, a
// 16x16 forward substitution and one state update -- no token-to-token
// dependency chain inside the chunk.  deltanet_q4_impl paid a dependent
// 4-lane reduction chain per token: 218 ms of a 4088-token Qwen3.8-27B
// prefill.  Same math; the summation order differs (GRIMOIRE_DN_VERIFY=n
// compares both on the first n layers).  Tokens past the end of the prompt
// are padded with a = 1, beta = 0, k = q = v = 0, which leaves u = 0.
// ---------------------------------------------------------------------
namespace {

constexpr int DNC = 16;                                   // tokens per chunk
constexpr int DNO_L = 0, DNO_M = DNC * DNC, DNO_EG = 2 * DNC * DNC,
              DNO_DL = DNO_EG + DNC, DNO_BT = DNO_DL + DNC, DNO_GL = DNO_BT + DNC;
constexpr int DNP = (DNO_GL + 1 + 3) / 4 * 4;             // floats per (head, chunk), float4-padded

// EMU: round every operand a matrix-unit version would feed the DPAS to bf16
// (k, q, the state in the dot products; L, Mq, r and u in the small
// products; the state-update weights and keys), accumulating in fp32 --
// GRIMOIRE_DN_BF16EMU=1, to measure that precision before building it.
inline float bfr(float x) { return float(bf16_rne(x)); }

template <int KD, bool EMU = false>
sycl::event dn_chunk_prep(sycl::queue& q, const DeltaNetPrefillParams& p, float* prep,
                          const std::vector<sycl::event>& deps) {
    constexpr int WG = DNC * DNC;
    const int NH = p.n_heads, NT = p.n_tokens, NC = (NT + DNC - 1) / DNC;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetPrefillParams pp = p;
        sycl::local_accessor<float, 1> ks(DNC * KD, h), qs(DNC * KD, h), g(DNC, h), bt(DNC, h);
        h.parallel_for(sycl::nd_range<1>(size_t(NH) * NC * WG, WG), [=](sycl::nd_item<1> it) {
            const int lid = int(it.get_local_id(0)), wg = int(it.get_group(0));
            const int head = wg / NC, c = wg % NC, t0 = c * DNC;
            const int nk = pp.n_k_heads ? pp.n_k_heads : pp.n_heads;
            const int khead = (nk == NH) ? head : head / (NH / nk);
            for (int e = lid; e < DNC * KD; e += WG) {
                const int t = t0 + e / KD, d = e % KD;
                const int64_t off = (int64_t(t) * nk + khead) * KD + d;
                ks[e] = t < NT ? pp.k[off] : 0.0f;
                qs[e] = t < NT ? pp.q[off] : 0.0f;
            }
            if (lid < DNC) {
                const int t = t0 + lid;
                g[lid] = t < NT ? sycl::log(sycl::fmax(pp.a[int64_t(t) * NH + head],
                                                        std::numeric_limits<float>::min()))
                                : 0.0f;
                bt[lid] = t < NT ? pp.beta[int64_t(t) * NH + head] : 0.0f;
            }
            sycl::group_barrier(it.get_group());
            if (lid == 0)
                for (int j = 1; j < DNC; ++j) g[j] += g[j - 1];
            sycl::group_barrier(it.get_group());
            const int j = lid / DNC, i = lid % DNC;
            float akk = 0.0f, aqk = 0.0f;
            for (int d = 0; d < KD; ++d) {
                const float ki = EMU ? bfr(ks[i * KD + d]) : ks[i * KD + d];
                akk = sycl::fma(EMU ? bfr(ks[j * KD + d]) : ks[j * KD + d], ki, akk);
                aqk = sycl::fma(EMU ? bfr(qs[j * KD + d]) : qs[j * KD + d], ki, aqk);
            }
            float* out = prep + (int64_t(head) * NC + c) * DNP;
            const float dec = sycl::exp(g[j] - g[i]);          // i <= j: <= 1
            const float lv = i < j ? bt[j] * dec * akk : 0.0f;
            const float mv = i <= j ? dec * aqk : 0.0f;
            out[DNO_L + j * DNC + i] = EMU ? bfr(lv) : lv;
            out[DNO_M + j * DNC + i] = EMU ? bfr(mv) : mv;
            if (lid < DNC) {
                out[DNO_EG + lid] = sycl::exp(g[lid]);
                out[DNO_DL + lid] = sycl::exp(g[DNC - 1] - g[lid]);
                out[DNO_BT + lid] = bt[lid];
            }
            if (lid == 0) out[DNO_GL] = sycl::exp(g[DNC - 1]);
        });
    });
}

template <int KD, bool EMU = false>
sycl::event dn_chunk_main(sycl::queue& q, const DeltaNetPrefillParams& p, const float* prep,
                          const std::vector<sycl::event>& deps) {
    constexpr int LPR = 4, GPS = SG_SIZE / LPR, KPL = KD / LPR, RR = 2;
    constexpr int SGS = 8, WG = SGS * SG_SIZE, ROWS = SGS * GPS * RR;   // 64 rows
    constexpr int C = DNC;
    constexpr int KQ4 = C * KD / (WG * 4);
    constexpr int VPW = C * ROWS / WG;
    constexpr int PP4 = DNP / 4;
    static_assert(KPL % 4 == 0 && KQ4 >= 1 && C * KD == KQ4 * WG * 4, "k split");
    constexpr int PPW = (PP4 + WG - 1) / WG;                 // prep float4 per work-item
    static_assert(VPW >= 1 && C * ROWS == VPW * WG && PPW <= 2, "chunk split");
    const int VD = p.v_dim, wg_per_head = VD / ROWS, NT = p.n_tokens, NC = (NT + C - 1) / C;
    const size_t n_wg = size_t(p.n_heads) * size_t(wg_per_head);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetPrefillParams pp = p;
        sycl::local_accessor<sycl::float4, 1> sk(C * KD / 4, h), sq(C * KD / 4, h), spr(PP4, h);
        sycl::local_accessor<float, 1> sv(C * ROWS, h);
        h.parallel_for(sycl::nd_range<1>(n_wg * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const auto sg = it.get_sub_group();
            const int lane = int(sg.get_local_id()[0]);
            const int lid = int(it.get_local_id(0)), sgi = lid / SG_SIZE;
            const int grp = lane / LPR, j4 = lane % LPR;
            const int wg = int(it.get_group(0));
            const int head = wg / wg_per_head, rbase = (wg % wg_per_head) * ROWS;
            const int nk = pp.n_k_heads ? pp.n_k_heads : pp.n_heads;
            const int khead = (nk == pp.n_heads) ? head : head / (pp.n_heads / nk);
            const int NH = pp.n_heads;
            auto rloc = [&](int r) { return (sgi * GPS + grp) * RR + r; };
            float* Sg = pp.state + int64_t(head) * VD * KD;
            float S[RR][KPL];
            #pragma unroll
            for (int r = 0; r < RR; ++r)
                #pragma unroll
                for (int i = 0; i < KPL; i += 4) {
                    const sycl::float4 v4 = *reinterpret_cast<const sycl::float4*>(
                        Sg + int64_t(rbase + rloc(r)) * KD + j4 * KPL + i);
                    S[r][i] = v4[0]; S[r][i + 1] = v4[1]; S[r][i + 2] = v4[2]; S[r][i + 3] = v4[3];
                }
            const float scale = sycl::rsqrt(float(KD));
            sycl::float4 rk[KQ4], rq[KQ4], rp[PPW];
            float rv[VPW];
            auto fetch = [&](int c) {
                const int t0 = c * C;
                #pragma unroll
                for (int i = 0; i < KQ4; ++i) {
                    const int e = (i * WG + lid) * 4;
                    const int t = t0 + e / KD;
                    const int64_t off = (int64_t(sycl::min(t, NT - 1)) * nk + khead) * KD + e % KD;
                    const sycl::float4 z(0.0f);
                    rk[i] = t < NT ? *reinterpret_cast<const sycl::float4*>(pp.k + off) : z;
                    rq[i] = t < NT ? *reinterpret_cast<const sycl::float4*>(pp.q + off) : z;
                }
                #pragma unroll
                for (int i = 0; i < VPW; ++i) {
                    const int e = i * WG + lid;
                    const int t = t0 + e / ROWS;
                    rv[i] = t < NT ? pp.v[(int64_t(t) * NH + head) * VD + rbase + e % ROWS] : 0.0f;
                }
                #pragma unroll
                for (int i = 0; i < PPW; ++i)
                    if (i * WG + lid < PP4)
                        rp[i] = reinterpret_cast<const sycl::float4*>(
                            prep + (int64_t(head) * NC + c) * DNP)[i * WG + lid];
            };
            fetch(0);
            for (int c = 0; c < NC; ++c) {
                #pragma unroll
                for (int i = 0; i < KQ4; ++i) { sk[i * WG + lid] = rk[i]; sq[i * WG + lid] = rq[i]; }
                #pragma unroll
                for (int i = 0; i < VPW; ++i) sv[i * WG + lid] = rv[i];
                #pragma unroll
                for (int i = 0; i < PPW; ++i)
                    if (i * WG + lid < PP4) spr[i * WG + lid] = rp[i];
                sycl::group_barrier(it.get_group());
                if (c + 1 < NC) fetch(c + 1);            // in flight while this chunk runs
                const float* pr = reinterpret_cast<const float*>(
                    spr.template get_multi_ptr<sycl::access::decorated::no>().get());
                // 1. dot products of the chunk-start state with every k_i and q_i
                float ks[RR][C], qs[RR][C];
                #pragma unroll
                for (int i = 0; i < C; ++i) {
                    float pk[RR], pq[RR];
                    #pragma unroll
                    for (int r = 0; r < RR; ++r) { pk[r] = 0.0f; pq[r] = 0.0f; }
                    #pragma unroll
                    for (int d = 0; d < KPL; d += 4) {
                        sycl::float4 k4 = sk[(i * KD + j4 * KPL + d) / 4];
                        sycl::float4 q4 = sq[(i * KD + j4 * KPL + d) / 4];
                        if constexpr (EMU)
                            for (int e = 0; e < 4; ++e) { k4[e] = bfr(k4[e]); q4[e] = bfr(q4[e]); }
                        #pragma unroll
                        for (int r = 0; r < RR; ++r) {
                            float s0v = S[r][d], s1v = S[r][d + 1], s2v = S[r][d + 2], s3v = S[r][d + 3];
                            if constexpr (EMU) { s0v = bfr(s0v); s1v = bfr(s1v); s2v = bfr(s2v); s3v = bfr(s3v); }
                            pk[r] = sycl::fma(s0v, k4[0], pk[r]);
                            pk[r] = sycl::fma(s1v, k4[1], pk[r]);
                            pk[r] = sycl::fma(s2v, k4[2], pk[r]);
                            pk[r] = sycl::fma(s3v, k4[3], pk[r]);
                            pq[r] = sycl::fma(s0v, q4[0], pq[r]);
                            pq[r] = sycl::fma(s1v, q4[1], pq[r]);
                            pq[r] = sycl::fma(s2v, q4[2], pq[r]);
                            pq[r] = sycl::fma(s3v, q4[3], pq[r]);
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < RR; ++r) { ks[r][i] = pk[r]; qs[r][i] = pq[r]; }
                }
                #pragma unroll
                for (int r = 0; r < RR; ++r)
                    #pragma unroll
                    for (int i = 0; i < C; ++i) {
                        ks[r][i] += sycl::permute_group_by_xor(sg, ks[r][i], 1);
                        ks[r][i] += sycl::permute_group_by_xor(sg, ks[r][i], 2);
                        qs[r][i] += sycl::permute_group_by_xor(sg, qs[r][i], 1);
                        qs[r][i] += sycl::permute_group_by_xor(sg, qs[r][i], 2);
                    }
                // 2. per row: u by forward substitution, outputs, u -> state weights
                const int t0 = c * C;
                #pragma unroll
                for (int r = 0; r < RR; ++r) {
                    const int rl = rloc(r);
                    #pragma unroll
                    for (int jj = 0; jj < C; ++jj) {      // ks[r][] becomes u
                        float u = pr[DNO_BT + jj] * (sv[jj * ROWS + rl] - pr[DNO_EG + jj] * ks[r][jj]);
                        if constexpr (EMU) u = bfr(u);
                        #pragma unroll
                        for (int i = 0; i < jj; ++i) u = sycl::fma(-pr[DNO_L + jj * C + i], ks[r][i], u);
                        ks[r][jj] = EMU ? bfr(u) : u;
                    }
                    #pragma unroll
                    for (int jj = 0; jj < C; ++jj) {
                        float o = pr[DNO_EG + jj] * qs[r][jj];
                        #pragma unroll
                        for (int i = 0; i <= jj; ++i) o = sycl::fma(pr[DNO_M + jj * C + i], ks[r][i], o);
                        if (j4 == jj % LPR && t0 + jj < NT)
                            pp.out[(int64_t(t0 + jj) * NH + head) * VD + rbase + rl] = o * scale;
                    }
                    #pragma unroll
                    for (int i = 0; i < C; ++i) {
                        const float wv = pr[DNO_DL + i] * ks[r][i];            // state weights
                        qs[r][i] = EMU ? bfr(wv) : wv;
                    }
                }
                // 3. state update: S = gamma_last S + sum_i w_i k_i
                const float gl = pr[DNO_GL];
                #pragma unroll
                for (int r = 0; r < RR; ++r)
                    #pragma unroll
                    for (int d = 0; d < KPL; ++d) S[r][d] *= gl;
                #pragma unroll
                for (int i = 0; i < C; ++i)
                    #pragma unroll
                    for (int d = 0; d < KPL; d += 4) {
                        sycl::float4 k4 = sk[(i * KD + j4 * KPL + d) / 4];
                        if constexpr (EMU)
                            for (int e = 0; e < 4; ++e) k4[e] = bfr(k4[e]);
                        #pragma unroll
                        for (int r = 0; r < RR; ++r) {
                            S[r][d]     = sycl::fma(qs[r][i], k4[0], S[r][d]);
                            S[r][d + 1] = sycl::fma(qs[r][i], k4[1], S[r][d + 1]);
                            S[r][d + 2] = sycl::fma(qs[r][i], k4[2], S[r][d + 2]);
                            S[r][d + 3] = sycl::fma(qs[r][i], k4[3], S[r][d + 3]);
                        }
                    }
                sycl::group_barrier(it.get_group());     // chunk fully read before it is overwritten
            }
            #pragma unroll
            for (int r = 0; r < RR; ++r)
                #pragma unroll
                for (int i = 0; i < KPL; i += 4)
                    *reinterpret_cast<sycl::float4*>(Sg + int64_t(rbase + rloc(r)) * KD + j4 * KPL + i) =
                        sycl::float4(S[r][i], S[r][i + 1], S[r][i + 2], S[r][i + 3]);
        });
    });
}

struct DnChunkScratch { float* prep = nullptr; size_t cap = 0; };
DnChunkScratch& dn_chunk_scratch(sycl::queue& q) {
    static std::mutex mu;
    static std::map<const sycl::queue*, DnChunkScratch> all;
    std::lock_guard<std::mutex> lk(mu);
    return all[&q];
}

} // namespace

// Opt-in (GRIMOIRE_DN_CHUNK16=1).  MEASURED 2026-09-26, Qwen3.8-27B 4088
// tokens: matches the sequential kernel to ~2e-6 relative (GRIMOIRE_DN_VERIFY)
// and reproduces the generated text byte for byte, but is SLOWER (321 vs
// 218 ms): with a row spread over 4 lanes the chunked form does about as
// many FMAs per token as the sequential one, spills 2.7 KB and reads the
// small per-chunk matrices from SLM.  It pays off only with the big
// products (K s0, Q s0, the state update) on the matrix unit.
bool deltanet_chunk16_supported(const DeltaNetPrefillParams& p) {
    static const bool on = std::getenv("GRIMOIRE_DN_CHUNK16") != nullptr;
    return on && (p.k_dim == 64 || p.k_dim == 128) && p.v_dim % 64 == 0 && p.n_tokens > 0 &&
           (p.n_k_heads == 0 || p.n_heads % p.n_k_heads == 0);
}

sycl::event launch_deltanet_prefill_chunk16(sycl::queue& q, const DeltaNetPrefillParams& p,
                                            const std::vector<sycl::event>& deps) {
    const int NC = (p.n_tokens + DNC - 1) / DNC;
    DnChunkScratch& s = dn_chunk_scratch(q);
    if (!grow(q, s.prep, s.cap, size_t(p.n_heads) * NC * DNP))
        throw std::runtime_error("deltanet chunk16: prep scratch allocation failed");
    static const bool emu = std::getenv("GRIMOIRE_DN_BF16EMU") != nullptr;
    if (p.k_dim == 64) {
        sycl::event e = emu ? dn_chunk_prep<64, true>(q, p, s.prep, deps)
                            : dn_chunk_prep<64>(q, p, s.prep, deps);
        return emu ? dn_chunk_main<64, true>(q, p, s.prep, {e}) : dn_chunk_main<64>(q, p, s.prep, {e});
    }
    sycl::event e = emu ? dn_chunk_prep<128, true>(q, p, s.prep, deps)
                        : dn_chunk_prep<128>(q, p, s.prep, deps);
    return emu ? dn_chunk_main<128, true>(q, p, s.prep, {e}) : dn_chunk_main<128>(q, p, s.prep, {e});
}

// ---------------------------------------------------------------------
// DeltaNet prefill on the MATRIX UNIT: chunked (16 tokens), ESIMD + DPAS.
//
// Same chunked algebra as dn_chunk_main (see its header), with every big
// product on DPAS in bf16 and fp32 accumulation; the state stays fp32.
// Precision measured before building it (GRIMOIRE_DN_BF16EMU on the SIMT
// chunk kernel, 48 layers, 5987 tokens): outputs <= 7.2e-3, state <= 5.4e-3
// relative to max, generated text byte-identical.
//
// One ESIMD thread owns 8 state rows of a value head; the state is 8 DPAS
// accumulator tiles [8 rows][16 d].  Per chunk (26 DPAS):
//   KS = S K^T, QS = S Q^T      A = bf16(state tile t), B = K^T / Q^T tile
//   R  = beta . (V^T - gamma . KS)            (V^T by a transposed 2-D load)
//   U  = R T^T,  O = gamma . QS + U Mq^T       (T = (I+L)^-1, Mq: dn_xmx_prep)
//   S  = gamma_last S + (U . dl) K            B = K tile, VNNI over tokens
// Each DPAS result [8 rows][16] is directly the next A operand -- no
// shuffles, no SLM, no barriers.
// ---------------------------------------------------------------------
namespace {

constexpr int DXC = 16;          // tokens per chunk

// k, q fp32 [M][nk][KD] -> KdT, QdT bf16 [nk][KD/2][Sp][2] (VNNI over d) and
// Kt bf16 [nk][Sp/2][KD][2] (VNNI over tokens); tokens >= M are zero.
sycl::event dn_xmx_pack(sycl::queue& q, const DeltaNetPrefillParams& p, int Sp,
                        sycl_bf16* KdT, sycl_bf16* QdT, sycl_bf16* Kt,
                        const std::vector<sycl::event>& deps) {
    const int KD = p.k_dim, M = p.n_tokens;
    const int nk = p.n_k_heads ? p.n_k_heads : p.n_heads;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetPrefillParams pp = p;
        h.parallel_for(sycl::range<3>(size_t(nk), size_t(Sp), size_t(KD)), [=](sycl::id<3> id) {
            const int kh = int(id[0]), s = int(id[1]), d = int(id[2]);
            const int64_t src = (int64_t(s) * nk + kh) * KD + d;
            const sycl_bf16 kv = bf16_rne(s < M ? pp.k[src] : 0.0f);
            const sycl_bf16 qv = bf16_rne(s < M ? pp.q[src] : 0.0f);
            const size_t dt = ((size_t(kh) * (KD / 2) + d / 2) * Sp + s) * 2 + (d & 1);
            KdT[dt] = kv;
            QdT[dt] = qv;
            Kt[((size_t(kh) * (Sp / 2) + s / 2) * KD + d) * 2 + (s & 1)] = kv;
        });
    });
}

// Per (value head, chunk): T = (I + L)^-1 and Mq as bf16 DPAS B tiles
// (VNNI over the contraction index i: [(i/2)*16 + j][i&1]), and the decays.
// prepB: [NH][NC][512] bf16 (T^T then Mq^T); prepF: [NH][NC][64] floats
// (gamma_j @0, gamma_last/gamma_i @16, beta_j @32, gamma_last @48).
sycl::event dn_xmx_prep(sycl::queue& q, const DeltaNetPrefillParams& p, int NC,
                        sycl_bf16* prepB, float* prepF, const std::vector<sycl::event>& deps) {
    constexpr int C = DXC, WG = C * C;
    const int NH = p.n_heads, KD = p.k_dim, M = p.n_tokens;
    const int nk = p.n_k_heads ? p.n_k_heads : NH;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetPrefillParams pp = p;
        sycl::local_accessor<float, 1> ks(C * 128, h), qs(C * 128, h);
        sycl::local_accessor<float, 1> g(C, h), bt(C, h), L(C * C, h), T(C * C, h);
        h.parallel_for(sycl::nd_range<1>(size_t(NH) * NC * WG, WG), [=](sycl::nd_item<1> it) {
            const int lid = int(it.get_local_id(0)), wg = int(it.get_group(0));
            const int head = wg / NC, c = wg % NC, t0 = c * C;
            const int khead = (nk == NH) ? head : head / (NH / nk);
            for (int e = lid; e < C * KD; e += WG) {
                const int t = t0 + e / KD, d = e % KD;
                const int64_t off = (int64_t(t) * nk + khead) * KD + d;
                ks[e] = t < M ? pp.k[off] : 0.0f;
                qs[e] = t < M ? pp.q[off] : 0.0f;
            }
            if (lid < C) {
                const int t = t0 + lid;
                g[lid] = t < M ? sycl::log(sycl::fmax(pp.a[int64_t(t) * NH + head],
                                                       std::numeric_limits<float>::min()))
                               : 0.0f;
                bt[lid] = t < M ? pp.beta[int64_t(t) * NH + head] : 0.0f;
            }
            sycl::group_barrier(it.get_group());
            if (lid == 0)
                for (int j = 1; j < C; ++j) g[j] += g[j - 1];
            sycl::group_barrier(it.get_group());
            const int j = lid / C, i = lid % C;
            float akk = 0.0f, aqk = 0.0f;
            for (int d = 0; d < KD; ++d) {
                const float ki = ks[i * KD + d];
                akk = sycl::fma(ks[j * KD + d], ki, akk);
                aqk = sycl::fma(qs[j * KD + d], ki, aqk);
            }
            const float dec = sycl::exp(g[j] - g[i]);
            L[j * C + i] = i < j ? bt[j] * dec * akk : 0.0f;
            const float mq = i <= j ? dec * aqk : 0.0f;
            T[j * C + i] = i == j ? 1.0f : 0.0f;
            sycl::group_barrier(it.get_group());
            // T = (I + L)^-1, unit lower triangular, one row at a time:
            // T[jj][i] = -sum_{m=i}^{jj-1} L[jj][m] T[m][i]   (i < jj)
            for (int jj = 1; jj < C; ++jj) {
                if (j == jj && i < jj) {
                    float t = 0.0f;
                    for (int m = i; m < jj; ++m) t = sycl::fma(L[jj * C + m], T[m * C + i], t);
                    T[jj * C + i] = -t;
                }
                sycl::group_barrier(it.get_group());
            }
            sycl_bf16* ob = prepB + (size_t(head) * NC + c) * 512;
            // B tile element (k = i, n = j): T^T[i][j] = T[j][i], Mq^T[i][j] = Mq[j][i]
            ob[((i / 2) * C + j) * 2 + (i & 1)] = bf16_rne(T[j * C + i]);
            ob[256 + ((i / 2) * C + j) * 2 + (i & 1)] = bf16_rne(mq);
            float* of = prepF + (size_t(head) * NC + c) * 64;
            if (lid < C) {
                of[lid] = sycl::exp(g[lid]);
                of[16 + lid] = sycl::exp(g[C - 1] - g[lid]);
                of[32 + lid] = bt[lid];
            }
            if (lid == 0) of[48] = sycl::exp(g[C - 1]);
        });
    });
}

// dn_xmx_prep on the matrix unit: one ESIMD thread per (value head, chunk).
// K K^T and Q K^T are 16x16x128 products -- 32 DPAS from the fp32 k/q tiles
// (rounded to bf16, as the precision emulation did) against the packed K^T;
// the decays via hardware log2/exp2; T = (I+L)^-1 by forward substitution on
// 16-wide register rows; the B tiles by an 8x16 dword transpose.  The SIMT
// version (dn_xmx_prep: SLM dots, 16 barriers) took 118 ms per prefill.
sycl::event dn_xmx_prep_es(sycl::queue& q, const DeltaNetPrefillParams& p, int NC, int Sp,
                           const sycl_bf16* KdT, sycl_bf16* prepB, float* prepF,
                           const std::vector<sycl::event>& deps) {
    constexpr int C = DXC, KD = 128, TPW = 8;
    const int NH = p.n_heads, M = p.n_tokens;
    const int nk = p.n_k_heads ? p.n_k_heads : NH;
    const size_t nth = size_t(NH) * NC, padded = (nth + TPW - 1) / TPW * TPW;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetPrefillParams pp = p;
        h.parallel_for(sycl::nd_range<1>(padded, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            const int tid = int(it.get_global_id(0));
            if (tid >= NH * NC) return;
            const int head = tid / NC, c = tid % NC, t0 = c * C;
            const int kh = (nk == NH) ? head : head / (NH / nk);
            const unsigned W = unsigned(nk) * KD * 4 - 1, Hs = unsigned(M) - 1;
            const uint32_t* kd = reinterpret_cast<const uint32_t*>(KdT + size_t(kh) * (KD / 2) * Sp * 2);
            es::simd<float, 128> akk0 = 0.0f, akk1 = 0.0f, aqk0 = 0.0f, aqk1 = 0.0f;
            #pragma unroll
            for (int sd = 0; sd < KD / 16; ++sd) {
                es::simd<float, 128> kf0 = es::load_2d<float, 16, 8>(pp.k, W, Hs, W, kh * KD + 16 * sd, t0);
                es::simd<float, 128> kf1 = es::load_2d<float, 16, 8>(pp.k, W, Hs, W, kh * KD + 16 * sd, t0 + 8);
                es::simd<float, 128> qf0 = es::load_2d<float, 16, 8>(pp.q, W, Hs, W, kh * KD + 16 * sd, t0);
                es::simd<float, 128> qf1 = es::load_2d<float, 16, 8>(pp.q, W, Hs, W, kh * KD + 16 * sd, t0 + 8);
                es::simd<uint32_t, 128> b = es::load_2d<uint32_t, 16, 8>(
                    kd, Sp * 4 - 1, KD / 2 - 1, Sp * 4 - 1, t0, 8 * sd);
                const es::simd<sycl_bf16, 256> bb = b.template bit_cast_view<sycl_bf16>().read();
                akk0 = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(akk0, bb, es::simd<sycl_bf16, 128>(kf0));
                akk1 = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(akk1, bb, es::simd<sycl_bf16, 128>(kf1));
                aqk0 = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(aqk0, bb, es::simd<sycl_bf16, 128>(qf0));
                aqk1 = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(aqk1, bb, es::simd<sycl_bf16, 128>(qf1));
            }
            const es::simd<int, 16> iv(0, 1);
            es::simd<uint32_t, 16> offs = (es::simd<uint32_t, 16>(iv) + unsigned(t0)) * unsigned(NH * 4) + unsigned(head * 4);
            es::simd_mask<16> valid = (iv + t0) < M;
            es::simd<float, 16> av = es::gather<float, 16>(pp.a, offs, valid);
            es::simd<float, 16> bv = es::gather<float, 16>(pp.beta, offs, valid);
            av.merge(es::simd<float, 16>(1.0f), !valid);
            bv.merge(es::simd<float, 16>(0.0f), !valid);
            es::simd<float, 16> g = es::log2(es::max(av, es::simd<float, 16>(std::numeric_limits<float>::min())));
            #pragma unroll
            for (int j = 1; j < C; ++j) g[j] = g[j] + g[j - 1];
            es::simd<float, 256> Lm, Mm, Tm = 0.0f;
            #pragma unroll
            for (int j = 0; j < C; ++j) {
                es::simd<float, 16> arow = j < 8 ? es::simd<float, 16>(akk0.template select<16, 1>(16 * j))
                                                 : es::simd<float, 16>(akk1.template select<16, 1>(16 * (j - 8)));
                es::simd<float, 16> qrow = j < 8 ? es::simd<float, 16>(aqk0.template select<16, 1>(16 * j))
                                                 : es::simd<float, 16>(aqk1.template select<16, 1>(16 * (j - 8)));
                const float gj = g[j], bj = bv[j];
                es::simd<float, 16> dec = es::exp2(gj - g);
                es::simd<float, 16> lr = arow * dec * bj;
                lr.merge(es::simd<float, 16>(0.0f), iv >= j);      // keep i < j
                es::simd<float, 16> mr = qrow * dec;
                mr.merge(es::simd<float, 16>(0.0f), iv > j);       // keep i <= j
                Lm.template select<16, 1>(16 * j) = lr;
                Mm.template select<16, 1>(16 * j) = mr;
            }
            // T = (I + L)^-1: T_j = e_j - sum_{m<j} L[j][m] T_m
            #pragma unroll
            for (int j = 0; j < C; ++j) {
                es::simd<float, 16> row = 0.0f;
                row[j] = 1.0f;
                #pragma unroll
                for (int m = 0; m < j; ++m) {
                    const float l = Lm[16 * j + m];
                    row -= l * es::simd<float, 16>(Tm.template select<16, 1>(16 * m));
                }
                Tm.template select<16, 1>(16 * j) = row;
            }
            // B tiles: dword (p, j) = (X[j][2p], X[j][2p+1]) -> transpose [j][p] -> [p][j]
            uint32_t* ob = reinterpret_cast<uint32_t*>(prepB + (size_t(head) * NC + c) * 512);
            #pragma unroll
            for (int w = 0; w < 2; ++w) {             // (lambdas are not allowed in ESIMD kernels)
                es::simd<sycl_bf16, 256> xb = w == 0 ? Tm : Mm;
                es::simd<uint32_t, 128> xd = xb.template bit_cast_view<uint32_t>().read();
                es::simd<uint32_t, 128> xt;
                #pragma unroll
                for (int pp2 = 0; pp2 < 8; ++pp2)
                    xt.template select<16, 1>(16 * pp2) = xd.template select<16, 8>(pp2);
                es::block_store<uint32_t, 128>(ob + 128 * w, xt);
            }
            float* of = prepF + (size_t(head) * NC + c) * 64;
            const float glast = g[C - 1];
            es::block_store<float, 16>(of, es::exp2(g));
            es::block_store<float, 16>(of + 16, es::exp2(glast - g));
            es::block_store<float, 16>(of + 32, bv);
            es::simd<float, 16> gl16 = es::exp2(es::simd<float, 16>(glast));
            es::block_store<float, 16>(of + 48, gl16);
        });
    });
}

template <int KD>
sycl::event dn_xmx_main(sycl::queue& q, const DeltaNetPrefillParams& p, int Sp,
                        const sycl_bf16* KdT, const sycl_bf16* QdT, const sycl_bf16* Kt,
                        const sycl_bf16* prepB, const float* prepF,
                        const std::vector<sycl::event>& deps) {
    constexpr int R = 8, C = DXC, NT_ = KD / 16, TPW = 8;
    const int NH = p.n_heads, VD = p.v_dim, M = p.n_tokens, NC = Sp / C;
    const int nk = p.n_k_heads ? p.n_k_heads : NH;
    const int rbh = VD / R;                       // threads per head
    const size_t nthreads = size_t(NH) * rbh;     // VD % (R*TPW) == 0 checked by the caller
    const float scale = 1.0f / std::sqrt(float(KD));
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetPrefillParams pp = p;
        h.parallel_for(sycl::nd_range<1>(nthreads, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            const int tid = int(it.get_global_id(0));
            const int head = tid / rbh, r0 = (tid % rbh) * R;
            const int kh = (nk == NH) ? head : head / (NH / nk);
            float* st = pp.state + size_t(head) * VD * KD;
            const uint32_t* kd = reinterpret_cast<const uint32_t*>(KdT + size_t(kh) * (KD / 2) * Sp * 2);
            const uint32_t* qd = reinterpret_cast<const uint32_t*>(QdT + size_t(kh) * (KD / 2) * Sp * 2);
            const uint32_t* kt = reinterpret_cast<const uint32_t*>(Kt + size_t(kh) * (Sp / 2) * KD * 2);
            es::simd<float, R * KD> S;
            #pragma unroll
            for (int t = 0; t < NT_; ++t)
                S.template select<R * 16, 1>(t * R * 16) =
                    es::load_2d<float, 16, R>(st, KD * 4 - 1, VD - 1, KD * 4 - 1, 16 * t, r0);
            // L1 prefetch two chunks ahead: every tile load of a chunk used to
            // wait out a full memory round trip (~8 us per chunk, 102 ms)
            constexpr auto PF = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            for (int c = 0; c < NC; ++c) {
                const int t0 = c * C;
                if (c + 2 < NC) {
                    const int tp = t0 + 2 * C;
                    #pragma unroll
                    for (int t = 0; t < NT_; ++t) {
                        es::prefetch_2d<uint32_t, 16, 8>(kd, Sp * 4 - 1, KD / 2 - 1, Sp * 4 - 1, tp, 8 * t, PF);
                        es::prefetch_2d<uint32_t, 16, 8>(qd, Sp * 4 - 1, KD / 2 - 1, Sp * 4 - 1, tp, 8 * t, PF);
                        es::prefetch_2d<uint32_t, 16, 8>(kt, KD * 4 - 1, Sp / 2 - 1, KD * 4 - 1, 16 * t, tp / 2, PF);
                    }
                    es::prefetch_2d<float, 8, 16>(pp.v, NH * VD * 4 - 1, M - 1, NH * VD * 4 - 1,
                                                  head * VD + r0, tp, PF);
                    const size_t rec = size_t(head) * NC + c + 2;
                    es::prefetch_2d<uint32_t, 16, 16>(reinterpret_cast<const uint32_t*>(prepB),
                        63, unsigned(NH * NC * 16) - 1, 63, 0, int(rec * 16), PF);
                    es::prefetch_2d<uint32_t, 16, 4>(reinterpret_cast<const uint32_t*>(prepF),
                        63, unsigned(NH * NC * 4) - 1, 63, 0, int(rec * 4), PF);
                }
                es::simd<float, R * 16> ks = 0.0f, qs = 0.0f;
                #pragma unroll
                for (int t = 0; t < NT_; ++t) {
                    const es::simd<sycl_bf16, R * 16> a = S.template select<R * 16, 1>(t * R * 16);
                    es::simd<uint32_t, 128> bk = es::load_2d<uint32_t, 16, 8>(
                        kd, Sp * 4 - 1, KD / 2 - 1, Sp * 4 - 1, t0, 8 * t);
                    es::simd<uint32_t, 128> bq = es::load_2d<uint32_t, 16, 8>(
                        qd, Sp * 4 - 1, KD / 2 - 1, Sp * 4 - 1, t0, 8 * t);
                    ks = xmx::dpas<8, R, float, float, sycl_bf16, sycl_bf16>(
                        ks, bk.template bit_cast_view<sycl_bf16>().read(), a);
                    qs = xmx::dpas<8, R, float, float, sycl_bf16, sycl_bf16>(
                        qs, bq.template bit_cast_view<sycl_bf16>().read(), a);
                }
                const float* pf = prepF + (size_t(head) * NC + c) * 64;
                const es::simd<float, 16> eg = es::block_load<float, 16>(pf);
                const es::simd<float, 16> dl = es::block_load<float, 16>(pf + 16);
                const es::simd<float, 16> bt = es::block_load<float, 16>(pf + 32);
                const es::simd<float, 16> gl4 = es::block_load<float, 16>(pf + 48);
                const float gl = gl4[0];
                // V^T tile [8 rows][16 tokens]: transposed 2-D load of v[t][head][r0..r0+7]
                es::simd<float, R * 16> vt = es::load_2d<float, R, 16, 1, true, false>(
                    pp.v, NH * VD * 4 - 1, M - 1, NH * VD * 4 - 1, head * VD + r0, t0);
                es::simd<float, R * 16> rr;
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    rr.template select<16, 1>(16 * r) =
                        bt * (es::simd<float, 16>(vt.template select<16, 1>(16 * r)) -
                              eg * es::simd<float, 16>(ks.template select<16, 1>(16 * r)));
                const sycl_bf16* pb = prepB + (size_t(head) * NC + c) * 512;
                es::simd<uint32_t, 128> tt = es::block_load<uint32_t, 128>(
                    reinterpret_cast<const uint32_t*>(pb));
                es::simd<uint32_t, 128> mt = es::block_load<uint32_t, 128>(
                    reinterpret_cast<const uint32_t*>(pb + 256));
                es::simd<float, R * 16> u = xmx::dpas<8, R, float, float, sycl_bf16, sycl_bf16>(
                    es::simd<float, R * 16>(0.0f), tt.template bit_cast_view<sycl_bf16>().read(),
                    es::simd<sycl_bf16, R * 16>(rr));
                es::simd<float, R * 16> o = xmx::dpas<8, R, float, float, sycl_bf16, sycl_bf16>(
                    es::simd<float, R * 16>(0.0f), mt.template bit_cast_view<sycl_bf16>().read(),
                    es::simd<sycl_bf16, R * 16>(u));
                es::simd<float, 16 * R> ot;                // [16 tokens][8 rows]
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const es::simd<float, 16> orow =
                        (es::simd<float, 16>(o.template select<16, 1>(16 * r)) +
                         eg * es::simd<float, 16>(qs.template select<16, 1>(16 * r))) * scale;
                    ot.template select<16, R>(r) = orow;
                }
                // 2-D stores are at most 8 rows high: tokens 0-7, then 8-15
                es::store_2d<float, R, 8>(pp.out, NH * VD * 4 - 1, M - 1, NH * VD * 4 - 1,
                                          head * VD + r0, t0, es::simd<float, 8 * R>(ot.template select<8 * R, 1>(0)));
                es::store_2d<float, R, 8>(pp.out, NH * VD * 4 - 1, M - 1, NH * VD * 4 - 1,
                                          head * VD + r0, t0 + 8, es::simd<float, 8 * R>(ot.template select<8 * R, 1>(8 * R)));
                es::simd<float, R * 16> uw;
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    uw.template select<16, 1>(16 * r) = es::simd<float, 16>(u.template select<16, 1>(16 * r)) * dl;
                const es::simd<sycl_bf16, R * 16> ua = uw;
                S *= gl;
                #pragma unroll
                for (int t = 0; t < NT_; ++t) {
                    es::simd<uint32_t, 128> bkt = es::load_2d<uint32_t, 16, 8>(
                        kt, KD * 4 - 1, Sp / 2 - 1, KD * 4 - 1, 16 * t, t0 / 2);
                    S.template select<R * 16, 1>(t * R * 16) = xmx::dpas<8, R, float, float, sycl_bf16, sycl_bf16>(
                        es::simd<float, R * 16>(S.template select<R * 16, 1>(t * R * 16)),
                        bkt.template bit_cast_view<sycl_bf16>().read(), ua);
                }
            }
            #pragma unroll
            for (int t = 0; t < NT_; ++t)
                es::store_2d<float, 16, R>(st, KD * 4 - 1, VD - 1, KD * 4 - 1, 16 * t, r0,
                                           es::simd<float, R * 16>(S.template select<R * 16, 1>(t * R * 16)));
        });
    });
}

struct DnXmxScratch {
    sycl_bf16* kd = nullptr; size_t kd_cap = 0;
    sycl_bf16* qd = nullptr; size_t qd_cap = 0;
    sycl_bf16* kt = nullptr; size_t kt_cap = 0;
    sycl_bf16* pb = nullptr; size_t pb_cap = 0;
    float*     pf = nullptr; size_t pf_cap = 0;
};
DnXmxScratch& dn_xmx_scratch(sycl::queue& q) {
    static std::mutex mu;
    static std::map<const sycl::queue*, DnXmxScratch> all;
    std::lock_guard<std::mutex> lk(mu);
    return all[&q];
}

} // namespace

// Default since 2026-09-26.  MEASURED, Qwen3.8-27B 4088 tokens: DN recurrence
// 219 -> 94 ms (pack 27, prep 11, main 56), prefill 2.09 -> 1.99 s; 48-layer
// GRIMOIRE_DN_VERIFY vs the fp32 sequential kernel: outputs <= 6.7e-3, state
// <= 5.8e-3 relative to max; generated text byte-identical.
// GRIMOIRE_DN_NOXMX=1 falls back to the sequential kernel.
bool deltanet_xmx_supported(const DeltaNetPrefillParams& p) {
    static const bool off = std::getenv("GRIMOIRE_DN_NOXMX") != nullptr;
    return !off && p.k_dim == 128 && p.v_dim % 64 == 0 && p.n_tokens > 0 &&
           (p.n_k_heads == 0 || p.n_heads % p.n_k_heads == 0);
}

sycl::event launch_deltanet_prefill_xmx(sycl::queue& q, const DeltaNetPrefillParams& p,
                                        const std::vector<sycl::event>& deps) {
    const int Sp = (p.n_tokens + DXC - 1) / DXC * DXC, NC = Sp / DXC;
    const int nk = p.n_k_heads ? p.n_k_heads : p.n_heads;
    DnXmxScratch& s = dn_xmx_scratch(q);
    const size_t pk = size_t(nk) * Sp * p.k_dim;
    if (!grow(q, s.kd, s.kd_cap, pk) || !grow(q, s.qd, s.qd_cap, pk) || !grow(q, s.kt, s.kt_cap, pk) ||
        !grow(q, s.pb, s.pb_cap, size_t(p.n_heads) * NC * 512) ||
        !grow(q, s.pf, s.pf_cap, size_t(p.n_heads) * NC * 64))
        throw std::runtime_error("deltanet xmx: scratch allocation failed");
    // GRIMOIRE_DN_XMX_TIMING=1: wait after each kernel, print totals at exit.
    static const bool timed = std::getenv("GRIMOIRE_DN_XMX_TIMING") != nullptr;
    struct Tot { double pack = 0, prep = 0, main = 0; int n = 0;
                 ~Tot() { if (n) std::fprintf(stderr, "  DN xmx: %d calls  pack %.1f ms  prep %.1f ms  main %.1f ms\n",
                                              n, pack, prep, main); } };
    static Tot tot;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    static const bool old_prep = std::getenv("GRIMOIRE_DN_XMX_PREP_OLD") != nullptr;
    if (timed) q.wait();                        // earlier work must not count as pack
    auto t0 = now();
    sycl::event e1 = dn_xmx_pack(q, p, Sp, s.kd, s.qd, s.kt, deps);
    if (timed) { e1.wait(); auto t1 = now(); tot.pack += ms(t0, t1); t0 = t1; }
    sycl::event e2 = old_prep ? dn_xmx_prep(q, p, NC, s.pb, s.pf, {e1})
                              : dn_xmx_prep_es(q, p, NC, Sp, s.kd, s.pb, s.pf, {e1});
    if (timed) { e2.wait(); auto t1 = now(); tot.prep += ms(t0, t1); t0 = t1; }
    sycl::event e3 = dn_xmx_main<128>(q, p, Sp, s.kd, s.qd, s.kt, s.pb, s.pf, {e2});
    if (timed) { e3.wait(); tot.main += ms(t0, now()); ++tot.n; }
    return e3;
}

bool gemm_fast_supported(const QuantWeight& w, int M) {
    static const bool off = std::getenv("GRIMOIRE_NO_FAST_GEMM") != nullptr;
    const size_t np = size_t(w.N + NC2 - 1) / NC2 * NC2;
    return !off && M >= 32 && w.payload && w.N % 16 == 0 && w.K % KC1 == 0 &&
           np * size_t(w.K) <= kMaxScratchElems;
}

namespace {

// Dequantize W once into the scratch, then one GEMM launch over all M rows.
template <int EPI>
sycl::event gemm_fast_run(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, void* out,
                          int M, const std::vector<sycl::event>& deps) {
    const int N = w.N, K = w.K;
    const bool timed = stage_timing();
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto lap = [&](double& acc, std::chrono::steady_clock::time_point& t0) {
        q.wait();
        const auto t1 = now();
        acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
    };
    if (timed) q.wait();             // earlier work must not count here
    auto t0 = now();
    if (EPI != 2 && fused_mxfp4_ok(w)) {     // N % NC2 == 0 checked there
        sycl::event e = gemm_mxfp4_fused<EPI>(q, w, x, out, M, deps);
        if (timed) { lap(stage_times().main, t0); ++stage_times().calls; }
        return e;
    }
    // N not a multiple of NC2 (DeltaNet's a|b gate projection, N = 96): the
    // scratch gets pitch Np with zeroed extra columns and the stores clip at
    // N.  It used to fall back to gemm_flt: 41 ms per Qwen prefill.
    const int Np = (N + NC2 - 1) / NC2 * NC2;
    Scratch& s = scratch_for(q);
    if (!grow(q, s.w, s.w_cap, size_t(Np) * K))
        throw std::runtime_error("gemm_fast: weight scratch allocation failed");
    sycl::event e;
    if (Np != N) {
        e = q.memset(s.w, 0, size_t(Np) * K * sizeof(sycl_bf16), deps);
        e = dequant_any(q, w, s.w, {e}, Np);
    } else if (EPI == 1) {
        e = w.fmt == Fmt::INT4 ? dequant_int4_stream(q, w, s.w, deps, N / 2)
                               : dequant_mxfp4_stream(q, w, s.w, deps, N / 2);
    } else {
        e = dequant_any(q, w, s.w, deps);
    }
    if (timed) { lap(stage_times().dq, t0); ++stage_times().calls; }
    e = gemm_bf16_vnni<EPI>(q, x, s.w, out, M, N, K, {e}, Np);
    if (timed) lap(stage_times().main, t0);
    return e;
}

} // namespace

sycl::event launch_gemm_fast(sycl::queue& q, const QuantWeight& w,
                             const sycl_bf16* x, float* y, int M,
                             const std::vector<sycl::event>& deps) {
    return gemm_fast_run<0>(q, w, x, y, M, deps);
}

// h[M][N] += x W^T in place (the residual add fused into the epilogue).
sycl::event launch_gemm_fast_residual(sycl::queue& q, const QuantWeight& w,
                                      const sycl_bf16* x, float* h, int M,
                                      const std::vector<sycl::event>& deps) {
    return gemm_fast_run<2>(q, w, x, h, M, deps);
}

bool gemm_fast_swiglu_supported(const QuantWeight& w, int M) {
    static const bool off = std::getenv("GRIMOIRE_NO_FUSED_SWIGLU") != nullptr;
    return !off && gemm_fast_supported(w, M) &&
           (w.fmt == Fmt::MXFP4 || int4_stream_ok(w)) && w.K % 128 == 0 &&
           w.N % NC2 == 0 && (w.N / 2) % 32 == 0;
}

sycl::event launch_gemm_fast_swiglu(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x,
                                    sycl_bf16* h, int M, const std::vector<sycl::event>& deps) {
    return gemm_fast_run<1>(q, w, x, h, M, deps);
}

// One tiered NVFP4 expert's whole FFN for its M rows, the block read only
// by the first three launches:
//   gate|up decoded SwiGLU-interleaved -> GEMM with silu(g*sg)*(u*su) fused,
//   bf16 h [M][I] -> down decoded -> GEMM, out fp32 [M][H] *= sd.
// *block_done = the event after which `block` is no longer read (a ring
// slot holding it can be refilled then).
sycl::event launch_nvfp4_expert_ffn(sycl::queue& q, const uint8_t* block,
                                    const NvExpertLayout& L, float sg, float su, float sd,
                                    const sycl_bf16* A, sycl_bf16* hbuf, float* out, int M,
                                    sycl_bf16* scratch, const std::vector<sycl::event>& deps,
                                    sycl::event* block_done) {
    const int H = L.H, I = L.I;
    sycl::event e = dequant_nvfp4_stream(q, block + L.gu_p, block + L.gu_s, 2 * I, H,
                                         scratch, deps, I);
    e = gemm_bf16_vnni<1, true>(q, A, scratch, hbuf, M, 2 * I, H, {e}, 0, sg, su, 0);
    e = dequant_nvfp4_stream(q, block + L.dn_p, block + L.dn_s, H, I, scratch, {e});
    if (block_done) *block_done = e;
    return gemm_bf16_vnni<0, true>(q, hbuf, scratch, out, M, H, I, {e}, 0, sd, sd, H);
}

// One tiered NVFP4 expert (b70/tiered_moe.hpp): decode its block into the
// caller's scratch, then out = A W^T with the projection scales in fp32.
sycl::event launch_nvfp4_expert_gemm(sycl::queue& q, const uint8_t* block,
                                     const NvExpertLayout& L, bool gate_up,
                                     float s0, float s1, const sycl_bf16* A, float* out,
                                     int M, sycl_bf16* scratch,
                                     const std::vector<sycl::event>& deps) {
    const int N = gate_up ? 2 * L.I : L.H, K = gate_up ? L.H : L.I;
    sycl::event e = dequant_nvfp4_stream(q, block + (gate_up ? L.gu_p : L.dn_p),
                                         block + (gate_up ? L.gu_s : L.dn_s),
                                         N, K, scratch, deps);
    return gemm_bf16_vnni<0, true>(q, A, scratch, out, M, N, K, {e}, 0,
                                   s0, gate_up ? s1 : s0, gate_up ? L.I : N);
}

// ---------------------------------------------------------------------
// Small-M MXFP4 GEMM (1..16 rows), every weight byte read ONCE for all rows:
//   Y[M][N] = X[M][K] (bf16) * W[N][K]^T
// This is batched decode's shape: serving c8 runs every dense projection
// and the lm_head at M = 8.  The grouped MoE kernel used as "one expert"
// is flat in M but ~2.5x slower than the M = 1 GEMV (handoff 10-03 s.6).
//
// One ESIMD thread owns 16 output columns x all rows x one K slice:
//   - W: two transposed 2-D block loads per 128 K (16 rows x 32 bytes), the
//     next 128 K prefetched; the 4 E8M0 scales of those 128 K in one
//     1-dword-wide 2-D load;
//   - dequant on the ALU, exact: nibble -> fp16 bits (= e2m1 * 2^-14) ->
//     fp32 -> top 16 bits = bf16, in VNNI order.  The power-of-two scale
//     2^(e - 127 + 14) multiplies each 32-K DPAS result per column, so the
//     only rounding is DPAS's own (e = 0 stays a normal float);
//   - X: 2-D block loads of 8 rows x 16 bf16, rows >= M read as zero;
//   - xmx::dpas<8,8>: 8 rows x 16 columns x 16 K per instruction.
// The K slices of a tile are one work-group, summed in SLM in a fixed
// order (deterministic).  Work-group = TPT tiles x KS slices.
// MEASURED 2026-10-03, tools/smallm_probe.cpp (weights streamed from DRAM),
// GB/s at M = 1 / 8: Ornith la_qkv 455 / 158, z/q 326 / 201, out 347 / 215,
// lm_head 538 / 463; Qwen3.8-27B o_proj 389 / 108, ffn gate_up 555 / 551.
// ---------------------------------------------------------------------
namespace {

struct SmallmPlan { int KS, kc, TPT; };
SmallmPlan smallm_plan(int N, int K) {
    const int tiles = N / 16;
    int KS = 1;
    while (KS < 16 && tiles * KS < 4096 && K / (KS * 2) >= 128) KS *= 2;
    int kc = (K + KS - 1) / KS;
    kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= 32) TPT *= 2;
    return {KS, kc, TPT};
}

template <int RBN>
sycl::event mxfp4_smallm_impl(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                              float* Y, int M, const std::vector<sycl::event>& deps) {
    constexpr int CT = 16;
    const int N = w.N, K = w.K;
    const SmallmPlan p = smallm_plan(N, K);
    const int KS = p.KS, kc = p.kc, TPT = p.TPT;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(w.payload);
    const uint32_t* sclw = static_cast<const uint32_t*>(w.scales);
    const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(N) - 1, PP = unsigned(w.row_bytes) - 1;
    const unsigned SW = unsigned(K) / 32 - 1, SP = unsigned(w.row_scales) - 1;
    const unsigned XW = unsigned(K) * 2 - 1, XH = unsigned(M) - 1;
    const unsigned YW = unsigned(N) * 4 - 1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS),
                       [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            es::slm_init<32 * RBN * 128 * 4>();
            const int lid = int(it.get_local_id(0));
            const int tw = lid / KS, ks = lid % KS;
            const int tile = int(it.get_group(0)) * TPT + tw;
            const bool live = tile < tiles;
            const int n0 = (live ? tile : 0) * CT;
            const int kb = ks * kc;
            const int ke = kb + kc < K ? kb + kc : K;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            es::simd<float, RBN * 128> acc = 0.0f;
            if (live) {
                for (int k = kb; k < ke; k += 128) {
                    if (k + 128 < ke)
                        es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (k + 128) / 8, n0, PFH);
                    es::simd<uint32_t, 8 * CT> tw0 =
                        es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8, n0);
                    es::simd<uint32_t, 8 * CT> tw1 =
                        es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8 + 8, n0);
                    es::simd<uint32_t, CT> sw = es::load_2d<uint32_t, 1, CT>(sclw, SW, PH, SP, k / 128, n0);
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const int kk = k + 32 * b;
                        es::simd<sycl_bf16, 128> a0[RBN], a1[RBN];
                        #pragma unroll
                        for (int rb = 0; rb < RBN; ++rb) {
                            a0[rb] = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk, rb * 8);
                            a1[rb] = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk + 16, rb * 8);
                        }
                        es::simd<uint32_t, 16> ev = (sw >> (8 * b)) & 0xFFu;
                        es::simd<uint32_t, 16> sbits = (ev + 14u) << 23;
                        es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                        es::simd<uint32_t, 128> vb[2];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int dw = 4 * b + j;
                            es::simd<uint32_t, 16> wv;
                            if (dw < 8) wv = tw0.template select<16, 1>(dw * CT);
                            else        wv = tw1.template select<16, 1>((dw - 8) * CT);
                            #pragma unroll
                            for (int qb = 0; qb < 4; ++qb) {
                                es::simd<uint32_t, 16> tb = qb ? (wv >> (8 * qb)) : wv;
                                es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                                es::simd<uint32_t, 16> hb = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                                es::simd<sycl::half, 32> hv = hb.template bit_cast_view<sycl::half>().read();
                                es::simd<float, 32> fv = hv;
                                es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                                es::simd<uint16_t, 32> bb = fb >> 16;
                                const int kp = 4 * j + qb;
                                vb[kp >> 3].template select<16, 1>((kp & 7) * 16) =
                                    bb.template bit_cast_view<uint32_t>().read();
                            }
                        }
                        #pragma unroll
                        for (int rb = 0; rb < RBN; ++rb) {
                            es::simd<float, 128> tmp = 0.0f;
                            tmp = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                tmp, vb[0].template bit_cast_view<sycl_bf16>().read(), a0[rb]);
                            tmp = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                tmp, vb[1].template bit_cast_view<sycl_bf16>().read(), a1[rb]);
                            #pragma unroll
                            for (int r = 0; r < 8; ++r)
                                acc.template select<16, 1>(rb * 128 + 16 * r) +=
                                    tmp.template select<16, 1>(16 * r) * sc;
                        }
                    }
                }
            }
            if (KS > 1) {
                es::slm_block_store<float, RBN * 128>(lid * RBN * 128 * 4, acc);
                es::barrier();
                if (ks != 0 || !live) return;
                for (int i = 1; i < KS; ++i)
                    acc += es::slm_block_load<float, RBN * 128>((tw * KS + i) * RBN * 128 * 4);
            } else if (!live) {
                return;
            }
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb)
                es::store_2d<float, 16, 8>(Y, YW, XH, YW, n0, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>(rb * 128)));
        });
    });
}

bool aligned64(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 63) == 0; }

// BF16 weights, the same small-M contract and thread design without any
// decode: a transposed 2-D load of the weight's dwords (two consecutive K
// of one output row each) IS the DPAS B operand in VNNI order.  For the
// BF16-critical tensors (MoE router, DeltaNet gate projections), which the
// batched step ran through launch_gemv_batch (weights re-read per 4 rows,
// SIMT FMAs): "post norm + route" was 2.3 ms of a 21.8 ms Ornith c8 step.
template <int RBN>
sycl::event bf16_smallm_impl(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                             float* Y, int M, const std::vector<sycl::event>& deps) {
    constexpr int CT = 16;
    const int N = w.N, K = w.K;
    // K slices of >= 64 (two 32-K steps): the BF16 GEMMs here are small (the
    // router is N=256) and run latency-bound on too few threads otherwise
    const int tiles = N / CT;
    int KS = 1;
    while (KS < 16 && tiles * KS < 4096 && K / (KS * 2) >= 64) KS *= 2;
    int kc = (K + KS - 1) / KS;
    kc = (kc + 31) / 32 * 32;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= 32) TPT *= 2;
    const int groups = (tiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(w.payload);
    const unsigned PW = unsigned(K) * 2 - 1, PH = unsigned(N) - 1, PP = unsigned(w.row_bytes) - 1;
    const unsigned XW = unsigned(K) * 2 - 1, XH = unsigned(M) - 1;
    const unsigned YW = unsigned(N) * 4 - 1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS),
                       [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            es::slm_init<32 * RBN * 128 * 4>();
            const int lid = int(it.get_local_id(0));
            const int tw = lid / KS, ks = lid % KS;
            const int tile = int(it.get_group(0)) * TPT + tw;
            const bool live = tile < tiles;
            const int n0 = (live ? tile : 0) * CT;
            const int kb = ks * kc;
            const int ke = kb + kc < K ? kb + kc : K;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            es::simd<float, RBN * 128> acc = 0.0f;
            if (live) {
                for (int k = kb; k < ke; k += 32) {
                    if (k + 64 < ke)
                        es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (k + 64) / 2, n0, PFH);
                    es::simd<uint32_t, 128> b0 =
                        es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 2, n0);
                    es::simd<uint32_t, 128> b1 =
                        es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 2 + 8, n0);
                    #pragma unroll
                    for (int rb = 0; rb < RBN; ++rb) {
                        es::simd<sycl_bf16, 128> a0 = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, k, rb * 8);
                        es::simd<sycl_bf16, 128> a1 = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, k + 16, rb * 8);
                        es::simd<float, 128> c = acc.template select<128, 1>(rb * 128);
                        c = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                            c, b0.template bit_cast_view<sycl_bf16>().read(), a0);
                        c = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                            c, b1.template bit_cast_view<sycl_bf16>().read(), a1);
                        acc.template select<128, 1>(rb * 128) = c;
                    }
                }
            }
            if (KS > 1) {
                es::slm_block_store<float, RBN * 128>(lid * RBN * 128 * 4, acc);
                es::barrier();
                if (ks != 0 || !live) return;
                for (int i = 1; i < KS; ++i)
                    acc += es::slm_block_load<float, RBN * 128>((tw * KS + i) * RBN * 128 * 4);
            } else if (!live) {
                return;
            }
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb)
                es::store_2d<float, 16, 8>(Y, YW, XH, YW, n0, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>(rb * 128)));
        });
    });
}

// Grouped SwiGLU gate_up with the weights as the DPAS A operand ("AW"): a
// thread owns 8 h columns of one expert, i.e. 8 gate rows + the matching 8
// up rows, read as 64-byte-wide 2-D tiles (8 rows x 128 K); the expert's
// routed rows are the B operand (transposed 2-D loads give VNNI
// [8 k-pairs][16 tokens], tokens >= rows read as zero).  C = [8 features]
// [16 tokens], so one DPAS covers every routed row of the expert (<= 16).
// MEASURED 2026-10-03, tools/moe_smallm_probe.cpp, Ornith gate_up from
// DRAM, 8 tokens x top-8 (~57 experts): 215 -> 168 us (297 -> 380 GB/s);
// 16 tokens 392 -> 287 us.  The 16-columns-per-thread kernel stays faster
// for down (K = 512), so this is gate_up only.  At ~1.3 ALU instructions
// per weight byte the E2M1 decode is now the limit (an SLM lookup table
// was slower: 219 us).
sycl::event moe_mxfp4_smallm_aw_gate_up(sycl::queue& q, const QuantWeight& w, int Ne,
                                        const sycl_bf16* A, sycl_bf16* out, const int32_t* tile_e,
                                        const int32_t* off, const int32_t* cnt, int T,
                                        const std::vector<sycl::event>& deps) {
    constexpr int NB = 2;                    // gate block + up block
    constexpr int ACC = NB * 128;
    constexpr int WGMAX = 16;
    const int K = w.K;
    const int cols = Ne / 2;
    const int ctiles = cols / 8;
    int KS = 1;
    while (KS < 16 && T * ctiles * KS < 4096 && K / (KS * 2) >= 128 && KS * 2 <= WGMAX) KS *= 2;
    int kc = (K + KS - 1) / KS;
    kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= WGMAX && TPT * 2 <= ctiles) TPT *= 2;
    const int CG = (ctiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(w.payload);
    const uint32_t* sclw = static_cast<const uint32_t*>(w.scales);
    const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(w.N) - 1, PP = unsigned(w.row_bytes) - 1;
    const unsigned SW = unsigned(K) / 32 - 1, SP = unsigned(w.row_scales) - 1;
    const unsigned XW = unsigned(K) * 2 - 1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * CG * TPT * KS, size_t(TPT) * KS),
                       [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            es::slm_init<WGMAX * ACC * 4>();
            const int lid = int(it.get_local_id(0));
            const int tw = lid / KS, ks = lid % KS;
            const int g0 = int(it.get_group(0));
            const int t = g0 / CG, ct = (g0 % CG) * TPT + tw;
            const int e = tile_e[t];
            if (e < 0) return;                        // whole work-group: same t
            const int rows = cnt[e];
            if (rows <= 0) return;
            const bool live = ct < ctiles;
            const int c0 = (live ? ct : 0) * 8;
            const uint32_t* Aw = reinterpret_cast<const uint32_t*>(A + size_t(off[e]) * K);
            const unsigned XH = unsigned(rows) - 1;
            const int wr[NB] = {e * Ne + c0, e * Ne + cols + c0};
            const int kb = ks * kc;
            const int ke = kb + kc < K ? kb + kc : K;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            const es::simd<uint32_t, 16> shv = (es::simd<uint32_t, 16>(0, 1) & 3u) << 3;
            es::simd<float, ACC> acc = 0.0f;
            if (live) {
                for (int k = kb; k < ke; k += 128) {
                    es::simd<uint32_t, 128> wt[NB];       // [8 rows][16 dwords = 128 K]
                    es::simd<uint32_t, 8> sw[NB];         // [8 rows] 4 E8M0 bytes
                    #pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        if (k + 128 < ke)
                            es::prefetch_2d<uint32_t, 16, 8>(payw, PW, PH, PP, (k + 128) / 8, wr[b], PFH);
                        wt[b] = es::load_2d<uint32_t, 16, 8>(payw, PW, PH, PP, k / 8, wr[b]);
                        sw[b] = es::load_2d<uint32_t, 1, 8>(sclw, SW, PH, SP, k / 128, wr[b]);
                    }
                    #pragma unroll
                    for (int g = 0; g < 4; ++g) {             // 32-K blocks
                        es::simd<uint32_t, 128> bt0 = es::load_2d<uint32_t, 8, 16, 1, true, false>(
                            Aw, XW, XH, XW, (k + 32 * g) / 2, 0);
                        es::simd<uint32_t, 128> bt1 = es::load_2d<uint32_t, 8, 16, 1, true, false>(
                            Aw, XW, XH, XW, (k + 32 * g) / 2 + 8, 0);
                        #pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            es::simd<uint32_t, 64> a0, a1;   // A tiles [8 rows][8 dwords]
                            #pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                es::simd<uint32_t, 16> rep =
                                    wt[b].template replicate_vs_w_hs<4, 1, 4, 0>(r * 16 + 4 * g);
                                es::simd<uint32_t, 16> tb = rep >> shv;
                                es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                                es::simd<uint32_t, 16> hb = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                                es::simd<sycl::half, 32> hv = hb.template bit_cast_view<sycl::half>().read();
                                es::simd<float, 32> fv = hv;
                                es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                                es::simd<uint16_t, 32> bb = fb >> 16;
                                es::simd<uint32_t, 16> pr = bb.template bit_cast_view<uint32_t>().read();
                                a0.template select<8, 1>(r * 8) = pr.template select<8, 1>(0);
                                a1.template select<8, 1>(r * 8) = pr.template select<8, 1>(8);
                            }
                            es::simd<float, 128> tmp = 0.0f;
                            tmp = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                tmp, bt0.template bit_cast_view<sycl_bf16>().read(),
                                a0.template bit_cast_view<sycl_bf16>().read());
                            tmp = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                tmp, bt1.template bit_cast_view<sycl_bf16>().read(),
                                a1.template bit_cast_view<sycl_bf16>().read());
                            es::simd<uint32_t, 8> ev = (sw[b] >> (8 * g)) & 0xFFu;
                            es::simd<uint32_t, 8> sbits = (ev + 14u) << 23;
                            es::simd<float, 8> sc8 = sbits.template bit_cast_view<float>().read();
                            #pragma unroll
                            for (int r = 0; r < 8; ++r)
                                acc.template select<16, 1>(b * 128 + 16 * r) +=
                                    tmp.template select<16, 1>(16 * r) * sc8[r];
                        }
                    }
                }
            }
            if (KS > 1) {
                es::slm_block_store<float, ACC>(lid * ACC * 4, acc);
                es::barrier();
                if (ks != 0 || !live) return;
                for (int i = 1; i < KS; ++i)
                    acc += es::slm_block_load<float, ACC>((tw * KS + i) * ACC * 4);
            } else if (!live) {
                return;
            }
            uint16_t* He = reinterpret_cast<uint16_t*>(out + size_t(off[e]) * cols);
            const int nr = rows < 16 ? rows : 16;
            for (int n = 0; n < nr; ++n) {
                es::simd<float, 8> gt = acc.template select<8, 16>(n);
                es::simd<float, 8> up = acc.template select<8, 16>(128 + n);
                es::simd<float, 8> hv = gt * es::inv(1.0f + es::exp2(gt * -1.4426950408889634f)) * up;
                es::simd<uint32_t, 8> u = hv.template bit_cast_view<uint32_t>().read();
                es::simd<uint32_t, 8> r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                es::simd<uint32_t, 4> pk = r.template select<4, 2>(0) | (r.template select<4, 2>(1) << 16);
                es::block_store<uint32_t, 4>(reinterpret_cast<uint32_t*>(He + size_t(n) * cols + c0), pk);
            }
        });
    });
}

// The same thread design over a grouped MoE layer: one work-group = one
// (expert tile, TPT column tiles) pair and its K slices.  At batched-decode
// sizes every touched expert has 1..16 routed rows (top-k experts are
// distinct per token, so rows <= M), i.e. one or two 8-row DPAS blocks, and
// each touched expert's weights are read once.  EPI 1: w holds [gate; up]
// per expert, the thread owns gate columns j..j+15 AND the matching up
// columns, and writes h = silu(gate) * up as bf16 [rows][Ne/2]; EPI 0:
// fp32 [rows][Ne].  Rows of expert e live at off[e], cnt[e] in A and out.
template <int EPI, int RBN>
sycl::event moe_mxfp4_smallm_impl(sycl::queue& q, const QuantWeight& w, int Ne,
                                  const sycl_bf16* A, void* out, const int32_t* tile_e,
                                  const int32_t* off, const int32_t* cnt, int T,
                                  const std::vector<sycl::event>& deps) {
    constexpr int CT = 16;
    constexpr int NCB = EPI == 1 ? 2 : 1;
    constexpr int ACC = NCB * RBN * 128;            // floats per thread
    constexpr int WGMAX = 32 / (NCB * RBN) < 8 ? 8 : 32 / (NCB * RBN);
    const int K = w.K;
    const int cols = EPI == 1 ? Ne / 2 : Ne;
    const int ctiles = cols / CT;
    int KS = 1;
    while (KS < 16 && T * ctiles * KS < 4096 && K / (KS * 2) >= 128 && KS * 2 <= WGMAX) KS *= 2;
    int kc = (K + KS - 1) / KS;
    kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= WGMAX && TPT * 2 <= ctiles) TPT *= 2;
    const int CG = (ctiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(w.payload);
    const uint32_t* sclw = static_cast<const uint32_t*>(w.scales);
    const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(w.N) - 1, PP = unsigned(w.row_bytes) - 1;
    const unsigned SW = unsigned(K) / 32 - 1, SP = unsigned(w.row_scales) - 1;
    const unsigned XW = unsigned(K) * 2 - 1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * CG * TPT * KS, size_t(TPT) * KS),
                       [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            es::slm_init<WGMAX * ACC * 4>();
            const int lid = int(it.get_local_id(0));
            const int tw = lid / KS, ks = lid % KS;
            const int g = int(it.get_group(0));
            const int t = g / CG, ct = (g % CG) * TPT + tw;
            const int e = tile_e[t];
            if (e < 0) return;                        // whole work-group: same t
            const int rows = cnt[e];
            if (rows <= 0) return;
            const bool live = ct < ctiles;
            const int j0 = (live ? ct : 0) * CT;
            const int nrb = (rows + 7) / 8;
            const sycl_bf16* Ae = A + size_t(off[e]) * K;
            const unsigned XH = unsigned(rows) - 1;
            int wr[NCB];
            wr[0] = e * Ne + j0;
            if constexpr (NCB == 2) wr[1] = e * Ne + cols + j0;
            const int kb = ks * kc;
            const int ke = kb + kc < K ? kb + kc : K;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            es::simd<float, ACC> acc = 0.0f;
            if (live) {
                for (int k = kb; k < ke; k += 128) {
                    es::simd<uint32_t, 8 * CT> tw0[NCB], tw1[NCB];
                    es::simd<uint32_t, CT> sw[NCB];
                    #pragma unroll
                    for (int cb = 0; cb < NCB; ++cb) {
                        if (k + 128 < ke)
                            es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (k + 128) / 8, wr[cb], PFH);
                        tw0[cb] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8, wr[cb]);
                        tw1[cb] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8 + 8, wr[cb]);
                        sw[cb] = es::load_2d<uint32_t, 1, CT>(sclw, SW, PH, SP, k / 128, wr[cb]);
                    }
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const int kk = k + 32 * b;
                        es::simd<sycl_bf16, 128> a0[RBN], a1[RBN];
                        #pragma unroll
                        for (int rb = 0; rb < RBN; ++rb) {
                            if (rb < nrb) {
                                a0[rb] = es::load_2d<sycl_bf16, 16, 8>(Ae, XW, XH, XW, kk, rb * 8);
                                a1[rb] = es::load_2d<sycl_bf16, 16, 8>(Ae, XW, XH, XW, kk + 16, rb * 8);
                            }
                        }
                        #pragma unroll
                        for (int cb = 0; cb < NCB; ++cb) {
                            es::simd<uint32_t, 16> ev = (sw[cb] >> (8 * b)) & 0xFFu;
                            es::simd<uint32_t, 16> sbits = (ev + 14u) << 23;
                            es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                            es::simd<uint32_t, 128> vb[2];
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const int dw = 4 * b + j;
                                es::simd<uint32_t, 16> wv;
                                if (dw < 8) wv = tw0[cb].template select<16, 1>(dw * CT);
                                else        wv = tw1[cb].template select<16, 1>((dw - 8) * CT);
                                #pragma unroll
                                for (int qb = 0; qb < 4; ++qb) {
                                    es::simd<uint32_t, 16> tb = qb ? (wv >> (8 * qb)) : wv;
                                    es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                                    es::simd<uint32_t, 16> hb = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                                    es::simd<sycl::half, 32> hv = hb.template bit_cast_view<sycl::half>().read();
                                    es::simd<float, 32> fv = hv;
                                    es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                                    es::simd<uint16_t, 32> bb = fb >> 16;
                                    const int kp = 4 * j + qb;
                                    vb[kp >> 3].template select<16, 1>((kp & 7) * 16) =
                                        bb.template bit_cast_view<uint32_t>().read();
                                }
                            }
                            #pragma unroll
                            for (int rb = 0; rb < RBN; ++rb) {
                                if (rb < nrb) {
                                    es::simd<float, 128> tmp = 0.0f;
                                    tmp = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                        tmp, vb[0].template bit_cast_view<sycl_bf16>().read(), a0[rb]);
                                    tmp = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                        tmp, vb[1].template bit_cast_view<sycl_bf16>().read(), a1[rb]);
                                    #pragma unroll
                                    for (int r = 0; r < 8; ++r)
                                        acc.template select<16, 1>((rb * NCB + cb) * 128 + 16 * r) +=
                                            tmp.template select<16, 1>(16 * r) * sc;
                                }
                            }
                        }
                    }
                }
            }
            if (KS > 1) {
                es::slm_block_store<float, ACC>(lid * ACC * 4, acc);
                es::barrier();
                if (ks != 0 || !live) return;
                for (int i = 1; i < KS; ++i)
                    acc += es::slm_block_load<float, ACC>((tw * KS + i) * ACC * 4);
            } else if (!live) {
                return;
            }
            if constexpr (EPI == 0) {
                float* Oe = static_cast<float*>(out) + size_t(off[e]) * Ne;
                const unsigned OW = unsigned(Ne) * 4 - 1;
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb)
                    if (rb < nrb)
                        es::store_2d<float, 16, 8>(Oe, OW, XH, OW, j0, rb * 8,
                            es::simd<float, 128>(acc.template select<128, 1>(rb * 128)));
            } else {
                uint32_t* He = reinterpret_cast<uint32_t*>(static_cast<sycl_bf16*>(out) + size_t(off[e]) * cols);
                const unsigned OW = unsigned(cols) * 2 - 1;
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb)
                    if (rb < nrb) {
                        es::simd<float, 128> gt = acc.template select<128, 1>((rb * 2 + 0) * 128);
                        es::simd<float, 128> up = acc.template select<128, 1>((rb * 2 + 1) * 128);
                        es::simd<float, 128> hv = gt * es::inv(1.0f + es::exp2(gt * -1.4426950408889634f)) * up;
                        es::simd<uint32_t, 128> u = hv.template bit_cast_view<uint32_t>().read();
                        es::simd<uint32_t, 128> r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                        es::simd<uint32_t, 64> lo = r.template select<64, 2>(0), hi = r.template select<64, 2>(1);
                        es::simd<uint32_t, 64> pk = lo | (hi << 16);
                        es::store_2d<uint32_t, 8, 8>(He, OW, XH, OW, j0 / 2, rb * 8, pk);
                    }
            }
        });
    });
}

} // namespace

// 2-D block loads need 64-byte-aligned bases and 16-byte pitches; the
// scales must be contiguous rows (row_scales = K/32) of a multiple of 16
// bytes.  Anything else keeps the caller's path.
bool mxfp4_smallm_ok(const QuantWeight& w, int M, const void* X, const void* Y) {
    return w.fmt == Fmt::MXFP4 && w.payload && w.scales && M >= 1 && M <= 16 &&
           w.K % 512 == 0 && w.N % 16 == 0 && w.row_bytes >= int64_t(w.K / 2) &&
           w.row_bytes % 16 == 0 && int64_t(w.row_scales) * 32 == w.K &&
           aligned64(w.payload) && aligned64(w.scales) && aligned64(X) && aligned64(Y);
}

sycl::event launch_mxfp4_smallm(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                                float* Y, int M, const std::vector<sycl::event>& deps) {
    return M <= 8 ? mxfp4_smallm_impl<1>(q, w, X, Y, M, deps)
                  : mxfp4_smallm_impl<2>(q, w, X, Y, M, deps);
}

bool bf16_smallm_ok(const QuantWeight& w, int M, const void* X, const void* Y) {
    return w.fmt == Fmt::BF16 && w.payload && M >= 1 && M <= 16 && w.K % 128 == 0 &&
           w.N % 16 == 0 && w.row_bytes >= int64_t(w.K) * 2 && w.row_bytes % 16 == 0 &&
           aligned64(w.payload) && aligned64(X) && aligned64(Y);
}

sycl::event launch_bf16_smallm(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                               float* Y, int M, const std::vector<sycl::event>& deps) {
    return M <= 8 ? bf16_smallm_impl<1>(q, w, X, Y, M, deps)
                  : bf16_smallm_impl<2>(q, w, X, Y, M, deps);
}

// INT4 groups of 64 or 128 with contiguous [N][K/GS] scale and zero rows.
bool int4_smallm_ok(const QuantWeight& w, int M, const void* X, const void* Y) {
    if (w.fmt != Fmt::INT4 || !w.payload || !w.scales || !w.zeros || M < 1 || M > 16) return false;
    const int gs = 1 << w.int4_gshift();
    return (gs == 64 || gs == 128) && int64_t(w.row_scales) * gs == w.K && w.K % 512 == 0 &&
           w.N % 16 == 0 && w.row_bytes >= int64_t(w.K / 2) && w.row_bytes % 16 == 0 &&
           aligned64(w.payload) && aligned64(X) && aligned64(Y);
}

// M > 8 (and every M with GRIMOIRE_I4_SMALLM_256=1): this library's 256
// registers and the MXFP4 kernel's split-K plan.  M <= 8 runs from
// gemv_decode.cpp (launch_int4_smallm).
sycl::event launch_int4_smallm_wide(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                                    float* Y, int M, const std::vector<sycl::event>& deps) {
    const bool g64 = (1 << w.int4_gshift()) == 64;
    const SmallmPlan p = smallm_plan(w.N, w.K);
    if (M <= 8) return g64 ? int4_smallm_impl<1, 64>(q, w, X, Y, M, p.KS, p.kc, p.TPT, deps)
                           : int4_smallm_impl<1, 128>(q, w, X, Y, M, p.KS, p.kc, p.TPT, deps);
    return g64 ? int4_smallm_impl<2, 64>(q, w, X, Y, M, p.KS, p.kc, p.TPT, deps)
               : int4_smallm_impl<2, 128>(q, w, X, Y, M, p.KS, p.kc, p.TPT, deps);
}

// Grouped: every row offset must keep the 2-D bases 64-byte aligned, i.e.
// K bf16, Ne fp32 and Ne/2 bf16 per row are multiples of 64 bytes.
bool moe_mxfp4_smallm_ok(const QuantWeight& w, int Ne, bool swiglu, int M,
                         const void* A, const void* out) {
    const int cols = swiglu ? Ne / 2 : Ne;
    return w.fmt == Fmt::MXFP4 && w.payload && w.scales && M >= 1 && M <= 16 && Ne > 0 &&
           w.N % Ne == 0 && w.K % 512 == 0 && cols % 32 == 0 && (!swiglu || Ne % 2 == 0) &&
           w.row_bytes >= int64_t(w.K / 2) && w.row_bytes % 16 == 0 &&
           int64_t(w.row_scales) * 32 == w.K && aligned64(w.payload) && aligned64(w.scales) &&
           aligned64(A) && aligned64(out);
}

sycl::event launch_moe_mxfp4_smallm(sycl::queue& q, const QuantWeight& w, int Ne, bool swiglu,
                                    const sycl_bf16* A, void* out, const int32_t* tile_e,
                                    const int32_t* off, const int32_t* cnt, int T, int M,
                                    const std::vector<sycl::event>& deps) {
    static const bool aw = [] { const char* e = std::getenv("GRIMOIRE_MOE_AW"); return !(e && *e == '0'); }();
    if (swiglu && aw && (Ne / 2) % 8 == 0)
        return moe_mxfp4_smallm_aw_gate_up(q, w, Ne, A, static_cast<sycl_bf16*>(out), tile_e, off,
                                           cnt, T, deps);
    if (swiglu)
        return M <= 8 ? moe_mxfp4_smallm_impl<1, 1>(q, w, Ne, A, out, tile_e, off, cnt, T, deps)
                      : moe_mxfp4_smallm_impl<1, 2>(q, w, Ne, A, out, tile_e, off, cnt, T, deps);
    return M <= 8 ? moe_mxfp4_smallm_impl<0, 1>(q, w, Ne, A, out, tile_e, off, cnt, T, deps)
                  : moe_mxfp4_smallm_impl<0, 2>(q, w, Ne, A, out, tile_e, off, cnt, T, deps);
}

} // namespace b70
