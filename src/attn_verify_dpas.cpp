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
// Speculative-verify attention on the matrix engine: head_dim 256, FP8 E4M3 KV.
//
// MEASURED 2026-10-08 (Qwen3.8-27B GPTQ, MTP K=6, one B70): the verify pass of a
// speculative step spent 7.6 ms in attention at 2K context and 22.1 ms at 8K, for 16
// layers x 7 query rows -- 1.4 ms per layer to read a 34 MB cache, ~25x below the
// card's memory speed.  The per-row decode kernel (flash_decode_esimd256_rows) runs
// every query head of every row on the vector ALU, decodes each K/V byte once per two
// heads, and spilled 7.7 KB of registers per thread.
//
// Here one ESIMD thread owns one KV head, one split of the keys and RB = 8 query rows,
// where a row is a (verify token, query head) pair of that KV head: G = 6 heads x 7
// tokens = 42 rows = 6 threads.  Per 16-key block, K and V are read straight from the
// FP8 cache and widened to fp16 in registers -- E4M3 widens exactly (the byte's 7
// magnitude bits at fp16 bits 7..13 and its sign at bit 15 are the value times 2^-8; the
// 2^8 folds into q and into the written partial) -- into the packed (VNNI) layout the
// matrix unit reads; S = Q K^T and O += P V are fp16 dpas with fp32 accumulation; the
// online softmax runs in the log2 domain.  Partials use the rows kernel's layout
// ([row][head][split][256] + m, l) and launch_verify_merge combines them.
//
// The thread holds Q (8 x 256 fp16, 4 KB) and O (8 x 256 fp32, 8 KB): it needs the
// 256-register file, which ESIMD kernels only get from the -doubleGRF backend option
// (grf_size<256> and -cl-intel-256-GRF-per-thread both left the old kernel spilling), so
// this file is built into its own library, bin/libgrimoire_attn.so.
// =====================================================================
#include "attn_verify.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <cstdlib>
#include <limits>

namespace b70 {
namespace {
namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;

// E4M3 byte -> the fp16 bit pattern of (value * 2^-8), in the low 16 bits.
template <int N>
SYCL_ESIMD_FUNCTION inline es::simd<uint32_t, N> e4m3_h16(es::simd<uint8_t, N> b) {
    es::simd<uint32_t, N> u = b;
    return ((u & 0x7F) << 7) | ((u & 0x80) << 8);
}

// Every split re-reads nothing, so the split count only trades parallelism against merge
// work.  MEASURED 2026-10-08 (tools/bench_verify_attn.cpp, one B70): the best counts put
// ~768 threads on the card -- 7 rows (6 threads per KV head): 32 splits at 2K (59.5 us)
// and at 8K (130 us); 1 row: 128 splits at 8K (97 us vs 157 at 32).  So the count is
// sized from a thread target, at least 32 keys per split.
//   GRIMOIRE_VERIFY_THREADS=<n>          the target (default 768)
//   GRIMOIRE_VERIFY_KEYS_PER_SPLIT=<k>   fixed keys per split instead (sweeps)
constexpr int kMinKeysPerSplit = 32;
int env_int(const char* name, int fallback) {
    const char* e = std::getenv(name);
    return e ? std::atoi(e) : fallback;
}
} // namespace

bool verify_dpas_ok(const AttnParams& p) {
    return p.head_dim == 256 && p.num_kv_heads > 0 && p.num_heads % p.num_kv_heads == 0 &&
           p.window_left <= 0 && !p.qbits && (p.seq_cap % 16) == 0;
}

int verify_dpas_splits(const AttnParams& p, const VerifyGroups& g, int len_max) {
    static const int kps = env_int("GRIMOIRE_VERIFY_KEYS_PER_SPLIT", 0);
    static const int target = env_int("GRIMOIRE_VERIFY_THREADS", 768);
    int n;
    if (kps >= 16) {
        n = (len_max + kps - 1) / kps;
    } else {
        int max_rows = 1;
        for (int i = 0; i < g.n; ++i) max_rows = g.nrows[i] > max_rows ? g.nrows[i] : max_rows;
        const int G = p.num_heads / p.num_kv_heads;
        const int tiles = (max_rows * G + 7) / 8;
        const int per_split = (g.n > 0 ? g.n : 1) * p.num_kv_heads * tiles;
        n = (target + per_split - 1) / per_split;
        const int by_len = (len_max + kMinKeysPerSplit - 1) / kMinKeysPerSplit;
        if (n > by_len) n = by_len;
    }
    if (n > MAX_SPLITS) n = MAX_SPLITS;
    return n > 0 ? n : 1;
}

sycl::event launch_verify_attn_dpas(sycl::queue& q, const AttnParams& p, const VerifyGroups& g,
                                    int64_t kv_stride, const int32_t* d_lens, int splits_max,
                                    const std::vector<sycl::event>& deps) {
    constexpr int HD = 256, RB = 8, NK = 16, DT = HD / 16;
    const int G = p.num_heads / p.num_kv_heads;
    const int KVH = p.num_kv_heads, H = p.num_heads;
    int max_rows = 1;
    for (int i = 0; i < g.n; ++i) max_rows = g.nrows[i] > max_rows ? g.nrows[i] : max_rows;
    const int tiles = (max_rows * G + RB - 1) / RB;
    const int smax = splits_max;
    const AttnParams pp = p;
    const VerifyGroups vg = g;
    const size_t nthreads = size_t(g.n) * KVH * tiles * smax;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(nthreads, 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            constexpr float NINF = -std::numeric_limits<float>::infinity();
            int id = int(it.get_group(0));
            const int part = id % smax; id /= smax;
            const int tile = id % tiles; id /= tiles;
            const int kvh = id % KVH;
            const int grp = id / KVH;
            const int row0 = vg.row0[grp], nrows = vg.nrows[grp];
            const int nr = nrows * G;
            const int j0 = tile * RB;
            if (j0 >= nr) return;
            const int len_max = d_lens[row0 + nrows - 1];
            int nsp = (len_max + kMinKeysPerSplit - 1) / kMinKeysPerSplit;
            if (nsp > smax) nsp = smax;
            if (nsp < 1) nsp = 1;
            int per = (len_max + nsp - 1) / nsp;
            per = (per + NK - 1) / NK * NK;
            const int s_beg = part * per;
            const int s_end = s_beg + per < len_max ? s_beg + per : len_max;

            int orow[RB], ohead[RB];
            es::simd<int, RB> lim = 0;               // row i sees keys < lim[i]
            #pragma unroll
            for (int i = 0; i < RB; ++i) {
                const int j = j0 + i;
                if (j < nr) {
                    orow[i] = row0 + j / G;
                    ohead[i] = kvh * G + j % G;
                    lim[i] = d_lens[orow[i]];
                } else {
                    orow[i] = -1;
                    ohead[i] = 0;
                }
            }
            if (part >= nsp || s_beg >= s_end) {
                #pragma unroll
                for (int i = 0; i < RB; ++i) {
                    if (orow[i] < 0) continue;
                    const int64_t pidx = (int64_t(orow[i]) * H + ohead[i]) * smax + part;
                    pp.part_m[pidx] = NINF;
                    pp.part_l[pidx] = 0.0f;
                }
                return;
            }

            // Q: DT chunks of [RB][16] fp16, scaled into the log2 domain (and by 2^8 for K's 2^-8).
            const float qs = pp.softmax_scale * 256.0f * 1.4426950408889634f;
            es::simd<fp16, RB * HD> qa = 0.0f;
            #pragma unroll
            for (int i = 0; i < RB; ++i) {
                if (orow[i] < 0) continue;
                es::simd<float, HD> qf =
                    es::block_load<float, HD>(pp.q + (int64_t(orow[i]) * H + ohead[i]) * HD) * qs;
                #pragma unroll
                for (int dt = 0; dt < DT; ++dt)
                    qa.template select<16, 1>((dt * RB + i) * 16) = qf.template select<16, 1>(16 * dt);
            }

            const int64_t cbase = int64_t(vg.slot[grp]) * kv_stride;
            const uint32_t* kh32 = reinterpret_cast<const uint32_t*>(
                pp.k_cache + cbase + int64_t(kvh) * HD * pp.seq_cap);
            const uint32_t* vh32 = reinterpret_cast<const uint32_t*>(
                pp.v_cache + cbase + int64_t(kvh) * pp.seq_cap * HD);
            const unsigned KW = unsigned(pp.seq_cap) - 1;    // K: [256 rows][seq_cap bytes]
            const unsigned VH = unsigned(pp.seq_cap) - 1;    // V: [seq_cap rows][256 bytes]

            es::simd<float, RB * HD> o = 0.0f;               // DT tiles of [RB][16]
            es::simd<float, RB> m = NINF, l = 0.0f;
            const es::simd<int, NK> col(0, 1);
            for (int s0 = s_beg; s0 < s_end; s0 += NK) {
                // S = Q K^T over 256 dims: 8 loads of [32 dims][16 keys], 2 dpas each.
                es::simd<float, RB * NK> s = 0.0f;
                #pragma unroll
                for (int dc = 0; dc < HD / 32; ++dc) {
                    es::simd<uint32_t, 128> kr = es::load_2d<uint32_t, 4, 32>(
                        kh32, KW, HD - 1, KW, s0 / 4, 32 * dc);
                    es::simd<uint8_t, 512> kb = kr.template bit_cast_view<uint8_t>();
                    #pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        es::simd<uint32_t, 128> b;
                        #pragma unroll
                        for (int pr = 0; pr < 8; ++pr) {
                            es::simd<uint32_t, 16> lo =
                                e4m3_h16<16>(kb.template select<16, 1>((16 * hh + 2 * pr) * 16));
                            es::simd<uint32_t, 16> hi =
                                e4m3_h16<16>(kb.template select<16, 1>((16 * hh + 2 * pr + 1) * 16));
                            b.template select<16, 1>(16 * pr) = lo | (hi << 16);
                        }
                        const int dt = 2 * dc + hh;
                        s = xmx::dpas<8, RB, float, float, fp16, fp16>(
                            s, b.template bit_cast_view<fp16>().read(),
                            es::simd<fp16, RB * 16>(qa.template select<RB * 16, 1>(dt * RB * 16)));
                    }
                }
                // mask: each row its own length (causal within the verify block)
                #pragma unroll
                for (int i = 0; i < RB; ++i) {
                    const int li = lim[i];
                    es::simd<float, NK> v = s.template select<NK, 1>(NK * i);
                    v.merge(es::simd<float, NK>(NINF), col + s0 >= li);
                    s.template select<NK, 1>(NK * i) = v;
                }
                es::simd<float, RB> bm;
                #pragma unroll
                for (int i = 0; i < RB; ++i)
                    bm[i] = es::hmax<float>(es::simd<float, NK>(s.template select<NK, 1>(NK * i)));
                es::simd<float, RB> mn = es::max(m, bm);
                es::simd<float, RB> ms = mn;
                ms.merge(es::simd<float, RB>(0.0f), mn == NINF);
                es::simd<float, RB> corr = es::exp2(m - ms);   // m = -inf -> 0
                es::simd<float, RB> rs;
                #pragma unroll
                for (int i = 0; i < RB; ++i) {
                    const float msi = ms[i];
                    es::simd<float, NK> pv = es::exp2(es::simd<float, NK>(s.template select<NK, 1>(NK * i)) - msi);
                    s.template select<NK, 1>(NK * i) = pv;
                    rs[i] = es::reduce<float>(pv, std::plus<>());
                }
                l = l * corr + rs;
                m = mn;
                float cr[RB];
                #pragma unroll
                for (int i = 0; i < RB; ++i) cr[i] = corr[i];
                #pragma unroll
                for (int dt = 0; dt < DT; ++dt)
                    #pragma unroll
                    for (int i = 0; i < RB; ++i)
                        o.template select<16, 1>((dt * RB + i) * 16) *= cr[i];
                const es::simd<fp16, RB * NK> pa = s;
                // O += P V: 4 loads of [16 keys][64 dims], 4 dpas each (VNNI over key pairs).
                #pragma unroll
                for (int vc = 0; vc < 4; ++vc) {
                    es::simd<uint32_t, 256> vr = es::load_2d<uint32_t, 16, 16>(
                        vh32, HD - 1, VH, HD - 1, 16 * vc, s0);
                    es::simd<uint8_t, 1024> vb = vr.template bit_cast_view<uint8_t>();
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        es::simd<uint32_t, 128> b;
                        #pragma unroll
                        for (int pr = 0; pr < 8; ++pr) {
                            es::simd<uint32_t, 16> lo =
                                e4m3_h16<16>(vb.template select<16, 1>((2 * pr) * 64 + 16 * w));
                            es::simd<uint32_t, 16> hi =
                                e4m3_h16<16>(vb.template select<16, 1>((2 * pr + 1) * 64 + 16 * w));
                            b.template select<16, 1>(16 * pr) = lo | (hi << 16);
                        }
                        const int dt = 4 * vc + w;
                        o.template select<RB * 16, 1>(dt * RB * 16) =
                            xmx::dpas<8, RB, float, float, fp16, fp16>(
                                es::simd<float, RB * 16>(o.template select<RB * 16, 1>(dt * RB * 16)),
                                b.template bit_cast_view<fp16>().read(), pa);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < RB; ++i) {
                if (orow[i] < 0) continue;
                const int64_t pidx = (int64_t(orow[i]) * H + ohead[i]) * smax + part;
                es::simd<float, HD> orow_v;
                #pragma unroll
                for (int dt = 0; dt < DT; ++dt)
                    orow_v.template select<16, 1>(16 * dt) = o.template select<16, 1>((dt * RB + i) * 16);
                es::block_store<float, HD>(pp.partials + pidx * HD, orow_v * 256.0f);   // undo V's 2^-8
                pp.part_m[pidx] = m[i];
                pp.part_l[pidx] = l[i];
            }
        });
    });
}

sycl::event launch_verify_merge(sycl::queue& q, const AttnParams& p, int rows, int splits_max,
                                const std::vector<sycl::event>& deps) {
    const AttnParams pp = p;
    const int smax = splits_max, HD = p.head_dim;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows) * pp.num_heads * HD), [=](sycl::id<1> idx) {
            const int64_t i = int64_t(idx[0]);
            const int d = int(i % HD);
            const int64_t rh = i / HD;
            const float* pm = pp.part_m + rh * smax;
            const float* pl = pp.part_l + rh * smax;
            float mx = -std::numeric_limits<float>::infinity();
            for (int s = 0; s < smax; ++s) mx = sycl::fmax(mx, pm[s]);
            float lsum = 0.0f, acc = 0.0f;
            for (int s = 0; s < smax; ++s) {
                const float ms = pm[s];
                if (sycl::isinf(ms)) continue;
                const float e = sycl::exp2(ms - mx);
                lsum += pl[s] * e;
                acc += pp.partials[(rh * smax + s) * HD + d] * e;
            }
            pp.out[rh * HD + d] = lsum > 0.0f ? acc / lsum : 0.0f;
        });
    });
}

} // namespace b70
