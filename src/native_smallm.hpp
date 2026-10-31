// GGUF native weights against 2..8 bf16 activation rows on DPAS (speculative verify, batched decode).
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Y[M][N] = X[M][K] W^T for W in Q4_K / Q5_K / Q6_K / Q8_0 / IQ4_XS / IQ4_NL blocks.  The thread
// design of int4_smallm_fdq_impl (int4_smallm.hpp): a thread owns 16 weight rows -- the DPAS
// columns, one per lane -- and a K slice; KS threads split K and add their partial sums through
// SLM.  Per 256 K each lane gathers its own row's block bytes (LSC gathers of several dwords per
// address: the result holds dword v of every lane at [16 v, 16 v + 16)), and the quants become
// EXACT bf16 integers for the B operand: 128 + code for the unsigned 4- / 5- / 6-bit codes (OR into
// 0x4300), the two signed nibbles of a Q8_0 int8 the same way, the IQ4 table value itself
// (nq_iq4_value).  A second DPAS against an all-ones operand gives sum x per scale group, so a group
// adds  sc * T - cs * sum x  with  cs = code offset * sc (+ the K-quant min).  The only roundings
// are the bf16 activations and fp32 accumulation, as in the INT4 / FP8 small-M kernels.
// A dword of four codes gives the B pairs (code 0, code 2) and (code 1, code 3) with two masks, so
// every 4-K block of A is reordered [x0 x2 x1 x3] in registers.
// 2026-10-29: before this, a GGUF verify pass took launch_gemv_batch (SIMT, weights re-read per 4
// rows), so MTP could not pay on --proj native.
#pragma once
#include "kernels.hpp"
#include "native_qk_esimd.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

namespace b70 {
namespace nsm {
namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using u32x16 = es::simd<uint32_t, 16>;
using f32x16 = es::simd<float, 16>;

// 16 K of X (rows rb*8 .. rb*8+7) as the DPAS A operand, each 4-K block reordered [x0 x2 x1 x3].
template <int RBN>
SYCL_ESIMD_FUNCTION inline void load_a(const sycl_bf16* X, unsigned XW, unsigned XH, int kk,
                                       es::simd<sycl_bf16, 128> (&a)[RBN]) {
    #pragma unroll
    for (int rb = 0; rb < RBN; ++rb) {
        es::simd<sycl_bf16, 128> t = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk, rb * 8);
        es::simd<uint16_t, 128> s = t.template bit_cast_view<uint16_t>();
        es::simd<uint16_t, 128> d;
        d.template select<32, 4>(0) = s.template select<32, 4>(0);
        d.template select<32, 4>(1) = s.template select<32, 4>(2);
        d.template select<32, 4>(2) = s.template select<32, 4>(1);
        d.template select<32, 4>(3) = s.template select<32, 4>(3);
        a[rb] = d.template bit_cast_view<sycl_bf16>().read();
    }
}

// Four codes (< 128, one per byte of each lane's dword) as B pairs 2c = (code 0, code 2) and
// 2c + 1 = (code 1, code 3), each bf16 (128 + code).
SYCL_ESIMD_FUNCTION inline void put_codes(es::simd<uint32_t, 128>& vb, int c, u32x16 d) {
    vb.template select<16, 1>(32 * c) = (d & 0x00FF00FFu) | 0x43004300u;
    vb.template select<16, 1>(32 * c + 16) = ((d >> 8) & 0x00FF00FFu) | 0x43004300u;
}
// Four exact float values (one per lane each) as the same pairs, bf16 by truncation (exact here).
SYCL_ESIMD_FUNCTION inline void put_vals(es::simd<uint32_t, 128>& vb, int c, f32x16 v0, f32x16 v1,
                                         f32x16 v2, f32x16 v3) {
    u32x16 b0 = v0.template bit_cast_view<uint32_t>().read(), b1 = v1.template bit_cast_view<uint32_t>().read();
    u32x16 b2 = v2.template bit_cast_view<uint32_t>().read(), b3 = v3.template bit_cast_view<uint32_t>().read();
    vb.template select<16, 1>(32 * c) = (b0 >> 16) | (b2 & 0xFFFF0000u);
    vb.template select<16, 1>(32 * c + 16) = (b1 >> 16) | (b3 & 0xFFFF0000u);
}
// The IQ4 value of the four nibbles at bit offset sh of each byte of d, as four float vectors.
SYCL_ESIMD_FUNCTION inline void iq4_vals(es::simd<uint32_t, 128>& vb, int c, u32x16 d, int sh) {
    u32x16 n = (d >> sh) & 0x0F0F0F0Fu;
    put_vals(vb, c, nq_iq4_value(es::convert<float>(n & 0xFFu)),
             nq_iq4_value(es::convert<float>((n >> 8) & 0xFFu)),
             nq_iq4_value(es::convert<float>((n >> 16) & 0xFFu)),
             nq_iq4_value(es::convert<float>(n >> 24)));
}
SYCL_ESIMD_FUNCTION inline f32x16 half_lo(u32x16 w) {
    es::simd<uint16_t, 16> h = w & 0xFFFFu;
    es::simd<sycl::half, 16> hh = h.template bit_cast_view<sycl::half>().read();
    return f32x16(hh);
}
// dword i of a lane's 2-byte misaligned run: the high half of gathered dword i and the low half
// of dword i+1, moved as 16-bit elements.  (a >> 16) | (b << 16) is a funnel shift to LLVM, which
// the ESIMD back end took as an outside function: IGC "VLD: The list of SPIR-V modules contains
// more than one module with an entry point!" (2026-10-29).
SYCL_ESIMD_FUNCTION inline u32x16 realign(u32x16 a, u32x16 b) {
    es::simd<uint16_t, 32> r;
    auto av = a.template bit_cast_view<uint16_t>();
    auto bv = b.template bit_cast_view<uint16_t>();
    r.template select<16, 2>(0) = av.template select<16, 2>(1);
    r.template select<16, 2>(1) = bv.template select<16, 2>(0);
    return r.template bit_cast_view<uint32_t>().read();
}
// ND dwords from each lane's address, dword v of every lane at [16 v, 16 v + 16): gathers of at
// most 8 dwords an address.
// The cache-hint properties force the LSC lowering for one dword an address too (the legacy
// gather it takes otherwise did not compile next to the others: "VLD: ... more than one module
// with an entry point").
template <int ND>
SYCL_ESIMD_FUNCTION inline es::simd<uint32_t, 16 * ND> gath(const uint32_t* p, u32x16 off) {
    constexpr auto H = sycl::ext::oneapi::experimental::properties{
        es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
    if constexpr (ND <= 4 || ND == 8) {
        return es::gather<uint32_t, 16 * ND, ND>(p, off, H);
    } else {
        es::simd<uint32_t, 16 * ND> r;
        #pragma unroll
        for (int i = 0; i < ND / 8; ++i)
            r.template select<128, 1>(128 * i) = es::gather<uint32_t, 128, 8>(p, off + uint32_t(32 * i), H);
        if constexpr (ND % 8 != 0)
            r.template select<16 * (ND % 8), 1>(128 * (ND / 8)) = gath<ND % 8>(p, off + uint32_t(32 * (ND / 8)));
        return r;
    }
}
// Dword i of a section read from its aligned start (ODD = false) or from 2 bytes before it.
template <bool ODD, int NV>
SYCL_ESIMD_FUNCTION inline u32x16 sec(es::simd<uint32_t, NV>& v, int i) {
    if constexpr (ODD) return realign(v.template select<16, 1>(16 * i), v.template select<16, 1>(16 * (i + 1)));
    else return v.template select<16, 1>(16 * i);
}

template <int RBN>
SYCL_ESIMD_FUNCTION inline void dpas2(es::simd<float, 128> (&tmp)[RBN], es::simd<float, 128> (&tsx)[RBN],
                                      es::simd<uint32_t, 128>& vb, const es::simd<sycl_bf16, 128> (&a)[RBN],
                                      const es::simd<sycl_bf16, 256>& ones, bool sums) {
    #pragma unroll
    for (int rb = 0; rb < RBN; ++rb) {
        tmp[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
            tmp[rb], vb.template bit_cast_view<sycl_bf16>().read(), a[rb]);
        if (sums)
            tsx[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(tsx[rb], ones, a[rb]);
    }
}
// acc += sc * tmp - cs * tsx (per lane = weight row), then clear tmp / tsx.
template <int RBN>
SYCL_ESIMD_FUNCTION inline void fold(es::simd<float, RBN * 128>& acc, es::simd<float, 128> (&tmp)[RBN],
                                     es::simd<float, 128> (&tsx)[RBN], f32x16 sc, f32x16 cs, bool sums) {
    #pragma unroll
    for (int rb = 0; rb < RBN; ++rb) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            f32x16 v = tmp[rb].template select<16, 1>(16 * r) * sc;
            if (sums) v -= tsx[rb].template select<16, 1>(16 * r) * cs;
            acc.template select<16, 1>(rb * 128 + 16 * r) += v;
        }
        tmp[rb] = 0.0f;
        tsx[rb] = 0.0f;
    }
}
// One Q6_K block (210 bytes) of 16 rows.  Block 2j of a row starts 4-byte aligned (rows hold an
// even number of blocks), block 2j+1 two bytes later: ODD reads every section from 2 bytes before
// it and realigns.  Per half (128 values): ql 64 B at 64 h, qh 32 B at 128 + 32 h, 8 int8 scales
// at 192 + 8 h; d (fp16) at 208.  value = d * sc[idx / 16] * (q - 32), q = ql nibble | qh 2 bits.
template <bool ODD, int RBN>
SYCL_ESIMD_FUNCTION inline void q6k_block(const uint32_t* pay, u32x16 roff, uint32_t u, int k,
                                          const sycl_bf16* X, unsigned XW, unsigned XH,
                                          es::simd<float, RBN * 128>& acc, es::simd<float, 128> (&tmp)[RBN],
                                          es::simd<float, 128> (&tsx)[RBN], es::simd<uint32_t, 128>& vb,
                                          es::simd<sycl_bf16, 128> (&a)[RBN],
                                          const es::simd<sycl_bf16, 256>& ones) {
    constexpr int E = ODD ? 1 : 0;                                   // the extra dword to realign
    const u32x16 blk = roff + (u / 2) * 420u + (ODD ? 208u : 0u);  // 4-byte aligned
    u32x16 dwd = gath<1>(pay, blk + 208u);
    const f32x16 d = half_lo(ODD ? u32x16(dwd >> 16) : dwd);
    #pragma unroll
    for (int hb = 0; hb < 2; ++hb) {
        es::simd<uint32_t, 16 * (16 + E)> ql = gath<16 + E>(pay, blk + uint32_t(64 * hb));
        es::simd<uint32_t, 16 * (8 + E)> qh = gath<8 + E>(pay, blk + uint32_t(128 + 32 * hb));
        es::simd<uint32_t, 16 * (2 + E)> scw = gath<2 + E>(pay, blk + uint32_t(192 + 8 * hb));
        const u32x16 s03 = sec<ODD>(scw, 0), s47 = sec<ODD>(scw, 1);   // 8 int8 scales
        #pragma unroll
        for (int L = 0; L < 4; ++L) {
            #pragma unroll
            for (int s = 0; s < 2; ++s) {                            // 16 values a scale = a DPAS
                const int si = 2 * L + s;
                u32x16 sw = (si < 4 ? s03 : s47) << (24 - 8 * (si & 3));
                es::simd<int32_t, 16> sv = sw.template bit_cast_view<int32_t>().read();
                const f32x16 ds = d * es::convert<float>(es::simd<int32_t, 16>(sv >> 24));
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const u32x16 lo = (sec<ODD>(ql, (L % 2) * 8 + 4 * s + c) >> (4 * (L / 2))) & 0x0F0F0F0Fu;
                    const u32x16 hi = (sec<ODD>(qh, 4 * s + c) >> (2 * L)) & 0x03030303u;
                    put_codes(vb, c, lo | (hi << 4));
                }
                load_a<RBN>(X, XW, XH, k + 128 * hb + 32 * L + 16 * s, a);
                dpas2<RBN>(tmp, tsx, vb, a, ones, true);
                fold<RBN>(acc, tmp, tsx, ds, ds * 160.0f, true);
            }
        }
    }
}
} // namespace nsm

// Keep the dense capture free of expert-table pointers and dimensions. With the
// engine's code-split=off build, constexpr ternaries still captured their unused
// operands on the host while AOT removed them from the device argument list.
// A typed layout gives each kernel only the state it actually consumes.
struct NativeSmallmDenseLayout {
    int M;
    SYCL_ESIMD_FUNCTION int expert(int) const { return 0; }
    SYCL_ESIMD_FUNCTION bool active(int) const { return true; }
    SYCL_ESIMD_FUNCTION int rows(int) const { return M; }
    SYCL_ESIMD_FUNCTION int tile(int g, int tw, int tpt) const { return g*tpt+tw; }
    SYCL_ESIMD_FUNCTION const sycl_bf16* input(const sycl_bf16* p, int) const { return p; }
    SYCL_ESIMD_FUNCTION float* output(float* p, int) const { return p; }
    SYCL_ESIMD_FUNCTION const uint32_t* weights(const uint32_t* p, int) const { return p; }
};
struct NativeSmallmGroupedLayout {
    int groups, N, K;
    uint32_t rbytes;
    const int32_t *table, *off, *cnt;
    SYCL_ESIMD_FUNCTION int expert(int g) const { return table[g/groups]; }
    SYCL_ESIMD_FUNCTION bool active(int e) const { return e>=0 && cnt[e]!=0; }
    SYCL_ESIMD_FUNCTION int rows(int e) const { return cnt[e]; }
    SYCL_ESIMD_FUNCTION int tile(int g, int tw, int tpt) const { return (g%groups)*tpt+tw; }
    SYCL_ESIMD_FUNCTION const sycl_bf16* input(const sycl_bf16* p, int e) const {
        return p+size_t(off[e])*K;
    }
    SYCL_ESIMD_FUNCTION float* output(float* p, int e) const { return p+size_t(off[e])*N; }
    SYCL_ESIMD_FUNCTION const uint32_t* weights(const uint32_t* p, int e) const {
        return p+size_t(e)*N*rbytes/4;
    }
};

template <Fmt F, int RBN, bool GROUPED = false>
sycl::event native_smallm_impl(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                               float* Y, int M, int KS, int kc, int TPT,
                               const std::vector<sycl::event>& deps,
                               int Ne = 0, const int32_t* tile_e = nullptr,
                               const int32_t* off = nullptr, const int32_t* cnt = nullptr,
                               int T = 1) {
    constexpr int CT = 16;
    const int N = GROUPED ? Ne : w.N, K = w.K;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    const uint32_t* pay_base = reinterpret_cast<const uint32_t*>(w.payload);
    const uint32_t rbytes = uint32_t(w.row_bytes);
    const unsigned XW = unsigned(K) * 2 - 1;
    const unsigned YW = unsigned(N) * 4 - 1;
    const auto layout = [&] {
        if constexpr (GROUPED) return NativeSmallmGroupedLayout{groups,N,K,rbytes,tile_e,off,cnt};
        else return NativeSmallmDenseLayout{M};
    }();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * groups * TPT * KS, size_t(TPT) * KS),
                       [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            using nsm::u32x16;
            using nsm::f32x16;
            es::slm_init<32 * RBN * 128 * 4>();
            const int lid = int(it.get_local_id(0));
            const int tw = lid / KS, ks = lid % KS;
            const int g = int(it.get_group(0));
            const int e = layout.expert(g);
            // Uniform across the work-group, before any split-K barrier.
            if (!layout.active(e)) return;
            const int rows = layout.rows(e);
            const unsigned XH = unsigned(rows) - 1;
            const sycl_bf16* Xe = layout.input(X,e);
            float* Ye = layout.output(Y,e);
            const uint32_t* pay = layout.weights(pay_base,e);
            const int tile = layout.tile(g,tw,TPT);
            const bool live = tile < tiles;
            const int n0 = (live ? tile : 0) * CT;
            const int kb = ks * kc;
            const int ke = kb + kc < K ? kb + kc : K;
            const u32x16 roff = (u32x16(0, 1) + uint32_t(n0)) * rbytes;   // each lane's row
            const es::simd<sycl_bf16, 256> ones = sycl_bf16(1.0f);
            es::simd<float, RBN * 128> acc = 0.0f;
            es::simd<float, 128> tmp[RBN], tsx[RBN];
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb) { tmp[rb] = 0.0f; tsx[rb] = 0.0f; }
            es::simd<uint32_t, 128> vb;
            es::simd<sycl_bf16, 128> a[RBN];
            if (live) {
                for (int k = kb; k < ke; k += 256) {
                    const uint32_t u = uint32_t(k / 256);              // 256-value unit of every row
                    if constexpr (F == Fmt::Q4_K || F == Fmt::Q5_K) {
                        constexpr uint32_t BS = F == Fmt::Q4_K ? 144 : 176, QS = F == Fmt::Q4_K ? 16 : 48;
                        const u32x16 base = roff + u * BS;
                        es::simd<uint32_t, 64> hdr = nsm::gath<4>(pay, base);
                        es::simd<uint32_t, 512> qs = nsm::gath<32>(pay, base + QS);
                        es::simd<uint32_t, 128> qh;
                        if constexpr (F == Fmt::Q5_K) qh = nsm::gath<8>(pay, base + 16u);
                        const u32x16 h0 = hdr.template select<16, 1>(0), h1 = hdr.template select<16, 1>(16);
                        const u32x16 h2 = hdr.template select<16, 1>(32), h3 = hdr.template select<16, 1>(48);
                        const f32x16 d = nsm::half_lo(h0), dm = nsm::half_lo(h0 >> 16);
                        #pragma unroll
                        for (int g = 0; g < 8; ++g) {
                            u32x16 sc, mn;
                            const int sh = 8 * (g & 3);
                            if (g < 4) { sc = (h1 >> sh) & 63u; mn = (h2 >> sh) & 63u; }
                            else {
                                sc = ((h3 >> sh) & 15u) | ((h1 >> (sh + 2)) & 48u);
                                mn = ((h3 >> (sh + 4)) & 15u) | ((h2 >> (sh + 2)) & 48u);
                            }
                            const f32x16 ds = d * es::convert<float>(sc), mins = dm * es::convert<float>(mn);
                            #pragma unroll
                            for (int hf = 0; hf < 2; ++hf) {           // 16 K a DPAS
                                #pragma unroll
                                for (int c = 0; c < 4; ++c) {
                                    const int j = 4 * hf + c;          // dword of the group's 32 bytes
                                    u32x16 dw = (qs.template select<16, 1>(16 * ((g / 2) * 8 + j)) >>
                                                 (4 * (g % 2))) & 0x0F0F0F0Fu;
                                    if constexpr (F == Fmt::Q5_K)
                                        dw |= ((qh.template select<16, 1>(16 * j) >> g) & 0x01010101u) << 4;
                                    nsm::put_codes(vb, c, dw);
                                }
                                nsm::load_a<RBN>(Xe, XW, XH, k + 32 * g + 16 * hf, a);
                                nsm::dpas2<RBN>(tmp, tsx, vb, a, ones, true);
                            }
                            nsm::fold<RBN>(acc, tmp, tsx, ds, ds * 128.0f + mins, true);
                        }
                    } else if constexpr (F == Fmt::IQ4_XS) {
                        const u32x16 base = roff + u * 136u;
                        es::simd<uint32_t, 32> hdr = nsm::gath<2>(pay, base);
                        es::simd<uint32_t, 512> qs = nsm::gath<32>(pay, base + 8u);
                        const u32x16 h0 = hdr.template select<16, 1>(0), sl = hdr.template select<16, 1>(16);
                        const f32x16 d = nsm::half_lo(h0);
                        const u32x16 shi = h0 >> 16;
                        #pragma unroll
                        for (int g = 0; g < 8; ++g) {
                            const u32x16 sc = ((sl >> (4 * g)) & 15u) | (((shi >> (2 * g)) & 3u) << 4);
                            const f32x16 ds = d * (es::convert<float>(sc) - 32.0f);
                            #pragma unroll
                            for (int hf = 0; hf < 2; ++hf) {           // low nibbles = values 0..15
                                #pragma unroll
                                for (int c = 0; c < 4; ++c)
                                    nsm::iq4_vals(vb, c, qs.template select<16, 1>(16 * (4 * g + c)), 4 * hf);
                                nsm::load_a<RBN>(Xe, XW, XH, k + 32 * g + 16 * hf, a);
                                nsm::dpas2<RBN>(tmp, tsx, vb, a, ones, false);
                            }
                            nsm::fold<RBN>(acc, tmp, tsx, ds, ds, false);
                        }
                    } else if constexpr (F == Fmt::IQ4_NL || F == Fmt::Q8_0) {
                        // 8 blocks of 32 a unit; two blocks share a 4-byte aligned window: 36 B
                        // (IQ4_NL, 9 dwords) or 68 B (Q8_0, 17 dwords).  The even block's quants
                        // start 2 bytes into the window, the odd block's are aligned.
                        constexpr uint32_t BS = F == Fmt::Q8_0 ? 34 : 18, WD = BS / 2;   // window dwords
                        constexpr int QD = F == Fmt::Q8_0 ? 8 : 4;                     // quant dwords a block
                        #pragma unroll
                        for (int pr = 0; pr < 4; ++pr) {
                            const u32x16 base = roff + u * (8 * BS) + uint32_t(pr) * (2 * BS);
                            es::simd<uint32_t, 16 * WD> win = nsm::gath<WD>(pay, base);
                            #pragma unroll
                            for (int bl = 0; bl < 2; ++bl) {
                                const u32x16 dwd = win.template select<16, 1>(16 * (bl ? WD / 2 : 0));
                                const f32x16 d = nsm::half_lo(bl ? u32x16(dwd >> 16) : dwd);
                                u32x16 qd[QD];
                                #pragma unroll
                                for (int i = 0; i < QD; ++i)
                                    qd[i] = bl ? u32x16(win.template select<16, 1>(16 * (WD / 2 + 1 + i)))
                                               : nsm::realign(win.template select<16, 1>(16 * i),
                                                              win.template select<16, 1>(16 * (i + 1)));
                                const int k0 = k + 64 * pr + 32 * bl;
                                if constexpr (F == Fmt::IQ4_NL) {
                                    #pragma unroll
                                    for (int hf = 0; hf < 2; ++hf) {   // low nibbles = values 0..15
                                        #pragma unroll
                                        for (int c = 0; c < 4; ++c) nsm::iq4_vals(vb, c, qd[c], 4 * hf);
                                        nsm::load_a<RBN>(Xe, XW, XH, k0 + 16 * hf, a);
                                        nsm::dpas2<RBN>(tmp, tsx, vb, a, ones, false);
                                    }
                                    nsm::fold<RBN>(acc, tmp, tsx, d, d, false);
                                } else {
                                    // int8 q = 16 h + l: low nibbles l (128 + l) on tmp, high
                                    // nibbles h + 8 (128 + h + 8) on tsx's twin; sum q x =
                                    // 16 Th + Tl - 2304 sum x.
                                    es::simd<float, 128> th[RBN];
                                    #pragma unroll
                                    for (int rb = 0; rb < RBN; ++rb) th[rb] = 0.0f;
                                    #pragma unroll
                                    for (int hf = 0; hf < 2; ++hf) {
                                        #pragma unroll
                                        for (int c = 0; c < 4; ++c) nsm::put_codes(vb, c, qd[4 * hf + c] & 0x0F0F0F0Fu);
                                        nsm::load_a<RBN>(Xe, XW, XH, k0 + 16 * hf, a);
                                        nsm::dpas2<RBN>(tmp, tsx, vb, a, ones, true);
                                        #pragma unroll
                                        for (int c = 0; c < 4; ++c)
                                            nsm::put_codes(vb, c, ((qd[4 * hf + c] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
                                        #pragma unroll
                                        for (int rb = 0; rb < RBN; ++rb)
                                            th[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                                th[rb], vb.template bit_cast_view<sycl_bf16>().read(), a[rb]);
                                    }
                                    #pragma unroll
                                    for (int rb = 0; rb < RBN; ++rb) tmp[rb] += th[rb] * 16.0f;
                                    nsm::fold<RBN>(acc, tmp, tsx, d, d * 2304.0f, true);
                                }
                            }
                        }
                    } else {
                        static_assert(F == Fmt::Q6_K);
                        if (u & 1u) nsm::q6k_block<true, RBN>(pay, roff, u, k, Xe, XW, XH, acc, tmp, tsx, vb, a, ones);
                        else nsm::q6k_block<false, RBN>(pay, roff, u, k, Xe, XW, XH, acc, tmp, tsx, vb, a, ones);
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
                es::store_2d<float, 16, 8>(Ye, YW, XH, YW, n0, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>(rb * 128)));
        });
    });
}

// The layout the kernel reads: whole 256-value units per row (Q6_K: an even number of blocks, so
// block 2j starts 4-byte aligned), packed rows, 4-byte aligned units, 16 rows a tile, byte offsets
// that fit 32 bits.  M <= 8: one 8-row DPAS block (RBN 1, gemv_decode.cpp's 128 registers);
// 9..16: two (RBN 2, libgrimoire_gemm.so's 256).
inline bool native_smallm_layout_ok(const QuantWeight& w, int M) {
    const bool fmt = w.fmt == Fmt::Q4_K || w.fmt == Fmt::Q5_K || w.fmt == Fmt::Q6_K ||
                     w.fmt == Fmt::Q8_0 || w.fmt == Fmt::IQ4_XS || w.fmt == Fmt::IQ4_NL;
    return fmt && w.payload && !w.scales && !w.zeros && M >= 1 && M <= 16 && w.N % 16 == 0 &&
           w.K % 256 == 0 && (w.fmt != Fmt::Q6_K || (w.K / 256) % 2 == 0) &&
           w.row_bytes == bytes_per_row(w.fmt, w.K) && w.row_bytes % 4 == 0 &&
           uint64_t(w.N) * uint64_t(w.row_bytes) < (uint64_t(1) << 32) &&
           reinterpret_cast<uintptr_t>(w.payload) % 64 == 0;
}

// Rows are already packed by expert. A table entry names one expert whose
// count is <= M; negative entries are padding. Ne is the expert's output
// width, not the concatenated E*Ne width in w.N.
inline bool native_moe_smallm_layout_ok(const QuantWeight& w, int Ne, int M) {
    return native_smallm_layout_ok(w, M) && Ne > 0 && Ne % 16 == 0 &&
           w.N % Ne == 0;
}

template <Fmt F, int RBN>
sycl::event native_moe_smallm_launch_t(sycl::queue& q, const QuantWeight& w, int Ne,
    const sycl_bf16* X, float* Y, int M, const int32_t* tile_e,
    const int32_t* off, const int32_t* cnt, int T,
    const std::vector<sycl::event>& deps) {
    const int tiles = Ne / 16, K = w.K;
    int KS = 1;
    while (KS < 32 && int64_t(T) * tiles * KS < 4096 && K / (KS * 2) >= 256)
        KS *= 2;
    int kc = ((K + KS - 1) / KS + 255) / 256 * 256;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= 32 && TPT * 2 <= tiles) TPT *= 2;
    return native_smallm_impl<F, RBN, true>(q, w, X, Y, M, KS, kc, TPT, deps,
                                            Ne, tile_e, off, cnt, T);
}

template <int RBN>
sycl::event native_moe_smallm_launch(sycl::queue& q, const QuantWeight& w, int Ne,
    const sycl_bf16* X, float* Y, int M, const int32_t* tile_e,
    const int32_t* off, const int32_t* cnt, int T,
    const std::vector<sycl::event>& deps) {
    if (!native_moe_smallm_layout_ok(w, Ne, M) || !X || !Y ||
        reinterpret_cast<uintptr_t>(X) % 64 || reinterpret_cast<uintptr_t>(Y) % 64 ||
        !tile_e || !off || !cnt || T <= 0 || M > RBN * 8)
        throw std::runtime_error("native_moe_smallm_launch: invalid layout or rows");
#define B70_MOE_NATIVE_SMALLM(F) case Fmt::F: \
    return native_moe_smallm_launch_t<Fmt::F, RBN>(q,w,Ne,X,Y,M,tile_e,off,cnt,T,deps)
    switch (w.fmt) {
    B70_MOE_NATIVE_SMALLM(Q4_K);
    B70_MOE_NATIVE_SMALLM(Q5_K);
    B70_MOE_NATIVE_SMALLM(Q6_K);
    B70_MOE_NATIVE_SMALLM(Q8_0);
    B70_MOE_NATIVE_SMALLM(IQ4_XS);
    B70_MOE_NATIVE_SMALLM(IQ4_NL);
    default: throw std::runtime_error("native_moe_smallm_launch: unsupported format");
    }
#undef B70_MOE_NATIVE_SMALLM
}

// launch_int4_smallm's split-K plan with 256-K steps.
template <Fmt F, int RBN>
sycl::event native_smallm_launch_t(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                                   float* Y, int M, const std::vector<sycl::event>& deps) {
    const int K = w.K, tiles = w.N / 16;
    int KS = tiles >= 1024 || K >= 12288 ? 4 : std::min(32, std::max(1, K / 256));
    int kc = (K + KS - 1) / KS;
    kc = (kc + 255) / 256 * 256;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= 32) TPT *= 2;
    return native_smallm_impl<F, RBN>(q, w, X, Y, M, KS, kc, TPT, deps);
}

// RBN 1 for M <= 8, 2 for M <= 16 (the caller's translation unit decides which it compiles).
template <int RBN>
sycl::event native_smallm_launch(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                                 float* Y, int M, const std::vector<sycl::event>& deps) {
    switch (w.fmt) {
        case Fmt::Q4_K:   return native_smallm_launch_t<Fmt::Q4_K, RBN>(q, w, X, Y, M, deps);
        case Fmt::Q5_K:   return native_smallm_launch_t<Fmt::Q5_K, RBN>(q, w, X, Y, M, deps);
        case Fmt::Q6_K:   return native_smallm_launch_t<Fmt::Q6_K, RBN>(q, w, X, Y, M, deps);
        case Fmt::Q8_0:   return native_smallm_launch_t<Fmt::Q8_0, RBN>(q, w, X, Y, M, deps);
        case Fmt::IQ4_XS: return native_smallm_launch_t<Fmt::IQ4_XS, RBN>(q, w, X, Y, M, deps);
        case Fmt::IQ4_NL: return native_smallm_launch_t<Fmt::IQ4_NL, RBN>(q, w, X, Y, M, deps);
        default: break;
    }
    throw std::runtime_error("native_smallm_launch: not a native GGUF format");
}

} // namespace b70
