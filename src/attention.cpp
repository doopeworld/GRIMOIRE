// =====================================================================
//  attention.cpp  --  FlashDecoding for the token-generation phase
//
//  The recurrence here is the one validated bit-for-bit against a
//  materialized softmax in tests/test_attention.cpp. Two properties are
//  load-bearing and easy to lose when hand-porting:
//
//   1. When the running maximum moves, BOTH the denominator and the
//      output accumulator must be rescaled by exp(m_old - m_new).
//      Rescaling only the denominator (as the source blueprint does)
//      produces output that is wrong by 150-700%, not marginally wrong.
//
//   2. The running (m, l) statistics must be sub-group UNIFORM. If each
//      lane keeps a private maximum, its partial sums are normalized
//      against a different constant, and the final reduce_over_group
//      adds quantities that are not commensurable.
//
//  Layout note: k_cache is stored D-major, [head][head_dim][seq_cap].
//  That makes lane L's read of K[s0+L][d] contiguous across the
//  sub-group, so a 16-lane score step is one 64-byte transaction instead
//  of 16 scattered ones. v_cache stays D-minor because the accumulator
//  is partitioned over d, which already gives lane-contiguous reads.
//  Appending a token writes K strided and V contiguous -- a few hundred
//  bytes per step, irrelevant next to streaming the whole cache.
// =====================================================================
#include "kernels.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <limits>

namespace b70 {

// Split-K width for single-token decode. launch_flash_decode and
// launch_flash_merge MUST derive this identically -- the merge walks
// [head][split] partials and a mismatch reads the wrong lanes.
//
// GRAPH_SPLITS(8) never scaled with context depth. Measured 2026-09-04 on
// Qwen3.8-27B at 4778 tokens: 3650 us per full-attention layer, 54.8 of 82 ms
// per token, an achieved 4.6 GB/s against a 602 GB/s roofline -- exactly the
// latency-bound regime this kernel's own comment warns about. Scaling splits
// with seq_len took the layer to 1197 us and the token to 43.9 ms.
//
// 128 keys per split is the measured optimum. 64 was tried and is WORSE
// (TG 22.1 vs 28.7): more partials means more merge rounding, which costs
// speculative draft acceptance.
static int keys_per_split() {
    // keys per split; tunable so the decode/merge balance can be swept without
    // a rebuild. More splits = more parallelism in flash_decode but a more
    // expensive flash_merge (it walks [head][split] partials serially).
    static const int KPS = [] {
        const char* e = std::getenv("GRIMOIRE_ATTN_KEYS_PER_SPLIT");
        // DEFAULT 32, not 128. Measured 2026-09-05 on Qwen3.8-27B at 2623
        // tokens, W4A8, coherence PASSED -- keys/split against per-layer cost:
        //     128 -> flash_decode 697 us, merge  6, token 38.4 ms, tg 19.4
        //      64 -> flash_decode 357 us, merge 10, token 33.0 ms, tg 21.8
        //      32 -> flash_decode 186 us, merge 19, token 30.5 ms, tg 23.3  <-
        //      16 -> flash_decode 314 us, merge 29, token 32.7 ms, tg 21.4
        // 128 keys/split under-parallelises badly (21 splits x 24 heads = 504
        // sub-groups on a 256-EU card). 16 over-splits: 128 splits leaves only
        // ~20 keys each and the merge grows. The optimum is a split COUNT near
        // 80-96, so this is keys-per-split only as a proxy -- at much deeper
        // context MAX_SPLITS(128) caps it, which degrades gracefully toward
        // wider splits. Sweep with the env var if the context is far from 2.6k.
        const int v = e ? std::atoi(e) : 32;
        return v > 0 ? v : 32;
    }();
    return KPS;
}

// The split count single-token decode uses for seq_len keys: ~kps keys per
// split, at least min_splits, at most cap.  The host sizes launches with it
// and, under graph capture, every kernel re-derives it on the device from
// the live length (AttnParams::capture), so the two cannot disagree.
static inline int splits_for(int seq_len, int min_splits, int kps, int cap) {
    int want = (seq_len + kps - 1) / kps;
    if (want < min_splits) want = min_splits;
    if (want > cap) want = cap;
    return want > 0 ? want : 1;
}

// Launch geometry.  Direct submission knows the length and sizes for it.  A
// recorded launch is replayed at every later length, so it is sized for the
// whole cache and the kernels use the live count (the rest return at once).
static inline int decode_splits(const AttnParams& p) {
    return splits_for(p.capture ? p.seq_cap : p.seq_len, p.splits, keys_per_split(), MAX_SPLITS);
}

template <int MAXD>
static sycl::event launch_flash_decode_impl(sycl::queue& q, const AttnParams& p,
                                const std::vector<sycl::event>& deps) {
    const int HD  = p.head_dim;
    const int DPL = HD / SG_SIZE; (void)DPL;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const AttnParams pp = p;

        // SPLIT-K OVER THE SEQUENCE.
        //
        // The obvious mapping -- one sub-group per head -- launches only
        // num_heads sub-groups. With 32 heads on a 256-EU B70 that leaves
        // the machine ~96% idle, and the kernel becomes latency-bound:
        // time scales with seq_len while achieved bandwidth stays pinned
        // at a few GB/s no matter how long the context gets.
        //
        // Instead each head is split into SPLITS chunks of the sequence.
        // Every chunk runs its own independent online softmax and writes
        // a partial (acc, m, l) triple; a second pass merges them with
        // the same rescale rule. This is standard FlashDecoding split-K,
        // and it is what turns the kernel from latency-bound into
        // bandwidth-bound.
        const int splits = decode_splits(p);
        const int kps = keys_per_split();
        h.parallel_for(
            sycl::nd_range<1>(size_t(pp.num_heads) * size_t(splits) * SG_SIZE,
                              size_t(SG_SIZE)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  gid  = int(it.get_group(0));
                const int  head = gid / splits;
                const int  part = gid % splits;
                if (head >= pp.num_heads) return;

                // This chunk's slice. seq_len comes from device memory
                // when the kernel is running inside a recorded graph.
                const int seq   = pp.d_seq_len ? pp.d_seq_len[0] : pp.seq_len;
                if ((pp.gate_le > 0 && seq > pp.gate_le) || (pp.gate_gt > 0 && seq <= pp.gate_gt))
                    return;                          // the other gated kernel owns this length
                // Recorded launch: the split count direct submission would
                // use at THIS length; the merge reads only that many.
                const int nsp   = pp.capture ? splits_for(seq, pp.splits, kps, splits) : splits;
                if (part >= nsp) return;
                // Sliding attention: everything before the window is
                // masked to -inf, which contributes nothing, so drop it
                // from the scan instead of scoring and discarding it.
                const int lo    = pp.window_left > 0
                                ? sycl::max(0, seq - pp.window_left) : 0;
                const int span  = seq - lo;
                const int per   = (span + nsp - 1) / nsp;
                const int s_beg = lo + part * per;
                const int s_end = sycl::min(s_beg + per, seq);

                // Grouped-query attention: several query heads share one
                // KV head.
                const int kvh = head / (pp.num_heads / pp.num_kv_heads);

                const float* qh = pp.q + int64_t(head) * pp.head_dim;
                const uint8_t* kh = pp.k_cache + int64_t(kvh) * pp.head_dim * pp.seq_cap;
                const uint8_t* vh = pp.v_cache + int64_t(kvh) * pp.seq_cap * pp.head_dim;

                // Every lane scores a different key, so every lane needs
                // the WHOLE q vector -- it cannot be partitioned like the
                // output accumulator. head_dim floats is 512 B, resident
                // in L1 across the entire scan.
                // MAXD bounds head_dim/SG_SIZE. This model uses
                // head_dim 256, so 16 accumulator slots per lane -- an
                // 8-slot array silently computes only the first half of
                // every head and leaves dims 128..255 holding whatever
                // the previous token left there.
                static_assert(MAXD == MAX_DPL || MAXD == MAX_DPL_WIDE,
                              "kernels.hpp owns the accumulator widths");
                const int dpl = HD / SG_SIZE;
                float m   = -std::numeric_limits<float>::infinity();
                float l   = 0.0f;
                float acc[MAXD];
                #pragma unroll
                for (int d = 0; d < MAXD; ++d) acc[d] = 0.0f;

                for (int s0 = s_beg; s0 < s_end; s0 += SG_SIZE) {
                    const int s = s0 + lane;

                    // ---- score one key per lane ----------------------
                    // K is D-major: kh[d * seq_cap + s], so the 16 lanes
                    // read 16 consecutive floats for each d.
                    float score = -std::numeric_limits<float>::infinity();
                    bool live = s < s_end;
                    if (live && pp.qbits) {        // QSA: selected blocks + tail
                        const int tail = seq / pp.qrat * pp.qrat;
                        const int b = s / pp.qrat;
                        live = s >= tail || ((pp.qbits[b >> 5] >> (b & 31)) & 1u);
                    }
                    if (live) {
                        // 16 K bytes in flight per lane before their FMAs:
                        // the plain loop issued one dependent load per dim
                        // (256 L2 round trips per 16-key block -- the whole
                        // 62.8 us of a short-context layer, MEASURED
                        // 2026-10-01).  Same FMA order over d, so the score
                        // is bit-identical.
                        float dot = 0.0f;
                        const uint8_t* kp = kh + s;
                        int d0 = 0;
                        for (; d0 + 16 <= HD; d0 += 16) {
                            uint8_t kb[16];
                            #pragma unroll
                            for (int u = 0; u < 16; ++u) kb[u] = kp[int64_t(d0 + u) * pp.seq_cap];
                            #pragma unroll
                            for (int u = 0; u < 16; ++u)
                                dot = sycl::fma(qh[d0 + u], e4m3_to_f32(kb[u]), dot);
                        }
                        for (; d0 < HD; ++d0)
                            dot = sycl::fma(qh[d0], e4m3_to_f32(kp[int64_t(d0) * pp.seq_cap]), dot);
                        score = dot * pp.softmax_scale;
                    }

                    // ---- sub-group-uniform online softmax ------------
                    const float mblk = sycl::reduce_over_group(
                        sg, score, sycl::maximum<float>());
                    const float mnew = sycl::fmax(m, mblk);

                    // exp(-inf - -inf) is NaN; the first block must not
                    // scale an accumulator that is still exactly zero.
                    const float corr = sycl::isinf(m) ? 0.0f : sycl::exp(m - mnew);

                    const float pj = sycl::isinf(score) ? 0.0f : sycl::exp(score - mnew);
                    const float psum = sycl::reduce_over_group(sg, pj, sycl::plus<float>());

                    l = sycl::fma(l, corr, psum);
                    #pragma unroll
                    for (int d = 0; d < MAXD; ++d)
                        if (d < dpl) acc[d] *= corr;

                    // ---- accumulate V --------------------------------
                    // Every lane visits all SG_SIZE keys of the block but
                    // only the output dims it owns. pj travels by
                    // broadcast, so V is read exactly once.
                    // Four keys of V bytes in flight before their FMAs; the
                    // per-dim FMA chain still runs over j in order.
                    const int nk = sycl::min(SG_SIZE, s_end - s0);
                    for (int j0 = 0; j0 < nk; j0 += 4) {
                        uint8_t vb[4][MAXD];
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int j = sycl::min(j0 + u, nk - 1);
                            const uint8_t* vrow = vh + int64_t(s0 + j) * pp.head_dim;
                            #pragma unroll
                            for (int d = 0; d < MAXD; ++d)
                                vb[u][d] = d < dpl ? vrow[lane + d * SG_SIZE] : uint8_t(0);
                        }
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int j = j0 + u;
                            const float pb = sycl::group_broadcast(sg, pj, j);
                            if (j < nk) {
                                #pragma unroll
                                for (int d = 0; d < MAXD; ++d)
                                    if (d < dpl)
                                        acc[d] = sycl::fma(pb, e4m3_to_f32(vb[u][d]), acc[d]);
                            }
                        }
                    }
                    m = mnew;
                }

                // Emit the UNNORMALIZED partial plus its (m, l) so the
                // merge pass can combine chunks exactly. Dividing by l
                // here would discard the information needed to rescale.
                const int64_t pidx = int64_t(head) * splits + part;
                float* po = pp.partials + pidx * pp.head_dim;
                #pragma unroll
                for (int d = 0; d < MAXD; ++d)
                    if (d < dpl) po[lane + d * SG_SIZE] = acc[d];
                if (lane == 0) {
                    pp.part_m[pidx] = (s_beg >= s_end) ? -std::numeric_limits<float>::infinity() : m;
                    pp.part_l[pidx] = (s_beg >= s_end) ? 0.0f : l;
                }
            });
    });
}
// ---------------------------------------------------------------------
// GQA-shared split-K decode.
//
// launch_flash_decode_impl gives every QUERY head its own sub-group, so a KV
// head's bytes are fetched once per query head that shares it (4x on K2, 6x
// on Qwen3.8-27B), and it fetches them one byte per lane: a lane scores one
// key, so each of the head_dim K loads moves 16 useful bytes, and V costs
// head_dim/16 byte gathers per key.  MEASURED on K2 at a 5768-token context:
// 421 us per layer (+32 merge) for 11.8 MB of FP8 KV, two-thirds of the
// whole decode token.  Load-message count, not bandwidth, is the limit --
// batching those loads (same message count) measured no change at all.
//
// Here one sub-group owns (kv head, split, chunk of GC query heads):
//   K: a lane owns KPL=4 consecutive keys, so ONE 32-bit load per dim gives
//      4 keys, and every loaded K value is used by all GC heads;
//   V: a lane owns HD/16 CONTIGUOUS dims, one 8- or 16-byte load per key,
//      again shared by the GC heads.
// Dot products keep the old kernel's operands and FMA order over d, so the
// scores are bit-identical; the online softmax runs over 64-key blocks
// instead of 16, which changes rounding only.  The split partition and the
// [head][split][head_dim] partials are unchanged, so launch_flash_merge is
// untouched.  GRIMOIRE_FLASH_DECODE_OLD=1 restores the per-head kernel.
// ---------------------------------------------------------------------
template <int HD, int GC>
static sycl::event flash_decode_gqa(sycl::queue& q, const AttnParams& p,
                                    const std::vector<sycl::event>& deps) {
    constexpr int DPL = HD / SG_SIZE;          // V dims per lane
    constexpr int KPL = 4;                     // keys per lane per block
    constexpr int KB  = SG_SIZE * KPL;         // 64 keys per block
    static_assert(DPL % 8 == 0, "V is read 8 bytes at a time");
    const int G      = p.num_heads / p.num_kv_heads;
    const int chunks = G / GC;
    const int splits = decode_splits(p);
    const int kps = keys_per_split();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const AttnParams pp = p;
        h.parallel_for(
            sycl::nd_range<1>(size_t(pp.num_kv_heads) * chunks * splits * SG_SIZE,
                              size_t(SG_SIZE)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                constexpr float NINF = -std::numeric_limits<float>::infinity();
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  gid  = int(it.get_group(0));
                const int  part = gid % splits;
                const int  rest = gid / splits;
                const int  kvh  = rest / chunks;
                const int  head0 = kvh * G + (rest % chunks) * GC;

                // The same slice launch_flash_decode_impl gives this split.
                const int seq   = pp.d_seq_len ? pp.d_seq_len[0] : pp.seq_len;
                if ((pp.gate_le > 0 && seq > pp.gate_le) || (pp.gate_gt > 0 && seq <= pp.gate_gt))
                    return;                          // the other gated kernel owns this length
                const int nsp   = pp.capture ? splits_for(seq, pp.splits, kps, splits) : splits;
                if (part >= nsp) return;             // recorded launch: live count only
                const int lo    = pp.window_left > 0
                                ? sycl::max(0, seq - pp.window_left) : 0;
                const int span  = seq - lo;
                const int per   = (span + nsp - 1) / nsp;
                const int s_beg = lo + part * per;
                const int s_end = sycl::min(s_beg + per, seq);

                const uint8_t* kh = pp.k_cache + int64_t(kvh) * HD * pp.seq_cap;
                const uint8_t* vh = pp.v_cache + int64_t(kvh) * pp.seq_cap * HD;
                const float*   qb = pp.q + int64_t(head0) * HD;

                float m[GC], l[GC], acc[GC][DPL];
                #pragma unroll
                for (int g = 0; g < GC; ++g) {
                    m[g] = NINF; l[g] = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < DPL; ++i) acc[g][i] = 0.0f;
                }

                // Blocks start 4-aligned so each lane's 4 keys are one aligned
                // 32-bit word; keys before s_beg are masked.  seq_cap % 4 == 0
                // (checked at dispatch) keeps the last word inside its row.
                for (int b0 = s_beg & ~(KPL - 1); b0 < s_end; b0 += KB) {
                    const int sk = b0 + lane * KPL;

                    // ---- scores: 4 keys x GC heads per 32-bit K load ------
                    float sc[GC][KPL];
                    #pragma unroll
                    for (int g = 0; g < GC; ++g)
                        #pragma unroll
                        for (int k = 0; k < KPL; ++k) sc[g][k] = 0.0f;
                    if (sk < s_end) {
                        const uint8_t* kc = kh + sk;
                        #pragma unroll 8
                        for (int d = 0; d < HD; ++d) {
                            const uint32_t kw = *reinterpret_cast<const uint32_t*>(
                                kc + int64_t(d) * pp.seq_cap);
                            float kf[KPL];
                            #pragma unroll
                            for (int k = 0; k < KPL; ++k)
                                kf[k] = e4m3_to_f32(uint8_t(kw >> (8 * k)));
                            #pragma unroll
                            for (int g = 0; g < GC; ++g) {
                                const float qd = qb[g * HD + d];
                                #pragma unroll
                                for (int k = 0; k < KPL; ++k)
                                    sc[g][k] = sycl::fma(qd, kf[k], sc[g][k]);
                            }
                        }
                    }
                    bool live[KPL];
                    #pragma unroll
                    for (int k = 0; k < KPL; ++k) {
                        const int s = sk + k;
                        live[k] = s >= s_beg && s < s_end;
                        if (live[k] && pp.qbits) {        // QSA: selected blocks + tail
                            const int tail = seq / pp.qrat * pp.qrat;
                            const int b = s / pp.qrat;
                            live[k] = s >= tail || ((pp.qbits[b >> 5] >> (b & 31)) & 1u);
                        }
                    }

                    // ---- sub-group-uniform online softmax, per head ------
                    float pr[GC][KPL];
                    #pragma unroll
                    for (int g = 0; g < GC; ++g) {
                        float sv[KPL], mx = NINF;
                        #pragma unroll
                        for (int k = 0; k < KPL; ++k) {
                            sv[k] = live[k] ? sc[g][k] * pp.softmax_scale : NINF;
                            mx = sycl::fmax(mx, sv[k]);
                        }
                        const float mblk = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
                        const float mnew = sycl::fmax(m[g], mblk);
                        // exp(-inf - -inf) is NaN; a still-empty accumulator
                        // must not be scaled by it.
                        const float corr = sycl::isinf(m[g]) ? 0.0f : sycl::exp(m[g] - mnew);
                        float ps = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < KPL; ++k) {
                            pr[g][k] = sycl::isinf(sv[k]) ? 0.0f : sycl::exp(sv[k] - mnew);
                            ps += pr[g][k];
                        }
                        l[g] = sycl::fma(l[g], corr,
                                         sycl::reduce_over_group(sg, ps, sycl::plus<float>()));
                        #pragma unroll
                        for (int i = 0; i < DPL; ++i) acc[g][i] *= corr;
                        m[g] = mnew;
                    }

                    // ---- V: one 8/16-byte load per key, keys ascending ---
                    const int kn = sycl::min(KB, s_end - b0);
                    for (int o = 0; o * KPL < kn; ++o) {
                        #pragma unroll
                        for (int kk = 0; kk < KPL; ++kk) {
                            const int j = o * KPL + kk;
                            if (j >= kn) break;
                            float pj[GC];
                            bool any = false;
                            #pragma unroll
                            for (int g = 0; g < GC; ++g) {
                                pj[g] = sycl::group_broadcast(sg, pr[g][kk], o);
                                any |= pj[g] != 0.0f;
                            }
                            if (!any) continue;          // masked key: no V read
                            const uint8_t* vr = vh + int64_t(b0 + j) * HD + lane * DPL;
                            float vf[DPL];
                            #pragma unroll
                            for (int w = 0; w < DPL / 8; ++w) {
                                const uint64_t vw = *reinterpret_cast<const uint64_t*>(vr + 8 * w);
                                #pragma unroll
                                for (int b = 0; b < 8; ++b)
                                    vf[8 * w + b] = e4m3_to_f32(uint8_t(vw >> (8 * b)));
                            }
                            #pragma unroll
                            for (int g = 0; g < GC; ++g)
                                #pragma unroll
                                for (int i = 0; i < DPL; ++i)
                                    acc[g][i] = sycl::fma(pj[g], vf[i], acc[g][i]);
                        }
                    }
                }

                // UNNORMALIZED partials + (m, l), exactly what the merge reads.
                #pragma unroll
                for (int g = 0; g < GC; ++g) {
                    const int64_t pidx = int64_t(head0 + g) * splits + part;
                    float* po = pp.partials + pidx * HD + lane * DPL;
                    #pragma unroll
                    for (int i = 0; i < DPL; ++i) po[i] = acc[g][i];
                    if (lane == 0) {
                        pp.part_m[pidx] = (s_beg >= s_end) ? NINF : m[g];
                        pp.part_l[pidx] = (s_beg >= s_end) ? 0.0f : l[g];
                    }
                }
            });
    });
}


// ---------------------------------------------------------------------
// ESIMD split-K decode attention, head_dim 256, FP8 E4M3 KV.
//
// The SIMT kernels give a lane one key and walk head_dim serially: at short
// context one 16-key block costs ~40 us per layer (one thread per EU, a
// dependent chain -- MEASURED 2026-10-01: splits 8/16/32/64 all 38-42 us),
// and the GQA kernel is 110 us there.  Here one thread owns GH query heads of
// one KV head and one split: 16 keys per step, scores vectorized over the
// keys (s[h] += q[h][d] * K[d][s0..s0+15]), V accumulated as whole 256-dim
// rows, every K/V byte decoded once for GH heads.  E4M3 decodes exactly
// through fp16: the byte's 7 magnitude bits at fp16 bits 7..13 plus the sign
// at bit 15 ARE the value times 2^-8 (subnormals included); the 2^8 folds
// into q (scores) and into the written partial (V).  Partials keep the
// [head][split][head_dim] + (m, l) layout, so launch_flash_merge is shared.
// GRIMOIRE_FLASH_ESIMD=0 = the SIMT kernels.
// ---------------------------------------------------------------------
namespace {
namespace es = sycl::ext::intel::esimd;

template <int N>
SYCL_ESIMD_FUNCTION inline es::simd<float, N> e4m3x2m8(es::simd<uint8_t, N> b) {
    es::simd<uint16_t, N> u = b;
    es::simd<uint16_t, N> hb = ((u & 0x7F) << 7) | ((u & 0x80) << 8);
    es::simd<sycl::half, N> hv = hb.template bit_cast_view<sycl::half>();
    return es::simd<float, N>(hv);
}

template <int GH>
sycl::event flash_decode_esimd256(sycl::queue& q, const AttnParams& p,
                                  const std::vector<sycl::event>& deps) {
    constexpr int HD = 256, NK = 16;
    const int splits = decode_splits(p);
    const int kps = keys_per_split();
    const int G = p.num_heads / p.num_kv_heads;
    const int hgroups = G / GH;
    const AttnParams pp = p;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(pp.num_kv_heads) * hgroups * splits, 1),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int gid = int(it.get_group(0));
            const int part = gid % splits;
            const int rest = gid / splits;
            const int hg = rest % hgroups;
            const int kvh = rest / hgroups;
            const int h0 = kvh * G + hg * GH;
            const int seq = pp.d_seq_len ? pp.d_seq_len[0] : pp.seq_len;
            // Recorded launch: use the split count direct submission would
            // launch at this length (splits_for, written out -- ESIMD code
            // gets only what it can see inline); the merge reads that many.
            int nsp = splits;
            if (pp.capture) {
                nsp = (seq + kps - 1) / kps;
                if (nsp < pp.splits) nsp = pp.splits;
                if (nsp > splits) nsp = splits;
                if (nsp < 1) nsp = 1;
            }
            if (part >= nsp) return;
            // 16-aligned split boundaries keep every tile load aligned
            int per = (seq + nsp - 1) / nsp;
            per = (per + NK - 1) / NK * NK;
            const int s_beg = part * per;
            const int s_end = s_beg + per < seq ? s_beg + per : seq;
            const float NEG = -std::numeric_limits<float>::infinity();
            if (s_beg >= s_end) {
                for (int g = 0; g < GH; ++g) {
                    const int64_t pidx = int64_t(h0 + g) * splits + part;
                    pp.part_m[pidx] = NEG;
                    pp.part_l[pidx] = 0.0f;
                }
                return;
            }
            const uint8_t* kh = pp.k_cache + int64_t(kvh) * HD * pp.seq_cap;
            const uint8_t* vh = pp.v_cache + int64_t(kvh) * pp.seq_cap * HD;
            const uint32_t* kh32 = reinterpret_cast<const uint32_t*>(kh);
            const unsigned SW = unsigned(pp.seq_cap) - 1;     // K surface: HD rows of seq_cap bytes
            const float qs = pp.softmax_scale * 256.0f;      // undo K's 2^-8
            es::simd<float, HD> qv[GH];
            es::simd<float, HD> acc[GH];
            float m[GH], l[GH];
            #pragma unroll
            for (int g = 0; g < GH; ++g) {
                qv[g] = es::block_load<float, HD>(pp.q + int64_t(h0 + g) * HD) * qs;
                acc[g] = 0.0f;
                m[g] = NEG;
                l[g] = 0.0f;
            }
            es::simd<int, NK> lane(0, 1);
            for (int s0 = s_beg; s0 < s_end; s0 += NK) {
                const int nk = s_end - s0 < NK ? s_end - s0 : NK;
                // every load of the block in flight at once: one memory
                // latency per 16 keys instead of one per dependent step
                es::simd<uint32_t, 128> kt[HD / 32];          // [32 dims][16 keys] each
                #pragma unroll
                for (int c = 0; c < HD / 32; ++c)
                    kt[c] = es::load_2d<uint32_t, 4, 32>(kh32, SW, HD - 1, SW, s0 / 4, 32 * c);
                es::simd<uint8_t, HD> vb[NK];
                #pragma unroll
                for (int j = 0; j < NK; ++j)
                    vb[j] = es::block_load<uint8_t, HD>(vh + int64_t(s0 + j) * HD);
                es::simd<float, NK> sc[GH];
                #pragma unroll
                for (int g = 0; g < GH; ++g) sc[g] = 0.0f;
                #pragma unroll
                for (int c = 0; c < HD / 32; ++c) {
                    es::simd<uint8_t, 512> kb = kt[c].template bit_cast_view<uint8_t>();
                    es::simd<float, 512> kf = e4m3x2m8<512>(kb);
                    es::simd<float, 32> qc[GH];
                    #pragma unroll
                    for (int g = 0; g < GH; ++g) qc[g] = qv[g].template select<32, 1>(32 * c);
                    #pragma unroll
                    for (int r = 0; r < 32; ++r) {
                        #pragma unroll
                        for (int g = 0; g < GH; ++g)
                            sc[g] += float(qc[g][r]) * kf.template select<16, 1>(16 * r);
                    }
                }
                es::simd_mask<NK> valid = lane < nk;
                es::simd<float, NK> pr[GH];
                #pragma unroll
                for (int g = 0; g < GH; ++g) {
                    es::simd<float, NK> sv = es::merge(sc[g], es::simd<float, NK>(NEG), valid);
                    const float mb = es::hmax<float>(sv);
                    const float mn = m[g] > mb ? m[g] : mb;
                    const float corr = (m[g] == NEG) ? 0.0f : sycl::exp(m[g] - mn);
                    es::simd<float, NK> e = es::exp(sv - mn);
                    pr[g] = es::merge(e, es::simd<float, NK>(0.0f), valid);
                    l[g] = l[g] * corr + es::reduce<float>(pr[g], std::plus<>());
                    acc[g] *= corr;
                    m[g] = mn;
                }
                #pragma unroll
                for (int j = 0; j < NK; ++j) {
                    es::simd<float, HD> vf = e4m3x2m8<HD>(vb[j]);
                    #pragma unroll
                    for (int g = 0; g < GH; ++g) acc[g] += float(pr[g][j]) * vf;   // pr = 0 past nk
                }
            }
            #pragma unroll
            for (int g = 0; g < GH; ++g) {
                const int64_t pidx = int64_t(h0 + g) * splits + part;
                es::block_store<float, HD>(pp.partials + pidx * HD, acc[g] * 256.0f);   // undo V's 2^-8
                pp.part_m[pidx] = m[g];
                pp.part_l[pidx] = l[g];
            }
        });
    });
}
bool flash_esimd_on() {
    static const bool v = [] { const char* e = std::getenv("GRIMOIRE_FLASH_ESIMD");
        return !(e && *e == '0'); }();
    return v;
}
} // namespace

sycl::event launch_flash_decode(sycl::queue& q, const AttnParams& p,
                                const std::vector<sycl::event>& deps) {
    if (flash_esimd_on() && p.head_dim == 256 && p.num_kv_heads > 0 &&
        p.num_heads % p.num_kv_heads == 0 && (p.num_heads / p.num_kv_heads) % 2 == 0 &&
        p.window_left <= 0 && !p.qbits && (p.seq_cap % 16) == 0)
        return flash_decode_esimd256<2>(q, p, deps);
    static const bool old = std::getenv("GRIMOIRE_FLASH_DECODE_OLD") != nullptr;
    // GC (heads per sub-group) is the largest divisor of the GQA group that
    // keeps GC * head_dim/16 accumulators at 48 or fewer per lane.
    // Inside a recorded decode graph the context length is only known on the
    // device, and neither kernel wins everywhere.  MEASURED 2026-09-30, Ornith
    // hd256 (G=8), per full-attention layer: ~50 keys -> per-head kernel 61 us,
    // GQA kernel 110 us; 5.7K keys -> 420 vs 128 us.  So both go into the graph
    // with complementary length gates and exactly one runs.
    // GRIMOIRE_FLASH_DECODE_SPLIT_T=<keys> moves the crossover (0 = GQA only).
    static const int split_t = [] {
        const char* e = std::getenv("GRIMOIRE_FLASH_DECODE_SPLIT_T");
        return e ? std::atoi(e) : 768; }();
    if (!old && split_t > 0 && p.d_seq_len && p.gate_le == 0 && p.gate_gt == 0 &&
        p.num_kv_heads > 0 && p.num_heads % p.num_kv_heads == 0 && (p.seq_cap % 4) == 0 &&
        (p.head_dim == 128 || p.head_dim == 256)) {
        AttnParams a = p; a.gate_le = split_t;
        sycl::event e0 = p.head_dim > MAX_HEAD_DIM
            ? launch_flash_decode_impl<MAX_DPL_WIDE>(q, a, deps)
            : launch_flash_decode_impl<MAX_DPL>(q, a, deps);
        AttnParams b = p; b.gate_gt = split_t;
        return launch_flash_decode(q, b, {e0});
    }
    if (!old && p.num_kv_heads > 0 && p.num_heads % p.num_kv_heads == 0 &&
        (p.seq_cap % 4) == 0) {
        const int G = p.num_heads / p.num_kv_heads;
        if (p.head_dim == 128) {
            if (G % 4 == 0) return flash_decode_gqa<128, 4>(q, p, deps);
            if (G % 3 == 0) return flash_decode_gqa<128, 3>(q, p, deps);
            if (G % 2 == 0) return flash_decode_gqa<128, 2>(q, p, deps);
            return flash_decode_gqa<128, 1>(q, p, deps);
        }
        if (p.head_dim == 256) {
            if (G % 3 == 0) return flash_decode_gqa<256, 3>(q, p, deps);
            if (G % 2 == 0) return flash_decode_gqa<256, 2>(q, p, deps);
            return flash_decode_gqa<256, 1>(q, p, deps);
        }
    }
    return p.head_dim > MAX_HEAD_DIM
         ? launch_flash_decode_impl<MAX_DPL_WIDE>(q, p, deps)
         : launch_flash_decode_impl<MAX_DPL>(q, p, deps);
}

// ---------------------------------------------------------------------
// Merge pass. Combines the per-chunk partials using the same rescale
// rule as the online softmax itself:
//     m   = max_i m_i
//     l   = sum_i l_i * exp(m_i - m)
//     acc = sum_i acc_i * exp(m_i - m)
// One sub-group per (head, dim-tile) -- see the geometry note below.
// ---------------------------------------------------------------------
sycl::event launch_flash_merge(sycl::queue& q, const AttnParams& p,
                               const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const AttnParams pp = p;
        const int msplits = decode_splits(p);    // the partials' stride
        const int kps = keys_per_split();
        // One sub-group per (head, dim-tile) instead of one per head. The old
        // geometry launched num_heads = 24 sub-groups for the whole merge --
        // 1.2% of a 256-EU card -- so the merge cost grew linearly with the
        // split count and ate the whole flash_decode win: at 64 splits it was
        // 187 us to reduce 1.6 MB, an achieved 8 GB/s. Tiling the output dim
        // gives num_heads * (head_dim / SG_SIZE) = 384 sub-groups on this model
        // and makes each lane own exactly ONE output dim, so the partial reads
        // are lane-contiguous 64-byte transactions.
        //
        // exp(m_i - m) is also hoisted: the old loop recomputed it once per
        // (split, dim) pair, i.e. head_dim/SG_SIZE = 16 times more often than
        // needed. The arithmetic and the summation order are unchanged, so the
        // result stays bit-identical -- this must not perturb the draft
        // acceptance that split width is known to affect.
        const int dtiles = (pp.head_dim + SG_SIZE - 1) / SG_SIZE;
        h.parallel_for(
            sycl::nd_range<1>(size_t(pp.num_heads) * size_t(dtiles) * SG_SIZE,
                              size_t(SG_SIZE)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int lane = int(it.get_sub_group().get_local_id()[0]);
                const int gid  = int(it.get_group(0));
                const int head = gid / dtiles;
                const int tile = gid % dtiles;
                if (head >= pp.num_heads) return;
                // How many of the msplits slots hold this step's splits: all
                // of them for direct submission, the live count for a
                // recorded launch (the decode kernels used the same one).
                const int seq = pp.d_seq_len ? pp.d_seq_len[0] : pp.seq_len;
                const int splits = pp.capture ? splits_for(seq, pp.splits, kps, msplits) : msplits;
                const int d = tile * SG_SIZE + lane;
                if (d >= pp.head_dim) return;

                const float* pm = pp.part_m + int64_t(head) * msplits;
                const float* pl = pp.part_l + int64_t(head) * msplits;

                float m = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < splits; ++i)
                    m = sycl::fmax(m, pm[i]);

                // l and the output accumulator share one exp() per split.
                float l = 0.0f, a = 0.0f;
                for (int i = 0; i < splits; ++i) {
                    const float mi = pm[i];
                    if (sycl::isinf(mi)) continue;
                    const float e = sycl::exp(mi - m);
                    l += pl[i] * e;
                    a += pp.partials[(int64_t(head) * msplits + i) * pp.head_dim + d] * e;
                }

                const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
                pp.out[int64_t(head) * pp.head_dim + d] = a * inv;
            });
    });
}

// ---------------------------------------------------------------------
// Small-batch FlashDecoding for speculative verification.  A work-group owns
// one (query head, sequence split), and one subgroup owns each query row.  The
// subgroups share K/V staged in SLM, so M queries stream the cache once rather
// than M times.  The split-K workspace is row-major:
//   [query][head][split][head_dim], [query][head][split].
// ---------------------------------------------------------------------
template <int MAXD>
static sycl::event launch_flash_decode_batched_impl(
    sycl::queue& q, const float* qv, const uint8_t* k_cache,
    const uint8_t* v_cache, float* out, int tokens, int base_seq_len,
    int num_heads, int num_kv_heads, int head_dim, int seq_cap,
    float softmax_scale, float* partials, float* part_m, float* part_l,
    int splits, const std::vector<sycl::event>& deps) {
    constexpr int KT = SG_SIZE;
    constexpr int MAX_WG = 512;
    const int q_per_kv = num_heads / num_kv_heads;
    // One subgroup handles one (row, query-head-within-KV-head).  A 16-row
    // DFlash verifier block made the old single group 2048 work-items on
    // Qwen3.5 (q_per_kv=4), beyond B70's 1024 limit. Tile rows while keeping
    // the proven 512-work-item ceiling used by the block-attention kernel.
    const int rows_per_wg = std::max(1, std::min(
        tokens, MAX_WG / (q_per_kv * SG_SIZE)));
    const int row_tiles = (tokens + rows_per_wg - 1) / rows_per_wg;
    const int wg = rows_per_wg * q_per_kv * SG_SIZE;
    const int max_seq = base_seq_len + tokens - 1;

    sycl::event scan = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> ks(size_t(head_dim) * KT, h);
        sycl::local_accessor<float, 1> vs(size_t(KT) * head_dim, h);
        h.parallel_for(
            sycl::nd_range<1>(
                size_t(num_kv_heads) * splits * row_tiles * wg, wg),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int qslot = int(sg.get_group_id()[0]);
                const int gid = int(it.get_group(0));
                const int tile = gid % row_tiles;
                const int block = gid / row_tiles;
                const int row = tile * rows_per_wg + qslot / q_per_kv;
                const bool row_valid = row < tokens;
                const int safe_row = row_valid ? row : 0;
                const int q_in_kv = qslot % q_per_kv;
                const int lid = int(it.get_local_id(0));
                const int kvh = block / splits;
                const int part = block % splits;
                const int head = kvh * q_per_kv + q_in_kv;
                const int row_seq = base_seq_len + row;
                const int per = (max_seq + splits - 1) / splits;
                const int s_beg = part * per;
                const int s_end = sycl::min(s_beg + per, max_seq);
                const float* qh = qv +
                    (int64_t(safe_row) * num_heads + head) * head_dim;
                const uint8_t* kh = k_cache +
                    int64_t(kvh) * head_dim * seq_cap;
                const uint8_t* vh = v_cache +
                    int64_t(kvh) * seq_cap * head_dim;
                float* ksl = ks.template get_multi_ptr<
                    sycl::access::decorated::no>().get();
                float* vsl = vs.template get_multi_ptr<
                    sycl::access::decorated::no>().get();

                static_assert(MAXD == MAX_DPL || MAXD == MAX_DPL_WIDE,
                              "kernels.hpp owns the accumulator widths");
                const int dpl = head_dim / SG_SIZE;
                float m = -std::numeric_limits<float>::infinity();
                float l = 0.0f;
                float acc[MAXD];
                #pragma unroll
                for (int d = 0; d < MAXD; ++d) acc[d] = 0.0f;

                for (int s0 = s_beg; s0 < s_end; s0 += KT) {
                    for (int x = lid; x < head_dim * KT; x += wg) {
                        const int d = x / KT;
                        const int k = x % KT;
                        const int s = s0 + k;
                        ksl[x] = s < s_end ? e4m3_to_f32(
                            kh[int64_t(d) * seq_cap + s]) : 0.0f;
                        vsl[int64_t(k) * head_dim + d] = s < s_end
                            ? e4m3_to_f32(vh[int64_t(s) * head_dim + d]) : 0.0f;
                    }
                    sycl::group_barrier(it.get_group());

                    const int s = s0 + lane;
                    float score = -std::numeric_limits<float>::infinity();
                    if (row_valid && s < s_end && s < row_seq) {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                            dot = sycl::fma(qh[d],
                                ksl[int64_t(d) * KT + lane], dot);
                        score = dot * softmax_scale;
                    }
                    const float mb = sycl::reduce_over_group(
                        sg, score, sycl::maximum<float>());
                    const float mn = sycl::fmax(m, mb);
                    const float corr = sycl::isinf(m)
                        ? 0.0f : sycl::exp(m - mn);
                    const float p = sycl::isinf(score)
                        ? 0.0f : sycl::exp(score - mn);
                    l = sycl::fma(l, corr, sycl::reduce_over_group(
                        sg, p, sycl::plus<float>()));
                    #pragma unroll
                    for (int d = 0; d < MAXD; ++d)
                        if (d < dpl) acc[d] *= corr;
                    for (int j = 0; j < KT && s0 + j < s_end; ++j) {
                        const float pb = sycl::group_broadcast(sg, p, j);
                        #pragma unroll
                        for (int d = 0; d < MAXD; ++d)
                            if (d < dpl)
                                acc[d] = sycl::fma(pb,
                                    vsl[int64_t(j) * head_dim + lane +
                                        d * SG_SIZE], acc[d]);
                    }
                    m = mn;
                    sycl::group_barrier(it.get_group());
                }

                if (row_valid) {
                    const int64_t pidx =
                        (int64_t(row) * num_heads + head) * splits + part;
                    float* po = partials + pidx * head_dim;
                    #pragma unroll
                    for (int d = 0; d < MAXD; ++d)
                        if (d < dpl) po[lane + d * SG_SIZE] = acc[d];
                    if (lane == 0) {
                        part_m[pidx] = (s_beg >= s_end || s_beg >= row_seq)
                            ? -std::numeric_limits<float>::infinity() : m;
                        part_l[pidx] = (s_beg >= s_end || s_beg >= row_seq)
                            ? 0.0f : l;
                    }
                }
            });
    });

    return q.submit([&](sycl::handler& h) {
        h.depends_on(scan);
        h.parallel_for(
            sycl::nd_range<1>(size_t(tokens) * num_heads * SG_SIZE,
                              size_t(SG_SIZE)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int lane = int(it.get_sub_group().get_local_id()[0]);
                const int qr_head = int(it.get_group(0));
                const int row = qr_head / num_heads;
                const int head = qr_head % num_heads;
                const int64_t base =
                    (int64_t(row) * num_heads + head) * splits;
                float m = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < splits; ++i)
                    m = sycl::fmax(m, part_m[base + i]);
                float l = 0.0f;
                for (int i = 0; i < splits; ++i) {
                    const float mi = part_m[base + i];
                    if (!sycl::isinf(mi))
                        l += part_l[base + i] * sycl::exp(mi - m);
                }
                float* o = out +
                    (int64_t(row) * num_heads + head) * head_dim;
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                for (int d = lane; d < head_dim; d += SG_SIZE) {
                    float a = 0.0f;
                    for (int i = 0; i < splits; ++i) {
                        const int64_t pidx = base + i;
                        const float mi = part_m[pidx];
                        if (!sycl::isinf(mi))
                            a += partials[pidx * head_dim + d] *
                                sycl::exp(mi - m);
                    }
                    o[d] = a * inv;
                }
            });
    });
}
sycl::event launch_flash_decode_batched(
    sycl::queue& q, const float* qv, const uint8_t* k_cache,
    const uint8_t* v_cache, float* out, int tokens, int base_seq_len,
    int num_heads, int num_kv_heads, int head_dim, int seq_cap,
    float softmax_scale, float* partials, float* part_m, float* part_l,
    int splits, const std::vector<sycl::event>& deps) {
    return head_dim > MAX_HEAD_DIM
         ? launch_flash_decode_batched_impl<MAX_DPL_WIDE>(q, qv, k_cache, v_cache, out, tokens, base_seq_len, num_heads,
           num_kv_heads, head_dim, seq_cap, softmax_scale, partials,
           part_m, part_l, splits, deps)
         : launch_flash_decode_batched_impl<MAX_DPL>(q, qv, k_cache, v_cache, out, tokens, base_seq_len, num_heads,
           num_kv_heads, head_dim, seq_cap, softmax_scale, partials,
           part_m, part_l, splits, deps);
}

// ---------------------------------------------------------------------
//  QSA sparse attention (Qwen4-Exp).
//
//  The same online-softmax flash math as launch_flash_decode, with ONE
//  difference: the key loop is driven by a GATHERED INDEX ARRAY rather
//  than a contiguous [lo, seq) range.  That is what QSA is -- the indexer
//  picks which blocks a query may see, and this attends to exactly those.
//
//  A -1 entry is a HOLE and must contribute nothing.  Treating it as key
//  0 is the silent failure here: the row still softmaxes to something
//  plausible, so a short sequence quietly attends to its first token
//  repeatedly instead of to fewer keys.
//
//  Templated on the accumulator width like every other flash kernel in
//  this file (see kernels.hpp), so a head wider than MAX_HEAD_DIM takes
//  the 32-slot instantiation rather than overrunning device stack.
// ---------------------------------------------------------------------
template <int MAXD>
static sycl::event launch_qsa_attention_impl(
    sycl::queue& q, const float* qv, const uint8_t* k_cache,
    const uint8_t* v_cache, const int32_t* idx, float* out, int rows,
    int n_heads, int kv_heads, int head_dim, int seq_cap, int n_idx,
    float softmax_scale, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(rows) * n_heads * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  gid  = int(it.get_group(0));
                const int  row  = gid / n_heads;
                const int  head = gid % n_heads;
                const int  kvh  = head / (n_heads / kv_heads);

                static_assert(MAXD == MAX_DPL || MAXD == MAX_DPL_WIDE,
                              "kernels.hpp owns the accumulator widths");
                const int dpl = head_dim / SG_SIZE;
                const float* qh = qv + (int64_t(row) * n_heads + head) * head_dim;
                const uint8_t* kh = k_cache + int64_t(kvh) * head_dim * seq_cap;
                const uint8_t* vh = v_cache + int64_t(kvh) * seq_cap * head_dim;
                const int32_t* ri = idx + int64_t(row) * n_idx;

                float m = -std::numeric_limits<float>::infinity();
                float l = 0.0f;
                float acc[MAXD];
                #pragma unroll
                for (int d = 0; d < MAXD; ++d) acc[d] = 0.0f;

                for (int i0 = 0; i0 < n_idx; i0 += SG_SIZE) {
                    const int i = i0 + lane;
                    const int s = (i < n_idx) ? ri[i] : -1;
                    float score = -std::numeric_limits<float>::infinity();
                    if (s >= 0 && s < seq_cap) {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                            dot = sycl::fma(qh[d], e4m3_to_f32(
                                kh[int64_t(d) * seq_cap + s]), dot);
                        score = dot * softmax_scale;
                    }
                    const float mblk = sycl::reduce_over_group(
                        sg, score, sycl::maximum<float>());
                    const float mnew = sycl::fmax(m, mblk);
                    const float corr = sycl::isinf(m) ? 0.0f : sycl::exp(m - mnew);
                    const float pj = sycl::isinf(score) ? 0.0f
                                                        : sycl::exp(score - mnew);
                    const float psum = sycl::reduce_over_group(
                        sg, pj, sycl::plus<float>());
                    l = sycl::fma(l, corr, psum);
                    #pragma unroll
                    for (int d = 0; d < MAXD; ++d)
                        if (d < dpl) acc[d] *= corr;
                    for (int j = 0; j < SG_SIZE; ++j) {
                        if (i0 + j >= n_idx) break;
                        const int sj = ri[i0 + j];
                        if (sj < 0 || sj >= seq_cap) continue;   // a hole
                        const float pb = sycl::group_broadcast(sg, pj, j);
                        const uint8_t* vrow = vh + int64_t(sj) * head_dim;
                        #pragma unroll
                        for (int d = 0; d < MAXD; ++d)
                            if (d < dpl)
                                acc[d] = sycl::fma(pb, e4m3_to_f32(
                                    vrow[lane + d * SG_SIZE]), acc[d]);
                    }
                    m = mnew;
                }
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                float* o = out + (int64_t(row) * n_heads + head) * head_dim;
                #pragma unroll
                for (int d = 0; d < MAXD; ++d)
                    if (d < dpl) o[lane + d * SG_SIZE] = acc[d] * inv;
            });
    });
}

sycl::event launch_qsa_attention(
    sycl::queue& q, const float* qv, const uint8_t* k_cache,
    const uint8_t* v_cache, const int32_t* idx, float* out, int rows,
    int n_heads, int kv_heads, int head_dim, int seq_cap, int n_idx,
    float softmax_scale, const std::vector<sycl::event>& deps) {
    return head_dim > MAX_HEAD_DIM
         ? launch_qsa_attention_impl<MAX_DPL_WIDE>(q, qv, k_cache, v_cache,
             idx, out, rows, n_heads, kv_heads, head_dim, seq_cap, n_idx,
             softmax_scale, deps)
         : launch_qsa_attention_impl<MAX_DPL>(q, qv, k_cache, v_cache,
             idx, out, rows, n_heads, kv_heads, head_dim, seq_cap, n_idx,
             softmax_scale, deps);
}

} // namespace b70
