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
#include <limits>

namespace b70 {

sycl::event launch_flash_decode(sycl::queue& q, const AttnParams& p,
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
        const int splits = pp.splits;
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
                const int per   = (seq + splits - 1) / splits;
                const int s_beg = part * per;
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
                // MAX_DPL bounds head_dim/SG_SIZE. This model uses
                // head_dim 256, so 16 accumulator slots per lane -- an
                // 8-slot array silently computes only the first half of
                // every head and leaves dims 128..255 holding whatever
                // the previous token left there.
                constexpr int MAX_DPL = 16;
                const int dpl = HD / SG_SIZE;
                float m   = -std::numeric_limits<float>::infinity();
                float l   = 0.0f;
                float acc[MAX_DPL];
                #pragma unroll
                for (int d = 0; d < MAX_DPL; ++d) acc[d] = 0.0f;

                for (int s0 = s_beg; s0 < s_end; s0 += SG_SIZE) {
                    const int s = s0 + lane;

                    // ---- score one key per lane ----------------------
                    // K is D-major: kh[d * seq_cap + s], so the 16 lanes
                    // read 16 consecutive floats for each d.
                    float score = -std::numeric_limits<float>::infinity();
                    if (s < s_end) {
                        float dot = 0.0f;
                        for (int d = 0; d < HD; ++d)
                            dot = sycl::fma(qh[d], e4m3_to_f32(
                                kh[int64_t(d) * pp.seq_cap + s]), dot);
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
                    for (int d = 0; d < MAX_DPL; ++d)
                        if (d < dpl) acc[d] *= corr;

                    // ---- accumulate V --------------------------------
                    // Every lane visits all SG_SIZE keys of the block but
                    // only the output dims it owns. pj travels by
                    // broadcast, so V is read exactly once.
                    for (int j = 0; j < SG_SIZE; ++j) {
                        if (s0 + j >= s_end) break;
                        const float pb = sycl::group_broadcast(sg, pj, j);
                        const uint8_t* vrow = vh + int64_t(s0 + j) * pp.head_dim;
                        #pragma unroll
                        for (int d = 0; d < MAX_DPL; ++d)
                            if (d < dpl)
                                acc[d] = sycl::fma(pb, e4m3_to_f32(
                                    vrow[lane + d * SG_SIZE]), acc[d]);
                    }
                    m = mnew;
                }

                // Emit the UNNORMALIZED partial plus its (m, l) so the
                // merge pass can combine chunks exactly. Dividing by l
                // here would discard the information needed to rescale.
                const int64_t pidx = int64_t(head) * splits + part;
                float* po = pp.partials + pidx * pp.head_dim;
                #pragma unroll
                for (int d = 0; d < MAX_DPL; ++d)
                    if (d < dpl) po[lane + d * SG_SIZE] = acc[d];
                if (lane == 0) {
                    pp.part_m[pidx] = (s_beg >= s_end) ? -std::numeric_limits<float>::infinity() : m;
                    pp.part_l[pidx] = (s_beg >= s_end) ? 0.0f : l;
                }
            });
    });
}

// ---------------------------------------------------------------------
// Merge pass. Combines the per-chunk partials using the same rescale
// rule as the online softmax itself:
//     m   = max_i m_i
//     l   = sum_i l_i * exp(m_i - m)
//     acc = sum_i acc_i * exp(m_i - m)
// One sub-group per head; trivially cheap next to the scan.
// ---------------------------------------------------------------------
sycl::event launch_flash_merge(sycl::queue& q, const AttnParams& p,
                               const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const AttnParams pp = p;
        h.parallel_for(
            sycl::nd_range<1>(size_t(pp.num_heads) * SG_SIZE, size_t(SG_SIZE)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int lane = int(it.get_sub_group().get_local_id()[0]);
                const int head = int(it.get_group(0));
                if (head >= pp.num_heads) return;
                const int splits = pp.splits;

                float m = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < splits; ++i)
                    m = sycl::fmax(m, pp.part_m[int64_t(head) * splits + i]);

                float l = 0.0f;
                for (int i = 0; i < splits; ++i) {
                    const float mi = pp.part_m[int64_t(head) * splits + i];
                    if (sycl::isinf(mi)) continue;
                    l += pp.part_l[int64_t(head) * splits + i] * sycl::exp(mi - m);
                }

                float* o = pp.out + int64_t(head) * pp.head_dim;
                const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
                for (int d = lane; d < pp.head_dim; d += SG_SIZE) {
                    float a = 0.0f;
                    for (int i = 0; i < splits; ++i) {
                        const int64_t pidx = int64_t(head) * splits + i;
                        const float mi = pp.part_m[pidx];
                        if (sycl::isinf(mi)) continue;
                        a += pp.partials[pidx * pp.head_dim + d] * sycl::exp(mi - m);
                    }
                    o[d] = a * inv;
                }
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
sycl::event launch_flash_decode_batched(
    sycl::queue& q, const float* qv, const uint8_t* k_cache,
    const uint8_t* v_cache, float* out, int tokens, int base_seq_len,
    int num_heads, int num_kv_heads, int head_dim, int seq_cap,
    float softmax_scale, float* partials, float* part_m, float* part_l,
    int splits, const std::vector<sycl::event>& deps) {
    constexpr int KT = SG_SIZE;
    const int q_per_kv = num_heads / num_kv_heads;
    const int wg = tokens * q_per_kv * SG_SIZE;
    const int max_seq = base_seq_len + tokens - 1;

    sycl::event scan = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> ks(size_t(head_dim) * KT, h);
        sycl::local_accessor<float, 1> vs(size_t(KT) * head_dim, h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(num_kv_heads) * splits * wg, wg),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int qslot = int(sg.get_group_id()[0]);
                const int row = qslot / q_per_kv;
                const int q_in_kv = qslot % q_per_kv;
                const int lid = int(it.get_local_id(0));
                const int gid = int(it.get_group(0));
                const int kvh = gid / splits;
                const int part = gid % splits;
                const int head = kvh * q_per_kv + q_in_kv;
                const int row_seq = base_seq_len + row;
                const int per = (max_seq + splits - 1) / splits;
                const int s_beg = part * per;
                const int s_end = sycl::min(s_beg + per, max_seq);
                const float* qh = qv +
                    (int64_t(row) * num_heads + head) * head_dim;
                const uint8_t* kh = k_cache +
                    int64_t(kvh) * head_dim * seq_cap;
                const uint8_t* vh = v_cache +
                    int64_t(kvh) * seq_cap * head_dim;
                float* ksl = ks.template get_multi_ptr<
                    sycl::access::decorated::no>().get();
                float* vsl = vs.template get_multi_ptr<
                    sycl::access::decorated::no>().get();

                constexpr int MAX_DPL = 16;
                const int dpl = head_dim / SG_SIZE;
                float m = -std::numeric_limits<float>::infinity();
                float l = 0.0f;
                float acc[MAX_DPL];
                #pragma unroll
                for (int d = 0; d < MAX_DPL; ++d) acc[d] = 0.0f;

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
                    if (s < s_end && s < row_seq) {
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
                    for (int d = 0; d < MAX_DPL; ++d)
                        if (d < dpl) acc[d] *= corr;
                    for (int j = 0; j < KT && s0 + j < s_end; ++j) {
                        const float pb = sycl::group_broadcast(sg, p, j);
                        #pragma unroll
                        for (int d = 0; d < MAX_DPL; ++d)
                            if (d < dpl)
                                acc[d] = sycl::fma(pb,
                                    vsl[int64_t(j) * head_dim + lane +
                                        d * SG_SIZE], acc[d]);
                    }
                    m = mn;
                    sycl::group_barrier(it.get_group());
                }

                const int64_t pidx =
                    (int64_t(row) * num_heads + head) * splits + part;
                float* po = partials + pidx * head_dim;
                #pragma unroll
                for (int d = 0; d < MAX_DPL; ++d)
                    if (d < dpl) po[lane + d * SG_SIZE] = acc[d];
                if (lane == 0) {
                    part_m[pidx] = (s_beg >= s_end || s_beg >= row_seq)
                        ? -std::numeric_limits<float>::infinity() : m;
                    part_l[pidx] = (s_beg >= s_end || s_beg >= row_seq)
                        ? 0.0f : l;
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

} // namespace b70
