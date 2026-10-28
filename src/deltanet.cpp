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
//  deltanet.cpp  --  Gated DeltaNet decode step
//
//  30 of the 40 layers in Qwen3.5-MoE use this instead of attention.
//
//  DECOMPOSITION
//  -------------
//  The state is [v_dim x k_dim] per value head. One work-item owns one
//  v_dim ROW of that state, and a row's update needs only the shared
//  q, k, its own v[i], and the scalars a and b:
//
//      w      = dot(S[i], k)
//      S[i]  := a*S[i] + b*(v[i] - a*w)*k
//      out[i] = dot(S[i], q)
//
//  No cross-item communication, no barriers beyond staging q and k in
//  SLM. The read of S[i] and the write-back are the same 512 bytes, so
//  the step is purely bandwidth bound: 2 x 60 MiB per token across all
//  30 layers, about 0.24 ms at 500 GB/s, against ~3.1 ms for the MoE
//  weights. Roughly 8% of the decode budget.
//
//  The `a *` inside the correction term is load-bearing. Dropping it
//  leaves output that looks fine and drifts with context: measured 2%
//  error at 8 tokens, 20% at 4096 (tests/test_deltanet.cpp). The
//  row-wise form here is bit-exact against the reference.
// =====================================================================
#include "kernels.hpp"
#include <cmath>

namespace b70 {



namespace {
bool dn_step_fast() {
    static const bool v = []{ const char* e = std::getenv("B70_DN_STEP");
        return !(e && *e && std::atoi(e) == 0); }();
    return v;
}
constexpr int kDnRowsPerWG = 16;    // sub-groups per work-group = rows per WG
} // namespace

// Original mapping: one WORK-ITEM per state row, walking that row with
// `for j in 0..KD`.  Adjacent lanes are then KD*4 = 512 bytes apart, so every
// load instruction touches sixteen different cache lines, and with n_heads
// work-groups the whole kernel is 4096 work-items on a 256-EU card.  Measured
// 37.8 us/layer, 1.81 ms/token, 106 GB/s -- 18% of roofline.  Kept as the
// fallback for shapes the fast path cannot take, and under B70_DN_STEP=0.
static sycl::event deltanet_step_rowwise(sycl::queue& q_, const DeltaNetParams& p,
                                 const std::vector<sycl::event>& deps) {
    const int KD = p.k_dim, VD = p.v_dim;

    return q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetParams pp = p;

        // q and k are re-read by every row of the head, so they belong
        // in SLM. v is not: each row touches exactly one element.
        sycl::local_accessor<float, 1> qs(KD, h);
        sycl::local_accessor<float, 1> ks(KD, h);

        h.parallel_for(
            sycl::nd_range<1>(size_t(pp.n_heads) * size_t(VD), size_t(VD)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int head = int(it.get_group(0));
                const int row  = int(it.get_local_id(0));      // v_dim index
                if (head >= pp.n_heads) return;

                float* qsp = qs.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* ksp = ks.template get_multi_ptr<sycl::access::decorated::no>().get();
                // q/k are shared by pairs of v-heads (repeat_interleave 2)
                const int nk    = pp.n_k_heads ? pp.n_k_heads : pp.n_heads;
                const int khead = (nk == pp.n_heads) ? head
                                : head / (pp.n_heads / nk);

                const float av = pp.a[head];
                const float bv = pp.beta[head];
                const float vi = pp.v[int64_t(head) * VD + row];

                for (int j = row; j < KD; j += VD) {
                    qsp[j] = pp.q[int64_t(khead) * KD + j];
                    ksp[j] = pp.k[int64_t(khead) * KD + j];
                }
                sycl::group_barrier(it.get_group());



                float* S = pp.state + (int64_t(head) * VD + row) * KD;

                // Pass 1: w = S[i] . k
                float w = 0.0f;
                for (int j = 0; j < KD; ++j) w = sycl::fma(S[j], ksp[j], w);

                // Pass 2: fused update and readout. Doing both in one
                // sweep means the row is touched once, which is the
                // whole point when the kernel is bandwidth bound.
                const float corr = bv * (vi - av * w);
                float o = 0.0f;
                for (int j = 0; j < KD; ++j) {
                    const float s = sycl::fma(av, S[j], corr * ksp[j]);
                    S[j] = s;
                    o = sycl::fma(s, qsp[j], o);
                }
                // The reference divides the read by sqrt(k_dim):
                //     output = einsum('bhkv,bhk->bhv', state, q) / sqrt(d_k)
                // Omitting it scales every linear-attention layer's output
                // by 11.3x at k_dim=128. Thirty layers of that is not a
                // subtle degradation -- it saturates the residual stream.
                pp.out[int64_t(head) * VD + row] = o * sycl::rsqrt(float(KD));
            });
    });
}

// ---------------------------------------------------------------------
//  Fast decode step: SUB-GROUP per state row.
//
//  Two changes over deltanet_step_rowwise, both aimed at the same 18%:
//
//   1. Lanes stride the k dimension instead of owning whole rows, so the
//      sixteen lanes of a sub-group read sixteen CONSECUTIVE floats -- one
//      coalesced 64-byte line per load instead of sixteen lines.
//   2. The row is held in registers between the two passes.  The row-wise
//      form reads S once for w = dot(S,k) and again for the update, moving
//      the state three times; here it moves twice, read and write.
//
//  Parallelism goes from n_heads work-groups (4096 work-items) to
//  n_heads*v_dim sub-groups (65536 work-items).
//
//  NOTE: the dot products are now sub-group reductions rather than a
//  sequential sum, so the summation ORDER differs from the row-wise form.
//  The result is not bit-identical -- it is a floating-point reassociation,
//  not a change of algorithm.
// ---------------------------------------------------------------------
template <int PL>          // floats per lane = k_dim / SG_SIZE
static sycl::event dn_step_sg(sycl::queue& q_, const DeltaNetParams& p,
                              const std::vector<sycl::event>& deps) {
    const int KD = p.k_dim, VD = p.v_dim;
    const int wg_threads   = kDnRowsPerWG * SG_SIZE;
    const int wgs_per_head = VD / kDnRowsPerWG;
    const size_t groups    = size_t(p.n_heads) * size_t(wgs_per_head);

    return q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetParams pp = p;
        sycl::local_accessor<float, 1> qs(KD, h);
        sycl::local_accessor<float, 1> ks(KD, h);
        const int wgt = wg_threads, wph = wgs_per_head;

        h.parallel_for(
            sycl::nd_range<1>(groups * size_t(wg_threads), size_t(wg_threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  sgid = int(sg.get_group_id()[0]);
                const int  lid  = int(it.get_local_id(0));
                const int  gid  = int(it.get_group(0));
                const int  head = gid / wph;
                const int  row  = (gid % wph) * kDnRowsPerWG + sgid;

                float* qsp = qs.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* ksp = ks.template get_multi_ptr<sycl::access::decorated::no>().get();
                const int nk    = pp.n_k_heads ? pp.n_k_heads : pp.n_heads;
                const int khead = (nk == pp.n_heads) ? head
                                : head / (pp.n_heads / nk);
                for (int j = lid; j < KD; j += wgt) {
                    qsp[j] = pp.q[int64_t(khead) * KD + j];
                    ksp[j] = pp.k[int64_t(khead) * KD + j];
                }
                sycl::group_barrier(it.get_group());

                const float av = pp.a[head];
                const float bv = pp.beta[head];
                const float vi = pp.v[int64_t(head) * VD + row];
                float* S = pp.state + (int64_t(head) * VD + row) * KD;

                float sv[PL], kv[PL];
                float w = 0.0f;
                #pragma unroll
                for (int u = 0; u < PL; ++u) {
                    const int j = u * SG_SIZE + lane;
                    sv[u] = S[j];
                    kv[u] = ksp[j];
                    w = sycl::fma(sv[u], kv[u], w);
                }
                w = sycl::reduce_over_group(sg, w, sycl::plus<float>());

                const float corr = bv * (vi - av * w);
                float o = 0.0f;
                #pragma unroll
                for (int u = 0; u < PL; ++u) {
                    const int j = u * SG_SIZE + lane;
                    const float sN = sycl::fma(av, sv[u], corr * kv[u]);
                    S[j] = sN;
                    o = sycl::fma(sN, qsp[j], o);
                }
                o = sycl::reduce_over_group(sg, o, sycl::plus<float>());
                if (lane == 0)
                    pp.out[int64_t(head) * VD + row] = o * sycl::rsqrt(float(KD));
            });
    });
}

sycl::event launch_deltanet_step(sycl::queue& q_, const DeltaNetParams& p,
                                 const std::vector<sycl::event>& deps) {
    const int KD = p.k_dim, VD = p.v_dim;
    if (dn_step_fast() && KD % SG_SIZE == 0 && VD % kDnRowsPerWG == 0) {
        switch (KD / SG_SIZE) {
            case 4:  return dn_step_sg<4> (q_, p, deps);
            case 8:  return dn_step_sg<8> (q_, p, deps);
            case 16: return dn_step_sg<16>(q_, p, deps);
            default: break;
        }
    }
    return deltanet_step_rowwise(q_, p, deps);
}

// dn_step_sg over `rows` rows of a batched decode in ONE launch: the work-
// groups of row r are dn_step_sg's work-groups for that row's pointers and
// state slot.  Every line of arithmetic is dn_step_sg's, so each row is
// bit-identical to its own launch_deltanet_step.  One launch per DeltaNet
// layer instead of one per row (8 rows x 30 layers = 240 launches a step).
template <int PL>
static sycl::event dn_step_sg_rows(sycl::queue& q_, const DeltaNetParams& p, int rows,
                                   const RowSlots& rs, int64_t state_stride,
                                   const std::vector<sycl::event>& deps) {
    const int KD = p.k_dim, VD = p.v_dim;
    const int wg_threads   = kDnRowsPerWG * SG_SIZE;
    const int wgs_per_head = VD / kDnRowsPerWG;
    const int per_row      = p.n_heads * wgs_per_head;

    return q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetParams pp = p;
        const RowSlots slots = rs;
        const int64_t sst = state_stride;
        sycl::local_accessor<float, 1> qs(KD, h);
        sycl::local_accessor<float, 1> ks(KD, h);
        const int wgt = wg_threads, wph = wgs_per_head, prow = per_row;

        h.parallel_for(
            sycl::nd_range<1>(size_t(per_row) * size_t(rows) * size_t(wg_threads), size_t(wg_threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  sgid = int(sg.get_group_id()[0]);
                const int  lid  = int(it.get_local_id(0));
                const int  r    = int(it.get_group(0)) / prow;
                const int  gid  = int(it.get_group(0)) % prow;
                const int  head = gid / wph;
                const int  row  = (gid % wph) * kDnRowsPerWG + sgid;

                float* qsp = qs.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* ksp = ks.template get_multi_ptr<sycl::access::decorated::no>().get();
                const int nk    = pp.n_k_heads ? pp.n_k_heads : pp.n_heads;
                const int khead = (nk == pp.n_heads) ? head
                                : head / (pp.n_heads / nk);
                const float* qr = pp.q + int64_t(r) * nk * KD;
                const float* kr = pp.k + int64_t(r) * nk * KD;
                for (int j = lid; j < KD; j += wgt) {
                    qsp[j] = qr[int64_t(khead) * KD + j];
                    ksp[j] = kr[int64_t(khead) * KD + j];
                }
                sycl::group_barrier(it.get_group());

                const float av = pp.a[int64_t(r) * pp.n_heads + head];
                const float bv = pp.beta[int64_t(r) * pp.n_heads + head];
                const float vi = pp.v[(int64_t(r) * pp.n_heads + head) * VD + row];
                float* S = pp.state + int64_t(slots.slot[r]) * sst + (int64_t(head) * VD + row) * KD;

                float sv[PL], kv[PL];
                float w = 0.0f;
                #pragma unroll
                for (int u = 0; u < PL; ++u) {
                    const int j = u * SG_SIZE + lane;
                    sv[u] = S[j];
                    kv[u] = ksp[j];
                    w = sycl::fma(sv[u], kv[u], w);
                }
                w = sycl::reduce_over_group(sg, w, sycl::plus<float>());

                const float corr = bv * (vi - av * w);
                float o = 0.0f;
                #pragma unroll
                for (int u = 0; u < PL; ++u) {
                    const int j = u * SG_SIZE + lane;
                    const float sN = sycl::fma(av, sv[u], corr * kv[u]);
                    S[j] = sN;
                    o = sycl::fma(sN, qsp[j], o);
                }
                o = sycl::reduce_over_group(sg, o, sycl::plus<float>());
                if (lane == 0)
                    pp.out[(int64_t(r) * pp.n_heads + head) * VD + row] = o * sycl::rsqrt(float(KD));
            });
    });
}

sycl::event launch_deltanet_step_rows(sycl::queue& q_, const DeltaNetParams& p, int rows,
                                      const RowSlots& rs, int64_t state_stride,
                                      const std::vector<sycl::event>& deps) {
    const int KD = p.k_dim, VD = p.v_dim;
    if (dn_step_fast() && KD % SG_SIZE == 0 && VD % kDnRowsPerWG == 0 && rows <= kMaxRowSlots) {
        switch (KD / SG_SIZE) {
            case 4:  return dn_step_sg_rows<4> (q_, p, rows, rs, state_stride, deps);
            case 8:  return dn_step_sg_rows<8> (q_, p, rows, rs, state_stride, deps);
            case 16: return dn_step_sg_rows<16>(q_, p, rows, rs, state_stride, deps);
            default: break;
        }
    }
    // any other shape: one launch per row, exactly as before
    const int nk = p.n_k_heads ? p.n_k_heads : p.n_heads;
    sycl::event e;
    for (int r = 0; r < rows; ++r) {
        DeltaNetParams sp = p;
        sp.q = p.q + int64_t(r) * nk * KD;
        sp.k = p.k + int64_t(r) * nk * KD;
        sp.v = p.v + int64_t(r) * p.n_heads * VD;
        sp.a = p.a + int64_t(r) * p.n_heads;
        sp.beta = p.beta + int64_t(r) * p.n_heads;
        sp.out = p.out + int64_t(r) * p.n_heads * VD;
        sp.state = p.state + int64_t(rs.slot[r]) * state_stride;
        e = launch_deltanet_step(q_, sp, r ? std::vector<sycl::event>{} : deps);
    }
    return e;
}

// ---------------------------------------------------------------------
// Speculative verify: the rows of each conversation in ONE launch, state read-only.
//
// MEASURED 2026-10-09 (Qwen3.8-27B GPTQ, MTP K=6, one B70): the verify pass ran
// launch_deltanet_step once per row and copied the 3 MB state after each row as a rollback
// snapshot -- 14 launches and ~84 MB of state traffic per DeltaNet layer, 67 us per layer
// (3.2 ms of a 41 ms verify), and ~670 of the pass's ~1700 kernel launches.  Here each
// sub-group reads its state row once and keeps it in registers across the rows with
// dn_step_sg's exact arithmetic (bit-identical outputs).  Nothing is written back: the
// rows' k / v / a / beta are saved instead, and the commit replays the accepted prefix
// (launch_deltanet_replay) -- one read and one write of the state per conversation.
// ---------------------------------------------------------------------
bool deltanet_verify_ok(const DeltaNetParams& p) {
    const int PL = p.k_dim / SG_SIZE;
    return dn_step_fast() && p.k_dim % SG_SIZE == 0 && p.v_dim % kDnRowsPerWG == 0 &&
           (PL == 4 || PL == 8 || PL == 16);
}

int64_t deltanet_saved_row_floats(const DeltaNetParams& p) {
    const int nk = p.n_k_heads ? p.n_k_heads : p.n_heads;
    return int64_t(nk) * p.k_dim + int64_t(p.n_heads) * p.v_dim + 2 * int64_t(p.n_heads);
}

template <int PL>
static sycl::event dn_verify_sg(sycl::queue& q_, const DeltaNetParams& p, const DnGroups& g,
                                int64_t state_stride, float* saved,
                                const std::vector<sycl::event>& deps) {
    const int KD = p.k_dim, VD = p.v_dim;
    const int wg_threads   = kDnRowsPerWG * SG_SIZE;
    const int wgs_per_head = VD / kDnRowsPerWG;
    const int per_group    = p.n_heads * wgs_per_head;
    const int64_t srow     = deltanet_saved_row_floats(p);

    return q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetParams pp = p;
        const DnGroups gg = g;
        const int64_t sst = state_stride;
        float* const sv_base = saved;
        const int wph = wgs_per_head, pg = per_group;

        h.parallel_for(
            sycl::nd_range<1>(size_t(per_group) * size_t(g.n) * size_t(wg_threads), size_t(wg_threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  sgid = int(sg.get_group_id()[0]);
                const int  grp  = int(it.get_group(0)) / pg;
                const int  gid  = int(it.get_group(0)) % pg;
                const int  head = gid / wph;
                const int  row  = (gid % wph) * kDnRowsPerWG + sgid;
                const int  nk    = pp.n_k_heads ? pp.n_k_heads : pp.n_heads;
                const int  khead = (nk == pp.n_heads) ? head : head / (pp.n_heads / nk);
                const bool k_owner = row == 0 && (nk == pp.n_heads || head % (pp.n_heads / nk) == 0);
                const int64_t voff = int64_t(nk) * KD;
                const int64_t aoff = voff + int64_t(pp.n_heads) * VD;

                const float* S = pp.state + int64_t(gg.slot[grp]) * sst + (int64_t(head) * VD + row) * KD;
                float sv[PL];
                #pragma unroll
                for (int u = 0; u < PL; ++u) sv[u] = S[u * SG_SIZE + lane];

                const int r0 = gg.row0[grp], nr = gg.nrows[grp];
                for (int t = 0; t < nr; ++t) {
                    const int r = r0 + t;
                    const float* kr = pp.k + (int64_t(r) * nk + khead) * KD;
                    const float* qr = pp.q + (int64_t(r) * nk + khead) * KD;
                    float kv[PL];
                    float w = 0.0f;
                    #pragma unroll
                    for (int u = 0; u < PL; ++u) {
                        kv[u] = kr[u * SG_SIZE + lane];
                        w = sycl::fma(sv[u], kv[u], w);
                    }
                    w = sycl::reduce_over_group(sg, w, sycl::plus<float>());

                    const float av = pp.a[int64_t(r) * pp.n_heads + head];
                    const float bv = pp.beta[int64_t(r) * pp.n_heads + head];
                    const float vi = pp.v[(int64_t(r) * pp.n_heads + head) * VD + row];
                    const float corr = bv * (vi - av * w);
                    float o = 0.0f;
                    #pragma unroll
                    for (int u = 0; u < PL; ++u) {
                        const float sN = sycl::fma(av, sv[u], corr * kv[u]);
                        sv[u] = sN;
                        o = sycl::fma(sN, qr[u * SG_SIZE + lane], o);
                    }
                    o = sycl::reduce_over_group(sg, o, sycl::plus<float>());
                    if (lane == 0)
                        pp.out[(int64_t(r) * pp.n_heads + head) * VD + row] = o * sycl::rsqrt(float(KD));

                    // this row's inputs, for the commit's replay
                    float* sr = sv_base + int64_t(r) * srow;
                    if (lane == 0) sr[voff + int64_t(head) * VD + row] = vi;
                    if (row == 0 && lane == 0) {
                        sr[aoff + head] = av;
                        sr[aoff + pp.n_heads + head] = bv;
                    }
                    if (k_owner) {
                        #pragma unroll
                        for (int u = 0; u < PL; ++u) sr[int64_t(khead) * KD + u * SG_SIZE + lane] = kv[u];
                    }
                }
            });
    });
}

template <int PL>
static sycl::event dn_replay_sg(sycl::queue& q_, const DeltaNetParams& p, const float* saved, int n,
                                const std::vector<sycl::event>& deps) {
    const int KD = p.k_dim, VD = p.v_dim;
    const int wg_threads   = kDnRowsPerWG * SG_SIZE;
    const int wgs_per_head = VD / kDnRowsPerWG;
    const size_t groups    = size_t(p.n_heads) * size_t(wgs_per_head);
    const int64_t srow     = deltanet_saved_row_floats(p);

    return q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetParams pp = p;
        const float* const sv_base = saved;
        const int wph = wgs_per_head, nrows = n;

        h.parallel_for(
            sycl::nd_range<1>(groups * size_t(wg_threads), size_t(wg_threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  sgid = int(sg.get_group_id()[0]);
                const int  gid  = int(it.get_group(0));
                const int  head = gid / wph;
                const int  row  = (gid % wph) * kDnRowsPerWG + sgid;
                const int  nk    = pp.n_k_heads ? pp.n_k_heads : pp.n_heads;
                const int  khead = (nk == pp.n_heads) ? head : head / (pp.n_heads / nk);
                const int64_t voff = int64_t(nk) * KD;
                const int64_t aoff = voff + int64_t(pp.n_heads) * VD;

                float* S = pp.state + (int64_t(head) * VD + row) * KD;
                float sv[PL];
                #pragma unroll
                for (int u = 0; u < PL; ++u) sv[u] = S[u * SG_SIZE + lane];
                for (int t = 0; t < nrows; ++t) {
                    const float* sr = sv_base + int64_t(t) * srow;
                    const float* kr = sr + int64_t(khead) * KD;
                    float kv[PL];
                    float w = 0.0f;
                    #pragma unroll
                    for (int u = 0; u < PL; ++u) {
                        kv[u] = kr[u * SG_SIZE + lane];
                        w = sycl::fma(sv[u], kv[u], w);
                    }
                    w = sycl::reduce_over_group(sg, w, sycl::plus<float>());
                    const float av = sr[aoff + head];
                    const float bv = sr[aoff + pp.n_heads + head];
                    const float vi = sr[voff + int64_t(head) * VD + row];
                    const float corr = bv * (vi - av * w);
                    #pragma unroll
                    for (int u = 0; u < PL; ++u) sv[u] = sycl::fma(av, sv[u], corr * kv[u]);
                }
                #pragma unroll
                for (int u = 0; u < PL; ++u) S[u * SG_SIZE + lane] = sv[u];
            });
    });
}

sycl::event launch_deltanet_verify_rows(sycl::queue& q_, const DeltaNetParams& p, const DnGroups& g,
                                        int64_t state_stride, float* saved,
                                        const std::vector<sycl::event>& deps) {
    switch (p.k_dim / SG_SIZE) {
        case 4:  return dn_verify_sg<4> (q_, p, g, state_stride, saved, deps);
        case 8:  return dn_verify_sg<8> (q_, p, g, state_stride, saved, deps);
        default: return dn_verify_sg<16>(q_, p, g, state_stride, saved, deps);
    }
}

sycl::event launch_deltanet_replay(sycl::queue& q_, const DeltaNetParams& p, const float* saved, int n,
                                   const std::vector<sycl::event>& deps) {
    switch (p.k_dim / SG_SIZE) {
        case 4:  return dn_replay_sg<4> (q_, p, saved, n, deps);
        case 8:  return dn_replay_sg<8> (q_, p, saved, n, deps);
        default: return dn_replay_sg<16>(q_, p, saved, n, deps);
    }
}

// ---------------------------------------------------------------------
// Causal depthwise conv1d over the packed qkv projection, kernel width 4.
// The ring buffer holds the last (K-1) tokens per channel; at decode
// there is exactly one new token, so this is a short dot product and a
// shift. Channels are independent, so it is one flat parallel_for.
// ---------------------------------------------------------------------


sycl::event launch_causal_conv1d(sycl::queue& q_, const ConvParams& p,
                                 const std::vector<sycl::event>& deps) {
    return q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const ConvParams pp = p;
        h.parallel_for(sycl::range<1>(size_t(pp.channels)), [=](sycl::id<1> id) {
            const int c = int(id[0]);
            const int K = pp.kernel;
            const bf16_t* w = pp.weight + int64_t(c) * K;
            float* r = pp.ring + int64_t(c) * (K - 1);

            float acc = 0.0f;
            for (int t = 0; t < K - 1; ++t) acc = sycl::fma(r[t], bf16_to_f32(w[t]), acc);
            const float xc = pp.x[c];
            acc = sycl::fma(xc, bf16_to_f32(w[K - 1]), acc);

            // shift the window: oldest out, this token in
            for (int t = 0; t < K - 2; ++t) r[t] = r[t + 1];
            r[K - 2] = xc;

            // SiLU, as the architecture applies after the conv
            pp.out[c] = acc / (1.0f + sycl::exp(-acc));
        });
    });
}

// Decode-only fusion: convolution+SiLU feeds Q/K L2 normalization directly.
// One subgroup owns one packed head.  V heads take the same convolution path
// but deliberately skip normalization.  This removes a dependency-bound
// launch from every DeltaNet layer without changing the stored intermediates.
sycl::event launch_causal_conv1d_l2norm(sycl::queue& q_, const ConvParams& p,
                                        int norm_heads, int head_dim,
                                        const std::vector<sycl::event>& deps) {
    const int heads = (p.channels + head_dim - 1) / head_dim;
    return q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const ConvParams pp = p;
        h.parallel_for(
            sycl::nd_range<1>(size_t(heads) * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int head = int(it.get_group(0));
                float ss = 0.0f;
                for (int d = lane; d < head_dim; d += SG_SIZE) {
                    const int c = head * head_dim + d;
                    if (c >= pp.channels) continue;
                    const int K = pp.kernel;
                    const bf16_t* w = pp.weight + int64_t(c) * K;
                    float* r = pp.ring + int64_t(c) * (K - 1);
                    float acc = 0.0f;
                    for (int t = 0; t < K - 1; ++t)
                        acc = sycl::fma(r[t], bf16_to_f32(w[t]), acc);
                    const float xc = pp.x[c];
                    acc = sycl::fma(xc, bf16_to_f32(w[K - 1]), acc);
                    for (int t = 0; t < K - 2; ++t) r[t] = r[t + 1];
                    r[K - 2] = xc;
                    const float y = acc / (1.0f + sycl::exp(-acc));
                    pp.out[c] = y;
                    if (head < norm_heads) ss = sycl::fma(y, y, ss);
                }
                if (head < norm_heads) {
                    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                    const float inv = sycl::rsqrt(ss + 1e-6f);
                    for (int d = lane; d < head_dim; d += SG_SIZE) {
                        const int c = head * head_dim + d;
                        if (c < pp.channels) pp.out[c] *= inv;
                    }
                }
            });
    });
}

} // namespace b70
