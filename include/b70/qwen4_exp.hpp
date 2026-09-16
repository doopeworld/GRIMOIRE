// =====================================================================
//  qwen4_exp.hpp -- host reference for the operators Qwen4-Exp
//  (Qwen3.8-Flash-Next) needs that no other GRIMOIRE model uses.  The
//  SYCL kernels are the parallel form of exactly this math; keep them in
//  step, and diff them here in bin/test_k2_kernels.
//
//  EXTRACTED, NOT INFERRED, from the vLLM implementation in ref/:
//    ref/qwen4_exp_common_hyperconnection.py   GatedResidual, the norm
//    ref/qwen4_exp_config.py                   the shapes and their rules
//
//  Qwen4ExpTextConfig INHERITS Qwen3NextConfig, so the Gated-DeltaNet +
//  MoE body is a shape this engine already runs.  What is NEW is the
//  RESIDUAL STRUCTURE around every block:
//
//  HYPERCONNECTIONS.  The residual stream is hc_count (4) streams wide,
//  i.e. [hc_count * hidden], not [hidden].  Each block sees ONE hidden-
//  wide input produced by mix(), and its output is injected back into
//  ALL FOUR streams by combine() with a per-stream gate:
//
//      mix:      normed = grouped_rmsnorm(hyper)          per stream
//                g = silu(W_down @ normed / hc)           [lowrank]
//                g = sigmoid(W_up @ g)                    [hc*hidden]
//                out[h] = mean_c( g[c][h] * normed[c][h] )
//
//      combine:  inj[c] = 2 * sigmoid((W_inj @ normed)[c] / hc)
//                out[c][h] = hyper[c][h] + block[h] * inj[c]
//
//  Two things here are silent when wrong, so they are spelled out:
//
//  1. THE DIVISION BY hc_count IS INSIDE THE GATE, BEFORE THE
//     NONLINEARITY -- silu(down(x)/hc) and sigmoid(inj(x)/hc), not
//     silu(down(x))/hc.  Moving it outside changes the gate's operating
//     point and leaves the model fluent.
//
//  2. THE NORM IS `(1 + w)`, NOT `w` (rule 11 territory).  The class is
//     called GroupedGEMMARMSNorm and gemma-4 is the one architecture in
//     this engine that applies its weight DIRECTLY -- but this one does
//     `normalized * (1.0 + self.weight)` with the weight initialised to
//     ZEROS (ref/qwen4_exp_common_hyperconnection.py:87).  Reading the
//     name instead of the forward gives exactly the wrong answer, which
//     is the mistake rule 11 exists to stop.  It is also GROUPED: each
//     hidden-wide stream is normalised on its own variance, and the
//     affine weight spans the whole hc*hidden row.
// =====================================================================
#ifndef B70_QWEN4_EXP_HPP
#define B70_QWEN4_EXP_HPP

#include <cmath>
#include <cstddef>
#include <vector>
#include <limits>
#include <algorithm>

namespace b70 {
namespace qwen4_exp {

inline float silu(float x)    { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// GroupedGemmaRMSNorm over a [hc_count * hidden] row.
//
// group_size == hidden in the reference, so the variance is taken over
// each hidden-wide stream separately -- NOT over the whole hc*hidden row.
// Using one variance for the row couples the streams and is silent.
// The weight is hc*hidden long and applied as (1 + w), per the note above.
inline void hc_norm(const float* x, const float* w, float* out,
                    int hc_count, int hidden, float eps) {
    for (int c = 0; c < hc_count; ++c) {
        const float* xs = x + size_t(c) * hidden;
        float* os = out + size_t(c) * hidden;
        double acc = 0.0;
        for (int i = 0; i < hidden; ++i) acc += double(xs[i]) * double(xs[i]);
        const float inv = 1.0f / std::sqrt(float(acc / hidden) + eps);
        for (int i = 0; i < hidden; ++i)
            os[i] = xs[i] * inv * (1.0f + w[size_t(c) * hidden + i]);
    }
}

// mix(): the gated mean that collapses hc streams to one block input.
//
// `down` is [lowrank, hc*hidden] and `up` is [hc*hidden, lowrank], both
// row-major as nn.Linear stores them (out_features, in_features).
// `normed` must be the hc_norm output; combine() needs the SAME buffer,
// which is why mix returns it rather than recomputing.
inline void hc_mix(const float* normed, const float* down, const float* up,
                   float* out, int hc_count, int hidden, int lowrank) {
    std::vector<float> g(size_t(lowrank), 0.0f);
    const int wide = hc_count * hidden;
    for (int r = 0; r < lowrank; ++r) {
        double acc = 0.0;
        for (int i = 0; i < wide; ++i)
            acc += double(down[size_t(r) * wide + i]) * double(normed[i]);
        // The divide is INSIDE silu -- see note 1 above.
        g[size_t(r)] = silu(float(acc) / float(hc_count));
    }
    for (int h = 0; h < hidden; ++h) {
        float acc = 0.0f;
        for (int c = 0; c < hc_count; ++c) {
            const int i = c * hidden + h;
            double u = 0.0;
            for (int r = 0; r < lowrank; ++r)
                u += double(up[size_t(i) * lowrank + r]) * double(g[size_t(r)]);
            acc += sigmoid(float(u)) * normed[i];
        }
        out[h] = acc / float(hc_count);          // .mean(dim=-2)
    }
}

// combine(): inject one hidden-wide block output back into every stream.
//
// `inj` is [hc_count, hc*hidden].  `hyper` is the UNNORMALISED stream --
// the reference adds to `hyper_input`, not to the normalised copy, and
// using the normalised one would drop the residual the layer is built on.
inline void hc_combine(const float* hyper, const float* normed,
                       const float* block, const float* inj, float* out,
                       int hc_count, int hidden) {
    const int wide = hc_count * hidden;
    for (int c = 0; c < hc_count; ++c) {
        double acc = 0.0;
        for (int i = 0; i < wide; ++i)
            acc += double(inj[size_t(c) * wide + i]) * double(normed[i]);
        // 2 * sigmoid(x / hc): the factor two is not decoration, it makes
        // the gate's midpoint 1.0 so an untrained injection is identity.
        const float w = 2.0f * sigmoid(float(acc) / float(hc_count));
        for (int h = 0; h < hidden; ++h)
            out[size_t(c) * hidden + h] = hyper[size_t(c) * hidden + h] + block[h] * w;
    }
}

// ---------------------------------------------------------------------
//  QSA -- Qwen Sparse Attention.
//
//  Extracted from the reference implementation in vLLM's own test,
//  tests/models/qwen4_exp/test_qsa_reference.py, which is plain torch
//  (the shipped path is CUDA kernels).  Three stages:
//
//    1. INDEX.  An MQA "indexer" scores every compressed key block:
//         scores = einsum("rhd,rnd->rnh", q, keys)
//         logits = relu(scores).sum(dim=-1) / sqrt(d)
//       Note the order: relu FIRST, then sum ACROSS HEADS, then scale.
//       Summing before the relu, or scaling before it, changes which
//       blocks win and is completely silent.
//    2. SELECT.  Top block_topk = indexer_budget / compress_ratio blocks
//       per query, over that query's visible range only.
//    3. EXPAND.  Each chosen block b becomes compress_ratio consecutive
//       TOKENS [b*ratio, b*ratio + ratio), truncated to indexer_budget
//       and masked to < seq_len.  Selecting one token therefore brings
//       its whole block -- that is the "coarser index" QSA is built on.
//
//  Attention then runs over the GATHERED token list, not a range:
//         scores = einsum("hd,khd->hk", q, keys)
//         p      = softmax(scores * softmax_scale)
//  which is the ordinary flash math with the key loop driven by an index
//  array.  -1 entries are holes and contribute nothing.
// ---------------------------------------------------------------------

// Stage 1.  `q` is [n_heads][head_dim] for one query row, `keys` is
// [n_blocks][head_dim] -- MQA, so ONE key stream is shared by every
// indexer head (the config requires indexer_kv_heads == 1).
// `visible` is how many blocks this query may see; the rest are -inf.
inline void qsa_index_logits(const float* q, const float* keys, float* logits,
                             int n_heads, int head_dim, int n_blocks,
                             int visible) {
    const float inv = 1.0f / std::sqrt(float(head_dim));
    for (int n = 0; n < n_blocks; ++n) {
        if (n >= visible) {
            logits[n] = -std::numeric_limits<float>::infinity();
            continue;
        }
        const float* kn = keys + size_t(n) * head_dim;
        float acc = 0.0f;
        for (int h = 0; h < n_heads; ++h) {
            const float* qh = q + size_t(h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += qh[d] * kn[d];
            // relu per head, THEN summed.  Not sum-then-relu.
            acc += dot > 0.0f ? dot : 0.0f;
        }
        logits[n] = acc * inv;
    }
}

// Stage 2.  Top-`topk` block indices by logit, descending.  Fewer than
// topk visible blocks leaves the tail as -1, which stage 3 treats as a
// hole -- it must not be clamped to block 0, which would silently make
// every short sequence attend to its first block repeatedly.
inline void qsa_topk_blocks(const float* logits, int n_blocks, int visible,
                            int topk, int* out) {
    for (int i = 0; i < topk; ++i) out[i] = -1;
    std::vector<int> idx;
    idx.reserve(size_t(visible > 0 ? visible : 0));
    for (int n = 0; n < n_blocks && n < visible; ++n) idx.push_back(n);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return logits[a] > logits[b]; });
    const int width = int(idx.size()) < topk ? int(idx.size()) : topk;
    for (int i = 0; i < width; ++i) out[i] = idx[size_t(i)];
}

// Stage 3.  Block indices -> token indices.  `out` is indexer_budget
// wide; entries outside the sequence, and holes from stage 2, are -1.
inline void qsa_expand_blocks(const int* blocks, int block_topk,
                              int compress_ratio, int token_topk,
                              int seq_len, int* out) {
    int w = 0;
    for (int b = 0; b < block_topk && w < token_topk; ++b) {
        for (int r = 0; r < compress_ratio && w < token_topk; ++r, ++w) {
            const int t = blocks[b] >= 0 ? blocks[b] * compress_ratio + r : -1;
            out[w] = (t >= 0 && t < seq_len) ? t : -1;
        }
    }
    for (; w < token_topk; ++w) out[w] = -1;
}

// The attention itself, over the gathered list.  Plain softmax here
// because this is the reference; the kernel uses the same online form
// every other flash path in this engine does.
inline void qsa_attention(const float* q, const float* k, const float* v,
                          const int* idx, float* out, int n_heads,
                          int kv_heads, int head_dim, int n_idx, int seq_cap,
                          float softmax_scale) {
    const int repeats = n_heads / (kv_heads > 0 ? kv_heads : 1);
    for (int h = 0; h < n_heads; ++h) {
        const int kvh = h / (repeats > 0 ? repeats : 1);
        float m = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n_idx; ++i) {
            if (idx[i] < 0) continue;
            const float* kk = k + (size_t(kvh) * seq_cap + idx[i]) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += q[size_t(h)*head_dim+d] * kk[d];
            dot *= softmax_scale;
            if (dot > m) m = dot;
        }
        float l = 0.0f;
        std::vector<float> acc(size_t(head_dim), 0.0f);
        for (int i = 0; i < n_idx; ++i) {
            if (idx[i] < 0) continue;
            const float* kk = k + (size_t(kvh) * seq_cap + idx[i]) * head_dim;
            const float* vv = v + (size_t(kvh) * seq_cap + idx[i]) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += q[size_t(h)*head_dim+d] * kk[d];
            const float p = std::exp(dot * softmax_scale - m);
            l += p;
            for (int d = 0; d < head_dim; ++d) acc[size_t(d)] += p * vv[d];
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (int d = 0; d < head_dim; ++d)
            out[size_t(h) * head_dim + d] = acc[size_t(d)] * inv;
    }
}

} // namespace qwen4_exp
} // namespace b70
#endif
