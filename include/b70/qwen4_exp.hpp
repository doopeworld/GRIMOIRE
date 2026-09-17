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
#include <cstdint>
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
//       THE INCOMPLETE BLOCK IS THEN APPENDED UNCONDITIONALLY: stage 1
//       can only see complete blocks, so without this a query cannot
//       attend to its own token.  See stage 3 below.
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

// Stage 3.  Block indices -> token indices, PLUS THE TAIL.
//
// The tail is the part this engine got wrong first and it is not
// optional: `visible` in stage 1 counts only COMPLETE blocks
// (min((pos+1)/ratio, seq_len/ratio) in the reference's
// _qsa_select_paged_reference), so the tokens of the block still being
// filled -- INCLUDING THE QUERY'S OWN TOKEN -- can never be selected by
// the indexer.  The reference appends them unconditionally:
//
//     tail_start = (pos + 1) / ratio * ratio
//     tail       = tail_start .. pos            (at most ratio-1 entries)
//
// (tests/models/qwen4_exp/test_qsa_reference.py,
// _expand_qsa_indices_reference).  Drop it and a decode step attends to
// everything except the most recent few tokens, which is fluent and
// wrong -- exactly the class this whole file is organised around.  The
// selected blocks are all below tail_start, so the two never overlap.
//
// `out` is therefore token_topk + compress_ratio - 1 wide.
//
// ONE DELIBERATE DIFFERENCE FROM THE REFERENCE, and it changes no value:
// vLLM stable-sorts the valid entries to the front and passes their count
// as the attention kernel's loop bound.  This engine's QSA attention
// skips holes inside the loop instead, so the gathered set and the ORDER
// of the valid entries are identical and the compaction would only move
// the -1s.  Nothing downstream may treat position as meaningful.
inline int qsa_expand_width(int token_topk, int compress_ratio) {
    return token_topk + compress_ratio - 1;
}

inline void qsa_expand_blocks(const int* blocks, int block_topk,
                              int compress_ratio, int token_topk,
                              int seq_len, int query_pos, int* out) {
    int w = 0;
    for (int b = 0; b < block_topk && w < token_topk; ++b) {
        for (int r = 0; r < compress_ratio && w < token_topk; ++r, ++w) {
            const int t = blocks[b] >= 0 ? blocks[b] * compress_ratio + r : -1;
            out[w] = (t >= 0 && t < seq_len) ? t : -1;
        }
    }
    for (; w < token_topk; ++w) out[w] = -1;
    // the incomplete block the indexer cannot see
    const int visible    = query_pos + 1;
    const int tail_start = visible / compress_ratio * compress_ratio;
    for (int i = 0; i < compress_ratio - 1; ++i, ++w) {
        const int t = tail_start + i;
        out[w] = (i < visible - tail_start && t < seq_len) ? t : -1;
    }
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

// ---------------------------------------------------------------------
//  PLE -- the per-layer n-gram embedding.
//
//  Extracted from the kernel and its docstring in vLLM,
//  vllm/models/qwen4_exp/nvidia/ops/ple.py (copied to
//  ref/qwen4_exp_nvidia_ngram_embedding.py / ple_layer.py).  The
//  docstring states the whole contract:
//
//    ids:  offset[h] + (xor_i(token[t-i] * multiplier[i]) % size[h])
//    gate: d = dot(RMSNorm(key), RMSNorm(query)) / sqrt(H)
//          g = sigmoid(sign(d) * sqrt(max(abs(d), 1e-6)))
//    conv: silu(sum_k weight[k] * history[t + k*dilation]) added to the
//          gated output, updating a persistent history
//
//  A NOTE ON SIZE, because it decides whether this model can run here at
//  all: the table is ngram_vocab_size_base (20,000,000) rows and vLLM
//  keeps it in PINNED HOST memory (Qwen4ExpPLEPinnedHostEmbedding), not
//  in VRAM.  It is a sparse gather -- only the rows for the current
//  tokens are touched -- so host residency is the intended design rather
//  than a fallback.  GRIMOIRE is otherwise VRAM-only; this one table is
//  the exception the architecture expects.
//
//  Two details below are silent when wrong:
//
//  1. EACH HEAD HAS ITS OWN N-GRAM ORDER.  head h covers order
//     h / heads_per_ngram + 2, so with heads_per_ngram 8 the first eight
//     heads are BIGRAMS and the next eight TRIGRAMS.  The xor for a head
//     includes a predecessor only while shift < order.  Mixing every
//     predecessor into every head makes them all the same order, which
//     still hashes, still indexes, and still produces text.
//
//  2. THE EOS BOUNDARY IS STICKY.  Walking backwards, once a predecessor
//     is the EOS token every older position is treated as EOS too.  That
//     is what stops an n-gram spanning two documents.  Dropping it reads
//     across the boundary and is invisible in any single-document test.
// ---------------------------------------------------------------------

// n-gram ids for ONE token position.  `hist` is the preceding tokens,
// newest last, at least ngram_context_len long; `hist_len` may be
// shorter at the start of a sequence, where the missing predecessors
// read as 0 exactly as the kernel's masked load does.
inline void ple_ngram_ids(int token, const int* hist, int hist_len,
                          const int64_t* multipliers, const int64_t* sizes,
                          const int64_t* offsets, int ngram_context_len,
                          int heads_per_ngram, int ngram_heads,
                          int eos_token_id, int64_t* out) {
    for (int h = 0; h < ngram_heads; ++h) {
        const int order = h / heads_per_ngram + 2;
        int64_t mixed = int64_t(token) * multipliers[0];
        bool crossed = false;
        for (int shift = 1; shift <= ngram_context_len; ++shift) {
            const int at = hist_len - shift;
            int64_t cand = (at >= 0) ? int64_t(hist[at]) : 0;
            // sticky EOS: everything older than a boundary is EOS
            if (crossed) cand = eos_token_id;
            if (cand == eos_token_id) crossed = true;
            if (order > shift)
                mixed ^= cand * multipliers[shift];
        }
        int64_t r = sizes[h] ? (mixed % sizes[h]) : 0;
        if (r < 0) r += sizes[h];           // torch.remainder semantics
        out[h] = r + offsets[h];
    }
}

// The PLE gate, and the two rows it produces.
//
// This is the whole of ops/ple.py's _ple_gate_kernel, and three things in
// it are easy to lose:
//
//  * IT IS PER STREAM.  key, hidden and both outputs are [hc*H] wide and
//    every quantity below -- the two norms, the dot, the gate -- is
//    computed independently for each hidden-wide stream.  `value` is only
//    H wide and the SAME value row is gated into all of them.
//  * THE NORMS HAVE WEIGHTS.  "d = dot(RMSNorm(key), RMSNorm(query))"
//    reads like a plain normalisation; Qwen4ExpPLEGroupedNorm applies
//    (1 + w) after it (ref/qwen4_exp_nvidia_ple_layer.py:63), with a
//    separate weight for every element of the hc*H row.  Dropping the
//    affine leaves a gate in (0,1) that still gates.
//  * THE SECOND OUTPUT IS NORMED FROM THE FIRST.  conv_input is
//    norm_conv(gated), not norm_conv(value) and not gated itself.
//
// `query` is the multi-stream hidden state.  The sqrt-of-magnitude
// shaping keeps the gate responsive near zero and the 1e-6 floor is what
// stops sqrt'(0) blowing up.
inline void ple_gate(const float* key, const float* value, const float* query,
                     const float* nk, const float* nq, const float* ncw,
                     float* gated, float* conv_in,
                     int hc_count, int H, float eps) {
    for (int c = 0; c < hc_count; ++c) {
        const float* k = key   + size_t(c) * H;
        const float* x = query + size_t(c) * H;
        const float* wk = nk + size_t(c) * H;
        const float* wq = nq + size_t(c) * H;
        double ks = 0.0, qs = 0.0;
        for (int i = 0; i < H; ++i) { ks += double(k[i])*k[i]; qs += double(x[i])*x[i]; }
        const float ki = 1.0f / std::sqrt(float(ks / H) + eps);
        const float qi = 1.0f / std::sqrt(float(qs / H) + eps);
        double dot = 0.0;
        for (int i = 0; i < H; ++i)
            dot += double(k[i]*ki*(1.0f+wk[i])) * double(x[i]*qi*(1.0f+wq[i]));
        const float d   = float(dot) / std::sqrt(float(H));
        const float sgn = d < 0.0f ? -1.0f : 1.0f;
        const float mag = std::fabs(d) > 1e-6f ? std::fabs(d) : 1e-6f;
        const float g   = sigmoid(sgn * std::sqrt(mag));

        float* go = gated + size_t(c) * H;
        for (int i = 0; i < H; ++i) go[i] = g * value[i];
        double gs = 0.0;
        for (int i = 0; i < H; ++i) gs += double(go[i]) * go[i];
        const float gi = 1.0f / std::sqrt(float(gs / H) + eps);
        const float* wc = ncw + size_t(c) * H;
        float* co = conv_in + size_t(c) * H;
        for (int i = 0; i < H; ++i) co[i] = go[i] * gi * (1.0f + wc[i]);
    }
}

// The dilated depthwise short convolution that finishes a PLE layer.
//
//     out[t] = hidden[t] + gated[t] + silu( sum_k w[k] * conv_in[t - (K-1-k)*D] )
//
// D is ngram_size, K is ple_conv_kernel_size, and the weight is per
// CHANNEL (the conv is depthwise over all hc*H of them).  Taps before the
// start of the sequence read zero, which is what an all-zero conv state
// gives at position 0.  The two residuals are BOTH added: the reference
// accumulates the convolution into the gated output and then adds the
// unmodified multi-stream state (ops/ple.py, _ple_conv_kernel tail), so a
// PLE layer REPLACES the state rather than contributing to it.
inline void ple_conv(const float* conv_in, const float* gated,
                     const float* hidden, const float* w, float* out,
                     int t, int channels, int kernel, int dilation) {
    for (int c = 0; c < channels; ++c) {
        float acc = 0.0f;
        for (int k = 0; k < kernel; ++k) {
            const int at = t - (kernel - 1 - k) * dilation;
            if (at < 0) continue;
            acc += w[size_t(c) * kernel + k] * conv_in[size_t(at) * channels + c];
        }
        out[size_t(c)] = hidden[size_t(c)] + gated[size_t(c)] + silu(acc);
    }
}


// ---------------------------------------------------------------------
//  The n-gram table's LAYOUT is computed, not stored.
//
//  Nothing in the checkpoint says how big each head's slice of the table
//  is or where it starts: Qwen4ExpNGramEmbedding derives both, and a
//  reader that guesses "equal slices of ngram_vocab_size_base" lands one
//  row out on the second head and silently reads a neighbour's rows.
//
//  sizes[h]   = the (global_head+1)-th PRIME strictly greater than
//               ngram_vocab_size_base - 1, where global_head is
//               ple_dense_layer_id * ngram_heads + h.  Primes, because
//               the index is a modulus.
//  offsets[h] = the running sum of the sizes before it.
//  multipliers[i] = 2 * (splitmix64(seed + 0x9E3779B97F4A7C15*(i+1)
//                        + 10007*ple_dense_layer_id) % half) + 1,
//               with half = ((2^63 - 1) / vocab_size) / 2, so every
//               multiplier is ODD and a token times it cannot overflow
//               into the sign bit.
//
//  Source: ref/qwen4_exp_nvidia_ngram_embedding.py:535-630.
// ---------------------------------------------------------------------
inline uint64_t splitmix64(uint64_t v) {
    v += 0x9E3779B97F4A7C15ull;
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
    return v ^ (v >> 31);
}

inline uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return uint64_t((unsigned __int128)a * b % m);
}

inline uint64_t powmod64(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    b %= m;
    while (e) { if (e & 1) r = mulmod64(r, b, m); b = mulmod64(b, b, m); e >>= 1; }
    return r;
}

// Deterministic Miller-Rabin with the reference's own witness set.
inline bool is_prime_64(uint64_t v) {
    if (v < 2) return false;
    for (uint64_t p : {2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,
                       29ull,31ull,37ull})
        if (v % p == 0) return v == p;
    uint64_t d = v - 1; int s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    for (uint64_t a : {2ull,325ull,9375ull,28178ull,450775ull,9780504ull,
                       1795265022ull}) {
        if (a % v == 0) continue;
        uint64_t x = powmod64(a, d, v);
        if (x == 1 || x == v - 1) continue;
        bool ok = false;
        for (int i = 1; i < s; ++i) {
            x = mulmod64(x, x, v);
            if (x == v - 1) { ok = true; break; }
        }
        if (!ok) return false;
    }
    return true;
}

inline uint64_t nth_prime_after(uint64_t start, int count) {
    uint64_t prime = start;
    for (int i = 0; i < count; ++i) {
        uint64_t c = prime + 1;
        if (c <= 2) { prime = 2; continue; }
        if ((c & 1) == 0) ++c;
        while (!is_prime_64(c)) c += 2;
        prime = c;
    }
    return prime;
}

inline void ngram_multipliers(int ngram_size, int64_t vocab_size, int64_t seed,
                              int ple_dense_layer_id, int64_t* out) {
    const int64_t max_mul = std::numeric_limits<int64_t>::max() / vocab_size;
    const int64_t half    = max_mul / 2 > 0 ? max_mul / 2 : 1;
    const uint64_t base   = uint64_t(seed) +
                            10007ull * uint64_t(ple_dense_layer_id);
    for (int i = 0; i < ngram_size; ++i) {
        const uint64_t v = base + 0x9E3779B97F4A7C15ull * uint64_t(i + 1);
        out[i] = 2 * int64_t(splitmix64(v) % uint64_t(half)) + 1;
    }
}

// Returns the total row count; sizes and offsets are ngram_heads long.
inline int64_t ngram_vocab_layout(int64_t vocab_base, int ngram_heads,
                                  int ple_dense_layer_id, int64_t* sizes,
                                  int64_t* offsets) {
    int64_t off = 0;
    for (int h = 0; h < ngram_heads; ++h) {
        const int global_head = ple_dense_layer_id * ngram_heads + h;
        sizes[h]   = int64_t(nth_prime_after(uint64_t(vocab_base - 1),
                                             global_head + 1));
        offsets[h] = off;
        off += sizes[h];
    }
    return off;
}

} // namespace qwen4_exp
} // namespace b70
#endif
