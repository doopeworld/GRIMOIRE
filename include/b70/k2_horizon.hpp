// =====================================================================
//  k2_horizon.hpp -- host reference for the three operators K2-Horizon
//  needs that no other GRIMOIRE model uses.  The SYCL kernels are the
//  parallel form of exactly this math; keep them in step.
//
//  Extracted from modeling_k2_horizon.py rather than inferred.  The
//  architecture is Qwen3-MoE with three substitutions:
//
//    1. GROUPED RMSNorm (layernorm_num_groups=2): the variance is taken
//       per contiguous group, not over the whole row.  With 2 groups at
//       hidden 2560 that is two independent 1280-wide normalizations,
//       and the weight is still applied across the full row afterwards.
//    2. SOFTPLUS attention gate (beta=log 2), applied to the attention
//       output BEFORE o_proj.
//    3. SIGMOID router whose bias steers SELECTION ONLY -- the returned
//       weights come from the unbiased scores.  Used identically by the
//       100-expert MoE and by MoVA's 64 value experts.
//
//  MoVA itself needs no operator here: it is v_proj replaced by a routed
//  bank of [kv_heads*head_dim, hidden] experts, top-4, each output put
//  through SiLU and accumulated with the router weight.  That is the
//  router below plus the existing grouped-GEMM path.
// =====================================================================
#ifndef B70_K2_HORIZON_HPP
#define B70_K2_HORIZON_HPP

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <vector>

namespace b70 {
namespace k2 {

// ---- 1. grouped RMSNorm ---------------------------------------------
// out[i] = w[i] * x[i] * rsqrt(mean(x[g]^2) + eps), g = group of i.
// n_groups == 1 degenerates to the standard RMSNorm already in ops.cpp,
// which is what the three dense layers and every other model use.
inline void rmsnorm_grouped(const float* x, const float* w, float* out,
                            int n, int n_groups, float eps) {
    const int g = n / n_groups;                 // caller guarantees n % n_groups == 0
    for (int k = 0; k < n_groups; ++k) {
        const float* xs = x + size_t(k) * g;
        double ss = 0;
        for (int i = 0; i < g; ++i) ss += double(xs[i]) * xs[i];
        const float s = 1.0f / std::sqrt(float(ss / g) + eps);
        for (int i = 0; i < g; ++i) {
            const size_t o = size_t(k) * g + i;
            out[o] = xs[i] * s * w[o];          // weight spans the FULL row
        }
    }
}

// ---- 2. softplus gate ------------------------------------------------
// torch.nn.functional.softplus(x, beta) = log1p(exp(beta*x)) / beta, and
// falls back to the identity once beta*x exceeds threshold (20 by
// default) so the exp cannot overflow.  K2 passes beta = log 2, which
// makes this exactly log2(1 + 2^x) -- but compute it in the torch form
// so the threshold behaviour matches bit for bit.
inline float softplus(float x, float beta) {
    const float bx = beta * x;
    if (bx > 20.0f) return x;                   // torch's linear region
    return std::log1p(std::exp(bx)) / beta;
}

inline void softplus_gate(const float* attn, const float* gate, float* out,
                          size_t n, float beta) {
    for (size_t i = 0; i < n; ++i) out[i] = attn[i] * softplus(gate[i], beta);
}

// ---- 3. sigmoid router with selection-only bias ----------------------
// Shared by the MoE block and MoVA.  The bias is added ONLY to the
// values top-k ranks on; the weight that multiplies an expert's output
// is gathered from the UNBIASED score.  Getting that backwards is
// silent: the routes stay plausible and only the mixture is wrong.
//
// normalize follows norm_topk_prob for the MoE block; MoVA normalizes
// whenever top_k > 1.  Both are true for this checkpoint, but keep the
// flag so the two call sites stay honest.
inline void router_topk(const float* logits, const float* bias,
                        int n_experts, int top_k,
                        bool sigmoid_score, bool normalize, float scaling,
                        int* out_idx, float* out_w) {
    const size_t ne = size_t(n_experts);      // named: vector(size_t(x)) is a vexing parse
    std::vector<float> score(ne);
    for (int e = 0; e < n_experts; ++e) {
        const float l = logits[e];
        score[size_t(e)] = sigmoid_score ? 1.0f / (1.0f + std::exp(-l)) : l;
    }
    if (!sigmoid_score) {                       // softmax branch, float32
        float mx = score[0];
        for (int e = 1; e < n_experts; ++e) mx = std::max(mx, score[size_t(e)]);
        double sum = 0;
        for (int e = 0; e < n_experts; ++e) {
            score[size_t(e)] = std::exp(score[size_t(e)] - mx);
            sum += score[size_t(e)];
        }
        for (int e = 0; e < n_experts; ++e) score[size_t(e)] /= float(sum);
    }

    // rank on the biased value, keep the unbiased one as the weight
    std::vector<int> order(ne);
    for (int e = 0; e < n_experts; ++e) order[size_t(e)] = e;
    std::vector<float> sel(ne);
    for (int e = 0; e < n_experts; ++e)
        sel[size_t(e)] = score[size_t(e)] + (bias ? bias[e] : 0.0f);
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
        [&](int a, int b) {
            if (sel[size_t(a)] != sel[size_t(b)]) return sel[size_t(a)] > sel[size_t(b)];
            return a < b;                        // deterministic tie-break
        });

    double wsum = 0;
    for (int i = 0; i < top_k; ++i) {
        out_idx[i] = order[size_t(i)];
        out_w[i]   = score[size_t(order[size_t(i)])];
        wsum += out_w[i];
    }
    if (normalize && wsum > 0)
        for (int i = 0; i < top_k; ++i) out_w[i] = float(out_w[i] / wsum);
    if (scaling != 1.0f)
        for (int i = 0; i < top_k; ++i) out_w[i] *= scaling;
}

// ---- layer classification -------------------------------------------
// K2HorizonDecoderLayer.__init__:
//   is_sparse = (idx not in mlp_only_layers)
//               and (num_experts > 0 and (idx+1) % decoder_sparse_step == 0)
// A sparse layer gets MoVA attention AND the routed MoE; a dense layer
// gets ordinary attention and one intermediate_size MLP.  For this
// checkpoint that is layers 0-2 dense and 3-47 sparse, so the two shapes
// coexist in one model and the loader cannot assume either.
inline bool is_sparse_layer(int idx, const int* mlp_only, int n_mlp_only,
                            int n_experts, int sparse_step) {
    for (int i = 0; i < n_mlp_only; ++i) if (mlp_only[i] == idx) return false;
    if (n_experts <= 0) return false;
    if (sparse_step <= 0) return false;
    return ((idx + 1) % sparse_step) == 0;
}

// ---- MoVA value mixture ---------------------------------------------
// value = sum_j w_j * silu(expert_{idx_j} @ x).  The SiLU sits on the
// EXPERT OUTPUT, not on the input, and applies before the router weight
// scales it -- combine_routed_experts(activation=F.silu).
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

inline void mova_combine(const float* const* expert_out, const float* w,
                         int top_k, int dim, float* out) {
    for (int d = 0; d < dim; ++d) out[d] = 0.0f;
    for (int j = 0; j < top_k; ++j) {
        const float* e = expert_out[j];
        for (int d = 0; d < dim; ++d) out[d] += w[j] * silu(e[d]);
    }
}

}  // namespace k2
}  // namespace b70

#endif
