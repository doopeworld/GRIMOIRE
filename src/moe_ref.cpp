// =====================================================================
//  moe_ref.cpp  --  host reference MoE. This is the oracle the fused
//  SYCL kernel is checked against.
// =====================================================================
#include "b70/moe.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace b70 {

void route_ref(const float* logits, int num_experts, int top_k,
               bool norm_topk, Routing& out) {
    // softmax over all experts
    float m = logits[0];
    for (int e = 1; e < num_experts; ++e) m = std::max(m, logits[e]);
    double z = 0.0;
    std::vector<float> p(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        p[e] = std::exp(logits[e] - m);
        z += p[e];
    }
    for (int e = 0; e < num_experts; ++e) p[e] = float(p[e] / z);

    // top-k by probability. Partial selection: with 256 experts and k=8
    // a full sort would be wasteful, and on device this becomes a
    // sub-group bitonic top-k.
    std::vector<int> idx(num_experts);
    for (int e = 0; e < num_experts; ++e) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) {
                          if (p[a] != p[b]) return p[a] > p[b];
                          return a < b;          // deterministic tie-break
                      });

    out.k = top_k;
    double s = 0.0;
    for (int j = 0; j < top_k; ++j) {
        out.expert[j] = idx[j];
        out.weight[j] = p[idx[j]];
        s += out.weight[j];
    }
    if (norm_topk && s > 0.0)
        for (int j = 0; j < top_k; ++j) out.weight[j] = float(out.weight[j] / s);
}

static inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

// Single expert: down( silu(gate.x) * (up.x) )
static void expert_ffn(const QuantWeight& gate_up, const QuantWeight& down,
                       int64_t g_row, int64_t u_row, int64_t d_row,
                       int H, int I, const float* x, float* acc, float scale) {
    std::vector<float> h(I);
    for (int i = 0; i < I; ++i) {
        double g = 0.0, u = 0.0;
        for (int c = 0; c < H; ++c) {
            g += double(gate_up.at(int(g_row) + i, c)) * x[c];
            u += double(gate_up.at(int(u_row) + i, c)) * x[c];
        }
        h[i] = silu(float(g)) * float(u);
    }
    for (int o = 0; o < H; ++o) {
        double a = 0.0;
        for (int i = 0; i < I; ++i)
            a += double(down.at(int(d_row) + o, i)) * h[i];
        acc[o] += scale * float(a);
    }
}

void moe_forward_ref(const MoeLayer& L, const Routing& r,
                     const float* x, float* y) {
    const int H = L.cfg.hidden, I = L.cfg.inter;
    for (int o = 0; o < H; ++o) y[o] = 0.0f;

    for (int j = 0; j < r.k; ++j) {
        const int e = r.expert[j];
        expert_ffn(L.gate_up, L.down,
                   L.gate_row(e), L.up_row(e), L.down_row(e),
                   H, I, x, y, r.weight[j]);
    }

    if (L.has_shared) {
        const int SI = L.cfg.shared_inter;
        expert_ffn(L.shared_gate_up, L.shared_down, 0, SI, 0,
                   H, SI, x, y, 1.0f);
    }
}

} // namespace b70
