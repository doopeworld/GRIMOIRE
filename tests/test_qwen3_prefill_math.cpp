// Host check of the scalar pieces used by prefill.cpp's M>1 Q/K kernels.
// The reference follows modeling_qwen3.py: plain-w RMSNorm, then full RoPE,
// then causal GQA with 32 query heads and 8 key/value heads.
#include "b70/prefill_qk_math.hpp"
#include "b70/qwen35.hpp"
#include "b70/formats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace b70;
namespace b70 { float g_norm_weight_offset = 1.0f; }

int main() {
    Qwen35Config cfg;
    cfg.is_qwen3 = true;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 8;
    cfg.head_dim = 128;
    cfg.partial_rope = 1.0f;
    cfg.rope_theta = 1000000.0f;
    constexpr int M = 4, Q = 32, K = 8, D = 128, H = 4096;
    constexpr float eps = 1e-6f;
    if (cfg.norm_weight_offset() != 0.0f || cfg.partial_rope != 1.0f)
        return 1;
    g_norm_weight_offset = cfg.norm_weight_offset();
    if (prefill_default_weight_offset() != 0.0f) return 1;

    std::vector<float> q(M * Q * D), k(M * K * D), v(M * K * D);
    std::vector<float> q_pref = q, k_pref = k, q_ref, k_ref;
    std::vector<float> qw(D), kw(D), hw(H), hidden(M * H);
    for (int d = 0; d < D; ++d) {
        qw[d] = bf16_to_f32(f32_to_bf16(0.62f + 0.004f * (d % 31)));
        kw[d] = bf16_to_f32(f32_to_bf16(0.74f + 0.003f * (d % 23)));
    }
    for (int d = 0; d < H; ++d)
        hw[d] = bf16_to_f32(f32_to_bf16(0.8f + 0.002f * (d % 73)));
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = 0.5f * std::sin(0.017f * float(i)) + 0.3f * std::cos(0.031f * float(i));
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = 0.6f * std::cos(0.023f * float(i)) + 0.2f * std::sin(0.011f * float(i));
        v[i] = std::sin(0.013f * float(i));
    }
    for (size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = 0.4f * std::sin(0.009f * float(i)) + 0.2f;
    q_pref = q; k_pref = k; q_ref = q; k_ref = k;

    // Same per-token/per-head addressing and scalar math as batched prefill.
    auto prefill_heads = [&](std::vector<float>& x, int heads,
                             const std::vector<float>& w) {
        for (int t = 0; t < M; ++t) for (int h = 0; h < heads; ++h) {
            float* p = x.data() + prefill_head_offset(t, h, heads, D);
            float sum = 0;
            for (int d = 0; d < D; ++d) sum += p[d] * p[d];
            const float scale = 1.0f / std::sqrt(sum / D + eps);
            for (int d = 0; d < D; ++d)
                p[d] *= scale * prefill_norm_weight(w[d], prefill_default_weight_offset());
            for (int j = 0; j < D / 2; ++j) {
                const float angle = float(17 + t) *
                    std::pow(cfg.rope_theta, -2.0f * j / D);
                prefill_rope_pair(p[j], p[j + D / 2],
                                  std::cos(angle), std::sin(angle),
                                  p[j], p[j + D / 2]);
            }
        }
    };
    prefill_heads(q_pref, Q, qw);
    prefill_heads(k_pref, K, kw);
    // A fixed 1+w prefill default, as used before this fix, must trip the
    // check for the actual non-unit Qwen3 norm weights.
    g_norm_weight_offset = 1.0f;
    std::vector<float> q_old = q;
    prefill_heads(q_old, Q, qw);
    g_norm_weight_offset = cfg.norm_weight_offset();

    // Independent sequential decode / HF formula, one token at a time.
    auto decode_heads = [&](std::vector<float>& x, int heads,
                            const std::vector<float>& w) {
        for (int t = 0; t < M; ++t) for (int h = 0; h < heads; ++h) {
            const size_t base = size_t(t) * heads * D + size_t(h) * D;
            double ss = 0;
            for (int d = 0; d < D; ++d) ss += double(x[base+d]) * x[base+d];
            const float inv = float(1.0 / std::sqrt(ss / D + eps));
            for (int d = 0; d < D; ++d) x[base+d] *= inv * w[d];
            for (int j = 0; j < D / 2; ++j) {
                const double a = (17 + t) / std::pow(double(cfg.rope_theta),
                                                      2.0 * j / D);
                const float left = x[base+j], right = x[base+j+D/2];
                x[base+j] = float(left * std::cos(a) - right * std::sin(a));
                x[base+j+D/2] = float(left * std::sin(a) + right * std::cos(a));
            }
        }
    };
    decode_heads(q_ref, Q, qw);
    decode_heads(k_ref, K, kw);

    float max_diff = 0;
    for (size_t i = 0; i < q.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(q_pref[i] - q_ref[i]));
    for (size_t i = 0; i < k.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(k_pref[i] - k_ref[i]));
    float old_diff = 0;
    for (size_t i = 0; i < q.size(); ++i)
        old_diff = std::max(old_diff, std::fabs(q_old[i] - q_ref[i]));
    if (old_diff < 0.1f) {
        std::fprintf(stderr, "test did not detect the old 1+w convention\n");
        return 1;
    }

    // Include the prefill whole-row RMSNorm, where the same default-offset
    // mistake also shifted the input, post-attention and final norms.
    for (int t = 0; t < M; ++t) {
        const float* row = hidden.data() + size_t(t) * H;
        double ss = 0;
        for (int d = 0; d < H; ++d) ss += double(row[d]) * row[d];
        const float inv = float(1.0 / std::sqrt(ss / H + eps));
        for (int d = 0; d < H; ++d) {
            const float got = row[d] * inv *
                prefill_norm_weight(hw[d], prefill_default_weight_offset());
            const float ref = row[d] * inv * hw[d];
            max_diff = std::max(max_diff, std::fabs(got - ref));
        }
    }

    // Causal GQA: query head h reads KV head h/4. Check every row, not just
    // row zero, so the M>1 stride and shared KV mapping are exercised.
    for (int t = 0; t < M; ++t) for (int h = 0; h < Q; ++h) {
        float score_pref[M], score_ref[M], vmax = -1e30f, rmax = -1e30f;
        for (int s = 0; s <= t; ++s) {
            score_pref[s] = score_ref[s] = 0;
            for (int d = 0; d < D; ++d) {
                score_pref[s] += q_pref[prefill_head_offset(t,h,Q,D)+d] *
                    k_pref[prefill_head_offset(s,h/4,K,D)+d];
                score_ref[s] += q_ref[(size_t(t)*Q+h)*D+d] *
                    k_ref[(size_t(s)*K+h/4)*D+d];
            }
            score_pref[s] /= std::sqrt(float(D));
            score_ref[s] /= std::sqrt(float(D));
            vmax = std::max(vmax, score_pref[s]);
            rmax = std::max(rmax, score_ref[s]);
        }
        float z_pref=0,z_ref=0,out_pref=0,out_ref=0;
        for (int s=0;s<=t;++s) {
            const float a=std::exp(score_pref[s]-vmax);
            const float b=std::exp(score_ref[s]-rmax);
            z_pref+=a; z_ref+=b;
            out_pref+=a*v[(size_t(s)*K+h/4)*D+17];
            out_ref+=b*v[(size_t(s)*K+h/4)*D+17];
        }
        max_diff=std::max(max_diff,std::fabs(out_pref/z_pref-out_ref/z_ref));
    }
    if (max_diff > 2e-4f) {
        std::fprintf(stderr, "Qwen3 prefill/decode math differs: %.7g\n", max_diff);
        return 1;
    }
    std::printf("Qwen3 M=4 plain-w prefill/decode Q/K, RoPE, GQA: pass (max %.7g)\n",
                max_diff);
    return 0;
}
