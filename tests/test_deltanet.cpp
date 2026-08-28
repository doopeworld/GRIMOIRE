// =====================================================================
//  test_deltanet.cpp  --  the Gated DeltaNet recurrence, validated on
//  the host before it is written as a kernel.
//
//  Shapes come from Ornith-1.0-35B / Qwen3.6-35B-A3B:
//    in_proj_qkv [8192,2048] = q 16x128 | k 16x128 | v 32x128
//    in_proj_z   [4096,2048] = output gate, 32 heads x 128
//    in_proj_a   [32,2048]   per-head decay
//    in_proj_b   [32,2048]   per-head beta
//    conv1d      [8192,1,4]  depthwise causal conv over qkv
//    A_log, dt_bias [32]
//
//  The state is [v_dim x k_dim] per value head -- 128x128 floats x 32
//  heads = 2 MiB per layer, 60 MiB across all 30 linear layers, and
//  CONSTANT in context length. A conventional KV cache at this model's
//  262144-token context would be orders of magnitude larger.
//
//  Recurrence (gated delta rule), per head, per token:
//      w   = S_{t-1} k                      (v_dim vector)
//      S_t = a*S_{t-1} + b*(v - a*w) k^T    (rank-1 update)
//      o   = S_t q
//  with a = exp(-exp(A_log)*softplus(dt+dt_bias)) and b = sigmoid(beta).
//
//  The subtlety worth pinning down: the decay `a` must be applied to the
//  OLD state inside the correction term as well (`v - a*w`), not just to
//  the state itself. Dropping it there still produces smooth, plausible
//  output that drifts from the reference as context grows -- the kind of
//  bug that shows up as "the model gets confused in long conversations"
//  rather than as an obvious failure.
// =====================================================================
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

static constexpr int SG = 16;

static inline float softplus(float x) {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}
static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ---------------------------------------------------------------------
// Reference: the recurrence written as plainly as possible.
// ---------------------------------------------------------------------
static void deltanet_ref(const float* q, const float* k, const float* v,
                         float a, float b, float* S, float* out,
                         int KD, int VD) {
    std::vector<float> w(VD, 0.0f);
    for (int i = 0; i < VD; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < KD; ++j) acc += S[size_t(i) * KD + j] * k[j];
        w[i] = acc;
    }
    for (int i = 0; i < VD; ++i) {
        const float corr = b * (v[i] - a * w[i]);
        for (int j = 0; j < KD; ++j)
            S[size_t(i) * KD + j] = a * S[size_t(i) * KD + j] + corr * k[j];
    }
    for (int i = 0; i < VD; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < KD; ++j) acc += S[size_t(i) * KD + j] * q[j];
        out[i] = acc;
    }
}

// ---------------------------------------------------------------------
// The parallel decomposition the SYCL kernel uses: one work-item per
// v_dim row of the state. Each item owns S[i][0..KD-1] and needs only
// the shared q, k, v[i], a, b -- so the whole step is embarrassingly
// parallel across rows with no cross-item communication at all.
// ---------------------------------------------------------------------
static void deltanet_rowwise(const float* q, const float* k, const float* v,
                             float a, float b, float* S, float* out,
                             int KD, int VD) {
    for (int i = 0; i < VD; ++i) {              // -> one work-item each
        float* row = S + size_t(i) * KD;
        float w = 0.0f;
        for (int j = 0; j < KD; ++j) w += row[j] * k[j];
        const float corr = b * (v[i] - a * w);
        float o = 0.0f;
        for (int j = 0; j < KD; ++j) {
            row[j] = a * row[j] + corr * k[j];
            o += row[j] * q[j];                 // fused: no second pass
        }
        out[i] = o;
    }
}

// The bug worth guarding against.
static void deltanet_missing_decay(const float* q, const float* k, const float* v,
                                   float a, float b, float* S, float* out,
                                   int KD, int VD) {
    for (int i = 0; i < VD; ++i) {
        float* row = S + size_t(i) * KD;
        float w = 0.0f;
        for (int j = 0; j < KD; ++j) w += row[j] * k[j];
        const float corr = b * (v[i] - w);      // <-- decay dropped here
        float o = 0.0f;
        for (int j = 0; j < KD; ++j) {
            row[j] = a * row[j] + corr * k[j];
            o += row[j] * q[j];
        }
        out[i] = o;
    }
}

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double se = 0, sr = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        se += double(a[i] - b[i]) * (a[i] - b[i]);
        sr += double(b[i]) * b[i];
    }
    return std::sqrt(se / (sr + 1e-30));
}

int main() {
    std::printf("=== Gated DeltaNet recurrence ===\n\n");
    const int KD = 128, VD = 128;      // linear_key_head_dim, linear_value_head_dim
    int fails = 0;

    std::printf("  %-8s %14s %16s\n", "tokens", "row-wise", "missing-decay");
    for (int T : {1, 8, 64, 512, 4096}) {
        std::mt19937 rng(T * 7 + 1);
        std::normal_distribution<float> nd(0.0f, 1.0f);

        std::vector<float> Sref(size_t(VD) * KD, 0.0f);
        std::vector<float> Srow = Sref, Sbad = Sref;
        std::vector<float> oref(VD), orow(VD), obad(VD);

        double worst_row = 0, worst_bad = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<float> q(KD), k(KD), v(VD);
            for (auto& x : q) x = nd(rng);
            for (auto& x : k) x = nd(rng);
            for (auto& x : v) x = nd(rng);

            // q and k are L2-normalized in this architecture; without it
            // the delta rule is not contractive and the state diverges.
            float nq = 0, nk = 0;
            for (int j = 0; j < KD; ++j) { nq += q[j] * q[j]; nk += k[j] * k[j]; }
            nq = 1.0f / std::sqrt(nq + 1e-6f);
            nk = 1.0f / std::sqrt(nk + 1e-6f);
            for (int j = 0; j < KD; ++j) { q[j] *= nq; k[j] *= nk; }

            const float A_log = nd(rng) * 0.5f, dt = nd(rng), dt_bias = 0.1f;
            const float a = std::exp(-std::exp(A_log) * softplus(dt + dt_bias));
            const float b = sigmoidf(nd(rng));

            deltanet_ref(q.data(), k.data(), v.data(), a, b, Sref.data(), oref.data(), KD, VD);
            deltanet_rowwise(q.data(), k.data(), v.data(), a, b, Srow.data(), orow.data(), KD, VD);
            deltanet_missing_decay(q.data(), k.data(), v.data(), a, b, Sbad.data(), obad.data(), KD, VD);

            worst_row = std::max(worst_row, rel(orow, oref));
            worst_bad = std::max(worst_bad, rel(obad, oref));
        }
        std::printf("  %-8d %14.3e %16.3e %s\n", T, worst_row, worst_bad,
                    worst_row < 1e-5 ? "" : " <-- FAIL");
        if (!(worst_row < 1e-5)) ++fails;
    }

    // The state must stay bounded no matter how long the context runs.
    // If it grows, the model degrades silently at long context.
    {
        std::mt19937 rng(99);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> S(size_t(VD) * KD, 0.0f), o(VD);
        double first = 0, last = 0;
        for (int t = 0; t < 20000; ++t) {
            std::vector<float> q(KD), k(KD), v(VD);
            for (auto& x : q) x = nd(rng);
            for (auto& x : k) x = nd(rng);
            for (auto& x : v) x = nd(rng);
            float nq = 0, nk = 0;
            for (int j = 0; j < KD; ++j) { nq += q[j]*q[j]; nk += k[j]*k[j]; }
            nq = 1.0f/std::sqrt(nq+1e-6f); nk = 1.0f/std::sqrt(nk+1e-6f);
            for (int j = 0; j < KD; ++j) { q[j]*=nq; k[j]*=nk; }
            const float a = std::exp(-std::exp(0.0f) * softplus(nd(rng)));
            const float b = sigmoidf(nd(rng));
            deltanet_rowwise(q.data(), k.data(), v.data(), a, b, S.data(), o.data(), KD, VD);
            double n = 0;
            for (float x : S) n += double(x) * x;
            n = std::sqrt(n);
            if (t == 100)   first = n;
            if (t == 19999) last  = n;
        }
        std::printf("\n  state norm at t=100: %.3f   at t=20000: %.3f   ratio %.3f\n",
                    first, last, last / first);
        if (!(last < first * 3.0 && std::isfinite(last))) {
            std::printf("  FAIL: state is not bounded over long context\n");
            ++fails;
        } else {
            std::printf("  state stays bounded -- safe for 262144-token context\n");
        }
    }

    // Memory, the reason this architecture is worth the trouble.
    {
        const int nlin = 30, heads = 32;
        const double state_mb = double(heads) * KD * VD * 4 * nlin / 1048576.0;
        std::printf("\n  DeltaNet state : %.1f MiB total, constant in context length\n", state_mb);
        for (int ctx : {8192, 65536, 262144}) {
            // 10 full-attention layers, 2 kv heads, head_dim 256, fp16
            const double kv_mb = 2.0 * 10 * 2 * 256 * ctx * 2 / 1048576.0;
            std::printf("  KV cache @%6d ctx : %8.1f MiB  (10 full-attn layers only)\n",
                        ctx, kv_mb);
        }
    }

    std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
