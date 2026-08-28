// =====================================================================
//  test_attention.cpp  --  proves the online-softmax recurrence used by
//  the FlashDecoding kernel, by emulating a 16-lane sub-group on the CPU
//  and comparing against a naive materialized softmax.
//
//  It also reproduces the blueprint's original recurrence so the failure
//  mode is visible rather than asserted.
// =====================================================================
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <limits>

static constexpr int SG = 16;    // Xe2 sub-group width we compile for

// ---------------------------------------------------------------------
// Reference: materialize all scores, subtract the max, normalize.
// ---------------------------------------------------------------------
static void attention_naive(const float* q, const float* K, const float* V,
                            float* out, int seq, int hd) {
    std::vector<double> s(seq);
    const double scale = 1.0 / std::sqrt(double(hd));
    for (int j = 0; j < seq; ++j) {
        double d = 0.0;
        for (int t = 0; t < hd; ++t) d += double(q[t]) * K[size_t(j) * hd + t];
        s[j] = d * scale;
    }
    double m = *std::max_element(s.begin(), s.end());
    double Z = 0.0;
    for (int j = 0; j < seq; ++j) { s[j] = std::exp(s[j] - m); Z += s[j]; }
    for (int t = 0; t < hd; ++t) {
        double a = 0.0;
        for (int j = 0; j < seq; ++j) a += s[j] * V[size_t(j) * hd + t];
        out[t] = float(a / Z);
    }
}

// ---------------------------------------------------------------------
// The corrected kernel, lane-for-lane.
//
// Each iteration the sub-group consumes SG key positions: lane L scores
// key s0+L, the running (m, l) statistics are reduced across the whole
// sub-group so they stay UNIFORM, and the output accumulator is
// partitioned across lanes (lane L owns dims L, L+SG, L+2*SG, ...).
//
// The two things the blueprint got wrong:
//   (a) when the running max moves, the *accumulator* must be rescaled
//       by exp(m_old - m_new), not just the denominator;
//   (b) the max must be reduced across lanes before use. Per-lane maxima
//       make the partial sums incommensurable, so the final
//       reduce_over_group adds numbers that were normalized differently.
// ---------------------------------------------------------------------
static void attention_online(const float* q, const float* K, const float* V,
                             float* out, int seq, int hd, bool buggy) {
    const float scale = 1.0f / std::sqrt(float(hd));
    const int   dpl   = hd / SG;              // accumulator dims per lane

    std::vector<float> m(SG, -std::numeric_limits<float>::infinity());
    std::vector<float> l(SG, 0.0f);
    std::vector<std::vector<float>> acc(SG, std::vector<float>(dpl, 0.0f));

    for (int s0 = 0; s0 < seq; s0 += SG) {
        // ---每 lane scores one key -----------------------------------
        float score[SG];
        for (int L = 0; L < SG; ++L) {
            const int j = s0 + L;
            if (j >= seq) { score[L] = -std::numeric_limits<float>::infinity(); continue; }
            float d = 0.0f;
            for (int t = 0; t < hd; ++t) d += q[t] * K[size_t(j) * hd + t];
            score[L] = d * scale;
        }

        if (buggy) {
            // ---- blueprint recurrence: per-lane max, no acc rescale ---
            for (int L = 0; L < SG; ++L) {
                const int j = s0 + L;
                if (j >= seq) continue;
                const float old = m[L];
                if (score[L] > m[L]) {
                    m[L] = score[L];
                    l[L] = l[L] * std::exp(old - m[L]) + std::exp(score[L] - m[L]);
                } else {
                    l[L] += std::exp(score[L] - m[L]);
                }
                const float p = std::exp(score[L] - m[L]);
                for (int d = 0; d < dpl; ++d)
                    acc[L][d] += p * V[size_t(j) * hd + (L + d * SG)];
            }
            continue;
        }

        // ---- corrected: sub-group reductions keep m and l uniform ----
        float mblk = -std::numeric_limits<float>::infinity();
        for (int L = 0; L < SG; ++L) mblk = std::max(mblk, score[L]);   // reduce_over_group(maximum)
        const float mnew = std::max(m[0], mblk);
        const float corr = (m[0] == -std::numeric_limits<float>::infinity())
                         ? 0.0f : std::exp(m[0] - mnew);

        float psum = 0.0f;
        float p[SG];
        for (int L = 0; L < SG; ++L) {
            p[L] = (score[L] == -std::numeric_limits<float>::infinity())
                 ? 0.0f : std::exp(score[L] - mnew);
            psum += p[L];                                                // reduce_over_group(plus)
        }

        for (int L = 0; L < SG; ++L) {
            l[L] = l[L] * corr + psum;                                   // uniform across lanes
            for (int d = 0; d < dpl; ++d) acc[L][d] *= corr;             // THE missing rescale
            // every lane walks all SG keys of this block, touching only
            // the output dims it owns; p[j] arrives by group_broadcast
            for (int j = 0; j < SG; ++j) {
                if (s0 + j >= seq) break;
                for (int d = 0; d < dpl; ++d)
                    acc[L][d] += p[j] * V[size_t(s0 + j) * hd + (L + d * SG)];
            }
            m[L] = mnew;
        }
    }

    for (int L = 0; L < SG; ++L)
        for (int d = 0; d < dpl; ++d)
            out[L + d * SG] = acc[L][d] / l[L];
}

// ---------------------------------------------------------------------
static double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        se += double(a[i] - b[i]) * (a[i] - b[i]);
        sr += double(b[i]) * b[i];
    }
    return std::sqrt(se / (sr + 1e-30));
}

int main() {
    std::printf("=== FlashDecoding online-softmax validation ===\n\n");
    const int hd = 128;
    int fails = 0;

    std::printf("  %-8s %-10s %14s %14s\n", "seq_len", "spread", "corrected", "blueprint");
    for (int seq : {16, 64, 129, 512, 1024, 4096}) {
        // "spread" widens the logit range, which is what exercises the
        // running-max path. Real models hit this the moment a strong
        // match appears late in a long context.
        for (float spread : {1.0f, 8.0f}) {
            std::mt19937 rng(seq * 31 + int(spread));
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> q(hd), K(size_t(seq) * hd), V(size_t(seq) * hd);
            for (auto& v : q) v = nd(rng) * spread;
            for (auto& v : K) v = nd(rng);
            for (auto& v : V) v = nd(rng);

            std::vector<float> ref(hd), got(hd), bad(hd);
            attention_naive(q.data(), K.data(), V.data(), ref.data(), seq, hd);
            attention_online(q.data(), K.data(), V.data(), got.data(), seq, hd, false);
            attention_online(q.data(), K.data(), V.data(), bad.data(), seq, hd, true);

            const double eg = rel_err(got, ref);
            const double eb = rel_err(bad, ref);
            std::printf("  %-8d %-10.1f %14.3e %14.3e %s\n",
                        seq, spread, eg, eb, eg < 1e-5 ? "" : "  <-- FAIL");
            if (!(eg < 1e-5)) ++fails;
        }
    }

    std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
