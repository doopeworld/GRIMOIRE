// =====================================================================
//  test_moe.cpp  --  validates the fused grouped-expert layout.
//
//  The risk being tested: in the packed layout an expert's gate, up and
//  down rows are found by arithmetic on the expert id, not by a pointer
//  table. An off-by-one in gate_row/up_row/down_row does not crash and
//  does not produce obviously wrong numbers -- it silently computes
//  with the neighbouring expert's weights. The model would still emit
//  fluent text, just subtly wrong text. So it gets checked against a
//  naive per-expert-tensor reference.
// =====================================================================
#include "b70/moe.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

using namespace b70;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    std::printf("  FAIL %s:%d  ", __FILE__, __LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

static inline float silu_h(float v) { return v / (1.0f + std::exp(-v)); }

// ---------------------------------------------------------------------
// Independent reference: experts held as SEPARATE fp32 tensors, the way
// they come out of a checkpoint. No packing, no offset arithmetic.
// ---------------------------------------------------------------------
struct LooseExpert {
    std::vector<float> gate, up, down;   // [I][H], [I][H], [H][I]
};

static void loose_forward(const std::vector<LooseExpert>& ex, const Routing& r,
                          int H, int I, const float* x, float* y) {
    for (int o = 0; o < H; ++o) y[o] = 0.0f;
    std::vector<float> h(I);
    for (int j = 0; j < r.k; ++j) {
        const LooseExpert& E = ex[r.expert[j]];
        for (int i = 0; i < I; ++i) {
            double g = 0.0, u = 0.0;
            for (int c = 0; c < H; ++c) {
                g += double(E.gate[size_t(i) * H + c]) * x[c];
                u += double(E.up  [size_t(i) * H + c]) * x[c];
            }
            h[i] = silu_h(float(g)) * float(u);
        }
        for (int o = 0; o < H; ++o) {
            double a = 0.0;
            for (int i = 0; i < I; ++i) a += double(E.down[size_t(o) * I + i]) * h[i];
            y[o] += r.weight[j] * float(a);
        }
    }
}

// ---------------------------------------------------------------------
static void test_layout() {
    // Small but structurally identical to the real thing: gate/up
    // interleaved per expert, down expert-major.
    const int H = 64, I = 32, E = 16, K = 4;
    std::printf("Packed expert-major layout vs loose per-expert tensors\n");
    std::printf("  H=%d I=%d experts=%d top_k=%d\n", H, I, E, K);

    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    // Round the source weights to bf16 up front. The packed side is
    // stored as bf16, so without this we would be measuring bf16
    // rounding (~3e-3) instead of whether the offset arithmetic is
    // right. With it, any disagreement is a real layout bug.
    auto gen = [&]() { return bf16_to_f32(f32_to_bf16(nd(rng))); };

    std::vector<LooseExpert> loose(E);
    // Packed: gate_up is [E * 2I][H], down is [E * H][I]
    std::vector<float> pack_gu(size_t(E) * 2 * I * H);
    std::vector<float> pack_dn(size_t(E) * H * I);

    for (int e = 0; e < E; ++e) {
        loose[e].gate.resize(size_t(I) * H);
        loose[e].up.resize(size_t(I) * H);
        loose[e].down.resize(size_t(H) * I);
        for (auto& v : loose[e].gate) v = gen();
        for (auto& v : loose[e].up)   v = gen();
        for (auto& v : loose[e].down) v = gen();

        // Pack exactly as the converter will.
        for (int i = 0; i < I; ++i)
            for (int c = 0; c < H; ++c) {
                pack_gu[(size_t(e) * 2 * I + i) * H + c]     = loose[e].gate[size_t(i) * H + c];
                pack_gu[(size_t(e) * 2 * I + I + i) * H + c] = loose[e].up  [size_t(i) * H + c];
            }
        for (int o = 0; o < H; ++o)
            for (int i = 0; i < I; ++i)
                pack_dn[(size_t(e) * H + o) * I + i] = loose[e].down[size_t(o) * I + i];
    }

    // Wrap the packed buffers as bf16 QuantWeights, which is what the
    // kernel actually indexes.
    MoeLayer L;
    L.cfg.hidden = H; L.cfg.inter = I; L.cfg.num_experts = E; L.cfg.top_k = K;
    L.cfg.shared_inter = 0;
    L.has_shared = false;

    PackedWeight pgu = quantize(pack_gu.data(), E * 2 * I, H, Fmt::BF16);
    PackedWeight pdn = quantize(pack_dn.data(), E * H, I, Fmt::BF16);
    L.gate_up = pgu.view();
    L.down    = pdn.view();

    // Verify the offset arithmetic lands on the right rows.
    for (int e = 0; e < E; ++e) {
        CHECK(L.gate_row(e) == int64_t(e) * 2 * I, "gate_row(%d)", e);
        CHECK(L.up_row(e)   == int64_t(e) * 2 * I + I, "up_row(%d)", e);
        CHECK(L.down_row(e) == int64_t(e) * H, "down_row(%d)", e);
        // spot-check that the packed value at that offset really is
        // this expert's weight, not a neighbour's
        const float a = L.gate_up.at(int(L.gate_row(e)) + 1, 3);
        const float b = loose[e].gate[size_t(1) * H + 3];
        CHECK(a == b, "gate mismatch expert %d: %g vs %g", e, a, b);
        const float c = L.gate_up.at(int(L.up_row(e)) + 2, 5);
        const float d = loose[e].up[size_t(2) * H + 5];
        CHECK(c == d, "up mismatch expert %d: %g vs %g", e, c, d);
        const float f = L.down.at(int(L.down_row(e)) + 4, 6);
        const float g = loose[e].down[size_t(4) * I + 6];
        CHECK(f == g, "down mismatch expert %d: %g vs %g", e, f, g);
    }

    // Full forward, several random routings including adversarial ones.
    std::vector<float> x(H);
    for (auto& v : x) v = nd(rng) * 4.0f;

    double worst = 0.0;
    for (int trial = 0; trial < 8; ++trial) {
        Routing r;
        r.k = K;
        std::vector<int> perm(E);
        for (int e = 0; e < E; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        float ws = 0.0f;
        for (int j = 0; j < K; ++j) {
            r.expert[j] = perm[j];
            r.weight[j] = std::fabs(nd(rng)) + 0.05f;
            ws += r.weight[j];
        }
        for (int j = 0; j < K; ++j) r.weight[j] /= ws;

        std::vector<float> y_pack(H), y_loose(H);
        moe_forward_ref(L, r, x.data(), y_pack.data());
        loose_forward(loose, r, H, I, x.data(), y_loose.data());

        double se = 0, sr = 0;
        for (int o = 0; o < H; ++o) {
            se += double(y_pack[o] - y_loose[o]) * (y_pack[o] - y_loose[o]);
            sr += double(y_loose[o]) * y_loose[o];
        }
        worst = std::max(worst, std::sqrt(se / sr));
    }
    std::printf("  worst rel error over 8 random routings: %.3e\n", worst);
    CHECK(worst < 1e-6, "packed layout disagrees with loose reference: %.3e", worst);

    // Adversarial: route to expert 0 and the last expert, the two most
    // likely to expose an offset bug.
    {
        Routing r; r.k = 2;
        r.expert[0] = 0;     r.weight[0] = 0.5f;
        r.expert[1] = E - 1; r.weight[1] = 0.5f;
        std::vector<float> a(H), b(H);
        moe_forward_ref(L, r, x.data(), a.data());
        loose_forward(loose, r, H, I, x.data(), b.data());
        double se = 0, sr = 0;
        for (int o = 0; o < H; ++o) {
            se += double(a[o] - b[o]) * (a[o] - b[o]);
            sr += double(b[o]) * b[o];
        }
        CHECK(std::sqrt(se / sr) < 1e-6, "boundary experts mismatch");
        std::printf("  boundary experts (0 and %d) correct\n", E - 1);
    }
}

// ---------------------------------------------------------------------
static void test_routing() {
    std::printf("\nRouter top-k selection\n");
    const int E = 256, K = 8;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> logits(E);
    for (auto& v : logits) v = nd(rng);

    Routing r;
    route_ref(logits.data(), E, K, true, r);
    CHECK(r.k == K, "k wrong");

    // selected must be the K largest logits
    std::vector<float> sorted = logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    for (int j = 0; j < K; ++j)
        CHECK(logits[r.expert[j]] >= sorted[K - 1] - 1e-9f,
              "slot %d picked a non-top-k expert", j);

    // no duplicates
    for (int a = 0; a < K; ++a)
        for (int b = a + 1; b < K; ++b)
            CHECK(r.expert[a] != r.expert[b], "duplicate expert %d", r.expert[a]);

    // renormalized weights sum to 1
    float s = 0.0f;
    for (int j = 0; j < K; ++j) s += r.weight[j];
    CHECK(std::fabs(s - 1.0f) < 1e-5f, "weights sum to %g", s);

    // descending
    for (int j = 1; j < K; ++j)
        CHECK(r.weight[j] <= r.weight[j - 1] + 1e-9f, "weights not descending");
    std::printf("  top-8 of 256: no duplicates, weights sum to %.6f\n", s);
}

// ---------------------------------------------------------------------
// The whole point of the fused kernel, in numbers.
// ---------------------------------------------------------------------
static void test_budget() {
    std::printf("\nQwen3.5-35B-A3B decode budget, per token\n");
    MoeConfig c;   // real config: 2048 / 512 / 256 experts / top-8
    const int layers = 40;

    const long long naive  = 1LL * layers * c.top_k * 3;   // one launch per expert per matrix
    const long long fused  = 1LL * layers * 2;
    const double    us     = 5.0;                          // Level Zero submit, optimistic

    std::printf("  launches/token   naive %lld   fused %lld   (%.0fx fewer)\n",
                naive, fused, double(naive) / double(fused));
    std::printf("  launch overhead  naive %.2f ms   fused %.2f ms\n",
                naive * us / 1000.0, fused * us / 1000.0);

    // bytes actually streamed
    for (Fmt f : { Fmt::MXFP4, Fmt::INT4, Fmt::MXFP8 }) {
        const double bpe = bits_per_elem(f) / 8.0;
        double gb = double(layers) * double(c.top_k) * 3.0
                  * double(c.hidden) * double(c.inter) * bpe / 1e9;
        gb += double(layers) * 3.0 * double(c.hidden) * double(c.shared_inter) * bpe / 1e9;
        const double ms_roof = gb / 608.0 * 1000.0;
        std::printf("  %-8s MoE weights %.2f GB -> %.2f ms roofline, %.0f tok/s\n",
                    fmt_name(f), gb, ms_roof, 1000.0 / ms_roof);
        std::printf("           with naive launch overhead: %.2f ms -> %.0f tok/s  (%.0f%% of roofline)\n",
                    ms_roof + naive * us / 1000.0,
                    1000.0 / (ms_roof + naive * us / 1000.0),
                    100.0 * ms_roof / (ms_roof + naive * us / 1000.0));
        CHECK(ms_roof > 0, "budget sanity");
    }
}

int main() {
    std::printf("=== fused grouped-expert MoE tests ===\n\n");
    test_layout();
    test_routing();
    test_budget();
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
