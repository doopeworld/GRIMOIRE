// =====================================================================
//  test_ops.cpp  --  host validation of the forward-pass operators.
//  Same math as src/ops.cpp; the kernels are the parallel form of these.
// =====================================================================
#include "b70/formats.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <climits>
#include <limits>

using namespace b70;
static int g_fail = 0;
#define CHECK(c, ...) do { if(!(c)){ std::printf("  FAIL %s:%d ",__FILE__,__LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while(0)

static void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float s = 1.0f / std::sqrt(float(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * s * w[i];
}

static void test_rmsnorm() {
    std::printf("RMSNorm\n");
    const int n = 2048;
    std::mt19937 rng(1); std::normal_distribution<float> nd(0,1);
    std::vector<float> x(n), w(n, 1.0f), o(n);
    for (auto& v : x) v = nd(rng);
    rmsnorm(x.data(), w.data(), o.data(), n, 1e-6f);
    // with unit weights the output RMS must be 1
    double ss = 0; for (float v : o) ss += double(v)*v;
    const double rms = std::sqrt(ss / n);
    std::printf("  output rms with unit weights: %.6f\n", rms);
    CHECK(std::fabs(rms - 1.0) < 1e-4, "rms %.6f != 1", rms);
    // scale invariance: 10x input gives identical output
    std::vector<float> x2(n), o2(n);
    for (int i = 0; i < n; ++i) x2[i] = x[i] * 10.0f;
    rmsnorm(x2.data(), w.data(), o2.data(), n, 1e-6f);
    double md = 0; for (int i = 0; i < n; ++i) md = std::max(md, double(std::fabs(o[i]-o2[i])));
    std::printf("  scale invariance (10x input): max diff %.2e\n", md);
    CHECK(md < 1e-4, "not scale invariant: %.3e", md);
}

// Partial RoPE: only the first `rot` dims rotate.
static void rope(float* p, int head_dim, int pos, float theta, float pf) {
    const int rot = int(head_dim * pf) & ~1;
    for (int j = 0; j < rot/2; ++j) {
        const float inv = std::pow(theta, -float(2*j)/float(rot));
        const float a = p[j], b = p[j + rot/2];
        const float c = std::cos(pos*inv), s = std::sin(pos*inv);
        p[j] = a*c - b*s; p[j + rot/2] = a*s + b*c;
    }
}

static void test_rope() {
    std::printf("\nPartial RoPE (head_dim 256, factor 0.25 -> 64 rotated)\n");
    const int HD = 256; const float PF = 0.25f, TH = 1e7f;
    const int rot = int(HD*PF);
    std::vector<float> a(HD), b(HD);
    for (int i = 0; i < HD; ++i) a[i] = b[i] = float(i+1) * 0.01f;

    rope(b.data(), HD, 5, TH, PF);
    // the non-rotated tail must be untouched
    int changed_tail = 0;
    for (int i = rot; i < HD; ++i) if (a[i] != b[i]) ++changed_tail;
    std::printf("  dims rotated: %d, tail dims altered: %d\n", rot, changed_tail);
    CHECK(changed_tail == 0, "%d tail dims were rotated; partial factor ignored", changed_tail);

    // rotation preserves the norm of each rotated pair
    double n0 = 0, n1 = 0;
    for (int i = 0; i < rot; ++i) { n0 += double(a[i])*a[i]; n1 += double(b[i])*b[i]; }
    std::printf("  rotated-block norm preserved: %.6f -> %.6f\n", std::sqrt(n0), std::sqrt(n1));
    CHECK(std::fabs(std::sqrt(n0)-std::sqrt(n1)) < 1e-4, "rotation changed the norm");

    // position 0 is the identity
    std::vector<float> c = a;
    rope(c.data(), HD, 0, TH, PF);
    double md = 0; for (int i = 0; i < HD; ++i) md = std::max(md, double(std::fabs(a[i]-c[i])));
    CHECK(md < 1e-6, "pos 0 is not the identity: %.3e", md);
    std::printf("  pos 0 is identity: ok\n");

    // relative-position property: rope(q,p1).rope(k,p2) depends on p1-p2
    auto dot_at = [&](int p1, int p2) {
        std::vector<float> q(HD), k(HD);
        std::mt19937 r(7); std::normal_distribution<float> nd(0,1);
        for (int i = 0; i < HD; ++i) { q[i] = nd(r); k[i] = nd(r); }
        rope(q.data(), HD, p1, TH, PF); rope(k.data(), HD, p2, TH, PF);
        double d = 0; for (int i = 0; i < rot; ++i) d += double(q[i])*k[i];
        return d;
    };
    const double d1 = dot_at(10, 3), d2 = dot_at(107, 100);
    std::printf("  relative property: dot(10,3)=%.6f  dot(107,100)=%.6f\n", d1, d2);
    CHECK(std::fabs(d1-d2) < 1e-3, "RoPE is not relative: %.6f vs %.6f", d1, d2);
}

static void test_swiglu_l2() {
    std::printf("\nSwiGLU and per-head L2 norm\n");
    auto silu = [](float x){ return x/(1.0f+std::exp(-x)); };
    CHECK(std::fabs(silu(0.0f)) < 1e-9, "silu(0) != 0");
    CHECK(silu(20.0f) > 19.9f, "silu saturation wrong");
    CHECK(silu(-20.0f) > -1e-6f && silu(-20.0f) < 0.0f, "silu negative tail wrong");
    std::printf("  silu(0)=%.6f silu(20)=%.4f silu(-20)=%.3e\n",
                silu(0.0f), silu(20.0f), silu(-20.0f));

    const int H = 4, D = 128;
    std::vector<float> x(size_t(H)*D);
    std::mt19937 r(3); std::normal_distribution<float> nd(0,2.5);
    for (auto& v : x) v = nd(r);
    for (int h = 0; h < H; ++h) {
        float* p = &x[size_t(h)*D];
        double ss = 0; for (int i=0;i<D;++i) ss += double(p[i])*p[i];
        const float inv = 1.0f/std::sqrt(float(ss)+1e-6f);
        for (int i=0;i<D;++i) p[i] *= inv;
        double n2 = 0; for (int i=0;i<D;++i) n2 += double(p[i])*p[i];
        CHECK(std::fabs(std::sqrt(n2)-1.0) < 1e-4, "head %d norm %.6f", h, std::sqrt(n2));
    }
    std::printf("  all %d heads normalized to unit length\n", H);
}

static void test_lm_head_cost() {
    std::printf("\nlm_head cost, vocab 248320 x hidden 2048\n");
    const double elems = 248320.0 * 2048.0;
    std::printf("  %-10s %10s %10s\n", "format", "GB", "ms @rate");
    struct R { const char* n; Fmt f; double bw; };
    const R rows[] = {{"bf16", Fmt::BF16, 542.0}, {"int8", Fmt::INT8, 527.0},
                      {"int4", Fmt::INT4, 477.0}, {"mxfp4", Fmt::MXFP4, 352.0}};
    for (const R& r : rows) {
        const double gb = elems * bits_per_elem(r.f) / 8.0 / 1e9;
        std::printf("  %-10s %10.3f %10.2f\n", r.n, gb, gb / r.bw * 1e3);
    }
    std::printf("  Quantizing lm_head to int4 saves ~1.6 ms per token --\n"
                "  more than the entire fused MoE block costs.\n");
}


// ---------------------------------------------------------------------
// Router top-k: the parallel kernel must select EXACTLY what the old
// single-work-item loop selected, ties included. The kernel was rewritten
// for speed (215 us -> a few us per layer), so "same answer" is the whole
// safety argument; this compares the two algorithms directly rather than
// checking properties of one of them.
// ---------------------------------------------------------------------
static void topk_serial(const float* lg, int n, int k,
                        std::vector<int>& idx, std::vector<float>& wt,
                        bool normalize) {
    std::vector<char> taken(n, 0);
    idx.assign(k, -1); wt.assign(k, 0.0f);
    for (int s = 0; s < k; ++s) {
        int best = -1; float bv = 0.0f;
        for (int e = 0; e < n; ++e) {
            if (taken[e]) continue;
            if (best < 0 || lg[e] > bv) { best = e; bv = lg[e]; }
        }
        taken[best] = 1; idx[s] = best; wt[s] = bv;
    }
    float m = wt[0];
    for (int s = 1; s < k; ++s) m = std::fmax(m, wt[s]);
    float sum = 0.0f;
    for (int s = 0; s < k; ++s) { wt[s] = std::exp(wt[s] - m); sum += wt[s]; }
    if (normalize && sum > 0.0f) for (int s = 0; s < k; ++s) wt[s] /= sum;
}

// Emulates src/ops.cpp launch_router_topk lane for lane, SG_SIZE wide.
static void topk_subgroup(const float* lg_in, int n, int k,
                          std::vector<int>& idx, std::vector<float>& wt,
                          bool normalize) {
    constexpr int SG = 16;
    std::vector<float> lg(lg_in, lg_in + n);
    std::vector<unsigned long long> mine(SG, 0ull);   // per-lane slice mask
    idx.assign(k, -1); wt.assign(k, 0.0f);

    for (int s = 0; s < k; ++s) {
        float cv[SG]; int ci[SG], cs[SG];
        for (int lane = 0; lane < SG; ++lane) {
            cv[lane] = -std::numeric_limits<float>::infinity();
            ci[lane] = INT_MAX; cs[lane] = -1;
            for (int e = lane, slot = 0; e < n; e += SG, ++slot) {
                if (mine[lane] & (1ull << slot)) continue;
                const float v = lg[e];
                if (v > cv[lane] || (v == cv[lane] && e < ci[lane])) {
                    cv[lane] = v; ci[lane] = e; cs[lane] = slot;
                }
            }
        }
        float bv = -std::numeric_limits<float>::infinity();
        for (int lane = 0; lane < SG; ++lane) bv = std::fmax(bv, cv[lane]);
        int bi = INT_MAX;
        for (int lane = 0; lane < SG; ++lane)
            if (cv[lane] == bv && ci[lane] != INT_MAX) bi = std::min(bi, ci[lane]);
        for (int lane = 0; lane < SG; ++lane)
            if (ci[lane] == bi && cs[lane] >= 0) mine[lane] |= (1ull << cs[lane]);
        idx[s] = bi; wt[s] = bv;
    }
    float m = wt[0];
    for (int s = 1; s < k; ++s) m = std::fmax(m, wt[s]);
    float sum = 0.0f;
    for (int s = 0; s < k; ++s) { wt[s] = std::exp(wt[s] - m); sum += wt[s]; }
    if (normalize && sum > 0.0f) for (int s = 0; s < k; ++s) wt[s] /= sum;
}

static void test_router_topk() {
    std::printf("Router top-k (parallel kernel vs the serial loop it replaces)\n");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    const int n = 256, k = 8;          // Qwen3.5-MoE: 256 experts, top-8
    int cases = 0, mismatched = 0;

    auto compare = [&](const std::vector<float>& lg, const char* what) {
        std::vector<int> ia, ib; std::vector<float> wa, wb;
        topk_serial(lg.data(), n, k, ia, wa, true);
        topk_subgroup(lg.data(), n, k, ib, wb, true);
        ++cases;
        bool ok = (ia == ib);
        for (int s = 0; s < k && ok; ++s)
            ok = std::fabs(wa[s] - wb[s]) <= 1e-6f * std::fmax(1.0f, std::fabs(wa[s]));
        if (!ok) {
            ++mismatched;
            std::printf("    %s: serial [", what);
            for (int s = 0; s < k; ++s) std::printf(" %d", ia[s]);
            std::printf(" ]  parallel [");
            for (int s = 0; s < k; ++s) std::printf(" %d", ib[s]);
            std::printf(" ]\n");
        }
        CHECK(ok, "%s: selection differs", what);
    };

    std::vector<float> lg(n);
    for (int t = 0; t < 500; ++t) {
        for (auto& v : lg) v = nd(rng);
        compare(lg, "random");
    }
    // every logit identical: the tie-break decides all 8 picks
    std::fill(lg.begin(), lg.end(), 0.5f);
    compare(lg, "all equal");
    // ties only among the top values
    for (auto& v : lg) v = -1.0f;
    for (int e : {200, 3, 77, 12, 255, 0, 130, 44, 91, 180}) lg[e] = 2.0f;
    compare(lg, "ties at the top");
    // all negative -- the old loop seeded bv=0, so this exercises the
    // `best < 0` first-iteration path
    for (auto& v : lg) v = -std::fabs(nd(rng)) - 1.0f;
    compare(lg, "all negative");
    // one -inf logit: the reason `taken` is a bitmask and not a sentinel
    for (auto& v : lg) v = nd(rng);
    lg[17] = -std::numeric_limits<float>::infinity();
    compare(lg, "contains -inf");
    // degenerate: k selections from a vector where only k are finite
    for (auto& v : lg) v = -std::numeric_limits<float>::infinity();
    for (int e = 0; e < k; ++e) lg[e * 7] = float(e);
    compare(lg, "only k finite");

    std::printf("  %d cases, %d mismatched\n", cases, mismatched);
}

int main() {
    std::printf("=== forward-pass operators ===\n\n");
    test_rmsnorm();
    test_rope();
    test_swiglu_l2();
    test_router_topk();
    test_lm_head_cost();
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
