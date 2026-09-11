// =====================================================================
//  test_k2_horizon.cpp -- host validation of the K2-Horizon operators.
//  Each case pins a property taken from modeling_k2_horizon.py, not a
//  golden number produced by this same code.
// =====================================================================
#include "b70/k2_horizon.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace b70;
static int g_fail = 0;
#define CHECK(c, ...) do { if(!(c)){ std::printf("  FAIL %s:%d ",__FILE__,__LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while(0)

// the standard whole-row RMSNorm every other model uses
static void rmsnorm_plain(const float* x, const float* w, float* o, int n, float eps) {
    double ss = 0; for (int i=0;i<n;++i) ss += double(x[i])*x[i];
    const float s = 1.0f/std::sqrt(float(ss/n)+eps);
    for (int i=0;i<n;++i) o[i] = x[i]*s*w[i];
}

static void test_grouped_rmsnorm() {
    std::printf("Grouped RMSNorm (layernorm_num_groups=2)\n");
    const int n = 2560, eps_groups = 2;          // the real hidden size
    std::mt19937 rng(7); std::normal_distribution<float> nd(0,1);
    std::vector<float> x(n), w(n,1.0f), og(n), op(n);
    for (auto& v : x) v = nd(rng);

    // n_groups == 1 must be bit-identical to the existing kernel
    k2::rmsnorm_grouped(x.data(), w.data(), og.data(), n, 1, 1e-6f);
    rmsnorm_plain(x.data(), w.data(), op.data(), n, 1e-6f);
    double md = 0; for (int i=0;i<n;++i) md = std::max(md, double(std::fabs(og[i]-op[i])));
    std::printf("  n_groups=1 vs plain RMSNorm: max diff %.3e\n", md);
    CHECK(md < 1e-6, "n_groups=1 must degenerate to plain RMSNorm");

    // each group independently normalized -> each half has unit RMS
    k2::rmsnorm_grouped(x.data(), w.data(), og.data(), n, eps_groups, 1e-6f);
    for (int g = 0; g < eps_groups; ++g) {
        double ss = 0; const int gn = n/eps_groups;
        for (int i=0;i<gn;++i) { const float v = og[size_t(g)*gn+i]; ss += double(v)*v; }
        const double rms = std::sqrt(ss/gn);
        std::printf("  group %d output rms: %.6f\n", g, rms);
        CHECK(std::fabs(rms-1.0) < 1e-4, "group %d rms %.6f != 1", g, rms);
    }

    // THE distinguishing property: scaling one group must not touch the
    // other.  Plain RMSNorm couples them through a shared variance; if a
    // kernel regresses to whole-row norm this is the case that catches it.
    std::vector<float> x2 = x, g2(n), p2(n);
    for (int i = 0; i < n/2; ++i) x2[size_t(i)] *= 10.0f;
    k2::rmsnorm_grouped(x2.data(), w.data(), g2.data(), n, 2, 1e-6f);
    double far = 0, plain_far = 0;
    for (int i = n/2; i < n; ++i) far = std::max(far, double(std::fabs(g2[size_t(i)]-og[size_t(i)])));
    rmsnorm_plain(x.data(),  w.data(), op.data(), n, 1e-6f);
    rmsnorm_plain(x2.data(), w.data(), p2.data(), n, 1e-6f);
    for (int i = n/2; i < n; ++i) plain_far = std::max(plain_far, double(std::fabs(p2[size_t(i)]-op[size_t(i)])));
    std::printf("  10x on group 0 -> group 1 moves by %.3e (plain RMSNorm: %.3e)\n", far, plain_far);
    CHECK(far < 1e-5, "groups are coupled -- this is whole-row norm, not grouped");
    CHECK(plain_far > 1e-2, "sanity: plain RMSNorm should couple them");
}

static void test_softplus_gate() {
    std::printf("\nSoftplus attention gate (beta=log 2)\n");
    const float beta = std::log(2.0f);
    // beta=log2 makes softplus exactly log2(1 + 2^x)
    double worst = 0;
    for (float x = -12.0f; x <= 12.0f; x += 0.25f) {
        const float got  = k2::softplus(x, beta);
        const float want = std::log2(1.0f + std::exp2(x));
        worst = std::max(worst, double(std::fabs(got-want)));
    }
    std::printf("  vs closed form log2(1+2^x) over [-12,12]: max diff %.3e\n", worst);
    CHECK(worst < 1e-5, "softplus(beta=log2) != log2(1+2^x)");

    std::printf("  softplus(0) = %.6f (must be 1)\n", k2::softplus(0.0f, beta));
    CHECK(std::fabs(k2::softplus(0.0f,beta)-1.0f) < 1e-6, "softplus(0) != 1");
    // strictly positive: the gate can never flip the sign of attn_output
    bool positive = true;
    for (float x = -60.0f; x <= 60.0f; x += 0.5f) positive &= k2::softplus(x,beta) > 0.0f;
    CHECK(positive, "gate must stay positive");
    // torch's linear region above threshold=20, and no overflow past it
    const float big = 100.0f;
    std::printf("  softplus(100) = %.4f (linear region, must be ~100)\n", k2::softplus(big,beta));
    CHECK(std::fabs(k2::softplus(big,beta)-big) < 1e-3, "threshold fallback wrong");
    CHECK(std::isfinite(k2::softplus(1e4f,beta)), "overflowed instead of going linear");
}

static void test_router() {
    std::printf("\nSigmoid router, bias steers SELECTION ONLY\n");
    const int n = 4, k = 2;
    const float logits[n] = {0.0f, 1.0f, 2.0f, 3.0f};
    auto sig = [](float v){ return 1.0f/(1.0f+std::exp(-v)); };

    // no bias: top-2 are the two largest logits, experts 3 and 2
    int idx[k]; float w[k];
    k2::router_topk(logits, nullptr, n, k, true, true, 1.0f, idx, w);
    std::printf("  no bias      -> experts %d,%d\n", idx[0], idx[1]);
    CHECK(idx[0]==3 && idx[1]==2, "unbiased top-2 wrong");

    // bias pulls expert 0 to the front, but its WEIGHT must still be
    // sigmoid(0.0)=0.5 -- unbiased.  Adding the bias into the weight is
    // the silent bug this pins down.
    const float bias[n] = {10.0f, 0.0f, 0.0f, 0.0f};
    k2::router_topk(logits, bias, n, k, true, false, 1.0f, idx, w);
    std::printf("  bias[0]=10   -> experts %d,%d  weights %.6f,%.6f\n",
                idx[0], idx[1], w[0], w[1]);
    CHECK(idx[0]==0 && idx[1]==3, "bias did not steer selection");
    CHECK(std::fabs(w[0]-sig(0.0f)) < 1e-6, "weight %.6f carries the bias; must be sigmoid(logit)=%.6f",
          w[0], sig(0.0f));
    CHECK(std::fabs(w[1]-sig(3.0f)) < 1e-6, "second weight wrong");

    // normalize + router_scaling_factor 2.5: weights must sum to 2.5
    k2::router_topk(logits, bias, n, k, true, true, 2.5f, idx, w);
    const double sum = double(w[0]) + w[1];
    std::printf("  normalized x2.5 -> %.6f + %.6f = %.6f\n", w[0], w[1], sum);
    CHECK(std::fabs(sum-2.5) < 1e-5, "normalized weights must sum to the scaling factor");
    // and the ratio must still be the unbiased one
    const double want_ratio = double(sig(0.0f))/sig(3.0f);
    CHECK(std::fabs(double(w[0])/w[1] - want_ratio) < 1e-5, "ratio distorted by bias");

    // softmax branch still available (score_func="softmax")
    k2::router_topk(logits, nullptr, n, k, false, true, 1.0f, idx, w);
    CHECK(std::fabs(double(w[0])+w[1]-1.0) < 1e-5, "softmax+normalize must sum to 1");

    // MoVA geometry: 64 experts, top-4, every route distinct and in range
    std::mt19937 rng(3); std::normal_distribution<float> nd(0,1);
    std::vector<float> lg(64), bs(64);
    int bad = 0, dup = 0;
    for (int t = 0; t < 500; ++t) {
        for (int e=0;e<64;++e){ lg[size_t(e)]=nd(rng); bs[size_t(e)]=nd(rng)*0.1f; }
        int mi[4]; float mw[4];
        k2::router_topk(lg.data(), bs.data(), 64, 4, true, true, 2.5f, mi, mw);
        for (int a=0;a<4;++a){
            if (mi[a]<0||mi[a]>=64) ++bad;
            for (int b=0;b<a;++b) if (mi[a]==mi[b]) ++dup;
        }
    }
    std::printf("  MoVA 64x top-4, 500 tokens: %d out-of-range, %d duplicate\n", bad, dup);
    CHECK(bad==0 && dup==0, "MoVA routing produced invalid routes");
}

static void test_mova_combine() {
    std::printf("\nMoVA value mixture (SiLU on the EXPERT OUTPUT)\n");
    const int dim = 8, k = 2;
    std::vector<float> e0(dim), e1(dim), out(dim);
    for (int i=0;i<dim;++i){ e0[size_t(i)] = float(i)-3.0f; e1[size_t(i)] = 2.0f-float(i); }
    const float* ep[k] = { e0.data(), e1.data() };
    const float w[k] = { 0.75f, 0.25f };
    k2::mova_combine(ep, w, k, dim, out.data());
    double worst = 0;
    for (int i=0;i<dim;++i) {
        const double want = 0.75*k2::silu(e0[size_t(i)]) + 0.25*k2::silu(e1[size_t(i)]);
        worst = std::max(worst, std::fabs(out[size_t(i)]-want));
    }
    std::printf("  vs w*silu(expert): max diff %.3e\n", worst);
    CHECK(worst < 1e-6, "mixture is not w*silu(expert_out)");
    // SiLU before the weight, not after: these differ, and swapping them
    // is invisible unless the weights are asymmetric like they are here.
    const double swapped = k2::silu(0.75f*e0[0]) + k2::silu(0.25f*e1[0]);
    CHECK(std::fabs(out[0]-swapped) > 1e-3, "sanity: silu/weight order must matter here");
}

static void test_layer_classification() {
    std::printf("\nLayer classification (mlp_only_layers=[0,1,2], step=1)\n");
    const int mlp_only[3] = {0,1,2};
    int dense = 0, sparse = 0;
    for (int i = 0; i < 48; ++i)
        (k2::is_sparse_layer(i, mlp_only, 3, 100, 1) ? sparse : dense)++;
    std::printf("  48 layers -> %d dense, %d sparse\n", dense, sparse);
    CHECK(dense == 3 && sparse == 45, "expected 3 dense + 45 sparse, got %d/%d", dense, sparse);
    for (int i = 0; i < 3; ++i)
        CHECK(!k2::is_sparse_layer(i, mlp_only, 3, 100, 1), "layer %d must be dense", i);
    CHECK(k2::is_sparse_layer(3, mlp_only, 3, 100, 1), "layer 3 must be sparse");
    CHECK(k2::is_sparse_layer(47, mlp_only, 3, 100, 1), "layer 47 must be sparse");
    // a dense model (num_experts=0) has no sparse layer at all
    CHECK(!k2::is_sparse_layer(9, mlp_only, 3, 0, 1), "num_experts=0 must disable sparsity");
    // decoder_sparse_step=2 would make every other layer sparse
    CHECK(k2::is_sparse_layer(3, mlp_only, 3, 100, 2), "step=2: layer 3 -> (3+1)%%2==0, sparse");
    CHECK(k2::is_sparse_layer(4, mlp_only, 3, 100, 2) == false, "step=2: layer 4 -> (5)%%2!=0 dense");
}

int main() {
    std::printf("=== K2-Horizon operators ===\n\n");
    test_grouped_rmsnorm();
    test_softplus_gate();
    test_router();
    test_mova_combine();
    test_layer_classification();
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
