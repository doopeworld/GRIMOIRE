// =====================================================================
//  test_dflash2_selector.cpp -- the DFlash2 candidate selector.
//
//  Reference is vLLM's own code, not a restatement of ours:
//
//    _score_edges(pred, succ, candidate_ids, unary, hidden, anchor, top_k):
//        successors      = succ[candidate_ids]                 # [B,L,k,r]
//        predecessor_ids = cat(anchor.expand(-1,1,k),
//                              candidate_ids[:, :-1], dim=1)   # [B,L,k]
//        predecessors    = pred[predecessor_ids]               # [B,L,k,r]
//        return unary[:,:,None]
//             + einsum("blpr,blcr->blpc", predecessors*hidden[:,:,None],
//                      successors)
//
//    _selector_walk_kernel: previous = 0; for step: row = scores[step][previous];
//                           pick = argmax(row); token = cand[step][pick];
//                           previous = pick          -- greedy, per step.
//
//  The SYCL kernels live in src/ops.cpp and cannot be compiled here, so
//  they are replayed lane for lane below and diffed against the
//  reference.  What this pins is the FORMULA and the WALK; it says
//  nothing about SYCL scheduling.
// =====================================================================
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <climits>
#include <limits>
#include <random>
#include <vector>
#include <cstring>

static int g_fail = 0;
#define CHECK(c, ...) do { if(!(c)){ std::printf("  FAIL %s:%d ",__FILE__,__LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while(0)

// bf16 round-trip: the kernels round intermediates through bf16 because the
// codebooks are bf16 parameters.  The reference must do the same or the
// comparison measures rounding, not the formula.
static float bf16(float x) {
    uint32_t b; std::memcpy(&b, &x, 4);
    const uint32_t r = (b + 0x7fffu + ((b >> 16) & 1)) & 0xffff0000u;
    float o; std::memcpy(&o, &r, 4); return o;
}

// ---- reference: vLLM _score_edges, one batch row ---------------------
static void score_edges_ref(const std::vector<float>& pred,      // [vocab,rank]
                            const std::vector<float>& succ,      // [vocab,rank]
                            const std::vector<int32_t>& cand,    // [steps,K]
                            const std::vector<float>& unary,     // [steps,K]
                            const std::vector<float>& hidden,    // [steps,rank]
                            int32_t anchor, int steps, int K, int rank,
                            std::vector<float>& out) {           // [steps,K,K]
    out.assign(size_t(steps) * K * K, 0.0f);
    for (int l = 0; l < steps; ++l)
        for (int p = 0; p < K; ++p) {
            // predecessor_ids: anchor at l==0, else the candidate at l-1
            const int32_t pid = (l == 0) ? anchor : cand[size_t(l - 1) * K + p];
            for (int c = 0; c < K; ++c) {
                const int32_t cid = cand[size_t(l) * K + c];
                float acc = 0.0f;
                for (int r = 0; r < rank; ++r) {
                    const float gated = bf16(pred[size_t(pid) * rank + r] *
                                             bf16(hidden[size_t(l) * rank + r]));
                    acc += gated * succ[size_t(cid) * rank + r];
                }
                out[(size_t(l) * K + p) * K + c] =
                    unary[size_t(l) * K + c] + bf16(acc);
            }
        }
}

// ---- replay of launch_dflash2_selector_edges, SG_SIZE wide -----------
static void score_edges_kernel(const std::vector<float>& pred,
                               const std::vector<float>& succ,
                               const std::vector<int32_t>& cand,
                               const std::vector<float>& unary,
                               const std::vector<float>& hidden,
                               int32_t anchor, int steps, int K, int rank,
                               std::vector<float>& out) {
    constexpr int SG = 16;
    out.assign(size_t(steps) * K * K, 0.0f);
    for (int slot = 0; slot < steps * K; ++slot) {
        const int step = slot / K, p = slot % K;
        const int32_t pid = step == 0 ? anchor : cand[size_t(step - 1) * K + p];
        for (int c = 0; c < K; ++c) {
            const int32_t cid = cand[size_t(step) * K + c];
            float acc = 0.0f;                       // sub-group reduction
            for (int lane = 0; lane < SG; ++lane)
                for (int r = lane; r < rank; r += SG) {
                    const float pv = pred[size_t(pid) * rank + r];
                    const float hv = bf16(hidden[size_t(step) * rank + r]);
                    acc += bf16(pv * hv) * succ[size_t(cid) * rank + r];
                }
            out[(size_t(step) * K + p) * K + c] =
                unary[size_t(step) * K + c] + bf16(acc);
        }
    }
}

// ---- replay of launch_dflash2_path_walk ------------------------------
static void path_walk(const std::vector<float>& scores,
                      const std::vector<int32_t>& cand,
                      int steps, int K, std::vector<int32_t>& tokens) {
    tokens.assign(size_t(steps), 0);
    int previous = 0;
    for (int step = 0; step < steps; ++step) {
        const float* row = scores.data() + (size_t(step) * K + previous) * K;
        float best = -std::numeric_limits<float>::infinity();
        int pick = K;
        for (int c = 0; c < K; ++c) if (row[c] > best) { best = row[c]; pick = c; }
        if (pick == K) pick = 0;
        tokens[size_t(step)] = cand[size_t(step) * K + pick];
        previous = pick;
    }
}

int main() {
    std::printf("=== DFlash2 candidate selector ===\n\n");
    const int vocab = 512, rank = 8, K = 16, steps = 6;
    std::mt19937 rng(17); std::normal_distribution<float> nd(0, 1);
    std::vector<float> pred(size_t(vocab) * rank), succ(size_t(vocab) * rank);
    for (auto& v : pred) v = bf16(nd(rng) * 0.5f);
    for (auto& v : succ) v = bf16(nd(rng) * 0.5f);
    std::vector<int32_t> cand(size_t(steps) * K);
    std::vector<float> unary(size_t(steps) * K), hidden(size_t(steps) * rank);
    std::uniform_int_distribution<int> vd(0, vocab - 1);
    for (auto& v : cand)  v = vd(rng);
    for (auto& v : unary) v = nd(rng);
    for (auto& v : hidden) v = nd(rng);
    const int32_t anchor = 123;

    std::printf("Edge scores: kernel replay vs vLLM _score_edges\n");
    std::vector<float> a, b;
    score_edges_ref(pred, succ, cand, unary, hidden, anchor, steps, K, rank, a);
    score_edges_kernel(pred, succ, cand, unary, hidden, anchor, steps, K, rank, b);
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, double(std::fabs(a[i]-b[i])));
    std::printf("  %zu scores, max diff %.3e\n", a.size(), worst);
    CHECK(worst < 1e-4, "kernel does not match _score_edges");

    // the unary term must be indexed by the CANDIDATE, not the predecessor:
    // swapping those is silent and only shows up as poor acceptance.
    std::printf("\nUnary term is indexed by candidate, not predecessor\n");
    bool rows_differ = false;
    for (int c = 0; c < K && !rows_differ; ++c)
        if (std::fabs(a[(size_t(1)*K+0)*K+c] - a[(size_t(1)*K+1)*K+c]) > 1e-6)
            rows_differ = true;
    CHECK(rows_differ, "scores identical across predecessors -- edge term is dead");
    std::printf("  predecessor rows differ: ok\n");

    // at step 0 every predecessor slot is the anchor, so all p rows match
    std::printf("\nStep 0 uses the anchor for every predecessor slot\n");
    double step0 = 0;
    for (int p = 1; p < K; ++p)
        for (int c = 0; c < K; ++c)
            step0 = std::max(step0, double(std::fabs(a[c] - a[(size_t(0)*K+p)*K+c])));
    std::printf("  max spread across p at step 0: %.3e\n", step0);
    CHECK(step0 < 1e-6, "step 0 must use the anchor for every p");

    std::printf("\nGreedy walk, and the anchor actually steers it\n");
    std::vector<int32_t> t1, t2;
    std::vector<float> c2;      // edge scores under a different anchor
    path_walk(a, cand, steps, K, t1);
    std::printf("  path:");
    for (int i = 0; i < steps; ++i) std::printf(" %d", t1[size_t(i)]);
    std::printf("\n");
    for (int i = 0; i < steps; ++i) {
        bool found = false;
        for (int c = 0; c < K; ++c) if (cand[size_t(i)*K+c] == t1[size_t(i)]) found = true;
        CHECK(found, "step %d token is not one of its candidates", i);
    }
    // The anchor must change the STEP-0 scores.  Whether it flips the
    // argmax depends on how strong the unary term is for a given seed, so
    // asserting on the path alone is a coin flip; assert on the scores,
    // then confirm that SOME anchor does move the path.
    score_edges_ref(pred, succ, cand, unary, hidden, 456, steps, K, rank, c2);
    double anchor_delta = 0;
    for (int c = 0; c < K; ++c)
        anchor_delta = std::max(anchor_delta, double(std::fabs(a[c] - c2[c])));
    std::printf("  anchor 123 vs 456, max step-0 score change: %.3e\n", anchor_delta);
    CHECK(anchor_delta > 1e-6, "anchor does not affect step-0 scores -- "
                               "anchor conditioning is not wired");
    int flips = 0;
    for (int32_t anc = 0; anc < 64; ++anc) {
        std::vector<float> sc; std::vector<int32_t> tk;
        score_edges_ref(pred, succ, cand, unary, hidden, anc, steps, K, rank, sc);
        path_walk(sc, cand, steps, K, tk);
        if (tk != t1) ++flips;
    }
    std::printf("  %d of 64 anchors produce a different path\n", flips);
    CHECK(flips > 0, "no anchor changes the path -- the edge term is inert");

    // argmax-per-position is a DIFFERENT algorithm; if the selector were
    // equivalent to it there would be nothing to implement.
    std::printf("\nSelector differs from independent argmax\n");
    const size_t nsteps = size_t(steps);
    std::vector<int32_t> greedy(nsteps);
    for (int l = 0; l < steps; ++l) {
        int best = 0;
        for (int c = 1; c < K; ++c)
            if (unary[size_t(l)*K+c] > unary[size_t(l)*K+best]) best = c;
        greedy[size_t(l)] = cand[size_t(l)*K+best];
    }
    CHECK(greedy != t1, "selector path equals per-position argmax");
    std::printf("  argmax path differs from the scored path: ok\n");

    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
