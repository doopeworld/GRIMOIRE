// =====================================================================
//  test_k2_kernels_device.cpp -- RUN the K2 and DFlash2 selector kernels
//  on a real SYCL device and diff them against the host references.
//
//  tests/test_k2_horizon.cpp and tests/test_dflash2_selector.cpp pin the
//  MATH by re-implementing each kernel on the host.  That catches a wrong
//  formula and nothing else: a kernel that never ran cannot fail them.
//  This tool submits the actual kernels from src/ops.cpp, so it also
//  covers the parts a host replay cannot -- sub-group reductions, the
//  group_barrier placement in the grouped norm (which has no host
//  analogue at all), the private top-k mask, and the padded-row stride
//  in topk16_rows.
//
//  It needs a SYCL device, any SYCL device.  The B70 is the one that
//  matters, but an OpenCL CPU device exercises the same kernel source and
//  is what makes this runnable off the Tower.  Per rule 8 this is still
//  not a substitute for generating text on the card.
//
//  Build (see the tail of build_b70.sh):
//    icpx -fsycl -fsycl-targets=spir64 -O2 -std=c++20 -I include -I src \
//         tools/test_k2_kernels_device.cpp src/ops.cpp -o bin/test_k2_kernels
// =====================================================================
#include "kernels.hpp"
#include "b70/k2_horizon.hpp"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

using namespace b70;

// Declared here exactly as grimoire.cpp declares it: the decode-path
// RMSNorm lives in ops.cpp and is not in kernels.hpp.
namespace b70 {
sycl::event launch_rmsnorm_residual(sycl::queue&, float*, const float*, const bf16_t*,
                                    float*, int, float, const std::vector<sycl::event>&);
}

static int g_fail = 0;
#define CHECK(c, ...) do { if(!(c)){ std::printf("  FAIL %s:%d ",__FILE__,__LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while(0)

static double worst_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double w = 0;
    for (size_t i = 0; i < a.size(); ++i) w = std::max(w, double(std::fabs(a[i] - b[i])));
    return w;
}

// bf16 round-trip, matching the selector test's reference
static float bf16r(float x) {
    uint32_t b; std::memcpy(&b, &x, 4);
    const uint32_t r = (b + 0x7fffu + ((b >> 16) & 1)) & 0xffff0000u;
    float o; std::memcpy(&o, &r, 4); return o;
}

// =====================================================================
int main() {
    sycl::queue q{sycl::property::queue::in_order()};
    std::printf("=== K2 / DFlash2 kernels on a real device ===\n");
    std::printf("device: %s\n\n",
        q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(20260912);
    std::normal_distribution<float> nd(0.f, 1.f);

    // ---- 1. grouped RMSNorm -----------------------------------------
    // Both conventions are covered: weight_offset 0 is K2 (weight applied
    // directly) and 1 is every other model ((1 + w)).  Getting that wrong
    // scales every norm output by ~10x and is otherwise silent.
    for (int n_groups : {1, 2, 4})
    for (float woff : {0.0f, 1.0f}) {
        const int tokens = 7, hidden = 512;
        std::vector<float> h(size_t(tokens) * hidden), r0(h.size()), r1(h.size());
        std::vector<bf16_t> w(static_cast<size_t>(hidden));
        for (auto& v : h)  v = nd(rng);
        for (auto& v : r0) v = nd(rng) * 0.25f;
        for (auto& v : r1) v = nd(rng) * 0.25f;
        for (auto& v : w)  v = f32_to_bf16(nd(rng) * 0.1f);

        float* d_h  = sycl::malloc_device<float>(h.size(), q);
        float* d_r0 = sycl::malloc_device<float>(h.size(), q);
        float* d_r1 = sycl::malloc_device<float>(h.size(), q);
        float* d_o  = sycl::malloc_device<float>(h.size(), q);
        bf16_t* d_w = sycl::malloc_device<bf16_t>(w.size(), q);
        q.memcpy(d_h, h.data(), h.size() * 4);
        q.memcpy(d_r0, r0.data(), r0.size() * 4);
        q.memcpy(d_r1, r1.data(), r1.size() * 4);
        q.memcpy(d_w, w.data(), w.size() * sizeof(bf16_t)).wait();

        launch_rmsnorm_grouped(q, d_h, d_r0, d_r1, d_w, d_o, nullptr,
                               tokens, hidden, n_groups, 1e-6f, woff).wait();

        std::vector<float> got(h.size()), got_h(h.size());
        q.memcpy(got.data(), d_o, got.size() * 4);
        q.memcpy(got_h.data(), d_h, got_h.size() * 4).wait();

        // reference: residuals fold into h first, then grouped RMSNorm
        std::vector<float> want(h.size()), want_h(h.size()), wf(static_cast<size_t>(hidden));
        for (size_t i = 0; i < h.size(); ++i) want_h[i] = h[i] + r0[i] + r1[i];
        for (int i = 0; i < hidden; ++i) wf[size_t(i)] = woff + bf16_to_f32(w[size_t(i)]);
        for (int t = 0; t < tokens; ++t)
            k2::rmsnorm_grouped(want_h.data() + size_t(t) * hidden, wf.data(),
                                want.data() + size_t(t) * hidden,
                                hidden, n_groups, 1e-6f);

        const double dn = worst_abs(want, got), dh = worst_abs(want_h, got_h);
        std::printf("grouped rmsnorm  groups=%d offset=%.0f  out %.3e  residual %.3e\n",
                    n_groups, double(woff), dn, dh);
        CHECK(dn < 2e-5, "grouped rmsnorm output mismatch");
        CHECK(dh < 1e-6, "residual accumulation into h mismatch");
        sycl::free(d_h,q); sycl::free(d_r0,q); sycl::free(d_r1,q);
        sycl::free(d_o,q); sycl::free(d_w,q);
    }

    // no-residual form: h must be read, not written
    {
        const int tokens = 3, hidden = 256, n_groups = 2;
        std::vector<float> h(size_t(tokens) * hidden);
        std::vector<bf16_t> w(static_cast<size_t>(hidden));
        for (auto& v : h) v = nd(rng);
        for (auto& v : w) v = f32_to_bf16(nd(rng) * 0.1f);
        float* d_h = sycl::malloc_device<float>(h.size(), q);
        float* d_o = sycl::malloc_device<float>(h.size(), q);
        bf16_t* d_w = sycl::malloc_device<bf16_t>(w.size(), q);
        q.memcpy(d_h, h.data(), h.size() * 4);
        q.memcpy(d_w, w.data(), w.size() * sizeof(bf16_t)).wait();
        launch_rmsnorm_grouped(q, d_h, nullptr, nullptr, d_w, d_o, nullptr,
                               tokens, hidden, n_groups, 1e-6f, 0.0f).wait();
        std::vector<float> got(h.size()), back(h.size());
        q.memcpy(got.data(), d_o, got.size() * 4);
        q.memcpy(back.data(), d_h, back.size() * 4).wait();
        std::vector<float> want(h.size()), wf(static_cast<size_t>(hidden));
        for (int i = 0; i < hidden; ++i) wf[size_t(i)] = bf16_to_f32(w[size_t(i)]);
        for (int t = 0; t < tokens; ++t)
            k2::rmsnorm_grouped(h.data() + size_t(t) * hidden, wf.data(),
                                want.data() + size_t(t) * hidden, hidden, n_groups, 1e-6f);
        std::printf("grouped rmsnorm  no residual                out %.3e  h untouched %s\n",
                    worst_abs(want, got), back == h ? "yes" : "NO");
        CHECK(worst_abs(want, got) < 2e-5, "no-residual grouped rmsnorm mismatch");
        CHECK(back == h, "kernel wrote h when no residual was given");
        sycl::free(d_h,q); sycl::free(d_o,q); sycl::free(d_w,q);
    }

    // ---- 2. softplus gate -------------------------------------------
    // Includes the torch threshold=20 linear region in both directions:
    // below it the log1p form, above it the identity.  A kernel that
    // skips the fallback returns inf for a large gate.
    {
        const int n = 4096;
        const float beta = 0.6931472f;          // log 2
        std::vector<float> attn(static_cast<size_t>(n)), gate(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) attn[size_t(i)] = nd(rng);
        for (int i = 0; i < n; ++i) gate[size_t(i)] = nd(rng) * 12.0f;
        gate[0] = 100.0f; gate[1] = 40.0f; gate[2] = -100.0f; gate[3] = 0.0f;
        float* d_a = sycl::malloc_device<float>(size_t(n), q);
        float* d_g = sycl::malloc_device<float>(size_t(n), q);
        float* d_o = sycl::malloc_device<float>(size_t(n), q);
        q.memcpy(d_a, attn.data(), size_t(n)*4);
        q.memcpy(d_g, gate.data(), size_t(n)*4).wait();
        launch_softplus_gate(q, d_a, d_g, d_o, n, beta).wait();
        std::vector<float> got(static_cast<size_t>(n)), want(static_cast<size_t>(n));
        q.memcpy(got.data(), d_o, size_t(n)*4).wait();
        k2::softplus_gate(attn.data(), gate.data(), want.data(), size_t(n), beta);
        double rel = 0;
        bool finite = true;
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(got[size_t(i)])) finite = false;
            const double d = std::fabs(got[size_t(i)] - want[size_t(i)]);
            rel = std::max(rel, d / std::max(1e-3, double(std::fabs(want[size_t(i)]))));
        }
        std::printf("softplus gate    beta=log2, |gate| up to 100  rel %.3e  finite %s\n",
                    rel, finite ? "yes" : "NO");
        CHECK(rel < 1e-6, "softplus gate mismatch");
        CHECK(finite, "softplus gate produced a non-finite value");
        sycl::free(d_a,q); sycl::free(d_g,q); sycl::free(d_o,q);
    }

    // ---- 3. sigmoid router, selection-only bias -----------------------
    // Both real shapes: MoVA (64 experts, top-4) and the MoE block
    // (100 experts, top-8), plus a count that is not a multiple of the
    // sub-group width.
    for (auto shape : std::vector<std::array<int,2>>{{64,4},{100,8},{37,3}}) {
        const int n_experts = shape[0], top_k = shape[1], tokens = 64;
        const float scaling = 2.5f;
        std::vector<float> logits(size_t(tokens) * n_experts);
        std::vector<bf16_t> bias(static_cast<size_t>(n_experts));
        std::vector<float> biasf(static_cast<size_t>(n_experts));
        for (auto& v : logits) v = nd(rng) * 2.0f;
        for (int e = 0; e < n_experts; ++e) {
            bias[size_t(e)] = f32_to_bf16(nd(rng) * 0.2f);
            biasf[size_t(e)] = bf16_to_f32(bias[size_t(e)]);
        }
        float* d_l = sycl::malloc_device<float>(logits.size(), q);
        bf16_t* d_b = sycl::malloc_device<bf16_t>(bias.size(), q);
        int32_t* d_e = sycl::malloc_device<int32_t>(size_t(tokens)*top_k, q);
        float* d_w = sycl::malloc_device<float>(size_t(tokens)*top_k, q);
        q.memcpy(d_l, logits.data(), logits.size()*4);
        q.memcpy(d_b, bias.data(), bias.size()*sizeof(bf16_t)).wait();
        launch_router_topk_k2(q, d_l, d_b, tokens, n_experts, top_k,
                              d_e, d_w, true, scaling).wait();
        std::vector<int32_t> ge(size_t(tokens)*top_k);
        std::vector<float>   gw(size_t(tokens)*top_k);
        q.memcpy(ge.data(), d_e, ge.size()*4);
        q.memcpy(gw.data(), d_w, gw.size()*4).wait();

        int idx_mismatch = 0; double wdiff = 0;
        for (int t = 0; t < tokens; ++t) {
            std::vector<int> wi(static_cast<size_t>(top_k));
            std::vector<float> ww(static_cast<size_t>(top_k));
            k2::router_topk(logits.data() + size_t(t)*n_experts, biasf.data(),
                            n_experts, top_k, true, true, scaling,
                            wi.data(), ww.data());
            for (int s = 0; s < top_k; ++s) {
                if (ge[size_t(t)*top_k+s] != wi[size_t(s)]) ++idx_mismatch;
                wdiff = std::max(wdiff, double(std::fabs(gw[size_t(t)*top_k+s]-ww[size_t(s)])));
            }
        }
        std::printf("router top-k     %3d experts, top-%d          "
                    "%d/%d index mismatches, weight %.3e\n",
                    n_experts, top_k, idx_mismatch, tokens*top_k, wdiff);
        CHECK(idx_mismatch == 0, "router selected different experts than the reference");
        CHECK(wdiff < 1e-5, "router weights differ -- bias may be folded into the weight");
        sycl::free(d_l,q); sycl::free(d_b,q); sycl::free(d_e,q); sycl::free(d_w,q);
    }

    // the bias must move SELECTION and not the returned weight
    {
        const int n_experts = 64, top_k = 4, tokens = 32;
        std::vector<float> logits(size_t(tokens)*n_experts);
        for (auto& v : logits) v = nd(rng) * 2.0f;
        std::vector<bf16_t> bias(static_cast<size_t>(n_experts));
        for (int e = 0; e < n_experts; ++e)
            bias[size_t(e)] = f32_to_bf16(e < 4 ? 5.0f : 0.0f);   // force 0..3 in
        float* d_l = sycl::malloc_device<float>(logits.size(), q);
        bf16_t* d_b = sycl::malloc_device<bf16_t>(bias.size(), q);
        int32_t* d_e = sycl::malloc_device<int32_t>(size_t(tokens)*top_k, q);
        float* d_w = sycl::malloc_device<float>(size_t(tokens)*top_k, q);
        q.memcpy(d_l, logits.data(), logits.size()*4);
        q.memcpy(d_b, bias.data(), bias.size()*sizeof(bf16_t)).wait();
        launch_router_topk_k2(q, d_l, d_b, tokens, n_experts, top_k,
                              d_e, d_w, false, 1.0f).wait();
        std::vector<int32_t> ge(size_t(tokens)*top_k);
        std::vector<float> gw(size_t(tokens)*top_k);
        q.memcpy(ge.data(), d_e, ge.size()*4);
        q.memcpy(gw.data(), d_w, gw.size()*4).wait();
        bool routed = true; double wdiff = 0;
        for (int t = 0; t < tokens; ++t) {
            std::vector<int> seen(ge.begin()+size_t(t)*top_k, ge.begin()+size_t(t+1)*top_k);
            std::sort(seen.begin(), seen.end());
            for (int s = 0; s < top_k; ++s) if (seen[size_t(s)] != s) routed = false;
            for (int s = 0; s < top_k; ++s) {
                const int e = ge[size_t(t)*top_k+s];
                const float unbiased = 1.0f/(1.0f+std::exp(-logits[size_t(t)*n_experts+e]));
                wdiff = std::max(wdiff, double(std::fabs(gw[size_t(t)*top_k+s]-unbiased)));
            }
        }
        std::printf("router bias      +5 on experts 0-3            "
                    "selection forced %s, weight-vs-unbiased %.3e\n",
                    routed ? "yes" : "NO", wdiff);
        CHECK(routed, "a large bias did not steer selection");
        CHECK(wdiff < 1e-6, "the returned weight carries the bias -- it must be UNBIASED");
        sycl::free(d_l,q); sycl::free(d_b,q); sycl::free(d_e,q); sycl::free(d_w,q);
    }

    // ---- 4. MoVA accumulate: out += w * silu(in) ---------------------
    {
        const int n = 4096, top_k = 4;
        std::vector<float> out(size_t(n), 0.0f);
        std::vector<std::vector<float>> ex(size_t(top_k), std::vector<float>(static_cast<size_t>(n)));
        std::vector<float> w(static_cast<size_t>(top_k));
        for (auto& e : ex) for (auto& v : e) v = nd(rng);
        for (auto& v : w) v = 0.1f + std::fabs(nd(rng));
        float* d_in = sycl::malloc_device<float>(size_t(n), q);
        float* d_out = sycl::malloc_device<float>(size_t(n), q);
        q.memset(d_out, 0, size_t(n)*4).wait();
        for (int j = 0; j < top_k; ++j) {
            q.memcpy(d_in, ex[size_t(j)].data(), size_t(n)*4).wait();
            launch_silu_scale_accum(q, d_in, d_out, w[size_t(j)], n).wait();
        }
        std::vector<float> got(static_cast<size_t>(n));
        q.memcpy(got.data(), d_out, size_t(n)*4).wait();
        std::vector<const float*> ptr(static_cast<size_t>(top_k));
        for (int j = 0; j < top_k; ++j) ptr[size_t(j)] = ex[size_t(j)].data();
        k2::mova_combine(ptr.data(), w.data(), top_k, n, out.data());
        std::printf("mova accumulate  4 experts, silu on output    %.3e\n",
                    worst_abs(out, got));
        CHECK(worst_abs(out, got) < 1e-5, "MoVA accumulate mismatch");
        sycl::free(d_in,q); sycl::free(d_out,q);
    }

    // ---- 5. DFlash2 candidate selector -------------------------------
    // The whole chain, on device: top-16 per row over a PADDED logit
    // stride, edge scores, then the greedy walk.
    {
        const int vocab = 512, rank = 8, K = 16, steps = 6;
        const int stride = vocab + 37;          // padded row, as the draft head is
        std::vector<float> logits(size_t(steps) * stride, -1e30f);
        for (int s = 0; s < steps; ++s)
            for (int v = 0; v < vocab; ++v)
                logits[size_t(s)*stride + v] = nd(rng);
        std::vector<float> pred(size_t(vocab)*rank), succ(size_t(vocab)*rank);
        for (auto& v : pred) v = bf16r(nd(rng)*0.5f);
        for (auto& v : succ) v = bf16r(nd(rng)*0.5f);
        std::vector<bf16_t> predb(pred.size()), succb(succ.size());
        for (size_t i = 0; i < pred.size(); ++i) predb[i] = f32_to_bf16(pred[i]);
        for (size_t i = 0; i < succ.size(); ++i) succb[i] = f32_to_bf16(succ[i]);
        std::vector<float> hidden(size_t(steps)*rank);
        for (auto& v : hidden) v = nd(rng);
        const int32_t anchor = 123;

        float* d_logits = sycl::malloc_device<float>(logits.size(), q);
        int32_t* d_ids = sycl::malloc_device<int32_t>(size_t(steps)*K, q);
        float* d_unary = sycl::malloc_device<float>(size_t(steps)*K, q);
        bf16_t* d_pred = sycl::malloc_device<bf16_t>(predb.size(), q);
        bf16_t* d_succ = sycl::malloc_device<bf16_t>(succb.size(), q);
        float* d_hidden = sycl::malloc_device<float>(hidden.size(), q);
        float* d_scores = sycl::malloc_device<float>(size_t(steps)*K*K, q);
        int32_t* d_tok = sycl::malloc_device<int32_t>(size_t(steps), q);
        q.memcpy(d_logits, logits.data(), logits.size()*4);
        q.memcpy(d_pred, predb.data(), predb.size()*sizeof(bf16_t));
        q.memcpy(d_succ, succb.data(), succb.size()*sizeof(bf16_t));
        q.memcpy(d_hidden, hidden.data(), hidden.size()*4).wait();

        // topk16_rows takes the ROW WIDTH it should scan; the padded tail
        // must never win a slot.
        launch_topk16_rows(q, d_logits, steps, stride, d_ids, d_unary).wait();
        std::vector<int32_t> ids(size_t(steps)*K);
        std::vector<float> unary(size_t(steps)*K);
        q.memcpy(ids.data(), d_ids, ids.size()*4);
        q.memcpy(unary.data(), d_unary, unary.size()*4).wait();

        int bad_id = 0, bad_order = 0;
        for (int s = 0; s < steps; ++s) {
            std::vector<float> row(logits.begin()+size_t(s)*stride,
                                   logits.begin()+size_t(s)*stride+stride);
            std::vector<int> order(static_cast<size_t>(stride));
            for (int i = 0; i < stride; ++i) order[size_t(i)] = i;
            std::partial_sort(order.begin(), order.begin()+K, order.end(),
                [&](int a, int b){ return row[size_t(a)] > row[size_t(b)]; });
            for (int c = 0; c < K; ++c) {
                if (ids[size_t(s)*K+c] != order[size_t(c)]) ++bad_order;
                if (ids[size_t(s)*K+c] >= vocab) ++bad_id;
            }
        }
        std::printf("topk16_rows      padded stride %d over %d real  "
                    "%d out-of-range, %d order mismatches\n",
                    stride, vocab, bad_id, bad_order);
        CHECK(bad_id == 0, "top-16 selected a padded slot");
        CHECK(bad_order == 0, "top-16 order differs from a host partial_sort");

        launch_dflash2_selector_edges(q, d_pred, d_succ, d_ids, d_unary,
                                      d_hidden, anchor, d_scores, steps, K, rank).wait();
        std::vector<float> scores(size_t(steps)*K*K);
        q.memcpy(scores.data(), d_scores, scores.size()*4).wait();

        // reference: vLLM _score_edges
        std::vector<float> want(size_t(steps)*K*K);
        for (int l = 0; l < steps; ++l)
            for (int p = 0; p < K; ++p) {
                const int32_t pid = (l==0) ? anchor : ids[size_t(l-1)*K+p];
                for (int c = 0; c < K; ++c) {
                    const int32_t cid = ids[size_t(l)*K+c];
                    float acc = 0;
                    for (int r = 0; r < rank; ++r)
                        acc += bf16r(pred[size_t(pid)*rank+r]*bf16r(hidden[size_t(l)*rank+r]))
                             * succ[size_t(cid)*rank+r];
                    want[(size_t(l)*K+p)*K+c] = unary[size_t(l)*K+c] + bf16r(acc);
                }
            }
        const double sd = worst_abs(want, scores);
        std::printf("selector edges   %zu scores                   %.3e\n",
                    want.size(), sd);
        CHECK(sd < 1e-3, "edge scores differ from vLLM _score_edges");

        // step 0 must use the anchor for every predecessor slot
        double step0 = 0;
        for (int p = 1; p < K; ++p)
            for (int c = 0; c < K; ++c)
                step0 = std::max(step0, double(std::fabs(scores[size_t(c)] -
                                 scores[(size_t(0)*K+p)*K+c])));
        CHECK(step0 < 1e-6, "step 0 does not use the anchor for every predecessor");

        launch_dflash2_path_walk(q, d_scores, d_ids, d_tok, steps, K).wait();
        std::vector<int32_t> tok(static_cast<size_t>(steps));
        q.memcpy(tok.data(), d_tok, tok.size()*4).wait();

        std::vector<int32_t> wtok(static_cast<size_t>(steps));
        int previous = 0;
        for (int s = 0; s < steps; ++s) {
            const float* row = scores.data() + (size_t(s)*K + previous)*K;
            int pick = 0;
            for (int c = 1; c < K; ++c) if (row[c] > row[pick]) pick = c;
            wtok[size_t(s)] = ids[size_t(s)*K + pick];
            previous = pick;
        }
        std::printf("path walk        greedy from the anchor       path");
        for (int s = 0; s < steps; ++s) std::printf(" %d", tok[size_t(s)]);
        std::printf("\n");
        CHECK(tok == wtok, "device walk differs from the sequential greedy reference");

        // and it must not be per-position argmax, or there is nothing here
        std::vector<int32_t> greedy(static_cast<size_t>(steps));
        for (int s = 0; s < steps; ++s) {
            int best = 0;
            for (int c = 1; c < K; ++c)
                if (unary[size_t(s)*K+c] > unary[size_t(s)*K+best]) best = c;
            greedy[size_t(s)] = ids[size_t(s)*K+best];
        }
        CHECK(greedy != tok, "selector path equals per-position argmax");

        sycl::free(d_logits,q); sycl::free(d_ids,q); sycl::free(d_unary,q);
        sycl::free(d_pred,q); sycl::free(d_succ,q); sycl::free(d_hidden,q);
        sycl::free(d_scores,q); sycl::free(d_tok,q);
    }

    // ---- 6. the norm CONVENTION, and prefill vs decode ----------------
    // set_norm_convention is called once in build() so that decode
    // (launch_rmsnorm_residual, M=1) and prefill
    // (launch_rmsnorm_residual_batched) cannot end up on different
    // conventions.  If they ever did it would be silent: the model would
    // simply be worse.  Check that the two entry points agree bit for bit
    // under BOTH conventions, and that the default is unchanged for every
    // non-K2 model.
    {
        const int hidden = 512, tokens = 5;
        std::vector<float> h(size_t(tokens) * hidden), r(h.size());
        std::vector<bf16_t> w(static_cast<size_t>(hidden));
        for (auto& v : h) v = nd(rng);
        for (auto& v : r) v = nd(rng) * 0.25f;
        for (auto& v : w) v = f32_to_bf16(nd(rng) * 0.1f);

        float* d_h = sycl::malloc_device<float>(h.size(), q);
        float* d_r = sycl::malloc_device<float>(r.size(), q);
        float* d_o = sycl::malloc_device<float>(h.size(), q);
        bf16_t* d_w = sycl::malloc_device<bf16_t>(w.size(), q);
        q.memcpy(d_r, r.data(), r.size() * 4);
        q.memcpy(d_w, w.data(), w.size() * sizeof(bf16_t)).wait();

        struct Conv { int groups; float offset; const char* name; };
        for (auto c : { Conv{1, 1.0f, "qwen (1+w), whole row"},
                        Conv{2, 0.0f, "k2 (w), 2 groups"} }) {
            set_norm_convention(c.groups, c.offset);
            CHECK(norm_is_grouped(hidden) == (c.groups > 1 || c.offset != 1.0f),
                  "norm_is_grouped disagrees with the convention it was given");

            // batched (prefill) over all tokens
            q.memcpy(d_h, h.data(), h.size() * 4).wait();
            launch_rmsnorm_residual_batched(q, d_h, d_r, nullptr, d_w, d_o,
                                            tokens, hidden, 1e-6f, nullptr, {},
                                            c.offset).wait();
            std::vector<float> batched(h.size());
            q.memcpy(batched.data(), d_o, batched.size() * 4).wait();

            // one token at a time (decode)
            std::vector<float> single(h.size());
            for (int t = 0; t < tokens; ++t) {
                q.memcpy(d_h, h.data() + size_t(t) * hidden, size_t(hidden) * 4).wait();
                launch_rmsnorm_residual(q, d_h, d_r + size_t(t) * hidden, d_w, d_o,
                                        hidden, 1e-6f, {}).wait();
                q.memcpy(single.data() + size_t(t) * hidden, d_o,
                         size_t(hidden) * 4).wait();
            }
            const double split = worst_abs(batched, single);
            std::printf("norm convention  %-24s prefill vs decode %.3e\n",
                        c.name, split);
            CHECK(split < 1e-6, "prefill and decode are on DIFFERENT norm conventions");

            // and against the host reference for this convention
            std::vector<float> want(h.size()), wf(static_cast<size_t>(hidden));
            for (int i = 0; i < hidden; ++i)
                wf[size_t(i)] = c.offset + bf16_to_f32(w[size_t(i)]);
            std::vector<float> hr(h.size());
            for (size_t i = 0; i < h.size(); ++i) hr[i] = h[i] + r[i];
            for (int t = 0; t < tokens; ++t)
                k2::rmsnorm_grouped(hr.data() + size_t(t) * hidden, wf.data(),
                                    want.data() + size_t(t) * hidden,
                                    hidden, c.groups, 1e-6f);
            CHECK(worst_abs(want, batched) < 2e-5,
                  "%s: output does not match the host reference", c.name);
        }
        set_norm_convention(1, 1.0f);            // leave the default in place
        CHECK(!norm_is_grouped(hidden), "default convention must not be grouped");
        sycl::free(d_h,q); sycl::free(d_r,q); sycl::free(d_o,q); sycl::free(d_w,q);
    }

    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
