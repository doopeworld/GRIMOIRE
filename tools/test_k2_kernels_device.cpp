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
#include "b70/qwen4_exp.hpp"

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
sycl::event launch_rmsnorm_heads(sycl::queue&, float*, const bf16_t*, int, int,
                                 float, bool, const std::vector<sycl::event>&);
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

    // ---- 3b. GeGLU, and that it is NOT SwiGLU ------------------------
    // Gemma asks for gelu_pytorch_tanh.  Running silu in its place loads
    // cleanly and produces the wrong model's output, so this checks both
    // that GeGLU matches its reference AND that it visibly differs from
    // the kernel it would be confused with.
    {
        const int rows = 5, inter = 96;
        std::vector<float> gu(size_t(rows) * 2 * inter);
        for (auto& v : gu) v = nd(rng) * 1.5f;

        float* d_gu  = sycl::malloc_device<float>(gu.size(), q);
        float* d_out = sycl::malloc_device<float>(size_t(rows) * inter, q);
        q.memcpy(d_gu, gu.data(), gu.size() * 4).wait();
        launch_geglu_batched(q, d_gu, d_out, rows, inter).wait();
        std::vector<float> got(size_t(rows) * inter);
        q.memcpy(got.data(), d_out, got.size() * 4).wait();

        // Host reference, written from the formula the reference uses.
        auto gelu = [](double x) {
            const double a = 0.7978845608028654, b = 0.044715;
            return 0.5 * x * (1.0 + std::tanh(a * (x + b * x * x * x)));
        };
        std::vector<float> want(got.size()), silu(got.size());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < inter; ++c) {
                const double g = gu[size_t(r) * 2 * inter + c];
                const double u = gu[size_t(r) * 2 * inter + inter + c];
                want[size_t(r) * inter + c] = float(gelu(g) * u);
                silu[size_t(r) * inter + c] = float(g / (1.0 + std::exp(-g)) * u);
            }
        const double d_ref  = worst_abs(want, got);
        const double d_silu = worst_abs(silu, got);
        std::printf("geglu            %d x %d   vs reference %.3e   vs swiglu %.3e\n",
                    rows, inter, d_ref, d_silu);
        CHECK(d_ref < 1e-5, "GeGLU does not match gelu_pytorch_tanh(gate)*up");
        // If this ever passes, the kernel is silu and the check above is
        // measuring nothing.
        CHECK(d_silu > 1e-2, "GeGLU is indistinguishable from SwiGLU here, so "
                             "the reference comparison proves nothing");
        sycl::free(d_gu, q); sycl::free(d_out, q);
    }

    // ---- 3c. proportional RoPE, and that it is NOT partial_rope -------
    // gemma-4 full-attention layers.  The exponent divides by the FULL
    // head_dim and dim i pairs with i + head_dim/2; partial_rope divides
    // by the rotated width and pairs j with j + rot/2.  Both differences
    // are invisible in the output.
    {
        const int heads = 3, head_dim = 64, pos = 7;
        const float theta = 1000000.0f, factor = 0.25f;
        std::vector<float> x0(size_t(heads) * head_dim);
        for (auto& v : x0) v = nd(rng);

        float* d_x = sycl::malloc_device<float>(x0.size(), q);
        int32_t* d_p = sycl::malloc_device<int32_t>(1, q);
        q.memcpy(d_p, &pos, sizeof(int)).wait();
        q.memcpy(d_x, x0.data(), x0.size() * 4).wait();
        launch_rope_proportional(q, d_x, heads, head_dim, d_p, theta, factor).wait();
        std::vector<float> got(x0.size());
        q.memcpy(got.data(), d_x, got.size() * 4).wait();

        // Host reference, straight from ref/gemma4_proportional_rope.py.
        const int half = head_dim / 2;
        const int angles = int(double(factor) * head_dim / 2.0);
        std::vector<float> want = x0;
        for (int h2 = 0; h2 < heads; ++h2)
            for (int i = 0; i < angles; ++i) {
                const double inv = std::pow(double(theta),
                                            -double(2 * i) / double(head_dim));
                const double ang = pos * inv;
                const double c = std::cos(ang), sn = std::sin(ang);
                const double a = x0[size_t(h2) * head_dim + i];
                const double b = x0[size_t(h2) * head_dim + i + half];
                want[size_t(h2) * head_dim + i]        = float(a * c - b * sn);
                want[size_t(h2) * head_dim + i + half] = float(a * sn + b * c);
            }
        const double d_ref = worst_abs(want, got);

        // What GRIMOIRE's partial_rope would have produced instead.
        const int rot = int(head_dim * factor) & ~1;
        std::vector<float> partial = x0;
        for (int h2 = 0; h2 < heads; ++h2)
            for (int j = 0; j < rot / 2; ++j) {
                const double inv = std::pow(double(theta),
                                            -double(2 * j) / double(rot));
                const double ang = pos * inv;
                const double c = std::cos(ang), sn = std::sin(ang);
                const double a = x0[size_t(h2) * head_dim + j];
                const double b = x0[size_t(h2) * head_dim + j + rot / 2];
                partial[size_t(h2) * head_dim + j]           = float(a * c - b * sn);
                partial[size_t(h2) * head_dim + j + rot / 2] = float(a * sn + b * c);
            }
        const double d_partial = worst_abs(partial, got);
        std::printf("rope proportional %d heads hd=%d f=%.2f  vs reference %.3e"
                    "   vs partial_rope %.3e\n",
                    heads, head_dim, double(factor), d_ref, d_partial);
        CHECK(d_ref < 1e-5, "proportional RoPE does not match the reference");
        CHECK(d_partial > 1e-2, "proportional RoPE is indistinguishable from "
                                "partial_rope here, so the comparison proves "
                                "nothing");
        // Dimensions past the rotated band must be untouched, not scaled.
        int touched = 0;
        for (int h2 = 0; h2 < heads; ++h2)
            for (int i = angles; i < half; ++i)
                if (got[size_t(h2)*head_dim+i] != x0[size_t(h2)*head_dim+i] ||
                    got[size_t(h2)*head_dim+i+half] != x0[size_t(h2)*head_dim+i+half])
                    ++touched;
        CHECK(touched == 0, "%d unrotated dims were modified", touched);
        sycl::free(d_x, q); sycl::free(d_p, q);
    }

    // ---- 3d. batched proportional RoPE == the single-token one --------
    // Prefill and decode must apply the SAME rotation.  They are separate
    // kernels in separate files (ops.cpp, prefill.cpp), which is exactly
    // how a prefill/decode divergence gets in: nothing downstream notices,
    // the KV cache simply holds keys rotated one way and the queries
    // another.  Drive both over several positions and require agreement,
    // and require that the batched kernel is still distinguishable from
    // the ordinary partial_rope one on the same input.
    {
        const int QH = 2, KVH = 1, HD = 64, M = 5, start = 3;
        const float theta = 1000000.0f, factor = 0.25f, eps = 1e-6f;
        std::vector<float> q0(size_t(M) * QH * HD), k0(size_t(M) * KVH * HD);
        for (auto& v : q0) v = nd(rng);
        for (auto& v : k0) v = nd(rng);
        // Scaleless norms on both sides, so any difference is the rotation.
        std::vector<bf16_t> ones_q(HD), ones_k(HD);
        for (int i = 0; i < HD; ++i) ones_q[size_t(i)] = ones_k[size_t(i)] = f32_to_bf16(0.0f);

        auto up = [&](const std::vector<float>& h) {
            float* d = sycl::malloc_device<float>(h.size(), q);
            q.memcpy(d, h.data(), h.size() * 4).wait(); return d;
        };
        bf16_t* d_wq = sycl::malloc_device<bf16_t>(size_t(HD), q);
        bf16_t* d_wk = sycl::malloc_device<bf16_t>(size_t(HD), q);
        q.memcpy(d_wq, ones_q.data(), size_t(HD) * sizeof(bf16_t)).wait();
        q.memcpy(d_wk, ones_k.data(), size_t(HD) * sizeof(bf16_t)).wait();

        float* bq = up(q0); float* bk = up(k0);
        launch_qk_norm_rope_proportional_batched(q, bq, bk, d_wq, d_wk, M, QH,
            KVH, HD, start, theta, factor, eps).wait();
        std::vector<float> gotq(q0.size()), gotk(k0.size());
        q.memcpy(gotq.data(), bq, gotq.size() * 4).wait();
        q.memcpy(gotk.data(), bk, gotk.size() * 4).wait();

        // Row by row through the DECODE kernels: norm, then the single-token
        // proportional rotation at that row's own position.
        std::vector<float> wantq(q0.size()), wantk(k0.size());
        int32_t* d_pos = sycl::malloc_device<int32_t>(1, q);
        for (int m = 0; m < M; ++m) {
            const int p = start + m;
            q.memcpy(d_pos, &p, sizeof(int)).wait();
            float* rq = up(std::vector<float>(
                q0.begin() + size_t(m) * QH * HD,
                q0.begin() + size_t(m + 1) * QH * HD));
            float* rk = up(std::vector<float>(
                k0.begin() + size_t(m) * KVH * HD,
                k0.begin() + size_t(m + 1) * KVH * HD));
            launch_rmsnorm_heads(q, rq, d_wq, QH,  HD, eps, true, {}).wait();
            launch_rmsnorm_heads(q, rk, d_wk, KVH, HD, eps, true, {}).wait();
            launch_rope_proportional(q, rq, QH,  HD, d_pos, theta, factor).wait();
            launch_rope_proportional(q, rk, KVH, HD, d_pos, theta, factor).wait();
            q.memcpy(wantq.data() + size_t(m) * QH * HD, rq,
                     size_t(QH) * HD * 4).wait();
            q.memcpy(wantk.data() + size_t(m) * KVH * HD, rk,
                     size_t(KVH) * HD * 4).wait();
            sycl::free(rq, q); sycl::free(rk, q);
        }
        const double dq = worst_abs(wantq, gotq), dk = worst_abs(wantk, gotk);

        // And the ordinary batched kernel on the same input, to prove the
        // agreement above is not the trivial "both did nothing".
        float* pq = up(q0); float* pk = up(k0);
        launch_qk_norm_rope_batched(q, pq, pk, d_wq, d_wk, M, QH, KVH, HD,
                                    start, theta, factor, eps).wait();
        std::vector<float> partq(q0.size());
        q.memcpy(partq.data(), pq, partq.size() * 4).wait();
        const double d_partial = worst_abs(partq, gotq);

        std::printf("rope proportional batched M=%d  vs decode q %.3e k %.3e"
                    "   vs partial_rope %.3e\n", M, dq, dk, d_partial);
        CHECK(dq < 1e-5 && dk < 1e-5,
              "batched proportional RoPE disagrees with the decode kernel");
        CHECK(d_partial > 1e-2,
              "batched proportional RoPE is indistinguishable from "
              "partial_rope here, so the comparison proves nothing");
        sycl::free(bq, q); sycl::free(bk, q);
        sycl::free(pq, q); sycl::free(pk, q);
        sycl::free(d_wq, q); sycl::free(d_wk, q); sycl::free(d_pos, q);
    }

    // ---- 4a. quantize-then-append == append-then-quantize -------------
    // The expert-major pack quantizes each expert on its own and appends,
    // so the host never holds 64 of them as f32 at once.  That is only
    // exact because every format here is ROW-LOCAL -- BF16 elementwise,
    // FP8/INT8 per output channel, INT4 per group within a row, MX per
    // 32-element block within a row.  Assert it instead of arguing it: if
    // any format ever grew a cross-row term, the packed weight would stay
    // finite and decode to slightly wrong numbers everywhere.
    for (Fmt fmt : {Fmt::BF16, Fmt::FP8_E4M3, Fmt::FP8_E5M2, Fmt::INT8,
                    Fmt::INT4, Fmt::MXFP8, Fmt::MXFP4}) {
        const int P = 4, N = 24, K = 128;
        std::vector<float> whole(size_t(P) * N * K);
        for (auto& v : whole) v = nd(rng) * 0.7f;
        PackedWeight all = quantize(whole.data(), P * N, K, fmt);

        PackedWeight app;
        app.fmt = fmt; app.N = P * N; app.K = K;
        for (int i = 0; i < P; ++i) {
            PackedWeight one = quantize(whole.data() + size_t(i) * N * K, N, K, fmt);
            if (i == 0) { app.row_bytes = one.row_bytes; app.row_scales = one.row_scales; }
            app.payload.insert(app.payload.end(), one.payload.begin(), one.payload.end());
            app.scales_raw.insert(app.scales_raw.end(),
                                  one.scales_raw.begin(), one.scales_raw.end());
            app.zeros.insert(app.zeros.end(), one.zeros.begin(), one.zeros.end());
        }
        const bool same_bytes = app.payload == all.payload &&
                                app.scales_raw == all.scales_raw &&
                                app.zeros == all.zeros &&
                                app.row_bytes == all.row_bytes &&
                                app.row_scales == all.row_scales;
        std::printf("pack append      %-9s %d x [%d,%d]                   %s\n",
                    fmt_name(fmt), P, N, K, same_bytes ? "byte-identical" : "DIFFERS");
        CHECK(same_bytes, "appending per-piece quantization is not identical to "
                          "quantizing the concatenation (%s)", fmt_name(fmt));
    }

    // ---- 4b. MoVA value projection, experts packed EXPERT-MAJOR ------
    // The kernel that replaces the per-expert GEMV plus the host readback.
    // It must produce EXACTLY what that path produced, so the reference
    // here is the same one: dequantize through QuantWeight::at(), which is
    // what every GRIMOIRE kernel is required to agree with, then
    // k2::mova_combine over the routed expert outputs.
    //
    // Every format, because the expert-major index (expert*N + row) walks
    // the scale array differently per format -- a per-channel scale is one
    // value per packed row, a group scale is a row of them, and getting
    // that wrong reads another expert's scales while still producing
    // finite, plausible values.
    for (Fmt fmt : {Fmt::BF16, Fmt::FP8_E4M3, Fmt::FP8_E5M2, Fmt::INT8,
                    Fmt::INT4, Fmt::MXFP8, Fmt::MXFP4}) {
        const int E = 6, N = 32, K = 128, M = 3, top_k = 3;
        std::vector<float> host(size_t(E) * N * K);
        for (auto& v : host) v = nd(rng) * 0.5f;
        PackedWeight pw = quantize(host.data(), E * N, K, fmt);

        // Upload the packed weight as the engine does.
        uint8_t* d_pay = sycl::malloc_device<uint8_t>(pw.payload.size(), q);
        q.memcpy(d_pay, pw.payload.data(), pw.payload.size()).wait();
        uint8_t* d_sc = nullptr;
        if (!pw.scales_raw.empty()) {
            d_sc = sycl::malloc_device<uint8_t>(pw.scales_raw.size(), q);
            q.memcpy(d_sc, pw.scales_raw.data(), pw.scales_raw.size()).wait();
        }
        uint8_t* d_zr = nullptr;
        if (!pw.zeros.empty()) {
            d_zr = sycl::malloc_device<uint8_t>(pw.zeros.size(), q);
            q.memcpy(d_zr, pw.zeros.data(), pw.zeros.size()).wait();
        }
        QuantWeight dw = pw.view();
        dw.payload = d_pay; dw.scales = d_sc; dw.zeros = d_zr;

        std::vector<float> x(size_t(M) * K);
        for (auto& v : x) v = nd(rng);
        // A deliberate out-of-range route in the last slot of row 1: the
        // router emits -1 when it fills fewer than top_k slots, and the
        // kernel must skip it rather than index past the experts.
        std::vector<int32_t> rex(size_t(M) * top_k);
        std::vector<float>   rwt(size_t(M) * top_k);
        for (int m = 0; m < M; ++m)
            for (int j = 0; j < top_k; ++j) {
                rex[size_t(m)*top_k+j] = (m * 2 + j * 3) % E;
                rwt[size_t(m)*top_k+j] = 0.15f + 0.2f * float(j);
            }
        rex[size_t(1)*top_k + top_k - 1] = -1;

        float*   d_x   = sycl::malloc_device<float>(x.size(), q);
        int32_t* d_rex = sycl::malloc_device<int32_t>(rex.size(), q);
        float*   d_rwt = sycl::malloc_device<float>(rwt.size(), q);
        float*   d_y   = sycl::malloc_device<float>(size_t(M) * N, q);
        q.memcpy(d_x, x.data(), x.size()*4).wait();
        q.memcpy(d_rex, rex.data(), rex.size()*sizeof(int32_t)).wait();
        q.memcpy(d_rwt, rwt.data(), rwt.size()*sizeof(float)).wait();
        q.memset(d_y, 0, size_t(M)*N*4).wait();

        launch_mova_value_packed(q, dw, d_x, d_rex, d_rwt, d_y,
                                 M, N, E, top_k).wait();
        std::vector<float> got(size_t(M) * N);
        q.memcpy(got.data(), d_y, size_t(M)*N*4).wait();

        // Reference: dequantize on the host through the SAME accessor the
        // kernel is required to match, project, then combine.
        const QuantWeight hw = pw.view();
        std::vector<float> want(size_t(M) * N, 0.0f);
        for (int m = 0; m < M; ++m) {
            std::vector<std::vector<float>> outs;
            std::vector<float> ws;
            for (int j = 0; j < top_k; ++j) {
                const int e = rex[size_t(m)*top_k+j];
                if (e < 0 || e >= E) continue;
                std::vector<float> o(size_t(N), 0.0f);
                for (int n2 = 0; n2 < N; ++n2) {
                    double a = 0.0;
                    for (int k = 0; k < K; ++k)
                        a += double(hw.at(e * N + n2, k)) * double(x[size_t(m)*K+k]);
                    o[size_t(n2)] = float(a);
                }
                outs.push_back(std::move(o));
                ws.push_back(rwt[size_t(m)*top_k+j]);
            }
            std::vector<const float*> ptr(outs.size());
            for (size_t i = 0; i < outs.size(); ++i) ptr[i] = outs[i].data();
            k2::mova_combine(ptr.data(), ws.data(), int(outs.size()), N,
                             want.data() + size_t(m)*N);
        }
        const double d = worst_abs(want, got);
        std::printf("mova packed      %-9s E=%d N=%d K=%d top_k=%d      %.3e\n",
                    fmt_name(fmt), E, N, K, top_k, d);
        // The tolerance is on the ACCUMULATION order, not the dequant: the
        // host sums in double and the kernel in float over K=128 terms.
        CHECK(d < 2e-3, "packed MoVA differs from the per-expert reference (%s)",
              fmt_name(fmt));
        // The skipped route must actually be skipped: if the kernel read
        // expert -1 it would still write finite numbers.
        CHECK(std::isfinite(got[size_t(1)*N]), "packed MoVA produced a non-finite value");
        sycl::free(d_pay,q); if(d_sc) sycl::free(d_sc,q); if(d_zr) sycl::free(d_zr,q);
        sycl::free(d_x,q); sycl::free(d_rex,q); sycl::free(d_rwt,q); sycl::free(d_y,q);
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

    // ---- Qwen4-Exp HyperConnections --------------------------------
    // The residual stream is hc_count wide.  Three things here are silent
    // when wrong, so each is diffed against b70/qwen4_exp.hpp rather than
    // eyeballed:
    //   * the norm is GROUPED per stream and applies (1 + w), NOT w --
    //     the class is called GroupedGemmaRMSNorm and gemma-4 is the one
    //     model here that applies its weight directly (rule 11)
    //   * the /hc_count sits INSIDE silu and sigmoid, not outside
    //   * combine adds to the UNNORMALISED stream
    {
        const int rows = 3, hc = 4, hidden = 32, lowrank = 16;
        const int wide = hc * hidden;
        std::mt19937 rng(20260916);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<float> hyper(size_t(rows) * wide), blk(size_t(rows) * hidden);
        std::vector<bf16_t> nw((size_t(wide)));
        for (auto& v : hyper) v = d(rng);
        for (auto& v : blk)   v = d(rng);
        for (auto& v : nw)    v = f32_to_bf16(d(rng) * 0.1f);

        float* d_hyper = sycl::malloc_device<float>(hyper.size(), q);
        float* d_norm  = sycl::malloc_device<float>(hyper.size(), q);
        float* d_blk   = sycl::malloc_device<float>(blk.size(), q);
        float* d_out   = sycl::malloc_device<float>(hyper.size(), q);
        bf16_t* d_nw   = sycl::malloc_device<bf16_t>(nw.size(), q);
        q.memcpy(d_hyper, hyper.data(), hyper.size()*sizeof(float));
        q.memcpy(d_blk, blk.data(), blk.size()*sizeof(float));
        q.memcpy(d_nw, nw.data(), nw.size()*sizeof(bf16_t)).wait();

        // --- grouped norm -------------------------------------------
        launch_hc_norm(q, d_hyper, d_nw, d_norm, rows, hc, hidden, 1e-6f, {});
        q.wait();
        std::vector<float> got(hyper.size()), want(hyper.size());
        q.memcpy(got.data(), d_norm, got.size()*sizeof(float)).wait();
        std::vector<float> wf((size_t(wide)));
        for (int i = 0; i < wide; ++i) wf[size_t(i)] = bf16_to_f32(nw[size_t(i)]);
        for (int r = 0; r < rows; ++r)
            qwen4_exp::hc_norm(hyper.data() + size_t(r)*wide, wf.data(),
                               want.data() + size_t(r)*wide, hc, hidden, 1e-6f);
        std::printf("%-16s %-40s %.3e\n", "hc norm",
                    "grouped per stream, (1 + w)", worst_abs(want, got));
        CHECK(worst_abs(want, got) < 2e-5,
              "hc_norm does not match the host reference");

        // A ones-valued weight under (1 + w) is a DOUBLING, and under
        // plain w it is the identity.  If the kernel ever drifts to the
        // gemma-4 convention this separates them by 2x rather than by
        // rounding, which the tolerance above would not catch on its own.
        {
            std::vector<bf16_t> ones(size_t(wide), f32_to_bf16(1.0f));
            q.memcpy(d_nw, ones.data(), ones.size()*sizeof(bf16_t)).wait();
            launch_hc_norm(q, d_hyper, d_nw, d_norm, rows, hc, hidden, 1e-6f, {});
            q.wait();
            std::vector<float> g2(hyper.size()), plain(hyper.size());
            q.memcpy(g2.data(), d_norm, g2.size()*sizeof(float)).wait();
            std::vector<float> zw(size_t(wide), 0.0f);
            for (int r = 0; r < rows; ++r)
                qwen4_exp::hc_norm(hyper.data() + size_t(r)*wide, zw.data(),
                                   plain.data() + size_t(r)*wide, hc, hidden, 1e-6f);
            // plain == the (1+0) result == normalized.  got should be 2x it.
            double worst = 0.0;
            for (size_t i = 0; i < g2.size(); ++i)
                worst = std::max(worst, std::abs(double(g2[i]) - 2.0*plain[i]));
            std::printf("%-16s %-40s %.3e\n", "hc norm w=1",
                        "(1+w) doubles; plain w would not", worst);
            CHECK(worst < 2e-5, "hc_norm is not applying (1 + w)");
            q.memcpy(d_nw, nw.data(), nw.size()*sizeof(bf16_t)).wait();
            launch_hc_norm(q, d_hyper, d_nw, d_norm, rows, hc, hidden, 1e-6f, {});
            q.wait();
        }

        // --- gated mean ---------------------------------------------
        // Stand in for the up projection with random logits: the kernel
        // owns sigmoid + the mean, and that is what is being pinned.
        std::vector<float> upo(size_t(rows) * wide);
        for (auto& v : upo) v = d(rng) * 3.0f;
        float* d_upo = sycl::malloc_device<float>(upo.size(), q);
        float* d_mix = sycl::malloc_device<float>(size_t(rows)*hidden, q);
        q.memcpy(d_upo, upo.data(), upo.size()*sizeof(float)).wait();
        launch_hc_gated_mean(q, d_upo, d_norm, d_mix, rows, hc, hidden, {});
        q.wait();
        std::vector<float> mix_got(size_t(rows)*hidden), mix_want(size_t(rows)*hidden);
        q.memcpy(mix_got.data(), d_mix, mix_got.size()*sizeof(float)).wait();
        q.memcpy(got.data(), d_norm, got.size()*sizeof(float)).wait();
        for (int r = 0; r < rows; ++r)
            for (int hh = 0; hh < hidden; ++hh) {
                float acc = 0.0f;
                for (int c = 0; c < hc; ++c) {
                    const size_t i = size_t(r)*wide + size_t(c)*hidden + hh;
                    acc += qwen4_exp::sigmoid(upo[i]) * got[i];
                }
                mix_want[size_t(r)*hidden + hh] = acc / float(hc);
            }
        std::printf("%-16s %-40s %.3e\n", "hc gated mean",
                    "sigmoid(up) * normed, averaged", worst_abs(mix_want, mix_got));
        CHECK(worst_abs(mix_want, mix_got) < 2e-5,
              "hc_gated_mean does not match the host reference");

        // --- combine -------------------------------------------------
        std::vector<float> injo(size_t(rows) * hc);
        for (auto& v : injo) v = d(rng) * 3.0f;
        float* d_injo = sycl::malloc_device<float>(injo.size(), q);
        q.memcpy(d_injo, injo.data(), injo.size()*sizeof(float)).wait();
        launch_hc_combine(q, d_hyper, d_injo, d_blk, d_out, rows, hc, hidden, {});
        q.wait();
        std::vector<float> cgot(hyper.size()), cwant(hyper.size());
        q.memcpy(cgot.data(), d_out, cgot.size()*sizeof(float)).wait();
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < hc; ++c) {
                const float wgt = 2.0f * qwen4_exp::sigmoid(
                    injo[size_t(r)*hc + c] / float(hc));
                for (int hh = 0; hh < hidden; ++hh) {
                    const size_t i = size_t(r)*wide + size_t(c)*hidden + hh;
                    // the UNNORMALISED stream, deliberately
                    cwant[i] = hyper[i] + blk[size_t(r)*hidden + hh] * wgt;
                }
            }
        std::printf("%-16s %-40s %.3e\n", "hc combine",
                    "adds to the UNNORMALISED stream", worst_abs(cwant, cgot));
        CHECK(worst_abs(cwant, cgot) < 2e-5,
              "hc_combine does not match the host reference");

        // And it must NOT be the normalised stream: with these inputs the
        // two differ by ~1, so a swap is loud rather than a rounding
        // difference.  Without this the check above passes either way when
        // the injection gate happens to be near 1.
        {
            std::vector<float> swapped(hyper.size());
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < hc; ++c) {
                    const float wgt = 2.0f * qwen4_exp::sigmoid(
                        injo[size_t(r)*hc + c] / float(hc));
                    for (int hh = 0; hh < hidden; ++hh) {
                        const size_t i = size_t(r)*wide + size_t(c)*hidden + hh;
                        swapped[i] = got[i] + blk[size_t(r)*hidden + hh] * wgt;
                    }
                }
            const double apart = worst_abs(swapped, cgot);
            std::printf("%-16s %-40s %.3e\n", "hc combine A/B",
                        "vs adding to the NORMALISED stream", apart);
            CHECK(apart > 1e-3,
                  "combine cannot tell the raw stream from the normalised one "
                  "on this input, so the check above proves nothing");
        }

        sycl::free(d_hyper,q); sycl::free(d_norm,q); sycl::free(d_blk,q);
        sycl::free(d_out,q); sycl::free(d_nw,q); sycl::free(d_upo,q);
        sycl::free(d_mix,q); sycl::free(d_injo,q);
    }

    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
