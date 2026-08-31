// =====================================================================
//  ops.cpp  --  the elementwise operators that glue the layers together
//
//  None of these are bandwidth-critical on their own: a hidden state is
//  2048 floats, 8 KB. What matters is that there are MANY of them --
//  two norms per layer, RoPE on 10 layers, residuals everywhere -- so
//  every one is a kernel launch at ~5 us. 40 layers x 4 small ops is
//  160 launches, 0.8 ms, which is comparable to the entire fused MoE
//  block. They are therefore written to be fused wherever the data
//  dependencies allow, not as one kernel per mathematical operation.
// =====================================================================
#include "kernels.hpp"
#include <climits>
#include <cmath>
#include <limits>

namespace b70 {

// Stage-1 partials for the two-stage argmax. Allocated once by the engine
// (Grimoire::build) rather than per call: 512 floats plus 512 ints.
float*   g_argmax_pv = nullptr;
int32_t* g_argmax_pi = nullptr;


// ---------------------------------------------------------------------
// RMSNorm, fused with the residual add that always precedes it.
//
//   h    = h + residual          (in place, needed by the NEXT residual)
//   out  = h / sqrt(mean(h^2) + eps) * weight
//
// Doing these separately costs an extra launch and an extra round trip
// of the hidden state for no benefit -- the norm has to read h anyway.
// One work-group, sub-group reductions, hidden state staged in SLM.
// ---------------------------------------------------------------------
namespace {
// RMSNorm runs as a SINGLE work-group: the reduction needs every element
// visible to one barrier.  At WG=256 that is 256 threads on a 256-EU card,
// and it showed up as 14 us per norm -- 70 KB of traffic at ~5 GB/s, 128
// launches per token, 1.81 ms.  A wider work-group is the one knob that
// adds memory parallelism without splitting the reduction across groups.
int norm_wg() {
    static const int v = []{ const char* e = std::getenv("B70_NORM_WG");
        int x = (e && *e) ? std::atoi(e) : 1024;
        return (x == 256 || x == 512 || x == 1024) ? x : 1024; }();
    return v;
}
} // namespace

// ---------------------------------------------------------------------
//  RMSNorm across MANY work-groups.
//
//  The single-work-group form has to keep every element under one barrier,
//  so the whole norm runs on one Xe-core: 70 KB of traffic in 14 us at
//  WG=256, 7 us at WG=1024, against a 602 GB/s card.  Two kernels remove
//  the constraint -- pass 1 reduces per work-group into a partials array,
//  pass 2 sums the partials (G is ~20, so every thread can do it out of
//  L1) and writes the output.  Two launches cost ~1 us each here, against
//  the ~4 us saved, and the partials are summed in a fixed order so the
//  result stays run-to-run identical.
// ---------------------------------------------------------------------
namespace {
bool norm_split() {
    static const bool v = []{ const char* e = std::getenv("B70_NORM_SPLIT");
        return !(e && *e && std::atoi(e) == 0); }();
    return v;
}

float* norm_partials(sycl::queue& q, int need) {
    static float* buf = nullptr;
    static int    cap = 0;
    static sycl::queue* owner = nullptr;
    if (!buf || cap < need) {
        if (buf) sycl::free(buf, *owner);
        buf = sycl::malloc_device<float>(size_t(need), q);
        cap = need;
        owner = &q;
    }
    return buf;
}

sycl::event rmsnorm_split(sycl::queue& q, float* h, const float* r0,
                          const float* r1, const bf16_t* weight, float* out,
                          int n, float eps, bool store_h,
                          const std::vector<sycl::event>& deps) {
    constexpr int WG = 256;
    const int G = (n + WG - 1) / WG;
    float* part = norm_partials(q, G);

    auto e1 = q.submit([&](sycl::handler& hd) {
        hd.depends_on(deps);
        sycl::local_accessor<float, 1> sl(WG / SG_SIZE, hd);
        hd.parallel_for(
            sycl::nd_range<1>(size_t(G) * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg  = it.get_sub_group();
                const int  lid = int(it.get_local_id(0));
                const int  g   = int(it.get_group(0));
                const int  i   = g * WG + lid;
                float v = 0.0f;
                if (i < n) {
                    v = h[i];
                    if (r0) v += r0[i];
                    if (r1) v += r1[i];
                    if (store_h) h[i] = v;
                }
                float ss = sycl::reduce_over_group(sg, v * v, sycl::plus<float>());
                float* slp = sl.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                if (sg.get_local_id()[0] == 0) slp[sg.get_group_id()[0]] = ss;
                sycl::group_barrier(it.get_group());
                if (lid == 0) {
                    float t = 0.0f;
                    for (int j = 0; j < WG / SG_SIZE; ++j) t += slp[j];
                    part[g] = t;
                }
            });
    });

    return q.submit([&](sycl::handler& hd) {
        hd.depends_on(e1);
        hd.parallel_for(
            sycl::nd_range<1>(size_t(G) * WG, WG),
            [=](sycl::nd_item<1> it) {
                const int lid = int(it.get_local_id(0));
                const int g   = int(it.get_group(0));
                float total = 0.0f;
                for (int j = 0; j < G; ++j) total += part[j];
                const float scale = sycl::rsqrt(total / float(n) + eps);
                const int i = g * WG + lid;
                if (i < n)
                    out[i] = h[i] * scale * (1.0f + bf16_to_f32(weight[i]));
            });
    });
}
} // namespace

sycl::event launch_rmsnorm_residual(sycl::queue& q, float* h, const float* residual,
                                    const bf16_t* weight, float* out,
                                    int n, float eps,
                                    const std::vector<sycl::event>& deps = {}) {
    if (norm_split() && n >= 1024)
        return rmsnorm_split(q, h, residual, nullptr, weight, out, n, eps,
                             residual != nullptr, deps);
    const int WG = norm_wg();
    return q.submit([&](sycl::handler& hd) {
        hd.depends_on(deps);
        sycl::local_accessor<float, 1> partial(WG / SG_SIZE, hd);

        hd.parallel_for(
            sycl::nd_range<1>(WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg  = it.get_sub_group();
                const int  lid = int(it.get_local_id(0));
                const int  sgid= int(sg.get_group_id()[0]);
                const int  lane= int(sg.get_local_id()[0]);
                const int  wgz = int(it.get_local_range(0));
                const int  nsg = wgz / SG_SIZE;

                float ss = 0.0f;
                for (int i = lid; i < n; i += wgz) {
                    const float v = residual ? (h[i] + residual[i]) : h[i];
                    if (residual) h[i] = v;          // keep for the next residual
                    ss = sycl::fma(v, v, ss);
                }
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                if (lane == 0) partial[sgid] = ss;
                sycl::group_barrier(it.get_group());

                float total = 0.0f;
                for (int i = 0; i < nsg; ++i) total += partial[i];
                const float scale = sycl::rsqrt(total / float(n) + eps);

                // ZERO-CENTERED: Qwen3.5 stores the norm weight centred
                // on zero and applies (1 + w). The parameter is
                // initialised to zeros, so a stored value near 0 means a
                // scale near ONE. Multiplying by w alone shrinks every
                // norm output by ~10x -- forty times over, which leaves
                // the residual stream at the embedding and the logits
                // near-random.
                for (int i = lid; i < n; i += wgz)
                    out[i] = h[i] * scale * (1.0f + bf16_to_f32(weight[i]));
            });
    });
}

// Residual join for the two independent MoE branches, folded into the norm.
sycl::event launch_rmsnorm_residual2(sycl::queue& q, float* h,
                                     const float* r0, const float* r1,
                                     const bf16_t* weight, float* out,
                                     int n, float eps,
                                     const std::vector<sycl::event>& deps) {
    if (norm_split() && n >= 1024)
        return rmsnorm_split(q, h, r0, r1, weight, out, n, eps, true, deps);
    const int WG = norm_wg();
    return q.submit([&](sycl::handler& hd) {
        hd.depends_on(deps);
        sycl::local_accessor<float, 1> partial(WG / SG_SIZE, hd);
        hd.parallel_for(sycl::nd_range<1>(WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lid = int(it.get_local_id(0));
                const int sgid = int(sg.get_group_id()[0]);
                const int lane = int(sg.get_local_id()[0]);
                const int wgz = int(it.get_local_range(0));
                float ss = 0.0f;
                for (int i = lid; i < n; i += wgz) {
                    const float v = h[i] + r0[i] + r1[i];
                    h[i] = v; ss = sycl::fma(v, v, ss);
                }
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                if (lane == 0) partial[sgid] = ss;
                sycl::group_barrier(it.get_group());
                float total = 0.0f;
                for (int i = 0; i < wgz / SG_SIZE; ++i) total += partial[i];
                const float scale = sycl::rsqrt(total / float(n) + eps);
                for (int i = lid; i < n; i += wgz)
                    out[i] = h[i] * scale * (1.0f + bf16_to_f32(weight[i]));
            });
    });
}

// =====================================================================
//  W4A8 support kernels.
// =====================================================================

// MXFP4 (E2M1 nibbles, E8M0 per-32 scales) -> SYMMETRIC signed int4, group
// 128, packed low-nibble-first, plus one f32 scale per (row, group).
//
// NOTE: this quantizes an already-quantized weight.  It is correct and it is
// what lets the W4A8 path run against the existing MXFP4 artifact, but the
// shippable source is the BF16 original -- do not mistake this for the final
// quality path.
//
// Two passes over the group rather than staging 128 floats per work-item:
// a 128-float private array spills, and the re-read hits L1.
sycl::event launch_mxfp4_to_int4sym(sycl::queue& q, const uint8_t* payload,
                                    const uint8_t* mxscales,
                                    int64_t row_bytes, int64_t row_scales,
                                    uint8_t* out, float* ws, int N, int K,
                                    const std::vector<sycl::event>& deps) {
    constexpr int G = 128;
    const int kg = K / G;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(size_t(N), size_t(kg)), [=](sycl::id<2> id) {
            const int n = int(id[0]);
            const int g = int(id[1]);
            const int k0 = g * G;
            auto val = [&](int k) {
                const uint8_t byte = payload[int64_t(n) * row_bytes + (k >> 1)];
                const uint8_t nib  = (k & 1) ? uint8_t(byte >> 4) : uint8_t(byte & 0x0F);
                const float  sc    = e8m0_to_f32(
                    mxscales[int64_t(n) * row_scales + k / kMXBlock]);
                return e2m1_to_f32(nib) * sc;
            };
            float amax = 0.0f;
            for (int j = 0; j < G; ++j)
                amax = sycl::fmax(amax, sycl::fabs(val(k0 + j)));
            const float sc  = (amax > 0.0f) ? amax / 7.0f : 1.0f;
            const float inv = 1.0f / sc;
            ws[int64_t(n) * kg + g] = sc;
            for (int j = 0; j < G; j += 2) {
                int q0 = int(sycl::round(val(k0 + j)     * inv));
                int q1 = int(sycl::round(val(k0 + j + 1) * inv));
                q0 = sycl::clamp(q0, -8, 7);
                q1 = sycl::clamp(q1, -8, 7);
                out[int64_t(n) * (K / 2) + (k0 + j) / 2] =
                    uint8_t((uint32_t(q0) & 0x0Fu) | ((uint32_t(q1) & 0x0Fu) << 4));
            }
        });
    });
}

// ---------------------------------------------------------------------
// Rotary embedding, PARTIAL.
//
// Qwen3.5 sets partial_rotary_factor = 0.25, so with head_dim 256 only
// the first 64 dimensions are rotated and the remaining 192 pass through
// untouched. Rotating the whole head -- the default assumption -- gives
// a model that produces fluent text with no sense of word order beyond a
// few tokens, which is a miserable thing to debug after the fact.
//
// The config also specifies mrope with sections [11,11,10]. For text
// only, all three sections share the same position index, so mrope
// degenerates exactly to standard RoPE and is handled as such.
// ---------------------------------------------------------------------
sycl::event launch_rope(sycl::queue& q, float* x, int n_heads, int head_dim,
                        int pos, float theta, float partial_factor,
                        const std::vector<sycl::event>& deps = {}) {
    const int rot = int(head_dim * partial_factor) & ~1;   // must be even
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_heads) * (rot / 2)),
            [=](sycl::id<1> id) {
                const int t    = int(id[0]);
                const int head = t / (rot / 2);
                const int j    = t % (rot / 2);

                const float inv = sycl::exp(-float(2 * j) / float(rot)
                                            * sycl::log(theta));
                const float ang = float(pos) * inv;
                const float c = sycl::cos(ang), s = sycl::sin(ang);

                float* p = x + int64_t(head) * head_dim;
                const float a = p[j], b = p[j + rot / 2];
                p[j]           = a * c - b * s;
                p[j + rot / 2] = a * s + b * c;
            });
    });
}

// ---------------------------------------------------------------------
// SwiGLU: out[i] = silu(gate[i]) * up[i]
// ---------------------------------------------------------------------
sycl::event launch_swiglu(sycl::queue& q, const float* gate, const float* up,
                          float* out, int n,
                          const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            const int i = int(id[0]);
            const float g = gate[i];
            out[i] = (g / (1.0f + sycl::exp(-g))) * up[i];
        });
    });
}

// ---------------------------------------------------------------------
// L2 normalization per head. Gated DeltaNet requires q and k normalized;
// without it the delta rule is not contractive and the recurrent state
// diverges over a long context.
// ---------------------------------------------------------------------
sycl::event launch_l2norm_heads(sycl::queue& q, float* x, int n_heads, int dim,
                                const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(n_heads) * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  head = int(it.get_group(0));
                float* p = x + int64_t(head) * dim;

                float ss = 0.0f;
                for (int i = lane; i < dim; i += SG_SIZE) ss = sycl::fma(p[i], p[i], ss);
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                const float inv = sycl::rsqrt(ss + 1e-6f);
                for (int i = lane; i < dim; i += SG_SIZE) p[i] *= inv;
            });
    });
}

sycl::event launch_l2norm_heads_pair_bf16(sycl::queue& q,const float* qsrc,
    const float* ksrc,sycl_bf16* qdst,sycl_bf16* kdst,int n_heads,int dim,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(2)*n_heads*SG_SIZE,SG_SIZE),
          [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(SG_SIZE)]]{
            auto sg=it.get_sub_group();const int lane=int(sg.get_local_id()[0]);
            const int g=int(it.get_group(0)),head=g>=n_heads?g-n_heads:g;
            const float* src=(g>=n_heads?ksrc:qsrc)+int64_t(head)*dim;
            sycl_bf16* dst=(g>=n_heads?kdst:qdst)+int64_t(head)*dim;
            float ss=0.0f;
            for(int i=lane;i<dim;i+=SG_SIZE)ss=sycl::fma(src[i],src[i],ss);
            ss=sycl::reduce_over_group(sg,ss,sycl::plus<float>());
            float inv=sycl::rsqrt(ss+1e-6f);
            if(g<n_heads)inv*=sycl::rsqrt(float(dim));
            for(int i=lane;i<dim;i+=SG_SIZE)dst[i]=sycl_bf16(src[i]*inv);
          });});
}

sycl::event launch_l2norm_heads_pair_bf16_io(sycl::queue& q,sycl_bf16* qv,
    sycl_bf16* kv,int n_heads,int dim,const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(2)*n_heads*SG_SIZE,SG_SIZE),
          [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(SG_SIZE)]]{
            auto sg=it.get_sub_group();const int lane=int(sg.get_local_id()[0]);
            const int g=int(it.get_group(0)),head=g>=n_heads?g-n_heads:g;
            sycl_bf16* p=(g>=n_heads?kv:qv)+int64_t(head)*dim;
            float ss=0.0f;
            for(int i=lane;i<dim;i+=SG_SIZE){float v=float(p[i]);ss=sycl::fma(v,v,ss);}
            ss=sycl::reduce_over_group(sg,ss,sycl::plus<float>());
            float inv=sycl::rsqrt(ss+1e-6f);
            if(g<n_heads)inv*=sycl::rsqrt(float(dim));
            for(int i=lane;i<dim;i+=SG_SIZE)p[i]=sycl_bf16(float(p[i])*inv);
          });});
}

// ---------------------------------------------------------------------
// Argmax over the logits. A full 248320-wide softmax is unnecessary for
// greedy decoding, and for sampling only the top-k matters -- so the
// kernel returns the max and its index, and the host samples from a
// short list rather than normalizing a quarter-million floats.
// ---------------------------------------------------------------------
sycl::event launch_argmax(sycl::queue& q, const float* logits, int n,
                          int32_t* out_idx, float* out_val,
                          const std::vector<sycl::event>& deps = {}) {
    // Two stages. The single-work-group version scanned all 248320 logits
    // on one Xe-core: measured 165.7 us per token against 10.8 us for this
    // one, and it sat on the critical path between every pair of tokens.
    //
    // Stage 1: kArgmaxGroups work-groups each reduce a strided slice.
    // Stage 2: one sub-group reduces the partials. Tie-break is "lowest
    // index wins" in BOTH stages, so decoding stays deterministic and
    // matches the serial reference.
    constexpr int WG = 256;
    constexpr int NG = kArgmaxGroups;
    // Locals: a SYCL kernel may not capture a non-const global.
    float* const   pv = g_argmax_pv;
    int32_t* const pi = g_argmax_pi;

    q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1>   bv(WG / SG_SIZE, h);
        sycl::local_accessor<int32_t, 1> bi(WG / SG_SIZE, h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(NG) * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lid  = int(it.get_local_id(0));
                const int  sgid = int(sg.get_group_id()[0]);
                const int  lane = int(sg.get_local_id()[0]);
                const int  nsg  = WG / SG_SIZE;
                const int  g    = int(it.get_group(0));

                float   best = -std::numeric_limits<float>::infinity();
                int32_t bidx = INT_MAX;
                for (int i = g * WG + lid; i < n; i += NG * WG)
                    if (logits[i] > best) { best = logits[i]; bidx = i; }

                const float gm = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
                const int32_t gi = sycl::reduce_over_group(
                    sg, (best == gm) ? bidx : std::numeric_limits<int32_t>::max(),
                    sycl::minimum<int32_t>());
                if (lane == 0) { bv[sgid] = gm; bi[sgid] = gi; }
                sycl::group_barrier(it.get_group());
                if (lid == 0) {
                    float   m = bv[0];
                    int32_t x = bi[0];
                    for (int i = 1; i < nsg; ++i)
                        if (bv[i] > m || (bv[i] == m && bi[i] < x)) { m = bv[i]; x = bi[i]; }
                    pv[g] = m; pi[g] = x;
                }
            });
    });

    return q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(it.get_local_id(0));
                float   best = -std::numeric_limits<float>::infinity();
                int32_t bidx = INT_MAX;
                for (int i = lane; i < NG; i += SG_SIZE)
                    if (pv[i] > best || (pv[i] == best && pi[i] < bidx)) {
                        best = pv[i]; bidx = pi[i];
                    }
                const float gm = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
                const int32_t gi = sycl::reduce_over_group(
                    sg, (best == gm) ? bidx : std::numeric_limits<int32_t>::max(),
                    sycl::minimum<int32_t>());
                if (lane == 0) { *out_val = gm; *out_idx = gi; }
            });
    });
}

sycl::event launch_embed(sycl::queue& q, const bf16_t* table, int token,
                         float* out, int n,
                         const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            const int i = int(id[0]);
            out[i] = bf16_to_f32(table[int64_t(token) * n + i]);
        });
    });
}

} // namespace b70

namespace b70 {

// ---------------------------------------------------------------------
// Gated DeltaNet gate computation.
//   alpha = exp(-exp(A_log) * softplus(a + dt_bias))    per-head decay
//   beta  = sigmoid(b)                                  per-head write gate
// Both are tiny [n_heads] vectors; one flat launch.
// ---------------------------------------------------------------------
sycl::event launch_deltanet_gates(sycl::queue& q, const float* a_raw,
                                  const float* b_raw, const bf16_t* A_log,
                                  const bf16_t* dt_bias, float* alpha,
                                  float* beta, int n_heads,
                                  const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_heads)), [=](sycl::id<1> id) {
            const int i = int(id[0]);
            const float dt = a_raw[i] + bf16_to_f32(dt_bias[i]);
            // softplus, guarded: for large dt the exp overflows and the
            // decay silently becomes NaN, which poisons the state forever.
            const float sp = dt > 20.0f ? dt : sycl::log(1.0f + sycl::exp(dt));
            alpha[i] = sycl::exp(-sycl::exp(bf16_to_f32(A_log[i])) * sp);
            beta[i]  = 1.0f / (1.0f + sycl::exp(-b_raw[i]));
        });
    });
}

// out[i] = x[i] * silu(z[i])   -- the DeltaNet output gate
sycl::event launch_gate_silu(sycl::queue& q, const float* x, const float* z,
                             float* out, int n,
                             const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            const int i = int(id[0]);
            const float zv = z[i];
            out[i] = x[i] * (zv / (1.0f + sycl::exp(-zv)));
        });
    });
}

// DeltaNet's per-head RMSNorm and output gate in one pass.
sycl::event launch_rmsnorm_gate_silu(sycl::queue& q, float* x, const float* z,
                                     const bf16_t* w, int n_heads, int dim,
                                     float eps,
                                     const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(n_heads) * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                float* p = x + int64_t(it.get_group(0)) * dim;
                const float* g = z + int64_t(it.get_group(0)) * dim;
                float ss = 0.0f;
                for (int i = lane; i < dim; i += SG_SIZE) ss = sycl::fma(p[i], p[i], ss);
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                const float scale = sycl::rsqrt(ss / float(dim) + eps);
                for (int i = lane; i < dim; i += SG_SIZE) {
                    const float zv = g[i];
                    p[i] = p[i] * scale * bf16_to_f32(w[i])
                         * (zv / (1.0f + sycl::exp(-zv)));
                }
            });
    });
}

// Q and K normalization plus partial RoPE in one kernel.
sycl::event launch_qk_norm_rope(sycl::queue& q, float* qv, float* kv,
                                const bf16_t* qw, const bf16_t* kw,
                                int q_heads, int k_heads, int dim,
                                const int32_t* d_pos, float theta,
                                float partial_factor, float eps,
                                const std::vector<sycl::event>& deps) {
    const int rot = int(dim * partial_factor) & ~1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(q_heads + k_heads) * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int head = int(it.get_group(0));
                const bool is_q = head < q_heads;
                const int hi = is_q ? head : head - q_heads;
                float* p = (is_q ? qv : kv) + int64_t(hi) * dim;
                const bf16_t* w = is_q ? qw : kw;
                float ss = 0.0f;
                for (int i = lane; i < dim; i += SG_SIZE) ss = sycl::fma(p[i], p[i], ss);
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                const float scale = sycl::rsqrt(ss / float(dim) + eps);
                for (int i = lane; i < dim; i += SG_SIZE)
                    p[i] *= scale * (1.0f + bf16_to_f32(w[i]));
                sycl::group_barrier(sg);
                const int pos = *d_pos;
                for (int j = lane; j < rot / 2; j += SG_SIZE) {
                    const float inv = sycl::exp(-float(2 * j) / float(rot) * sycl::log(theta));
                    const float ang = float(pos) * inv;
                    const float c = sycl::cos(ang), s = sycl::sin(ang);
                    const float a = p[j], b = p[j + rot / 2];
                    p[j] = a * c - b * s;
                    p[j + rot / 2] = a * s + b * c;
                }
            });
    });
}

// Per-head RMSNorm sharing one [dim] weight vector across all heads.
sycl::event launch_rmsnorm_heads(sycl::queue& q, float* x, const bf16_t* w,
                                 int n_heads, int dim, float eps,
                                 bool zero_centered,
                                 const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(n_heads) * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                float* p = x + int64_t(it.get_group(0)) * dim;
                float ss = 0.0f;
                for (int i = lane; i < dim; i += SG_SIZE) ss = sycl::fma(p[i], p[i], ss);
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                const float s = sycl::rsqrt(ss / float(dim) + eps);
                // Two conventions in this model, both per-head:
                //   RMSNormGated  (DeltaNet)  init ones  -> w * x
                //   RMSNorm       (q/k norms) init zeros -> (1 + w) * x
                for (int i = lane; i < dim; i += SG_SIZE) {
                    const float wv = bf16_to_f32(w[i]);
                    p[i] = p[i] * s * (zero_centered ? (1.0f + wv) : wv);
                }
            });
    });
}

// dst += src
sycl::event launch_add(sycl::queue& q, float* dst, const float* src, int n,
                       const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)),
                       [=](sycl::id<1> id) { dst[id[0]] += src[id[0]]; });
    });
}

sycl::event launch_add_f16_round(sycl::queue& q,float* dst,const float* src,
    int n,const std::vector<sycl::event>& deps){
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)),[=](sycl::id<1> id){
            dst[id[0]]=float(sycl::half(dst[id[0]]+src[id[0]]));
        });
    });
}

// Copy a projected K/V vector into the paged cache at `pos`.
// K is stored D-major so the attention kernel's 16-lane score step reads
// contiguous floats; V stays D-minor because its accumulator is
// partitioned over d. See src/attention.cpp.
sycl::event launch_kv_append(sycl::queue& q, const float* k, const float* v,
                             float* k_cache, float* v_cache, int pos,
                             int n_kv_heads, int head_dim, int seq_cap,
                             const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_kv_heads) * head_dim),
            [=](sycl::id<1> id) {
                const int t    = int(id[0]);
                const int head = t / head_dim;
                const int d    = t % head_dim;
                k_cache[(int64_t(head) * head_dim + d) * seq_cap + pos] = k[t];
                v_cache[(int64_t(head) * seq_cap + pos) * head_dim + d] = v[t];
            });
    });
}

// Top-k over the router logits, on device. n_experts is 256 here, so a
// single work-group with a partial sort is far cheaper than a full sort.
namespace {
// Fixed-width specialization: with n_experts == SG_SIZE * SLOTS every lane
// holds its whole slice in registers and the SLM staging disappears.
// Measured on a B70 at n_experts=256: 16.6 us staged through SLM, 7.4 us
// in registers, against a 4.7 us floor for an empty 16-thread kernel. The
// generic path below still handles any other expert count.
template <int SLOTS>
sycl::event router_topk_fixed(sycl::queue& q, const float* logits, int top_k,
                              int32_t* out_expert, float* out_weight,
                              bool normalize,
                              const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(it.get_local_id(0));

                float r[SLOTS];
                #pragma unroll
                for (int slot = 0; slot < SLOTS; ++slot)
                    r[slot] = logits[lane + slot * SG_SIZE];

                uint64_t mine = 0;
                for (int s = 0; s < top_k; ++s) {
                    float cv = -std::numeric_limits<float>::infinity();
                    int   ci = INT_MAX, cs = -1;
                    #pragma unroll
                    for (int slot = 0; slot < SLOTS; ++slot) {
                        if (mine & (1ull << slot)) continue;
                        const int e = lane + slot * SG_SIZE;
                        if (r[slot] > cv || (r[slot] == cv && e < ci)) {
                            cv = r[slot]; ci = e; cs = slot;
                        }
                    }
                    const float bv = sycl::reduce_over_group(sg, cv,
                                         sycl::maximum<float>());
                    const int   bi = sycl::reduce_over_group(
                                         sg, (cv == bv && ci != INT_MAX) ? ci : INT_MAX,
                                         sycl::minimum<int>());
                    if (ci == bi && cs >= 0) mine |= (1ull << cs);
                    if (lane == 0) { out_expert[s] = bi; out_weight[s] = bv; }
                }

                if (lane == 0) {
                    float m = out_weight[0];
                    for (int s = 1; s < top_k; ++s) m = sycl::fmax(m, out_weight[s]);
                    float sum = 0.0f;
                    for (int s = 0; s < top_k; ++s) {
                        out_weight[s] = sycl::exp(out_weight[s] - m);
                        sum += out_weight[s];
                    }
                    if (normalize && sum > 0.0f)
                        for (int s = 0; s < top_k; ++s) out_weight[s] /= sum;
                }
            });
    });
}
} // namespace

sycl::event launch_router_topk(sycl::queue& q, const float* logits,
                               int n_experts, int top_k,
                               int32_t* out_expert, float* out_weight,
                               bool normalize,
                               const std::vector<sycl::event>& deps = {}) {
    // One sub-group, n_experts logits staged in SLM, top_k rounds of a
    // sub-group argmax.
    //
    // The previous version ran on nd_range(1,1) -- a SINGLE work-item
    // walking top_k * n_experts dependent global loads with a 512-byte
    // private `taken` array that the compiler puts in scratch. Measured
    // on a B70 that cost 215 us per layer, 8.6 ms per token across 40
    // layers: more than half of decode, for 0.26 MB of weights. It was
    // invisible on an out-of-order queue because it overlapped with the
    // real work it was blocking.
    //
    // Tie-break is preserved EXACTLY: the serial loop used strict `>`
    // while scanning e ascending, so equal logits keep the LOWEST index.
    // The parallel form reproduces that by reducing the value first, then
    // taking the minimum index among the lanes that hold it. Selection is
    // tracked in a bitmask rather than by writing -inf into the staged
    // logits, so a genuinely -inf logit cannot be mistaken for an expert
    // that was already picked.
    // Qwen3.5-MoE routes 256 experts; that lands on the register path.
    if (n_experts == SG_SIZE * 16)
        return router_topk_fixed<16>(q, logits, top_k, out_expert, out_weight,
                                     normalize, deps);

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1>    lg(size_t(n_experts), h);

        h.parallel_for(
            sycl::nd_range<1>(SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(it.get_local_id(0));

                for (int e = lane; e < n_experts; e += SG_SIZE) lg[e] = logits[e];
                sycl::group_barrier(it.get_group());

                // Every lane owns a strided slice of the logits, so
                // "already selected" is a PRIVATE bitmask over that slice
                // -- 16 elements at n_experts=256. The winner of each
                // round comes out of the reduction on every lane, so no
                // lane needs to publish anything and the per-round
                // barrier disappears: 8 barriers and 8 SLM round-trips
                // per layer, x40 layers, for a 256-element selection.
                // uint64 slice mask: one bit per element this lane owns,
                // so n_experts up to SG_SIZE*64 = 1024 is covered. The
                // loop it replaced was bounded at 512 by a private array.
                uint64_t mine = 0;
                for (int s = 0; s < top_k; ++s) {
                    float cv = -std::numeric_limits<float>::infinity();
                    int   ci = INT_MAX;
                    int   cs = -1;
                    for (int e = lane, slot = 0; e < n_experts; e += SG_SIZE, ++slot) {
                        if (mine & (1ull << slot)) continue;
                        const float v = lg[e];
                        if (v > cv || (v == cv && e < ci)) { cv = v; ci = e; cs = slot; }
                    }
                    const float bv = sycl::reduce_over_group(sg, cv,
                                         sycl::maximum<float>());
                    const int   bi = sycl::reduce_over_group(
                                         sg, (cv == bv && ci != INT_MAX) ? ci : INT_MAX,
                                         sycl::minimum<int>());
                    if (ci == bi && cs >= 0) mine |= (1ull << cs);
                    if (lane == 0) { out_expert[s] = bi; out_weight[s] = bv; }
                }

                // softmax over the selected weights; top_k is tiny, so one
                // lane is the cheapest correct thing here.
                if (lane == 0) {
                    float m = out_weight[0];
                    for (int s = 1; s < top_k; ++s) m = sycl::fmax(m, out_weight[s]);
                    float sum = 0.0f;
                    for (int s = 0; s < top_k; ++s) {
                        out_weight[s] = sycl::exp(out_weight[s] - m);
                        sum += out_weight[s];
                    }
                    if (normalize && sum > 0.0f)
                        for (int s = 0; s < top_k; ++s) out_weight[s] /= sum;
                }
            });
    });
}

// One subgroup per prompt token. Routing tables are laid out [M][top_k].
// The reduction and lowest-index tie break are identical to decode.
sycl::event launch_router_topk_batched(
    sycl::queue& q, const float* logits, int tokens, int n_experts, int top_k,
    int32_t* out_expert, float* out_weight, bool normalize,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(tokens) * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int t = int(it.get_group(0));
                const float* row = logits + int64_t(t) * n_experts;
                int32_t* oe = out_expert + int64_t(t) * top_k;
                float* ow = out_weight + int64_t(t) * top_k;
                uint64_t taken = 0;
                for (int s = 0; s < top_k; ++s) {
                    float cv = -std::numeric_limits<float>::infinity();
                    int ci = INT_MAX, cs = -1;
                    for (int e = lane, slot = 0; e < n_experts;
                         e += SG_SIZE, ++slot) {
                        if (slot < 64 && (taken & (1ull << slot))) continue;
                        const float v = row[e];
                        if (v > cv || (v == cv && e < ci)) {
                            cv = v; ci = e; cs = slot;
                        }
                    }
                    const float bv = sycl::reduce_over_group(
                        sg, cv, sycl::maximum<float>());
                    const int bi = sycl::reduce_over_group(
                        sg, (cv == bv && ci != INT_MAX) ? ci : INT_MAX,
                        sycl::minimum<int>());
                    if (ci == bi && cs >= 0 && cs < 64) taken |= 1ull << cs;
                    if (lane == 0) { oe[s] = bi; ow[s] = bv; }
                }
                if (lane == 0 && normalize) {
                    float m = ow[0];
                    for (int s = 1; s < top_k; ++s) m = sycl::fmax(m, ow[s]);
                    float sum = 0.0f;
                    for (int s = 0; s < top_k; ++s) {
                        ow[s] = sycl::exp(ow[s] - m); sum += ow[s];
                    }
                    if (sum > 0.0f)
                        for (int s = 0; s < top_k; ++s) ow[s] /= sum;
                }
            });
    });
}

sycl::event launch_router_topk_bf16_batched(
    sycl::queue& q, const sycl_bf16* logits, int tokens, int n_experts,
    int top_k, int32_t* out_expert, float* out_weight, bool normalize,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(tokens) * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int t = int(it.get_group(0));
                const sycl_bf16* row = logits + int64_t(t) * n_experts;
                int32_t* oe = out_expert + int64_t(t) * top_k;
                float* ow = out_weight + int64_t(t) * top_k;
                uint64_t taken = 0;
                for (int s = 0; s < top_k; ++s) {
                    float cv = -std::numeric_limits<float>::infinity();
                    int ci = INT_MAX, cs = -1;
                    for (int e = lane, slot = 0; e < n_experts;
                         e += SG_SIZE, ++slot) {
                        if (slot < 64 && (taken & (1ull << slot))) continue;
                        const float v = float(row[e]);
                        if (v > cv || (v == cv && e < ci)) {
                            cv = v; ci = e; cs = slot;
                        }
                    }
                    const float bv = sycl::reduce_over_group(
                        sg, cv, sycl::maximum<float>());
                    const int bi = sycl::reduce_over_group(
                        sg, (cv == bv && ci != INT_MAX) ? ci : INT_MAX,
                        sycl::minimum<int>());
                    if (ci == bi && cs >= 0 && cs < 64) taken |= 1ull << cs;
                    if (lane == 0) { oe[s] = bi; ow[s] = bv; }
                }
                if (lane == 0 && normalize) {
                    float m = ow[0];
                    for (int s = 1; s < top_k; ++s) m = sycl::fmax(m, ow[s]);
                    float sum = 0.0f;
                    for (int s = 0; s < top_k; ++s) {
                        ow[s] = sycl::exp(ow[s] - m); sum += ow[s];
                    }
                    if (sum > 0.0f)
                        for (int s = 0; s < top_k; ++s) ow[s] /= sum;
                }
            });
    });
}

// x[i] *= sigmoid(gate)  -- the shared-expert scalar gate
sycl::event launch_scale_by_sigmoid(sycl::queue& q, float* x, const float* g,
                                    int n,
                                    const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            x[id[0]] *= 1.0f / (1.0f + sycl::exp(-g[0]));
        });
    });
}

} // namespace b70

namespace b70 {

// ---------------------------------------------------------------------
// Device-position variants.
//
// A recorded command graph is replayed unchanged every token, so nothing
// inside it may capture a host value that changes. RoPE's position and
// the KV write index both change per token, so they are read from device
// memory instead of baked into the kernel at submit time. This is what
// makes graph capture possible at all.
// ---------------------------------------------------------------------
sycl::event launch_rope_dev(sycl::queue& q, float* x, int n_heads, int head_dim,
                            const int32_t* d_pos, float theta, float partial_factor,
                            const std::vector<sycl::event>& deps = {}) {
    const int rot = int(head_dim * partial_factor) & ~1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_heads) * (rot / 2)),
            [=](sycl::id<1> id) {
                const int t    = int(id[0]);
                const int head = t / (rot / 2);
                const int j    = t % (rot / 2);
                const int pos  = d_pos[0];

                const float inv = sycl::exp(-float(2 * j) / float(rot)
                                            * sycl::log(theta));
                const float ang = float(pos) * inv;
                const float c = sycl::cos(ang), s = sycl::sin(ang);
                float* p = x + int64_t(head) * head_dim;
                const float a = p[j], b = p[j + rot / 2];
                p[j]           = a * c - b * s;
                p[j + rot / 2] = a * s + b * c;
            });
    });
}

sycl::event launch_kv_append_dev(sycl::queue& q, const float* k, const float* v,
                                 uint8_t* k_cache, uint8_t* v_cache,
                                 const int32_t* d_pos, int n_kv_heads,
                                 int head_dim, int seq_cap,
                                 const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_kv_heads) * head_dim),
            [=](sycl::id<1> id) {
                const int t    = int(id[0]);
                const int head = t / head_dim;
                const int d    = t % head_dim;
                const int pos  = d_pos[0];
                if (pos >= seq_cap) return;
                k_cache[(int64_t(head) * head_dim + d) * seq_cap + pos] =
                    f32_to_e4m3(k[t]);
                v_cache[(int64_t(head) * seq_cap + pos) * head_dim + d] =
                    f32_to_e4m3(v[t]);
            });
    });
}

// Advance the position. Runs inside the graph, so replay is self-contained.
sycl::event launch_incr_pos(sycl::queue& q, int32_t* d_pos,
                            const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() { d_pos[0] += 1; });
    });
}

sycl::event launch_incr_pos2(sycl::queue& q, int32_t* a, int32_t* b,
                             const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() { ++(*a); ++(*b); });
    });
}

} // namespace b70

namespace b70 {

// ---------------------------------------------------------------------
// De-interleave [q | gate] per head.
//
// With attn_output_gate the q projection emits 2*head_dim per head:
// query first, then the gate. They are interleaved PER HEAD, not
// concatenated as two halves of the whole tensor -- getting that wrong
// mixes gate values into the queries of the next head.
// ---------------------------------------------------------------------
sycl::event launch_split_qgate(sycl::queue& q, const float* src, float* qout,
                               float* gout, int n_heads, int head_dim,
                               const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_heads) * head_dim),
            [=](sycl::id<1> id) {
                const int t    = int(id[0]);
                const int head = t / head_dim;
                const int d    = t % head_dim;
                const int64_t base = int64_t(head) * 2 * head_dim;
                qout[t] = src[base + d];
                gout[t] = src[base + head_dim + d];
            });
    });
}

// x *= sigmoid(gate), elementwise
sycl::event launch_gate_sigmoid_mul(sycl::queue& q, float* x, const float* g,
                                    int n,
                                    const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            const int i = int(id[0]);
            x[i] *= 1.0f / (1.0f + sycl::exp(-g[i]));
        });
    });
}

} // namespace b70

namespace b70 {

sycl::event launch_dflash_store_tap(
    sycl::queue& q, const float* src, float* taps, int rows, int hidden,
    int tap_count, int start_pos, int tap, const std::vector<sycl::event>& deps,
    bool fp16_round) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows) * hidden), [=](sycl::id<1> id) {
            const int64_t z = id[0];
            const int row = int(z / hidden), col = int(z % hidden);
            taps[(int64_t(start_pos + row) * tap_count + tap) * hidden + col] =
                fp16_round ? float(sycl::half(src[z])) : src[z];
        });
    });
}

sycl::event launch_dflash_store_tap_dev(
    sycl::queue& q, const float* src, float* taps, int hidden,
    int tap_count, const int32_t* position, int tap,
    const std::vector<sycl::event>& deps, bool fp16_round) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(hidden)), [=](sycl::id<1> id) {
            const int col = int(id[0]);
            taps[(int64_t(*position) * tap_count + tap) * hidden + col] =
                fp16_round ? float(sycl::half(src[col])) : src[col];
        });
    });
}

// Diagnostic: sum of squares, max magnitude, and NaN/Inf counts of a
// device buffer. Copying the whole tensor back per stage would dominate
// the timing, so the reduction runs on device and only 4 floats return.
sycl::event launch_probe(sycl::queue& q, const float* x, int n, float* out4,
                         const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.single_task([=]() {
            float ss = 0.0f, mx = 0.0f;
            int nan_n = 0, inf_n = 0;
            for (int i = 0; i < n; ++i) {
                const float v = x[i];
                if (v != v) { ++nan_n; continue; }
                if (sycl::isinf(v)) { ++inf_n; continue; }
                ss += v * v;
                const float a = v < 0 ? -v : v;
                if (a > mx) mx = a;
            }
            out4[0] = sycl::sqrt(ss / float(n));   // rms
            out4[1] = mx;
            out4[2] = float(nan_n);
            out4[3] = float(inf_n);
        });
    });
}

sycl::event launch_dflash2_grouped_conv(
    sycl::queue& q, const float* x, const float* coefficients,
    const bf16_t* base, float* out, int rows, int hidden, int taps,
    int group_size, int block_size, int side,
    const std::vector<sycl::event>& deps) {
    const int groups=hidden/group_size;
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows)*hidden),[=](sycl::id<1> id){
            const int64_t z=id[0];
            const int r=int(z/hidden),c=int(z-int64_t(r)*hidden);
            const int g=c/group_size,pos=r%block_size;
            float acc=0.0f;
            for(int t=0;t<taps&&t<=pos;++t){
                const float bv=bf16_to_f32(base[(side*taps+t)*hidden+c]);
                const int64_t ci=((int64_t(r)*2+side)*taps+t)*groups+g;
                const float k=float(sycl_bf16(bv+coefficients[ci]));
                const float term=float(sycl_bf16(k*x[int64_t(r-t)*hidden+c]));
                acc=t==0?term:float(sycl_bf16(acc+term));
            }
            out[z]=acc;
        });
    });
}

sycl::event launch_topk16_rows(
    sycl::queue& q, const float* logits, int rows, int vocab,
    int32_t* out_ids, float* out_values,
    const std::vector<sycl::event>& deps) {
    constexpr int K=16;
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(rows)*SG_SIZE,SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg=it.get_sub_group();
                const int lane=int(sg.get_local_id()[0]);
                const int row=int(it.get_group(0));
                int32_t chosen[K];
                #pragma unroll
                for(int j=0;j<K;++j)chosen[j]=-1;
                for(int pick=0;pick<K;++pick){
                    float best=-std::numeric_limits<float>::infinity();
                    int32_t bid=INT_MAX;
                    for(int v=lane;v<vocab;v+=SG_SIZE){
                        bool used=false;
                        #pragma unroll
                        for(int j=0;j<K;++j)if(j<pick&&chosen[j]==v)used=true;
                        if(used)continue;
                        const float x=logits[int64_t(row)*vocab+v];
                        if(x>best||(x==best&&v<bid)){best=x;bid=v;}
                    }
                    const float win=sycl::reduce_over_group(sg,best,sycl::maximum<float>());
                    const int32_t mine=(best==win)?bid:INT_MAX;
                    const int32_t wid=sycl::reduce_over_group(sg,mine,sycl::minimum<int32_t>());
                    chosen[pick]=wid;
                    if(lane==0){
                        out_ids[int64_t(row)*K+pick]=wid;
                        out_values[int64_t(row)*K+pick]=win;
                    }
                }
            });
    });
}

sycl::event launch_dflash2_selector_edges(
    sycl::queue& q, const bf16_t* predecessor, const bf16_t* successor,
    const int32_t* candidate_ids, const float* unary,
    const float* projected_hidden, int32_t anchor_token,
    float* scores, int steps, int top_k, int rank,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(steps)*top_k*SG_SIZE,SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg=it.get_sub_group();
                const int lane=int(sg.get_local_id()[0]);
                const int slot=int(it.get_group(0));
                const int step=slot/top_k,p=slot%top_k;
                const int32_t pid=step==0?anchor_token:
                    candidate_ids[int64_t(step-1)*top_k+p];
                for(int c=0;c<top_k;++c){
                    const int32_t cid=candidate_ids[int64_t(step)*top_k+c];
                    float acc=0.0f;
                    for(int r=lane;r<rank;r+=SG_SIZE){
                        const float pv=bf16_to_f32(predecessor[int64_t(pid)*rank+r]);
                        const float hv=float(sycl_bf16(projected_hidden[int64_t(step)*rank+r]));
                        const float gated=float(sycl_bf16(pv*hv));
                        acc=sycl::fma(gated,
                            bf16_to_f32(successor[int64_t(cid)*rank+r]),acc);
                    }
                    acc=sycl::reduce_over_group(sg,acc,sycl::plus<float>());
                    if(lane==0)scores[(int64_t(step)*top_k+p)*top_k+c]=
                        unary[int64_t(step)*top_k+c]+float(sycl_bf16(acc));
                }
            });
    });
}

sycl::event launch_dflash2_path_walk(
    sycl::queue& q, const float* scores, const int32_t* candidate_ids,
    int32_t* tokens, int steps, int top_k,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.single_task([=](){
            int previous=0;
            for(int step=0;step<steps;++step){
                const float* row=scores+(int64_t(step)*top_k+previous)*top_k;
                float best=-std::numeric_limits<float>::infinity();
                int pick=top_k;
                for(int c=0;c<top_k;++c)if(row[c]>best){best=row[c];pick=c;}
                if(pick==top_k)pick=0;
                tokens[step]=candidate_ids[int64_t(step)*top_k+pick];
                previous=pick;
            }
        });
    });
}

} // namespace b70
