// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

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
#include <sycl/ext/intel/esimd.hpp>
#include <limits>
#include "kernels.hpp"
#include "b70/int8_embedding.hpp"
#include <climits>
#include <cmath>
#include <limits>

namespace b70 {

// The norm convention for the loaded model, set once by Grimoire::build.
// K2 takes the variance per contiguous group and applies the weight
// DIRECTLY (its parameter is initialised to ones); Qwen3.5 takes it over
// the whole row and applies (1 + w).  Holding it here rather than at ~60
// call sites means a path cannot silently keep the wrong one.
int   g_norm_groups = 1;
float g_norm_weight_offset = 1.0f;
void set_norm_convention(int groups, float weight_offset) {
    g_norm_groups = groups > 0 ? groups : 1;
    g_norm_weight_offset = weight_offset;
}
// What set_norm_convention last installed.  The capability banner has to
// report the convention that is actually LIVE: it used to re-derive it
// from cfg with a second ternary that only knew about K2, so a gemma-4
// load printed "(1 + w)" while the kernels ran plain w.  A banner line
// that disagrees with the engine is worse than no banner -- the whole
// point of printing it is that a reader can trust it.
void get_norm_convention(int* groups, float* weight_offset) {
    if (groups) *groups = g_norm_groups;
    if (weight_offset) *weight_offset = g_norm_weight_offset;
}
bool norm_is_grouped(int hidden) {
    return (g_norm_groups > 1 || g_norm_weight_offset != 1.0f)
        && g_norm_groups > 0 && (hidden % g_norm_groups) == 0;
}


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
    if (norm_is_grouped(n))
        return launch_rmsnorm_grouped(q, h, residual, nullptr, weight, out,
                                      nullptr, 1, n, g_norm_groups, eps,
                                      g_norm_weight_offset, deps);
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
    // K2's grouped norm (plain w, per-group variance) -- as in
    // launch_rmsnorm_residual.  Without this the decode MoE join silently
    // applied Qwen's ungrouped (1 + w) to every K2 layer after the first.
    if (norm_is_grouped(n))
        return launch_rmsnorm_grouped(q, h, r0, r1, weight, out, nullptr, 1, n,
                                      g_norm_groups, eps, g_norm_weight_offset, deps);
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

// FP16 [N,K] -> symmetric signed int4, group 128. This is the quality path
// for proposal-only weights that are available from the original checkpoint:
// quantize once from the same FP16 values used by the reference drafter.
sycl::event launch_f16_to_int4sym(sycl::queue& q, const sycl::half* src,
                                  uint8_t* out, float* ws, int N, int K,
                                  const std::vector<sycl::event>& deps) {
    constexpr int G = 128;
    const int kg = K / G;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(size_t(N), size_t(kg)),
            [=](sycl::id<2> id) {
                const int n = int(id[0]);
                const int g = int(id[1]);
                const int k0 = g * G;
                const sycl::half* row = src + int64_t(n) * K;
                float amax = 0.0f;
                for (int j = 0; j < G; ++j)
                    amax = sycl::fmax(amax, sycl::fabs(float(row[k0 + j])));
                const float sc = (amax > 0.0f) ? amax / 7.0f : 1.0f;
                const float inv = 1.0f / sc;
                ws[int64_t(n) * kg + g] = sc;
                for (int j = 0; j < G; j += 2) {
                    int q0 = int(sycl::round(float(row[k0 + j]) * inv));
                    int q1 = int(sycl::round(float(row[k0 + j + 1]) * inv));
                    q0 = sycl::clamp(q0, -8, 7);
                    q1 = sycl::clamp(q1, -8, 7);
                    out[int64_t(n) * (K / 2) + (k0 + j) / 2] =
                        uint8_t((uint32_t(q0) & 0x0Fu) |
                                ((uint32_t(q1) & 0x0Fu) << 4));
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
// GeGLU: gelu(gate) * up, the tanh approximation.
//
// Gemma asks for "gelu_pytorch_tanh", which is NOT the error-function
// gelu and NOT silu.  Substituting silu -- which is what this engine did
// until the loader learned to read hidden_act -- loads cleanly and
// produces fluent text from the wrong model.
//
//   gelu_tanh(x) = 0.5x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
//
// The two constants are PyTorch's, kept as literals so this reads the
// same as the reference rather than being rederived.
// ---------------------------------------------------------------------
inline float gelu_tanh(float x) {
    constexpr float kA = 0.7978845608028654f;   // sqrt(2/pi)
    constexpr float kB = 0.044715f;
    return 0.5f * x * (1.0f + sycl::tanh(kA * (x + kB * x * x * x)));
}

sycl::event launch_geglu(sycl::queue& q, const float* gate, const float* up,
                         float* out, int n,
                         const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            const int i = int(id[0]);
            out[i] = gelu_tanh(gate[i]) * up[i];
        });
    });
}

// Batched form over a [rows][2*inter] gate|up block, matching
// launch_swiglu_batched's layout so the two are interchangeable at the
// call site and only the activation differs.
sycl::event launch_geglu_batched(sycl::queue& q, const float* gu, float* out,
                                 int rows, int inter,
                                 const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows) * size_t(inter)),
            [=](sycl::id<1> id) {
                const int r = int(id[0] / inter), c = int(id[0] % inter);
                const float* row = gu + int64_t(r) * 2 * inter;
                out[int64_t(r) * inter + c] = gelu_tanh(row[c]) * row[inter + c];
            });
    });
}

// ---------------------------------------------------------------------
// Proportional RoPE -- gemma-4 full-attention layers.
//
// This is NOT this engine's partial_rope, and the difference is silent.
// From ref/gemma4_proportional_rope.py:
//
//   rope_angles = int(factor * head_dim / 2)
//   inv_freq[i] = base ** -(2i / HEAD_DIM)   for i < rope_angles
//   inv_freq[i] = 0                          for the rest
//
// Two things follow.  The exponent divides by the FULL head_dim, not by
// the rotated width -- so at head_dim 512 and factor 0.25 the
// frequencies are base^-(2i/512), where partial_rope would give
// base^-(2i/128).  And the cos/sin table stays head_dim/2 wide, so the
// pairing is dim i with dim i + head_dim/2 (i + 256), where partial_rope
// pairs j with j + rot/2 (j + 64).  Different frequencies AND a
// different pairing; either alone changes every number.
//
// Dimensions past rope_angles have inv_freq 0, which is cos 1 / sin 0 --
// the identity.  They are left untouched rather than multiplied by one.
// ---------------------------------------------------------------------
sycl::event launch_rope_proportional(sycl::queue& q, float* x, int n_heads,
                                     int head_dim, const int32_t* d_pos,
                                     float theta, float partial_factor,
                                     const std::vector<sycl::event>& deps,
                                     float freq_divisor) {
    const int half   = head_dim / 2;
    const int angles = int(partial_factor * float(head_dim) / 2.0f);
    const int rot    = angles < half ? angles : half;
    if (rot <= 0) return q.submit([&](sycl::handler& h) {
        h.depends_on(deps); h.single_task([=](){}); });
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_heads) * size_t(rot)),
            [=](sycl::id<1> id) {
                const int t    = int(id[0]);
                const int head = t / rot;
                const int i    = t % rot;
                const int pos  = d_pos[0];
                // exponent over head_dim, NOT over the rotated width
                // inv_freq /= factor, the reference's last line.  1.0 is
                // the identity, which is every checkpoint seen so far.
                const float inv = sycl::exp(-float(2 * i) / float(head_dim)
                                            * sycl::log(theta)) / freq_divisor;
                const float ang = float(pos) * inv;
                const float c = sycl::cos(ang), s = sycl::sin(ang);
                float* p = x + int64_t(head) * head_dim;
                const float a = p[i], b = p[i + half];   // pairing over half
                p[i]        = a * c - b * s;
                p[i + half] = a * s + b * c;
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

sycl::event launch_embed_int8(sycl::queue& q, const int8_t* table,
    const bf16_t* scales, int token, float* out, int hidden,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(hidden)), [=](sycl::id<1> id) {
            out[id[0]] = int8_embedding_value(table[int64_t(token) * hidden + id[0]], scales[token]);
        });
    });
}

sycl::event launch_embed_int8_batched(sycl::queue& q, const int8_t* table,
    const bf16_t* scales, const int32_t* tokens, float* out, int count,
    int hidden, int begin, int rows, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(count) * hidden), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int row = tokens[i / hidden] - begin;
            out[i] = row >= 0 && row < rows
                ? int8_embedding_value(table[int64_t(row) * hidden + i % hidden], scales[row])
                : 0.0f;
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
            // log1p(exp(dt)), not log(1 + exp(dt)): for negative dt the
            // 1.0f + addition discards exp(dt)'s low bits before the log,
            // which is ~2.5e-4 relative error against torch's softplus.
            // Measured on the K2 gate, same formula; free to avoid.
            const float sp = dt > 20.0f ? dt : sycl::log1p(sycl::exp(dt));
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

// Same as launch_rmsnorm_gate_silu with the gate through a SIGMOID:
// Qwen4-Exp's DeltaNet output norm is RMSNormGated(activation=
// output_gate_type) and its config says "sigmoid" -- norm(x)*w*sigmoid(z),
// where Qwen3-Next/3.5 use silu(z).
sycl::event launch_rmsnorm_gate_sigmoid(sycl::queue& q, float* x, const float* z,
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
                    p[i] = p[i] * scale * bf16_to_f32(w[i]) * (1.0f / (1.0f + sycl::exp(-zv)));
                }
            });
    });
}

// launch_gate_sigmoid_mul_bf16_out with the gate read from the q
// projection's interleaved output qg ([token][head][q | gate]) -- no split.
sycl::event launch_gate_sigmoid_mul_bf16_out_qg(sycl::queue& q, const float* x, const float* qg,
                                                sycl_bf16* out, int tokens, int heads, int dim,
                                                const std::vector<sycl::event>& deps) {
    const size_t n = size_t(tokens) * heads * dim;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const size_t th = i / size_t(dim), d = i % size_t(dim);   // th = token*heads + head
            const float g = qg[th * 2 * size_t(dim) + size_t(dim) + d];
            const float s = 1.0f / (1.0f + sycl::exp(-g));
            out[i] = sycl_bf16(x[i] * s);
        });
    });
}

// launch_rmsnorm_gate_silu followed by launch_f32_to_bf16, in one pass, for
// the DeltaNet output projection's input: same values, x left untouched.
sycl::event launch_rmsnorm_gate_silu_bf16_out(sycl::queue& q, const float* x, const float* z,
                                              const bf16_t* w, sycl_bf16* out, int n_heads,
                                              int dim, float eps,
                                              const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(n_heads) * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const float* p = x + int64_t(it.get_group(0)) * dim;
                const float* g = z + int64_t(it.get_group(0)) * dim;
                sycl_bf16* o = out + int64_t(it.get_group(0)) * dim;
                float ss = 0.0f;
                for (int i = lane; i < dim; i += SG_SIZE) ss = sycl::fma(p[i], p[i], ss);
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                const float scale = sycl::rsqrt(ss / float(dim) + eps);
                for (int i = lane; i < dim; i += SG_SIZE) {
                    const float zv = g[i];
                    o[i] = sycl_bf16(p[i] * scale * bf16_to_f32(w[i])
                                     * (zv / (1.0f + sycl::exp(-zv))));
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
                // No q/k norm weight (K2-Horizon: query_key_norm=false) means
                // no q/k norm at all -- RoPE only.  Reading w[] through the
                // null pointer faulted on the device and hung the engine.
                if (w) {
                    float ss = 0.0f;
                    for (int i = lane; i < dim; i += SG_SIZE) ss = sycl::fma(p[i], p[i], ss);
                    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                    const float scale = sycl::rsqrt(ss / float(dim) + eps);
                    for (int i = lane; i < dim; i += SG_SIZE)
                        p[i] *= scale * (1.0f + bf16_to_f32(w[i]));
                }
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

// x *= k, in place.  gemma-4 needs it twice: the embedding is scaled by
// sqrt(hidden) on lookup (Gemma4TextScaledWordEmbedding), and each decoder
// layer multiplies the residual stream by its own layer_scalar after the
// residual add.  Both are silent when omitted -- the model stays fluent and
// drifts.
sycl::event launch_scale(sycl::queue& q, float* x, float k, int n,
                         const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)),
                       [=](sycl::id<1> id) { x[id[0]] *= k; });
    });
}

// logits = tanh(logits / cap) * cap.  gemma-4's final_logit_softcapping is
// 30.0.  Monotonic, so greedy decoding picks the same token -- but it is
// not a no-op for sampling, and it is exactly the kind of omission that
// leaves output plausible.
sycl::event launch_logit_softcap(sycl::queue& q, float* logits, float cap,
                                 int n, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const float inv = 1.0f / cap;
        h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            logits[id[0]] = sycl::tanh(logits[id[0]] * inv) * cap;
        });
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
                int sel_e = 0; float sel_v = -std::numeric_limits<float>::infinity();
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
                    if (top_k <= SG_SIZE) { if (lane == s) { sel_e = bi; sel_v = bv; } }
                    else if (lane == 0) { out_expert[s] = bi; out_weight[s] = bv; }
                }

                // Lane s holds selection s.  This repeats the serial softmax
                // below, same operations in the same order, on broadcast
                // registers (bit-identical) instead of 24 dependent global
                // round trips through out_weight: MEASURED 2026-10-01 the
                // decode top-k was 10.2 us per layer, 40 layers a token.
                if (top_k <= SG_SIZE) {
                    float m = sycl::select_from_group(sg, sel_v, 0);
                    for (int s = 1; s < top_k; ++s) m = sycl::fmax(m, sycl::select_from_group(sg, sel_v, s));
                    float sum = 0.0f, my_w = 0.0f;
                    for (int s = 0; s < top_k; ++s) {
                        const float e = sycl::exp(sycl::select_from_group(sg, sel_v, s) - m);
                        sum += e;
                        if (lane == s) my_w = e;
                    }
                    if (normalize && sum > 0.0f) my_w /= sum;
                    if (lane < top_k) { out_expert[lane] = sel_e; out_weight[lane] = my_w; }
                } else if (lane == 0) {
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

// Top-k over 256 router logits in ONE ESIMD thread, all in registers: per
// round a vector max, the lowest index holding it (the serial kernel's
// tie-break), then that lane is retired.  The softmax runs over the picks in
// pick order with the same operations as the sub-group kernel, so the
// weights are bit-identical for the same logits.  MEASURED 2026-10-01
// (tools/decode_probe.cpp): 3.2 us device; the sub-group kernel was 8.2-8.8 us
// per layer in-model.  GRIMOIRE_TOPK_ESIMD=0 = the sub-group kernel.
template <int E>
sycl::event router_topk_esimd(sycl::queue& q, const float* logits, int top_k,
                              int32_t* oe, float* ow, bool normalize,
                              const std::vector<sycl::event>& deps) {
    namespace es = sycl::ext::intel::esimd;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
            es::simd<float, E> v = es::block_load<float, E>(logits);
            es::simd<int, E> idx(0, 1);
            es::simd_mask<E> live = 1;
            es::simd<float, 16> sv = 0.0f;
            es::simd<int, 16> se = 0;
            const float NEG = -std::numeric_limits<float>::infinity();
            for (int s = 0; s < top_k; ++s) {
                es::simd<float, E> vv = es::merge(v, es::simd<float, E>(NEG), live);
                const float m = es::hmax<float>(vv);
                es::simd<int, E> cand = es::merge(idx, es::simd<int, E>(0x7fffffff), live && (vv == m));
                const int i = es::hmin<int>(cand);
                sv.select<1, 1>(s) = m;
                se.select<1, 1>(s) = i;
                live.template select<1, 1>(i) = 0;
            }
            float mx = sv[0];
            for (int s = 1; s < top_k; ++s) { const float c = sv.select<1, 1>(s)[0]; mx = c > mx ? c : mx; }
            float sum = 0.0f;
            es::simd<float, 16> ev = 0.0f;
            for (int s = 0; s < top_k; ++s) {
                const float e = sycl::exp(float(sv.select<1, 1>(s)[0]) - mx);
                ev.select<1, 1>(s) = e;
                sum += e;
            }
            if (normalize && sum > 0.0f) ev = ev / sum;
            for (int s = 0; s < top_k; ++s) {
                oe[s] = se.select<1, 1>(s)[0];
                ow[s] = ev.select<1, 1>(s)[0];
            }
        });
    });
}
// router_topk_esimd for `tokens` rows in one launch, one thread per token:
// the same operations per token (bit-identical).  MEASURED 2026-10-04: the
// sub-group batched kernel cost ~21 us per layer at every batch size --
// 0.84 ms of an Ornith batched decode step.
template <int E>
sycl::event router_topk_esimd_rows(sycl::queue& q, const float* logits, int tokens, int top_k,
                                   int32_t* out_e, float* out_w, bool normalize,
                                   const std::vector<sycl::event>& deps) {
    namespace es = sycl::ext::intel::esimd;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(tokens), 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int t = int(it.get_group(0));
            int32_t* oe = out_e + int64_t(t) * top_k;
            float* ow = out_w + int64_t(t) * top_k;
            es::simd<float, E> v = es::block_load<float, E>(logits + int64_t(t) * E);
            es::simd<int, E> idx(0, 1);
            es::simd_mask<E> live = 1;
            es::simd<float, 16> sv = 0.0f;
            es::simd<int, 16> se = 0;
            const float NEG = -std::numeric_limits<float>::infinity();
            for (int s = 0; s < top_k; ++s) {
                es::simd<float, E> vv = es::merge(v, es::simd<float, E>(NEG), live);
                const float m = es::hmax<float>(vv);
                es::simd<int, E> cand = es::merge(idx, es::simd<int, E>(0x7fffffff), live && (vv == m));
                const int i = es::hmin<int>(cand);
                sv.select<1, 1>(s) = m;
                se.select<1, 1>(s) = i;
                live.template select<1, 1>(i) = 0;
            }
            float mx = sv[0];
            for (int s = 1; s < top_k; ++s) { const float c = sv.select<1, 1>(s)[0]; mx = c > mx ? c : mx; }
            float sum = 0.0f;
            es::simd<float, 16> ev = 0.0f;
            for (int s = 0; s < top_k; ++s) {
                const float e = sycl::exp(float(sv.select<1, 1>(s)[0]) - mx);
                ev.select<1, 1>(s) = e;
                sum += e;
            }
            if (normalize && sum > 0.0f) ev = ev / sum;
            for (int s = 0; s < top_k; ++s) {
                oe[s] = se.select<1, 1>(s)[0];
                ow[s] = ev.select<1, 1>(s)[0];
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
    static const bool topk_es = [] { const char* e = std::getenv("GRIMOIRE_TOPK_ESIMD");
        return !(e && *e == '0'); }();
    if (topk_es && n_experts == 256 && top_k >= 1 && top_k <= 16)
        return router_topk_esimd<256>(q, logits, top_k, out_expert, out_weight, normalize, deps);
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
                int sel_e = 0; float sel_v = -std::numeric_limits<float>::infinity();
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
                    if (top_k <= SG_SIZE) { if (lane == s) { sel_e = bi; sel_v = bv; } }
                    else if (lane == 0) { out_expert[s] = bi; out_weight[s] = bv; }
                }

                // softmax over the selected weights; top_k is tiny, so one
                // lane is the cheapest correct thing here.
                // Lane s holds selection s.  This repeats the serial softmax
                // below, same operations in the same order, on broadcast
                // registers (bit-identical) instead of 24 dependent global
                // round trips through out_weight: MEASURED 2026-10-01 the
                // decode top-k was 10.2 us per layer, 40 layers a token.
                if (top_k <= SG_SIZE) {
                    float m = sycl::select_from_group(sg, sel_v, 0);
                    for (int s = 1; s < top_k; ++s) m = sycl::fmax(m, sycl::select_from_group(sg, sel_v, s));
                    float sum = 0.0f, my_w = 0.0f;
                    for (int s = 0; s < top_k; ++s) {
                        const float e = sycl::exp(sycl::select_from_group(sg, sel_v, s) - m);
                        sum += e;
                        if (lane == s) my_w = e;
                    }
                    if (normalize && sum > 0.0f) my_w /= sum;
                    if (lane < top_k) { out_expert[lane] = sel_e; out_weight[lane] = my_w; }
                } else if (lane == 0) {
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
    static const bool topk_es = [] { const char* e = std::getenv("GRIMOIRE_TOPK_ESIMD");
        return !(e && *e == '0'); }();
    if (topk_es && n_experts == 256 && top_k >= 1 && top_k <= 16)
        return router_topk_esimd_rows<256>(q, logits, tokens, top_k, out_expert, out_weight,
                                           normalize, deps);
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
                int sel_e = 0; float sel_v = -std::numeric_limits<float>::infinity();
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
                    if (top_k <= SG_SIZE) { if (lane == s) { sel_e = bi; sel_v = bv; } }
                    else if (lane == 0) { oe[s] = bi; ow[s] = bv; }
                }
                if (top_k <= SG_SIZE) {
                    float my_w = sel_v;
                    if (normalize) {
                        float m = sycl::select_from_group(sg, sel_v, 0);
                        for (int s = 1; s < top_k; ++s) m = sycl::fmax(m, sycl::select_from_group(sg, sel_v, s));
                        float sum = 0.0f;
                        for (int s = 0; s < top_k; ++s) {
                            const float e = sycl::exp(sycl::select_from_group(sg, sel_v, s) - m);
                            sum += e;
                            if (lane == s) my_w = e;
                        }
                        if (sum > 0.0f) my_w /= sum;
                    }
                    if (lane < top_k) { oe[lane] = sel_e; ow[lane] = my_w; }
                } else if (lane == 0 && normalize) {
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
                int sel_e = 0; float sel_v = -std::numeric_limits<float>::infinity();
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
                    if (top_k <= SG_SIZE) { if (lane == s) { sel_e = bi; sel_v = bv; } }
                    else if (lane == 0) { oe[s] = bi; ow[s] = bv; }
                }
                if (top_k <= SG_SIZE) {
                    float my_w = sel_v;
                    if (normalize) {
                        float m = sycl::select_from_group(sg, sel_v, 0);
                        for (int s = 1; s < top_k; ++s) m = sycl::fmax(m, sycl::select_from_group(sg, sel_v, s));
                        float sum = 0.0f;
                        for (int s = 0; s < top_k; ++s) {
                            const float e = sycl::exp(sycl::select_from_group(sg, sel_v, s) - m);
                            sum += e;
                            if (lane == s) my_w = e;
                        }
                        if (sum > 0.0f) my_w /= sum;
                    }
                    if (lane < top_k) { oe[lane] = sel_e; ow[lane] = my_w; }
                } else if (lane == 0 && normalize) {
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

// launch_gate_sigmoid_mul followed by launch_f32_to_bf16, in one pass, for a
// result that only feeds a GEMM: out = bf16(x * sigmoid(g)), the same values.
sycl::event launch_gate_sigmoid_mul_bf16_out(sycl::queue& q, const float* x, const float* g,
                                             sycl_bf16* out, size_t n,
                                             const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const float s = 1.0f / (1.0f + sycl::exp(-g[i]));
            out[i] = sycl_bf16(x[i] * s);
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

// Top-16 per row, ordered by value descending then index ascending -- the
// first 16 elements of that strict total order, so the result is unique and
// any correct method returns exactly what the loop form below returns.
//
// The loop form gives a row to ONE sub-group that makes 16 passes over the
// whole vocabulary, re-checking the picks so far on every element: MEASURED
// ~65 ms per 248K-entry row, 974 ms of a ~1 s DFlash2 draft (15 rows, one
// launch each).  Here a row gets a 512-lane work-group: one strided pass in
// which each lane keeps its own sorted top-16 in registers (most elements are
// rejected by a single compare), then a per-sub-group merge and a final
// merge of the 32 sub-group lists through SLM.  Both merges take the next
// pick as "the best element strictly after the previous pick", which needs
// no list of what was already chosen.  All rows in one launch; `stride` lets
// padded logits rows (draft_logits_stride) go in that same launch.
namespace {
inline bool topk_better(float av, int32_t ai, float bv, int32_t bi) {
    return av > bv || (av == bv && ai < bi);
}
}
sycl::event launch_topk16_rows_strided(
    sycl::queue& q, const float* logits, int rows, int vocab, int64_t stride,
    int32_t* out_ids, float* out_values,
    const std::vector<sycl::event>& deps) {
    constexpr int K = 16, SGS = 32, WG = SGS * SG_SIZE;
    constexpr float NINF = -std::numeric_limits<float>::infinity();
    constexpr float PINF =  std::numeric_limits<float>::infinity();
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        sycl::local_accessor<float, 1>   s_val(size_t(SGS) * K, h);
        sycl::local_accessor<int32_t, 1> s_idx(size_t(SGS) * K, h);
        h.parallel_for(sycl::nd_range<1>(size_t(rows) * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg  = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int sgi  = int(sg.get_group_id()[0]);
                const int lid  = int(it.get_local_id(0));
                const int row  = int(it.get_group(0));
                const float* lr = logits + int64_t(row) * stride;

                // this lane's sorted top-16 (best first)
                float   val[K];
                int32_t idx[K];
                #pragma unroll
                for (int j = 0; j < K; ++j) { val[j] = NINF; idx[j] = INT_MAX; }
                for (int v = lid; v < vocab; v += WG) {
                    float cv = lr[v];
                    int32_t ci = v;
                    if (!topk_better(cv, ci, val[K - 1], idx[K - 1])) continue;
                    #pragma unroll
                    for (int j = 0; j < K; ++j)            // insert by carrying
                        if (topk_better(cv, ci, val[j], idx[j])) {
                            const float tv = val[j]; const int32_t ti = idx[j];
                            val[j] = cv; idx[j] = ci; cv = tv; ci = ti;
                        }
                }

                // merge the 16 lanes' lists -> this sub-group's top-16
                float tv = PINF; int32_t ti = -1;           // "before everything"
                float mv = NINF; int32_t mi = INT_MAX;       // lane `pick` keeps pick `pick`
                for (int pick = 0; pick < K; ++pick) {
                    float bv = NINF; int32_t bi = INT_MAX;
                    #pragma unroll
                    for (int j = 0; j < K; ++j)
                        if (topk_better(tv, ti, val[j], idx[j]) &&
                            topk_better(val[j], idx[j], bv, bi)) { bv = val[j]; bi = idx[j]; }
                    const float   win = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
                    const int32_t wid = sycl::reduce_over_group(
                        sg, bv == win ? bi : INT_MAX, sycl::minimum<int32_t>());
                    if (lane == pick) { mv = win; mi = wid; }
                    tv = win; ti = wid;
                }
                s_val[sgi * K + lane] = mv;
                s_idx[sgi * K + lane] = mi;
                sycl::group_barrier(it.get_group());

                // sub-group 0 merges the 32 lists (512 candidates, 32 per lane)
                if (sgi == 0) {
                    float   cv[SGS];
                    int32_t cidx[SGS];
                    #pragma unroll
                    for (int t = 0; t < SGS; ++t) {
                        cv[t]   = s_val[t * K + lane];
                        cidx[t] = s_idx[t * K + lane];
                    }
                    float fv = PINF; int32_t fi = -1;
                    for (int pick = 0; pick < K; ++pick) {
                        float bv = NINF; int32_t bi = INT_MAX;
                        #pragma unroll
                        for (int t = 0; t < SGS; ++t)
                            if (topk_better(fv, fi, cv[t], cidx[t]) &&
                                topk_better(cv[t], cidx[t], bv, bi)) { bv = cv[t]; bi = cidx[t]; }
                        const float   win = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
                        const int32_t wid = sycl::reduce_over_group(
                            sg, bv == win ? bi : INT_MAX, sycl::minimum<int32_t>());
                        if (lane == 0) {
                            out_ids[int64_t(row) * K + pick]    = wid;
                            out_values[int64_t(row) * K + pick] = win;
                        }
                        fv = win; fi = wid;
                    }
                }
            });
    });
}

sycl::event launch_topk16_rows(
    sycl::queue& q, const float* logits, int rows, int vocab,
    int32_t* out_ids, float* out_values,
    const std::vector<sycl::event>& deps) {
    // GRIMOIRE_TOPK16_LOOP=1 keeps the original one-sub-group-per-row form
    // below as the reference for an A/B.
    static const bool loop = std::getenv("GRIMOIRE_TOPK16_LOOP") != nullptr;
    if (!loop)
        return launch_topk16_rows_strided(q, logits, rows, vocab, vocab,
                                          out_ids, out_values, deps);
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


// =====================================================================
//  K2-Horizon operators.  The host reference and the properties these
//  must satisfy live in include/b70/k2_horizon.hpp and
//  tests/test_k2_horizon.cpp; keep the two in step.
// =====================================================================

// GROUPED RMSNorm (layernorm_num_groups).  The variance is taken per
// contiguous group, so hidden 2560 with 2 groups is two independent
// 1280-wide normalizations, and the weight still spans the full row.
//
// zero_centered is FALSE for K2: K2HorizonRMSNorm initialises its weight
// to ONES and multiplies directly.  Qwen3.5 stores the weight centred on
// zero and applies (1 + w); using that convention here would scale every
// norm output by roughly 10x, forty-eight times over.
//
// One work-group per row, groups walked in sequence -- n_groups is 2, so
// there is nothing to gain from splitting them across work-groups and the
// sequential form lets the partial-sum scratch be reused.
sycl::event launch_rmsnorm_grouped(sycl::queue& q, float* h,
                                   const float* r0, const float* r1,
                                   const bf16_t* weight, float* out,
                                   sycl_bf16* out_bf,
                                   int tokens, int hidden, int n_groups,
                                   float eps, float weight_offset,
                                   const std::vector<sycl::event>& deps) {
    const int WG = norm_wg();
    const int gn = hidden / n_groups;   // caller guarantees hidden % n_groups == 0
    return q.submit([&](sycl::handler& hd) {
        hd.depends_on(deps);
        sycl::local_accessor<float, 1> partial(WG / SG_SIZE, hd);
        hd.parallel_for(
            sycl::nd_range<1>(size_t(tokens) * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg  = it.get_sub_group();
                const int  lid = int(it.get_local_id(0));
                const int  sgid= int(sg.get_group_id()[0]);
                const int  lane= int(sg.get_local_id()[0]);
                const int  wgz = int(it.get_local_range(0));
                const int  nsg = wgz / SG_SIZE;
                const int64_t row = int64_t(it.get_group(0)) * hidden;

                for (int g = 0; g < n_groups; ++g) {
                    const int64_t base = row + int64_t(g) * gn;
                    float ss = 0.0f;
                    for (int i = lid; i < gn; i += wgz) {
                        float v = h[base + i];
                        if (r0) v += r0[base + i];
                        if (r1) v += r1[base + i];
                        if (r0 || r1) h[base + i] = v;   // keep for the next residual
                        ss = sycl::fma(v, v, ss);
                    }
                    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                    if (lane == 0) partial[sgid] = ss;
                    sycl::group_barrier(it.get_group());

                    float total = 0.0f;
                    for (int i = 0; i < nsg; ++i) total += partial[i];
                    const float scale = sycl::rsqrt(total / float(gn) + eps);

                    for (int i = lid; i < gn; i += wgz) {
                        const int64_t o = base + i;
                        const float wv = bf16_to_f32(weight[int64_t(g) * gn + i]);
                        const float y = h[o] * scale * (weight_offset + wv);
                        if (out)    out[o] = y;
                        if (out_bf) out_bf[o] = sycl_bf16(y);
                    }
                    // partial[] is reused by the next group: no thread may
                    // race ahead and overwrite it while others still read.
                    sycl::group_barrier(it.get_group());
                }
            });
    });
}

// SOFTPLUS attention gate.  out = attn * softplus(gate, beta), applied to
// the attention output BEFORE o_proj.  torch falls back to the identity
// once beta*x passes threshold (20) so the exp cannot overflow; match that
// or a large gate produces inf instead of a linear response.
sycl::event launch_softplus_gate(sycl::queue& q, const float* attn,
                                 const float* gate, float* out,
                                 int64_t n, float beta,
                                 const std::vector<sycl::event>& deps) {
    constexpr int WG = 256;
    const int64_t groups = (n + WG - 1) / WG;
    return q.submit([&](sycl::handler& hd) {
        hd.depends_on(deps);
        hd.parallel_for(
            sycl::nd_range<1>(size_t(groups) * WG, WG),
            [=](sycl::nd_item<1> it) {
                const int64_t i = int64_t(it.get_global_id(0));
                if (i >= n) return;
                const float gv = gate[i];
                const float bx = beta * gv;
                // log1p(exp(bx)), NOT log(1 + exp(bx)).  For a negative
                // gate exp(bx) is small and the 1.0f + addition throws
                // away its low bits before the log ever sees them:
                // measured against torch's own softplus that is ~2.5e-4
                // relative error around bx = -10, and it is zero with
                // log1p.  Free to fix, so there is no reason to carry it.
                const float sp = bx > 20.0f ? gv
                               : sycl::log1p(sycl::exp(bx)) / beta;
                out[i] = attn[i] * sp;
            });
    });
}

// SIGMOID router with a SELECTION-ONLY bias.
//
// The bias is added to the value top-k ranks on; the weight returned for
// an expert is the UNBIASED sigmoid.  Folding the bias into the weight is
// silent -- the routes stay plausible and only the mixture is wrong -- so
// the winner's score is recomputed from its own logit rather than carried
// through the reduction.
//
// Normalization is a plain sum, NOT the softmax launch_router_topk_batched
// applies: K2 divides the gathered sigmoids by their sum and then
// multiplies by router_scaling_factor.
sycl::event launch_router_topk_k2(
    sycl::queue& q, const float* logits, const bf16_t* bias,
    int tokens, int n_experts, int top_k,
    int32_t* out_expert, float* out_weight,
    bool normalize, float scaling,
    const std::vector<sycl::event>& deps) {
    // Register-resident form: every expert's sigmoid is computed ONCE and
    // kept in its lane, so a selection pass is PL compares and two
    // sub-group reductions.  The loop below reloads the logits and redoes
    // exp() for every expert on every one of the top_k passes -- measured
    // 16.9 us for 8 of K2's 100 experts, per MoE layer and again for MoVA.
    // Same ranking (lowest index wins a tie), same winner weight (the
    // unbiased sigmoid of its own logit), same normalise-then-scale order,
    // so the two produce identical routes and weights.
    constexpr int PL = 8;
    if (n_experts <= SG_SIZE * PL) {
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
                    constexpr float ninf = -std::numeric_limits<float>::infinity();
                    float sel[PL], sc[PL];
                    #pragma unroll
                    for (int i = 0; i < PL; ++i) {
                        const int e = lane + i * SG_SIZE;
                        sc[i] = 0.0f; sel[i] = ninf;
                        if (e < n_experts) {
                            sc[i]  = 1.0f / (1.0f + sycl::exp(-row[e]));
                            sel[i] = sc[i] + (bias ? bf16_to_f32(bias[e]) : 0.0f);
                        }
                    }
                    for (int s = 0; s < top_k; ++s) {
                        // e rises with i inside a lane: strict > keeps the
                        // lowest index on a tie, as the loop form does
                        float cv = ninf; int ci = INT_MAX;
                        #pragma unroll
                        for (int i = 0; i < PL; ++i)
                            if (sel[i] > cv) { cv = sel[i]; ci = lane + i * SG_SIZE; }
                        const float bv = sycl::reduce_over_group(
                            sg, cv, sycl::maximum<float>());
                        const int bi = sycl::reduce_over_group(
                            sg, (cv == bv && ci != INT_MAX) ? ci : INT_MAX,
                            sycl::minimum<int>());
                        float w = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < PL; ++i)
                            if (lane + i * SG_SIZE == bi) { w = sc[i]; sel[i] = ninf; }
                        w = sycl::reduce_over_group(sg, w, sycl::plus<float>());
                        if (lane == 0) { oe[s] = bi; ow[s] = w; }   // UNBIASED
                    }
                    if (lane == 0) {
                        if (normalize) {
                            float sum = 0.0f;
                            for (int s = 0; s < top_k; ++s) sum += ow[s];
                            if (sum > 0.0f)
                                for (int s = 0; s < top_k; ++s) ow[s] /= sum;
                        }
                        if (scaling != 1.0f)
                            for (int s = 0; s < top_k; ++s) ow[s] *= scaling;
                    }
                });
        });
    }
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
                        // rank on sigmoid(logit) + bias, never on the logit:
                        // the bias is defined against the post-sigmoid score.
                        const float sc = 1.0f / (1.0f + sycl::exp(-row[e]));
                        const float v  = sc + (bias ? bf16_to_f32(bias[e]) : 0.0f);
                        if (v > cv || (v == cv && e < ci)) { cv = v; ci = e; cs = slot; }
                    }
                    const float bv = sycl::reduce_over_group(
                        sg, cv, sycl::maximum<float>());
                    const int bi = sycl::reduce_over_group(
                        sg, (cv == bv && ci != INT_MAX) ? ci : INT_MAX,
                        sycl::minimum<int>());
                    if (ci == bi && cs >= 0 && cs < 64) taken |= 1ull << cs;
                    if (lane == 0) {
                        oe[s] = bi;
                        ow[s] = 1.0f / (1.0f + sycl::exp(-row[bi]));  // UNBIASED
                    }
                }
                if (lane == 0) {
                    if (normalize) {
                        float sum = 0.0f;
                        for (int s = 0; s < top_k; ++s) sum += ow[s];
                        if (sum > 0.0f)
                            for (int s = 0; s < top_k; ++s) ow[s] /= sum;
                    }
                    if (scaling != 1.0f)
                        for (int s = 0; s < top_k; ++s) ow[s] *= scaling;
                }
            });
    });
}


// MoVA accumulate: out += w * silu(in).  The SiLU is on the EXPERT
// OUTPUT and applies before the router weight scales it, per
// combine_routed_experts(activation=F.silu).
// ---------------------------------------------------------------------
// MoVA value projection over experts packed EXPERT-MAJOR.
//
// The bring-up path issued one GEMV per (token, route) and brought the
// routing table back to the HOST to decide which expert each one needed:
// two synchronisations per layer at M=1, and one per layer plus a stall
// at prefill.  At 4096 tokens over 45 sparse layers the readback is what
// makes K2 prefill unusable -- not the arithmetic, which is unchanged.
//
// Packing the E experts into a single [E*N][K] weight removes it.  Expert
// e's output row n is global row e*N + n, so QuantWeight's own indexing
// resolves payload and scales with NO new decode logic -- the packed
// weight is simply a taller matrix of the same format, and `at()` on it
// is the same reference these kernels have always had to match.  The
// routing table can then stay on the device: this kernel reads rex/rwt
// directly and never tells the host what it found.
//
// One work-item owns one (token, output row) pair and walks its own top-k
// routes, so the accumulation is private -- no atomics, and the summation
// order is fixed by j, which keeps the result reproducible run to run.
//
// NOT used under tensor parallel: TP shards each expert's rows across
// ranks, and row-sharding a matrix that has been concatenated along N
// would cut across expert boundaries.  The per-expert path stays for
// that case.
// ---------------------------------------------------------------------
template <Fmt F>
static sycl::event mova_value_packed_impl(
        sycl::queue& q, const QuantWeight& w, const float* x,
        const int32_t* rex, const float* rwt, float* y,
        int M, int N, int E, int top_k,
        const std::vector<sycl::event>& deps) {
    const int K = w.K;
    constexpr int WG = 128;
    const size_t total  = size_t(M) * size_t(N);
    const size_t groups = (total + WG - 1) / WG;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;          // POD: pointers and strides only
        h.parallel_for(
            sycl::nd_range<1>(groups * WG, WG),
            [=](sycl::nd_item<1> it) {
                const size_t g = it.get_global_id(0);
                if (g >= total) return;
                const int m = int(g / size_t(N));
                const int n = int(g % size_t(N));
                const float* xr = x + size_t(m) * K;
                float sum = 0.0f;
                for (int j = 0; j < top_k; ++j) {
                    const int e = rex[size_t(m) * top_k + j];
                    if (e < 0 || e >= E) continue;   // router produced no route
                    const int64_t row = int64_t(e) * N + n;
                    const uint8_t* rp = wc.payload + row * wc.row_bytes;
                    float a = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        float sc;
                        if constexpr (F == Fmt::BF16) {
                            sc = 1.0f;
                        } else if constexpr (F == Fmt::FP8_E4M3 ||
                                             F == Fmt::FP8_E5M2 ||
                                             F == Fmt::INT8) {
                            sc = static_cast<const float*>(wc.scales)[row];
                        } else if constexpr (F == Fmt::INT4) {
                            sc = bf16_to_f32(static_cast<const bf16_t*>(wc.scales)
                                     [row * wc.row_scales + (k >> wc.int4_gshift())]);
                        } else {
                            sc = e8m0_to_f32(static_cast<const uint8_t*>(wc.scales)
                                     [row * wc.row_scales + k / kMXBlock]);
                        }
                        float wv;
                        if constexpr (F == Fmt::INT4) {
                            const uint8_t z = wc.zeros
                                ? wc.zeros[row * wc.row_scales + (k >> wc.int4_gshift())]
                                : uint8_t(0);
                            wv = decode_int4(rp, k, sc, z);
                        } else {
                            wv = decode_elem<F>(rp, k, sc);
                        }
                        a = sycl::fma(wv, xr[k], a);
                    }
                    // silu THEN weight, matching launch_silu_scale_accum:
                    // the router weight scales the activated expert output,
                    // it does not enter the activation.
                    const float sl = a / (1.0f + sycl::exp(-a));
                    sum = sycl::fma(rwt[size_t(m) * top_k + j], sl, sum);
                }
                y[size_t(m) * N + n] = sum;
            });
    });
}

sycl::event launch_mova_value_packed(
        sycl::queue& q, const QuantWeight& w, const float* x,
        const int32_t* rex, const float* rwt, float* y,
        int M, int N, int E, int top_k,
        const std::vector<sycl::event>& deps) {
    switch (w.fmt) {
        case Fmt::BF16:     return mova_value_packed_impl<Fmt::BF16>(q,w,x,rex,rwt,y,M,N,E,top_k,deps);
        case Fmt::FP8_E4M3: return mova_value_packed_impl<Fmt::FP8_E4M3>(q,w,x,rex,rwt,y,M,N,E,top_k,deps);
        case Fmt::FP8_E5M2: return mova_value_packed_impl<Fmt::FP8_E5M2>(q,w,x,rex,rwt,y,M,N,E,top_k,deps);
        case Fmt::INT8:     return mova_value_packed_impl<Fmt::INT8>(q,w,x,rex,rwt,y,M,N,E,top_k,deps);
        case Fmt::INT4:     return mova_value_packed_impl<Fmt::INT4>(q,w,x,rex,rwt,y,M,N,E,top_k,deps);
        case Fmt::MXFP8:    return mova_value_packed_impl<Fmt::MXFP8>(q,w,x,rex,rwt,y,M,N,E,top_k,deps);
        case Fmt::MXFP4:    return mova_value_packed_impl<Fmt::MXFP4>(q,w,x,rex,rwt,y,M,N,E,top_k,deps);
    }
    return {};
}

sycl::event launch_silu_scale_accum(sycl::queue& q, const float* in, float* out,
                                    float w, int n,
                                    const std::vector<sycl::event>& deps) {
    constexpr int WG = 256;
    const int groups = (n + WG - 1) / WG;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(groups) * WG, WG),
            [=](sycl::nd_item<1> it) {
                const int i = int(it.get_global_id(0));
                if (i >= n) return;
                const float v = in[i];
                out[i] += w * (v / (1.0f + sycl::exp(-v)));
            });
    });
}

// ---------------------------------------------------------------------
//  Qwen4-Exp (Qwen3.8-Flash-Next) HyperConnections.
//
//  The residual stream is hc_count streams wide.  mix() collapses it to
//  one hidden-wide block input; combine() injects the block output back
//  into every stream.  Host reference and the full derivation:
//  include/b70/qwen4_exp.hpp -- these are its parallel form and
//  bin/test_k2_kernels diffs them against it.
//
//  The two projections (down/up in mix, inject in combine) are ordinary
//  matmuls and go through the engine's existing GEMV, so what is here is
//  only the part that has no existing kernel: the grouped norm, the
//  gated mean, and the injection.
// ---------------------------------------------------------------------

// GroupedGemmaRMSNorm: one variance PER hc stream, weight applied as
// (1 + w) across the whole hc*hidden row.  One work-group per (row,
// stream) so the reduction stays inside a group.
sycl::event launch_hc_norm(sycl::queue& q, const float* x, const bf16_t* w,
                           float* out, int rows, int hc_count, int hidden,
                           float eps, const std::vector<sycl::event>& deps) {
    constexpr int WG = 256;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> red(WG, h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(rows) * hc_count * WG, WG),
            [=](sycl::nd_item<1> it) {
                const int gid  = int(it.get_group(0));
                const int row  = gid / hc_count;
                const int c    = gid % hc_count;
                const int lane = int(it.get_local_id(0));
                const int64_t base = (int64_t(row) * hc_count + c) * hidden;
                float acc = 0.0f;
                for (int i = lane; i < hidden; i += WG) {
                    const float v = x[base + i];
                    acc += v * v;
                }
                red[lane] = acc;
                sycl::group_barrier(it.get_group());
                for (int s = WG / 2; s; s >>= 1) {
                    if (lane < s) red[lane] += red[lane + s];
                    sycl::group_barrier(it.get_group());
                }
                const float inv = sycl::rsqrt(red[0] / float(hidden) + eps);
                // The affine weight spans hc*hidden, so it is indexed by
                // the stream too -- not reused across streams.
                for (int i = lane; i < hidden; i += WG)
                    out[base + i] = x[base + i] * inv *
                                    (1.0f + bf16_to_f32(w[size_t(c) * hidden + i]));
            });
    });
}

// mix() tail: gate = sigmoid(up_out), then the gated MEAN over streams.
// `up_out` is the [rows][hc*hidden] result of the up projection, which
// the caller produces with the ordinary GEMV.
sycl::event launch_hc_gated_mean(sycl::queue& q, const float* up_out,
                                 const float* normed, float* out, int rows,
                                 int hc_count, int hidden,
                                 const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows) * hidden),
            [=](sycl::id<1> id) {
                const int row = int(id[0] / hidden);
                const int hh  = int(id[0] % hidden);
                const int64_t rb = int64_t(row) * hc_count * hidden;
                float acc = 0.0f;
                for (int c = 0; c < hc_count; ++c) {
                    const int64_t i = rb + int64_t(c) * hidden + hh;
                    acc += (1.0f / (1.0f + sycl::exp(-up_out[i]))) * normed[i];
                }
                out[int64_t(row) * hidden + hh] = acc / float(hc_count);
            });
    });
}

// combine(): out[c][h] = hyper[c][h] + block[h] * 2*sigmoid(inj_out[c]/hc).
// `inj_out` is the [rows][hc_count] result of the injection projection.
//
// NOTE the residual is the UNNORMALISED `hyper`, not `normed`.  Adding to
// the normalised copy loses the stream the layer is built on and is
// silent -- the shapes are identical either way.
// ---------------------------------------------------------------------
// Fused hyper-connection mix for ONE token (decode), BF16 weights.  The
// same arithmetic as launch_hc_combine + launch_hc_norm + GEMV(down) +
// launch_hc_silu + GEMV(inject) + GEMV(up) + launch_hc_gated_mean, in
// three launches instead of seven (96 mixes per Flash-Next token: the
// seven small launches cost ~9.5 ms of it).
// ---------------------------------------------------------------------
// A: [combine] + grouped (1 + w) RMSNorm.  One work-group per stream.
sycl::event launch_hc_combine_norm1(sycl::queue& q, float* hyper, const float* pend,
                                    const float* pinj, const bf16_t* w, float* normed,
                                    int hc_count, int hidden, float eps,
                                    const std::vector<sycl::event>& deps) {
    constexpr int WG = 256;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(hc_count) * WG, WG), [=](sycl::nd_item<1> it) {
            const int c = int(it.get_group(0)), lane = int(it.get_local_id(0));
            float* x = hyper + int64_t(c) * hidden;
            float z = 0.0f;
            if (pend) z = 2.0f / (1.0f + sycl::exp(-(pinj[c] / float(hc_count))));
            float acc = 0.0f;
            for (int i = lane; i < hidden; i += WG) {
                float v = x[i];
                if (pend) { v += pend[i] * z; x[i] = v; }
                acc += v * v;
            }
            acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
            const float inv = sycl::rsqrt(acc / float(hidden) + eps);
            const bf16_t* ws = w + int64_t(c) * hidden;
            float* o = normed + int64_t(c) * hidden;
            for (int i = lane; i < hidden; i += WG)
                o[i] = x[i] * inv * (1.0f + bf16_to_f32(ws[i]));
        });
    });
}

namespace {
// dot of a bf16 row with an fp32 vector, one sub-group, 8 elements per lane
// per step (n % 128 == 0).
inline float sg_dot_bf16(const sycl::sub_group& sg, const bf16_t* row, const float* x, int n) {
    const int lane = int(sg.get_local_id()[0]);
    float acc = 0.0f;
    for (int k = lane * 8; k < n; k += SG_SIZE * 8) {
        const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(row + k);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t u = v[j];
            acc = sycl::fma(sycl::bit_cast<float>(u << 16), x[k + 2 * j], acc);
            acc = sycl::fma(sycl::bit_cast<float>(u & 0xFFFF0000u), x[k + 2 * j + 1], acc);
        }
    }
    return sycl::reduce_over_group(sg, acc, sycl::plus<float>());
}
} // namespace

// B: lora[r] = silu(down[r] . normed / hc) for r < lowrank, and (when inj_w)
// inj[c] = inject[c] . normed.  One sub-group per output row.
sycl::event launch_hc_down_inject1(sycl::queue& q, const bf16_t* down, const bf16_t* inj_w,
                                   const float* normed, float* lora, float* inj,
                                   int lowrank, int hc_count, int wide,
                                   const std::vector<sycl::event>& deps) {
    const int rows = lowrank + (inj_w ? hc_count : 0);
    const int groups = (rows + WG_SUBGROUPS - 1) / WG_SUBGROUPS;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * WG_SUBGROUPS * SG_SIZE,
                                         size_t(WG_SUBGROUPS) * SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int r = int(it.get_group(0)) * WG_SUBGROUPS + int(sg.get_group_id()[0]);
                if (r >= rows) return;
                const bool is_inj = r >= lowrank;
                const bf16_t* row = is_inj ? inj_w + int64_t(r - lowrank) * wide
                                           : down + int64_t(r) * wide;
                const float d = sg_dot_bf16(sg, row, normed, wide);
                if (sg.get_local_id()[0] == 0) {
                    if (is_inj) inj[r - lowrank] = d;
                    else { const float v = d / float(hc_count); lora[r] = v / (1.0f + sycl::exp(-v)); }
                }
            });
    });
}

// C: block[h] = (1/hc) sum_c sigmoid(up[c*H+h] . lora) * normed[c*H+h].
// One sub-group per h; lora staged in SLM.
sycl::event launch_hc_up_mean1(sycl::queue& q, const bf16_t* up, const float* lora,
                               const float* normed, float* block, int hc_count,
                               int hidden, int lowrank,
                               const std::vector<sycl::event>& deps) {
    const int groups = (hidden + WG_SUBGROUPS - 1) / WG_SUBGROUPS;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> sl(size_t(lowrank), h);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * WG_SUBGROUPS * SG_SIZE,
                                         size_t(WG_SUBGROUPS) * SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int lid = int(it.get_local_id(0));
                for (int i = lid; i < lowrank; i += WG_SUBGROUPS * SG_SIZE) sl[i] = lora[i];
                sycl::group_barrier(it.get_group());
                const auto sg = it.get_sub_group();
                const int hh = int(it.get_group(0)) * WG_SUBGROUPS + int(sg.get_group_id()[0]);
                if (hh >= hidden) return;
                const float* ls = sl.template get_multi_ptr<sycl::access::decorated::no>().get();
                float acc = 0.0f;
                for (int c = 0; c < hc_count; ++c) {
                    const int64_t i = int64_t(c) * hidden + hh;
                    const float g = sg_dot_bf16(sg, up + i * lowrank, ls, lowrank);
                    acc += normed[i] / (1.0f + sycl::exp(-g));
                }
                if (sg.get_local_id()[0] == 0) block[hh] = acc / float(hc_count);
            });
    });
}

sycl::event launch_hc_combine(sycl::queue& q, const float* hyper,
                              const float* inj_out, const float* block,
                              float* out, int rows, int hc_count, int hidden,
                              const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows) * hc_count * hidden),
            [=](sycl::id<1> id) {
                const int64_t i = int64_t(id[0]);
                const int hh  = int(i % hidden);
                const int c   = int((i / hidden) % hc_count);
                const int row = int(i / (int64_t(hidden) * hc_count));
                const float z = inj_out[int64_t(row) * hc_count + c] / float(hc_count);
                const float wgt = 2.0f / (1.0f + sycl::exp(-z));
                out[i] = hyper[i] + block[int64_t(row) * hidden + hh] * wgt;
            });
    });
}

// ---------------------------------------------------------------------
//  QSA -- Qwen Sparse Attention (Qwen4-Exp).
//
//  Host reference and the full derivation: include/b70/qwen4_exp.hpp.
//  Three stages, and the attention that consumes them is in attention.cpp
//  because it is the existing flash math with the key loop driven by an
//  index array instead of a range.
// ---------------------------------------------------------------------

// Stage 1.  MQA indexer: relu per head, THEN summed across heads, THEN
// scaled by 1/sqrt(head_dim).  One sub-group per (row, block).
//
// The order is the whole algorithm.  Summing before the relu, or scaling
// before it, changes which blocks are selected -- and a wrong selection
// still produces fluent text, because attention over the wrong 2048
// tokens is still attention.
sycl::event launch_qsa_index_logits(sycl::queue& q, const float* qv,
                                    const float* keys, float* logits,
                                    int rows, int n_heads, int head_dim,
                                    int n_blocks, const int32_t* visible,
                                    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(rows) * n_blocks * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int gid  = int(it.get_group(0));
                const int row  = gid / n_blocks;
                const int n    = gid % n_blocks;
                const int vis  = visible ? visible[row] : n_blocks;
                if (n >= vis) {
                    if (lane == 0)
                        logits[int64_t(row) * n_blocks + n] =
                            -std::numeric_limits<float>::infinity();
                    return;
                }
                // ONE key cache, SHARED by every query row.  The blocks
                // are the sequence's, not the query's: striding this by
                // row reads past the blocks that were actually pooled and
                // into whatever the allocation happened to hold, which is
                // a wrong block ranking at M > 1 and perfectly fluent.
                // (Decode is M == 1, so it never saw it.)
                const float* kn = keys + int64_t(n) * head_dim;
                float acc = 0.0f;
                for (int hh = 0; hh < n_heads; ++hh) {
                    const float* qh = qv +
                        (int64_t(row) * n_heads + hh) * head_dim;
                    float dot = 0.0f;
                    for (int d = lane; d < head_dim; d += SG_SIZE)
                        dot = sycl::fma(qh[d], kn[d], dot);
                    dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
                    // relu per head, before the sum
                    if (lane == 0) acc += dot > 0.0f ? dot : 0.0f;
                }
                if (lane == 0)
                    logits[int64_t(row) * n_blocks + n] =
                        acc / sycl::sqrt(float(head_dim));
            });
    });
}

// RoPE over [rows][heads][dim] where row r sits at position first + r.
// The engine's other rope launchers take ONE position (decode) or fuse
// the norm (the batched qk path); the QSA indexer's query needs neither
// and must not borrow the fused one, whose norm convention differs.
sycl::event launch_rope_rows(sycl::queue& q, float* x, int rows, int heads,
                             int dim, int first, float theta,
                             float partial_factor,
                             const std::vector<sycl::event>& deps) {
    const int rot = int(dim * partial_factor) & ~1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows) * heads * (rot / 2)),
            [=](sycl::id<1> id) {
                const int per = heads * (rot / 2);
                const int r   = int(id[0] / per);
                const int rem = int(id[0] % per);
                const int hh  = rem / (rot / 2);
                const int i   = rem % (rot / 2);
                const float pos = float(first + r);
                // EXACTLY launch_rope_dev's expression, not an equivalent
                // one: the QSA indexer's query is roped here in the
                // batched path and there in decode, and a low-precision
                // powr would rank blocks slightly differently, which
                // selects different tokens and is fluent.
                const float inv = sycl::exp(-float(2 * i) / float(rot)
                                            * sycl::log(theta));
                const float ang = pos * inv;
                const float cs = sycl::cos(ang), sn = sycl::sin(ang);
                float* v = x + (int64_t(r) * heads + hh) * dim;
                const float a = v[i], b = v[i + rot / 2];
                v[i]           = a * cs - b * sn;
                v[i + rot / 2] = a * sn + b * cs;
            });
    });
}

// Copy `width` floats out of each row of a wider array.  The indexer's
// projection emits [q | k] per token, so the key rows are strided by the
// whole projection width and a flat memcpy would interleave them.
sycl::event launch_copy_rows_strided(sycl::queue& q, const float* src,
                                     float* dst, int rows, int src_stride,
                                     int dst_stride, int width,
                                     const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows) * width),
            [=](sycl::id<1> id) {
                const int r = int(id[0] / width);
                const int c = int(id[0] % width);
                dst[int64_t(r) * dst_stride + c] =
                    src[int64_t(r) * src_stride + c];
            });
    });
}

// Per-row QSA metadata: how many COMPLETE key blocks this row may see,
// the sequence length its expansion is clipped to, and its own position.
// Kept in device memory so no host round trip sits inside the layer loop.
sycl::event launch_qsa_row_meta(sycl::queue& q, int32_t* visible,
                                int32_t* seq_len, int32_t* query_pos,
                                int rows, int first, int total,
                                int compress_ratio,
                                const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows)), [=](sycl::id<1> id) {
            const int r = int(id[0]);
            const int p = first + r;
            const int a = (p + 1) / compress_ratio;
            const int b = total / compress_ratio;
            visible[r]   = a < b ? a : b;
            seq_len[r]   = total;
            query_pos[r] = p;
        });
    });
}

// Stage 1b.  Pool the raw index keys of one completed group.
//
// A block is COMPLETE when its last token arrives, and the reference
// pools the ratio raw keys as they were projected -- before any norm and
// before RoPE (ops/qsa.py, _compress_qsa_groups_kernel: it accumulates
// raw_keys and divides by COMPRESS_RATIO, and the caller norms and ropes
// the POOLED result afterwards).  Norming or roping first is a different
// key and a different block ranking.
sycl::event launch_qsa_pool_blocks(sycl::queue& q, const float* raw,
                                   float* pooled, int first_block,
                                   int n_blocks, int compress_ratio,
                                   int head_dim,
                                   const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_blocks) * head_dim),
            [=](sycl::id<1> id) {
                const int b = int(id[0] / head_dim);
                const int d = int(id[0] % head_dim);
                const int t0 = (first_block + b) * compress_ratio;
                float acc = 0.0f;
                for (int r = 0; r < compress_ratio; ++r)
                    acc += raw[int64_t(t0 + r) * head_dim + d];
                pooled[int64_t(b) * head_dim + d] = acc / float(compress_ratio);
            });
    });
}

// Stage 1c.  RoPE the pooled keys at the position of the FIRST token of
// each group (ops/qsa.py: first_position = end_position - ratio + 1), not
// the last and not the block index.  NeoX-style halves, matching
// launch_rope_dev.
sycl::event launch_qsa_rope_blocks(sycl::queue& q, float* keys,
                                   int first_block, int n_blocks,
                                   int head_dim, int compress_ratio,
                                   float theta, float partial_factor,
                                   const std::vector<sycl::event>& deps) {
    const int rot = int(head_dim * partial_factor) & ~1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_blocks) * (rot / 2)),
            [=](sycl::id<1> id) {
                const int b = int(id[0] / (rot / 2));
                const int i = int(id[0] % (rot / 2));
                const float pos = float((first_block + b) * compress_ratio);
                const float inv = sycl::exp(-float(2 * i) / float(rot)
                                            * sycl::log(theta));
                const float ang = pos * inv;
                const float cs = sycl::cos(ang), sn = sycl::sin(ang);
                float* k = keys + int64_t(b) * head_dim;
                const float a = k[i], bb = k[i + rot / 2];
                k[i]           = a * cs - bb * sn;
                k[i + rot / 2] = a * sn + bb * cs;
            });
    });
}

// Stage 2.  Top-k blocks per row, DESCENDING, holes left as -1.
//
// One work-group per row, one masked max-reduction per output slot.  That
// is O(topk * n_blocks / WG) and it is not the fastest selection there
// is; it is the one whose result is the reference's result, which is what
// the gate compares against.  A cheaper threshold scan can replace it
// once there is a number to beat (rule 8).
sycl::event launch_qsa_topk_blocks(sycl::queue& q, const float* logits,
                                   int32_t* out, int rows, int n_blocks,
                                   int topk, const int32_t* visible,
                                   const std::vector<sycl::event>& deps) {
    // One work-group per query row.  Two cases:
    //  * vis <= topk: every visible block is selected -- no ranking at all
    //    (every row of a prompt shorter than indexer_budget).
    //  * vis >  topk: find the topk-th largest score by a 4-pass radix
    //    select on order-preserving 32-bit keys (256-bin SLM histogram per
    //    pass), then one ordered scan keeps every key above the threshold
    //    and the lowest-index ties at it -- the same set, and the same tie
    //    rule, as the previous argmax-per-slot loop, which rescanned all
    //    blocks for each of the topk picks (O(topk^2 * n)) and cost ~500 ms
    //    per full-attention layer at 6K tokens.  The set is emitted in
    //    INDEX order; attention over it does not depend on the order.
    constexpr int WG = 256;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> hist(256, h);
        sycl::local_accessor<uint32_t, 1> sel(4, h);   // prefix, mask, remaining
        h.parallel_for(sycl::nd_range<1>(size_t(rows) * WG, WG),
            [=](sycl::nd_item<1> it) {
                const int row  = int(it.get_group(0));
                const int lane = int(it.get_local_id(0));
                const float* lg = logits + int64_t(row) * n_blocks;
                int32_t* o = out + int64_t(row) * topk;
                int vis = visible ? visible[row] : n_blocks;
                vis = vis < n_blocks ? vis : n_blocks;
                if (vis <= topk) {
                    for (int k = lane; k < topk; k += WG) o[k] = k < vis ? k : -1;
                    return;
                }
                auto key_of = [](float f) {
                    const uint32_t u = sycl::bit_cast<uint32_t>(f);
                    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                };
                if (lane == 0) { sel[0] = 0u; sel[1] = 0u; sel[2] = uint32_t(topk); }
                for (int pass = 0; pass < 4; ++pass) {
                    const int shift = 24 - 8 * pass;
                    hist[lane] = 0u;
                    sycl::group_barrier(it.get_group());
                    const uint32_t prefix = sel[0], mask = sel[1];
                    for (int n = lane; n < vis; n += WG) {
                        const uint32_t k = key_of(lg[n]);
                        if ((k & mask) == prefix) {
                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group,
                                             sycl::access::address_space::local_space>
                                a(hist[(k >> shift) & 0xFFu]);
                            a.fetch_add(1u);
                        }
                    }
                    sycl::group_barrier(it.get_group());
                    if (lane == 0) {
                        uint32_t need = sel[2], acc = 0u;
                        int b = 255;
                        for (; b > 0; --b) {
                            if (acc + hist[b] >= need) break;
                            acc += hist[b];
                        }
                        sel[0] = prefix | (uint32_t(b) << shift);
                        sel[1] = mask | (0xFFu << shift);
                        sel[2] = need - acc;          // still needed AT this prefix
                    }
                    sycl::group_barrier(it.get_group());
                }
                if (lane == 0) {
                    const uint32_t thr = sel[0];
                    uint32_t ties = sel[2];
                    int c = 0;
                    for (int n = 0; n < vis && c < topk; ++n) {
                        const uint32_t k = key_of(lg[n]);
                        if (k > thr) o[c++] = n;
                        else if (k == thr && ties) { o[c++] = n; --ties; }
                    }
                    for (; c < topk; ++c) o[c] = -1;
                }
            });
    });
}

// Selected block lists [rows][topk] (-1 = hole) -> bitmap [rows][words] for
// the masked flash kernel (launch_flash_prefill_qsa).
sycl::event launch_qsa_block_bits(sycl::queue& q, const int32_t* blocks, int rows,
                                  int topk, uint32_t* bits, int words,
                                  const std::vector<sycl::event>& deps) {
    sycl::event z = q.memset(bits, 0, size_t(rows) * words * sizeof(uint32_t), deps);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(z);
        h.parallel_for(sycl::range<1>(size_t(rows) * topk), [=](sycl::id<1> id) {
            const int row = int(id[0] / topk);
            const int b = blocks[id[0]];
            if (b < 0 || (b >> 5) >= words) return;
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                a(bits[size_t(row) * words + (b >> 5)]);
            a.fetch_or(1u << (b & 31));
        });
    });
}

sycl::event launch_qsa_expand_blocks(sycl::queue& q, const int32_t* blocks,
                                     int32_t* out, int rows, int block_topk,
                                     int compress_ratio, int token_topk,
                                     const int32_t* seq_len,
                                     const int32_t* query_pos,
                                     const std::vector<sycl::event>& deps) {
    const int width = token_topk + compress_ratio - 1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows) * width),
            [=](sycl::id<1> id) {
                const int row = int(id[0] / width);
                const int w   = int(id[0] % width);
                const int len = seq_len ? seq_len[row] : 0;
                int t = -1;
                if (w < token_topk) {
                    const int b = w / compress_ratio;
                    const int r = w % compress_ratio;
                    if (b < block_topk) {
                        const int bi = blocks[int64_t(row) * block_topk + b];
                        if (bi >= 0) {
                            const int cand = bi * compress_ratio + r;
                            if (cand >= 0 && cand < len) t = cand;
                        }
                    }
                } else {
                    const int i          = w - token_topk;
                    const int visible    = (query_pos ? query_pos[row] : 0) + 1;
                    const int tail_start = visible / compress_ratio * compress_ratio;
                    const int cand       = tail_start + i;
                    if (i < visible - tail_start && cand < len) t = cand;
                }
                out[int64_t(row) * width + w] = t;
            });
    });
}

// ---------------------------------------------------------------------
//  PLE -- the per-layer n-gram embedding (Qwen4-Exp).
//  Host reference and derivation: include/b70/qwen4_exp.hpp.
// ---------------------------------------------------------------------

// n-gram ids.  One work-item per (token, head).
//
// head h covers n-gram ORDER h/heads_per_ngram + 2, and only predecessors
// with shift < order enter its xor.  Mixing every predecessor into every
// head collapses the orders into one and is completely silent -- the
// hashes are still valid table indices.
//
// The EOS walk is STICKY: once a predecessor is EOS, every older one is
// treated as EOS too, which is what keeps an n-gram inside one document.
sycl::event launch_ple_ngram_ids(sycl::queue& q, const int32_t* tokens,
                                 int64_t* out, int first, int n_tokens,
                                 const int64_t* multipliers,
                                 const int64_t* sizes, const int64_t* offsets,
                                 int ngram_context_len, int heads_per_ngram,
                                 int ngram_heads, int eos_token_id,
                                 const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n_tokens) * ngram_heads),
            [=](sycl::id<1> id) {
                const int r  = int(id[0] / ngram_heads);
                const int hh = int(id[0] % ngram_heads);
                const int t  = first + r;
                const int order = hh / heads_per_ngram + 2;
                int64_t mixed = int64_t(tokens[t]) * multipliers[0];
                bool crossed = false;
                for (int shift = 1; shift <= ngram_context_len; ++shift) {
                    const int at = t - shift;
                    // before the sequence: EOS, as the reference's
                    // eos-filled previous_context (HF modeling_qwen4_exp
                    // Qwen4ExpTextNGramEmbedding.forward) -- not 0.
                    int64_t cand = (at >= 0) ? int64_t(tokens[at]) : int64_t(eos_token_id);
                    if (crossed) cand = eos_token_id;
                    if (cand == eos_token_id) crossed = true;
                    if (order > shift) mixed ^= cand * multipliers[shift];
                }
                int64_t rem = sizes[hh] ? (mixed % sizes[hh]) : 0;
                if (rem < 0) rem += sizes[hh];  // torch.remainder semantics
                out[int64_t(r) * ngram_heads + hh] = rem + offsets[hh];
            });
    });
}

// The PLE gate: d = dot(rms(key), rms(query)) / sqrt(H), then
// sigmoid(sign(d) * sqrt(max(|d|, 1e-6))).
//
// key and query are normalised SEPARATELY.  Normalising the concatenated
// pair, or skipping one, changes d's scale and the gate still lands in
// (0,1) -- nothing downstream would notice.
// The PLE gate.  One work-group per (token, stream): the norms are
// GROUPED by hidden, so every stream has its own variance, its own
// affine weights and its own gate scalar.  `value` is H wide and the
// same row is gated into every stream.
//
// Both outputs are produced here because the second is normed from the
// first: conv_in = norm_conv(gated), not norm_conv(value).
//
// The affine weights are the part that is silent when dropped: the
// docstring says "d = dot(RMSNorm(key), RMSNorm(query))" and
// Qwen4ExpPLEGroupedNorm applies (1 + w) inside that RMSNorm.
sycl::event launch_ple_gate(sycl::queue& q, const float* key,
                            const float* value, const float* query,
                            const bf16_t* nk, const bf16_t* nq,
                            const bf16_t* ncw, float* gated, float* conv_in,
                            int rows, int hc_count, int H, float eps,
                            const std::vector<sycl::event>& deps) {
    constexpr int WG = 128;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> rk(WG, h), rq(WG, h), rd(WG, h);
        h.parallel_for(sycl::nd_range<1>(size_t(rows) * hc_count * WG, WG),
            [=](sycl::nd_item<1> it) {
                const int gid  = int(it.get_group(0));
                const int row  = gid / hc_count;
                const int c    = gid % hc_count;
                const int lane = int(it.get_local_id(0));
                const int64_t base = (int64_t(row) * hc_count + c) * int64_t(H);
                const int64_t wb   = int64_t(c) * H;
                const float* k = key   + base;
                const float* x = query + base;
                const float* v = value + int64_t(row) * H;
                float ks = 0.0f, qs = 0.0f;
                for (int i = lane; i < H; i += WG) { ks += k[i]*k[i]; qs += x[i]*x[i]; }
                rk[lane] = ks; rq[lane] = qs;
                sycl::group_barrier(it.get_group());
                for (int s = WG/2; s; s >>= 1) {
                    if (lane < s) { rk[lane] += rk[lane+s]; rq[lane] += rq[lane+s]; }
                    sycl::group_barrier(it.get_group());
                }
                const float ki = sycl::rsqrt(rk[0]/float(H) + eps);
                const float qi = sycl::rsqrt(rq[0]/float(H) + eps);
                float dot = 0.0f;
                for (int i = lane; i < H; i += WG)
                    dot += (k[i]*ki*(1.0f + bf16_to_f32(nk[wb+i])))
                         * (x[i]*qi*(1.0f + bf16_to_f32(nq[wb+i])));
                rd[lane] = dot;
                sycl::group_barrier(it.get_group());
                for (int s = WG/2; s; s >>= 1) {
                    if (lane < s) rd[lane] += rd[lane+s];
                    sycl::group_barrier(it.get_group());
                }
                const float d   = rd[0] / sycl::sqrt(float(H));
                const float sgn = d < 0.0f ? -1.0f : 1.0f;
                const float mag = sycl::fabs(d) > 1e-6f ? sycl::fabs(d) : 1e-6f;
                const float g   = 1.0f / (1.0f + sycl::exp(-(sgn * sycl::sqrt(mag))));

                float* go = gated + base;
                for (int i = lane; i < H; i += WG) go[i] = g * v[i];
                sycl::group_barrier(it.get_group());
                float gs = 0.0f;
                for (int i = lane; i < H; i += WG) gs += go[i]*go[i];
                rk[lane] = gs;
                sycl::group_barrier(it.get_group());
                for (int s = WG/2; s; s >>= 1) {
                    if (lane < s) rk[lane] += rk[lane+s];
                    sycl::group_barrier(it.get_group());
                }
                const float gi = sycl::rsqrt(rk[0]/float(H) + eps);
                float* co = conv_in + base;
                for (int i = lane; i < H; i += WG)
                    co[i] = go[i] * gi * (1.0f + bf16_to_f32(ncw[wb+i]));
            });
    });
}

// The dilated depthwise short convolution that finishes a PLE layer.
// `conv_in` holds the current request's rows from position 0, so a tap
// before the start of the sequence is simply skipped -- which is what an
// all-zero conv state gives.  Both residuals are added: a PLE layer
// REPLACES the multi-stream state.
sycl::event launch_ple_conv(sycl::queue& q, const float* conv_in,
                            const float* gated, const float* hidden,
                            const bf16_t* w, float* out, int first, int rows,
                            int channels, int kernel, int dilation,
                            const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows) * channels),
            [=](sycl::id<1> id) {
                const int r = int(id[0] / channels);
                const int c = int(id[0] % channels);
                const int t = first + r;
                float acc = 0.0f;
                for (int k = 0; k < kernel; ++k) {
                    const int at = t - (kernel - 1 - k) * dilation;
                    if (at < 0) continue;
                    acc += bf16_to_f32(w[size_t(c) * kernel + k]) *
                           conv_in[int64_t(at) * channels + c];
                }
                const int64_t o = int64_t(r) * channels + c;
                out[o] = hidden[o] + gated[o] + acc / (1.0f + sycl::exp(-acc));
            });
    });
}

// The n-gram table gather.  `table` is HOST memory on purpose: 20,000,000
// rows is the architecture's own design (vLLM pins it rather than putting
// it in VRAM) and only the rows for the current tokens are read.  Every
// other weight in this engine is device-resident; this one is the
// exception the model expects.
// The table is BF16 or FP8-E4M3 with ONE global scale, which is what
// makes the real checkpoint's ~51B table a 51 GB host allocation instead
// of a 102 GB one.  The reference keeps the same two cases
// (Qwen4ExpPLEUnquantizedEmbeddingMethod / Qwen4ExpPLEFp8EmbeddingMethod)
// and rejects an FP8 table with no scale rather than assuming 1.0.
sycl::event launch_ple_embed_gather(sycl::queue& q, const void* table,
                                    bool fp8, float scale,
                                    const int64_t* ids, float* out, int rows,
                                    int ngram_heads, int head_dim,
                                    int64_t table_rows,
                                    const std::vector<sycl::event>& deps) {
    const int width = ngram_heads * head_dim;
    const uint8_t* raw = static_cast<const uint8_t*>(table);
    const bf16_t*  bf  = static_cast<const bf16_t*>(table);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows) * width),
            [=](sycl::id<1> id) {
                const int r  = int(id[0] / width);
                const int w  = int(id[0] % width);
                const int hh = w / head_dim;
                const int d  = w % head_dim;
                const int64_t row = ids[int64_t(r) * ngram_heads + hh];
                float v = 0.0f;
                if (row >= 0 && row < table_rows) {
                    const int64_t at = row * head_dim + d;
                    v = fp8 ? e4m3_to_f32(raw[at]) * scale : bf16_to_f32(bf[at]);
                }
                out[int64_t(r) * width + w] = v;
            });
    });
}

// mix(): silu(down_out / hc_count).  The divide is INSIDE the
// nonlinearity; moving it outside changes the gate's operating point and
// leaves the model fluent.
sycl::event launch_hc_silu(sycl::queue& q, float* x, int n, int hc_count,
                           const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            const float v = x[id[0]] / float(hc_count);
            x[id[0]] = v / (1.0f + sycl::exp(-v));
        });
    });
}

} // namespace b70
