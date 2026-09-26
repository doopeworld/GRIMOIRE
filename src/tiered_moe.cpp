// =====================================================================
//  tiered_moe.cpp -- NVFP4 routed-expert GEMVs that read each expert
//  through a pointer table (VRAM block or pinned host block).
//
//  Same shape as moe_kernels.cpp: two launches per layer, routing read
//  from device memory, activation staged once per work-group in SLM.
//  What differs is only where a row lives: eptr[e] + offset instead of
//  base + e * stride, so an expert can sit in VRAM or in host RAM.
//
//  NVFP4 row decode, per lane per step: 16 payload bytes = 32 elements =
//  two 16-wide blocks, each with its own E4M3 scale.  The element sum of
//  a block is formed first and scaled once (the scale is exact in bf16/f32,
//  so this is the same value as scaling every element), and the
//  projection's F32 scale is applied once per row at the end.
// =====================================================================
#include "kernels.hpp"
#include "b70/formats.hpp"
#include "b70/tiered_moe.hpp"

namespace b70 {
namespace {

inline float silu_t(float v) { return v / (1.0f + sycl::exp(-v)); }

// 32 elements of one NVFP4 row against 32 activations.  pv holds bytes
// 0..15 of the chunk (low nibble = even element); s2 the two block scales.
inline float nv_dot32(const sycl::uint4& pv, uint32_t s2, const float* xs,
                      const float* e2, const float* e4) {
    float a0 = 0.0f, a1 = 0.0f;
    #pragma unroll
    for (int d = 0; d < 2; ++d)
        #pragma unroll
        for (int b = 0; b < 8; ++b)
            a0 = sycl::fma(e2[(pv[d] >> (4 * b)) & 15u], xs[d * 8 + b], a0);
    #pragma unroll
    for (int d = 2; d < 4; ++d)
        #pragma unroll
        for (int b = 0; b < 8; ++b)
            a1 = sycl::fma(e2[(pv[d] >> (4 * b)) & 15u], xs[d * 8 + b], a1);
    return a0 * e4[s2 & 0xFFu] + a1 * e4[(s2 >> 8) & 0xFFu];
}

inline uint32_t load_s2(const uint8_t* p) {       // 2-byte aligned
    return uint32_t(*reinterpret_cast<const uint16_t*>(p));
}

// ---------------------------------------------------------------------
// gate + up.  One sub-group owns R (gate row, up row) pairs of one
// (token, slot); a work-group is WG_SUBGROUPS sub-groups of one token.
// ---------------------------------------------------------------------
template <int R>
sycl::event tmoe_gate_up_r(sycl::queue& q, const TieredMoeView& v,
                           const int32_t* d_expert, const float* x, float* h,
                           int M, const std::vector<sycl::event>& deps) {
    const NvExpertLayout L = v.lay;
    const int H = L.H, I = L.I, K = v.top_k;
    const int rows_per_wg = WG_SUBGROUPS * R;
    const int gpt = (K * I + rows_per_wg - 1) / rows_per_wg;
    const size_t n_groups = size_t(M) * gpt;
    const uint8_t* const* eptr = v.eptr;
    return q.submit([&](sycl::handler& hc) {
        hc.depends_on(deps);
        sycl::local_accessor<float, 1> slmx(size_t(H), hc);
        sycl::local_accessor<float, 1> e4s(256, hc);
        sycl::local_accessor<float, 1> e2s(16, hc);
        hc.parallel_for(
            sycl::nd_range<1>(n_groups * WG_SUBGROUPS * SG_SIZE, size_t(WG_SUBGROUPS) * SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int lid = int(it.get_local_id(0));
                const int lsz = WG_SUBGROUPS * SG_SIZE;
                float* e4 = e4s.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* e2 = e2s.template get_multi_ptr<sycl::access::decorated::no>().get();
                for (int b = lid; b < 256; b += lsz) e4[b] = e4m3_to_f32(uint8_t(b));
                if (lid < 16) e2[lid] = e2m1_to_f32(uint8_t(lid));
                const int token = int(it.get_group(0)) / gpt;
                const int lg    = int(it.get_group(0)) % gpt;
                const float* xt = x + int64_t(token) * H;
                for (int c = lid; c < H; c += lsz) slmx[c] = xt[c];
                sycl::group_barrier(it.get_group());
                const float* xs = slmx.template get_multi_ptr<sycl::access::decorated::no>().get();

                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  row0 = lg * rows_per_wg + int(sg.get_group_id()[0]) * R;

                const uint8_t* gp[R]; const uint8_t* up[R];
                const uint8_t* gs[R]; const uint8_t* us[R];
                float ga[R], ua[R], sgf[R], suf[R];
                bool  on[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    ga[r] = 0.0f; ua[r] = 0.0f; sgf[r] = 0.0f; suf[r] = 0.0f;
                    const int sr = row0 + r;
                    on[r] = sr < K * I;
                    gp[r] = up[r] = gs[r] = us[r] = nullptr;
                    if (!on[r]) continue;
                    const int slot = sr / I, i = sr % I;
                    const int e = d_expert[int64_t(token) * K + slot];
                    if (e < 0) { on[r] = false; continue; }
                    const uint8_t* blk = eptr[e];
                    gp[r] = blk + L.gu_p + size_t(i) * (H / 2);
                    up[r] = blk + L.gu_p + size_t(I + i) * (H / 2);
                    gs[r] = blk + L.gu_s + size_t(i) * (H / 16);
                    us[r] = blk + L.gu_s + size_t(I + i) * (H / 16);
                    const float* g3 = reinterpret_cast<const float*>(blk + L.gsc);
                    sgf[r] = g3[0]; suf[r] = g3[1];
                }
                for (int k0 = lane * 32; k0 < H; k0 += SG_SIZE * 32) {
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (!on[r]) continue;
                        const sycl::uint4 pg = *reinterpret_cast<const sycl::uint4*>(gp[r] + k0 / 2);
                        const sycl::uint4 pu = *reinterpret_cast<const sycl::uint4*>(up[r] + k0 / 2);
                        ga[r] += nv_dot32(pg, load_s2(gs[r] + k0 / 16), xs + k0, e2, e4);
                        ua[r] += nv_dot32(pu, load_s2(us[r] + k0 / 16), xs + k0, e2, e4);
                    }
                }
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float g = sycl::reduce_over_group(sg, ga[r], sycl::plus<float>());
                    const float u = sycl::reduce_over_group(sg, ua[r], sycl::plus<float>());
                    const int sr = row0 + r;
                    if (lane == 0 && sr < K * I)
                        h[int64_t(token) * K * I + sr] = on[r] ? silu_t(g * sgf[r]) * (u * suf[r]) : 0.0f;
                }
            });
    });
}

// ---------------------------------------------------------------------
// down + router-weighted sum over the k slots.  One sub-group owns R
// output rows of one token and walks all k experts.
// ---------------------------------------------------------------------
template <int R>
sycl::event tmoe_down_r(sycl::queue& q, const TieredMoeView& v,
                        const int32_t* d_expert, const float* d_weight,
                        const float* h, float* y, int M,
                        const std::vector<sycl::event>& deps) {
    const NvExpertLayout L = v.lay;
    const int H = L.H, I = L.I, K = v.top_k;
    const int rows_per_wg = WG_SUBGROUPS * R;
    const int gpt = (H + rows_per_wg - 1) / rows_per_wg;
    const size_t n_groups = size_t(M) * gpt;
    const uint8_t* const* eptr = v.eptr;
    return q.submit([&](sycl::handler& hc) {
        hc.depends_on(deps);
        sycl::local_accessor<float, 1> slmh(size_t(K) * I, hc);
        sycl::local_accessor<float, 1> e4s(256, hc);
        sycl::local_accessor<float, 1> e2s(16, hc);
        hc.parallel_for(
            sycl::nd_range<1>(n_groups * WG_SUBGROUPS * SG_SIZE, size_t(WG_SUBGROUPS) * SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int lid = int(it.get_local_id(0));
                const int lsz = WG_SUBGROUPS * SG_SIZE;
                float* e4 = e4s.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* e2 = e2s.template get_multi_ptr<sycl::access::decorated::no>().get();
                for (int b = lid; b < 256; b += lsz) e4[b] = e4m3_to_f32(uint8_t(b));
                if (lid < 16) e2[lid] = e2m1_to_f32(uint8_t(lid));
                const int token = int(it.get_group(0)) / gpt;
                const int lg    = int(it.get_group(0)) % gpt;
                const float* ht = h + int64_t(token) * K * I;
                for (int c = lid; c < K * I; c += lsz) slmh[c] = ht[c];
                sycl::group_barrier(it.get_group());
                const float* hs = slmh.template get_multi_ptr<sycl::access::decorated::no>().get();

                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  o0   = lg * rows_per_wg + int(sg.get_group_id()[0]) * R;
                float total[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) total[r] = 0.0f;
                for (int slot = 0; slot < K; ++slot) {
                    const int64_t route = int64_t(token) * K + slot;
                    const int   e  = d_expert[route];
                    const float rw = d_weight[route];
                    if (e < 0 || rw == 0.0f) continue;
                    const uint8_t* blk = eptr[e];
                    const float sd = reinterpret_cast<const float*>(blk + L.gsc)[2];
                    float acc[R];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
                    for (int k0 = lane * 32; k0 < I; k0 += SG_SIZE * 32) {
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            const int o = o0 + r;
                            if (o >= H) continue;
                            const sycl::uint4 pv = *reinterpret_cast<const sycl::uint4*>(
                                blk + L.dn_p + size_t(o) * (I / 2) + k0 / 2);
                            acc[r] += nv_dot32(pv, load_s2(blk + L.dn_s + size_t(o) * (I / 16) + k0 / 16),
                                               hs + slot * I + k0, e2, e4);
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < R; ++r) total[r] = sycl::fma(rw * sd, acc[r], total[r]);
                }
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int o = o0 + r;
                    const float t = sycl::reduce_over_group(sg, total[r], sycl::plus<float>());
                    if (lane == 0 && o < H) y[int64_t(token) * H + o] = t;
                }
            });
    });
}

} // namespace

sycl::event launch_tmoe_gate_up(sycl::queue& q, const TieredMoeView& v,
                                const int32_t* d_expert, const float* x, float* h,
                                int M, const std::vector<sycl::event>& deps) {
    return tmoe_gate_up_r<2>(q, v, d_expert, x, h, M, deps);
}

sycl::event launch_tmoe_down(sycl::queue& q, const TieredMoeView& v,
                             const int32_t* d_expert, const float* d_weight,
                             const float* h, float* y, int M,
                             const std::vector<sycl::event>& deps) {
    return tmoe_down_r<2>(q, v, d_expert, d_weight, h, y, M, deps);
}

} // namespace b70
