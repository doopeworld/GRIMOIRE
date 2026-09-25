// bench_flash_prefill -- candidate XMX causal prefill attention for the B70.
//
// Same contract as launch_flash_prefill (src/prefill.cpp):
//   q      fp32 [tokens][H][D]            (query t sits at position start+t)
//   k      e4m3 [KVH][D][seq_cap]         (d-major)
//   v      e4m3 [KVH][seq_cap][D]
//   out    fp32 [tokens][H][D]
// Causal over keys 0 .. start+t.  GQA: head h reads kv head h / (H/KVH).
//
// One work-group = one head x BQ=32 query rows, 8 sub-groups.  Per 32-key
// block: K and V are converted e4m3 -> bf16 into SLM in the packed (VNNI)
// layout the matrix unit reads, S = Q K^T and O += P V run on joint_matrix,
// and the online softmax runs on the S tile in SLM.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int SG = 16, TM = 8, TN = 16, TK = 16;

static float e4m3_to_f32_host(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 15 && m == 7) v = std::numeric_limits<float>::quiet_NaN();
    else if (e == 0) v = std::ldexp(float(m) / 8.0f, -6);
    else v = std::ldexp(1.0f + float(m) / 8.0f, e - 7);
    return s ? -v : v;
}

template <int D>
sycl::event flash_xmx(sycl::queue& q, const float* qv, const uint8_t* kc, const uint8_t* vc,
                      float* out, int tokens, int start, int H, int KVH, int seq_cap,
                      float scale, const bf16* lut /*256 entries, e4m3 -> bf16*/) {
    constexpr int BQ = 32, BK = 32, NSG = 8, WG = NSG * SG;
    constexpr int DC = D / 2;                 // O columns per sub-group (2 column halves)
    static_assert(D % 32 == 0, "D");
    const int qtiles = (tokens + BQ - 1) / BQ;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<bf16, 1> Qs(BQ * D, h);              // row-major A
        sycl::local_accessor<bf16, 1> KVs(2 * BK * D, h);         // K^T packed | V packed
        sycl::local_accessor<float, 1> Ss(BQ * BK, h);
        sycl::local_accessor<bf16, 1> Ps(BQ * BK, h);
        sycl::local_accessor<float, 1> stat(3 * BQ, h);           // m | l | corr
        sycl::local_accessor<bf16, 1> L(256, h);
        h.parallel_for(sycl::nd_range<1>(size_t(qtiles) * H * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]);
            const int lane = int(sg.get_local_id()[0]);
            const int lid = int(it.get_local_id(0));
            const int g = int(it.get_group(0));
            const int qt = g / H, hq = g % H;
            const int kvh = hq / (H / KVH);
            const int q0 = qt * BQ;
            bf16* qs = Qs.template get_multi_ptr<sycl::access::decorated::no>().get();
            bf16* ks = KVs.template get_multi_ptr<sycl::access::decorated::no>().get();
            bf16* vs = ks + BK * D;
            float* ss = Ss.template get_multi_ptr<sycl::access::decorated::no>().get();
            bf16* ps = Ps.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* mrow = stat.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* lrow = mrow + BQ;
            float* crow = lrow + BQ;
            bf16* lt = L.template get_multi_ptr<sycl::access::decorated::no>().get();
            for (int i = lid; i < 256; i += WG) lt[i] = lut[i];
            for (int i = lid; i < BQ * D; i += WG) {
                const int r = i / D, d = i % D, t = q0 + r;
                qs[i] = bf16(t < tokens ? qv[(int64_t(t) * H + hq) * D + d] * scale : 0.0f);
            }
            for (int i = lid; i < BQ; i += WG) {
                mrow[i] = -std::numeric_limits<float>::infinity(); lrow[i] = 0.0f;
            }
            const uint8_t* kh = kc + int64_t(kvh) * D * seq_cap;
            const uint8_t* vh = vc + int64_t(kvh) * seq_cap * D;
            const int last_q = sycl::min(tokens - 1, q0 + BQ - 1);
            const int kend = start + last_q + 1;                   // keys this tile needs
            // O accumulators: sub-group owns rows rb*8..+8, columns ch*DC..+DC.
            const int rb = sgid % 4, ch = sgid / 4;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> o[DC / TN];
            #pragma unroll
            for (int n = 0; n < DC / TN; ++n) mx::joint_matrix_fill(sg, o[n], 0.0f);
            sycl::group_barrier(it.get_group());
            for (int s0 = 0; s0 < kend; s0 += BK) {
                // ---- stage K^T (packed [D/2][BK][2]) and V (packed [BK/2][D][2])
                for (int i = lid; i < BK * D; i += WG) {
                    const int d = i / BK, j = i % BK, s = s0 + j;   // K: j fastest
                    ks[((d >> 1) * BK + j) * 2 + (d & 1)] =
                        s < kend ? lt[kh[int64_t(d) * seq_cap + s]] : bf16(0.0f);
                }
                for (int i = lid; i < BK * D; i += WG) {
                    const int j = i / D, d = i % D, s = s0 + j;     // V: d fastest
                    vs[((j >> 1) * D + d) * 2 + (j & 1)] =
                        s < kend ? lt[vh[int64_t(s) * D + d]] : bf16(0.0f);
                }
                sycl::group_barrier(it.get_group());
                // ---- S = Q K^T: sub-group owns rows (sgid%4)*8, cols (sgid/4)*16
                {
                    const int srb = sgid % 4, scb = sgid / 4;
                    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> sacc;
                    mx::joint_matrix_fill(sg, sacc, 0.0f);
                    #pragma unroll 4
                    for (int k = 0; k < D; k += TK) {
                        mx::joint_matrix<sycl::sub_group, bf16, mx::use::a, TM, TK,
                                         mx::layout::row_major> a;
                        mx::joint_matrix<sycl::sub_group, bf16, mx::use::b, TK, TN,
                                         mx::layout::ext_intel_packed> b;
                        mx::joint_matrix_load(sg, a,
                            Qs.template get_multi_ptr<sycl::access::decorated::no>() +
                                srb * TM * D + k, D);
                        mx::joint_matrix_load(sg, b,
                            KVs.template get_multi_ptr<sycl::access::decorated::no>() +
                                (k / 2) * BK * 2 + scb * TN * 2, BK * 2);
                        mx::joint_matrix_mad(sg, sacc, a, b, sacc);
                    }
                    mx::joint_matrix_store(sg, sacc,
                        Ss.template get_multi_ptr<sycl::access::decorated::no>() +
                            srb * TM * BK + scb * TN, BK, mx::layout::row_major);
                }
                sycl::group_barrier(it.get_group());
                // ---- online softmax: sub-group owns 4 rows, lanes own 2 columns
                #pragma unroll
                for (int rr = 0; rr < 4; ++rr) {
                    const int r = sgid * 4 + rr;
                    const int qpos = start + q0 + r;
                    const bool rowok = q0 + r < tokens;
                    float sv[2];
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int col = lane + c * SG, key = s0 + col;
                        sv[c] = (rowok && key <= qpos) ? ss[r * BK + col]
                                                       : -std::numeric_limits<float>::infinity();
                    }
                    const float bm = sycl::reduce_over_group(sg, sycl::fmax(sv[0], sv[1]),
                                                             sycl::maximum<float>());
                    const float mo = mrow[r];
                    const float mn = sycl::fmax(mo, bm);
                    const float corr = sycl::isinf(mo) ? 0.0f : sycl::exp(mo - mn);
                    float psum = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const float p = sycl::isinf(sv[c]) ? 0.0f : sycl::exp(sv[c] - mn);
                        psum += p;
                        ps[r * BK + lane + c * SG] = bf16(p);
                    }
                    psum = sycl::reduce_over_group(sg, psum, sycl::plus<float>());
                    if (lane == 0) {
                        mrow[r] = mn; lrow[r] = lrow[r] * corr + psum; crow[r] = corr;
                    }
                }
                sycl::group_barrier(it.get_group());
                // ---- O = O * corr + P V
                #pragma unroll
                for (int n = 0; n < DC / TN; ++n)
                    sycl::ext::intel::experimental::matrix::joint_matrix_apply(sg, o[n], [=](float& x, size_t row, size_t) {
                        x *= crow[rb * TM + int(row)];
                    });
                #pragma unroll
                for (int kk = 0; kk < BK; kk += TK) {
                    mx::joint_matrix<sycl::sub_group, bf16, mx::use::a, TM, TK,
                                     mx::layout::row_major> a;
                    mx::joint_matrix_load(sg, a,
                        Ps.template get_multi_ptr<sycl::access::decorated::no>() +
                            rb * TM * BK + kk, BK);
                    #pragma unroll
                    for (int n = 0; n < DC / TN; ++n) {
                        mx::joint_matrix<sycl::sub_group, bf16, mx::use::b, TK, TN,
                                         mx::layout::ext_intel_packed> b;
                        mx::joint_matrix_load(sg, b,
                            KVs.template get_multi_ptr<sycl::access::decorated::no>() + BK * D +
                                (kk / 2) * D * 2 + (ch * DC + n * TN) * 2, D * 2);
                        mx::joint_matrix_mad(sg, o[n], a, b, o[n]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            // ---- epilogue: O / l through SLM (reuses the K/V buffer as fp32 [BQ][D])
            float* ob = reinterpret_cast<float*>(ks);
            #pragma unroll
            for (int n = 0; n < DC / TN; ++n)
                mx::joint_matrix_store(sg, o[n],
                    sycl::address_space_cast<sycl::access::address_space::local_space,
                                             sycl::access::decorated::no>(ob) +
                        rb * TM * D + ch * DC + n * TN, D, mx::layout::row_major);
            sycl::group_barrier(it.get_group());
            for (int i = lid; i < BQ * D; i += WG) {
                const int r = i / D, d = i % D, t = q0 + r;
                if (t < tokens) {
                    const float l = lrow[r];
                    out[(int64_t(t) * H + hq) * D + d] = l > 0.0f ? ob[i] / l : 0.0f;
                }
            }
        });
    });
}

// ---------------------------------------------------------------------
// v2.  A pre-pass converts the e4m3 cache ONCE into the packed bf16 layouts
// the matrix unit reads (keys >= kend zeroed):
//   Kp [KVH][D/2][Sp][2]   B operand of S = Q K^T  (k = d,   n = key)
//   Vp [KVH][Sp/2][D][2]   B operand of O = P V    (k = key, n = d)
// Then every sub-group owns 8 query rows end to end: Q in its own SLM slice,
// K/V fragments straight from global memory, softmax on its own S/P slice.
// No work-group barrier anywhere in the key loop.
// ---------------------------------------------------------------------
template <int D>
void kv_pack(sycl::queue& q, const uint8_t* kc, const uint8_t* vc, bf16* Kp, bf16* Vp,
             int KVH, int seq_cap, int kend, int Sp, const bf16* lut) {
    q.parallel_for(sycl::range<3>(size_t(KVH), size_t(D), size_t(Sp)), [=](sycl::id<3> id) {
        const int h = int(id[0]), d = int(id[1]), s = int(id[2]);          // key fastest
        Kp[((size_t(h) * (D / 2) + d / 2) * Sp + s) * 2 + (d & 1)] =
            s < kend ? lut[kc[(size_t(h) * D + d) * seq_cap + s]] : bf16(0.0f);
    });
    q.parallel_for(sycl::range<3>(size_t(KVH), size_t(Sp), size_t(D)), [=](sycl::id<3> id) {
        const int h = int(id[0]), s = int(id[1]), d = int(id[2]);          // d fastest
        Vp[((size_t(h) * (Sp / 2) + s / 2) * D + d) * 2 + (s & 1)] =
            s < kend ? lut[vc[(size_t(h) * seq_cap + s) * D + d]] : bf16(0.0f);
    });
}

template <int D>
sycl::event flash_xmx2(sycl::queue& q, const float* qv, const bf16* Kp, const bf16* Vp,
                       float* out, int tokens, int start, int H, int KVH, int Sp, float scale) {
    constexpr int NSG = 8, RB = TM, BQ = NSG * RB, BK = 32, WG = NSG * SG, NF = D / TN;
    const int qtiles = (tokens + BQ - 1) / BQ;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<bf16, 1> Qs(NSG * RB * D, h);
        sycl::local_accessor<float, 1> Ss(NSG * RB * BK, h);
        sycl::local_accessor<bf16, 1> Ps(NSG * RB * BK, h);
        h.parallel_for(sycl::nd_range<1>(size_t(qtiles) * H * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            namespace ix = sycl::ext::intel::experimental::matrix;
            constexpr auto NO = sycl::access::decorated::no;
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_id()[0]), lane = int(sg.get_local_id()[0]);
            const int g = int(it.get_group(0)), qt = g / H, hq = g % H, kvh = hq / (H / KVH);
            const int r0 = qt * BQ + sgid * RB;
            if (r0 >= tokens) return;                   // uniform per sub-group; no WG barriers below
            auto qs_mp = Qs.template get_multi_ptr<NO>() + sgid * RB * D;
            auto ss_mp = Ss.template get_multi_ptr<NO>() + sgid * RB * BK;
            auto ps_mp = Ps.template get_multi_ptr<NO>() + sgid * RB * BK;
            bf16* qs = qs_mp.get(); float* ss = ss_mp.get(); bf16* ps = ps_mp.get();
            for (int i = lane; i < RB * D; i += SG) {
                const int r = i / D, d = i % D, t = r0 + r;
                qs[i] = bf16(t < tokens ? qv[(size_t(t) * H + hq) * D + d] * scale : 0.0f);
            }
            sycl::group_barrier(sg);
            auto kg = sycl::address_space_cast<sycl::access::address_space::global_space, NO>(
                Kp + size_t(kvh) * (D / 2) * Sp * 2);
            auto vg = sycl::address_space_cast<sycl::access::address_space::global_space, NO>(
                Vp + size_t(kvh) * (Sp / 2) * D * 2);
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> o[NF];
            #pragma unroll
            for (int n = 0; n < NF; ++n) mx::joint_matrix_fill(sg, o[n], 0.0f);
            float m[RB], l[RB];
            #pragma unroll
            for (int r = 0; r < RB; ++r) { m[r] = -std::numeric_limits<float>::infinity(); l[r] = 0.0f; }
            const int last_row = sycl::min(tokens - 1, r0 + RB - 1);
            const int kend = start + last_row + 1;
            for (int s0 = 0; s0 < kend; s0 += BK) {
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> sa[BK / TN];
                #pragma unroll
                for (int c = 0; c < BK / TN; ++c) mx::joint_matrix_fill(sg, sa[c], 0.0f);
                #pragma unroll 4
                for (int k = 0; k < D; k += TK) {
                    mx::joint_matrix<sycl::sub_group, bf16, mx::use::a, TM, TK,
                                     mx::layout::row_major> a;
                    mx::joint_matrix_load(sg, a, qs_mp + k, D);
                    #pragma unroll
                    for (int c = 0; c < BK / TN; ++c) {
                        mx::joint_matrix<sycl::sub_group, bf16, mx::use::b, TK, TN,
                                         mx::layout::ext_intel_packed> b;
                        mx::joint_matrix_load(sg, b,
                            kg + size_t(k / 2) * Sp * 2 + size_t(s0 + c * TN) * 2, size_t(Sp) * 2);
                        mx::joint_matrix_mad(sg, sa[c], a, b, sa[c]);
                    }
                }
                #pragma unroll
                for (int c = 0; c < BK / TN; ++c)
                    mx::joint_matrix_store(sg, sa[c], ss_mp + c * TN, BK, mx::layout::row_major);
                sycl::group_barrier(sg);
                float corr[RB];
                #pragma unroll
                for (int r = 0; r < RB; ++r) {
                    const int t = r0 + r, qpos = start + t;
                    const float v0 = (t < tokens && s0 + lane <= qpos) ? ss[r * BK + lane]
                                     : -std::numeric_limits<float>::infinity();
                    const float v1 = (t < tokens && s0 + lane + SG <= qpos) ? ss[r * BK + lane + SG]
                                     : -std::numeric_limits<float>::infinity();
                    const float bm = sycl::reduce_over_group(sg, sycl::fmax(v0, v1),
                                                             sycl::maximum<float>());
                    const float mn = sycl::fmax(m[r], bm);
                    corr[r] = sycl::isinf(m[r]) ? 0.0f : sycl::exp(m[r] - mn);
                    const float p0 = sycl::isinf(v0) ? 0.0f : sycl::exp(v0 - mn);
                    const float p1 = sycl::isinf(v1) ? 0.0f : sycl::exp(v1 - mn);
                    ps[r * BK + lane] = bf16(p0);
                    ps[r * BK + lane + SG] = bf16(p1);
                    l[r] = l[r] * corr[r] + sycl::reduce_over_group(sg, p0 + p1, sycl::plus<float>());
                    m[r] = mn;
                }
                sycl::group_barrier(sg);
                #pragma unroll
                for (int n = 0; n < NF; ++n)
                    ix::joint_matrix_apply(sg, o[n], [&](float& x, size_t row, size_t) { x *= corr[row]; });
                #pragma unroll
                for (int kk = 0; kk < BK; kk += TK) {
                    mx::joint_matrix<sycl::sub_group, bf16, mx::use::a, TM, TK,
                                     mx::layout::row_major> a;
                    mx::joint_matrix_load(sg, a, ps_mp + kk, BK);
                    #pragma unroll
                    for (int n = 0; n < NF; ++n) {
                        mx::joint_matrix<sycl::sub_group, bf16, mx::use::b, TK, TN,
                                         mx::layout::ext_intel_packed> b;
                        mx::joint_matrix_load(sg, b,
                            vg + size_t((s0 + kk) / 2) * D * 2 + size_t(n * TN) * 2, size_t(D) * 2);
                        mx::joint_matrix_mad(sg, o[n], a, b, o[n]);
                    }
                }
                sycl::group_barrier(sg);
            }
            float inv[RB];
            #pragma unroll
            for (int r = 0; r < RB; ++r) inv[r] = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
            #pragma unroll
            for (int n = 0; n < NF; ++n)
                ix::joint_matrix_apply(sg, o[n], [&](float& x, size_t row, size_t) { x *= inv[row]; });
            auto og = sycl::address_space_cast<sycl::access::address_space::global_space, NO>(out);
            if (r0 + RB <= tokens) {
                #pragma unroll
                for (int n = 0; n < NF; ++n)
                    mx::joint_matrix_store(sg, o[n], og + (size_t(r0) * H + hq) * D + n * TN,
                                           size_t(H) * D, mx::layout::row_major);
            } else {
                for (int n = 0; n < NF; ++n) {        // ragged last tile: through the S slice
                    mx::joint_matrix_store(sg, o[n], ss_mp, TN, mx::layout::row_major);
                    sycl::group_barrier(sg);
                    for (int e = lane; e < RB * TN; e += SG) {
                        const int r = e / TN, c = e % TN, t = r0 + r;
                        if (t < tokens) out[(size_t(t) * H + hq) * D + n * TN + c] = ss[e];
                    }
                    sycl::group_barrier(sg);
                }
            }
        });
    });
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    constexpr int D = 256;
    std::vector<bf16> hl(256);
    for (int i = 0; i < 256; ++i) hl[i] = bf16(e4m3_to_f32_host(uint8_t(i)));
    bf16* lut = sycl::malloc_device<bf16>(256, q);
    q.memcpy(lut, hl.data(), 256 * sizeof(bf16)).wait();

    auto run_case = [&](int tokens, int start, int H, int KVH, bool check, int iters) {
        const int seq_cap = start + tokens + 64;
        std::mt19937 rng(tokens * 131 + start);
        std::uniform_real_distribution<float> uq(-1.f, 1.f);
        std::vector<float> hq(size_t(tokens) * H * D);
        for (auto& x : hq) x = uq(rng);
        std::vector<uint8_t> hk(size_t(KVH) * D * seq_cap), hv(size_t(KVH) * seq_cap * D);
        for (auto& b : hk) { b = uint8_t(rng() & 0xff); if ((b & 0x7f) == 0x7f) b = 0x38; }
        for (auto& b : hv) { b = uint8_t(rng() & 0xff); if ((b & 0x7f) == 0x7f) b = 0x38; }
        float* dq = sycl::malloc_device<float>(hq.size(), q);
        uint8_t* dk = sycl::malloc_device<uint8_t>(hk.size(), q);
        uint8_t* dv = sycl::malloc_device<uint8_t>(hv.size(), q);
        float* dout = sycl::malloc_device<float>(hq.size(), q);
        q.memcpy(dq, hq.data(), hq.size() * 4); q.memcpy(dk, hk.data(), hk.size());
        q.memcpy(dv, hv.data(), hv.size()).wait();
        const float sc = 1.0f / std::sqrt(float(D));
        const int kend = start + tokens, Sp = (kend + 31) / 32 * 32;
        bf16* Kp = sycl::malloc_device<bf16>(size_t(KVH) * Sp * D, q);
        bf16* Vp = sycl::malloc_device<bf16>(size_t(KVH) * Sp * D, q);
        auto attn = [&]() {
            kv_pack<D>(q, dk, dv, Kp, Vp, KVH, seq_cap, kend, Sp, lut);
            return flash_xmx2<D>(q, dq, Kp, Vp, dout, tokens, start, H, KVH, Sp, sc);
        };
        attn().wait();
        double worst = 0;
        if (check) {
            std::vector<float> ho(hq.size());
            q.memcpy(ho.data(), dout, ho.size() * 4).wait();
            std::mt19937 pick(3);
            for (int n = 0; n < 48; ++n) {
                const int t = int(pick() % tokens), h = int(pick() % H), kvh = h / (H / KVH);
                const int kmax = start + t;
                std::vector<double> sc_(kmax + 1);
                double m = -1e300;
                for (int s = 0; s <= kmax; ++s) {
                    double dot = 0;
                    for (int d = 0; d < D; ++d)
                        dot += double(float(bf16(hq[(size_t(t) * H + h) * D + d] * sc))) *
                               double(e4m3_to_f32_host(hk[(size_t(kvh) * D + d) * seq_cap + s]));
                    sc_[s] = dot; m = std::max(m, dot);
                }
                double l = 0;
                for (int s = 0; s <= kmax; ++s) { sc_[s] = std::exp(sc_[s] - m); l += sc_[s]; }
                for (int d = 0; d < D; d += 7) {
                    double o = 0;
                    for (int s = 0; s <= kmax; ++s)
                        o += sc_[s] * double(e4m3_to_f32_host(hv[(size_t(kvh) * seq_cap + s) * D + d]));
                    o /= l;
                    const double got = ho[(size_t(t) * H + h) * D + d];
                    worst = std::max(worst, std::fabs(got - o) / (std::fabs(o) + 0.05));
                }
            }
        }
        double ms = 0;
        if (iters) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) attn();
            q.wait();
            ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3 / iters;
        }
        const double flop = 4.0 * H * D * (double(tokens) * start + double(tokens) * (tokens + 1) / 2.0);
        std::printf("tokens=%5d start=%4d H=%d KVH=%d  %s  %8.3f ms  %6.1f TFLOP/s\n",
                    tokens, start, H, KVH,
                    check ? (worst < 2e-2 ? "check OK " : "check BAD") : "         ",
                    ms, ms > 0 ? flop / (ms * 1e-3) / 1e12 : 0.0);
        if (check) std::printf("    worst relative error %.3e\n", worst);
        sycl::free(dq, q); sycl::free(dk, q); sycl::free(dv, q); sycl::free(dout, q);
        sycl::free(Kp, q); sycl::free(Vp, q);
    };
    run_case(200, 0, 4, 2, true, 0);        // ragged tile, GQA
    run_case(100, 37, 4, 1, true, 0);       // resumed prompt (start > 0)
    run_case(5987, 0, 24, 4, true, 5);      // Qwen3.8-27B, one layer
    run_case(5987, 0, 16, 2, false, 5);     // Ornith-1.5, one layer
    sycl::free(lut, q);
    return 0;
}
