// =====================================================================
//  prefill.cpp  --  prompt processing
//
//  Decode and prefill are different problems and need different kernels.
//
//  DECODE (batch 1) is bandwidth bound: every weight is read to produce
//  one token, so the whole game is streaming bytes.
//
//  PREFILL (batch M) is compute bound: each weight is read ONCE and used
//  M times. For a 4096-token prompt that is a 4096x reduction in weight
//  traffic, and the cost moves entirely into arithmetic.
//
//  Running decode M times instead would re-read 18 GB per token -- about
//  21 seconds for a 4096-token prompt. That is the difference between
//  ~200 t/s and several thousand.
// =====================================================================
#include "kernels.hpp"
#include "gemv_step.hpp"
#include <algorithm>
#include <limits>

namespace b70 {
namespace {

// ---------------------------------------------------------------------
// Batched GEMM: Y[M][N] = X[M][K] * W[N][K]^T
//
// One sub-group owns one output row n. It walks that weight row ONCE,
// keeping a slab of it in registers, and applies it to all M tokens.
// Weight traffic is N*K regardless of M, which is the entire point.
// ---------------------------------------------------------------------
template <Fmt F, int EPL>
sycl::event gemm_batched_impl(sycl::queue& q, const QuantWeight& w,
                              const float* x, float* y, int M,
                              const std::vector<sycl::event>& deps) {
    const int N = w.N, K = w.K;
    constexpr int MTILE = 8;               // tokens held in registers at once
    const int rows_per_wg = WG_SUBGROUPS;
    const int n_groups    = (N + rows_per_wg - 1) / rows_per_wg;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        sycl::local_accessor<float, 1> lut_slm(256, h);
        sycl::local_accessor<float, 1> e8m0_slm(256, h);
        sycl::local_accessor<float, 1> e2m1_slm(16, h);

        const int wg_threads = rows_per_wg * SG_SIZE;
        h.parallel_for(
            sycl::nd_range<1>(size_t(n_groups) * wg_threads, wg_threads),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  lid  = int(it.get_local_id(0));

                float* lut  = lut_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* slut = e8m0_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* nlut = e2m1_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
                if constexpr (F == Fmt::FP8_E4M3 || F == Fmt::MXFP8)
                    for (int b = lid; b < 256; b += wg_threads) lut[b] = e4m3_to_f32(uint8_t(b));
                else if constexpr (F == Fmt::FP8_E5M2)
                    for (int b = lid; b < 256; b += wg_threads) lut[b] = e5m2_to_f32(uint8_t(b));
                if constexpr (Traits<F>::block == kMXBlock)
                    for (int b = lid; b < 256; b += wg_threads) slut[b] = e8m0_to_f32(uint8_t(b));
                if constexpr (F == Fmt::MXFP4)
                    for (int b = lid; b < 16; b += wg_threads) nlut[b] = e2m1_to_f32(uint8_t(b));
                sycl::group_barrier(it.get_group());

                const int n = int(it.get_group(0)) * rows_per_wg + int(sg.get_group_id()[0]);
                if (n >= N) return;
                const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;

                // Process tokens in tiles so the decoded weight slab is
                // reused MTILE times before it leaves registers.
                for (int m0 = 0; m0 < M; m0 += MTILE) {
                    const int mt = (M - m0 < MTILE) ? (M - m0) : MTILE;
                    float acc[MTILE];
                    #pragma unroll
                    for (int t = 0; t < MTILE; ++t) acc[t] = 0.0f;

                    constexpr int STEP = SG_SIZE * EPL;
                    for (int base = 0; base + STEP <= K; base += STEP) {
                        const int k0 = base + lane * EPL;
                        for (int t = 0; t < mt; ++t)
                            acc[t] += GemvStep<F, EPL>::run(
                                wc, row, x + int64_t(m0 + t) * K, lut, slut, nlut, n, k0);
                    }
                    const int done = (K / STEP) * STEP;
                    for (int t = 0; t < mt; ++t) {
                        for (int k = done + lane; k < K; k += SG_SIZE)
                            acc[t] = sycl::fma(wc.at(n, k), x[int64_t(m0 + t) * K + k], acc[t]);
                        const float tot = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
                        if (lane == 0) y[int64_t(m0 + t) * N + n] = tot;
                    }
                }
            });
    });
}

} // namespace

sycl::event launch_gemm_batched(sycl::queue& q, const QuantWeight& w,
                                const float* x, float* y, int M,
                                const std::vector<sycl::event>& deps) {
    switch (w.fmt) {
        case Fmt::BF16:     return gemm_batched_impl<Fmt::BF16, 16>(q, w, x, y, M, deps);
        case Fmt::FP8_E4M3: return gemm_batched_impl<Fmt::FP8_E4M3, 16>(q, w, x, y, M, deps);
        case Fmt::FP8_E5M2: return gemm_batched_impl<Fmt::FP8_E5M2, 16>(q, w, x, y, M, deps);
        case Fmt::INT8:     return gemm_batched_impl<Fmt::INT8, 16>(q, w, x, y, M, deps);
        case Fmt::INT4:     return gemm_batched_impl<Fmt::INT4, 16>(q, w, x, y, M, deps);
        case Fmt::MXFP8:    return gemm_batched_impl<Fmt::MXFP8, 16>(q, w, x, y, M, deps);
        case Fmt::MXFP4:    return gemm_batched_impl<Fmt::MXFP4, 16>(q, w, x, y, M, deps);
    }
    return {};
}

// ---------------------------------------------------------------------
// DeltaNet prefill.
//
// The recurrence is sequential by construction: token t's state depends
// on t-1. It cannot be parallelised across tokens without the chunkwise
// matrix reformulation. But it does NOT need to be: the expensive part
// of a naive implementation is re-reading the 2 MiB state from VRAM for
// every token (4096 tokens x 30 layers = 245 GB, which would dominate
// everything).
//
// Instead one work-group owns one head and holds that head's entire
// [v_dim][k_dim] state in SLM -- 64 KiB for 128x128, inside the 128 KiB
// budget. It then walks all M tokens with the state resident, and writes
// it back once. State traffic becomes 2 MiB per LAYER instead of per
// token, and the sequential dependency costs nothing because 32 heads x
// 30 layers gives plenty of independent work to fill the machine.
// ---------------------------------------------------------------------
sycl::event launch_deltanet_prefill(sycl::queue& q, const DeltaNetPrefillParams& p,
                                    const std::vector<sycl::event>& deps) {
    const int KD = p.k_dim, VD = p.v_dim;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const DeltaNetPrefillParams pp = p;
        sycl::local_accessor<float, 1> smem(size_t(VD) * KD, h);   // the state
        sycl::local_accessor<float, 1> qs(KD, h);
        sycl::local_accessor<float, 1> ks(KD, h);

        h.parallel_for(
            sycl::nd_range<1>(size_t(pp.n_heads) * size_t(VD), size_t(VD)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int head = int(it.get_group(0));
                const int row  = int(it.get_local_id(0));
                if (head >= pp.n_heads) return;

                float* S   = smem.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* qsp = qs.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* ksp = ks.template get_multi_ptr<sycl::access::decorated::no>().get();

                // load this head's state into SLM once
                float* Sg = pp.state + int64_t(head) * VD * KD;
                for (int j = 0; j < KD; ++j) S[row * KD + j] = Sg[row * KD + j];
                sycl::group_barrier(it.get_group());

                for (int t = 0; t < pp.n_tokens; ++t) {
                    // q/k shared across pairs of v-heads, as in decode
                    const int nk    = pp.n_k_heads ? pp.n_k_heads : pp.n_heads;
                    const int khead = (nk == pp.n_heads) ? head : head / (pp.n_heads / nk);
                    const float* qt = pp.q + (int64_t(t) * nk + khead) * KD;
                    const float* kt = pp.k + (int64_t(t) * nk + khead) * KD;
                    const float* vt = pp.v + (int64_t(t) * pp.n_heads + head) * VD;

                        for (int j = row; j < KD; j += VD) { qsp[j] = qt[j]; ksp[j] = kt[j]; }
                    sycl::group_barrier(it.get_group());

                    const float av = pp.a[int64_t(t) * pp.n_heads + head];
                    const float bv = pp.beta[int64_t(t) * pp.n_heads + head];
                    const float vi = vt[row];

                    float* Sr = S + row * KD;
                    float w = 0.0f;
                    for (int j = 0; j < KD; ++j) w = sycl::fma(Sr[j], ksp[j], w);

                    const float corr = bv * (vi - av * w);
                    float o = 0.0f;
                    for (int j = 0; j < KD; ++j) {
                        const float sv = sycl::fma(av, Sr[j], corr * ksp[j]);
                        Sr[j] = sv;
                        o = sycl::fma(sv, qsp[j], o);
                    }
                    pp.out[(int64_t(t) * pp.n_heads + head) * VD + row] =
                        o * sycl::rsqrt(float(KD));
                    sycl::group_barrier(it.get_group());
                }

                // write the carried state back for the decode phase
                for (int j = 0; j < KD; ++j) Sg[row * KD + j] = S[row * KD + j];
            });
    });
}

// One work-group per token.  This is the batched counterpart of the
// decode residual+RMSNorm fusion and intentionally uses the same reduction
// order inside each row, making state-equivalence failures easy to localise.
sycl::event launch_rmsnorm_residual_batched(
    sycl::queue& q, float* h, const float* r0, const float* r1,
    const bf16_t* weight, float* out, int tokens, int hidden, float eps,
    sycl_bf16* out_bf,
    const std::vector<sycl::event>& deps, float weight_offset) {
    constexpr int WG = 256;
    return q.submit([&](sycl::handler& hd) {
        hd.depends_on(deps);
        sycl::local_accessor<float, 1> partial(WG / SG_SIZE, hd);
        hd.parallel_for(
            sycl::nd_range<1>(size_t(tokens) * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int t = int(it.get_group(0));
                const int lid = int(it.get_local_id(0));
                const int sgid = int(sg.get_group_id()[0]);
                const int lane = int(sg.get_local_id()[0]);
                float* ht = h + int64_t(t) * hidden;
                float* ot = out ? out + int64_t(t) * hidden : nullptr;
                sycl_bf16* obt = out_bf ? out_bf + int64_t(t) * hidden : nullptr;
                const float* a = r0 ? r0 + int64_t(t) * hidden : nullptr;
                const float* b = r1 ? r1 + int64_t(t) * hidden : nullptr;
                float ss = 0.0f;
                for (int i = lid; i < hidden; i += WG) {
                    float v = ht[i];
                    if (a) v += a[i];
                    if (b) v += b[i];
                    if (a || b) ht[i] = v;
                    ss = sycl::fma(v, v, ss);
                }
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                float* pt = partial.template get_multi_ptr<sycl::access::decorated::no>().get();
                if (lane == 0) pt[sgid] = ss;
                sycl::group_barrier(it.get_group());
                float total = 0.0f;
                for (int i = 0; i < WG / SG_SIZE; ++i) total += pt[i];
                const float scale = sycl::rsqrt(total / float(hidden) + eps);
                for (int i = lid; i < hidden; i += WG) {
                    const float v=ht[i]*scale*(weight_offset+bf16_to_f32(weight[i]));
                    if(ot)ot[i]=v;if(obt)obt[i]=sycl_bf16(v);
                }
            });
    });
}

sycl::event launch_rmsnorm_residual_f16_batched(
    sycl::queue& q, float* h, const float* residual, const bf16_t* weight,
    float* out, int tokens, int hidden, float eps,
    const std::vector<sycl::event>& deps, float weight_offset) {
    constexpr int WG=256;
    return q.submit([&](sycl::handler& hd){
        hd.depends_on(deps);
        sycl::local_accessor<float,1> partial(WG/SG_SIZE,hd);
        hd.parallel_for(sycl::nd_range<1>(size_t(tokens)*WG,WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg=it.get_sub_group();
                const int row=int(it.get_group(0));
                const int lid=int(it.get_local_id(0));
                const int lane=int(sg.get_local_id()[0]);
                const int sgid=int(sg.get_group_id()[0]);
                float* hr=h+int64_t(row)*hidden;
                const float* rr=residual?residual+int64_t(row)*hidden:nullptr;
                float* dst=out+int64_t(row)*hidden;
                float ss=0.0f;
                for(int i=lid;i<hidden;i+=WG){
                    const float v=float(sycl::half(hr[i]+(rr?rr[i]:0.0f)));
                    hr[i]=v;
                    ss=sycl::fma(v,v,ss);
                }
                ss=sycl::reduce_over_group(sg,ss,sycl::plus<float>());
                float* p=partial.template get_multi_ptr<
                    sycl::access::decorated::no>().get();
                if(lane==0)p[sgid]=ss;
                sycl::group_barrier(it.get_group());
                float total=0.0f;
                for(int i=0;i<WG/SG_SIZE;++i)total+=p[i];
                const float scale=sycl::rsqrt(total/float(hidden)+eps);
                for(int i=lid;i<hidden;i+=WG)
                    dst[i]=float(sycl::half(hr[i]*scale*
                        float(sycl::half(weight_offset+
                            bf16_to_f32(weight[i])))));
            });
    });
}

// ---------------------------------------------------------------------------
// FP16-storage activation variants.
//
// The f32 kernels above already round every intermediate through sycl::half
// before storing (dst = float(half(...))) to match Fusion's float16 dataflow,
// so the f32 buffers only ever hold fp16-representable values. Keeping the
// activations in fp16 storage is therefore numerically identical and halves
// the traffic, and removes the fp32<->fp16 conversion that otherwise wraps
// every oneDNN W4A16 GEMM (measured: 546 ms of a 2198 ms Muse prefill).
// ---------------------------------------------------------------------------
sycl::event launch_embed_f16_h(sycl::queue& q, const sycl::half* table,
    const int32_t* tokens, sycl::half* out, int count, int hidden,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(count)*hidden), [=](sycl::id<1> id) {
            const size_t i=id[0];
            const int row=int(i/hidden), col=int(i%hidden);
            out[i]=table[int64_t(tokens[row])*hidden+col];
        });
    });
}

sycl::event launch_add_f16_round_h(sycl::queue& q, sycl::half* dst,
    const sycl::half* src, int n, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(n)),[=](sycl::id<1> id){
            dst[id[0]]=sycl::half(float(dst[id[0]])+float(src[id[0]]));
        });
    });
}

sycl::event launch_gate_sigmoid_mul_h(sycl::queue& q, sycl::half* x,
    const sycl::half* gate, int tokens, int hidden,
    const std::vector<sycl::event>& deps) {
    constexpr size_t WG=256;
    const size_t padded=(size_t(hidden)+WG-1)/WG*WG;
    return q.submit([&](sycl::handler& h) { h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({size_t(tokens),padded},{1,WG}),
          [=](sycl::nd_item<2> it) {
            const int t=int(it.get_global_id(0)),d=int(it.get_global_id(1));
            if(d<hidden) {
                const int64_t i=int64_t(t)*hidden+d;
                const float xv=float(x[i]);
                const float gv=float(gate[i]);
                x[i]=sycl::half(xv/(1.0f+sycl::exp(-gv)));
            }
        });
    });
}

sycl::event launch_swiglu_h(sycl::queue& q, const sycl::half* gu,
    sycl::half* out, int tokens, int inter,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(tokens)*inter),[=](sycl::id<1> id){
            const int64_t z=id[0];
            const int row=int(z/inter),d=int(z%inter);
            const float gate=float(gu[int64_t(row)*2*inter+d]);
            const float up=float(gu[int64_t(row)*2*inter+inter+d]);
            out[z]=sycl::half((gate/(1.0f+sycl::exp(-gate)))*up);
        });
    });
}

sycl::event launch_dflash_store_tap_h(
    sycl::queue& q, const sycl::half* src, float* taps, int rows, int hidden,
    int tap_count, int start_pos, int tap,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(rows)*hidden), [=](sycl::id<1> id) {
            const int64_t z=id[0];
            const int row=int(z/hidden), col=int(z%hidden);
            taps[(int64_t(start_pos+row)*tap_count+tap)*hidden+col]=
                float(src[z]);
        });
    });
}

sycl::event launch_rmsnorm_residual_f16w_h(
    sycl::queue& q, sycl::half* hbuf, const sycl::half* residual,
    const sycl::half* weight, sycl::half* out, int tokens, int hidden,
    float eps, const std::vector<sycl::event>& deps, float weight_offset) {
    constexpr int WG=256;
    return q.submit([&](sycl::handler& hd){
        hd.depends_on(deps);
        sycl::local_accessor<float,1> partial(WG/SG_SIZE,hd);
        hd.parallel_for(sycl::nd_range<1>(size_t(tokens)*WG,WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg=it.get_sub_group();
                const int row=int(it.get_group(0));
                const int lid=int(it.get_local_id(0));
                const int lane=int(sg.get_local_id()[0]);
                const int sgid=int(sg.get_group_id()[0]);
                sycl::half* hr=hbuf+int64_t(row)*hidden;
                const sycl::half* rr=residual?residual+int64_t(row)*hidden:nullptr;
                sycl::half* dst=out+int64_t(row)*hidden;
                float ss=0.0f;
                for(int i=lid;i<hidden;i+=WG){
                    const float v=float(sycl::half(float(hr[i])+
                        (rr?float(rr[i]):0.0f)));
                    hr[i]=sycl::half(v);
                    ss=sycl::fma(v,v,ss);
                }
                ss=sycl::reduce_over_group(sg,ss,sycl::plus<float>());
                float* pp=partial.template get_multi_ptr<
                    sycl::access::decorated::no>().get();
                if(lane==0)pp[sgid]=ss;
                sycl::group_barrier(it.get_group());
                float total=0.0f;
                for(int i=0;i<WG/SG_SIZE;++i)total+=pp[i];
                const float scale=sycl::rsqrt(total/float(hidden)+eps);
                for(int i=lid;i<hidden;i+=WG)
                    dst[i]=sycl::half(float(hr[i])*scale*
                        (weight_offset+float(weight[i])));
            });
    });
}

sycl::event launch_rmsnorm_residual_f16w_batched(
    sycl::queue& q, float* h, const float* residual, const sycl::half* weight,
    float* out, int tokens, int hidden, float eps,
    const std::vector<sycl::event>& deps, float weight_offset) {
    constexpr int WG=256;
    return q.submit([&](sycl::handler& hd){
        hd.depends_on(deps);
        sycl::local_accessor<float,1> partial(WG/SG_SIZE,hd);
        hd.parallel_for(sycl::nd_range<1>(size_t(tokens)*WG,WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg=it.get_sub_group();
                const int row=int(it.get_group(0));
                const int lid=int(it.get_local_id(0));
                const int lane=int(sg.get_local_id()[0]);
                const int sgid=int(sg.get_group_id()[0]);
                float* hr=h+int64_t(row)*hidden;
                const float* rr=residual?residual+int64_t(row)*hidden:nullptr;
                float* dst=out+int64_t(row)*hidden;
                float ss=0.0f;
                for(int i=lid;i<hidden;i+=WG){
                    const float v=float(sycl::half(hr[i]+(rr?rr[i]:0.0f)));
                    hr[i]=v;
                    ss=sycl::fma(v,v,ss);
                }
                ss=sycl::reduce_over_group(sg,ss,sycl::plus<float>());
                float* p=partial.template get_multi_ptr<
                    sycl::access::decorated::no>().get();
                if(lane==0)p[sgid]=ss;
                sycl::group_barrier(it.get_group());
                float total=0.0f;
                for(int i=0;i<WG/SG_SIZE;++i)total+=p[i];
                const float scale=sycl::rsqrt(total/float(hidden)+eps);
                for(int i=lid;i<hidden;i+=WG)
                    dst[i]=float(sycl::half(hr[i]*scale*
                        (weight_offset+float(weight[i]))));
            });
    });
}

sycl::event launch_rmsnorm_residual_batched_quant(
    sycl::queue& q, float* h, const float* r0, const float* r1,
    const bf16_t* weight, float* out, sycl_bf16* out_bf,
    int8_t* out_q, float* out_scale, int tokens, int hidden, float eps,
    const std::vector<sycl::event>& deps) {
    constexpr int WG=256, NSG=WG/SG_SIZE;
    return q.submit([&](sycl::handler& hd) {
        hd.depends_on(deps);
        sycl::local_accessor<float,1> partial(2*NSG,hd);
        hd.parallel_for(sycl::nd_range<1>(size_t(tokens)*WG,WG),
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto sg=it.get_sub_group();int t=int(it.get_group(0));
            int lid=int(it.get_local_id(0)),lane=int(sg.get_local_id()[0]);
            int sgid=int(sg.get_group_id()[0]);
            float* ht=h+int64_t(t)*hidden;
            const float* a=r0?r0+int64_t(t)*hidden:nullptr;
            const float* b=r1?r1+int64_t(t)*hidden:nullptr;
            float ss=0.0f;
            for(int i=lid;i<hidden;i+=WG){float v=ht[i];if(a)v+=a[i];if(b)v+=b[i];
                if(a||b)ht[i]=v;ss=sycl::fma(v,v,ss);}
            ss=sycl::reduce_over_group(sg,ss,sycl::plus<float>());
            float* pt=partial.template get_multi_ptr<sycl::access::decorated::no>().get();
            if(lane==0)pt[sgid]=ss;sycl::group_barrier(it.get_group());
            float total=0.0f;for(int i=0;i<NSG;++i)total+=pt[i];
            const float rs=sycl::rsqrt(total/float(hidden)+eps);
            float mx=0.0f;
            for(int i=lid;i<hidden;i+=WG){float v=ht[i]*rs*(1.0f+bf16_to_f32(weight[i]));
                if(out)out[int64_t(t)*hidden+i]=v;if(out_bf)out_bf[int64_t(t)*hidden+i]=sycl_bf16(v);
                mx=sycl::fmax(mx,sycl::fabs(v));}
            mx=sycl::reduce_over_group(sg,mx,sycl::maximum<float>());
            if(lane==0)pt[NSG+sgid]=mx;sycl::group_barrier(it.get_group());
            float rowmax=0.0f;for(int i=0;i<NSG;++i)rowmax=sycl::fmax(rowmax,pt[NSG+i]);
            const float sc=rowmax>0.0f?rowmax/127.0f:1.0f,inv=1.0f/sc;
            if(lid==0)out_scale[t]=sc;
            for(int i=lid;i<hidden;i+=WG){float v=out_bf?float(out_bf[int64_t(t)*hidden+i]):
                    out[int64_t(t)*hidden+i];
                out_q[int64_t(t)*hidden+i]=int8_t(sycl::clamp(int(sycl::round(v*inv)),-127,127));}
          });
    });
}

sycl::event launch_rmsnorm_moe_residual_batched(sycl::queue& q,float* h,
    const sycl_bf16* routed,const int32_t* inverse,const float* route_weight,
    const float* shared,const bf16_t* weight,float* out,sycl_bf16* out_bf,
    int tokens,int top_k,int hidden,float eps,const std::vector<sycl::event>&deps){
    constexpr int WG=256,EPI=8;
    return q.submit([&](sycl::handler&hd){hd.depends_on(deps);
      sycl::local_accessor<float,1> partial(WG/SG_SIZE,hd);
      hd.parallel_for(sycl::nd_range<1>(size_t(tokens)*WG,WG),
       [=](sycl::nd_item<1>it)[[sycl::reqd_sub_group_size(SG_SIZE)]]{
        auto sg=it.get_sub_group();const int t=int(it.get_group(0));
        const int lid=int(it.get_local_id(0)),lane=int(sg.get_local_id()[0]);
        const int sgid=int(sg.get_group_id()[0]);int32_t route[8];float rw[8];
        #pragma unroll
        for(int s=0;s<8;++s){const int64_t r=int64_t(t)*top_k+s;
          route[s]=inverse[r];rw[s]=route_weight[r];}
        float ss=0.0f;float* ht=h+int64_t(t)*hidden;
        const float* sh=shared?shared+int64_t(t)*hidden:nullptr;
        for(int d=lid*EPI;d<hidden;d+=WG*EPI){
          float v[EPI];
          #pragma unroll
          for(int e=0;e<EPI;++e)v[e]=ht[d+e]+(sh?sh[d+e]:0.0f);
          #pragma unroll
          for(int s=0;s<8;++s){const sycl_bf16* row=routed+int64_t(route[s])*hidden+d;
            #pragma unroll
            for(int e=0;e<EPI;++e)v[e]=sycl::fma(rw[s],float(row[e]),v[e]);}
          #pragma unroll
          for(int e=0;e<EPI;++e){ht[d+e]=v[e];ss=sycl::fma(v[e],v[e],ss);}
        }
        ss=sycl::reduce_over_group(sg,ss,sycl::plus<float>());
        float* pt=partial.template get_multi_ptr<sycl::access::decorated::no>().get();
        if(lane==0)pt[sgid]=ss;sycl::group_barrier(it.get_group());
        float total=0.0f;for(int i=0;i<WG/SG_SIZE;++i)total+=pt[i];
        const float scale=sycl::rsqrt(total/float(hidden)+eps);
        float* ot=out?out+int64_t(t)*hidden:nullptr;
        sycl_bf16* obt=out_bf?out_bf+int64_t(t)*hidden:nullptr;
        for(int d=lid;d<hidden;d+=WG){const float v=ht[d]*scale*(1.0f+bf16_to_f32(weight[d]));
          if(ot)ot[d]=v;if(obt)obt[d]=sycl_bf16(v);}
       });});
}

// Depthwise causal convolution depends on earlier INPUTS, not earlier outputs.
// Therefore all prompt tokens are independent once the initial ring is fixed.
// The old implementation assigned one work-item per channel and serially
// walked thousands of tokens; that left nearly the entire B70 idle.
sycl::event launch_causal_conv1d_prefill(
    sycl::queue& q, const ConvParams& p, int tokens,
    const std::vector<sycl::event>& deps) {
    sycl::event compute = q.submit([&](sycl::handler& hd) {
        hd.depends_on(deps);
        const ConvParams pp = p;
        hd.parallel_for(sycl::range<2>(size_t(tokens),size_t(pp.channels)), [=](sycl::id<2> id) {
            const int t=int(id[0]),c=int(id[1]),K=pp.kernel;
            const bf16_t* w = pp.weight + int64_t(c) * K;
            const float* ring=pp.ring+int64_t(c)*(K-1);
            float acc=0.0f;
            for(int j=0;j<K-1;++j){
                const int src=t-(K-1-j);
                const float x=src>=0?pp.x[int64_t(src)*pp.channels+c]:ring[j+t];
                acc=sycl::fma(x,bf16_to_f32(w[j]),acc);
            }
            const float x=pp.x[int64_t(t)*pp.channels+c];
            acc=sycl::fma(x,bf16_to_f32(w[K-1]),acc);
            pp.out[int64_t(t)*pp.channels+c]=acc/(1.0f+sycl::exp(-acc));
        });
    });
    return q.submit([&](sycl::handler& hd){
        hd.depends_on(compute);const ConvParams pp=p;
        hd.parallel_for(sycl::range<1>(size_t(pp.channels)),[=](sycl::id<1> id){
            const int c=int(id[0]),K=pp.kernel;
            // For normal prompt chunks tokens >= K-1.  The short-chunk case
            // retains the still-needed tail of the initial history.
            for(int j=0;j<K-1;++j){
                if(tokens>=K-1)pp.ring[int64_t(c)*(K-1)+j]=pp.x[int64_t(tokens-(K-1)+j)*pp.channels+c];
                else if(j<K-1-tokens)
                    pp.ring[int64_t(c)*(K-1)+j]=pp.ring[int64_t(c)*(K-1)+j+tokens];
                else pp.ring[int64_t(c)*(K-1)+j]=pp.x[int64_t(j-(K-1-tokens))*pp.channels+c];
            }
        });
    });
}

sycl::event launch_causal_conv1d_split_prefill(
    sycl::queue& q,const ConvParams& p,int tokens,float* qv,float* kv,float* vv,
    sycl_bf16* vv_bf,
    int qk_size,int v_size,const std::vector<sycl::event>& deps){
    sycl::event compute=q.submit([&](sycl::handler&hd){hd.depends_on(deps);
      const ConvParams pp=p;
      hd.parallel_for(sycl::range<2>(size_t(tokens),size_t(pp.channels)),[=](sycl::id<2>id){
        const int t=int(id[0]),c=int(id[1]),K=pp.kernel;
        const bf16_t*w=pp.weight+int64_t(c)*K;
        const float*ring=pp.ring+int64_t(c)*(K-1);float acc=0.0f;
        for(int j=0;j<K-1;++j){const int src=t-(K-1-j);
          const float x=src>=0?pp.x[int64_t(src)*pp.channels+c]:ring[j+t];
          acc=sycl::fma(x,bf16_to_f32(w[j]),acc);}
        acc=sycl::fma(pp.x[int64_t(t)*pp.channels+c],bf16_to_f32(w[K-1]),acc);
        const float y=acc/(1.0f+sycl::exp(-acc));
        if(c<qk_size)qv[int64_t(t)*qk_size+c]=y;
        else if(c<2*qk_size)kv[int64_t(t)*qk_size+c-qk_size]=y;
        else if(vv_bf)vv_bf[int64_t(t)*v_size+c-2*qk_size]=sycl_bf16(y);
        else vv[int64_t(t)*v_size+c-2*qk_size]=y;
      });});
    return q.submit([&](sycl::handler&hd){hd.depends_on(compute);const ConvParams pp=p;
      hd.parallel_for(sycl::range<1>(size_t(pp.channels)),[=](sycl::id<1>id){
        const int c=int(id[0]),K=pp.kernel;
        for(int j=0;j<K-1;++j){
          if(tokens>=K-1)pp.ring[int64_t(c)*(K-1)+j]=pp.x[int64_t(tokens-(K-1)+j)*pp.channels+c];
          else if(j<K-1-tokens)pp.ring[int64_t(c)*(K-1)+j]=pp.ring[int64_t(c)*(K-1)+j+tokens];
          else pp.ring[int64_t(c)*(K-1)+j]=pp.x[int64_t(j-(K-1-tokens))*pp.channels+c];
        }});});
}

sycl::event launch_causal_conv1d_split_bf16_prefill(sycl::queue& q,
    const sycl_bf16* x,const bf16_t* weight,float* ring,int channels,int kernel,
    int tokens,sycl_bf16* qv,sycl_bf16* kv,sycl_bf16* vv,int qk_size,int v_size,
    const std::vector<sycl::event>& deps){
    auto compute=q.submit([&](sycl::handler&h){h.depends_on(deps);
      h.parallel_for(sycl::range<2>(size_t(tokens),size_t(channels)),[=](sycl::id<2>id){
        const int t=int(id[0]),c=int(id[1]);const bf16_t*w=weight+int64_t(c)*kernel;
        const float*r=ring+int64_t(c)*(kernel-1);float acc=0.0f;
        for(int j=0;j<kernel-1;++j){const int src=t-(kernel-1-j);
          const float xv=src>=0?float(x[int64_t(src)*channels+c]):r[j+t];
          acc=sycl::fma(xv,bf16_to_f32(w[j]),acc);}
        acc=sycl::fma(float(x[int64_t(t)*channels+c]),bf16_to_f32(w[kernel-1]),acc);
        const sycl_bf16 y=sycl_bf16(acc/(1.0f+sycl::exp(-acc)));
        if(c<qk_size)qv[int64_t(t)*qk_size+c]=y;
        else if(c<2*qk_size)kv[int64_t(t)*qk_size+c-qk_size]=y;
        else vv[int64_t(t)*v_size+c-2*qk_size]=y;
      });});
    return q.submit([&](sycl::handler&h){h.depends_on(compute);
      h.parallel_for(sycl::range<1>(size_t(channels)),[=](sycl::id<1>id){
        const int c=int(id[0]);
        for(int j=0;j<kernel-1;++j){
          if(tokens>=kernel-1)ring[int64_t(c)*(kernel-1)+j]=
            float(x[int64_t(tokens-(kernel-1)+j)*channels+c]);
          else if(j<kernel-1-tokens)ring[int64_t(c)*(kernel-1)+j]=
            ring[int64_t(c)*(kernel-1)+j+tokens];
          else ring[int64_t(c)*(kernel-1)+j]=
            float(x[int64_t(j-(kernel-1-tokens))*channels+c]);
        }});});
}

sycl::event launch_qk_norm_rope_batched(
    sycl::queue& q, float* qv, float* kv, const bf16_t* qw, const bf16_t* kw,
    int tokens, int q_heads, int k_heads, int dim, int start_pos,
    float theta, float partial_factor, float eps,
    const std::vector<sycl::event>& deps, float weight_offset) {
    const int rot = int(dim * partial_factor) & ~1;
    const int heads_per_token = q_heads + k_heads;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(tokens) * heads_per_token * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int gh = int(it.get_group(0));
                const int t = gh / heads_per_token;
                const int h0 = gh % heads_per_token;
                const bool isq = h0 < q_heads;
                const int hi = isq ? h0 : h0 - q_heads;
                float* p = isq ? qv + (int64_t(t) * q_heads + hi) * dim
                               : kv + (int64_t(t) * k_heads + hi) * dim;
                const bf16_t* w = isq ? qw : kw;
                float ss = 0.0f;
                for (int d = lane; d < dim; d += SG_SIZE)
                    ss = sycl::fma(p[d], p[d], ss);
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                const float scale = sycl::rsqrt(ss / float(dim) + eps);
                for (int d = lane; d < dim; d += SG_SIZE)
                    p[d] *= scale * (weight_offset + bf16_to_f32(w[d]));
                sycl::group_barrier(sg);
                const int pos = start_pos + t;
                for (int j = lane; j < rot / 2; j += SG_SIZE) {
                    const float inv = sycl::exp(-float(2 * j) / float(rot) * sycl::log(theta));
                    const float ang = float(pos) * inv;
                    const float cs = sycl::cos(ang), sn = sycl::sin(ang);
                    const float a = p[j], b = p[j + rot / 2];
                    p[j] = a * cs - b * sn;
                    p[j + rot / 2] = a * sn + b * cs;
                }
            });
    });
}

sycl::event launch_kv_append_batched(
    sycl::queue& q, const float* k, const float* v, uint8_t* k_cache,
    uint8_t* v_cache, int tokens, int start_pos, int n_kv_heads,
    int head_dim, int seq_cap, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(tokens) * n_kv_heads * head_dim),
            [=](sycl::id<1> id) {
                const int64_t z = int64_t(id[0]);
                const int d = int(z % head_dim);
                const int64_t th = z / head_dim;
                const int kh = int(th % n_kv_heads), t = int(th / n_kv_heads);
                const int pos = start_pos + t;
                if (pos >= seq_cap) return;
                const int64_t src = (int64_t(t) * n_kv_heads + kh) * head_dim + d;
                k_cache[(int64_t(kh) * head_dim + d) * seq_cap + pos] =
                    f32_to_e4m3(k[src]);
                v_cache[(int64_t(kh) * seq_cap + pos) * head_dim + d] =
                    f32_to_e4m3(v[src]);
            });
    });
}

sycl::event launch_f32_to_f16(sycl::queue& q, const float* src,
    sycl::half* dst, size_t count, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(count),
            [=](sycl::id<1> id) { dst[id[0]] = sycl::half(src[id[0]]); });
    });
}

sycl::event launch_f16_to_f32(sycl::queue& q, const sycl::half* src,
    float* dst, size_t count, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(count),
            [=](sycl::id<1> id) { dst[id[0]] = float(src[id[0]]); });
    });
}

sycl::event launch_qk_norm_rope_f16_batched(
    sycl::queue& q, const float* q_src, const float* k_src, sycl::half* q_dst,
    sycl::half* k_dst, const bf16_t* q_weight, const bf16_t* k_weight,
    int tokens, int q_heads, int k_heads, int head_dim, int start_pos,
    float theta, float eps, const std::vector<sycl::event>& deps,
    bool use_rope, float query_scale, float weight_offset) {
    const int heads_per_token = q_heads + k_heads;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(tokens) * heads_per_token * SG_SIZE,
                              SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int gh = int(it.get_group(0));
                const int row = gh / heads_per_token;
                const int packed_head = gh % heads_per_token;
                const bool is_q = packed_head < q_heads;
                const int head = is_q ? packed_head : packed_head - q_heads;
                const float* src = is_q
                    ? q_src + (int64_t(row) * q_heads + head) * head_dim
                    : k_src + (int64_t(row) * k_heads + head) * head_dim;
                sycl::half* dst = is_q
                    ? q_dst + (int64_t(row) * q_heads + head) * head_dim
                    : k_dst + (int64_t(row) * k_heads + head) * head_dim;
                const bf16_t* weight = is_q ? q_weight : k_weight;

                // vLLM's W4A16/F.linear result is FP16 before native RMSNorm.
                // Round the projected value first, then accumulate variance in
                // FP32 exactly as the native RMSNorm implementation does.
                float sum_sq = 0.0f;
                for (int d = lane; d < head_dim; d += SG_SIZE) {
                    const float x = float(sycl::half(src[d]));
                    sum_sq = sycl::fma(x, x, sum_sq);
                }
                sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
                const float scale = sycl::rsqrt(sum_sq / float(head_dim) + eps);
                for (int d = lane; d < head_dim; d += SG_SIZE) {
                    const float x = float(sycl::half(src[d]));
                    const float w=weight_offset+bf16_to_f32(weight[d]);
                    dst[d] = sycl::half(x * scale * w *
                        (is_q?query_scale:1.0f));
                }
                sycl::group_barrier(sg);

                // Muse uses NEOX half-split RoPE over the complete 128-wide
                // head. RMSNorm output is FP16 before RoPE, and RoPE writes
                // FP16 again.
                if(!use_rope)return;
                const int pos = start_pos + row;
                for (int j = lane; j < head_dim / 2; j += SG_SIZE) {
                    const float a = float(dst[j]);
                    const float b = float(dst[j + head_dim / 2]);
                    const float inv = sycl::exp(-float(2 * j) / float(head_dim)
                                                * sycl::log(theta));
                    const float angle = float(pos) * inv;
                    const float cs = sycl::cos(angle), sn = sycl::sin(angle);
                    dst[j] = sycl::half(a * cs - b * sn);
                    dst[j + head_dim / 2] = sycl::half(a * sn + b * cs);
                }
            });
    });
}

sycl::event launch_qkv_norm_rope_f16_fused(
    sycl::queue& q, const sycl::half* qkv_src, sycl::half* q_dst,
    sycl::half* k_dst, sycl::half* v_dst, const bf16_t* q_weight,
    const bf16_t* k_weight, int tokens, int q_heads, int k_heads,
    int head_dim, int start_pos, float theta, float eps,
    const std::vector<sycl::event>& deps, bool use_rope,
    float query_scale, float weight_offset) {
    const int q_width=q_heads*head_dim;
    const int k_width=k_heads*head_dim;
    const int row_width=q_width+2*k_width;
    const int heads_per_token=q_heads+k_heads;
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(tokens)*heads_per_token*SG_SIZE,SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg=it.get_sub_group();
                const int lane=int(sg.get_local_id()[0]);
                const int gh=int(it.get_group(0));
                const int row=gh/heads_per_token;
                const int packed_head=gh%heads_per_token;
                const bool is_q=packed_head<q_heads;
                const int head=is_q?packed_head:packed_head-q_heads;
                const int64_t src_base=int64_t(row)*row_width+
                    (is_q?int64_t(head)*head_dim:
                     int64_t(q_width)+int64_t(head)*head_dim);
                const int64_t dst_base=(int64_t(row)*(is_q?q_heads:k_heads)+head)
                    *head_dim;
                sycl::half* dst=is_q?q_dst+dst_base:k_dst+dst_base;
                const bf16_t* weight=is_q?q_weight:k_weight;
                float sum_sq=0.0f;
                for(int d=lane;d<head_dim;d+=SG_SIZE){
                    const float x=float(qkv_src[src_base+d]);
                    sum_sq=sycl::fma(x,x,sum_sq);
                }
                sum_sq=sycl::reduce_over_group(sg,sum_sq,sycl::plus<float>());
                const float scale=sycl::rsqrt(sum_sq/float(head_dim)+eps);
                for(int d=lane;d<head_dim;d+=SG_SIZE)
                    dst[d]=sycl::half(float(qkv_src[src_base+d])*scale*
                        (weight_offset+bf16_to_f32(weight[d]))*
                        (is_q?query_scale:1.0f));
                sycl::group_barrier(sg);
                if(!use_rope){
                    if(!is_q){
                        const int64_t v_src=int64_t(row)*row_width+q_width+
                            k_width+int64_t(head)*head_dim;
                        const int64_t v_dst_base=(int64_t(row)*k_heads+head)*
                            head_dim;
                        for(int d=lane;d<head_dim;d+=SG_SIZE)
                            v_dst[v_dst_base+d]=qkv_src[v_src+d];
                    }
                    return;
                }
                const int pos=start_pos+row;
                for(int j=lane;j<head_dim/2;j+=SG_SIZE){
                    const float a=float(dst[j]);
                    const float b=float(dst[j+head_dim/2]);
                    const float inv=sycl::exp(-float(2*j)/float(head_dim)*
                                              sycl::log(theta));
                    const float angle=float(pos)*inv;
                    const float cs=sycl::cos(angle),sn=sycl::sin(angle);
                    dst[j]=sycl::half(a*cs-b*sn);
                    dst[j+head_dim/2]=sycl::half(a*sn+b*cs);
                }
                if(!is_q){
                    const int64_t v_src=int64_t(row)*row_width+q_width+k_width+
                        int64_t(head)*head_dim;
                    const int64_t v_dst_base=(int64_t(row)*k_heads+head)*head_dim;
                    for(int d=lane;d<head_dim;d+=SG_SIZE)
                        v_dst[v_dst_base+d]=qkv_src[v_src+d];
                }
            });
    });
}

sycl::event launch_qkv_norm_rope_f16w_fused(
    sycl::queue& q, const sycl::half* qkv_src, sycl::half* q_dst,
    sycl::half* k_dst, sycl::half* v_dst, const sycl::half* q_weight,
    const sycl::half* k_weight, int tokens, int q_heads, int k_heads,
    int head_dim, int start_pos, float theta, float eps,
    const std::vector<sycl::event>& deps, bool use_rope,
    float query_scale, float weight_offset) {
    const int q_width=q_heads*head_dim;
    const int k_width=k_heads*head_dim;
    const int row_width=q_width+2*k_width;
    const int heads_per_token=q_heads+k_heads;
    return q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(tokens)*heads_per_token*SG_SIZE,SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg=it.get_sub_group();
                const int lane=int(sg.get_local_id()[0]);
                const int gh=int(it.get_group(0));
                const int row=gh/heads_per_token;
                const int packed_head=gh%heads_per_token;
                const bool is_q=packed_head<q_heads;
                const int head=is_q?packed_head:packed_head-q_heads;
                const int64_t src_base=int64_t(row)*row_width+
                    (is_q?int64_t(head)*head_dim:
                     int64_t(q_width)+int64_t(head)*head_dim);
                const int64_t dst_base=(int64_t(row)*(is_q?q_heads:k_heads)+head)
                    *head_dim;
                sycl::half* dst=is_q?q_dst+dst_base:k_dst+dst_base;
                const sycl::half* weight=is_q?q_weight:k_weight;
                float sum_sq=0.0f;
                for(int d=lane;d<head_dim;d+=SG_SIZE){
                    const float x=float(qkv_src[src_base+d]);
                    sum_sq=sycl::fma(x,x,sum_sq);
                }
                sum_sq=sycl::reduce_over_group(sg,sum_sq,sycl::plus<float>());
                const float scale=sycl::rsqrt(sum_sq/float(head_dim)+eps);
                for(int d=lane;d<head_dim;d+=SG_SIZE)
                    dst[d]=sycl::half(float(qkv_src[src_base+d])*scale*
                        (weight_offset+float(weight[d]))*
                        (is_q?query_scale:1.0f));
                sycl::group_barrier(sg);
                if(!use_rope){
                    if(!is_q){
                        const int64_t v_src=int64_t(row)*row_width+q_width+
                            k_width+int64_t(head)*head_dim;
                        const int64_t v_dst_base=(int64_t(row)*k_heads+head)*
                            head_dim;
                        for(int d=lane;d<head_dim;d+=SG_SIZE)
                            v_dst[v_dst_base+d]=qkv_src[v_src+d];
                    }
                    return;
                }
                const int pos=start_pos+row;
                for(int j=lane;j<head_dim/2;j+=SG_SIZE){
                    const float a=float(dst[j]);
                    const float b=float(dst[j+head_dim/2]);
                    const float inv=sycl::exp(-float(2*j)/float(head_dim)*
                                              sycl::log(theta));
                    const float angle=float(pos)*inv;
                    const float cs=sycl::cos(angle),sn=sycl::sin(angle);
                    dst[j]=sycl::half(a*cs-b*sn);
                    dst[j+head_dim/2]=sycl::half(a*sn+b*cs);
                }
                if(!is_q){
                    const int64_t v_src=int64_t(row)*row_width+q_width+k_width+
                        int64_t(head)*head_dim;
                    const int64_t v_dst_base=(int64_t(row)*k_heads+head)*head_dim;
                    for(int d=lane;d<head_dim;d+=SG_SIZE)
                        v_dst[v_dst_base+d]=qkv_src[v_src+d];
                }
            });
    });
}

sycl::event launch_kv_append_f16_paged(
    sycl::queue& q, const sycl::half* k, const sycl::half* v,
    sycl::half* k_cache, sycl::half* v_cache, int tokens, int start_pos,
    int n_kv_heads, int head_dim, int block_size,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(tokens) * n_kv_heads * head_dim),
            [=](sycl::id<1> id) {
                const int64_t z = int64_t(id[0]);
                const int d = int(z % head_dim);
                const int64_t th = z / head_dim;
                const int kh = int(th % n_kv_heads);
                const int row = int(th / n_kv_heads);
                const int pos = start_pos + row;
                const int block = pos / block_size;
                const int offset = pos % block_size;
                const int64_t src = (int64_t(row) * n_kv_heads + kh) * head_dim + d;
                const int64_t dst = ((int64_t(block) * block_size + offset)
                                     * n_kv_heads + kh) * head_dim + d;
                k_cache[dst] = k[src];
                v_cache[dst] = v[src];
            });
    });
}

sycl::event launch_dflash_context_kv_f16(
    sycl::queue& q, const float* fused_kv, sycl::half* all_k,
    sycl::half* all_v, const bf16_t* stacked_k_norm, int layers, int tokens,
    int kv_heads, int head_dim, int start_pos, float theta, float eps,
    const std::vector<sycl::event>& deps) {
    const int kv_width=kv_heads*head_dim;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(layers)*tokens*kv_heads*SG_SIZE,SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg=it.get_sub_group();
                const int lane=int(sg.get_local_id()[0]);
                const int group=int(it.get_group(0));
                const int kh=group%kv_heads;
                const int tr=group/kv_heads;
                const int row=tr%tokens;
                const int layer=tr/tokens;
                const int64_t fused_base=(int64_t(row)*layers*2+
                    int64_t(layer)*2)*kv_width+int64_t(kh)*head_dim;
                const int64_t out_base=(int64_t(layer)*tokens+row)*kv_width+
                    int64_t(kh)*head_dim;
                float sum_sq=0.0f;
                for(int d=lane;d<head_dim;d+=SG_SIZE){
                    const float kval=float(sycl::half(fused_kv[fused_base+d]));
                    sum_sq=sycl::fma(kval,kval,sum_sq);
                    all_v[out_base+d]=sycl::half(
                        fused_kv[fused_base+kv_width+d]);
                }
                sum_sq=sycl::reduce_over_group(sg,sum_sq,sycl::plus<float>());
                const float scale=sycl::rsqrt(sum_sq/float(head_dim)+eps);
                const bf16_t* weight=stacked_k_norm+int64_t(layer)*head_dim;
                for(int d=lane;d<head_dim;d+=SG_SIZE){
                    const float kval=float(sycl::half(fused_kv[fused_base+d]));
                    all_k[out_base+d]=sycl::half(
                        kval*scale*bf16_to_f32(weight[d]));
                }
                sycl::group_barrier(sg);
                const int pos=start_pos+row;
                for(int j=lane;j<head_dim/2;j+=SG_SIZE){
                    const float a=float(all_k[out_base+j]);
                    const float b=float(all_k[out_base+j+head_dim/2]);
                    const float inv=sycl::exp(-float(2*j)/float(head_dim)*
                                              sycl::log(theta));
                    const float angle=float(pos)*inv;
                    const float cs=sycl::cos(angle),sn=sycl::sin(angle);
                    all_k[out_base+j]=sycl::half(a*cs-b*sn);
                    all_k[out_base+j+head_dim/2]=sycl::half(a*sn+b*cs);
                }
            });
    });
}

sycl::event launch_dflash_context_kv_f16w(
    sycl::queue& q, const float* fused_kv, sycl::half* all_k,
    sycl::half* all_v, const sycl::half* stacked_k_norm, int layers, int tokens,
    int kv_heads, int head_dim, int start_pos, float theta, float eps,
    const std::vector<sycl::event>& deps) {
    const int kv_width=kv_heads*head_dim;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(
            sycl::nd_range<1>(size_t(layers)*tokens*kv_heads*SG_SIZE,SG_SIZE),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg=it.get_sub_group();
                const int lane=int(sg.get_local_id()[0]);
                const int group=int(it.get_group(0));
                const int kh=group%kv_heads;
                const int tr=group/kv_heads;
                const int row=tr%tokens;
                const int layer=tr/tokens;
                const int64_t fused_base=(int64_t(row)*layers*2+
                    int64_t(layer)*2)*kv_width+int64_t(kh)*head_dim;
                const int64_t out_base=(int64_t(layer)*tokens+row)*kv_width+
                    int64_t(kh)*head_dim;
                float sum_sq=0.0f;
                for(int d=lane;d<head_dim;d+=SG_SIZE){
                    const float kval=float(sycl::half(fused_kv[fused_base+d]));
                    sum_sq=sycl::fma(kval,kval,sum_sq);
                    all_v[out_base+d]=sycl::half(
                        fused_kv[fused_base+kv_width+d]);
                }
                sum_sq=sycl::reduce_over_group(sg,sum_sq,sycl::plus<float>());
                const float scale=sycl::rsqrt(sum_sq/float(head_dim)+eps);
                const sycl::half* weight=stacked_k_norm+int64_t(layer)*head_dim;
                for(int d=lane;d<head_dim;d+=SG_SIZE){
                    const float kval=float(sycl::half(fused_kv[fused_base+d]));
                    all_k[out_base+d]=sycl::half(kval*scale*float(weight[d]));
                }
                sycl::group_barrier(sg);
                const int pos=start_pos+row;
                for(int j=lane;j<head_dim/2;j+=SG_SIZE){
                    const float a=float(all_k[out_base+j]);
                    const float b=float(all_k[out_base+j+head_dim/2]);
                    const float inv=sycl::exp(-float(2*j)/float(head_dim)*
                                              sycl::log(theta));
                    const float angle=float(pos)*inv;
                    const float cs=sycl::cos(angle),sn=sycl::sin(angle);
                    all_k[out_base+j]=sycl::half(a*cs-b*sn);
                    all_k[out_base+j+head_dim/2]=sycl::half(a*sn+b*cs);
                }
            });
    });
}

// Causal chunk-prefill attention. One work-group owns a 16-query tile for one
// head. Its 16 subgroups each own one query, while all of them reuse the same
// 16-key K/V tile staged in SLM. The former one-subgroup-per-query kernel read
// the complete K/V history independently for every query.
sycl::event launch_flash_prefill(
    sycl::queue& q, const float* qv, const uint8_t* k_cache,
    const uint8_t* v_cache, float* out, int tokens, int start_pos,
    int num_heads, int num_kv_heads, int head_dim, int seq_cap,
    float softmax_scale, const std::vector<sycl::event>& deps) {
    constexpr int QT=16, KT=16, WG=QT*SG_SIZE;
    const int qtiles=(tokens+QT-1)/QT;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> ks(size_t(head_dim)*KT,h);
        sycl::local_accessor<float,1> vs(size_t(KT)*head_dim,h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(qtiles)*num_heads*WG,WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_id()[0]);
                const int qslot = int(sg.get_group_id()[0]);
                const int lid=int(it.get_local_id(0));
                const int g = int(it.get_group(0));
                const int qt=g/num_heads, qh0=g%num_heads;
                const int t=qt*QT+qslot;
                const int kvh = qh0 / (num_heads / num_kv_heads);
                const uint8_t* kh = k_cache + int64_t(kvh) * head_dim * seq_cap;
                const uint8_t* vh = v_cache + int64_t(kvh) * seq_cap * head_dim;
                float* ksl=ks.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* vsl=vs.template get_multi_ptr<sycl::access::decorated::no>().get();
                constexpr int MAX_DPL = 16;
                const int dpl = head_dim / SG_SIZE;
                float acc[MAX_DPL];
                #pragma unroll
                for (int j = 0; j < MAX_DPL; ++j) acc[j] = 0.0f;
                float m = -std::numeric_limits<float>::infinity(), l = 0.0f;
                const int end=t<tokens?start_pos+t+1:0;
                const int tile_last=sycl::min(tokens-1,qt*QT+QT-1);
                const int max_end=start_pos+tile_last+1;
                for (int s0 = 0; s0 < max_end; s0 += KT) {
                    for(int x=lid;x<head_dim*KT;x+=WG){
                        const int d=x/KT,k=x%KT,s=s0+k;
                        ksl[x]=s<max_end?e4m3_to_f32(kh[int64_t(d)*seq_cap+s]):0.0f;
                        vsl[int64_t(k)*head_dim+d]=s<max_end?
                            e4m3_to_f32(vh[int64_t(s)*head_dim+d]):0.0f;
                    }
                    sycl::group_barrier(it.get_group());
                    const int s = s0 + lane;
                    float score = -std::numeric_limits<float>::infinity();
                    if (t<tokens && s < end) {
                        const float* qr=qv+(int64_t(t)*num_heads+qh0)*head_dim;
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                            dot = sycl::fma(qr[d],ksl[int64_t(d)*KT+lane],dot);
                        score = dot * softmax_scale;
                    }
                    const float mb = sycl::reduce_over_group(sg, score, sycl::maximum<float>());
                    const float mn = sycl::fmax(m, mb);
                    const float corr = sycl::isinf(m) ? 0.0f : sycl::exp(m - mn);
                    const float p = sycl::isinf(score) ? 0.0f : sycl::exp(score - mn);
                    l = sycl::fma(l, corr, sycl::reduce_over_group(sg, p, sycl::plus<float>()));
                    for (int j = 0; j < MAX_DPL; ++j) if (j < dpl) acc[j] *= corr;
                    for (int j = 0; j < KT && s0 + j < end; ++j) {
                        const float pb = sycl::group_broadcast(sg, p, j);
                        for (int d = 0; d < MAX_DPL; ++d) if (d < dpl)
                            acc[d]=sycl::fma(pb,vsl[int64_t(j)*head_dim+lane+d*SG_SIZE],acc[d]);
                    }
                    m = mn;
                    sycl::group_barrier(it.get_group());
                }
                if(t<tokens){
                    float* o=out+(int64_t(t)*num_heads+qh0)*head_dim;
                    const float inv=l>0.0f?1.0f/l:0.0f;
                    for(int d=0;d<MAX_DPL;++d)if(d<dpl)o[lane+d*SG_SIZE]=acc[d]*inv;
                }
            });
    });
}

// DFlash block attention. Full-attention heads are normally non-causal; Muse's
// sliding-attention head is trained causally. K/V for context and query rows
// use the target's FP8 E4M3 cache layout.
sycl::event launch_dflash2_block_attention(
    sycl::queue& q, const float* qv, const uint8_t* k_cache,
    const uint8_t* v_cache, float* out, int tokens, int context_len,
    int num_heads, int num_kv_heads, int head_dim, int seq_cap,
    int sliding_window, bool causal, float softmax_scale,
    const std::vector<sycl::event>& deps) {
    constexpr int QT=16, KT=16, WG=QT*SG_SIZE;
    const int qtiles=(tokens+QT-1)/QT;
    const int end=context_len+tokens;
    const int begin=(causal&&sliding_window>0)
        ? std::max(0,context_len+1-sliding_window)
        : (sliding_window>0 ? std::max(0,end-sliding_window) : 0);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float,1> ks(size_t(head_dim)*KT,h);
        sycl::local_accessor<float,1> vs(size_t(KT)*head_dim,h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(qtiles)*num_heads*WG,WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg=it.get_sub_group();
                const int lane=int(sg.get_local_id()[0]);
                const int qslot=int(sg.get_group_id()[0]);
                const int lid=int(it.get_local_id(0));
                const int g=int(it.get_group(0));
                const int qt=g/num_heads,qh0=g%num_heads;
                const int t=qt*QT+qslot;
                const int kvh=qh0/(num_heads/num_kv_heads);
                const uint8_t* kh=k_cache+int64_t(kvh)*head_dim*seq_cap;
                const uint8_t* vh=v_cache+int64_t(kvh)*seq_cap*head_dim;
                float* ksl=ks.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* vsl=vs.template get_multi_ptr<sycl::access::decorated::no>().get();
                constexpr int MAX_DPL=16;
                const int dpl=head_dim/SG_SIZE;
                float acc[MAX_DPL];
                #pragma unroll
                for(int j=0;j<MAX_DPL;++j)acc[j]=0.0f;
                float m=-std::numeric_limits<float>::infinity(),l=0.0f;
                for(int s0=begin;s0<end;s0+=KT){
                    for(int x=lid;x<head_dim*KT;x+=WG){
                        const int d=x/KT,k=x%KT,s=s0+k;
                        ksl[x]=s<end?e4m3_to_f32(kh[int64_t(d)*seq_cap+s]):0.0f;
                        vsl[int64_t(k)*head_dim+d]=s<end?
                            e4m3_to_f32(vh[int64_t(s)*head_dim+d]):0.0f;
                    }
                    sycl::group_barrier(it.get_group());
                    const int s=s0+lane;
                    float score=-std::numeric_limits<float>::infinity();
                    const int query_end=causal ? context_len+t+1 : end;
                    const int query_begin=sliding_window>0
                        ? sycl::max(0,query_end-sliding_window) : 0;
                    if(t<tokens&&s>=query_begin&&s<query_end){
                        const float* qr=qv+(int64_t(t)*num_heads+qh0)*head_dim;
                        float dot=0.0f;
                        for(int d=0;d<head_dim;++d)
                            dot=sycl::fma(qr[d],ksl[int64_t(d)*KT+lane],dot);
                        score=dot*softmax_scale;
                    }
                    const float mb=sycl::reduce_over_group(sg,score,sycl::maximum<float>());
                    const float mn=sycl::fmax(m,mb);
                    const float corr=sycl::isinf(m)?0.0f:sycl::exp(m-mn);
                    const float p=sycl::isinf(score)?0.0f:sycl::exp(score-mn);
                    l=sycl::fma(l,corr,sycl::reduce_over_group(sg,p,sycl::plus<float>()));
                    for(int j=0;j<MAX_DPL;++j)if(j<dpl)acc[j]*=corr;
                    for(int j=0;j<KT&&s0+j<end;++j){
                        const float pb=sycl::group_broadcast(sg,p,j);
                        for(int d=0;d<MAX_DPL;++d)if(d<dpl)
                            acc[d]=sycl::fma(pb,vsl[int64_t(j)*head_dim+lane+d*SG_SIZE],acc[d]);
                    }
                    m=mn;
                    sycl::group_barrier(it.get_group());
                }
                if(t<tokens){
                    float* o=out+(int64_t(t)*num_heads+qh0)*head_dim;
                    const float inv=l>0.0f?1.0f/l:0.0f;
                    for(int d=0;d<MAX_DPL;++d)if(d<dpl)
                        o[lane+d*SG_SIZE]=acc[d]*inv;
                }
            });
    });
}

sycl::event launch_embed_batched(sycl::queue& q, const bf16_t* table,
    const int32_t* tokens, float* out, int count, int hidden,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) { h.depends_on(deps);
        // A range dimension is also the implicit work-group dimension on some
        // Level Zero paths. hidden=2048 exceeds B70's per-dimension limit of
        // 1024, so flatten the independent token/hidden coordinates.
        h.parallel_for(sycl::range<1>(size_t(count) * hidden), [=](sycl::id<1> id) {
            const int t = int(id[0] / hidden), d = int(id[0] % hidden);
            out[int64_t(t) * hidden + d] = bf16_to_f32(table[int64_t(tokens[t]) * hidden + d]);
        });
    });
}

sycl::event launch_embed_f16_batched(sycl::queue& q, const sycl::half* table,
    const int32_t* tokens, float* out, int count, int hidden,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(count)*hidden), [=](sycl::id<1> id) {
            const size_t i=id[0];
            const int row=int(i/hidden), col=int(i%hidden);
            out[i]=float(table[int64_t(tokens[row])*hidden+col]);
        });
    });
}

sycl::event launch_split_deltanet_qkv_batched(sycl::queue& q, const float* src,
    float* qv, float* kv, float* vv, int tokens, int qk_size, int v_size,
    const std::vector<sycl::event>& deps) {
    const int stride = 2 * qk_size + v_size;
    return q.submit([&](sycl::handler& h) { h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(tokens) * stride), [=](sycl::id<1> id) {
            const int64_t z = int64_t(id[0]); const int t = int(z / stride), d = int(z % stride);
            if (d < qk_size) qv[int64_t(t) * qk_size + d] = src[z];
            else if (d < 2 * qk_size) kv[int64_t(t) * qk_size + d - qk_size] = src[z];
            else vv[int64_t(t) * v_size + d - 2 * qk_size] = src[z];
        });
    });
}

sycl::event launch_split_dn_fused_projections(sycl::queue& q,const float* src,
    float* qkv,float* z,float* ab,int tokens,const std::vector<sycl::event>&deps){
 constexpr int Q=8192,Z=4096,A=64,S=Q+Z+A;
 return q.submit([&](sycl::handler&h){h.depends_on(deps);
  h.parallel_for(sycl::range<1>(size_t(tokens)*S),[=](sycl::id<1>id){
   int64_t x=id[0];int t=int(x/S),d=int(x%S);float v=src[x];
   if(d<Q)qkv[int64_t(t)*Q+d]=v;
   else if(d<Q+Z)z[int64_t(t)*Z+d-Q]=v;
   else ab[int64_t(t)*A+d-Q-Z]=v;
  });});
}

sycl::event launch_split_qgate_batched(sycl::queue& q, const float* src,
    float* qout, float* gout, int tokens, int heads, int dim,
    const std::vector<sycl::event>& deps) {
    const int half = heads * dim, stride = 2 * half;
    return q.submit([&](sycl::handler& h) { h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(tokens) * half), [=](sycl::id<1> id) {
            const int64_t z = int64_t(id[0]); const int t = int(z / half), hd = int(z % half);
            const int head = hd / dim, d = hd % dim;
            const int64_t base = int64_t(t) * stride + int64_t(head) * 2 * dim;
            qout[z] = src[base + d];
            gout[z] = src[base + dim + d];
        });
    });
}

sycl::event launch_split_qgate_bf16(sycl::queue&q,const sycl_bf16*src,
    sycl_bf16*qout,float*gout,int tokens,int heads,int dim,
    const std::vector<sycl::event>&deps){
 const int half=heads*dim,stride=2*half;
 return q.submit([&](sycl::handler&h){h.depends_on(deps);
  h.parallel_for(sycl::range<1>(size_t(tokens)*half),[=](sycl::id<1>id){
   int64_t z=id[0];int t=int(z/half),hd=int(z%half),head=hd/dim,d=hd%dim;
   int64_t base=int64_t(t)*stride+int64_t(head)*2*dim;
   qout[z]=src[base+d];gout[z]=float(src[base+dim+d]);});});
}

sycl::event launch_qk_norm_rope_bf16_batched(sycl::queue&q,sycl_bf16*qv,
    sycl_bf16*kv,const bf16_t*qw,const bf16_t*kw,int tokens,int q_heads,
    int k_heads,int dim,int start_pos,float theta,float partial_factor,float eps,
    const std::vector<sycl::event>&deps){
 const int rot=int(dim*partial_factor)&~1,hpt=q_heads+k_heads;
 return q.submit([&](sycl::handler&h){h.depends_on(deps);
  h.parallel_for(sycl::nd_range<1>(size_t(tokens)*hpt*SG_SIZE,SG_SIZE),
   [=](sycl::nd_item<1>it)[[sycl::reqd_sub_group_size(SG_SIZE)]]{
    auto sg=it.get_sub_group();int lane=int(sg.get_local_id()[0]);
    int gh=int(it.get_group(0)),t=gh/hpt,h0=gh%hpt;bool isq=h0<q_heads;
    int hi=isq?h0:h0-q_heads;sycl_bf16*p=isq?qv+(int64_t(t)*q_heads+hi)*dim:
      kv+(int64_t(t)*k_heads+hi)*dim;const bf16_t*w=isq?qw:kw;float ss=0;
    for(int d=lane;d<dim;d+=SG_SIZE){float v=float(p[d]);ss=sycl::fma(v,v,ss);}
    ss=sycl::reduce_over_group(sg,ss,sycl::plus<float>());
    float scale=sycl::rsqrt(ss/float(dim)+eps);int pos=start_pos+t;
    for(int j=lane;j<rot/2;j+=SG_SIZE){
      float a=float(p[j])*scale*(1.0f+bf16_to_f32(w[j]));
      float b=float(p[j+rot/2])*scale*(1.0f+bf16_to_f32(w[j+rot/2]));
      float inv=sycl::exp(-float(2*j)/float(rot)*sycl::log(theta));
      float ang=float(pos)*inv,cs=sycl::cos(ang),sn=sycl::sin(ang);
      p[j]=sycl_bf16(a*cs-b*sn);p[j+rot/2]=sycl_bf16(a*sn+b*cs);}
    for(int d=rot+lane;d<dim;d+=SG_SIZE)
      p[d]=sycl_bf16(float(p[d])*scale*(1.0f+bf16_to_f32(w[d])));
   });});
}

sycl::event launch_kv_append_bf16_batched(sycl::queue&q,const sycl_bf16*k,
    const sycl_bf16*v,uint8_t*k_cache,uint8_t*v_cache,int tokens,int start_pos,
    int n_kv_heads,int head_dim,int seq_cap,const std::vector<sycl::event>&deps){
 return q.submit([&](sycl::handler&h){h.depends_on(deps);
  h.parallel_for(sycl::range<1>(size_t(tokens)*n_kv_heads*head_dim),[=](sycl::id<1>id){
   int64_t z=id[0];int d=int(z%head_dim);int64_t th=z/head_dim;
   int kh=int(th%n_kv_heads),t=int(th/n_kv_heads),pos=start_pos+t;if(pos>=seq_cap)return;
   int64_t src=(int64_t(t)*n_kv_heads+kh)*head_dim+d;
   k_cache[(int64_t(kh)*head_dim+d)*seq_cap+pos]=f32_to_e4m3(float(k[src]));
   v_cache[(int64_t(kh)*seq_cap+pos)*head_dim+d]=f32_to_e4m3(float(v[src]));});});
}

sycl::event launch_deltanet_gates_batched(sycl::queue& q, const float* ab,
    const bf16_t* A_log, const bf16_t* dt_bias, float* alpha, float* beta,
    int tokens, int heads, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) { h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(tokens) * heads), [=](sycl::id<1> id) {
            const int64_t z = int64_t(id[0]); const int hd = int(z % heads), t = int(z / heads);
            const float dt = ab[int64_t(t) * 2 * heads + hd] + bf16_to_f32(dt_bias[hd]);
            const float sp = dt > 20.0f ? dt : sycl::log(1.0f + sycl::exp(dt));
            alpha[z] = sycl::exp(-sycl::exp(bf16_to_f32(A_log[hd])) * sp);
            beta[z] = 1.0f / (1.0f + sycl::exp(-ab[int64_t(t) * 2 * heads + heads + hd]));
        });
    });
}

// `stride` is the row pitch of the head-major gate_a/beta buffers, which is
// the chunk-padded token count (M + chunk_size - 1), not the number of real
// tokens.  The chunked GDN kernel reads a whole 64-token tile from the last
// chunk, so with stride == tokens that tile runs off the end of head h into
// head h+1's data.  vLLM sizes these as {num_v_heads, total_token + padding}
// and zero-fills them for exactly this reason.
sycl::event launch_deltanet_native_gates(sycl::queue& q, const float* ab,
    float* gate_a, float* beta, int tokens, int heads, int64_t stride,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){h.depends_on(deps);
        h.parallel_for(sycl::range<2>(size_t(tokens),size_t(heads)),[=](sycl::id<2> id){
            const int t=int(id[0]),hd=int(id[1]);
            gate_a[int64_t(hd)*stride+t]=ab[int64_t(t)*2*heads+hd];
            const float x=ab[int64_t(t)*2*heads+heads+hd];
            beta[int64_t(hd)*stride+t]=1.0f/(1.0f+sycl::exp(-x));
        });});
}

sycl::event launch_swiglu_batched(sycl::queue& q, const float* gu, float* out,
    int tokens, int inter, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) { h.depends_on(deps);
        // DFlash uses inter=6144. Keep large row widths out of an implicit
        // work-group dimension, whose B70 limit is 1024.
        h.parallel_for(sycl::range<1>(size_t(tokens) * inter), [=](sycl::id<1> id) {
            const int t = int(id[0] / inter), d = int(id[0] % inter);
            const int64_t z = int64_t(t) * inter + d;
            const float g = gu[int64_t(t) * 2 * inter + d];
            out[z] = (g / (1.0f + sycl::exp(-g))) * gu[int64_t(t) * 2 * inter + inter + d];
        });
    });
}

// SwiGLU over SEPARATE gate and up buffers, for the BesTLA prefill path (BesTLA has
// no fused 2*inter weight, so it produces gate and up as two [M,inter] results).
sycl::event launch_swiglu_f16_batched(sycl::queue& q, const float* gu,
    float* out, int tokens, int inter,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(tokens)*inter),[=](sycl::id<1> id){
            const int64_t z=id[0];
            const int row=int(z/inter),d=int(z%inter);
            const float gate=float(sycl::half(gu[int64_t(row)*2*inter+d]));
            const float up=float(sycl::half(
                gu[int64_t(row)*2*inter+inter+d]));
            out[z]=float(sycl::half((gate/(1.0f+sycl::exp(-gate)))*up));
        });
    });
}

sycl::event launch_swiglu_bf16_split(sycl::queue& q, const sycl_bf16* gate,
    const sycl_bf16* up, sycl_bf16* out, int tokens, int inter,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) { h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(tokens)*inter),[=](sycl::id<1> id){
            const int64_t z=id[0];
            const float g=float(gate[z]);
            out[z]=sycl_bf16((g/(1.0f+sycl::exp(-g)))*float(up[z]));
        }); });
}

// bf16 activations -> int8 with a per-ROW scale.  The existing
// launch_quantize_rows_int8 takes f32; prefill carries activations as bf16.
// One work-group per row, so at M=4096 that is 4096 groups -- plenty.
sycl::event launch_quantize_rows_int8_bf16(sycl::queue& q, const sycl_bf16* x,
                                           int8_t* xq, float* scales,
                                           int M, int K,
                                           const std::vector<sycl::event>& deps) {
    constexpr int WG = 256;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> lm(WG / SG_SIZE, h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(M) * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg  = it.get_sub_group();
                const int  row = int(it.get_group(0));
                const int  lid = int(it.get_local_id(0));
                const sycl_bf16* r = x + int64_t(row) * K;
                float amax = 0.0f;
                for (int k = lid; k < K; k += WG)
                    amax = sycl::fmax(amax, sycl::fabs(float(r[k])));
                amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
                float* lp = lm.template
                    get_multi_ptr<sycl::access::decorated::no>().get();
                if (sg.get_local_id()[0] == 0) lp[sg.get_group_id()[0]] = amax;
                sycl::group_barrier(it.get_group());
                float t = 0.0f;
                for (int i = 0; i < WG / SG_SIZE; ++i) t = sycl::fmax(t, lp[i]);
                const float sc  = (t > 0.0f) ? t / 127.0f : 1.0f;
                const float inv = 1.0f / sc;
                if (lid == 0) scales[row] = sc;
                int8_t* o = xq + int64_t(row) * K;
                for (int k = lid; k < K; k += WG) {
                    int v = int(sycl::round(float(r[k]) * inv));
                    o[k] = int8_t(sycl::clamp(v, -127, 127));
                }
            });
    });
}

sycl::event launch_swiglu_bf16(sycl::queue& q, const sycl_bf16* gu,
    sycl_bf16* out, int tokens, int inter,
    const std::vector<sycl::event>& deps) {
    constexpr size_t WG=256;
    const size_t padded=(size_t(inter)+WG-1)/WG*WG;
    return q.submit([&](sycl::handler& h) { h.depends_on(deps);
        // Preserve 2-D indexing without letting a 6K-wide row become an
        // implicit work-group dimension on Level Zero.
        h.parallel_for(sycl::nd_range<2>({size_t(tokens),padded},{1,WG}),
          [=](sycl::nd_item<2> it){
            const int t=int(it.get_global_id(0)), d=int(it.get_global_id(1));
            if(d>=inter)return;
            const int64_t z=int64_t(t)*inter+d;
            const float g=float(gu[int64_t(t)*2*inter+d]);
            out[z]=sycl_bf16((g/(1.0f+sycl::exp(-g)))*
                             float(gu[int64_t(t)*2*inter+inter+d]));
        }); });
}

sycl::event launch_swiglu_bf16_quant(sycl::queue& q,const sycl_bf16*gu,
    sycl_bf16*out,int8_t*out_q,float*out_scale,int tokens,int inter,
    const std::vector<sycl::event>&deps){
 constexpr int WG=256,NSG=WG/SG_SIZE;
 return q.submit([&](sycl::handler&h){h.depends_on(deps);
  sycl::local_accessor<float,1>partial(NSG,h);
  h.parallel_for(sycl::nd_range<1>(size_t(tokens)*WG,WG),
   [=](sycl::nd_item<1>it)[[sycl::reqd_sub_group_size(SG_SIZE)]]{
    auto sg=it.get_sub_group();int t=int(it.get_group(0));
    int lid=int(it.get_local_id(0)),lane=int(sg.get_local_id()[0]);
    int sgid=int(sg.get_group_id()[0]);float mx=0.0f;
    for(int d=lid;d<inter;d+=WG){float g=float(gu[int64_t(t)*2*inter+d]);
      float v=(g/(1.0f+sycl::exp(-g)))*float(gu[int64_t(t)*2*inter+inter+d]);
      out[int64_t(t)*inter+d]=sycl_bf16(v);mx=sycl::fmax(mx,sycl::fabs(v));}
    mx=sycl::reduce_over_group(sg,mx,sycl::maximum<float>());
    float*pt=partial.template get_multi_ptr<sycl::access::decorated::no>().get();
    if(lane==0)pt[sgid]=mx;sycl::group_barrier(it.get_group());
    float rowmax=0.0f;for(int i=0;i<NSG;++i)rowmax=sycl::fmax(rowmax,pt[i]);
    float sc=rowmax>0.0f?rowmax/127.0f:1.0f,inv=1.0f/sc;
    if(lid==0)out_scale[t]=sc;
    for(int d=lid;d<inter;d+=WG){float v=float(out[int64_t(t)*inter+d]);
      out_q[int64_t(t)*inter+d]=int8_t(sycl::clamp(int(sycl::round(v*inv)),-127,127));}
   });});
}

sycl::event launch_scale_by_sigmoid_batched(sycl::queue& q, float* x,
    const float* gate, int tokens, int hidden,
    const std::vector<sycl::event>& deps) {
    constexpr size_t WG=256;
    const size_t padded=(size_t(hidden)+WG-1)/WG*WG;
    return q.submit([&](sycl::handler& h) { h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({size_t(tokens),padded},{1,WG}),
          [=](sycl::nd_item<2> it) {
            const int t=int(it.get_global_id(0)),d=int(it.get_global_id(1));
            if(d<hidden)
                x[int64_t(t)*hidden+d]*=1.0f/(1.0f+sycl::exp(-gate[t]));
        });
    });
}

sycl::event launch_gate_sigmoid_mul_batched(sycl::queue& q, float* x,
    const float* gate, int tokens, int hidden,
    const std::vector<sycl::event>& deps) {
    constexpr size_t WG=256;
    const size_t padded=(size_t(hidden)+WG-1)/WG*WG;
    return q.submit([&](sycl::handler& h) { h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({size_t(tokens),padded},{1,WG}),
          [=](sycl::nd_item<2> it) {
            const int t=int(it.get_global_id(0)),d=int(it.get_global_id(1));
            if(d<hidden) {
                const int64_t i=int64_t(t)*hidden+d;
                const float xv=float(sycl::half(x[i]));
                const float gv=float(sycl::half(gate[i]));
                x[i]=float(sycl::half(xv/(1.0f+sycl::exp(-gv))));
            }
        });
    });
}

sycl::event launch_permute_rows_bf16(sycl::queue& q, const float* src,
    const int32_t* perm_token, sycl_bf16* dst, int rows, int width,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){h.depends_on(deps);
        h.parallel_for(sycl::range<2>(size_t(rows),size_t(width)),[=](sycl::id<2> id){
            const int r=int(id[0]), d=int(id[1]);
            dst[int64_t(r)*width+d]=sycl_bf16(src[int64_t(perm_token[r])*width+d]);
        });});
}

sycl::event launch_moe_unpermute(sycl::queue& q, const float* src,
    const int32_t* inverse, const float* route_weight, float* dst,
    int tokens, int top_k, int hidden,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){h.depends_on(deps);
        h.parallel_for(sycl::range<2>(size_t(tokens),size_t(hidden)),[=](sycl::id<2> id){
            const int t=int(id[0]), d=int(id[1]);
            const int64_t z=int64_t(t)*hidden+d;
            float v=0.0f;
            for(int s=0;s<top_k;++s){const int64_t r=int64_t(t)*top_k+s;
                v=sycl::fma(route_weight[r],src[int64_t(inverse[r])*hidden+d],v);}
            dst[z]=v;
        });});
}

sycl::event launch_moe_unpermute_bf16(sycl::queue& q, const sycl_bf16* src,
    const int32_t* inverse, const float* route_weight, float* dst,
    int tokens, int top_k, int hidden,
    const std::vector<sycl::event>& deps) {
    constexpr int WG=256, EPI=8;
    return q.submit([&](sycl::handler& h){h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(tokens)*WG,WG),
          [=](sycl::nd_item<1> it){
            const int t=int(it.get_group(0)),lid=int(it.get_local_id(0));
            int32_t route[8]; float weight[8];
            #pragma unroll
            for(int s=0;s<8;++s){
                const int64_t r=int64_t(t)*top_k+s;
                route[s]=inverse[r]; weight[s]=route_weight[r];
            }
            for(int d=lid*EPI;d<hidden;d+=WG*EPI){
                float v[EPI]={};
                #pragma unroll
                for(int s=0;s<8;++s){
                    const sycl_bf16* row=src+int64_t(route[s])*hidden+d;
                    #pragma unroll
                    for(int e=0;e<EPI;++e)
                        v[e]=sycl::fma(weight[s],float(row[e]),v[e]);
                }
                #pragma unroll
                for(int e=0;e<EPI;++e)dst[int64_t(t)*hidden+d+e]=v[e];
            }
        });});
}

void launch_moe_remap_bf16_top8(sycl::queue& q, const sycl_bf16* hidden,
    const int32_t* topk_ids, sycl_bf16* remapped, int32_t* rows_per_expert,
    int32_t* expert_offsets, int32_t* inverse, int tokens, int hidden_size,
    int num_experts) {
    constexpr int TOPK = 8;
    constexpr int WG = 256;
    const int routes = tokens * TOPK;
    q.memset(rows_per_expert, 0, size_t(num_experts) * sizeof(int32_t));

    // This is the framework-free form of Intel vLLM RowsPerExpertCount:
    // aggregate in SLM, reserve one contiguous range per work-group/expert,
    // then turn each route's local rank into an expert-local global rank.
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> counts(sycl::range<1>(num_experts), h);
        h.parallel_for(sycl::nd_range<1>(
            sycl::range<1>(size_t((routes + WG - 1) / WG) * WG),
            sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int lid = int(it.get_local_id(0));
            for (int e = lid; e < num_experts; e += WG) counts[e] = 0;
            it.barrier(sycl::access::fence_space::local_space);
            const int r = int(it.get_global_id(0));
            int expert = -1;
            if (r < routes) {
                expert = topk_ids[r];
                auto a = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                    sycl::memory_scope::work_group,
                    sycl::access::address_space::local_space>(counts[expert]);
                inverse[r] = a.fetch_add(1);
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int e = lid; e < num_experts; e += WG) {
                const int n = counts[e];
                if (n) {
                    auto a = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(rows_per_expert[e]);
                    counts[e] = a.fetch_add(n);
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (r < routes) inverse[r] += counts[expert];
        });
    });

    // Compute the expert-major base offsets once.
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> prefix(sycl::range<1>(num_experts), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(WG),
            sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int lid = int(it.get_local_id(0));
            prefix[lid] = lid == 0 ? 0 : rows_per_expert[lid - 1];
            it.barrier(sycl::access::fence_space::local_space);
            sycl::joint_inclusive_scan(it.get_group(), prefix.get_multi_ptr<
                sycl::access::decorated::no>().get(),
                prefix.get_multi_ptr<sycl::access::decorated::no>().get() + num_experts,
                prefix.get_multi_ptr<sycl::access::decorated::no>().get(), sycl::plus<int>{});
            expert_offsets[lid]=prefix[lid];
        });
    });

    // One work-group per token loads each 16-byte source vector once and fans
    // it out to all eight expert rows.  This is the useful part of Intel's
    // RemapHiddenStates schedule: the older route-major kernel reread every
    // 4 KiB source row eight times and launched 8x as many work-groups.
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(tokens) * WG),
            sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int lid=int(it.get_local_id(0));
            const int token=int(it.get_group(0));
            int dst[TOPK];
            #pragma unroll
            for(int s=0;s<TOPK;++s){
                const int r=token*TOPK+s;
                dst[s]=inverse[r]+expert_offsets[topk_ids[r]];
            }
            using V=sycl::vec<sycl_bf16,8>;
            for(int d=lid*8;d<hidden_size;d+=WG*8){
                const V v=*reinterpret_cast<const V*>(
                    hidden+int64_t(token)*hidden_size+d);
                #pragma unroll
                for(int s=0;s<TOPK;++s)
                    *reinterpret_cast<V*>(remapped+int64_t(dst[s])*hidden_size+d)=v;
            }
            if(lid==0){
                #pragma unroll
                for(int s=0;s<TOPK;++s)inverse[token*TOPK+s]=dst[s];
            }
        });
    });
}

} // namespace b70

namespace b70 {

// Activations must be bf16 for the XMX bf16 DPAS path.
sycl::event launch_f32_to_bf16(sycl::queue& q, const float* src, sycl_bf16* dst,
                               size_t n, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            dst[id[0]] = sycl_bf16(src[id[0]]);
        });
    });
}

sycl::event launch_bf16_to_f32(sycl::queue& q, const sycl_bf16* src, float* dst,
                               size_t n, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            dst[id[0]] = float(src[id[0]]);
        });
    });
}

sycl::event launch_rmsnorm_gate_silu_bf16_io(sycl::queue&q,
    const sycl_bf16*x,const float*z,const bf16_t*w,sycl_bf16*out,
    int n_heads,int dim,float eps,const std::vector<sycl::event>&deps){
 return q.submit([&](sycl::handler&h){h.depends_on(deps);
  h.parallel_for(sycl::nd_range<1>(size_t(n_heads)*SG_SIZE,SG_SIZE),
   [=](sycl::nd_item<1>it)[[sycl::reqd_sub_group_size(SG_SIZE)]]{
    auto sg=it.get_sub_group();int lane=int(sg.get_local_id()[0]);
    int64_t base=int64_t(it.get_group(0))*dim;float ss=0.0f;
    for(int i=lane;i<dim;i+=SG_SIZE){float v=float(x[base+i]);ss=sycl::fma(v,v,ss);}
    ss=sycl::reduce_over_group(sg,ss,sycl::plus<float>());
    float scale=sycl::rsqrt(ss/float(dim)+eps);
    for(int i=lane;i<dim;i+=SG_SIZE){float zv=z[base+i];
      out[base+i]=sycl_bf16(float(x[base+i])*scale*bf16_to_f32(w[i])*
        (zv/(1.0f+sycl::exp(-zv))));}
   });});
}

sycl::event launch_gate_sigmoid_mul_bf16_io(sycl::queue&q,
    const sycl_bf16*x,const float*gate,sycl_bf16*out,size_t n,
    const std::vector<sycl::event>&deps){
 return q.submit([&](sycl::handler&h){h.depends_on(deps);
  h.parallel_for(sycl::range<1>(n),[=](sycl::id<1>id){size_t i=id[0];
   float g=gate[i];out[i]=sycl_bf16(float(x[i])*(g/(1.0f+sycl::exp(-g))));});});
}

sycl::event launch_f32_to_bf16_scaled(sycl::queue& q, const float* src,
    sycl_bf16* dst, size_t n, float scale,
    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h){h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){
            dst[id[0]]=sycl_bf16(src[id[0]]*scale);
        });});
}

} // namespace b70

namespace b70 {

// ---------------------------------------------------------------------
// Dequantize a whole weight matrix to bf16 once.
//
// The XMX GEMM stages B tiles into SLM and dequantizes them there, which
// is right for decode -- but at batch M the same weight element is
// re-decoded ceil(M/WG_M) times. At M=4096 that is 64x redundant work,
// and it is why the kernel sits at ~3% of DPAS peak rather than being
// limited by the matrix engines.
//
// Prefill can afford to pay N*K once: an [8192][2048] matrix is 32 MiB
// in bf16, trivially reusable across all tokens of the prompt. After
// this the GEMM reads plain bf16 and the dequant disappears from the
// inner loop entirely.
// ---------------------------------------------------------------------
sycl::event launch_dequant_bf16(sycl::queue& q, const QuantWeight& w,
                                sycl_bf16* dst,
                                const std::vector<sycl::event>& deps) {
    const int N = w.N, K = w.K;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const QuantWeight wc = w;
        h.parallel_for(sycl::range<1>(size_t(N) * K), [=](sycl::id<1> id) {
            const int64_t i = int64_t(id[0]);
            const int n = int(i / K), k = int(i % K);
            dst[i] = sycl_bf16(wc.at(n, k));
        });
    });
}

} // namespace b70
