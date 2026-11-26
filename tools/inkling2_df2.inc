// GRIMOIRE -- Inkling 2 Mini: the DFlash2 drafter (incoai/Inkling-2-Mini-DFlash2), included by inkling2_gpu.cpp.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// INK_DFLASH2=<dir with config.json + model.safetensors>: drafts with the block-diffusion drafter + candidate selector
// instead of the release's DSpark head (mtp.safetensors).  The verify, commit and sampling paths are DSpark's.
//
// Reference (transcribed, not re-derived): SGLang python/sglang/srt/models/dflash.py (DFlash2DraftModel,
// DFlashDecoderLayer, DFlashGroupedConv, CandidateSelector, candidate_topk), srt/speculative/dflash_worker_v2.py
// (context K/V = k_norm(rope(k_proj(hidden_norm(fc(taps))))), block = [anchor, mask x 7] through the target's
// embedding), srt/models/inkling2.py (the target embedding includes embed_norm) and the checkpoint's own
// sglang_inkling_dflash2.py: the taps are DSpark's (the stream entering applications 5 / 23 / 41 and the final norm's
// output = TAP0..TAP3), candidates come from the target head over the 200058 real tokens, and the selector's softcap
// is applied AFTER the edge correction (unary logits stay raw).
//
// One draft = 5 layers over 8 rows:
//   h0 = embed_norm(embed[anchor]), embed_norm(embed[mask]) x 7;  residual stream r
//   per layer: x = rms(r);  (x', ka) = conv_in(x);  a = o(attn(qkv(x')));  r += conv_out(a, ka)
//              x = rms(r);  (x', km) = conv_in(x);  m = down(silu(gate x') * up x');  r += conv_out(m, km)
//   fin = rms(r, norm);  rows 1..7: top-16 candidates of the target head, edge scores
//   score[e][p][c] = 30 tanh((logit[e][c] + <A[pred] * P fin[e], B[cand[e][c]]>) / 30), pred = anchor / cand[e-1][p]
//   walk: argmax (greedy) or a draw from softmax(score / T) (q of the exact p/q verify).
// conv_in / conv_out: grouped dynamic 2-tap convolution over the block rows (row j sees row j - 1), coefficients
// base[side][tap][ch] + delta[row][side][tap][ch / 16], delta = kernel_projection(x) (576 = 2 x 2 x 144).
// Attention: 32 query / 8 KV heads of 128, q/k RMSNorm, NeoX RoPE theta 1e4, non-causal over the block, every layer a
// 2048 window (HF window: the key at distance 2047 is the oldest one seen).
// (b70/json.hpp is included at the top of inkling2_gpu.cpp: this file is included inside namespace ink)

class InkF2Q; class InkF2Emb; class InkF2Norm; class InkF2Conv; class InkF2QK; class InkF2AttS; class InkF2AttM;
class InkF2Silu; class InkF2XB; class InkF2Edge; class InkF2Cat; class InkF2HN; class InkF2KV; class InkF2FinB;

namespace {
// Per-head RMSNorm (128) + NeoX RoPE for one row of q or k; one sub-group per head (lanes 0..7 = dims 0..63).
inline void f2_head_norm_rope(sycl::sub_group sg, const float* in, float* outf, uint16_t* outb, const uint16_t* w, int pos, float eps,
                              float theta) {
    const int lane = int(sg.get_local_linear_id());
    float v[8], ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { v[e] = bfr(in[lane * 8 + e]); ss += v[e] * v[e]; }
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    const float r = 1.f / sycl::sqrt(ss / 128.f + eps);
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = bfr((v[e] * r) * bfw(w[lane * 8 + e]));
    float o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float partner = sycl::permute_group_by_xor(sg, v[e], 8);
        const int d = (lane & 7) * 8 + e;
        const float inv = sycl::pow(theta, -float(2 * d) / 128.f);
        const float ang = float(pos) * inv;
        const float cs = sycl::cos(ang), sn = sycl::sin(ang);
        o[e] = lane < 8 ? bfr(v[e] * cs - partner * sn) : bfr(v[e] * cs + partner * sn);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if (outf) outf[lane * 8 + e] = o[e];
        if (outb) outb[lane * 8 + e] = tobf(o[e]);
    }
}
}  // namespace

// INK_DFLASH2=<dir>: that DFlash2 drafter; INK_DFLASH2=0: the release's DSpark head; unset: DFlash2 when a drafter
// folder sits next to the model (<model>/../Inkling-2-Mini-DFlash2 or <model>/dflash2) and INK_I8 is on, else DSpark.
// MEASURED 11-26 (two B70, BetterBench quick, Inkling-2-Mini-Uncensored): DFlash2 179.2 vs DSpark 139.7 weighted.
void Engine::load_drafter(const std::string& model) {
    const char* e = std::getenv("INK_DFLASH2");
    std::string dir;
    if (e && *e) { if (std::string(e) != "0") dir = e; }
    else if (i8v_) {
        std::string m = model;
        while (m.size() > 1 && m.back() == '/') m.pop_back();
        const auto slash = m.find_last_of('/');
        const std::string parent = slash == std::string::npos ? "." : slash == 0 ? "/" : m.substr(0, slash);
        for (const std::string& c : {parent + "/Inkling-2-Mini-DFlash2", m + "/dflash2"}) {
            struct stat sb{};
            if (::stat((c + "/config.json").c_str(), &sb) == 0 && ::stat((c + "/model.safetensors").c_str(), &sb) == 0) { dir = c; break; }
        }
        if (!dir.empty()) std::printf("dflash2 drafter found at %s (INK_DFLASH2=0: the DSpark head)\n", dir.c_str());
    }
    if (!dir.empty()) load_df2(dir);
    else load_mtp(model + "/mtp.safetensors");
}

// BF16 row-major [N][K] -> INT8 row-major [N][Kp] (K zero-padded to 512), scale = row absmax / 127 (the DSpark drafter's
// INT8 rule), for b70::launch_fp8_smallm.
Engine::F2W Engine::f2_quant(const uint16_t* src, int N, int K) {
    F2W w;
    w.N = N; w.K = K; w.Kp = (K + 511) / 512 * 512;
    // INK_F2_I4=1: GRIMOIRE's INT4 (RTN, group 128) for the drafter's linears: half the bytes of INT8 on every draft
    static const bool i4 = std::getenv("INK_F2_I4") && std::atoi(std::getenv("INK_F2_I4")) == 1;
    if (i4) { w.q4 = quant4(src, N, K); return w; }
    // INK_F2_BF16=1: the checkpoint's BF16 weights through b70::launch_bf16_smallm (twice INT8's bytes)
    static const bool bf = std::getenv("INK_F2_BF16") && std::atoi(std::getenv("INK_F2_BF16")) == 1;
    if (bf) {
        w.Kp = K;
        w.B = dev_.alloc<uint16_t>(size_t(N) * K);
        q_.memcpy(w.B, src, size_t(N) * K * 2).wait();
        return w;
    }
    w.Q = dev_.alloc<int8_t>(size_t(N) * w.Kp);
    w.S = dev_.alloc<float>(N);
    uint16_t* tmp = sycl::malloc_device<uint16_t>(size_t(N) * K, q_);
    if (!tmp) throw std::runtime_error("dflash2: staging");
    q_.memcpy(tmp, src, size_t(N) * K * 2).wait();
    int8_t* Q = w.Q;
    float* S = w.S;
    const int Kp = w.Kp;
    q_.parallel_for<InkF2Q>(sycl::nd_range<1>(size_t(N) * 256, 256), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int lid = int(it.get_local_id(0));
        float mx = 0.f;
        for (int k = lid; k < K; k += 256) mx = sycl::fmax(mx, sycl::fabs(bfw(tmp[r * K + k])));
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        const float s = mx > 0.f ? mx / 127.f : 1.f;
        for (int k = lid; k < Kp; k += 256)
            Q[r * Kp + k] = k < K ? int8_t(sycl::clamp(sycl::rint(bfw(tmp[r * K + k]) / s), -127.f, 127.f)) : int8_t(0);
        if (lid == 0) S[r] = s;
    });
    q_.wait();
    sycl::free(tmp, q_);
    return w;
}

// y[M][N] = x[M][K] (pitch ldx, rounded to bf16) . W^T, INT8 weights (fp32 out, not rounded)
void Engine::f2_mm(const F2W& w, const float* x, int ldx, float* y, int M) {
    uint16_t* xb = f2_.xb;
    const int K = w.K, Kp = w.Kp;
    q_.parallel_for<InkF2XB>(sycl::range<1>(size_t(M) * Kp), [=](sycl::id<1> id) {
        const size_t m = id[0] / Kp, k = id[0] % Kp;
        xb[id[0]] = k < size_t(K) ? tobf(x[m * ldx + k]) : uint16_t(0);
    });
    if (w.q4.pay) { if (!i4_gemv(w.q4, 0, w.N, xb, y, M)) throw std::runtime_error("dflash2: INT4 GEMV refused"); return; }
    if (w.B) { if (!gsm_dense(w.B, w.N, K, xb, y, M)) throw std::runtime_error("dflash2: BF16 GEMV refused"); return; }
    if (!i8_small(w.Q, w.S, 0, w.N, Kp, xb, y, M)) throw std::runtime_error("dflash2: INT8 GEMV refused");
}

void Engine::load_df2(const std::string& dir) {
    if (!i8v_) throw std::runtime_error("dflash2: needs INK_I8=1 (the INT8 target head scores the candidates)");
    std::string cfgtxt;
    {
        std::ifstream f(dir + "/config.json");
        if (!f) throw std::runtime_error("dflash2: no " + dir + "/config.json");
        std::stringstream ss; ss << f.rdbuf(); cfgtxt = ss.str();
    }
    const b70::Json cfg = b70::Json::parse(cfgtxt);
    const b70::Json& dc = cfg.at("dflash_config");
    const int h = h_;
    F2& F = f2_;
    F.L = cfg.at("num_hidden_layers").integer();
    F.heads = cfg.at("num_attention_heads").integer();
    F.kvh = cfg.at("num_key_value_heads").integer();
    F.inter = cfg.at("intermediate_size").integer();
    F.win = cfg.at("sliding_window").integer();
    F.mask = dc.at("mask_token_id").integer();
    F.topk = dc.at("selector_top_k").integer();
    F.rank = dc.at("selector_rank").integer();
    F.eps = float(cfg.at("rms_norm_eps").number);
    F.theta = 10000.f;
    if (const auto* rp = cfg.find("rope_parameters")) if (const auto* t = rp->find("rope_theta")) F.theta = float(t->number);
    F.softcap = float(dc.at("final_logit_softcapping").number);
    const std::string mode = dc.text_or("selector_softcap_mode", "before_correction");
    {
        std::string why;
        if (cfg.at("hidden_size").integer() != h) why = "hidden_size";
        else if (cfg.at("head_dim").integer() != 128 || F.heads != 32 || F.kvh != 8) why = "attention shape (32 x 128 / 8 KV)";
        else if (dc.at("block_size").integer() != 8) why = "block_size 8";
        else if (dc.at("conv_kernel_size").integer() != 2 || dc.at("conv_group_size").integer() != 16 || h % 16) why = "conv 2 taps / group 16";
        else if (F.topk != 16 || F.rank <= 0 || F.rank % 16) why = "selector top-k 16";
        else if (mode != "after_correction") why = "selector_softcap_mode " + mode;
        else if (F.mask < 0 || F.mask >= c_.vocab) why = "mask_token_id";
        else if (F.win < 9 || F.win > 2048) why = "sliding_window (<= 2048)";
        else if (F.inter % 16) why = "intermediate_size";
        const auto& ids = dc.at("target_layer_ids").array;
        if (why.empty() && (ids.size() != 4 || ids[0].integer() != 5 || ids[1].integer() != 23 || ids[2].integer() != 41 || ids[3].integer() != 48))
            why = "target_layer_ids (5, 23, 41, 48: DSpark's taps)";
        const auto& lt = cfg.at("layer_types").array;
        if (why.empty() && int(lt.size()) != F.L) why = "layer_types";
        for (const auto& x : lt) if (why.empty() && x.text() != "sliding_attention") why = "every layer sliding";
        if (cfg.bool_or("is_causal", false)) why = "is_causal (the block is non-causal)";
        if (!why.empty()) throw std::runtime_error("dflash2 config outside this port: " + why);
    }
    auto st = std::make_shared<b70::SafeTensors>();
    std::string err;
    if (!st->open(dir + "/model.safetensors", err)) throw std::runtime_error("dflash2: " + err);
    auto T = [&](const std::string& n, size_t elems) {
        const auto* t = st->find(n);
        if (!t) throw std::runtime_error("dflash2: missing " + n);
        if (size_t(t->end - t->begin) != elems * 2) throw std::runtime_error("dflash2: size of " + n);
        return static_cast<const uint16_t*>(st->data(*t));
    };
    auto up16 = [&](const std::string& n, size_t elems) {
        uint16_t* d = dev_.alloc<uint16_t>(elems);
        q_.memcpy(d, T(n, elems), elems * 2).wait();
        return d;
    };
    auto cat = [&](std::vector<std::pair<std::string, size_t>> parts) {
        std::vector<uint16_t> v;
        for (auto& pr : parts) { const uint16_t* p = T(pr.first, pr.second); v.insert(v.end(), p, p + pr.second); }
        return v;
    };
    const int QD = F.heads * 128, KD = F.kvh * 128, I = F.inter, NG = h / 16;
    const size_t H = size_t(h);
    F.cap = std::min(limit_, 2048);
    F.Ly.resize(F.L);
    for (int l = 0; l < F.L; ++l) {
        const std::string s = "layers." + std::to_string(l) + ".";
        F2Layer& L = F.Ly[l];
        {
            const auto v = cat({{s + "self_attn.q_proj.weight", size_t(QD) * H}, {s + "self_attn.k_proj.weight", size_t(KD) * H}, {s + "self_attn.v_proj.weight", size_t(KD) * H}});
            L.qkv = f2_quant(v.data(), QD + 2 * KD, h);
            L.kvW = dev_.alloc<uint16_t>(size_t(2 * KD) * H);   // [k; v] BF16 for the context rows
            q_.memcpy(L.kvW, v.data() + size_t(QD) * H, size_t(2 * KD) * H * 2).wait();
        }
        L.o = f2_quant(T(s + "self_attn.o_proj.weight", H * QD), h, QD);
        {
            const auto v = cat({{s + "mlp.gate_proj.weight", size_t(I) * H}, {s + "mlp.up_proj.weight", size_t(I) * H}});
            L.gu = f2_quant(v.data(), 2 * I, h);
        }
        L.d = f2_quant(T(s + "mlp.down_proj.weight", H * I), h, I);
        L.ca = f2_quant(T(s + "attention_conv.kernel_projection.weight", size_t(4 * NG) * H), 4 * NG, h);
        L.cm = f2_quant(T(s + "mlp_conv.kernel_projection.weight", size_t(4 * NG) * H), 4 * NG, h);
        L.baseA = up16(s + "attention_conv.base_kernel", 4 * H);
        L.baseM = up16(s + "mlp_conv.base_kernel", 4 * H);
        L.nin = up16(s + "input_layernorm.weight", H);
        L.npost = up16(s + "post_attention_layernorm.weight", H);
        L.qn = up16(s + "self_attn.q_norm.weight", 128);
        L.kn = up16(s + "self_attn.k_norm.weight", 128);
        L.kc = dev_.zeros<uint16_t>(size_t(F.cap) * KD);
        L.vc = dev_.zeros<uint16_t>(size_t(F.cap) * KD);
    }
    F.fcW = up16("fc.weight", H * 4 * H);
    F.hnorm = up16("hidden_norm.weight", H);
    F.norm = up16("norm.weight", H);
    F.hpW = up16("candidate_selector.hidden_projection.weight", size_t(F.rank) * H);
    F.pred = up16("candidate_selector.predecessor_codebook", size_t(c_.vocab) * F.rank);
    F.succ = up16("candidate_selector.successor_codebook", size_t(c_.vocab) * F.rank);
    // scratch (8 block rows)
    const int GU = 2 * I, QKV = QD + 2 * KD;
    const int KPmax = std::max({(QD + 511) / 512 * 512, (I + 511) / 512 * 512, kp_});
    F.r = dev_.zeros<float>(8 * H); F.x = dev_.zeros<float>(8 * H); F.xc = dev_.zeros<float>(8 * H); F.cp = dev_.zeros<float>(8 * size_t(4 * NG));
    F.qkv = dev_.zeros<float>(8 * size_t(QKV)); F.att = dev_.zeros<float>(8 * size_t(QD)); F.o = dev_.zeros<float>(8 * H);
    F.gu = dev_.zeros<float>(8 * size_t(GU)); F.act = dev_.zeros<float>(8 * size_t(I)); F.fin = dev_.zeros<float>(8 * H);
    F.maskE = dev_.zeros<float>(H);
    F.hp = dev_.zeros<float>(8 * size_t(F.rank)); F.sc = dev_.zeros<float>(8 * 16 * 16);
    F.part = dev_.zeros<float>(size_t(kF2NS) * 8 * F.heads * 130);
    F.base = dev_.zeros<float>(8 * size_t(vrows_));
    F.xb = dev_.zeros<uint16_t>(8 * size_t(KPmax)); F.bk = dev_.zeros<uint16_t>(8 * size_t(KD)); F.bv = dev_.zeros<uint16_t>(8 * size_t(KD));
    F.finb = dev_.zeros<uint16_t>(8 * size_t(kp_));
    F.cat = dev_.zeros<float>(32 * 4 * H); F.catb = dev_.zeros<uint16_t>(size_t(F.cap) * 4 * H);
    F.ctx = dev_.zeros<float>(size_t(F.cap) * H); F.ctxn = dev_.zeros<float>(size_t(F.cap) * H); F.ctxb = dev_.zeros<uint16_t>(size_t(F.cap) * H);
    F.kv = dev_.zeros<float>(size_t(F.cap) * 2 * KD);
    F.cid = dev_.zeros<int>(8 * 16); F.cval = dev_.zeros<float>(8 * 16);
    if (!d_.dtok) d_.dtok = dev_.zeros<int>(8);
    q_.wait();
    f2_embed(F.mask, F.maskE, 1);   // embed_norm(embed[mask]): the same 7 rows every draft
    q_.wait();
    f2_st_ = st;
    F.ok = true;
    mtp_ = true;
    taps_on = true;
    std::printf("dflash2 drafter loaded (%d layers, block 8, selector top-%d rank %d, window %d, mask %d, %s layers); device total %.2f GiB\n",
                F.L, F.topk, F.rank, F.win, F.mask, F.Ly[0].qkv.q4.pay ? "INT4" : F.Ly[0].qkv.B ? "BF16" : "INT8", dev_.bytes / 1073741824.0);
}

// rows[0] = embed_norm(embed[tok]); with rows == 8 the other 7 rows are the mask row (maskE)
void Engine::f2_embed(int tok, float* dst, int rows) {
    const uint16_t *Em = embed_, *wn = embed_norm_;
    const float* mk = f2_.maskE;
    const int h = h_;
    const float eps = c_.eps;
    q_.parallel_for<InkF2Emb>(sycl::nd_range<1>(size_t(rows) * 256, 256), [=](sycl::nd_item<1> it) {
        const int m = int(it.get_group(0)), lid = int(it.get_local_id(0));
        if (m == 0) {
            const uint16_t* e = Em + size_t(tok) * h;
            float ss = 0.f;
            for (int i = lid; i < h; i += 256) { const float v = bfw(e[i]); ss += v * v; }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float rr = 1.f / sycl::sqrt(ss / float(h) + eps);
            for (int i = lid; i < h; i += 256) dst[i] = bfr((bfw(e[i]) * rr) * bfw(wn[i]));
        } else
            for (int i = lid; i < h; i += 256) dst[size_t(m) * h + i] = mk[i];
    });
}

// One DFlash2 block at positions p..p+7: d_.dtok = [t, drafts], dtok_h_ the same on the host, dn_ drafts proposed
// (dq_[i] = their q when dsp_ samples).
void Engine::f2_block(int t, int p) {
    F2& F = f2_;
    const int h = h_, QD = F.heads * 128, KD = F.kvh * 128, QKV = QD + 2 * KD, I = F.inter, NG = h / 16, NP = 4 * NG;
    const float eps = F.eps, theta = F.theta;
    float* r = F.r;
    f2_embed(t, r, 8);
    // r += add (bf16 add, when add) ; x = rms(r) * w
    auto addnorm = [&](const float* add, const uint16_t* w, float* x) {
        q_.parallel_for<InkF2Norm>(sycl::nd_range<1>(8 * 256, 256), [=](sycl::nd_item<1> it) {
            const int m = int(it.get_group(0)), lid = int(it.get_local_id(0));
            float* rr = r + size_t(m) * h;
            float ss = 0.f;
            for (int i = lid; i < h; i += 256) {
                const float v = add ? bfr(rr[i] + add[size_t(m) * h + i]) : rr[i];
                if (add) rr[i] = v;
                ss += v * v;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float rs = 1.f / sycl::sqrt(ss / float(h) + eps);
            for (int i = lid; i < h; i += 256) x[size_t(m) * h + i] = bfr((rr[i] * rs) * bfw(w[i]));
        });
    };
    // out[j] = c0 * in[j] + c1 * in[j - 1] (j >= 1), c_tap = bf(base[side][tap][ch] + bf(cp[j][side][tap][ch / 16]))
    auto conv = [&](const float* in, int ldin, const uint16_t* base, int side, float* out) {
        const float* cp = F.cp;
        q_.parallel_for<InkF2Conv>(sycl::range<1>(8 * size_t(h)), [=](sycl::id<1> id) {
            const int j = int(id[0] / h), c = int(id[0] % h), g = c / 16;
            const float* cr = cp + size_t(j) * NP + side * 2 * NG;
            const float c0 = bfr(bfw(base[size_t(side * 2) * h + c]) + bfr(cr[g]));
            float v = bfr(c0 * bfr(in[size_t(j) * ldin + c]));
            if (j >= 1) {
                const float c1 = bfr(bfw(base[size_t(side * 2 + 1) * h + c]) + bfr(cr[NG + g]));
                v = bfr(v + bfr(c1 * bfr(in[size_t(j - 1) * ldin + c])));
            }
            out[id[0]] = v;
        });
    };
    const int lo = std::max(0, p - (F.win - 1)), nctx = p - lo, cap = F.cap, win = F.win;
    const float scale = 1.f / std::sqrt(128.f);
    const int NH = F.heads, GQ = F.heads / F.kvh;
    for (int l = 0; l < F.L; ++l) {
        const F2Layer& L = F.Ly[l];
        addnorm(l ? F.o : nullptr, L.nin, F.x);   // F.o holds the previous layer's convolved MLP output
        f2_mm(L.ca, F.x, h, F.cp, 8);
        conv(F.x, h, L.baseA, 0, F.xc);
        f2_mm(L.qkv, F.xc, h, F.qkv, 8);
        {
            float* qkv = F.qkv;
            uint16_t *bk = F.bk, *bv = F.bv;
            const uint16_t *qn = L.qn, *kn = L.kn;
            const int nh = NH, nkv = F.kvh;
            q_.parallel_for<InkF2QK>(sycl::nd_range<1>(size_t(8) * (NH + 2 * F.kvh) * 16, (NH + 2 * F.kvh) * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int m = int(it.get_group(0)), hd = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
                float* row = qkv + size_t(m) * QKV;
                if (hd < nh) f2_head_norm_rope(sg, row + hd * 128, row + hd * 128, nullptr, qn, p + m, eps, theta);
                else if (hd < nh + nkv) f2_head_norm_rope(sg, row + QD + (hd - nh) * 128, nullptr, bk + size_t(m) * KD + (hd - nh) * 128, kn, p + m, eps, theta);
                else {
                    const int vh = hd - nh - nkv;
#pragma unroll
                    for (int e = 0; e < 8; ++e) bv[size_t(m) * KD + vh * 128 + lane * 8 + e] = tobf(bfr(row[QD + KD + vh * 128 + lane * 8 + e]));
                }
            });
        }
        {   // context keys [lo, p) in key slices (work-group = KV head x slice, its GQ x 8 queries share each K / V tile in
            // SLM), then a merge that adds the 8 block keys (non-causal).  Row m sees keys at >= p + m - (win - 1).
            const float* qkv = F.qkv;
            float *att = F.att, *part = F.part;
            const uint16_t *kc = L.kc, *vc = L.vc, *bk = F.bk, *bv = F.bv;
            const int NS = kF2NS, sl = std::max(1, (nctx + NS - 1) / NS), NQ = 8 * GQ;
            q_.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> qs(sycl::range<1>(size_t(NQ) * 128), cgh), ks(sycl::range<1>(16 * 132), cgh), vs(sycl::range<1>(16 * 128), cgh);
                cgh.parallel_for<InkF2AttS>(sycl::nd_range<1>(size_t(F.kvh) * NS * NQ * 16, size_t(NQ) * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    auto g = it.get_group();
                    auto sg = it.get_sub_group();
                    const int kvh = int(it.get_group(0)) / NS, sp = int(it.get_group(0)) % NS;
                    const int lid = int(it.get_local_id(0)), qi = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
                    const int nthr = NQ * 16;
                    const int m = qi / GQ, j = kvh * GQ + qi % GQ;
                    for (int e = lid; e < NQ * 128; e += nthr) { const int q2 = e / 128, d = e % 128; qs[e] = qkv[size_t(q2 / GQ) * QKV + (kvh * GQ + q2 % GQ) * 128 + d]; }
                    const int k0 = sp * sl, k1 = sycl::min(nctx, k0 + sl);
                    const int lom = p + m - (win - 1);
                    float mx = -INFINITY, lsum = 0.f, acc[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
                    for (int t0 = k0; t0 < k1; t0 += 16) {
                        sycl::group_barrier(g);
                        for (int e = lid; e < 16 * 128; e += nthr) {
                            const int kk = e / 128, d = e % 128, ii = t0 + kk;
                            const size_t rw = size_t((lo + (ii < k1 ? ii : k0)) % cap) * KD + kvh * 128 + d;
                            ks[kk * 132 + d] = bfw(kc[rw]);
                            vs[kk * 128 + d] = bfw(vc[rw]);
                        }
                        sycl::group_barrier(g);
                        const int nk = sycl::min(16, k1 - t0);
                        float s = -INFINITY;
                        if (lane < nk && lo + t0 + lane >= lom) {
                            s = 0.f;
                            for (int d = 0; d < 128; ++d) s += qs[qi * 128 + d] * ks[lane * 132 + d];
                            s *= scale;
                        }
                        const float tm = sycl::reduce_over_group(sg, s, sycl::maximum<float>());
                        const float nm = sycl::fmax(mx, tm);
                        if (nm == -INFINITY) continue;
                        const float corr = mx == -INFINITY ? 0.f : sycl::exp(mx - nm);
                        const float w = s == -INFINITY ? 0.f : sycl::exp(s - nm);
                        lsum = lsum * corr + sycl::reduce_over_group(sg, w, sycl::plus<float>());
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[e] *= corr;
                        for (int kk = 0; kk < nk; ++kk) {
                            const float wk = sycl::select_from_group(sg, w, kk);
#pragma unroll
                            for (int e = 0; e < 8; ++e) acc[e] += wk * vs[kk * 128 + lane * 8 + e];
                        }
                        mx = nm;
                    }
                    float* pp = part + ((size_t(sp) * 8 + m) * NH + j) * 130;
#pragma unroll
                    for (int e = 0; e < 8; ++e) pp[lane * 8 + e] = acc[e];
                    if (lane == 0) { pp[128] = mx; pp[129] = lsum; }
                });
            });
            q_.parallel_for<InkF2AttM>(sycl::nd_range<1>(size_t(8 * NH) * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int t2 = int(it.get_group(0)), m = t2 / NH, j = t2 % NH, kvh = j / GQ, lane = int(sg.get_local_linear_id());
                float qv[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) qv[e] = qkv[size_t(m) * QKV + j * 128 + lane * 8 + e];
                float bs[8];
                for (int b = 0; b < 8; ++b) {
                    float s = 0.f;
#pragma unroll
                    for (int e = 0; e < 8; ++e) s += qv[e] * bfw(bk[size_t(b) * KD + kvh * 128 + lane * 8 + e]);
                    bs[b] = sycl::reduce_over_group(sg, s, sycl::plus<float>()) * scale;
                }
                float gm = -INFINITY;
                for (int b = 0; b < 8; ++b) gm = sycl::fmax(gm, bs[b]);
                for (int s2 = 0; s2 < NS; ++s2) gm = sycl::fmax(gm, part[((size_t(s2) * 8 + m) * NH + j) * 130 + 128]);
                float sum = 0.f, acc[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[e] = 0.f;
                for (int s2 = 0; s2 < NS; ++s2) {
                    const float* pp = part + ((size_t(s2) * 8 + m) * NH + j) * 130;
                    if (pp[128] == -INFINITY) continue;
                    const float w = sycl::exp(pp[128] - gm);
                    sum += pp[129] * w;
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[e] += pp[lane * 8 + e] * w;
                }
                for (int b = 0; b < 8; ++b) {
                    const float w = sycl::exp(bs[b] - gm);
                    sum += w;
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[e] += w * bfw(bv[size_t(b) * KD + kvh * 128 + lane * 8 + e]);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) att[size_t(m) * QD + j * 128 + lane * 8 + e] = bfr(acc[e] / sum);
            });
        }
        f2_mm(L.o, F.att, QD, F.o, 8);
        conv(F.o, h, L.baseA, 1, F.xc);   // attention output convolution (coefficients of this sublayer's input)
        addnorm(F.xc, L.npost, F.x);
        f2_mm(L.cm, F.x, h, F.cp, 8);
        conv(F.x, h, L.baseM, 0, F.xc);
        f2_mm(L.gu, F.xc, h, F.gu, 8);
        {
            const float* gu = F.gu;
            float* act = F.act;
            q_.parallel_for<InkF2Silu>(sycl::range<1>(8 * size_t(I)), [=](sycl::id<1> id) {
                const size_t m = id[0] / I, i = id[0] % I;
                const float g = bfr(gu[m * 2 * I + i]), u = bfr(gu[m * 2 * I + I + i]);
                act[id[0]] = bfr((g / (1.f + sycl::exp(-g))) * u);
            });
        }
        f2_mm(L.d, F.act, I, F.xc, 8);
        conv(F.xc, h, L.baseM, 1, F.o);   // MLP output convolution -> added by the next norm
    }
    addnorm(F.o, F.norm, F.fin);
    // candidates: top-16 of the target's INT8 head (raw logits; the selector softcaps after the correction) for rows 1..7
    {
        uint16_t* fb = F.finb;
        const float* fin = F.fin;
        const int kp = kp_;
        q_.parallel_for<InkF2FinB>(sycl::range<1>(size_t(8) * kp), [=](sycl::id<1> id) {
            const size_t m = id[0] / kp, i = id[0] % kp;
            fb[id[0]] = i < size_t(h) ? tobf(fin[m * h + i]) : uint16_t(0);
        });
    }
    if (!i8_small(unR_, unembedS_, 0, vrows_, kp_, F.finb + kp_, F.base, 7)) throw std::runtime_error("dflash2: INT8 head refused");
    gemvm<0>(F.hpW, F.rank, h, F.fin + h, h, F.hp, F.rank, 7);
    topk_rows(F.base, vrows_, vvalid_, 7, v0_, 0.f, 16);
    float cv[8 * KC]; int ci[8 * KC];
    cands(7, cv, ci);
    for (int e = 0; e < 7; ++e)
        for (int c = 0; c < 16; ++c) { f2h_id_[e * 16 + c] = ci[e * KC + c]; f2h_v_[e * 16 + c] = cv[e * KC + c]; }
    q_.memcpy(F.cid, f2h_id_, sizeof(f2h_id_));
    q_.memcpy(F.cval, f2h_v_, sizeof(f2h_v_));
    {   // score[e][p][c] = softcap(unary[e][c] + bf(<bf(A[pred] * hp[e]), B[cand[e][c]]>)), pred = t (e = 0) / cand[e-1][p]
        const int* cid = F.cid;
        const float *cval = F.cval, *hp = F.hp;
        const uint16_t *A = F.pred, *B = F.succ;
        float* sc = F.sc;
        const int R = F.rank;
        const float cap = F.softcap;
        q_.parallel_for<InkF2Edge>(sycl::nd_range<1>(7 * 256, 256), [=](sycl::nd_item<1> it) {
            const int e = int(it.get_group(0)), lid = int(it.get_local_id(0)), pp = lid / 16, c = lid % 16;
            const int pid = e == 0 ? t : cid[(e - 1) * 16 + pp];
            const int cand = cid[e * 16 + c];
            const uint16_t *a = A + size_t(pid) * R, *b = B + size_t(cand) * R;
            const float* hv = hp + size_t(e) * R;
            float s = 0.f;
            for (int k = 0; k < R; ++k) s += bfr(bfw(a[k]) * hv[k]) * bfw(b[k]);
            const float v = cval[e * 16 + c] + bfr(s);
            sc[(e * 16 + pp) * 16 + c] = cap * sycl::tanh(v / cap);
        });
    }
    q_.memcpy(f2h_sc_, F.sc, sizeof(f2h_sc_)).wait();
    // the walk (host): greedy = argmax chain; sampling = a draw from softmax(score / T) per step (q of the verify)
    const int nd = std::min(dmax_, 7);
    dtok_h_[0] = t;
    dn_ = nd;
    double dcum = 1.0, conf = 1.0;
    static const double gtau = std::getenv("INK_F2_GTAU") ? std::atof(std::getenv("INK_F2_GTAU")) : 0.0;
    const bool samp = dsp_ && dsp_->temp > 0.f;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int prev = 0;
    for (int e = 0; e < nd; ++e) {
        const float* row = f2h_sc_ + (e * 16 + (e == 0 ? 0 : prev)) * 16;
        double pr[16], mx = row[0], s = 0.0;
        for (int c = 1; c < 16; ++c) mx = std::max(mx, double(row[c]));
        int best = 0;
        for (int c = 1; c < 16; ++c) if (row[c] > row[best]) best = c;
        int pick = best;
        if (samp) {
            // q = softmax(score / T) over the 16 candidates; INK_F2_QTRUNC=1 also applies the request's top-k / top-p
            // to q (vLLM order), so drafts avoid tokens outside the target's own support (any q keeps the output exact)
            static const bool qtrunc = std::getenv("INK_F2_QTRUNC") && std::atoi(std::getenv("INK_F2_QTRUNC")) == 1;
            const double T = std::max(1e-5, double(dsp_->temp));
            for (int c = 0; c < 16; ++c) pr[c] = std::exp((double(row[c]) - mx) / T);
            int ord[16];
            for (int c = 0; c < 16; ++c) ord[c] = c;
            std::stable_sort(ord, ord + 16, [&](int a, int b) { return row[a] > row[b]; });
            int k = 16;
            if (qtrunc && dsp_->top_k > 0 && dsp_->top_k < 16) k = dsp_->top_k;
            double tot = 0.0;
            for (int j = 0; j < k; ++j) tot += pr[ord[j]];
            int keep = k;
            if (qtrunc && dsp_->top_p < 1.f) {
                double c = 0.0;
                for (int j = 0; j < k; ++j) { c += pr[ord[j]] / tot; if (c >= double(dsp_->top_p)) { keep = j + 1; break; } }
            }
            for (int j = 0; j < keep; ++j) s += pr[ord[j]];
            Dist& d = dq_[e];
            d.n = keep;
            for (int j = 0; j < keep; ++j) { d.id[j] = f2h_id_[e * 16 + ord[j]]; d.p[j] = pr[ord[j]] / s; }
            const double u = U(*drng_);
            double acc = 0.0;
            int jp = keep - 1;
            for (int j = 0; j < keep; ++j) { acc += d.p[j]; if (u < acc) { jp = j; break; } }
            pick = ord[jp];
            dcum *= d.p[jp];
        } else if (dsp_) {   // temperature 0 through mtp_step_s: a point-mass q
            Dist& d = dq_[e];
            d.n = 1; d.id[0] = f2h_id_[e * 16 + best]; d.p[0] = 1.0;
        }
        if (!samp && gtau > 0.0) {   // greedy: verify up to the first draft the selector (T = 1) finds unlikely
            for (int c = 0; c < 16; ++c) { pr[c] = std::exp(double(row[c]) - mx); s += pr[c]; }
            conf *= pr[pick] / s;
        }
        dtok_h_[1 + e] = f2h_id_[e * 16 + pick];
        prev = pick;
        if ((samp && dtau_ > 0.0 && dcum < dtau_) || (!samp && gtau > 0.0 && conf < gtau)) { dn_ = e + 1; break; }
    }
    for (int e = dn_; e < 7; ++e) dtok_h_[1 + e] = dtok_h_[dn_];   // unused rows: any valid id
    q_.memcpy(d_.dtok, dtok_h_, 8 * sizeof(int)).wait();
}

// Context K/V for target rows m0.. at positions p0..: fc over the four taps, hidden_norm, every layer's k / v, k norm +
// RoPE, into the layer's ring (only the last cap rows can be seen again).
void Engine::f2_commit(int p0, int m0, int rows) {
    F2& F = f2_;
    if (rows > F.cap) { const int skip = rows - F.cap; m0 += skip; p0 += skip; rows = F.cap; }
    const int h = h_, KD = F.kvh * 128, cap = F.cap;
    const float eps = F.eps, theta = F.theta;
    {
        const float* pool = pool_;
        const size_t RS = rs_;
        float* cat = F.cat;
        uint16_t* catb = F.catb;
        const bool wf = rows < 32;
        q_.parallel_for<InkF2Cat>(sycl::range<1>(size_t(rows) * 4 * h), [=](sycl::id<1> id) {
            const size_t m = id[0] / (4 * size_t(h)), k = id[0] % (4 * size_t(h)), tp = k / h, i = k % h;
            const float v = bfr(pool[(TAP0 + tp) * RS + (m0 + m) * h + i]);
            catb[id[0]] = tobf(v);
            if (wf) cat[id[0]] = v;
        });
    }
    dense(F.fcW, h, 4 * h, F.cat, 4 * size_t(h), F.catb, F.ctx, h, rows, true);
    {
        const float* ctx = F.ctx;
        float* cn = F.ctxn;
        uint16_t* cb = F.ctxb;
        const uint16_t* w = F.hnorm;
        q_.parallel_for<InkF2HN>(sycl::nd_range<1>(size_t(rows) * 256, 256), [=](sycl::nd_item<1> it) {
            const size_t m = it.get_group(0);
            const int lid = int(it.get_local_id(0));
            float ss = 0.f;
            for (int i = lid; i < h; i += 256) { const float v = bfr(ctx[m * h + i]); ss += v * v; }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float rs = 1.f / sycl::sqrt(ss / float(h) + eps);
            for (int i = lid; i < h; i += 256) { const float y = bfr((bfr(ctx[m * h + i]) * rs) * bfw(w[i])); cn[m * h + i] = y; cb[m * h + i] = tobf(y); }
        });
    }
    for (int l = 0; l < F.L; ++l) {
        F2Layer& L = F.Ly[l];
        dense(L.kvW, 2 * KD, h, F.ctxn, h, F.ctxb, F.kv, 2 * KD, rows, true);
        const float* kv = F.kv;
        uint16_t *kc = L.kc, *vc = L.vc;
        const uint16_t* kn = L.kn;
        const int nkv = F.kvh;
        q_.parallel_for<InkF2KV>(sycl::nd_range<1>(size_t(rows) * 2 * nkv * 16, 2 * nkv * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            const int m = int(it.get_group(0)), hd = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
            const int pos = p0 + m;
            const size_t slot = size_t(pos % cap);
            const float* row = kv + size_t(m) * 2 * KD;
            if (hd < nkv) f2_head_norm_rope(sg, row + hd * 128, nullptr, kc + slot * KD + hd * 128, kn, pos, eps, theta);
            else {
                const int vh = hd - nkv;
#pragma unroll
                for (int e = 0; e < 8; ++e) vc[slot * KD + vh * 128 + lane * 8 + e] = tobf(row[KD + vh * 128 + lane * 8 + e]);
            }
        });
    }
}
