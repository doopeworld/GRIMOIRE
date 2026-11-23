// SPDX-License-Identifier: GPL-3.0-or-later
// Native Inkling 2 Mini text inference on Intel GPUs (B70 / B60): one sequence, greedy.
// Op-by-op port of src/inkling2_text_ref.cpp (same BF16 rounding points).
//
//   inkling2_gpu MODEL_DIR check IDS_FILE CONTEXT NEW_TOKENS [REF_LAST_LOGITS_F32|-] [CHUNK]
//   inkling2_gpu MODEL_DIR bench IDS_FILE CONTEXT PP_LIST TG [CHUNK]
//
// One visit loop runs M rows: M = 1 is the decode step (recorded once as a SYCL graph and
// replayed), M > 1 a prompt chunk (Grimoire's XMX/oneDNN GEMM for the dense weights, experts
// grouped by rows through inkling2::launch_expert_ffn, a batched attention kernel).
// Weights stay in their checkpoint formats: BF16 dense, NVFP4 / MXFP8 experts.
#include "b70/inkling2.hpp"
#include "b70/formats.hpp"
#include "b70/weights.hpp"
#include "kernels.hpp"
#include "inkling2_sycl.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sycl_ext = sycl::ext::oneapi::experimental;
using bf16x = sycl::ext::oneapi::bfloat16;

template <int M> class InkGemv;
class InkMix; class InkRms; class InkGateCat; class InkGateBlend; class InkEmbed; class InkPost; class InkAtt1; class InkAtt2;
class InkAttB; class InkAtt3; class InkResid; class InkTopk; class InkMlpConv; class InkArg1; class InkArg2; class InkPos; class InkGu; class InkDown;
class InkHist; class InkGather; class InkCombine;

namespace ink {

using b70::inkling2::Checkpoint;
using b70::inkling2::Config;
using b70::inkling2::ExpertMatrix;
using b70::inkling2::ExpertScratch;
using b70::inkling2::PackedFormat;
using b70::inkling2::Visit;

constexpr int kMaxBankRows = 16;
constexpr int kMixRows = 8;

inline float bfr(float x) { return b70::bf16_to_f32(b70::f32_to_bf16(x)); }
inline float bfw(uint16_t w) { return b70::bits_to_f32(uint32_t(w) << 16); }
inline uint16_t tobf(float x) { return b70::f32_to_bf16(x).bits; }
inline float sigm(float x) { return x >= 0.f ? 1.f / (1.f + sycl::exp(-x)) : sycl::exp(x) / (1.f + sycl::exp(x)); }
// Round-to-nearest-even to BF16 for finite values (no NaN branch).
inline float bfrf(float x) { uint32_t u = b70::f32_to_bits(x); u += 0x7FFFu + ((u >> 16) & 1u); return b70::bits_to_f32(u & 0xFFFF0000u); }

struct Dev {
    sycl::queue& q;
    std::vector<void*> ptrs;
    size_t bytes = 0;
    explicit Dev(sycl::queue& q) : q(q) {}
    template <class T> T* alloc(size_t n) {
        T* p = sycl::malloc_device<T>(std::max<size_t>(n, 1), q);
        if (!p) throw std::runtime_error("device allocation failed");
        ptrs.push_back(p);
        bytes += n * sizeof(T);
        return p;
    }
    template <class T> T* zeros(size_t n) { T* p = alloc<T>(n); q.memset(p, 0, n * sizeof(T)); return p; }
    ~Dev() { q.wait(); for (auto* p : ptrs) sycl::free(p, q); }
};

struct Layer {
    bool sliding = true, loop = false;
    PackedFormat fgu{}, fdn{};
    uint16_t *Wp = nullptr, *Wo = nullptr, *Wr = nullptr;
    float* rbias = nullptr;
    uint16_t *nq = nullptr, *nk = nullptr, *nv = nullptr, *nm = nullptr;
    uint16_t *qs = nullptr, *rs = nullptr, *table = nullptr;
    int E = 0;
    uint16_t *kw = nullptr, *vw = nullptr, *aconv = nullptr, *mconv = nullptr;
    uint16_t *cdown[4]{}, *cgain[4]{}, *cforget[4]{};
    uint8_t *gu = nullptr, *gus = nullptr, *dn = nullptr, *dns = nullptr;
    float *gua = nullptr, *gud = nullptr, *dna = nullptr, *dnd = nullptr;
    const float *hgua = nullptr, *hgud = nullptr, *hdna = nullptr, *hdnd = nullptr;  // host copies (prompt path)
    float *aq = nullptr, *mq = nullptr;
    // projection segments: A (from xq) = [q | rel | sink (+pad) | gate], B (xk) = [k | kproj], C (xv) = [v | vproj]
    int NA = 0, NB = 0, NC = 0;
};
constexpr int kOQ = 0;

struct Slot {
    float *khist = nullptr, *vhist = nullptr, *ahist = nullptr, *mhist = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
    int cap = 0, win = 0;
};

struct MixJob { int rows[kMixRows]; int n; const float* query; int out; };
struct MixBatch { MixJob j[4]; int count; };
struct NormJob { int in; const uint16_t* w; int out; uint16_t* outb; };
struct NormBatch { NormJob j[3]; int count; };
struct GateJob { int cur, msg, out; const uint16_t* gain; const uint16_t* forget; };
struct GateBatch { GateJob j[3]; int count; };

enum Row { HID, RES, X0, X1, X2, XM, N0, N1, N2, NM, M0, M1, M2, M3, PRE, PREV, G0, G1, G2, GM, Z, YB, BANK0, NROWS = BANK0 + kMaxBankRows };

class Engine {
public:
    Engine(sycl::queue& q, const std::string& dir, int limit, int chunk)
        : q_(q), ck_(Checkpoint::open(dir)), c_(ck_.config), dev_(q), limit_(limit), mmax_(std::max(1, chunk)) {
        if (limit < 1 || limit > c_.max_context) throw std::invalid_argument("bad context");
        visits_ = c_.visits();
        h_ = c_.hidden;
        rs_ = size_t(mmax_) * h_;
        load();
    }
    const Config& config() const { return c_; }
    void reset() { pos_ = 0; q_.memset(pos_d_, 0, sizeof(int)).wait(); }
    int step(int token, std::vector<float>* logits_out = nullptr);
    int prefill(const std::vector<int>& ids, std::vector<float>* logits_out = nullptr);
    bool use_graph = true;

private:
    sycl::queue& q_;
    Checkpoint ck_;
    const Config& c_;
    Dev dev_;
    int limit_, mmax_, h_ = 0, pos_ = 0;
    size_t rs_ = 0;
    std::vector<Visit> visits_;
    std::vector<Layer> L_;
    std::vector<Slot> S_;
    uint16_t *embed_ = nullptr, *embed_norm_ = nullptr, *unembed_ = nullptr, *fnorm_ = nullptr;
    float *oquery_ = nullptr, *xq_ = nullptr, *xkv_ = nullptr;
    float *pool_ = nullptr, *projA_ = nullptr, *projB_ = nullptr, *projC_ = nullptr, *scores_ = nullptr, *aout_ = nullptr;
    float *att_stat_ = nullptr, *att_part_ = nullptr;
    float *xcat_ = nullptr, *low_ = nullptr, *rlog_ = nullptr, *wts_ = nullptr, *act_ = nullptr, *dbuf_ = nullptr, *logits_ = nullptr, *amx_v_ = nullptr;
    uint16_t *nb_ = nullptr, *nmb_ = nullptr, *aoutb_ = nullptr, *xcatb_ = nullptr, *inb_ = nullptr, *outb_ = nullptr;
    int *sel_ = nullptr, *tok_ = nullptr, *amx_i_ = nullptr, *next_ = nullptr, *pos_d_ = nullptr, *gather_ = nullptr, *pairpos_ = nullptr;
    ExpertScratch xs_{};
    int maxNA_ = 0, maxNB_ = 0;
    std::unique_ptr<sycl_ext::command_graph<sycl_ext::graph_state::executable>> gexec_;

    float* row(int r, int m = 0) const { return pool_ + size_t(r) * rs_ + size_t(m) * h_; }
    const uint8_t* host(const std::string& n, size_t* nb = nullptr) const {
        const auto* t = ck_.file->find(n);
        if (!t) throw std::runtime_error("missing tensor " + n);
        if (nb) *nb = size_t(t->end - t->begin);
        return static_cast<const uint8_t*>(ck_.file->data(*t));
    }
    template <class T> T* up(const std::string& n) {
        size_t nb;
        const auto* p = host(n, &nb);
        T* d = dev_.alloc<T>(nb / sizeof(T));
        q_.memcpy(d, p, nb);
        return d;
    }
    void load();
    void run(int M);
    void dense(const uint16_t* W, int N, int K, const float* x, size_t ldx, const uint16_t* xb, float* y, size_t ldy, int M, bool round);
    template <int MODE> void gemv(const uint16_t* W, int rows, int K, const float* x0, float* y0, int end0, const float* x1, float* y1, int end1,
                                  const float* x2, float* y2, float cap = 0.f);
    void mix(const MixBatch& b, int M, int m0 = 0);
    void rms(const NormBatch& b, int M, int m0 = 0);
    void gates(const GateBatch& b, const Layer& Ly, const int* ctrl, int M);
    void experts_decode(const Layer& Ly);
    void experts_prompt(const Layer& Ly, int M);
    void hist_update(float* hist, const float* src, size_t ld, int n, int M);
};

void Engine::load() {
    const std::string root = "model.llm.";
    const int Q = c_.local.heads * c_.local.head_dim, K = c_.local.kv_heads * c_.local.head_dim;
    if (c_.full.heads * c_.full.head_dim != Q || c_.full.kv_heads * c_.full.head_dim != K || c_.full.head_dim != 128 || c_.local.head_dim != 128 ||
        c_.full.heads != c_.local.heads || c_.rel_dim != 16 || c_.conv != 4 || c_.top_k != 8 || c_.experts > 128 || h_ % 256 || c_.inter % 128)
        throw std::runtime_error("geometry outside this port (head_dim 128, d_rel 16, conv 4, top-8, <= 128 experts)");
    const auto t0 = std::chrono::steady_clock::now();
    embed_ = up<uint16_t>(root + "embed.weight");
    embed_norm_ = up<uint16_t>(root + "embed_norm.weight");
    unembed_ = up<uint16_t>(root + "unembed.weight");
    fnorm_ = up<uint16_t>(root + "norm.weight");
    oquery_ = up<float>(root + "attn_res.output_query");
    xq_ = up<float>(root + "middle_cycle_cross_pass_queries");
    xkv_ = up<float>(root + "middle_cycle_cross_pass_kv_queries");
    L_.resize(c_.layers);
    std::vector<uint16_t> stage;
    for (int i = 0; i < c_.layers; ++i) {
        Layer& Ly = L_[i];
        const auto s = root + "layers." + std::to_string(i) + ".";
        const auto t = s + "attn.";
        Ly.sliding = c_.sliding[i];
        Ly.loop = i >= c_.loop_start && i < c_.loop_end;
        const auto& a = Ly.sliding ? c_.local : c_.full;
        const int H = a.heads, KVH = a.kv_heads, R = c_.rel_dim;
        struct Part { std::string name; int rows; int pad; };
        std::vector<Part> parts = {{t + "wq_du.weight", Q, 0}, {t + "wr_du.weight", H * R, 0}, {t + "attn_sink_proj.weight", H, 32 - H},
                                   {t + "gate_attn.weight", Q, 0}, {t + "wk_dv.weight", K, 0}};
        if (!Ly.sliding) parts.push_back({t + "k_sconv.weight_proj", KVH * c_.conv, 0});
        parts.push_back({t + "wv_dv.weight", K, 0});
        if (!Ly.sliding) parts.push_back({t + "v_sconv.weight_proj", KVH * c_.conv, 0});
        Ly.NA = Q + H * R + 32 + Q;
        Ly.NB = K + (Ly.sliding ? 0 : KVH * c_.conv);
        Ly.NC = Ly.NB;
        if (H > 32 || Ly.NA % 16 || Ly.NB % 16) throw std::runtime_error("projection padding");
        stage.assign(size_t(Ly.NA + Ly.NB + Ly.NC) * h_, 0);
        int at = 0;
        for (auto& p : parts) {
            size_t nb;
            const auto* src = host(p.name, &nb);
            if (nb != size_t(p.rows) * h_ * 2) throw std::runtime_error("unexpected size " + p.name);
            std::memcpy(stage.data() + size_t(at) * h_, src, nb);
            at += p.rows + p.pad;
        }
        if (at != Ly.NA + Ly.NB + Ly.NC) throw std::runtime_error("projection layout");
        maxNA_ = std::max(maxNA_, Ly.NA); maxNB_ = std::max(maxNB_, Ly.NB);
        Ly.Wp = dev_.alloc<uint16_t>(stage.size());
        q_.memcpy(Ly.Wp, stage.data(), stage.size() * 2).wait();
        Ly.Wo = up<uint16_t>(t + "wo_ud.weight");
        Ly.Wr = up<uint16_t>(s + "mlp.gate.weight");
        Ly.rbias = up<float>(s + "mlp.gate.bias");
        if (Ly.sliding) { Ly.nq = Ly.nk = Ly.nv = up<uint16_t>(s + "attn_norm.weight"); }
        else { Ly.nq = up<uint16_t>(s + "q_input_norm.weight"); Ly.nk = up<uint16_t>(s + "k_input_norm.weight"); Ly.nv = up<uint16_t>(s + "v_input_norm.weight"); }
        Ly.nm = up<uint16_t>(s + "mlp_norm.weight");
        Ly.qs = up<uint16_t>(t + "q_l2_clip.scale");
        Ly.rs = up<uint16_t>(t + "rel_l2_clip.scale");
        Ly.table = up<uint16_t>(t + "rel_logits_proj.proj");
        Ly.E = Ly.sliding ? c_.window : c_.rel_extent;
        Ly.kw = up<uint16_t>(t + (Ly.sliding ? "k_sconv.weight" : "k_sconv.static_w"));
        Ly.vw = up<uint16_t>(t + (Ly.sliding ? "v_sconv.weight" : "v_sconv.static_w"));
        Ly.aconv = up<uint16_t>(s + "attn_sconv.weight");
        Ly.mconv = up<uint16_t>(s + "mlp_sconv.weight");
        if (Ly.loop) {
            const char* names[4] = {"attn", "k", "v", "mlp"};
            for (int j = 0; j < 4; ++j) {
                const auto p = s + "controllers." + names[j] + ".";
                Ly.cdown[j] = up<uint16_t>(p + "down_weight");
                Ly.cgain[j] = up<uint16_t>(p + "gain_weight");
                Ly.cforget[j] = up<uint16_t>(p + "forget_weight");
            }
        }
        Ly.fgu = ck_.layout[i].gate_up;
        Ly.fdn = ck_.layout[i].down;
        const auto e = s + "mlp.experts.";
        Ly.gu = up<uint8_t>(e + "w13_weight"); Ly.gus = up<uint8_t>(e + "w13_weight_scale");
        Ly.dn = up<uint8_t>(e + "w2_weight"); Ly.dns = up<uint8_t>(e + "w2_weight_scale");
        if (Ly.fgu == PackedFormat::NVFP4) {
            Ly.gua = up<float>(e + "w13_weight_amax"); Ly.gud = up<float>(e + "w13_weight_outer_scale_denominator");
            Ly.hgua = reinterpret_cast<const float*>(host(e + "w13_weight_amax"));
            Ly.hgud = reinterpret_cast<const float*>(host(e + "w13_weight_outer_scale_denominator"));
        }
        if (Ly.fdn == PackedFormat::NVFP4) {
            Ly.dna = up<float>(e + "w2_weight_amax"); Ly.dnd = up<float>(e + "w2_weight_outer_scale_denominator");
            Ly.hdna = reinterpret_cast<const float*>(host(e + "w2_weight_amax"));
            Ly.hdnd = reinterpret_cast<const float*>(host(e + "w2_weight_outer_scale_denominator"));
        }
        if (i) Ly.aq = up<float>(root + "attn_res.attn_queries." + std::to_string(i - 1));
        Ly.mq = up<float>(root + "attn_res.mlp_queries." + std::to_string(i));
        q_.wait();
    }
    S_.resize(c_.cache_slots());
    for (auto& v : visits_) {
        Slot& sl = S_[v.cache_slot];
        if (sl.cap) continue;
        // A sliding ring must also hold the rows of a prompt chunk written before they are read.
        sl.win = v.sliding ? std::min(limit_, c_.window) : limit_;
        sl.cap = v.sliding ? std::min(limit_, c_.window + mmax_) : limit_;
        sl.khist = dev_.zeros<float>(3 * size_t(K)); sl.vhist = dev_.zeros<float>(3 * size_t(K));
        sl.ahist = dev_.zeros<float>(3 * size_t(h_)); sl.mhist = dev_.zeros<float>(3 * size_t(h_));
        sl.kc = dev_.zeros<uint16_t>(size_t(sl.cap) * K); sl.vc = dev_.zeros<uint16_t>(size_t(sl.cap) * K);
    }
    const size_t mm = size_t(mmax_), TK = size_t(c_.top_k);
    pool_ = dev_.zeros<float>(size_t(NROWS) * rs_);
    projA_ = dev_.zeros<float>(mm * maxNA_); projB_ = dev_.zeros<float>(mm * maxNB_); projC_ = dev_.zeros<float>(mm * maxNB_);
    scores_ = dev_.zeros<float>(size_t(c_.local.heads) * limit_);
    {
        const size_t nch = (size_t(limit_) + 127) / 128;
        att_stat_ = dev_.zeros<float>(size_t(c_.local.heads) * nch * 2);
        att_part_ = dev_.zeros<float>(size_t(c_.local.heads) * nch * 128);
    }
    aout_ = dev_.zeros<float>(mm * Q); aoutb_ = dev_.zeros<uint16_t>(mm * Q);
    xcat_ = dev_.zeros<float>(3 * mm * 2 * h_); xcatb_ = dev_.zeros<uint16_t>(3 * mm * 2 * h_);
    low_ = dev_.zeros<float>(3 * mm * c_.gate_rank);
    nb_ = dev_.zeros<uint16_t>(3 * mm * h_); nmb_ = dev_.zeros<uint16_t>(mm * h_);
    rlog_ = dev_.zeros<float>(mm * c_.experts);
    sel_ = dev_.zeros<int>(mm * TK); wts_ = dev_.zeros<float>(mm * TK);
    act_ = dev_.zeros<float>(TK * c_.inter);
    dbuf_ = dev_.zeros<float>(TK * h_);
    logits_ = dev_.zeros<float>(c_.output_vocab);
    amx_v_ = dev_.zeros<float>(4096); amx_i_ = dev_.zeros<int>(4096);
    tok_ = dev_.zeros<int>(mm); next_ = dev_.zeros<int>(1); pos_d_ = dev_.zeros<int>(1);
    if (mmax_ > 1) {
        inb_ = dev_.zeros<uint16_t>(mm * TK * h_); outb_ = dev_.zeros<uint16_t>(mm * TK * h_);
        gather_ = dev_.zeros<int>(mm * TK); pairpos_ = dev_.zeros<int>(mm * TK);
        const auto sz = b70::inkling2::expert_scratch_sizes(mmax_, h_, c_.inter);
        xs_ = ExpertScratch{dev_.alloc<uint32_t>(sz.weight_pairs), dev_.alloc<float>(sz.projection), dev_.alloc<bf16x>(sz.activation)};
    }
    q_.wait();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("loaded %d layers, %d visits, %.2f GiB on the device in %.1f s (prompt chunk %d rows)\n", c_.layers, int(visits_.size()),
                dev_.bytes / 1073741824.0, s, mmax_);
}

template <int MODE>
void Engine::gemv(const uint16_t* W, int rows, int K, const float* x0, float* y0, int end0, const float* x1, float* y1, int end1,
                  const float* x2, float* y2, float cap) {
    // Two rows per sub-group (segment ends are even, so both rows read the same input).
    const size_t pairs = (size_t(rows) + 1) / 2, groups = (pairs + 7) / 8;
    q_.parallel_for<InkGemv<MODE>>(sycl::nd_range<1>(groups * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const int r = 2 * (int(it.get_group(0)) * 8 + int(sg.get_group_linear_id()));
        if (r >= rows) return;
        const bool two = r + 1 < rows;
        const float* x = r < end0 ? x0 : (r < end1 ? x1 : x2);
        const uint16_t* w0 = W + size_t(r) * K;
        const uint16_t* w1 = two ? w0 + K : w0;
        const int l = int(sg.get_local_linear_id());
        float s0 = 0.f, s1 = 0.f;
        for (int k = l * 8; k < K; k += 128) {
            const sycl::vec<uint16_t, 8> u0 = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(w0 + k);
            const sycl::vec<uint16_t, 8> u1 = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(w1 + k);
            const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(x + k);
            const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(x + k + 4);
            s0 = sycl::fma(bfw(u0[0]), a[0], s0); s0 = sycl::fma(bfw(u0[1]), a[1], s0); s0 = sycl::fma(bfw(u0[2]), a[2], s0); s0 = sycl::fma(bfw(u0[3]), a[3], s0);
            s0 = sycl::fma(bfw(u0[4]), b[0], s0); s0 = sycl::fma(bfw(u0[5]), b[1], s0); s0 = sycl::fma(bfw(u0[6]), b[2], s0); s0 = sycl::fma(bfw(u0[7]), b[3], s0);
            s1 = sycl::fma(bfw(u1[0]), a[0], s1); s1 = sycl::fma(bfw(u1[1]), a[1], s1); s1 = sycl::fma(bfw(u1[2]), a[2], s1); s1 = sycl::fma(bfw(u1[3]), a[3], s1);
            s1 = sycl::fma(bfw(u1[4]), b[0], s1); s1 = sycl::fma(bfw(u1[5]), b[1], s1); s1 = sycl::fma(bfw(u1[6]), b[2], s1); s1 = sycl::fma(bfw(u1[7]), b[3], s1);
        }
        s0 = sycl::reduce_over_group(sg, s0, sycl::plus<float>());
        s1 = sycl::reduce_over_group(sg, s1, sycl::plus<float>());
        if (l < 2 && (l == 0 || two)) {
            const float s = l == 0 ? s0 : s1;
            const int rr = r + l;
            float* y = rr < end0 ? y0 + rr : (rr < end1 ? y1 + (rr - end0) : y2 + (rr - end1));
            if constexpr (MODE == 0) *y = bfr(s);
            else if constexpr (MODE == 1) *y = s;
            else { const float lin = bfr(s); *y = cap * sycl::tanh(lin / cap); }
        }
    });
}

void Engine::dense(const uint16_t* W, int N, int K, const float* x, size_t ldx, const uint16_t* xb, float* y, size_t ldy, int M, bool round) {
    if (M >= 32 && xb) {
        b70::QuantWeight w;
        w.fmt = b70::Fmt::BF16; w.N = N; w.K = K; w.payload = reinterpret_cast<const uint8_t*>(W); w.row_bytes = int64_t(K) * 2;
        if (!b70::gemm_fast_supported(w, M) || ldy != size_t(N)) throw std::runtime_error("GEMM shape not supported");
        b70::launch_gemm_fast(q_, w, reinterpret_cast<const bf16x*>(xb), y, M);
        return;
    }
    for (int m = 0; m < M; ++m) {
        if (round) gemv<0>(W, N, K, x + m * ldx, y + m * ldy, N, nullptr, nullptr, N, nullptr, nullptr);
        else gemv<1>(W, N, K, x + m * ldx, y + m * ldy, N, nullptr, nullptr, N, nullptr, nullptr);
    }
}

void Engine::mix(const MixBatch& b, int M, int m0) {
    float* pool = pool_;
    const int h = h_;
    const size_t RS = rs_;
    const float f = 1.f / std::sqrt(float(h));
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(16 * 2 * kMixRows), cgh);
        sycl::local_accessor<float, 1> tot(sycl::range<1>(2 * kMixRows), cgh);
        cgh.parallel_for<InkMix>(sycl::nd_range<1>(size_t(b.count) * M * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int gi = int(it.get_group(0));
            const MixJob& J = b.j[gi / M];
            const size_t mo = size_t(m0 + gi % M) * h;
            const int lid = int(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
            const int n = J.n;
            float ss[kMixRows], qd[kMixRows];
#pragma unroll
            for (int r = 0; r < kMixRows; ++r) { ss[r] = 0.f; qd[r] = 0.f; }
            for (int i = lid; i < h; i += 256) {
                const float qv = J.query[i];
#pragma unroll
                for (int r = 0; r < kMixRows; ++r)
                    if (r < n) { const float v = pool[size_t(J.rows[r]) * RS + mo + i]; ss[r] = sycl::fma(v, v, ss[r]); qd[r] = sycl::fma(qv, v, qd[r]); }
            }
#pragma unroll
            for (int r = 0; r < kMixRows; ++r)
                if (r < n) {
                    const float a = sycl::reduce_over_group(sg, ss[r], sycl::plus<float>());
                    const float c = sycl::reduce_over_group(sg, qd[r], sycl::plus<float>());
                    if (lane == 0) { red[sgid * 2 * kMixRows + r] = a; red[sgid * 2 * kMixRows + kMixRows + r] = c; }
                }
            sycl::group_barrier(it.get_group());
            if (lid < 2 * kMixRows) { float t = 0.f; for (int g = 0; g < 16; ++g) t += red[g * 2 * kMixRows + lid]; tot[lid] = t; }
            sycl::group_barrier(it.get_group());
            float inv[kMixRows], sc[kMixRows];
            float high = -INFINITY;
#pragma unroll
            for (int r = 0; r < kMixRows; ++r)
                if (r < n) { inv[r] = 1.f / sycl::sqrt(tot[r] / float(h) + 1e-6f); sc[r] = tot[kMixRows + r] * inv[r] * f; high = sycl::fmax(high, sc[r]); }
            float sum = 0.f;
#pragma unroll
            for (int r = 0; r < kMixRows; ++r)
                if (r < n) { sc[r] = sycl::exp(sc[r] - high); sum += sc[r]; }
            float* out = pool + size_t(J.out) * RS + mo;
            for (int i = lid; i < h; i += 256) {
                float o = 0.f;
#pragma unroll
                for (int r = 0; r < kMixRows; ++r)
                    if (r < n) o += (pool[size_t(J.rows[r]) * RS + mo + i] * inv[r]) * (sc[r] / sum);
                out[i] = o;
            }
        });
    });
}

void Engine::rms(const NormBatch& b, int M, int m0) {
    float* pool = pool_;
    const int h = h_;
    const size_t RS = rs_;
    const float eps = c_.eps;
    q_.parallel_for<InkRms>(sycl::nd_range<1>(size_t(b.count) * M * 256, 256), [=](sycl::nd_item<1> it) {
        const int gi = int(it.get_group(0));
        const NormJob& J = b.j[gi / M];
        const int m = m0 + gi % M;
        const int lid = int(it.get_local_id(0));
        const float* x = pool + size_t(J.in) * RS + size_t(m) * h;
        float ss = 0.f;
        for (int i = lid; i < h; i += 256) ss += x[i] * x[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.f / sycl::sqrt(ss / float(h) + eps);
        float* y = pool + size_t(J.out) * RS + size_t(m) * h;
        for (int i = lid; i < h; i += 256) {
            const float v = bfr((x[i] * r) * bfw(J.w[i]));
            y[i] = v;
            if (J.outb) J.outb[size_t(m) * h + i] = tobf(v);
        }
    });
}

// gate(cur, msg): low = dense([norm(cur), norm(msg)], down, rank); out = (1 - tanh(forget.low)) cur + tanh(gain.low) msg
void Engine::gates(const GateBatch& b, const Layer& Ly, const int* ctrl, int M) {
    float* pool = pool_;
    float* xcat = xcat_;
    uint16_t* xcatb = xcatb_;
    float* low = low_;
    const int h = h_, rank = c_.gate_rank, MM = mmax_;
    const size_t RS = rs_;
    const float eps = c_.eps;
    q_.parallel_for<InkGateCat>(sycl::nd_range<1>(size_t(b.count) * M * 256, 256), [=](sycl::nd_item<1> it) {
        const int gi = int(it.get_group(0)), jb = gi / M, m = gi % M;
        const GateJob& J = b.j[jb];
        const int lid = int(it.get_local_id(0));
        const float* cur = pool + size_t(J.cur) * RS + size_t(m) * h;
        const float* msg = pool + size_t(J.msg) * RS + size_t(m) * h;
        float sc = 0.f, sm = 0.f;
        for (int i = lid; i < h; i += 256) { sc += cur[i] * cur[i]; sm += msg[i] * msg[i]; }
        sc = sycl::reduce_over_group(it.get_group(), sc, sycl::plus<float>());
        sm = sycl::reduce_over_group(it.get_group(), sm, sycl::plus<float>());
        const float ir = 1.f / sycl::sqrt(sc / float(h) + eps), im = 1.f / sycl::sqrt(sm / float(h) + eps);
        const size_t o = (size_t(jb) * MM + m) * 2 * h;
        for (int i = lid; i < h; i += 256) {
            const float a = bfr(cur[i] * ir), c = bfr(msg[i] * im);
            xcat[o + i] = a; xcat[o + h + i] = c;
            xcatb[o + i] = tobf(a); xcatb[o + h + i] = tobf(c);
        }
    });
    for (int j = 0; j < b.count; ++j)
        dense(Ly.cdown[ctrl[j]], rank, 2 * h, xcat + size_t(j) * MM * 2 * h, 2 * h, xcatb + size_t(j) * MM * 2 * h, low + size_t(j) * MM * rank, rank, M, true);
    const size_t total = size_t(b.count) * M * h, groups = (total + 7) / 8;
    q_.parallel_for<InkGateBlend>(sycl::nd_range<1>(groups * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const size_t t = it.get_group(0) * 8 + sg.get_group_linear_id();
        if (t >= total) return;
        const int i = int(t % h), m = int((t / h) % M), jb = int(t / (size_t(h) * M));
        const GateJob& J = b.j[jb];
        const int l = int(sg.get_local_linear_id());
        const float* lw = low + (size_t(jb) * MM + m) * rank;
        float gs = 0.f, fs = 0.f;
        for (int r = l; r < rank; r += 16) { const float lv = bfr(lw[r]); gs += lv * bfw(J.gain[size_t(i) * rank + r]); fs += lv * bfw(J.forget[size_t(i) * rank + r]); }
        gs = sycl::reduce_over_group(sg, gs, sycl::plus<float>());
        fs = sycl::reduce_over_group(sg, fs, sycl::plus<float>());
        if (l == 0) {
            const float c = pool[size_t(J.cur) * RS + size_t(m) * h + i], mm = pool[size_t(J.msg) * RS + size_t(m) * h + i];
            pool[size_t(J.out) * RS + size_t(m) * h + i] = (1.f - sycl::tanh(fs)) * c + sycl::tanh(gs) * mm;
        }
    });
}

// ---- routed experts, one decode row.  Same design as Grimoire's tiered NVFP4 MoE (the Ornith /
// Flash-Next fix): decode tables in SLM, one lookup + one FMA per weight, block sums scaled once.
// Inkling rounds every NVFP4 weight to BF16 after its outer scale: value(n) = bfr((e2m1(n)*sc)*outer).
// E2M1 magnitudes are 2^k (codes 1,2,4,6) or 1.5*2^k (codes 3,5,7) and every product before the
// outer multiply is exact, so value(n) = sign * 2^k * q with q1 = bfr(sc*outer) or
// q3 = bfr((1.5*sc)*outer): the same BF16 weights as the reference, summed in a different order.
// MXFP8 (E4M3 x E8M0) weights are exact in BF16 already (3 mantissa bits times a power of two).
inline void fill_tables(float* e4, float* wp, float* wt, int lid, int lsz) {
    for (int b = lid; b < 256; b += lsz) e4[b] = b70::e4m3_to_f32(uint8_t(b));
    if (lid < 16) {
        const int c = lid & 7;
        const float sgn = (lid & 8) ? -1.f : 1.f;
        const float mag = c == 0 ? 0.f : (c == 1 ? 0.5f : (c <= 3 ? 1.f : (c <= 5 ? 2.f : 4.f)));
        const bool t = c == 3 || c == 5 || c == 7;
        wp[lid] = t ? 0.f : sgn * mag;
        wt[lid] = t ? sgn * mag : 0.f;
    }
}
inline void nv8(uint32_t w, const float* xr, const float* wp, const float* wt, float& A, float& B) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const uint32_t n = (w >> (4 * t)) & 15u;
        A = sycl::fma(wp[n], xr[t], A);
        B = sycl::fma(wt[n], xr[t], B);
    }
}
inline float mx4(uint32_t w, const float* xr, const float* e4) {
    float a = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) a = sycl::fma(e4[(w >> (8 * t)) & 255u], xr[t], a);
    return a;
}
inline void load_x(const float* g, float* xr, int n) {
    for (int i = 0; i < n; i += 4) {
        const sycl::float4 v = *reinterpret_cast<const sycl::float4*>(g + i);
        xr[i] = v[0]; xr[i + 1] = v[1]; xr[i + 2] = v[2]; xr[i + 3] = v[3];
    }
}

void Engine::experts_decode(const Layer& Ly) {
    const int h = h_, I = c_.inter, TK = c_.top_k;
    const float gcap = c_.gate_cap, ucap = c_.up_cap;
    const int* sel = sel_;
    float* act = act_;
    float* dbuf = dbuf_;
    const float* x = row(NM);
    constexpr int R = 2, SGS = 8;
    {
        const bool NV = Ly.fgu == PackedFormat::NVFP4;
        const uint8_t *W = Ly.gu, *S = Ly.gus;
        const float *amax = Ly.gua, *den = Ly.gud;
        const size_t rb = NV ? h / 2 : h, sb = NV ? h / 16 : h / 32;
        const int steps = h / 256, wpb = I / (R * SGS);
        q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> e4s(256, cgh), wps(16, cgh), wts(16, cgh);
            cgh.parallel_for<InkGu>(sycl::nd_range<1>(size_t(TK) * wpb * SGS * 16, SGS * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid = int(it.get_local_id(0));
                float* e4 = e4s.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wp = wps.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wt = wts.template get_multi_ptr<sycl::access::decorated::no>().get();
                fill_tables(e4, wp, wt, lid, SGS * 16);
                sycl::group_barrier(it.get_group());
                auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_linear_id());
                const int s = int(it.get_group(0)) / wpb;
                const int i0 = (int(it.get_group(0)) % wpb) * R * SGS + int(sg.get_group_linear_id()) * R;
                const int e = sel[s];
                const float outer = NV ? amax[e] / den[0] : 1.f;
                float g[R], u[R];
#pragma unroll
                for (int r = 0; r < R; ++r) { g[r] = 0.f; u[r] = 0.f; }
                for (int k = 0; k < steps; ++k) {
                    const int blk = k * 16 + lane;
                    float xr[16];
                    load_x(x + size_t(blk) * 16, xr, 16);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const size_t gr = size_t(e) * 2 * I + 2 * size_t(i0 + r), ur = gr + 1;
                        if (NV) {
                            const sycl::uint2 pg = *reinterpret_cast<const sycl::uint2*>(W + gr * rb + size_t(blk) * 8);
                            const sycl::uint2 pu = *reinterpret_cast<const sycl::uint2*>(W + ur * rb + size_t(blk) * 8);
                            float ag = 0.f, bg = 0.f, au = 0.f, bu = 0.f;
                            nv8(pg[0], xr, wp, wt, ag, bg); nv8(pg[1], xr + 8, wp, wt, ag, bg);
                            nv8(pu[0], xr, wp, wt, au, bu); nv8(pu[1], xr + 8, wp, wt, au, bu);
                            const float sgc = e4[S[gr * sb + blk]], suc = e4[S[ur * sb + blk]];
                            g[r] += bfrf(sgc * outer) * ag + bfrf((1.5f * sgc) * outer) * bg;
                            u[r] += bfrf(suc * outer) * au + bfrf((1.5f * suc) * outer) * bu;
                        } else {
                            const sycl::uint4 pg = *reinterpret_cast<const sycl::uint4*>(W + gr * rb + size_t(blk) * 16);
                            const sycl::uint4 pu = *reinterpret_cast<const sycl::uint4*>(W + ur * rb + size_t(blk) * 16);
                            const float ag = mx4(pg[0], xr, e4) + mx4(pg[1], xr + 4, e4) + mx4(pg[2], xr + 8, e4) + mx4(pg[3], xr + 12, e4);
                            const float au = mx4(pu[0], xr, e4) + mx4(pu[1], xr + 4, e4) + mx4(pu[2], xr + 8, e4) + mx4(pu[3], xr + 12, e4);
                            g[r] += ag * b70::e8m0_to_f32(S[gr * sb + blk / 2]);
                            u[r] += au * b70::e8m0_to_f32(S[ur * sb + blk / 2]);
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float gg = sycl::reduce_over_group(sg, g[r], sycl::plus<float>());
                    float uu = sycl::reduce_over_group(sg, u[r], sycl::plus<float>());
                    if (lane == 0) {
                        gg = bfrf(gg); uu = bfrf(uu);
                        const float sw = (gcap * sycl::tanh(gg / gcap)) * sigm(gg) * (ucap * sycl::tanh(uu / ucap));
                        act[size_t(s) * I + i0 + r] = bfrf(sw);
                    }
                }
            });
        });
    }
    {
        const bool NV = Ly.fdn == PackedFormat::NVFP4;
        const uint8_t *W = Ly.dn, *S = Ly.dns;
        const float *amax = Ly.dna, *den = Ly.dnd;
        const size_t rb = NV ? I / 2 : I, sb = NV ? I / 16 : I / 32;
        const int steps = I / 128, wpb = h / (R * SGS);
        q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> e4s(256, cgh), wps(16, cgh), wts(16, cgh);
            cgh.parallel_for<InkDown>(sycl::nd_range<1>(size_t(TK) * wpb * SGS * 16, SGS * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid = int(it.get_local_id(0));
                float* e4 = e4s.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wp = wps.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wt = wts.template get_multi_ptr<sycl::access::decorated::no>().get();
                fill_tables(e4, wp, wt, lid, SGS * 16);
                sycl::group_barrier(it.get_group());
                auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_linear_id());
                const int s = int(it.get_group(0)) / wpb;
                const int j0 = (int(it.get_group(0)) % wpb) * R * SGS + int(sg.get_group_linear_id()) * R;
                const int e = sel[s];
                const float outer = NV ? amax[e] / den[0] : 1.f;
                const float* a = act + size_t(s) * I;
                float d[R];
#pragma unroll
                for (int r = 0; r < R; ++r) d[r] = 0.f;
                for (int k = 0; k < steps; ++k) {
                    const int c = k * 16 + lane;
                    float xr[8];
                    load_x(a + size_t(c) * 8, xr, 8);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const size_t rw = size_t(e) * h + j0 + r;
                        if (NV) {
                            const uint32_t pw = *reinterpret_cast<const uint32_t*>(W + rw * rb + size_t(c) * 4);
                            float A = 0.f, B = 0.f;
                            nv8(pw, xr, wp, wt, A, B);
                            const float sc = e4[S[rw * sb + c / 2]];
                            d[r] += bfrf(sc * outer) * A + bfrf((1.5f * sc) * outer) * B;
                        } else {
                            const sycl::uint2 pw = *reinterpret_cast<const sycl::uint2*>(W + rw * rb + size_t(c) * 8);
                            d[r] += (mx4(pw[0], xr, e4) + mx4(pw[1], xr + 4, e4)) * b70::e8m0_to_f32(S[rw * sb + c / 4]);
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float dd = sycl::reduce_over_group(sg, d[r], sycl::plus<float>());
                    if (lane == 0) dbuf[size_t(s) * h + j0 + r] = bfrf(dd);
                }
            });
        });
    }
}

// Prompt rows: group (row, slot) pairs by expert, run each used expert once over its rows with the
// PR's exact launch_expert_ffn (BF16 VNNI expansion + XMX GEMM), then combine per row in slot order.
void Engine::experts_prompt(const Layer& Ly, int M) {
    const int h = h_, I = c_.inter, TK = c_.top_k, E = c_.experts;
    std::vector<int> hsel(size_t(M) * TK);
    q_.memcpy(hsel.data(), sel_, hsel.size() * sizeof(int)).wait();
    std::vector<int> count(E, 0), off(E + 1, 0), gat(size_t(M) * TK), ppos(size_t(M) * TK);
    for (int v : hsel) count[v]++;
    for (int e = 0; e < E; ++e) off[e + 1] = off[e] + count[e];
    std::vector<int> fill(off.begin(), off.end() - 1);
    for (int m = 0; m < M; ++m)
        for (int s = 0; s < TK; ++s) { const int e = hsel[size_t(m) * TK + s]; const int p = fill[e]++; gat[p] = m; ppos[size_t(m) * TK + s] = p; }
    q_.memcpy(gather_, gat.data(), gat.size() * sizeof(int));
    q_.memcpy(pairpos_, ppos.data(), ppos.size() * sizeof(int));
    {
        const uint16_t* nmb = nmb_;
        uint16_t* inb = inb_;
        const int* g = gather_;
        const size_t total = size_t(M) * TK * h;
        q_.parallel_for<InkGather>(sycl::range<1>(total), [=](sycl::id<1> id) {
            const size_t t = id[0], p = t / h, ch = t % h;
            inb[t] = nmb[size_t(g[p]) * h + ch];
        });
    }
    const size_t rbg = Ly.fgu == PackedFormat::NVFP4 ? h / 2 : h, sbg = Ly.fgu == PackedFormat::NVFP4 ? h / 16 : h / 32;
    const size_t rbd = Ly.fdn == PackedFormat::NVFP4 ? I / 2 : I, sbd = Ly.fdn == PackedFormat::NVFP4 ? I / 16 : I / 32;
    for (int e = 0; e < E; ++e) {
        if (!count[e]) continue;
        const float og = Ly.fgu == PackedFormat::NVFP4 ? Ly.hgua[e] / Ly.hgud[0] : 1.f;
        const float od = Ly.fdn == PackedFormat::NVFP4 ? Ly.hdna[e] / Ly.hdnd[0] : 1.f;
        const ExpertMatrix gu{Ly.fgu, 2 * I, h, Ly.gu + size_t(e) * 2 * I * rbg, Ly.gus + size_t(e) * 2 * I * sbg, og};
        const ExpertMatrix dn{Ly.fdn, h, I, Ly.dn + size_t(e) * h * rbd, Ly.dns + size_t(e) * h * sbd, od};
        b70::inkling2::launch_expert_ffn(q_, gu, dn, count[e], reinterpret_cast<const bf16x*>(inb_) + size_t(off[e]) * h,
                                         reinterpret_cast<bf16x*>(outb_) + size_t(off[e]) * h, c_.gate_cap, c_.up_cap, xs_);
    }
    {
        const uint16_t* outb = outb_;
        const int* pp = pairpos_;
        const float* w = wts_;
        float* yb = row(YB);
        const size_t total = size_t(M) * h;
        q_.parallel_for<InkCombine>(sycl::range<1>(total), [=](sycl::id<1> id) {
            const size_t t = id[0], m = t / h, ch = t % h;
            float acc = 0.f;
            for (int s = 0; s < TK; ++s) acc += w[m * TK + s] * bfw(outb[size_t(pp[m * TK + s]) * h + ch]);
            yb[t] = bfr(acc);
        });
    }
}

void Engine::hist_update(float* hist, const float* src, size_t ld, int n, int M) {
    const int* posd = pos_d_;
    const int lo = std::max(0, M - 3);
    q_.parallel_for<InkHist>(sycl::range<1>(size_t(M - lo) * n), [=](sycl::id<1> id) {
        const int m = lo + int(id[0] / n), ch = int(id[0] % n);
        hist[size_t((*posd + m) % 3) * n + ch] = bfr(src[size_t(m) * ld + ch]);
    });
}

int Engine::step(int token, std::vector<float>* logits_out) {
    if (pos_ >= limit_) throw std::runtime_error("context full");
    q_.memcpy(tok_, &token, sizeof(int));
    if (use_graph && !gexec_) {
        sycl_ext::command_graph<sycl_ext::graph_state::modifiable> g(q_.get_context(), q_.get_device());
        g.begin_recording(q_);
        run(1);
        g.end_recording(q_);
        gexec_ = std::make_unique<sycl_ext::command_graph<sycl_ext::graph_state::executable>>(g.finalize());
    }
    if (gexec_) q_.ext_oneapi_graph(*gexec_);
    else run(1);
    int next = -1;
    if (logits_out) { logits_out->resize(c_.output_vocab); q_.memcpy(logits_out->data(), logits_, size_t(c_.output_vocab) * sizeof(float)); }
    q_.memcpy(&next, next_, sizeof(int)).wait();
    ++pos_;
    return next;
}

int Engine::prefill(const std::vector<int>& ids, std::vector<float>* logits_out) {
    if (pos_ + int(ids.size()) > limit_) throw std::runtime_error("context full");
    size_t at = 0;
    while (at < ids.size()) {
        const int M = int(std::min<size_t>(size_t(mmax_), ids.size() - at));
        q_.memcpy(tok_, ids.data() + at, size_t(M) * sizeof(int));
        if (M == 1 && gexec_) q_.ext_oneapi_graph(*gexec_);
        else run(M);
        pos_ += M;
        at += M;
    }
    int next = -1;
    if (logits_out) { logits_out->resize(c_.output_vocab); q_.memcpy(logits_out->data(), logits_, size_t(c_.output_vocab) * sizeof(float)); }
    q_.memcpy(&next, next_, sizeof(int)).wait();
    return next;
}

// M rows at positions *pos_d .. *pos_d + M - 1; advances *pos_d by M.  M == 1 has no host value that
// depends on the position, so it is recorded once and replayed.
void Engine::run(int M) {
    const int h = h_, MM = mmax_;
    const int* posd = pos_d_;
    float* pool = pool_;
    const size_t RS = rs_;
    const float eps = c_.eps;
    {   // hidden = rms(embed[token], embed_norm)
        const uint16_t *Em = embed_, *w = embed_norm_;
        const int* tok = tok_;
        q_.parallel_for<InkEmbed>(sycl::nd_range<1>(size_t(M) * 256, 256), [=](sycl::nd_item<1> it) {
            const int m = int(it.get_group(0)), lid = int(it.get_local_id(0));
            const uint16_t* e = Em + size_t(tok[m]) * h;
            float ss = 0.f;
            for (int i = lid; i < h; i += 256) { const float v = bfw(e[i]); ss += v * v; }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float r = 1.f / sycl::sqrt(ss / float(h) + eps);
            for (int i = lid; i < h; i += 256) pool[size_t(HID) * RS + size_t(m) * h + i] = bfr((bfw(e[i]) * r) * bfw(w[i]));
        });
    }
    auto copy = [&](int from, int to) { q_.memcpy(row(to), row(from), size_t(M) * h * sizeof(float)); };
    std::vector<int> bank, pre_bank, prev_bank;
    int next_bank = 0;
    const float conv_static = 1.f / std::sqrt(float(1 + c_.conv)), conv_dyn = 1.f / std::sqrt(float(1 + c_.conv * 2));
    auto job = [&](const std::vector<int>& b, int partial, const float* query, int out) {
        MixJob J{};
        if (b.size() + 1 > size_t(kMixRows)) throw std::runtime_error("bank too deep");
        for (size_t r = 0; r < b.size(); ++r) J.rows[r] = b[r];
        J.rows[b.size()] = partial;
        J.n = int(b.size()) + 1;
        J.query = query;
        J.out = out;
        return J;
    };
    for (const auto& v : visits_) {
        const int i = v.physical;
        const Layer& Ly = L_[i];
        const Slot& sl = S_[v.cache_slot];
        const auto& a = v.sliding ? c_.local : c_.full;
        const int H = a.heads, KVH = a.kv_heads, D = a.head_dim, K = KVH * D, R = c_.rel_dim, Q = H * D;
        if (i == c_.loop_start && v.pass == 0) { copy(HID, PRE); pre_bank = bank; }
        if (i == c_.loop_start && v.has_message) {
            copy(HID, PREV);
            prev_bank.assign(bank.begin() + pre_bank.size() + 1, bank.end());
            copy(PRE, HID);
            bank = pre_bank;
        }
        if (v.has_message) {
            const size_t off = size_t(i - c_.loop_start) * 2 * h;
            MixBatch mb{};
            mb.j[0] = job(prev_bank, PREV, xq_ + off, M0);
            mb.j[1] = job(prev_bank, PREV, xkv_ + off, M1);
            mb.j[2] = job(prev_bank, PREV, xkv_ + off + h, M2);
            mb.j[3] = job(prev_bank, PREV, xq_ + off + h, M3);
            mb.count = 4;
            mix(mb, M);
        }
        int in[3] = {HID, HID, HID};
        if (i) {
            MixBatch mb{};
            const int nin = v.sliding ? 1 : 3;
            for (int j = 0; j < nin; ++j) mb.j[j] = job(bank, HID, Ly.aq + (v.sliding ? 0 : size_t(j) * h), X0 + j);
            mb.count = nin;
            mix(mb, M);
            for (int j = 0; j < 3; ++j) in[j] = v.sliding ? X0 : X0 + j;
        }
        if (v.has_message) {
            GateBatch gb{};
            int ctrl[3] = {0, 1, 2};
            for (int j = 0; j < 3; ++j) gb.j[j] = GateJob{in[j], M0 + j, G0 + j, Ly.cgain[j], Ly.cforget[j]};
            gb.count = 3;
            gates(gb, Ly, ctrl, M);
            for (int j = 0; j < 3; ++j) in[j] = G0 + j;
        }
        int nrm[3];
        bool same = false;
        {
            NormBatch nb{};
            const bool gemm = M >= 32;
            if (v.sliding && !v.has_message) {
                nb.j[0] = NormJob{in[0], Ly.nq, N0, gemm ? nb_ : nullptr}; nb.count = 1;
                nrm[0] = nrm[1] = nrm[2] = N0;
                same = true;
            } else {
                const uint16_t* w[3] = {Ly.nq, Ly.nk, Ly.nv};
                for (int j = 0; j < 3; ++j) nb.j[j] = NormJob{in[j], w[j], N0 + j, gemm ? nb_ + size_t(j) * MM * h : nullptr};
                nb.count = 3;
                nrm[0] = N0; nrm[1] = N1; nrm[2] = N2;
            }
            rms(nb, M);
        }
        const int NA = Ly.NA, NB = Ly.NB, NC = Ly.NC;
        if (M == 1) {
            gemv<0>(Ly.Wp, NA + NB + NC, h, row(nrm[0]), projA_, NA, row(nrm[1]), projB_, NA + NB, row(nrm[2]), projC_);
        } else {
            const uint16_t* xb[3] = {nb_, same ? nb_ : nb_ + size_t(MM) * h, same ? nb_ : nb_ + 2 * size_t(MM) * h};
            dense(Ly.Wp, NA, h, row(nrm[0]), h, xb[0], projA_, NA, M, true);
            dense(Ly.Wp + size_t(NA) * h, NB, h, row(nrm[1]), h, xb[1], projB_, NB, M, true);
            dense(Ly.Wp + size_t(NA + NB) * h, NC, h, row(nrm[2]), h, xb[2], projC_, NC, M, true);
        }
        const int oq = kOQ, orel = Q, osink = Q + H * R, og = Q + H * R + 32;
        {   // per (row, head): q/rel clip, k/v short conv + clip / value norm, cache append
            float *pA = projA_, *pB = projB_, *pC = projC_;
            const uint16_t *qs = Ly.qs, *rsc = Ly.rs, *kw = Ly.kw, *vw = Ly.vw;
            float *kh = sl.khist, *vh = sl.vhist;
            uint16_t *kc = sl.kc, *vc = sl.vc;
            const int cap = sl.cap;
            const bool sliding = v.sliding;
            const float qk_clip = c_.qk_clip, rel_clip = c_.rel_clip, cscale = v.sliding ? conv_static : conv_dyn;
            const float alpha = c_.log_alpha, lfloor = float(c_.log_floor);
            const int taps = c_.conv, G = H + 2 * KVH;
            q_.parallel_for<InkPost>(sycl::nd_range<1>(size_t(M) * G * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int ga = int(it.get_group(0)), m = ga / G, grp = ga % G, d = int(it.get_local_id(0));
                auto g = it.get_group();
                const int pos = *posd + m;
                const float tau = sliding ? 1.f : 1.f + alpha * sycl::log(sycl::fmax(1.f, float(pos + 1) / lfloor));
                float* A = pA + size_t(m) * NA;
                if (grp < H) {
                    const int j = grp;
                    const float p = bfr(A[oq + j * D + d]) * bfw(qs[j * D + d]);
                    const float s = sycl::reduce_over_group(g, p * p, sycl::plus<float>());
                    const float f = sycl::fmin(1.f, qk_clip / (sycl::sqrt(s) + 1e-6f));
                    A[oq + j * D + d] = bfr(bfr(p * f) * tau);
                    auto sg = it.get_sub_group();
                    if (sg.get_group_linear_id() == 0) {
                        const float r = bfr(A[orel + j * R + d]) * bfw(rsc[j * R + d]);
                        const float s2 = sycl::reduce_over_group(sg, r * r, sycl::plus<float>());
                        const float f2 = sycl::fmin(1.f, rel_clip / (sycl::sqrt(s2) + 1e-6f));
                        A[orel + j * R + d] = bfr(bfr(r * f2) * tau);
                    }
                    return;
                }
                const bool isk = grp < H + KVH;
                const int kh_i = isk ? grp - H : grp - H - KVH;
                const int ch = kh_i * D + d;
                float* hist = isk ? kh : vh;
                const uint16_t* w = isk ? kw : vw;
                const float* P = isk ? pB : pC;
                const int ld = isk ? NB : NC;
                const float x = bfr(P[size_t(m) * ld + ch]);
                const float* dyn = P + size_t(m) * ld + K;
                float sum = 0.f;
                for (int lag = 0; lag <= sycl::min(pos, taps - 1); ++lag) {
                    const float xl = lag == 0 ? x : (m - lag >= 0 ? bfr(P[size_t(m - lag) * ld + ch]) : hist[size_t((pos - lag) % (taps - 1)) * K + ch]);
                    const float wt = sliding ? bfw(w[ch * taps + taps - 1 - lag]) : bfw(w[lag * K + ch]) + bfr(dyn[kh_i * taps + lag]);
                    sum += xl * wt;
                }
                float z = bfr((x + sum) * cscale);
                if (M == 1) hist[size_t(pos % (taps - 1)) * K + ch] = x;
                const float ss = sycl::reduce_over_group(g, z * z, sycl::plus<float>());
                if (isk) {
                    const float f = sycl::fmin(1.f, qk_clip / (sycl::sqrt(ss) + 1e-6f));
                    z = bfr(bfr(z * f) * 1.f);
                } else {
                    const float f = 1.f / sycl::sqrt(ss / float(D) + eps);
                    z = bfr(z * f);
                }
                (isk ? kc : vc)[size_t(pos % cap) * K + ch] = tobf(z);
            });
            if (M > 1) { hist_update(sl.khist, projB_, NB, K, M); hist_update(sl.vhist, projC_, NC, K, M); }
        }
        {   // attention over the cached window
            const int cap = sl.cap, win = sl.win, E = Ly.E, groups = H / KVH;
            const float scale = 1.f / std::sqrt(float(D));
            const float* pA = projA_;
            const uint16_t *kc = sl.kc, *vc = sl.vc, *table = Ly.table;
            float* aout = aout_;
            uint16_t* aoutb = aoutb_;
            if (M == 1) {
                // split-K decode attention: key blocks of 128 in parallel.  K1 scores + block max/sum,
                // K2 weights bf(exp(s - high) / sum) with the global max/sum + block partial P.V,
                // K3 sums the blocks in order and applies the output gate.
                float* scores = scores_;
                float* stat = att_stat_;
                float* part = att_part_;
                const int limit = limit_, NCH = (win + 127) / 128;
                q_.submit([&](sycl::handler& cgh) {
                    sycl::local_accessor<float, 1> qsm(sycl::range<1>(144), cgh), red(sycl::range<1>(16), cgh);
                    cgh.parallel_for<InkAtt1>(sycl::nd_range<1>(size_t(H) * NCH * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        const int gi = int(it.get_group(0)), j = gi / NCH, c = gi % NCH, lid = int(it.get_local_id(0));
                        const int pos = *posd, count = sycl::min(pos + 1, win), first = pos + 1 - count;
                        if (c * 128 >= count) return;
                        auto sg = it.get_sub_group();
                        const int sgid = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
                        qsm[lid] = pA[oq + j * D + lid];
                        if (lid < 16) qsm[128 + lid] = pA[orel + j * R + lid];
                        sycl::group_barrier(it.get_group());
                        const int ii = c * 128 + sgid * 16 + lane;
                        float s = -INFINITY;
                        if (ii < count) {
                            const uint16_t* kr = kc + size_t((first + ii) % cap) * K + (j / groups) * D;
                            float dot = 0.f;
                            for (int d0 = 0; d0 < 128; d0 += 8) {
                                const sycl::vec<uint16_t, 8> kv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(kr + d0);
#pragma unroll
                                for (int e = 0; e < 8; ++e) dot += qsm[d0 + e] * bfw(kv[e]);
                            }
                            const float qk = bfr(bfr(dot) * scale);
                            const int distance = pos - (first + ii);
                            float bias = 0.f;
                            if (distance < E)
                                for (int r = 0; r < 16; ++r) bias += qsm[128 + r] * bfw(table[size_t(r) * E + distance]);
                            s = bfr(qk + bfr(bias));
                            scores[size_t(j) * limit + ii] = s;
                        }
                        const float cmax = sycl::reduce_over_group(it.get_group(), s, sycl::maximum<float>());
                        const float csum = sycl::reduce_over_group(it.get_group(), ii < count ? sycl::exp(s - cmax) : 0.f, sycl::plus<float>());
                        if (lid == 0) { stat[(size_t(j) * NCH + c) * 2] = cmax; stat[(size_t(j) * NCH + c) * 2 + 1] = csum; }
                    });
                });
                q_.submit([&](sycl::handler& cgh) {
                    sycl::local_accessor<float, 1> wl(sycl::range<1>(128), cgh);
                    cgh.parallel_for<InkAtt2>(sycl::nd_range<1>(size_t(H) * NCH * 128, 128), [=](sycl::nd_item<1> it) {
                        const int gi = int(it.get_group(0)), j = gi / NCH, c = gi % NCH, d = int(it.get_local_id(0));
                        const int pos = *posd, count = sycl::min(pos + 1, win), first = pos + 1 - count;
                        if (c * 128 >= count) return;
                        const int nch = (count + 127) / 128;
                        const float sink = bfr(pA[osink + j]);
                        float high = sink;
                        for (int x = 0; x < nch; ++x) high = sycl::fmax(high, stat[(size_t(j) * NCH + x) * 2]);
                        float sum = sycl::exp(sink - high);
                        for (int x = 0; x < nch; ++x) sum += stat[(size_t(j) * NCH + x) * 2 + 1] * sycl::exp(stat[(size_t(j) * NCH + x) * 2] - high);
                        const int ii = c * 128 + d;
                        wl[d] = ii < count ? bfr(sycl::exp(scores[size_t(j) * limit + ii] - high) / sum) : 0.f;
                        sycl::group_barrier(it.get_group());
                        const int hd = (j / groups) * D, nk = sycl::min(128, count - c * 128);
                        float acc = 0.f;
                        for (int k = 0; k < nk; ++k) acc += wl[k] * bfw(vc[size_t((first + c * 128 + k) % cap) * K + hd + d]);
                        part[(size_t(j) * NCH + c) * 128 + d] = acc;
                    });
                });
                q_.parallel_for<InkAtt3>(sycl::nd_range<1>(size_t(H) * 128, 128), [=](sycl::nd_item<1> it) {
                    const int j = int(it.get_group(0)), d = int(it.get_local_id(0));
                    const int pos = *posd, count = sycl::min(pos + 1, win), nch = (count + 127) / 128;
                    float acc = 0.f;
                    for (int x = 0; x < nch; ++x) acc += part[(size_t(j) * NCH + x) * 128 + d];
                    aout[j * D + d] = bfr(bfr(acc) * sigm(bfr(pA[og + j * D + d])));
                });
            } else {
                // one sub-group per (row, head): lane-per-key scores, two passes (max/sum, then weighted values)
                const size_t sgs = size_t(M) * H, ng = (sgs + 7) / 8;
                q_.submit([&](sycl::handler& cgh) {
                    sycl::local_accessor<float, 1> qsm(sycl::range<1>(8 * 144), cgh);
                    cgh.parallel_for<InkAttB>(sycl::nd_range<1>(ng * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        auto sg = it.get_sub_group();
                        const int sgid = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
                        const size_t t = it.get_group(0) * 8 + sgid;
                        if (t >= sgs) return;
                        const int m = int(t / H), j = int(t % H);
                        const int pos = *posd + m, count = sycl::min(pos + 1, win), first = pos + 1 - count;
                        const float* A = pA + size_t(m) * NA;
                        float* qq = qsm.template get_multi_ptr<sycl::access::decorated::no>().get() + sgid * 144;
#pragma unroll
                        for (int e = 0; e < 8; ++e) qq[lane * 8 + e] = A[oq + j * D + lane * 8 + e];
                        qq[128 + lane] = A[orel + j * R + lane];
                        sycl::group_barrier(sg);
                        const int hd = (j / groups) * D;
                        auto score = [&](int ii) {
                            const uint16_t* kr = kc + size_t((first + ii) % cap) * K + hd;
                            float s = 0.f;
                            for (int d0 = 0; d0 < 128; d0 += 8) {
                                const sycl::vec<uint16_t, 8> kv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(kr + d0);
#pragma unroll
                                for (int e = 0; e < 8; ++e) s += qq[d0 + e] * bfw(kv[e]);
                            }
                            const float qk = bfr(bfr(s) * scale);
                            const int distance = pos - (first + ii);
                            float bias = 0.f;
                            if (distance < E)
                                for (int r = 0; r < 16; ++r) bias += qq[128 + r] * bfw(table[size_t(r) * E + distance]);
                            return bfr(qk + bfr(bias));
                        };
                        float mx = -INFINITY, l = 0.f;
                        for (int ii = lane; ii < count; ii += 16) {
                            const float s = score(ii);
                            if (s > mx) { l = l * sycl::exp(mx - s) + 1.f; mx = s; } else l += sycl::exp(s - mx);
                        }
                        const float gm = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
                        const float lsum = sycl::reduce_over_group(sg, mx == -INFINITY ? 0.f : l * sycl::exp(mx - gm), sycl::plus<float>());
                        const float sink = bfr(A[osink + j]);
                        const float high = sycl::fmax(sink, gm);
                        const float sum = sycl::exp(sink - high) + lsum * sycl::exp(gm - high);
                        float acc[8];
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[e] = 0.f;
                        for (int base = 0; base < count; base += 16) {
                            const int ii = base + lane;
                            const float w = ii < count ? bfr(sycl::exp(score(ii) - high) / sum) : 0.f;
                            const int nk = sycl::min(16, count - base);
                            for (int kk = 0; kk < nk; ++kk) {
                                const float wk = sycl::select_from_group(sg, w, kk);
                                const uint16_t* vr = vc + size_t((first + base + kk) % cap) * K + hd + lane * 8;
                                const sycl::vec<uint16_t, 8> vv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(vr);
#pragma unroll
                                for (int e = 0; e < 8; ++e) acc[e] += wk * bfw(vv[e]);
                            }
                        }
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            const int c = j * D + lane * 8 + e;
                            const float o = bfr(bfr(acc[e]) * sigm(bfr(A[og + c])));
                            aout[size_t(m) * Q + c] = o;
                            aoutb[size_t(m) * Q + c] = tobf(o);
                        }
                    });
                });
            }
        }
        dense(Ly.Wo, h, Q, aout_, Q, aoutb_, row(Z), h, M, true);
        int bank_row = -1;
        if (v.block_boundary) {
            if (next_bank >= kMaxBankRows) throw std::runtime_error("bank rows exhausted");
            bank_row = BANK0 + next_bank++;
        }
        {   // residual = sconv(z); bank push or + hidden
            const uint16_t* w = Ly.aconv;
            float* hist = sl.ahist;
            const int taps = c_.conv;
            const float cs = conv_static;
            const int br = bank_row;
            q_.parallel_for<InkResid>(sycl::range<1>(size_t(M) * h), [=](sycl::id<1> id) {
                const int m = int(id[0] / h), ch = int(id[0] % h);
                const int pos = *posd + m;
                const float* zr = pool + size_t(Z) * RS;
                const float x = bfr(zr[size_t(m) * h + ch]);
                float sum = 0.f;
                for (int lag = 0; lag <= sycl::min(pos, taps - 1); ++lag) {
                    const float xl = lag == 0 ? x : (m - lag >= 0 ? bfr(zr[size_t(m - lag) * h + ch]) : hist[size_t((pos - lag) % (taps - 1)) * h + ch]);
                    sum += xl * bfw(w[ch * taps + taps - 1 - lag]);
                }
                const float out = bfr((x + sum) * cs);
                if (M == 1) hist[size_t(pos % (taps - 1)) * h + ch] = x;
                const size_t o = size_t(m) * h + ch;
                const float hid = pool[size_t(HID) * RS + o];
                if (br >= 0) { pool[size_t(br) * RS + o] = hid; pool[size_t(RES) * RS + o] = out; }
                else pool[size_t(RES) * RS + o] = out + hid;
            });
            if (M > 1) hist_update(sl.ahist, row(Z), h, h, M);
        }
        if (bank_row >= 0) bank.push_back(bank_row);
        { MixBatch mb{}; mb.j[0] = job(bank, RES, Ly.mq, XM); mb.count = 1; mix(mb, M); }
        int xm = XM;
        if (v.has_message) {
            GateBatch gb{};
            int ctrl[1] = {3};
            gb.j[0] = GateJob{XM, M3, GM, Ly.cgain[3], Ly.cforget[3]};
            gb.count = 1;
            gates(gb, Ly, ctrl, M);
            xm = GM;
        }
        { NormBatch nb{}; nb.j[0] = NormJob{xm, Ly.nm, NM, M >= 2 ? nmb_ : nullptr}; nb.count = 1; rms(nb, M); }
        // router: sigmoid scores, top-k by score + bias, weights = scale * score / sum, sorted by expert id
        dense(Ly.Wr, c_.experts, h, row(NM), h, nmb_, rlog_, c_.experts, M, false);
        {
            const float* lg = rlog_;
            const float* bias = Ly.rbias;
            int* sel = sel_;
            float* wts = wts_;
            const int En = c_.experts, TK = c_.top_k;
            const float rscale = c_.route_scale;
            q_.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> key(sycl::range<1>(128), cgh), score(sycl::range<1>(128), cgh);
                sycl::local_accessor<int, 1> byrank(sycl::range<1>(16), cgh);
                sycl::local_accessor<float, 1> total(sycl::range<1>(1), cgh);
                cgh.parallel_for<InkTopk>(sycl::nd_range<1>(size_t(M) * 128, 128), [=](sycl::nd_item<1> it) {
                    const int m = int(it.get_group(0)), e = int(it.get_local_id(0));
                    auto g = it.get_group();
                    if (e < En) { const float sc = sigm(lg[size_t(m) * En + e]); score[e] = sc; key[e] = sc + bias[e]; }
                    sycl::group_barrier(g);
                    int rank = 1 << 30;
                    if (e < En) {
                        rank = 0;
                        for (int f = 0; f < En; ++f) rank += (key[f] > key[e] || (key[f] == key[e] && f < e)) ? 1 : 0;
                        if (rank < TK) byrank[rank] = e;
                    }
                    sycl::group_barrier(g);
                    if (e == 0) { float s = 0.f; for (int r = 0; r < TK; ++r) s += score[byrank[r]]; total[0] = s; }
                    sycl::group_barrier(g);
                    if (rank < TK) {
                        int at = 0;
                        for (int r = 0; r < TK; ++r) at += byrank[r] < e ? 1 : 0;
                        sel[size_t(m) * TK + at] = e;
                        wts[size_t(m) * TK + at] = rscale * (score[e] / (total[0] + 1e-20f));
                    }
                });
            });
        }
        if (M == 1) experts_decode(Ly);
        else experts_prompt(Ly, M);
        {   // hidden = bf(residual + sconv(y)); y = bf(sum_s w_s d_s) in slot (= ascending expert) order
            const uint16_t* w = Ly.mconv;
            float* hist = sl.mhist;
            const float* dbuf = dbuf_;
            const float* wts = wts_;
            const int TK = c_.top_k, taps = c_.conv;
            const float cs = conv_static;
            q_.parallel_for<InkMlpConv>(sycl::range<1>(size_t(M) * h), [=](sycl::id<1> id) {
                const int m = int(id[0] / h), ch = int(id[0] % h);
                const int pos = *posd + m;
                const float* yb = pool + size_t(YB) * RS;
                float x;
                if (M == 1) {
                    float acc = 0.f;
                    for (int s = 0; s < TK; ++s) acc += wts[s] * dbuf[size_t(s) * h + ch];
                    x = bfr(acc);
                } else x = yb[size_t(m) * h + ch];
                float sum = 0.f;
                for (int lag = 0; lag <= sycl::min(pos, taps - 1); ++lag) {
                    const float xl = lag == 0 ? x : (m - lag >= 0 ? yb[size_t(m - lag) * h + ch] : hist[size_t((pos - lag) % (taps - 1)) * h + ch]);
                    sum += xl * bfw(w[ch * taps + taps - 1 - lag]);
                }
                const float y = bfr((x + sum) * cs);
                if (M == 1) hist[size_t(pos % (taps - 1)) * h + ch] = x;
                const size_t o = size_t(m) * h + ch;
                pool[size_t(HID) * RS + o] = bfr(pool[size_t(RES) * RS + o] + y);
            });
            if (M > 1) hist_update(sl.mhist, row(YB), h, h, M);
        }
    }
    {   // last row only: output mix, norm, unembed with softcap, argmax
        MixBatch mb{};
        mb.j[0] = job(bank, HID, oquery_, X0); mb.count = 1;
        mix(mb, 1, M - 1);
        NormBatch nb{}; nb.j[0] = NormJob{X0, fnorm_, N0, nullptr}; nb.count = 1;
        rms(nb, 1, M - 1);
    }
    const int V = c_.output_vocab;
    gemv<2>(unembed_, V, h, row(N0, M - 1), logits_, V, nullptr, nullptr, V, nullptr, nullptr, c_.logit_cap);
    {
        const float* lg = logits_;
        float* bv = amx_v_;
        int* bi = amx_i_;
        int* nx = next_;
        const int per = 256 * 16, G = (V + per - 1) / per;
        q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lv(sycl::range<1>(256), cgh);
            sycl::local_accessor<int, 1> li(sycl::range<1>(256), cgh);
            cgh.parallel_for<InkArg1>(sycl::nd_range<1>(size_t(G) * 256, 256), [=](sycl::nd_item<1> it) {
                const int lid = int(it.get_local_id(0)), base = int(it.get_group(0)) * per;
                float best = -INFINITY; int bidx = 0x7fffffff;
                for (int t = 0; t < 16; ++t) { const int k = base + t * 256 + lid; if (k < V && (lg[k] > best || (lg[k] == best && k < bidx))) { best = lg[k]; bidx = k; } }
                lv[lid] = best; li[lid] = bidx;
                for (int s = 128; s > 0; s >>= 1) {
                    sycl::group_barrier(it.get_group());
                    if (lid < s && (lv[lid + s] > lv[lid] || (lv[lid + s] == lv[lid] && li[lid + s] < li[lid]))) { lv[lid] = lv[lid + s]; li[lid] = li[lid + s]; }
                }
                if (lid == 0) { bv[it.get_group(0)] = lv[0]; bi[it.get_group(0)] = li[0]; }
            });
        });
        q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lv(sycl::range<1>(256), cgh);
            sycl::local_accessor<int, 1> li(sycl::range<1>(256), cgh);
            cgh.parallel_for<InkArg2>(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
                const int lid = int(it.get_local_id(0));
                float best = -INFINITY; int bidx = 0x7fffffff;
                for (int k = lid; k < G; k += 256) if (bv[k] > best || (bv[k] == best && bi[k] < bidx)) { best = bv[k]; bidx = bi[k]; }
                lv[lid] = best; li[lid] = bidx;
                for (int s = 128; s > 0; s >>= 1) {
                    sycl::group_barrier(it.get_group());
                    if (lid < s && (lv[lid + s] > lv[lid] || (lv[lid + s] == lv[lid] && li[lid + s] < li[lid]))) { lv[lid] = lv[lid + s]; li[lid] = li[lid + s]; }
                }
                if (lid == 0) *nx = li[0];
            });
        });
    }
    int* pd = pos_d_;
    q_.single_task<InkPos>([=]() { *pd += M; });
}

std::vector<int> read_ids(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read IDs file");
    std::vector<int> ids;
    for (int v; in >> v;) ids.push_back(v);
    if (ids.empty()) throw std::runtime_error("empty IDs file");
    return ids;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace ink

int main(int argc, char** argv) try {
    using namespace ink;
    if (argc < 6) {
        std::fprintf(stderr, "usage: inkling2_gpu MODEL_DIR check IDS CONTEXT NEW_TOKENS [REF_LOGITS|-] [CHUNK]\n"
                             "       inkling2_gpu MODEL_DIR bench IDS CONTEXT PP_LIST TG [CHUNK]\n");
        return 2;
    }
    const std::string mode = argv[2];
    const auto ids = read_ids(argv[3]);
    const int ctx = std::atoi(argv[4]);
    const auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.size() != 1) throw std::runtime_error("expected exactly one visible GPU (set ONEAPI_DEVICE_SELECTOR)");
    sycl::queue q(devs[0], sycl::property_list{sycl::property::queue::in_order()});
    std::printf("%s\n", devs[0].get_info<sycl::info::device::name>().c_str());
    if (mode == "check") {
        const int gen = std::atoi(argv[5]);
        const int chunk = argc > 7 ? std::atoi(argv[7]) : 512;
        Engine eng(q, argv[1], ctx, chunk);
        if (const char* g = std::getenv("INK_GRAPH")) eng.use_graph = std::atoi(g) != 0;
        auto compare = [&](const std::vector<float>& logits, int next, const char* what) {
            if (argc < 7 || std::string(argv[6]) == "-") return;
            std::ifstream rf(argv[6], std::ios::binary);
            std::vector<float> ref(logits.size());
            if (!rf.read(reinterpret_cast<char*>(ref.data()), ref.size() * sizeof(float))) throw std::runtime_error("cannot read reference logits");
            double maxd = 0, sumd = 0;
            int ra = 0;
            for (size_t k = 0; k < ref.size(); ++k) { const double d = std::fabs(double(ref[k]) - logits[k]); maxd = std::max(maxd, d); sumd += d; if (ref[k] > ref[ra]) ra = int(k); }
            std::printf("%s vs CPU reference: argmax ref %d gpu %d, max |dlogit| %.4f, mean %.5f\n", what, ra, next, maxd, sumd / ref.size());
        };
        // A: prompt token by token (decode graph), B: prompt as batched chunks
        std::vector<float> la, lb;
        int na = -1;
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < ids.size(); ++i) na = eng.step(ids[i], i + 1 == ids.size() ? &la : nullptr);
        std::printf("A token-by-token: %zu tokens %.1f ms next=%d\n", ids.size(), ms_since(t0), na);
        compare(la, na, "A");
        std::vector<int> ga;
        for (int g = 0, nx = na; g < gen; ++g) { ga.push_back(nx); if (g + 1 < gen) nx = eng.step(nx); }
        eng.reset();
        t0 = std::chrono::steady_clock::now();
        const int nb = eng.prefill(ids, &lb);
        std::printf("B batched (chunk %d): %zu tokens %.1f ms next=%d\n", chunk, ids.size(), ms_since(t0), nb);
        compare(lb, nb, "B");
        double maxd = 0;
        for (size_t k = 0; k < la.size(); ++k) maxd = std::max(maxd, double(std::fabs(la[k] - lb[k])));
        std::printf("A vs B last-row logits max |d| %.4f\n", maxd);
        std::vector<int> gb;
        for (int g = 0, nx = nb; g < gen; ++g) { gb.push_back(nx); if (g + 1 < gen) nx = eng.step(nx); }
        int same = 0;
        while (same < gen && ga[same] == gb[same]) ++same;
        std::printf("greedy continuation: A and B agree on the first %d of %d tokens\nA=", same, gen);
        for (int g = 0; g < gen; ++g) std::printf(g ? ",%d" : "%d", ga[g]);
        std::printf("\nB=");
        for (int g = 0; g < gen; ++g) std::printf(g ? ",%d" : "%d", gb[g]);
        std::printf("\n");
        return 0;
    }
    if (mode == "bench") {
        std::vector<int> pps;
        {
            std::stringstream ss(argv[5]);
            std::string t;
            while (std::getline(ss, t, ',')) pps.push_back(std::atoi(t.c_str()));
        }
        const int tg = argc > 6 ? std::atoi(argv[6]) : 128;
        const int chunk = argc > 7 ? std::atoi(argv[7]) : 2048;
        Engine eng(q, argv[1], ctx, chunk);
        {   // warm-up: graph capture, GEMM kernels, expert path
            std::vector<int> w(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), size_t(std::min(chunk, 256))));
            int nx = eng.prefill(w);
            for (int i = 0; i < 4; ++i) nx = eng.step(nx);
            eng.reset();
        }
        std::printf("| test | tokens | time (ms) | tok/s |\n|---|---|---|---|\n");
        for (int pp : pps) {
            if (size_t(pp) > ids.size() || pp + tg > ctx) { std::printf("| pp%d | skipped (ids %zu, ctx %d) |\n", pp, ids.size(), ctx); continue; }
            eng.reset();
            std::vector<int> p(ids.begin(), ids.begin() + pp);
            auto t0 = std::chrono::steady_clock::now();
            int nx = eng.prefill(p);
            const double pms = ms_since(t0);
            t0 = std::chrono::steady_clock::now();
            for (int g = 0; g < tg; ++g) nx = eng.step(nx);
            const double tms = ms_since(t0);
            std::printf("| pp%d | %d | %.1f | %.1f |\n| tg%d @ %d | %d | %.1f | %.1f |\n", pp, pp, pms, 1000.0 * pp / pms, tg, pp, tg, tms, 1000.0 * tg / tms);
            std::fflush(stdout);
        }
        return 0;
    }
    throw std::invalid_argument("mode must be check or bench");
} catch (const std::exception& e) {
    std::fprintf(stderr, "inkling2_gpu: %s\n", e.what());
    return 1;
}
