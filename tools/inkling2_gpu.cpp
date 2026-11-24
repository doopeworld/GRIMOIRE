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
#include "b70/imatrix.hpp"
#include "kernels.hpp"
#include "inkling2_sycl.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

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
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <barrier>
#include <thread>

namespace sycl_ext = sycl::ext::oneapi::experimental;
using bf16x = sycl::ext::oneapi::bfloat16;

template <int M> class InkGemv;
class InkMix; class InkRms; class InkGateCat; class InkGateBlend; class InkEmbed; class InkPost; class InkAtt1; class InkAtt2;
class InkAttB; class InkAtt3; class InkResid; class InkTopk; class InkMlpConv; class InkArg1; class InkArg2; class InkPos; class InkGu; class InkDown;
class InkQuantT; template <int> class InkGemvT8; class InkMail; class InkRMN; class InkPosP; class InkPPPack; class InkPPUnpack; class InkGateLow; class InkRouteTopk; class InkAddRoundH; class InkMixS; class InkBias; class InkHist; class InkGather; class InkCombine; class InkDqNv; class InkDqMx; class InkActP; class InkOutB; class InkLowB; class InkGateBlendM; class InkAttT; class InkAddRound; class InkPartial;
template <int M> class InkGemvM;
template <int M> class InkGemvT;
template <int M> class InkGemvTM;
class InkDXB; class InkResidV; class InkMlpConvV; class InkSplitP; class InkAddRoundS; class InkFoldQ; class InkFinB; class InkPosV2; class InkRouteGrp; class InkArgC1; class InkArgC2; class InkGroupP; class InkGateBlend2; class InkRowQ; class InkGatherS; class InkGroup; class InkGuS; class InkDownS; class InkCombS; class InkCommit; class InkCopy; class InkMixNorm; class InkKVCtx;
class InkDEmbed; class InkDNorm; class InkAddRes; class InkQKRope; class InkDAtt; class InkSilu; class InkRnnIn; class InkRnn; class InkFold; class InkArgM1; class InkArgMV; class InkHistV; class InkPosV;

struct InkAttnArgs {
    const float* A; int NA, oq, orel, osink, og;
    int H, groups, E;
    const uint16_t* table; const uint16_t *kc, *vc;
    int cap, K, win; float scale;
    float* aout; uint16_t* aoutb; int Q;
    const int* posd; int M;
    const uint16_t* bias;   // [M][H][E] bf16 = bf(sum_r rel[r] * table[r][d])
    int nsplit = 0, chunk = 0; float* part = nullptr;   // > 0: key-split partials (verify rows, M <= 8)
};
sycl::event ink_attn_prompt(sycl::queue& q, const InkAttnArgs& g);
sycl::event ink_attn_esimd(sycl::queue& q, const InkAttnArgs& g);
sycl::event ink_dense_bf16(sycl::queue& q, const uint16_t* W, int N, int K, const uint16_t* x, float* y, int M);
struct InkXArgs {
    const uint8_t* codes; const uint8_t* scales;
    const float* amax; const float* den;
    const int* sel; int e0, en, topk;
    int N, K;
    const float* x; size_t ldx;
    float* out;
    int gate_up; float gcap, ucap;
    const uint32_t* lut;
};
sycl::event ink_xexp(sycl::queue& q, const InkXArgs& g);
struct InkEsArgs {
    const uint8_t* pay; const uint8_t* scl;
    const float* amax; const float* den;
    const int* sel; int e0, en, topk;
    const float* x; float* out;
    int H, I; float gcap, ucap;
    int own = -1;
};
bool ink_es_moe(sycl::queue& q, const InkEsArgs& gu, const InkEsArgs& dn);
struct InkI8Args {
    const int8_t* T; const float* sc; int N, K;
    const float *x0, *x1, *x2; float *y0, *y1, *y2; int end0, end1; int mode; float cap;
};
bool ink_es_gemv_i8(sycl::queue& q, const InkI8Args& a, int ks);
struct InkI8MArgs {
    const int8_t* T; const float* sc; int N, K, M;
    const float *x0, *x1, *x2; size_t ldx;
    float *y0, *y1, *y2; size_t ly0, ly1, ly2; int end0, end1; int mode; float cap;
};
bool ink_es_gemv_i8m(sycl::queue& q, const InkI8MArgs& a, int ks);
struct InkNvEsArgs {
    const uint8_t* pay; const uint8_t* scl; int Ne, K, rows_total;
    const uint16_t* A; float* out;
    const int* tiles; int T; const int* uinfo; const float* outer;
    const uint32_t* lut;
    uint16_t* act = nullptr; float gcap = 0.f, ucap = 0.f;
    int fmt = 0;
};
sycl::event ink_moe_nv_esimd(sycl::queue& q, const InkNvEsArgs& g);
struct InkSmArgs {
    const uint8_t* pay; const uint8_t* scl; int Ne, K, rows_total;
    const uint16_t* A; int slots;
    void* out;
    const int* gi; int e0, en;
    const float* amax; const float* den;
    const uint32_t* lut;
    float gcap, ucap;
    int U;
    int alu = 1;
    int own = -1;
};
sycl::event ink_moe_small_esimd(sycl::queue& q, const InkSmArgs& g, int fmt, int epi);
struct InkGBArgs {
    const float* low; int ldlow;
    const uint16_t* gain[3]; const uint16_t* forget[3];
    float* pool; size_t RS; int cur[3], msg[3], out[3];
    int count, M, h;
};
sycl::event ink_gate_blend_esimd(sycl::queue& q, const InkGBArgs& g);
sycl::event ink_xexp_pack(sycl::queue& q, const uint8_t* pay, const uint8_t* scl, int E, int N, int K, uint8_t* codes, uint8_t* scales);
sycl::event ink_moe_grouped(sycl::queue& q, int fmt, const uint8_t* pay, const uint8_t* scl, size_t rb, size_t sb,
                            int N, int K, const uint16_t* A, float* out, const int* tiles, int T, const int* uinfo,
                            const float* outer, const std::vector<sycl::event>& deps);

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

// INK_I4=1: a dense weight in GRIMOIRE's INT4 format (b70::quantize Fmt::INT4: RTN, group 128, bf16 scales, uint8 zeros),
// K zero-padded to a multiple of 512 for its small-M ESIMD kernels
struct Q4 { uint8_t* pay = nullptr; uint16_t* sc = nullptr; uint8_t* z = nullptr; int64_t rb = 0; int rs = 0, N = 0, K = 0; };
struct Layer {
    bool sliding = true, loop = false;
    PackedFormat fgu{}, fdn{};
    uint16_t *Wp = nullptr, *Wo = nullptr, *Wr = nullptr;
    int koff = 0, kfull = 0;  // full-attention conv static weights: this card's channel offset and the full width
    uint16_t *WpT = nullptr, *WoT = nullptr, *WrT = nullptr, *cdownT[4]{};  // decode tiles [N/16][K/8][16][8]
    int8_t *WpQ = nullptr, *WoQ = nullptr;   // INK_I8=1: the same tiles as int8, one fp32 scale per row
    int8_t *WpR = nullptr, *WoR = nullptr;   // the same int8 values row-major, K padded to 512 (GRIMOIRE INT8 small-M, verify)
    Q4 Wp4, Wo4;                               // INK_I4=1
    float *WpS = nullptr, *WoS = nullptr;
    float* rbias = nullptr;
    uint16_t *nq = nullptr, *nk = nullptr, *nv = nullptr, *nm = nullptr;
    uint16_t *qs = nullptr, *rs = nullptr, *table = nullptr;
    int E = 0;
    uint16_t *kw = nullptr, *vw = nullptr, *aconv = nullptr, *mconv = nullptr;
    uint16_t *cdown[4]{}, *cgain[4]{}, *cforget[4]{};
    uint8_t *gu = nullptr, *gus = nullptr, *dn = nullptr, *dns = nullptr;
    uint8_t *guX = nullptr, *gusX = nullptr, *dnX = nullptr, *dnsX = nullptr;   // INK_XEXP=1: DPAS decode layout
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
struct NormJob { int in; const uint16_t* w; int out; uint16_t* outb; int ldb = 0; };   // ldb: bf16 copy pitch (0 = hidden)
struct NormBatch { NormJob j[3]; int count; };
struct GateJob { int cur, msg, out; const uint16_t* gain; const uint16_t* forget; };
struct GateBatch { GateJob j[3]; int count; };

// Two-card exchange: pinned host slots, cross-queue events, one std::barrier per exchange (no host wait on the GPU).
// Two-card watchdog: every exchange stamps progress; no progress for 60 s -> print where each card is and exit.
static std::atomic<long long> g_tp_progress{0};
static std::atomic<int> g_tp_stage[2] = {};
static double g_tp_wait_ms[2][2] = {};
static constexpr uint32_t kEmpty = 0xFFFFFFFFu;   // never a computed fp32 (Intel NaNs are 0x7fc00000)   // per card: copy-out wait, thread meet
static long g_tp_n[2] = {};
static long long now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct TPX {
    static constexpr int NS = 8;
    float* hb[2][NS] = {};
    float* ib[2][NS] = {};   // small exchanges: card r's IN slots (pinned in r's context); the peer's host thread fills them
    // device mailboxes (INK_DEVX=1, after Xenithium's comm/mailbox_esimd.hpp, adapted to pinned host memory because the
    // Tower has no P2P): box[r] = card r's inbox, [2 parities][kBoxWords] words, every word its own flag (kEmpty until
    // the peer's system-scope store lands). Each card's kernel stores into the peer's inbox and spins on its own.
    static constexpr size_t kBoxWords = 8 * 2304;
    uint32_t* box[2] = {};
    uint32_t* err = nullptr;   // [2] spin timeouts
    int mcount[2] = {0, 0};
    sycl::event d2h[2][NS], h2d[2][NS];
    std::barrier<> bar{2};
    float bv[2][8] = {};
    int bi[2][8] = {};
    int counter[2] = {0, 0};
};

enum Row { HID, RES, X0, X1, X2, XM, N0, N1, N2, NM, M0, M1, M2, M3, PRE, PREV, G0, G1, G2, GM, Z, YB, TAP0, TAP1, TAP2, TAP3, BANK0, NROWS = BANK0 + kMaxBankRows };

class Engine {
public:
    double sect_ms_[8] = {};
    // ---- pipeline parallel (two cards, each with the full weights): this engine runs visits [vbeg_, vend_)
    struct PPState { std::vector<int> bank, pre_bank, prev_bank; int next_bank = 0; };
    int vbeg_ = 0, vend_ = -1;
    PPState pp_in_, pp_out_;
    void set_visits(int b, int e) { vbeg_ = b; vend_ = e; }
    int n_visits() const { return int(visits_.size()); }
    bool pp_first() const { return vbeg_ == 0; }
    bool pp_last() const { return vend_ < 0 || vend_ >= int(visits_.size()); }
    // rows that cross the split: HID, PRE, PREV, bank rows [BANK0, BANK0 + next_bank)
    static std::vector<int> pp_rows(int next_bank) {
        std::vector<int> r{HID, PRE, PREV};
        for (int b = 0; b < next_bank; ++b) r.push_back(BANK0 + b);
        return r;
    }
    // copy the crossing rows (bf16: every one is a copy of a bf16-rounded hidden state) into host[ rows ][M][h]
    void pp_export(uint16_t* host, int M) {
        const auto rows = pp_rows(pp_out_.next_bank);
        uint16_t* st = ppstage_;
        const int h = h_;
        for (size_t k = 0; k < rows.size(); ++k) {
            const float* src = row(rows[k]);
            uint16_t* d = st + k * size_t(M) * h;
            q_.parallel_for<InkPPPack>(sycl::range<1>(size_t(M) * h), [=](sycl::id<1> i) { d[i[0]] = tobf(src[i[0]]); });
        }
        q_.memcpy(host, st, rows.size() * size_t(M) * h * 2).wait();
    }
    void pp_import(const uint16_t* host, int M, const PPState& s) {
        pp_in_ = s;
        const auto rows = pp_rows(s.next_bank);
        uint16_t* st = ppstage_;
        const int h = h_;
        q_.memcpy(st, host, rows.size() * size_t(M) * h * 2);
        for (size_t k = 0; k < rows.size(); ++k) {
            float* dst = row(rows[k]);
            const uint16_t* sv = st + k * size_t(M) * h;
            q_.parallel_for<InkPPUnpack>(sycl::range<1>(size_t(M) * h), [=](sycl::id<1> i) { dst[i[0]] = bfw(sv[i[0]]); });
        }
    }
    // one chunk / one token through this engine's visits; the last stage returns the next token, the others -1
    int pp_run(const int* ids, int M) {
        if (pos_ + M > limit_) throw std::runtime_error("context full");
        q_.memcpy(tok_, ids, size_t(M) * sizeof(int));
        if (M == 1 && use_graph) {
            if (!gexec_) {
                sycl_ext::command_graph<sycl_ext::graph_state::modifiable> g(q_.get_context(), q_.get_device());
                g.begin_recording(q_);
                run(1);
                g.end_recording(q_);
                gexec_ = std::make_unique<sycl_ext::command_graph<sycl_ext::graph_state::executable>>(g.finalize());
            }
            q_.ext_oneapi_graph(*gexec_);
        } else run(M);
        pos_ += M;
        return pp_last() ? final_token() : -1;
    }
    uint16_t* ppstage_ = nullptr;
    void pp_alloc(int max_rows) { if (!ppstage_) ppstage_ = dev_.alloc<uint16_t>(size_t(max_rows) * mmax_ * h_); }
    void sync_all() { q_.wait(); }
    Engine(sycl::queue& q, const std::string& dir, int limit, int chunk, int rank = 0, int world = 1, TPX* tpx = nullptr)
        : q_(q), ck_(Checkpoint::open(dir)), c_(ck_.config), dev_(q), limit_(limit), mmax_(std::max(1, chunk)), rank_(rank), world_(world), tpx_(tpx) {
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
    void load_mtp(const std::string& path);
    std::vector<int> mtp_step(int t, const int* forced = nullptr);
    bool use_graph = true;
    int gpar_ = 0, gex_ = 0;   // two-card graph: mailbox count at capture, exchanges inside
    // two cards: a step recorded as a chain of graphs split at its exchanges (each segment ends with the copy-out of
    // its exchange; the next one starts with the add); replay = per segment one graph launch + the host exchange.
    // Keyed by M (1 decode, 8 verify).  INK_NO_SEG=1: kernels submitted one by one (old).
    struct SegStep {
        std::vector<std::unique_ptr<sycl_ext::command_graph<sycl_ext::graph_state::executable>>> g;
        std::vector<int> slot; std::vector<size_t> n;   // per exchange: slot index (from a multiple of NS), floats
    };
    SegStep seg_[9];
    bool seg_rec_ = false;
    std::unique_ptr<sycl_ext::command_graph<sycl_ext::graph_state::modifiable>> seg_cur_;
    SegStep* seg_dst_ = nullptr;
    bool seg_on_ = !std::getenv("INK_NO_SEG");
  public:
    void run_seg(int M);   // run(M) through the recorded chain (records it the first time)
  private:
    bool taps_on = false;

private:
    sycl::queue& q_;
    Checkpoint ck_;
    const Config& c_;
    Dev dev_;
    int limit_, mmax_, h_ = 0, pos_ = 0;
    int rank_ = 0, world_ = 1;
    TPX* tpx_ = nullptr;
    int e0_ = 0, en_ = 0, v0_ = 0, vrows_ = 0, vvalid_ = 0;
    // two cards: every expert on both cards (INK_NO_EPREP=1: half each); decode / verify split the USED experts by
    // parity at run time (slot / used-expert index), prompts keep the static halves [e0_, e0_ + en_)
    bool rep_ = false;
    int we0_ = 0, wen_ = 0, own_ = -1;
    float *peer_ = nullptr, *bestv_ = nullptr, *gate_g_ = nullptr, *gate_f_ = nullptr;
    uint16_t* lowb_ = nullptr;
    // pinned table (copied to tbl_d_ per visit): [0,128) used expert ids, [128,256) gate|up outer, [256,384) down outer,
    // [384,768) uinfo (local expert, first row, rows), [768, ...) grouped-GEMM tiles (u, row block)
    static constexpr size_t kTbl = 768 + 2 * 4096;
    float* elist_ = nullptr;
    uint16_t* bias_ = nullptr;
    uint32_t* xlut_ = nullptr;   // ink_xexp byte tables
    int* route_cnt_ = nullptr;    // fused router: finished work-groups (reset by the last)
    uint32_t* nvlut_ = nullptr;   // prompt ESIMD MoE: byte -> E2M1 bf16 pair (lo | hi << 16)
    uint32_t* nvlut16_ = nullptr; // verify ESIMD MoE: byte -> E2M1 fp16 pair
    float* tbl_d_ = nullptr;
    void allreduce(float* buf, size_t n);  // buf = bf(buf + peer's buf), both cards same bits
    int final_token(int m = 0);
    void final_tokens(int n, int* out);
    void argmax_rows_v(const float* lg, int ld, int V, int n, int* out, float* val, int base);
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
    uint16_t* tiled(const uint16_t* src, int N, int K);
    void quant_tiled(const uint16_t* T, int N, int K, int8_t*& q, float*& sc);
    template <int MODE> void gemvt8(const int8_t* T, const float* sc, int N, int K, int KS, const float* x0, float* y0, int end0, const float* x1,
                                    float* y1, int end1, const float* x2, float* y2, float cap = 0.f);
    bool i8_ = std::getenv("INK_I8") && std::atoi(std::getenv("INK_I8")) == 1;
    bool i8v_ = i8_ && !std::getenv("INK_NO_I8V");   // verify (2..8 rows) dense projections: b70::launch_fp8_smallm on int8
    int8_t* unR_ = nullptr;
    int8_t* rowq(const int8_t* tiled, int N, int K, int Kp);
    float* projAll_ = nullptr;   // verify, sliding visits: [8][NA + NB + NC] from one INT8 launch
    float* attpart_ = nullptr; size_t attpart_n_ = 0;   // verify attention key-split partials
    bool post_bias_done_ = false;   // this visit's bias table came from InkPost
    bool route_deferred_ = false;   // verify rows: experts_small does the router top-k (InkRouteGrp)
    // two cards, decode / verify: the exchange's add is done by the next reader (InkResid / InkMlpConv) -- the peer's
    // partial (pinned host slot) is added as the row is read; set by allreduce, consumed (cleared) by the reader
    const float* peer_in_ = nullptr;
    bool fold_add_ = false;
    void split_proj(int NA, int NB, int NC, int M);
    bool i8_small(const int8_t* W, const float* S, int r0, int n, int Kp, const uint16_t* X, float* Y, int M);
    // INK_GSM=1: decode / verify dense projections through Grimoire's own small-M ESIMD kernels (libgrimoire_gemm:
    // launch_bf16_smallm, the Qwen / Ornith decode + verify path) on the row-major BF16 weights
    bool gsm_ = std::getenv("INK_GSM") && std::atoi(std::getenv("INK_GSM")) == 1;
    bool gsm_dense(const uint16_t* W, int N, int K, const uint16_t* X, float* Y, int M);
    bool gsmv_ = !std::getenv("INK_NO_GSMV");   // verify router + gate down-projections: b70::launch_bf16_smallm
    bool i4_ = std::getenv("INK_I4") && std::atoi(std::getenv("INK_I4")) == 1;
    bool want_logits_ = false, last_raw_ = false;   // last_raw_: the recorded / last run left raw INT4 logits   // a caller reads logits_: keep them softcapped (no raw INT4 logits)
    int i4parts_ = std::getenv("INK_I4_PARTS") ? std::atoi(std::getenv("INK_I4_PARTS")) : 7;
    int kp_ = 0;                      // padded hidden for the INT4 kernels
    uint16_t *xpad_ = nullptr, *xpadU_ = nullptr;   // padded bf16 inputs [3][8][kp_], [8][kp_]
    Q4 un4_;
    Q4 quant4(const uint16_t* src, int N, int K);
    // INK_IMATRIX=<gguf>: importance per input channel (llama.cpp imatrix) for the INT4 rounding; parts = row counts
    // with their GGUF tensor name ("" = none)
    Q4 quant4(const uint16_t* src, int N, int K, const std::vector<std::pair<int, std::string>>& parts);
    std::unique_ptr<b70::ImportanceMatrix> imx_;
    bool i4_gemv(const Q4& w, int r0, int n, const uint16_t* X, float* Y, int M);
    template <int MODE> void gemvt(const uint16_t* T, int N, int K, int KS, const float* x0, float* y0, int end0, const float* x1, float* y1, int end1,
                                   const float* x2, float* y2);
    // tiled, M <= 8 rows: x rows at ldx; segment outputs y0/y1/y2 with their own row pitch; rows >= nvalid not written
    template <int MODE> void gemvtm(const uint16_t* T, int N, int K, int KS, int M, const float* x0, const float* x1, const float* x2, size_t ldx,
                                    float* y0, size_t ld0, int end0, float* y1, size_t ld1, int end1, float* y2, size_t ld2, int nvalid,
                                    float cap = 0.f);
    uint16_t* unembedT_ = nullptr;
    uint16_t* unembedFullT_ = nullptr;
    int8_t* unembedQ_ = nullptr;
    float* unembedS_ = nullptr;   // two cards + MTP: the drafter's full-vocab copy (tiled)
    void run(int M);
    void dense(const uint16_t* W, int N, int K, const float* x, size_t ldx, const uint16_t* xb, float* y, size_t ldy, int M, bool round);
    template <int MODE> void gemv(const uint16_t* W, int rows, int K, const float* x0, float* y0, int end0, const float* x1, float* y1, int end1,
                                  const float* x2, float* y2, float cap = 0.f);
    void mix(const MixBatch& b, int M, int m0 = 0);
    void mix_small(const MixBatch& b, int M, int m0);
    void resid_mix_norm(const uint16_t* aconv, float* ahist, int br, const MixJob& J, const uint16_t* nw, int nout);
    void rms(const NormBatch& b, int M, int m0 = 0);
    void gates(const GateBatch& b, const Layer& Ly, const int* ctrl, int M);
    void experts_decode(const Layer& Ly);
    void experts_prompt(const Layer& Ly, int M);
    void dequant_used(PackedFormat f, int N, int K, const uint8_t* pay, const uint8_t* scl, size_t rb, size_t sb, int nu, int outer_at, uint16_t* dst, int u0 = 0);
    void hist_update(float* hist, const float* src, size_t ld, int n, int M);
    void hist_update_v(int slot, int kind, const float* src, size_t ld, int n, int M);
    // small-M (verify) path
    struct SlotPtrs { float* ring[4]; int n[4]; };
    bool verify_ = false, all_logits_ = false, mtp_ = false;
    float *vbuf_ = nullptr, *actS_ = nullptr, *dbufS_ = nullptr, *logitsM_ = nullptr;
    float *logitsR_ = nullptr, *amxm_v_ = nullptr; int* amxm_i_ = nullptr;
  public:
    double mtp_ms_[3] = {0, 0, 0};   // INK_MTP_TIME: draft, verify, commit (host wall)
  private:   // INT8 verify unembed: raw logits, argmax partials
    void argmax_rows_cap(const float* lg, int ld, int V, int n, int* out, float* val, int base, float cap);
    uint16_t *xgS_ = nullptr, *actSb_ = nullptr;   // ESIMD verify experts: bf16 A rows in slot order, bf16 act
    SlotPtrs* slotptr_ = nullptr;
    int* g_info_ = nullptr;
    template <int MODE> void gemvm(const uint16_t* W, int rows, int K, const float* x, size_t ldx, float* y, size_t ldy, int M, float cap = 0.f);
    void experts_small(const Layer& Ly, int M);
    void commit_verify(int p, int a);
    void argmax_rows(const float* lg, int V, int n, int* out);
    // DSpark drafter
    struct DLayer { uint16_t *Wqkv, *Wkvc, *Wo, *Wgu, *Wd, *nin, *npost, *qn, *kn, *knc, *dfn, *kc, *vc; bool sliding;
                    int8_t *Q8[4] = {}; float* S8[4] = {}; int N8[4] = {}, K8[4] = {}, KP8[4] = {};   // INT8 qkv / o / gu / d (row-major, K padded)
                    int8_t* T8[4] = {}; };                                                          // the same, decode tiles
    struct Draft {
        std::shared_ptr<b70::SafeTensors> st;
        DLayer L[3];
        uint16_t *mask, *norm, *w1, *rnnW, *rnnB, *foldW, *mdfn;
        int8_t *foldQ = nullptr, *UR = nullptr; float *foldS = nullptr, *US = nullptr, *baseR = nullptr; uint16_t* finb = nullptr;   // INT8 drafter head
        uint16_t* xb8 = nullptr;   // INT8 drafter layers: bf16 padded input rows [8][<= 9216]
        float *alpha, *foldB;
        float *r, *x, *qkv, *att, *o, *gu, *act, *fin, *base, *state, *inp, *proj, *latent, *dlog, *frnn, *ctx, *kvc;
        uint16_t *ctxb, *bk, *bv;
        int* dtok;
    };
    Draft d_{};
    void dgemv8(const DLayer& L, int w, const float* x, float* y);   // drafter INT8 layer GEMV, 8 rows
    void mix_norm_taps(int alpha_row, const uint16_t* w, int m0, int rows, float* out, uint16_t* outb);
    void draft_commit(int p0, int m0, int rows);
    void draft_block(int t, int p);
};

void Engine::load() {
    const std::string root = "model.llm.";
    const int Q = c_.local.heads * c_.local.head_dim, K = c_.local.kv_heads * c_.local.head_dim;
    if (c_.full.heads * c_.full.head_dim != Q || c_.full.kv_heads * c_.full.head_dim != K || c_.full.head_dim != 128 || c_.local.head_dim != 128 ||
        c_.full.heads != c_.local.heads || c_.rel_dim != 16 || c_.conv != 4 || c_.top_k != 8 || c_.experts > 128 || h_ % 256 || c_.inter % 128)
        throw std::runtime_error("geometry outside this port (head_dim 128, d_rel 16, conv 4, top-8, <= 128 experts)");
    const auto t0 = std::chrono::steady_clock::now();
    en_ = c_.experts / world_; e0_ = rank_ * en_;
    rep_ = world_ > 1 && !std::getenv("INK_NO_EPREP");
    we0_ = rep_ ? 0 : e0_; wen_ = rep_ ? c_.experts : en_; own_ = rep_ ? rank_ : -1;
    vrows_ = c_.vocab / world_; v0_ = rank_ * vrows_; vvalid_ = std::max(0, std::min(vrows_, c_.output_vocab - v0_));
    if (c_.vocab % (16 * world_) || c_.experts % world_ || c_.local.kv_heads % world_) throw std::runtime_error("two-card split");
    embed_ = up<uint16_t>(root + "embed.weight");
    embed_norm_ = up<uint16_t>(root + "embed_norm.weight");
    {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(host(root + "unembed.weight")) + size_t(v0_) * h_;
        unembed_ = dev_.alloc<uint16_t>(size_t(vrows_) * h_);
        q_.memcpy(unembed_, src, size_t(vrows_) * h_ * 2).wait();
        unembedT_ = tiled(src, vrows_, h_);
        if (i8_) quant_tiled(unembedT_, vrows_, h_, unembedQ_, unembedS_);
        if (i8v_) unR_ = rowq(unembedQ_, vrows_, h_, (h_ + 511) / 512 * 512);
        if (i4_ && (i4parts_ & 4)) un4_ = quant4(src, vrows_, h_, {{vrows_, "output.weight"}});
    }
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
        const int H = a.heads / world_, KVH = a.kv_heads / world_, R = c_.rel_dim, D = a.head_dim;
        const int h0 = rank_ * H, kv0 = rank_ * KVH, Qr = H * D, Kr = KVH * D;
        struct Part { std::string name; int row0; int rows; int pad; };
        std::vector<Part> parts = {{t + "wq_du.weight", h0 * D, Qr, 0}, {t + "wr_du.weight", h0 * R, H * R, 0}, {t + "attn_sink_proj.weight", h0, H, 32 - H},
                                   {t + "gate_attn.weight", h0 * D, Qr, 0}, {t + "wk_dv.weight", kv0 * D, Kr, 0}};
        if (!Ly.sliding) parts.push_back({t + "k_sconv.weight_proj", kv0 * c_.conv, KVH * c_.conv, 0});
        parts.push_back({t + "wv_dv.weight", kv0 * D, Kr, 0});
        if (!Ly.sliding) parts.push_back({t + "v_sconv.weight_proj", kv0 * c_.conv, KVH * c_.conv, 0});
        Ly.NA = Qr + H * R + 32 + Qr;
        Ly.NB = Kr + (Ly.sliding ? 0 : KVH * c_.conv);
        Ly.NC = Ly.NB;
        if (H > 32 || Ly.NA % 16 || Ly.NB % 16) throw std::runtime_error("projection padding");
        stage.assign(size_t(Ly.NA + Ly.NB + Ly.NC) * h_, 0);
        int at = 0;
        for (auto& p : parts) {
            const auto* src = host(p.name);
            std::memcpy(stage.data() + size_t(at) * h_, src + size_t(p.row0) * h_ * 2, size_t(p.rows) * h_ * 2);
            at += p.rows + p.pad;
        }
        if (at != Ly.NA + Ly.NB + Ly.NC) throw std::runtime_error("projection layout");
        maxNA_ = std::max(maxNA_, Ly.NA); maxNB_ = std::max(maxNB_, Ly.NB);
        Ly.Wp = dev_.alloc<uint16_t>(stage.size());
        q_.memcpy(Ly.Wp, stage.data(), stage.size() * 2).wait();
        Ly.WpT = tiled(stage.data(), Ly.NA + Ly.NB + Ly.NC, h_);
        if (i8_) quant_tiled(Ly.WpT, Ly.NA + Ly.NB + Ly.NC, h_, Ly.WpQ, Ly.WpS);
        if (i8v_) Ly.WpR = rowq(Ly.WpQ, Ly.NA + Ly.NB + Ly.NC, h_, (h_ + 511) / 512 * 512);
        if (i4_ && (i4parts_ & 1)) {
            const std::string b = "blk." + std::to_string(i) + ".";
            Ly.Wp4 = quant4(stage.data(), Ly.NA + Ly.NB + Ly.NC, h_, {{Ly.NA, b + "attn_q.weight"}, {Ly.NB, b + "attn_k.weight"}, {Ly.NC, b + "attn_v.weight"}});
        }
        {   // this card's columns of the attention output projection
            const uint16_t* src = reinterpret_cast<const uint16_t*>(host(t + "wo_ud.weight"));
            std::vector<uint16_t> wo(size_t(h_) * Qr);
            for (int n = 0; n < h_; ++n) std::memcpy(wo.data() + size_t(n) * Qr, src + size_t(n) * Q + size_t(h0) * D, size_t(Qr) * 2);
            Ly.Wo = dev_.alloc<uint16_t>(wo.size());
            q_.memcpy(Ly.Wo, wo.data(), wo.size() * 2).wait();
            Ly.WoT = tiled(wo.data(), h_, Qr);
            if (i8_) quant_tiled(Ly.WoT, h_, Qr, Ly.WoQ, Ly.WoS);
            if (i8v_) Ly.WoR = rowq(Ly.WoQ, h_, Qr, (Qr + 511) / 512 * 512);
            if (i4_ && (i4parts_ & 2)) Ly.Wo4 = quant4(wo.data(), h_, Qr, {{h_, "blk." + std::to_string(i) + ".attn_output.weight"}});
        }
        Ly.Wr = up<uint16_t>(s + "mlp.gate.weight");
        Ly.WrT = tiled(reinterpret_cast<const uint16_t*>(host(s + "mlp.gate.weight")), c_.experts, h_);
        Ly.rbias = up<float>(s + "mlp.gate.bias");
        if (Ly.sliding) { Ly.nq = Ly.nk = Ly.nv = up<uint16_t>(s + "attn_norm.weight"); }
        else { Ly.nq = up<uint16_t>(s + "q_input_norm.weight"); Ly.nk = up<uint16_t>(s + "k_input_norm.weight"); Ly.nv = up<uint16_t>(s + "v_input_norm.weight"); }
        Ly.nm = up<uint16_t>(s + "mlp_norm.weight");
        Ly.qs = up<uint16_t>(t + "q_l2_clip.scale") + size_t(h0) * D;
        Ly.rs = up<uint16_t>(t + "rel_l2_clip.scale") + size_t(h0) * R;
        Ly.table = up<uint16_t>(t + "rel_logits_proj.proj");
        Ly.E = Ly.sliding ? c_.window : c_.rel_extent;
        Ly.kw = up<uint16_t>(t + (Ly.sliding ? "k_sconv.weight" : "k_sconv.static_w"));
        Ly.vw = up<uint16_t>(t + (Ly.sliding ? "v_sconv.weight" : "v_sconv.static_w"));
        if (Ly.sliding) { Ly.kw += size_t(kv0) * D * c_.conv; Ly.vw += size_t(kv0) * D * c_.conv; }
        Ly.koff = kv0 * D; Ly.kfull = a.kv_heads * D;
        Ly.aconv = up<uint16_t>(s + "attn_sconv.weight");
        Ly.mconv = up<uint16_t>(s + "mlp_sconv.weight");
        if (Ly.loop) {
            const char* names[4] = {"attn", "k", "v", "mlp"};
            for (int j = 0; j < 4; ++j) {
                const auto p = s + "controllers." + names[j] + ".";
                Ly.cdown[j] = up<uint16_t>(p + "down_weight");
                Ly.cdownT[j] = tiled(reinterpret_cast<const uint16_t*>(host(p + "down_weight")), c_.gate_rank, 2 * h_);
                Ly.cgain[j] = up<uint16_t>(p + "gain_weight");
                Ly.cforget[j] = up<uint16_t>(p + "forget_weight");
            }
        }
        Ly.fgu = ck_.layout[i].gate_up;
        Ly.fdn = ck_.layout[i].down;
        const auto e = s + "mlp.experts.";
        {   // this card's experts [e0, e0 + en)
            auto part = [&](const std::string& n) {
                size_t nb;
                const auto* src = host(n, &nb);
                const size_t per = nb / c_.experts;
                uint8_t* d = dev_.alloc<uint8_t>(per * wen_);
                q_.memcpy(d, src + per * we0_, per * wen_);
                return d;
            };
            Ly.gu = part(e + "w13_weight"); Ly.gus = part(e + "w13_weight_scale");
            Ly.dn = part(e + "w2_weight"); Ly.dns = part(e + "w2_weight_scale");
        }
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
        static const bool xexp = std::getenv("INK_XEXP") && std::atoi(std::getenv("INK_XEXP")) == 1;
        if (xexp && !xlut_) {
            // byte -> (lo nibble, hi nibble) bf16 pair of the 1x class {0.5,1,2,4} and of the 1.5x class {1,2,4} (x1.5 in q3)
            auto v = [](int n, int cls) -> uint32_t {
                const int mag = n & 7;
                const bool c3 = (mag & 1) && mag > 1;
                if (mag == 0 || c3 != (cls == 3)) return 0u;
                return uint32_t(0x3F00 + ((mag >> 1) << 7)) | ((n & 8) ? 0x8000u : 0u);
            };
            std::vector<uint32_t> L(512);
            for (int b = 0; b < 256; ++b) { L[b] = v(b & 15, 1) | (v(b >> 4, 1) << 16); L[256 + b] = v(b & 15, 3) | (v(b >> 4, 3) << 16); }
            xlut_ = dev_.alloc<uint32_t>(512);
            q_.memcpy(xlut_, L.data(), 512 * 4).wait();
        }
        if (xexp && Ly.fgu == PackedFormat::NVFP4 && Ly.fdn == PackedFormat::NVFP4) {
            const int I2 = 2 * c_.inter, hh = h_, II = c_.inter;
            if (rep_) throw std::runtime_error("INK_XEXP with replicated experts: not supported");
            Ly.guX = dev_.alloc<uint8_t>(size_t(en_) * I2 * hh / 2); Ly.gusX = dev_.alloc<uint8_t>(size_t(en_) * I2 * hh / 16);
            Ly.dnX = dev_.alloc<uint8_t>(size_t(en_) * hh * II / 2); Ly.dnsX = dev_.alloc<uint8_t>(size_t(en_) * hh * II / 16);
            ink_xexp_pack(q_, Ly.gu, Ly.gus, en_, I2, hh, Ly.guX, Ly.gusX);
            ink_xexp_pack(q_, Ly.dn, Ly.dns, en_, hh, II, Ly.dnX, Ly.dnsX);
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
        sl.kc = dev_.zeros<uint16_t>(size_t(sl.cap) * K / world_); sl.vc = dev_.zeros<uint16_t>(size_t(sl.cap) * K / world_);
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
    logits_ = dev_.zeros<float>(std::max(c_.output_vocab, c_.vocab));
    route_cnt_ = dev_.zeros<int>(16);
    {
        static const uint32_t mag[8] = {0x0000u, 0x3F00u, 0x3F80u, 0x3FC0u, 0x4000u, 0x4040u, 0x4080u, 0x40C0u};
        std::vector<uint32_t> L(256);
        for (int b = 0; b < 256; ++b) L[b] = (mag[b & 7] | ((b & 8) ? 0x8000u : 0u)) | ((mag[(b >> 4) & 7] | ((b & 0x80) ? 0x8000u : 0u)) << 16);
        nvlut_ = dev_.alloc<uint32_t>(256);
        q_.memcpy(nvlut_, L.data(), 256 * 4).wait();
        static const uint32_t mag16[8] = {0x0000u, 0x3800u, 0x3C00u, 0x3E00u, 0x4000u, 0x4200u, 0x4400u, 0x4600u};   // 0 .5 1 1.5 2 3 4 6
        for (int b = 0; b < 256; ++b) L[b] = (mag16[b & 7] | ((b & 8) ? 0x8000u : 0u)) | ((mag16[(b >> 4) & 7] | ((b & 0x80) ? 0x8000u : 0u)) << 16);
        nvlut16_ = dev_.alloc<uint32_t>(256);
        q_.memcpy(nvlut16_, L.data(), 256 * 4).wait();
    }
    if (i4_ || i8v_) { kp_ = (h_ + 511) / 512 * 512; xpad_ = dev_.zeros<uint16_t>(size_t(3) * 8 * kp_); xpadU_ = dev_.zeros<uint16_t>(size_t(8) * kp_); }
    bestv_ = dev_.zeros<float>(16);
    if (world_ > 1) {
        peer_ = dev_.zeros<float>(mm * h_);
        for (int k = 0; k < TPX::NS; ++k) {  // this card's outgoing slots, pinned in its own context
            tpx_->hb[rank_][k] = sycl::malloc_host<float>(mm * h_, q_.get_context());
            tpx_->ib[rank_][k] = sycl::malloc_host<float>(size_t(64) * h_, q_.get_context());
            if (!tpx_->hb[rank_][k] || !tpx_->ib[rank_][k]) throw std::runtime_error("pinned host exchange buffers");
        }
    }
    amx_v_ = dev_.zeros<float>(4096); amx_i_ = dev_.zeros<int>(4096);
    tok_ = dev_.zeros<int>(std::max<size_t>(mm, 8)); next_ = dev_.zeros<int>(16); pos_d_ = dev_.zeros<int>(1);
    vbuf_ = dev_.zeros<float>(S_.size() * 4 * 8 * size_t(h_));
    actS_ = dev_.zeros<float>(128 * size_t(c_.inter)); dbufS_ = dev_.zeros<float>(128 * size_t(h_));
    xgS_ = dev_.zeros<uint16_t>(128 * size_t(h_)); actSb_ = dev_.zeros<uint16_t>(128 * size_t(c_.inter));
    logitsM_ = dev_.zeros<float>(8 * size_t(c_.output_vocab));
    if (i8v_) logitsR_ = dev_.zeros<float>(8 * size_t(vrows_));
    if (i8v_) { int mx = 0; for (const auto& Ly : L_) mx = std::max(mx, Ly.NA + Ly.NB + Ly.NC); projAll_ = dev_.zeros<float>(8 * size_t(mx)); }
    amxm_v_ = dev_.zeros<float>(8 * 64); amxm_i_ = dev_.zeros<int>(8 * 64);
    g_info_ = dev_.zeros<int>(1024);
    {
        std::vector<SlotPtrs> sp(S_.size());
        for (size_t i = 0; i < S_.size(); ++i) {
            sp[i].ring[0] = S_[i].khist; sp[i].ring[1] = S_[i].vhist; sp[i].ring[2] = S_[i].ahist; sp[i].ring[3] = S_[i].mhist;
            sp[i].n[0] = K; sp[i].n[1] = K; sp[i].n[2] = h_; sp[i].n[3] = h_;
        }
        slotptr_ = dev_.alloc<SlotPtrs>(sp.size());
        q_.memcpy(slotptr_, sp.data(), sp.size() * sizeof(SlotPtrs)).wait();
    }
    if (mmax_ > 1) {
        inb_ = dev_.zeros<uint16_t>((mm * TK + 64) * h_);
        gather_ = dev_.zeros<int>(mm * TK); pairpos_ = dev_.zeros<int>(mm * TK);
        // expert scratch: row-major BF16 weights of one projection, fp32 GEMM rows, BF16 activations (+64 pad rows)
        const size_t R = mm * TK + 64;
        xs_ = ExpertScratch{dev_.alloc<uint32_t>(size_t(c_.inter) * h_), dev_.alloc<float>(R * std::max<size_t>(2 * c_.inter, h_)), dev_.alloc<bf16x>(R * c_.inter)};
        elist_ = sycl::malloc_host<float>(kTbl, q_.get_context());
        tbl_d_ = dev_.alloc<float>(kTbl);
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

uint16_t* Engine::tiled(const uint16_t* src, int N, int K) {
    if (N % 16 || K % 8) throw std::runtime_error("tiled: N % 16, K % 8");
    std::vector<uint16_t> T(size_t(N) * K);
    const size_t K8 = size_t(K) / 8;
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) T[(((size_t(n) / 16) * K8 + k / 8) * 16 + n % 16) * 8 + k % 8] = src[size_t(n) * K + k];
    uint16_t* d = dev_.alloc<uint16_t>(T.size());
    q_.memcpy(d, T.data(), T.size() * 2).wait();
    return d;
}

Q4 Engine::quant4(const uint16_t* src, int N, int K) { return quant4(src, N, K, {{N, std::string()}}); }

Q4 Engine::quant4(const uint16_t* src, int N, int K, const std::vector<std::pair<int, std::string>>& parts) {
    const int Kp = (K + 511) / 512 * 512;
    if (!imx_ && std::getenv("INK_IMATRIX")) {
        imx_ = std::make_unique<b70::ImportanceMatrix>();
        std::string err;
        if (!imx_->load(std::getenv("INK_IMATRIX"), err)) throw std::runtime_error("INK_IMATRIX: " + err);
        std::printf("imatrix: %zu entries from %s\n", imx_->entries(), std::getenv("INK_IMATRIX"));
    }
    bool any = false;
    for (auto& p : parts) any = any || (imx_ && !p.second.empty() && imx_->find(p.second, K, false));
    if (any) {
        // each part quantized with its own importance (K padded with zero importance), rows concatenated
        std::vector<uint8_t> pay, sc, zz;
        int64_t rb = 0; int rs = 0; int r0 = 0;
        for (auto& p : parts) {
            std::vector<float> f(size_t(p.first) * Kp, 0.f), imp(Kp, 0.f);
            for (int n = 0; n < p.first; ++n)
                for (int k = 0; k < K; ++k) f[size_t(n) * Kp + k] = bfw(src[size_t(r0 + n) * K + k]);
            const float* iv = imx_ ? imx_->find(p.second, K) : nullptr;
            if (iv) std::copy(iv, iv + K, imp.begin());
            else std::fill(imp.begin(), imp.begin() + K, 1.f);
            const b70::PackedWeight pw = b70::quantize(f.data(), p.first, Kp, b70::Fmt::INT4, imp.data());
            rb = pw.row_bytes; rs = pw.row_scales;
            pay.insert(pay.end(), pw.payload.begin(), pw.payload.end());
            sc.insert(sc.end(), pw.scales_raw.begin(), pw.scales_raw.end());
            zz.insert(zz.end(), pw.zeros.begin(), pw.zeros.end());
            r0 += p.first;
        }
        if (r0 != N) throw std::runtime_error("quant4: parts do not cover the rows");
        Q4 w;
        w.N = N; w.K = Kp; w.rb = rb; w.rs = rs;
        w.pay = dev_.alloc<uint8_t>(pay.size()); w.sc = dev_.alloc<uint16_t>(sc.size() / 2); w.z = dev_.alloc<uint8_t>(std::max<size_t>(zz.size(), 1));
        q_.memcpy(w.pay, pay.data(), pay.size()); q_.memcpy(w.sc, sc.data(), sc.size());
        if (!zz.empty()) q_.memcpy(w.z, zz.data(), zz.size());
        q_.wait();
        return w;
    }
    std::vector<float> f(size_t(N) * Kp, 0.f);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) f[size_t(n) * Kp + k] = bfw(src[size_t(n) * K + k]);
    const b70::PackedWeight pw = b70::quantize(f.data(), N, Kp, b70::Fmt::INT4);
    Q4 w;
    w.N = N; w.K = Kp; w.rb = pw.row_bytes; w.rs = pw.row_scales;
    w.pay = dev_.alloc<uint8_t>(pw.payload.size());
    w.sc = dev_.alloc<uint16_t>(pw.scales_raw.size() / 2);
    w.z = dev_.alloc<uint8_t>(std::max<size_t>(pw.zeros.size(), 1));
    q_.memcpy(w.pay, pw.payload.data(), pw.payload.size());
    q_.memcpy(w.sc, pw.scales_raw.data(), pw.scales_raw.size());
    if (!pw.zeros.empty()) q_.memcpy(w.z, pw.zeros.data(), pw.zeros.size());
    q_.wait();
    return w;
}

// rows [r0, r0 + n) of an INT4 weight through GRIMOIRE's small-M ESIMD kernel (libgrimoire_gemm launch_int4_smallm_wide)
bool Engine::i4_gemv(const Q4& q4, int r0, int n, const uint16_t* X, float* Y, int M) {
    if (!q4.pay) return false;
    b70::QuantWeight w;
    w.fmt = b70::Fmt::INT4; w.N = n; w.K = q4.K; w.row_bytes = q4.rb; w.row_scales = q4.rs;
    w.payload = q4.pay + size_t(r0) * q4.rb;
    w.scales = q4.sc + size_t(r0) * q4.rs;
    w.zeros = q4.z + size_t(r0) * q4.rs;
    if (!b70::int4_smallm_wide_ok(w, M, X, Y)) {
        static bool told = false;
        if (!told) { told = true; std::printf("INK_I4: int4_smallm refused N=%d K=%d M=%d (falls back)\n", n, q4.K, M); }
        return false;
    }
    b70::launch_int4_smallm_wide(q_, w, reinterpret_cast<const bf16x*>(X), Y, M);
    return true;
}

// int8 tiles [N/16][K/8][16][8] -> row-major [N][Kp], zero padding past K (same values as the decode path)
int8_t* Engine::rowq(const int8_t* T, int N, int K, int Kp) {
    int8_t* d = dev_.alloc<int8_t>(size_t(N) * Kp);
    const size_t K8 = size_t(K) / 8;
    q_.parallel_for<InkRowQ>(sycl::range<1>(size_t(N) * Kp), [=](sycl::id<1> id) {
        const size_t r = id[0] / Kp, k = id[0] % Kp;
        d[id[0]] = k < size_t(K) ? T[(((r / 16) * K8 + k / 8) * 16 + r % 16) * 8 + k % 8] : int8_t(0);
    });
    q_.wait();
    return d;
}

void Engine::dgemv8(const DLayer& L, int w, const float* x, float* y) {
    const int K = L.K8[w], Kp = L.KP8[w];
    static const int i8m_d = std::getenv("INK_I8M") ? std::atoi(std::getenv("INK_I8M")) : 0;
    if (i8m_d > 0 && L.T8[w] &&
        ink_es_gemv_i8m(q_, InkI8MArgs{L.T8[w], L.S8[w], L.N8[w], K, 8, x, x, x, size_t(K), y, y, y, size_t(L.N8[w]), size_t(L.N8[w]),
                                       size_t(L.N8[w]), L.N8[w], L.N8[w], 1, 0.f}, i8m_d))
        return;
    uint16_t* xb = d_.xb8;
    q_.parallel_for<InkDXB>(sycl::range<1>(size_t(8) * Kp), [=](sycl::id<1> id) {
        const size_t m = id[0] / Kp, k = id[0] % Kp;
        xb[id[0]] = k < size_t(K) ? tobf(x[m * K + k]) : uint16_t(0);
    });
    if (!i8_small(L.Q8[w], L.S8[w], 0, L.N8[w], Kp, xb, y, 8)) throw std::runtime_error("drafter INT8 layer refused");
}

void Engine::split_proj(int NA, int NB, int NC, int M) {
    const float* src = projAll_;
    float *a = projA_, *b = projB_, *c = projC_;
    const int N = NA + NB + NC;
    q_.parallel_for<InkSplitP>(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> id) {
        const int m = int(id[0] / N), r = int(id[0] % N);
        const float v = src[id[0]];
        if (r < NA) a[size_t(m) * NA + r] = v;
        else if (r < NA + NB) b[size_t(m) * NB + (r - NA)] = v;
        else c[size_t(m) * NC + (r - NA - NB)] = v;
    });
}

bool Engine::i8_small(const int8_t* W, const float* S, int r0, int n, int Kp, const uint16_t* X, float* Y, int M) {
    if (!W) return false;
    b70::QuantWeight w;
    w.fmt = b70::Fmt::INT8; w.N = n; w.K = Kp; w.row_bytes = Kp; w.row_scales = 1;
    w.payload = reinterpret_cast<const uint8_t*>(W) + size_t(r0) * Kp;
    w.scales = S + r0;
    if (!b70::fp8_smallm_ok(w, M, X, Y)) {
        static bool told = false;
        if (!told) { told = true; std::printf("INK_I8V: fp8_smallm refused N=%d K=%d M=%d r0=%d (falls back)\n", n, Kp, M, r0); }
        return false;
    }
    b70::launch_fp8_smallm(q_, w, reinterpret_cast<const bf16x*>(X), Y, M);
    return true;
}

bool Engine::gsm_dense(const uint16_t* W, int N, int K, const uint16_t* X, float* Y, int M) {
    b70::QuantWeight w;
    w.fmt = b70::Fmt::BF16; w.N = N; w.K = K; w.payload = reinterpret_cast<const uint8_t*>(W); w.row_bytes = int64_t(K) * 2;
    if (!b70::bf16_smallm_ok(w, M, X, Y)) {
        static bool told = false;
        if (!told) { told = true; std::printf("INK_GSM: bf16_smallm refused N=%d K=%d M=%d (falls back)\n", N, K, M); }
        return false;
    }
    b70::launch_bf16_smallm(q_, w, reinterpret_cast<const bf16x*>(X), Y, M);
    return true;
}

// INK_I8=1 (opt-in; changes numerics): a tiled BF16 weight as int8 in the same tile order, scale = row absmax / 127,
// q = rne(w / scale).  Decode reads half the bytes of the BF16 tiles.
void Engine::quant_tiled(const uint16_t* T, int N, int K, int8_t*& qo, float*& so) {
    qo = dev_.alloc<int8_t>(size_t(N) * K);
    so = dev_.alloc<float>(N);
    int8_t* q = qo;
    float* sc = so;
    const size_t K8 = size_t(K) / 8;
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(256), cgh);
        cgh.parallel_for<InkQuantT>(sycl::nd_range<1>(size_t(N) * 256, 256), [=](sycl::nd_item<1> it) {
            const size_t r = it.get_group(0);
            const int lid = int(it.get_local_id(0));
            auto at = [&](size_t k) { return (((r / 16) * K8 + k / 8) * 16 + r % 16) * 8 + k % 8; };
            float mx = 0.f;
            for (size_t k = lid; k < size_t(K); k += 256) mx = sycl::fmax(mx, sycl::fabs(bfw(T[at(k)])));
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            const float s = mx > 0.f ? mx / 127.f : 1.f;
            for (size_t k = lid; k < size_t(K); k += 256) q[at(k)] = int8_t(sycl::clamp(sycl::rint(bfw(T[at(k)]) / s), -127.f, 127.f));
            if (lid == 0) sc[r] = s;
        });
    });
    q_.wait();
}

template <int MODE>
void Engine::gemvt8(const int8_t* T, const float* sc, int N, int K, int KS, const float* x0, float* y0, int end0, const float* x1, float* y1,
                    int end1, const float* x2, float* y2, float cap) {
    const int tiles = N / 16, K8 = K / 8, per = K8 / KS;
    if (N % 16 || K8 % KS) throw std::runtime_error("gemvt8 geometry");
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(size_t(KS) * 16), cgh);
        cgh.parallel_for<InkGemvT8<MODE>>(sycl::nd_range<1>(size_t(tiles) * KS * 16, size_t(KS) * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            const int tile = int(it.get_group(0)), q = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
            const int r0 = tile * 16;
            const float* x = r0 < end0 ? x0 : (r0 < end1 ? x1 : x2);
            const int8_t* w = T + size_t(tile) * K8 * 128 + lane * 8;
            float a0 = 0.f, a1 = 0.f;
            for (int ko = q * per; ko < (q + 1) * per; ++ko) {
                const sycl::vec<int8_t, 8> wv = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(w + size_t(ko) * 128);
                const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(x + ko * 8);
                const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(x + ko * 8 + 4);
                a0 = sycl::fma(float(wv[0]), a[0], a0); a1 = sycl::fma(float(wv[1]), a[1], a1);
                a0 = sycl::fma(float(wv[2]), a[2], a0); a1 = sycl::fma(float(wv[3]), a[3], a1);
                a0 = sycl::fma(float(wv[4]), b[0], a0); a1 = sycl::fma(float(wv[5]), b[1], a1);
                a0 = sycl::fma(float(wv[6]), b[2], a0); a1 = sycl::fma(float(wv[7]), b[3], a1);
            }
            float acc = a0 + a1;
            if (KS > 1) {
                red[q * 16 + lane] = acc;
                sycl::group_barrier(it.get_group());
                if (q == 0) { acc = 0.f; for (int t = 0; t < KS; ++t) acc += red[t * 16 + lane]; }
            }
            if (q == 0) {
                const int r = r0 + lane;
                acc *= sc[r];
                float* y = r < end0 ? y0 + r : (r < end1 ? y1 + (r - end0) : y2 + (r - end1));
                if constexpr (MODE == 0) *y = bfr(acc);
                else if constexpr (MODE == 1) *y = acc;
                else { const float lin = bfr(acc); *y = cap * sycl::tanh(lin / cap); }
            }
        });
    });
}

// Decode GEMV on the tiled layout (urakozz's lane-per-column idea): a sub-group owns 16 output
// columns, each lane one column; KS sub-groups split K and merge through SLM in a fixed order.
template <int MODE>
void Engine::gemvt(const uint16_t* T, int N, int K, int KS, const float* x0, float* y0, int end0, const float* x1, float* y1, int end1,
                   const float* x2, float* y2) {
    const int tiles = N / 16, K8 = K / 8, per = K8 / KS;
    if (N % 16 || K8 % KS) throw std::runtime_error("gemvt geometry");
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(size_t(KS) * 16), cgh);
        cgh.parallel_for<InkGemvT<MODE>>(sycl::nd_range<1>(size_t(tiles) * KS * 16, size_t(KS) * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            const int tile = int(it.get_group(0)), q = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
            const int r0 = tile * 16;
            const float* x = r0 < end0 ? x0 : (r0 < end1 ? x1 : x2);
            const uint16_t* w = T + size_t(tile) * K8 * 128 + lane * 8;
            float a0 = 0.f, a1 = 0.f;
            for (int ko = q * per; ko < (q + 1) * per; ++ko) {
                const sycl::vec<uint16_t, 8> wv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(w + size_t(ko) * 128);
                const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(x + ko * 8);
                const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(x + ko * 8 + 4);
                a0 = sycl::fma(bfw(wv[0]), a[0], a0); a1 = sycl::fma(bfw(wv[1]), a[1], a1);
                a0 = sycl::fma(bfw(wv[2]), a[2], a0); a1 = sycl::fma(bfw(wv[3]), a[3], a1);
                a0 = sycl::fma(bfw(wv[4]), b[0], a0); a1 = sycl::fma(bfw(wv[5]), b[1], a1);
                a0 = sycl::fma(bfw(wv[6]), b[2], a0); a1 = sycl::fma(bfw(wv[7]), b[3], a1);
            }
            float acc = a0 + a1;
            if (KS > 1) {
                red[q * 16 + lane] = acc;
                sycl::group_barrier(it.get_group());
                if (q == 0) { acc = 0.f; for (int t = 0; t < KS; ++t) acc += red[t * 16 + lane]; }
            }
            if (q == 0) {
                const int r = r0 + lane;
                float* y = r < end0 ? y0 + r : (r < end1 ? y1 + (r - end0) : y2 + (r - end1));
                if constexpr (MODE == 0) *y = bfr(acc);
                else *y = acc;
            }
        });
    });
}

template <int MODE>
void Engine::gemvtm(const uint16_t* T, int N, int K, int KS, int M, const float* x0, const float* x1, const float* x2, size_t ldx,
                    float* y0, size_t ld0, int end0, float* y1, size_t ld1, int end1, float* y2, size_t ld2, int nvalid, float cap) {
    const int tiles = N / 16, K8 = K / 8, per = K8 / KS;
    if (N % 16 || K8 % KS || M < 1 || M > 8) throw std::runtime_error("gemvtm geometry");
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(KS > 1 ? size_t(KS) * 16 * 8 : 1), cgh);
        cgh.parallel_for<InkGemvTM<MODE>>(sycl::nd_range<1>(size_t(tiles) * KS * 16, size_t(KS) * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            const int tile = int(it.get_group(0)), q = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
            const int r0 = tile * 16;
            const float* x = r0 < end0 ? x0 : (r0 < end1 ? x1 : x2);
            const uint16_t* w = T + size_t(tile) * K8 * 128 + lane * 8;
            float acc[8];
#pragma unroll
            for (int m = 0; m < 8; ++m) acc[m] = 0.f;
            for (int ko = q * per; ko < (q + 1) * per; ++ko) {
                const sycl::vec<uint16_t, 8> wv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(w + size_t(ko) * 128);
                float wf[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) wf[e] = bfw(wv[e]);
#pragma unroll
                for (int m = 0; m < 8; ++m)
                    if (m < M) {
                        const float* xr = x + m * ldx + ko * 8;
                        const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(xr);
                        const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(xr + 4);
                        float s = acc[m];
                        s = sycl::fma(wf[0], a[0], s); s = sycl::fma(wf[1], a[1], s); s = sycl::fma(wf[2], a[2], s); s = sycl::fma(wf[3], a[3], s);
                        s = sycl::fma(wf[4], b[0], s); s = sycl::fma(wf[5], b[1], s); s = sycl::fma(wf[6], b[2], s); s = sycl::fma(wf[7], b[3], s);
                        acc[m] = s;
                    }
            }
            if (KS > 1) {
#pragma unroll
                for (int m = 0; m < 8; ++m) red[(q * 8 + m) * 16 + lane] = acc[m];
                sycl::group_barrier(it.get_group());
                if (q == 0)
#pragma unroll
                    for (int m = 0; m < 8; ++m) { float t = 0.f; for (int k = 0; k < KS; ++k) t += red[(k * 8 + m) * 16 + lane]; acc[m] = t; }
            }
            const int r = r0 + lane;
            if (q == 0 && r < nvalid) {
                float* y;
                size_t ld;
                if (r < end0) { y = y0 + r; ld = ld0; }
                else if (r < end1) { y = y1 + (r - end0); ld = ld1; }
                else { y = y2 + (r - end1); ld = ld2; }
#pragma unroll
                for (int m = 0; m < 8; ++m)
                    if (m < M) {
                        if constexpr (MODE == 0) y[m * ld] = bfr(acc[m]);
                        else if constexpr (MODE == 1) y[m * ld] = acc[m];
                        else { const float lin = bfr(acc[m]); y[m * ld] = cap * sycl::tanh(lin / cap); }
                    }
            }
        });
    });
}

void Engine::dense(const uint16_t* W, int N, int K, const float* x, size_t ldx, const uint16_t* xb, float* y, size_t ldy, int M, bool round) {
    // full prompt chunks: oneDNN (faster; its per-M kernels are built once for M = chunk); any other row count:
    // the own BF16 GEMM, which needs no per-M setup (oneDNN cost ~450 ms on a new 512-row prompt)
    static const bool use_onednn = std::getenv("INK_DENSE_ONEDNN") != nullptr;
    if (M >= 32 && xb && !use_onednn && K % 32 == 0 && M != mmax_) {
        if (ldy != size_t(N)) throw std::runtime_error("dense: output pitch must be N");
        ink_dense_bf16(q_, W, N, K, xb, y, M);
        return;
    }
    if (M >= 32 && xb) {
        b70::QuantWeight w;
        w.fmt = b70::Fmt::BF16; w.N = N; w.K = K; w.payload = reinterpret_cast<const uint8_t*>(W); w.row_bytes = int64_t(K) * 2;
        if (!b70::gemm_fast_supported(w, M) || ldy != size_t(N)) throw std::runtime_error("GEMM shape not supported");
        b70::launch_gemm_fast(q_, w, reinterpret_cast<const bf16x*>(xb), y, M);
        return;
    }
    if (M == 1) {
        if (round) gemv<0>(W, N, K, x, y, N, nullptr, nullptr, N, nullptr, nullptr);
        else gemv<1>(W, N, K, x, y, N, nullptr, nullptr, N, nullptr, nullptr);
        return;
    }
    for (int m0 = 0; m0 < M; m0 += 8) {
        const int n = std::min(8, M - m0);
        if (round) gemvm<0>(W, N, K, x + m0 * ldx, ldx, y + m0 * ldy, ldy, n);
        else gemvm<1>(W, N, K, x + m0 * ldx, ldx, y + m0 * ldy, ldy, n);
    }
}

// M = 1: residual = sconv(z) (+ hidden, or a bank push of hidden), the MLP bank mix and, when nout >= 0, its norm --
// one work-group, three launches' work (InkResid + InkMixS + InkRms).  A thread writes RES / the bank row for its own
// elements and reads only those back, and every reduction runs in the separate kernels' order: same bits.
void Engine::resid_mix_norm(const uint16_t* aconv, float* ahist, int br, const MixJob& J, const uint16_t* nw, int nout) {
    float* pool = pool_;
    const int h = h_;
    const size_t RS = rs_;
    const float f = 1.f / std::sqrt(float(h));
    const int* posd = pos_d_;
    const int taps = c_.conv;
    const float cs = 1.f / std::sqrt(float(1 + c_.conv)), eps = c_.eps;
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(64 * 2 * kMixRows), cgh);
        sycl::local_accessor<float, 1> tot(sycl::range<1>(2 * kMixRows), cgh);
        cgh.parallel_for<InkRMN>(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int lid = int(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
            const int pos = *posd;
            // residual (InkResid, M = 1)
            for (int ch = lid; ch < h; ch += 1024) {
                const float* zr = pool + size_t(Z) * RS;
                const float x = bfr(zr[ch]);
                float sum = 0.f;
                for (int lag = 0; lag <= sycl::min(pos, taps - 1); ++lag) {
                    const float xl = lag == 0 ? x : ahist[size_t((pos - lag) % (taps - 1)) * h + ch];
                    sum += xl * bfw(aconv[ch * taps + taps - 1 - lag]);
                }
                const float out = bfr((x + sum) * cs);
                ahist[size_t(pos % (taps - 1)) * h + ch] = x;
                const float hid = pool[size_t(HID) * RS + ch];
                if (br >= 0) { pool[size_t(br) * RS + ch] = hid; pool[size_t(RES) * RS + ch] = out; }
                else pool[size_t(RES) * RS + ch] = out + hid;
            }
            // bank mix (InkMixS)
            const int n = J.n;
            float ss[kMixRows], qd[kMixRows];
#pragma unroll
            for (int r = 0; r < kMixRows; ++r) { ss[r] = 0.f; qd[r] = 0.f; }
            for (int i = lid; i < h; i += 1024) {
                const float qv = J.query[i];
#pragma unroll
                for (int r = 0; r < kMixRows; ++r)
                    if (r < n) { const float v = pool[size_t(J.rows[r]) * RS + i]; ss[r] = sycl::fma(v, v, ss[r]); qd[r] = sycl::fma(qv, v, qd[r]); }
            }
#pragma unroll
            for (int r = 0; r < kMixRows; ++r)
                if (r < n) {
                    const float a = sycl::reduce_over_group(sg, ss[r], sycl::plus<float>());
                    const float c = sycl::reduce_over_group(sg, qd[r], sycl::plus<float>());
                    if (lane == 0) { red[sgid * 2 * kMixRows + r] = a; red[sgid * 2 * kMixRows + kMixRows + r] = c; }
                }
            sycl::group_barrier(it.get_group());
            if (lid < 2 * kMixRows) { float t = 0.f; for (int g = 0; g < 64; ++g) t += red[g * 2 * kMixRows + lid]; tot[lid] = t; }
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
            float* out = pool + size_t(J.out) * RS;
            float o3[3];
            float nss = 0.f;
#pragma unroll
            for (int t = 0; t < 3; ++t) {
                const int i = lid + t * 1024;
                o3[t] = 0.f;
                if (i >= h) continue;
                float o = 0.f;
#pragma unroll
                for (int r = 0; r < kMixRows; ++r)
                    if (r < n) o += (pool[size_t(J.rows[r]) * RS + i] * inv[r]) * (sc[r] / sum);
                out[i] = o;
                o3[t] = o;
                nss += o * o;
            }
            if (nout < 0) return;
            // its norm (InkRms, M = 1: the same per-thread sum and group reduction)
            nss = sycl::reduce_over_group(it.get_group(), nss, sycl::plus<float>());
            const float rr = 1.f / sycl::sqrt(nss / float(h) + eps);
            float* y = pool + size_t(nout) * RS;
#pragma unroll
            for (int t = 0; t < 3; ++t) {
                const int i = lid + t * 1024;
                if (i < h) y[i] = bfr((o3[t] * rr) * bfw(nw[i]));
            }
        });
    });
}

void Engine::mix_small(const MixBatch& b, int M, int m0) {
    float* pool = pool_;
    const int h = h_;
    const size_t RS = rs_;
    const float f = 1.f / std::sqrt(float(h));
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(64 * 2 * kMixRows), cgh);
        sycl::local_accessor<float, 1> tot(sycl::range<1>(2 * kMixRows), cgh);
        cgh.parallel_for<InkMixS>(sycl::nd_range<1>(size_t(b.count) * M * 1024, 1024), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
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
            for (int i = lid; i < h; i += 1024) {
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
            if (lid < 2 * kMixRows) { float t = 0.f; for (int g = 0; g < 64; ++g) t += red[g * 2 * kMixRows + lid]; tot[lid] = t; }
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
            for (int i = lid; i < h; i += 1024) {
                float o = 0.f;
#pragma unroll
                for (int r = 0; r < kMixRows; ++r)
                    if (r < n) o += (pool[size_t(J.rows[r]) * RS + mo + i] * inv[r]) * (sc[r] / sum);
                out[i] = o;
            }
        });
    });
}

void Engine::mix(const MixBatch& b, int M, int m0) {
    // decode / verify rows: the 1024-thread kernel (more work-groups' worth of threads for few rows)
    if (M * b.count < 64) { mix_small(b, M, m0); return; }
    // prompt rows: one 256-thread work-group per (job, row); each thread keeps its <= 9 elements of every bank row in registers
    float* pool = pool_;
    const int h = h_;
    const size_t RS = rs_;
    const float f = 1.f / std::sqrt(float(h));
    constexpr int WG = 256, PT = 9;
    if (h > WG * PT) throw std::runtime_error("mix: hidden > 2304");
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(16 * 2 * kMixRows), cgh);
        sycl::local_accessor<float, 1> tot(sycl::range<1>(2 * kMixRows), cgh);
        cgh.parallel_for<InkMix>(sycl::nd_range<1>(size_t(b.count) * M * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int gi = int(it.get_group(0));
            const MixJob& J = b.j[gi / M];
            const size_t mo = size_t(m0 + gi % M) * h;
            const int lid = int(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const int sgid = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
            const int n = J.n;
            float v[kMixRows][PT], qv[PT];
#pragma unroll
            for (int t = 0; t < PT; ++t) { const int i = lid + t * WG; qv[t] = i < h ? J.query[i] : 0.f; }
            float ss[kMixRows], qd[kMixRows];
#pragma unroll
            for (int r = 0; r < kMixRows; ++r) {
                ss[r] = 0.f; qd[r] = 0.f;
#pragma unroll
                for (int t = 0; t < PT; ++t) {
                    const int i = lid + t * WG;
                    v[r][t] = (r < n && i < h) ? pool[size_t(J.rows[r]) * RS + mo + i] : 0.f;
                    ss[r] = sycl::fma(v[r][t], v[r][t], ss[r]);
                    qd[r] = sycl::fma(qv[t], v[r][t], qd[r]);
                }
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
#pragma unroll
            for (int t = 0; t < PT; ++t) {
                const int i = lid + t * WG;
                if (i >= h) continue;
                float o = 0.f;
#pragma unroll
                for (int r = 0; r < kMixRows; ++r)
                    if (r < n) o += (v[r][t] * inv[r]) * (sc[r] / sum);
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
    const int WGN = M == 1 ? 1024 : 256;
    q_.parallel_for<InkRms>(sycl::nd_range<1>(size_t(b.count) * M * WGN, WGN), [=](sycl::nd_item<1> it) {
        const int gi = int(it.get_group(0));
        const NormJob& J = b.j[gi / M];
        const int m = m0 + gi % M;
        const int lid = int(it.get_local_id(0));
        const float* x = pool + size_t(J.in) * RS + size_t(m) * h;
        float ss = 0.f;
        for (int i = lid; i < h; i += WGN) ss += x[i] * x[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.f / sycl::sqrt(ss / float(h) + eps);
        float* y = pool + size_t(J.out) * RS + size_t(m) * h;
        for (int i = lid; i < h; i += WGN) {
            const float v = bfr((x[i] * r) * bfw(J.w[i]));
            y[i] = v;
            if (J.outb) J.outb[size_t(m) * (J.ldb ? J.ldb : h) + i] = tobf(v);
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
    static const bool old_gates = std::getenv("INK_OLD_GATES") != nullptr;
    if (M == 1 && !old_gates && b.count <= 3) {
        // decode: low[j][r] = bf(cdown[r] . [bf(cur * ir), bf(msg * im)]) -- one 256-thread work-group per output row,
        // each recomputing the two norms exactly as InkGateCat (same per-thread order and group reduction): one
        // launch over rank x jobs work-groups instead of InkGateCat (one work-group a job) + a 4-work-group GEMV
        struct CD { const uint16_t* w[3]; } cd{};
        for (int j = 0; j < b.count; ++j) cd.w[j] = Ly.cdown[ctrl[j]];
        q_.parallel_for<InkGateLow>(sycl::nd_range<1>(size_t(b.count) * rank * 256, 256), [=](sycl::nd_item<1> it) {
            const int gi = int(it.get_group(0)), jb = gi / rank, r = gi % rank;
            const GateJob J = jb == 0 ? b.j[0] : (jb == 1 ? b.j[1] : b.j[2]);   // no dynamic index (it spilled 5.6 KB)
            const int lid = int(it.get_local_id(0));
            const float* cur = pool + size_t(J.cur) * RS;
            const float* msg = pool + size_t(J.msg) * RS;
            float sc = 0.f, sm = 0.f;
            for (int i = lid; i < h; i += 256) { sc += cur[i] * cur[i]; sm += msg[i] * msg[i]; }
            sc = sycl::reduce_over_group(it.get_group(), sc, sycl::plus<float>());
            sm = sycl::reduce_over_group(it.get_group(), sm, sycl::plus<float>());
            const float ir = 1.f / sycl::sqrt(sc / float(h) + eps), im = 1.f / sycl::sqrt(sm / float(h) + eps);
            const uint16_t* wr = (jb == 0 ? cd.w[0] : (jb == 1 ? cd.w[1] : cd.w[2])) + size_t(r) * 2 * h;
            float acc = 0.f;
            for (int i = lid; i < h; i += 256) acc += bfr(cur[i] * ir) * bfw(wr[i]) + bfr(msg[i] * im) * bfw(wr[h + i]);
            acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
            if (lid == 0) low[size_t(jb) * MM * rank + r] = bfr(acc);
        });
    } else {
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
        if (M == 1) gemvt<0>(Ly.cdownT[ctrl[j]], rank, 2 * h, 32, xcat + size_t(j) * MM * 2 * h, low + size_t(j) * MM * rank, rank, nullptr, nullptr, rank, nullptr, nullptr);
        else if (M <= 8 && gsmv_ && gsm_dense(Ly.cdown[ctrl[j]], rank, 2 * h, xcatb + size_t(j) * MM * 2 * h, low + size_t(j) * MM * rank, M)) {
        } else if (M <= 8) {
            const float* xj = xcat + size_t(j) * MM * 2 * h;
            gemvtm<0>(Ly.cdownT[ctrl[j]], rank, 2 * h, 32, M, xj, xj, xj, 2 * h, low + size_t(j) * MM * rank, rank, rank, nullptr, 0, rank, nullptr, 0, rank);
        }
        else dense(Ly.cdown[ctrl[j]], rank, 2 * h, xcat + size_t(j) * MM * 2 * h, 2 * h, xcatb + size_t(j) * MM * 2 * h, low + size_t(j) * MM * rank, rank, M, true);
    }
    if (M >= 32) {
        // prompt rows: gain / forget as two GEMMs over the BF16-rounded low rows (K = rank), then an elementwise blend
        if (!gate_g_) { gate_g_ = dev_.alloc<float>(size_t(MM) * h); gate_f_ = dev_.alloc<float>(size_t(MM) * h); lowb_ = dev_.alloc<uint16_t>(size_t(MM) * rank); }
        float *G = gate_g_, *F = gate_f_;
        uint16_t* lb = lowb_;
        for (int j = 0; j < b.count; ++j) {
            const float* lw = low + size_t(j) * MM * rank;
            q_.parallel_for<InkLowB>(sycl::range<1>(size_t(M) * rank), [=](sycl::id<1> id) { lb[id[0]] = tobf(bfr(lw[id[0]])); });
            b70::QuantWeight wg, wf;
            wg.fmt = wf.fmt = b70::Fmt::BF16; wg.N = wf.N = h; wg.K = wf.K = rank; wg.row_bytes = wf.row_bytes = int64_t(rank) * 2;
            wg.payload = reinterpret_cast<const uint8_t*>(b.j[j].gain);
            wf.payload = reinterpret_cast<const uint8_t*>(b.j[j].forget);
            if (rank % 32 == 0 && !std::getenv("INK_DENSE_ONEDNN") && M != mmax_) {
                ink_dense_bf16(q_, b.j[j].gain, h, rank, lb, G, M);
                ink_dense_bf16(q_, b.j[j].forget, h, rank, lb, F, M);
            } else {
                b70::launch_gemm_fast(q_, wg, reinterpret_cast<const bf16x*>(lb), G, M);
                b70::launch_gemm_fast(q_, wf, reinterpret_cast<const bf16x*>(lb), F, M);
            }
            const GateJob J = b.j[j];
            q_.parallel_for<InkGateBlendM>(sycl::range<1>(size_t(M) * h), [=](sycl::id<1> id) {
                const size_t t = id[0];
                const float c = pool[size_t(J.cur) * RS + t], mm = pool[size_t(J.msg) * RS + t];
                pool[size_t(J.out) * RS + t] = (1.f - sycl::tanh(F[t])) * c + sycl::tanh(G[t]) * mm;
            });
        }
        return;
    }
    static const bool blend2 = !std::getenv("INK_OLD_BLEND");
    static const bool blende = !std::getenv("INK_NO_BLENDE");   // ESIMD DPAS gate blend (GRIMOIRE small-M pattern)
    if (blend2 && blende && rank == 64 && M >= 2 && M <= 8 && h % 32 == 0 && (RS * 4) % 64 == 0) {
        InkGBArgs ga{};
        ga.low = low; ga.ldlow = int(size_t(MM) * rank);
        for (int j = 0; j < b.count; ++j) { ga.gain[j] = b.j[j].gain; ga.forget[j] = b.j[j].forget; ga.cur[j] = b.j[j].cur; ga.msg[j] = b.j[j].msg; ga.out[j] = b.j[j].out; }
        ga.pool = pool; ga.RS = RS; ga.count = b.count; ga.M = M; ga.h = h;
        ink_gate_blend_esimd(q_, ga);
        return;
    }
    if (blend2 && rank == 64 && M >= 2 && M <= 8 && h % 128 == 0) {
        // one work-item = one output column of one job for all M rows: gain / forget rows read once (8 bf16 at a time),
        // the BF16-rounded low rows from SLM (the same value for the whole work-group: broadcast reads)
        const int count = b.count;
        const GateJob J0 = b.j[0], J1 = b.j[1], J2 = b.j[2];
        q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> ls(sycl::range<1>(3 * 8 * 64), cgh);
            cgh.parallel_for<InkGateBlend2>(sycl::nd_range<1>(size_t(count) * h, 128), [=](sycl::nd_item<1> it) {
                const int lid = int(it.get_local_id(0));
                const size_t g = it.get_global_id(0);
                const int jb = int(g / h), i = int(g % h);
                for (int t = lid; t < M * 64; t += 128) ls[(jb * 8 + t / 64) * 64 + t % 64] = bfr(low[(size_t(jb) * MM + t / 64) * 64 + t % 64]);
                sycl::group_barrier(it.get_group());
                const GateJob J = jb == 0 ? J0 : (jb == 1 ? J1 : J2);
                float ga[8], fa[8];
#pragma unroll
                for (int m = 0; m < 8; ++m) { ga[m] = 0.f; fa[m] = 0.f; }
                const sycl::uint4* gr = reinterpret_cast<const sycl::uint4*>(J.gain + size_t(i) * 64);
                const sycl::uint4* fr = reinterpret_cast<const sycl::uint4*>(J.forget + size_t(i) * 64);
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    const sycl::uint4 gv = gr[c], fv = fr[c];
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const uint32_t gw = gv[e >> 1], fw = fv[e >> 1];
                        const float gwe = bfw(uint16_t((e & 1) ? gw >> 16 : gw & 0xFFFFu)), fwe = bfw(uint16_t((e & 1) ? fw >> 16 : fw & 0xFFFFu));
#pragma unroll
                        for (int m = 0; m < 8; ++m)
                            if (m < M) { const float lv = ls[(jb * 8 + m) * 64 + c * 8 + e]; ga[m] += lv * gwe; fa[m] += lv * fwe; }
                    }
                }
#pragma unroll
                for (int m = 0; m < 8; ++m)
                    if (m < M) {
                        const size_t o = size_t(m) * h + i;
                        const float cv = pool[size_t(J.cur) * RS + o], mv = pool[size_t(J.msg) * RS + o];
                        pool[size_t(J.out) * RS + o] = (1.f - sycl::tanh(fa[m])) * cv + sycl::tanh(ga[m]) * mv;
                    }
            });
        });
        return;
    }
    const size_t total = size_t(b.count) * M * h, groups = (total + 7) / 8;
    q_.parallel_for<InkGateBlend>(sycl::nd_range<1>(groups * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const size_t t = it.get_group(0) * 8 + sg.get_group_linear_id();
        if (t >= total) return;
        const int i = int(t % h), m = int((t / h) % M), jb = int(t / (size_t(h) * M));
        const GateJob J = jb == 0 ? b.j[0] : (jb == 1 ? b.j[1] : b.j[2]);   // no dynamic index (it spilled 2.4 KB)
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
        wp[2 * lid] = t ? 0.f : sgn * mag;
        wp[2 * lid + 1] = t ? sgn * mag : 0.f;
        wt[lid] = 0.f;
    }
}
inline void nv8(uint32_t w, const float* xr, const float* wp, const float* wt, float& A, float& B) {
    // wp and wt interleaved: wp[2n] = class-1 weight, wp[2n+1] = class-1.5 weight (one 8-byte SLM read)
    (void)wt;
    const sycl::float2* t2 = reinterpret_cast<const sycl::float2*>(wp);
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const uint32_t n = (w >> (4 * t)) & 15u;
        const sycl::float2 v = t2[n];
        A = sycl::fma(v[0], xr[t], A);
        B = sycl::fma(v[1], xr[t], B);
    }
}
// NVFP4 block table: value(code c) = bfr((e2m1(c)*sc)*outer) = {0, q1/2, q1, q3, 2q1, 2q3, 4q1, 4q3} (exact).
inline sycl::vec<float, 8> nv_tab(float sc, float outer) {
    const float q1 = bfrf(sc * outer), q3 = bfrf((1.5f * sc) * outer);
    return sycl::vec<float, 8>(0.f, 0.5f * q1, q1, q3, 2.f * q1, 2.f * q3, 4.f * q1, 4.f * q3);
}
inline float nvt8(uint32_t w, const float* xr, const sycl::vec<float, 8>& tv, float acc) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const uint32_t n = (w >> (4 * t)) & 15u;
        const float v = tv[n & 7u];
        acc = sycl::fma((n & 8u) ? -v : v, xr[t], acc);
    }
    return acc;
}
// 16-entry code tables held in the sub-group's registers (lane j = code j), read by shuffle: no SLM traffic.
inline float tab_p(int lane) {
    const int c = lane & 7;
    const float sgn = (lane & 8) ? -1.f : 1.f;
    const float mag = c == 0 ? 0.f : (c == 1 ? 0.5f : (c <= 3 ? 1.f : (c <= 5 ? 2.f : 4.f)));
    return (c == 3 || c == 5 || c == 7) ? 0.f : sgn * mag;
}
inline float tab_t(int lane) {
    const int c = lane & 7;
    const float sgn = (lane & 8) ? -1.f : 1.f;
    const float mag = c <= 3 ? 1.f : (c <= 5 ? 2.f : 4.f);
    return (c == 3 || c == 5 || c == 7) ? sgn * mag : 0.f;
}
inline void nvs8(sycl::sub_group sg, uint32_t w, const float* xr, float tp, float tt, float& A, float& B) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const uint32_t n = (w >> (4 * t)) & 15u;
        A = sycl::fma(sycl::select_from_group(sg, tp, n), xr[t], A);
        B = sycl::fma(sycl::select_from_group(sg, tt, n), xr[t], B);
    }
}
// Byte table: tb[b] = (class-1 lo, class-1.5 lo, class-1 hi, class-1.5 hi) for byte b (two E2M1 codes).
inline void fill_btab(sycl::float4* tb, const float* wp, int lid, int lsz) {
    for (int b = lid; b < 256; b += lsz) tb[b] = sycl::float4(wp[2 * (b & 15)], wp[2 * (b & 15) + 1], wp[2 * (b >> 4)], wp[2 * (b >> 4) + 1]);
}
inline void nvb8(uint32_t w, const float* xr, const sycl::float4* tb, float& A, float& B) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const sycl::float4 t = tb[(w >> (8 * j)) & 255u];
        A = sycl::fma(t[0], xr[2 * j], A); B = sycl::fma(t[1], xr[2 * j], B);
        A = sycl::fma(t[2], xr[2 * j + 1], A); B = sycl::fma(t[3], xr[2 * j + 1], B);
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
    const int h = h_, I = c_.inter, TK = c_.top_k, e0 = e0_, en = en_, own = own_;
    // INK_ESMOE=1: GRIMOIRE's Ornith ESIMD decode MoE ported to NVFP4 (tools/inkling2_moe_gemm.cpp ink_es_moe)
    static const bool esmoe = std::getenv("INK_ESMOE") && std::atoi(std::getenv("INK_ESMOE")) == 1;
    if (esmoe && Ly.fgu == PackedFormat::NVFP4 && Ly.fdn == PackedFormat::NVFP4 &&
        ink_es_moe(q_, InkEsArgs{Ly.gu, Ly.gus, Ly.gua, Ly.gud, sel_, e0, en, TK, row(NM), act_, h, I, c_.gate_cap, c_.up_cap, own_},
                   InkEsArgs{Ly.dn, Ly.dns, Ly.dna, Ly.dnd, sel_, e0, en, TK, act_, dbuf_, h, I, 0.f, 0.f, own_}))
        return;
    if (Ly.guX) {   // DPAS decode (tools/inkling2_moe_gemm.cpp ink_xexp)
        ink_xexp(q_, InkXArgs{Ly.guX, Ly.gusX, Ly.gua, Ly.gud, sel_, e0, en, TK, 2 * I, h, row(NM), 0, act_, 1, c_.gate_cap, c_.up_cap, xlut_});
        ink_xexp(q_, InkXArgs{Ly.dnX, Ly.dnsX, Ly.dna, Ly.dnd, sel_, e0, en, TK, h, I, act_, size_t(I), dbuf_, 0, 0.f, 0.f, xlut_});
        return;
    }
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
        const int wpb = I / (R * SGS);
        constexpr int steps = 9;
        if (h != 256 * steps) throw std::runtime_error("decode experts: hidden 2304 only");
        q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> e4s(256, cgh), wps(32, cgh), wts(16, cgh);
            sycl::local_accessor<sycl::float4, 1> tbs(256, cgh);
            cgh.parallel_for<InkGu>(sycl::nd_range<1>(size_t(TK) * wpb * SGS * 16, SGS * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid = int(it.get_local_id(0));
                float* e4 = e4s.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wp = wps.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wt = wts.template get_multi_ptr<sycl::access::decorated::no>().get();
                sycl::float4* tb = tbs.template get_multi_ptr<sycl::access::decorated::no>().get();
                fill_tables(e4, wp, wt, lid, SGS * 16);
                sycl::group_barrier(it.get_group());
                fill_btab(tb, wp, lid, SGS * 16);
                sycl::group_barrier(it.get_group());
                auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_linear_id());
                const int s = int(it.get_group(0)) / wpb;
                const int i0 = (int(it.get_group(0)) % wpb) * R * SGS + int(sg.get_group_linear_id()) * R;
                const int e = sel[s];
                if (own >= 0 ? (s & 1) != own : (e < e0 || e >= e0 + en)) return;
                const int el = own >= 0 ? e : e - e0;
                const float outer = NV ? amax[e] / den[0] : 1.f;
                float g[R], u[R];
#pragma unroll
                for (int r = 0; r < R; ++r) { g[r] = 0.f; u[r] = 0.f; }
#pragma unroll
                for (int k = 0; k < steps; ++k) {
                    const int blk = k * 16 + lane;
                    float xr[16];
                    load_x(x + size_t(blk) * 16, xr, 16);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const size_t gr = size_t(el) * 2 * I + 2 * size_t(i0 + r), ur = gr + 1;
                        if (NV) {
                            const sycl::uint2 pg = *reinterpret_cast<const sycl::uint2*>(W + gr * rb + size_t(blk) * 8);
                            const sycl::uint2 pu = *reinterpret_cast<const sycl::uint2*>(W + ur * rb + size_t(blk) * 8);
                            float ag = 0.f, bg = 0.f, au = 0.f, bu = 0.f;
                            nvb8(pg[0], xr, tb, ag, bg); nvb8(pg[1], xr + 8, tb, ag, bg);
                            nvb8(pu[0], xr, tb, au, bu); nvb8(pu[1], xr + 8, tb, au, bu);
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
        const int wpb = h / (R * SGS);
        constexpr int steps = 9;
        if (I != 128 * steps) throw std::runtime_error("decode experts: inter 1152 only");
        q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> e4s(256, cgh), wps(32, cgh), wts(16, cgh);
            sycl::local_accessor<sycl::float4, 1> tbs(256, cgh);
            cgh.parallel_for<InkDown>(sycl::nd_range<1>(size_t(TK) * wpb * SGS * 16, SGS * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid = int(it.get_local_id(0));
                float* e4 = e4s.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wp = wps.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wt = wts.template get_multi_ptr<sycl::access::decorated::no>().get();
                sycl::float4* tb = tbs.template get_multi_ptr<sycl::access::decorated::no>().get();
                fill_tables(e4, wp, wt, lid, SGS * 16);
                sycl::group_barrier(it.get_group());
                fill_btab(tb, wp, lid, SGS * 16);
                sycl::group_barrier(it.get_group());
                auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_linear_id());
                const int s = int(it.get_group(0)) / wpb;
                const int j0 = (int(it.get_group(0)) % wpb) * R * SGS + int(sg.get_group_linear_id()) * R;
                const int e = sel[s];
                if (own >= 0 ? (s & 1) != own : (e < e0 || e >= e0 + en)) {
                    if (lane == 0)
                        for (int r = 0; r < R; ++r) dbuf[size_t(s) * h + j0 + r] = 0.f;
                    return;
                }
                const int el = own >= 0 ? e : e - e0;
                const float outer = NV ? amax[e] / den[0] : 1.f;
                const float* a = act + size_t(s) * I;
                float d[R];
#pragma unroll
                for (int r = 0; r < R; ++r) d[r] = 0.f;
#pragma unroll
                for (int k = 0; k < steps; ++k) {
                    const int c = k * 16 + lane;
                    float xr[8];
                    load_x(a + size_t(c) * 8, xr, 8);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const size_t rw = size_t(el) * h + j0 + r;
                        if (NV) {
                            const uint32_t pw = *reinterpret_cast<const uint32_t*>(W + rw * rb + size_t(c) * 4);
                            float A = 0.f, B = 0.f;
                            nvb8(pw, xr, tb, A, B);
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
    std::vector<int> used;
    for (int e = e0_; e < e0_ + en_; ++e) if (count[e]) used.push_back(e);
    int* ids = reinterpret_cast<int*>(elist_);   // the sel_ download above waited: no kernel still reads the list
    for (size_t u = 0; u < used.size(); ++u) {
        const int e = used[u];
        ids[u] = e;
        elist_[128 + u] = Ly.fgu == PackedFormat::NVFP4 ? Ly.hgua[e] / Ly.hgud[0] : 1.f;
        elist_[256 + u] = Ly.fdn == PackedFormat::NVFP4 ? Ly.hdna[e] / Ly.hdnd[0] : 1.f;
    }
    const int nu = int(used.size());
    if (nu) {
        uint16_t* wb = reinterpret_cast<uint16_t*>(xs_.weight_pairs);
        float* proj = xs_.projection;
        uint16_t* act = reinterpret_cast<uint16_t*>(xs_.activation);
        static const bool grouped = !std::getenv("INK_EXP_ONEDNN");
        int T = 0;
        int* ui = ids + 384;
        int* tl = ids + 768;
        for (int u = 0; u < nu; ++u) {
            const int e = used[u];
            ui[3 * u] = e - we0_; ui[3 * u + 1] = off[e]; ui[3 * u + 2] = count[e];
            for (int mb = 0; mb * 256 < count[e]; ++mb) { tl[2 * T] = u; tl[2 * T + 1] = mb; ++T; }
        }
        if (grouped) q_.memcpy(tbl_d_, elist_, (768 + 2 * size_t(T)) * sizeof(float));
        const int* tiles_d = reinterpret_cast<const int*>(tbl_d_) + 768;
        const int* ui_d = reinterpret_cast<const int*>(tbl_d_) + 384;
        const int fg = Ly.fgu == PackedFormat::NVFP4 ? 0 : 1, fd = Ly.fdn == PackedFormat::NVFP4 ? 0 : 1;
        static const bool nves = !std::getenv("INK_NO_NVES");   // GRIMOIRE's Ornith ESIMD prompt MoE, NVFP4 port
        static const bool fuse_act = !std::getenv("INK_NO_FUSEACT");   // softcapped SwiGLU in the gate_up epilogue
        const float gcap = c_.gate_cap, ucap = c_.up_cap;
        static const bool mxes = !std::getenv("INK_NO_MXES");   // the same ESIMD kernel for the MXFP8 layers
        bool act_done = false;
        const bool es_g = grouped && nves && (fg == 0 || mxes), es_d = grouped && nves && (fd == 0 || mxes);
        if (es_g && fuse_act) {
            ink_moe_nv_esimd(q_, InkNvEsArgs{Ly.gu, Ly.gus, 2 * I, h, wen_ * 2 * I, inb_, proj, tiles_d, T, ui_d, tbl_d_ + 128, nvlut_, act, gcap, ucap, fg});
            act_done = true;
        } else if (es_g)
            ink_moe_nv_esimd(q_, InkNvEsArgs{Ly.gu, Ly.gus, 2 * I, h, wen_ * 2 * I, inb_, proj, tiles_d, T, ui_d, tbl_d_ + 128, nvlut_, nullptr, 0.f, 0.f, fg});
        else if (grouped)
            ink_moe_grouped(q_, fg, Ly.gu, Ly.gus, rbg, sbg, 2 * I, h, inb_, proj, tiles_d, T, ui_d, tbl_d_ + 128, {});
        else
            for (int u = 0; u < nu; ++u) {
                const int e = used[u];
                dequant_used(Ly.fgu, 2 * I, h, Ly.gu, Ly.gus, rbg, sbg, 1, 128, wb, u);
                b70::QuantWeight wg;
                wg.fmt = b70::Fmt::BF16; wg.N = 2 * I; wg.K = h; wg.row_bytes = int64_t(h) * 2; wg.payload = reinterpret_cast<const uint8_t*>(wb);
                b70::launch_gemm_fast(q_, wg, reinterpret_cast<const bf16x*>(inb_) + size_t(off[e]) * h, proj + size_t(off[e]) * 2 * I, (count[e] + 31) / 32 * 32);
            }
        if (!act_done) q_.parallel_for<InkActP>(sycl::range<1>(size_t(M) * TK * I), [=](sycl::id<1> id) {
            const size_t i = id[0];
            // same formula as inkling2::launch_expert_ffn: BF16-rounded linear outputs, softcaps, sigmoid of the raw gate
            const float g = bfr(proj[2 * i]), u = bfr(proj[2 * i + 1]);
            const float z = sycl::exp(-sycl::fabs(g));
            const float sig = (g >= 0.f ? 1.f : z) / (1.f + z);
            act[i] = tobf((gcap * sycl::tanh(g / gcap)) * sig * (ucap * sycl::tanh(u / ucap)));
        });
        if (es_d)
            ink_moe_nv_esimd(q_, InkNvEsArgs{Ly.dn, Ly.dns, h, I, wen_ * h, act, proj, tiles_d, T, ui_d, tbl_d_ + 256, nvlut_, nullptr, 0.f, 0.f, fd});
        else if (grouped)
            ink_moe_grouped(q_, fd, Ly.dn, Ly.dns, rbd, sbd, h, I, act, proj, tiles_d, T, ui_d, tbl_d_ + 256, {});
        else
            for (int u = 0; u < nu; ++u) {
                const int e = used[u];
                dequant_used(Ly.fdn, h, I, Ly.dn, Ly.dns, rbd, sbd, 1, 256, wb, u);
                b70::QuantWeight wd;
                wd.fmt = b70::Fmt::BF16; wd.N = h; wd.K = I; wd.row_bytes = int64_t(I) * 2; wd.payload = reinterpret_cast<const uint8_t*>(wb);
                b70::launch_gemm_fast(q_, wd, reinterpret_cast<const bf16x*>(act) + size_t(off[e]) * I, proj + size_t(off[e]) * h, (count[e] + 31) / 32 * 32);
            }
    }
    {
        const float* outp = xs_.projection;   // fp32 expert outputs; bf-rounded here (was a separate bf16 pass)
        const int* pp = pairpos_;
        const int* sl = sel_;
        const float* w = wts_;
        float* yb = row(YB);
        const size_t total = size_t(M) * h;
        const int e0 = e0_, en = en_;
        const bool tp = world_ > 1;
        q_.parallel_for<InkCombine>(sycl::range<1>(total), [=](sycl::id<1> id) {
            const size_t t = id[0], m = t / h, ch = t % h;
            float acc = 0.f;
            for (int s = 0; s < TK; ++s) {
                const int e = sl[m * TK + s];
                if (e >= e0 && e < e0 + en) acc += w[m * TK + s] * bfr(outp[size_t(pp[m * TK + s]) * h + ch]);
            }
            yb[t] = tp ? acc : bfr(acc);
        });
    }
}

// Every used expert's projection decoded in ONE launch: exact Inkling weights (bfr((e2m1*scale)*outer), as
// expert_bf16_pair), row-major BF16, coalesced reads; expert u goes to dst + u*N*K.  Then oneDNN's BF16 GEMM per
// expert (the dense prompt path).  GEMM rows are padded to a multiple of 32 so oneDNN builds few primitives; pad
// rows read the next expert's inputs and their outputs are overwritten by the next expert's GEMM (in order).
void Engine::dequant_used(PackedFormat f, int N, int K, const uint8_t* pay, const uint8_t* scl, size_t rb, size_t sb,
                          int nu, int outer_at, uint16_t* dst, int u0) {
    const int* ids = reinterpret_cast<const int*>(elist_) + u0;
    const float* outer = elist_ + outer_at + u0;
    const int e0 = we0_;
    if (f == PackedFormat::NVFP4) {
        const int KB = K / 16;
        q_.parallel_for<InkDqNv>(sycl::range<1>(size_t(nu) * N * KB), [=](sycl::id<1> id) {
            const size_t t = id[0], u = t / (size_t(N) * KB), r = t % (size_t(N) * KB), n = r / KB, kb = r % KB;
            const size_t el = size_t(ids[u] - e0);
            const float o = outer[u];
            const uint8_t* pw = pay + (el * N + n) * rb;
            const float sc = b70::e4m3_to_f32(scl[(el * N + n) * sb + kb]);
            const sycl::uint2 v = *reinterpret_cast<const sycl::uint2*>(pw + kb * 8);
            sycl::vec<uint16_t, 16> ov;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const uint32_t byte = ((b < 4 ? v[0] : v[1]) >> (8 * (b & 3))) & 0xFFu;
                ov[2 * b] = tobf((b70::e2m1_to_f32(uint8_t(byte & 15)) * sc) * o);
                ov[2 * b + 1] = tobf((b70::e2m1_to_f32(uint8_t(byte >> 4)) * sc) * o);
            }
            *reinterpret_cast<sycl::vec<uint16_t, 16>*>(dst + (u * N + n) * size_t(K) + kb * 16) = ov;
        });
    } else {
        const int KB = K / 32;
        q_.parallel_for<InkDqMx>(sycl::range<1>(size_t(nu) * N * KB * 2), [=](sycl::id<1> id) {
            const size_t t = id[0] / 2, half = id[0] % 2, u = t / (size_t(N) * KB), r = t % (size_t(N) * KB), n = r / KB, kb = r % KB;
            const size_t el = size_t(ids[u] - e0);
            const float sc = b70::e8m0_to_f32(scl[(el * N + n) * sb + kb]);
            const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(pay + (el * N + n) * rb + kb * 32 + half * 16);
            sycl::vec<uint16_t, 16> ov;
#pragma unroll
            for (int b = 0; b < 16; ++b) ov[b] = tobf(b70::e4m3_to_f32(uint8_t((v[b / 4] >> (8 * (b & 3))) & 0xFFu)) * sc);
            *reinterpret_cast<sycl::vec<uint16_t, 16>*>(dst + (u * N + n) * size_t(K) + kb * 32 + half * 16) = ov;
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

void Engine::hist_update_v(int slot, int kind, const float* src, size_t ld, int n, int M) {
    float* vb = vbuf_ + (size_t(slot) * 4 + kind) * 8 * size_t(h_);
    const int h = h_;
    q_.parallel_for<InkHistV>(sycl::range<1>(size_t(M) * n), [=](sycl::id<1> id) {
        const int m = int(id[0] / n), ch = int(id[0] % n);
        vb[size_t(m) * h + ch] = bfr(src[size_t(m) * ld + ch]);
    });
}

void Engine::allreduce(float* buf, size_t n) {
    // Host-mediated: copy out, wait for our own copy, meet the other card's thread, copy its part in.
    // No event of one device is ever waited on by the other device's queue.
    TPX& x = *tpx_;
    const int me = rank_, peer = 1 - rank_;
    if (x.box[0] && n <= TPX::kBoxWords) {
        // device-side exchange: no host wait, so a whole decode step can be one graph
        const int par = x.mcount[me]++ & 1;
        uint32_t* to = x.box[peer] + size_t(par) * TPX::kBoxWords;
        uint32_t* own = x.box[me] + size_t(par) * TPX::kBoxWords;
        uint32_t* err = x.err + me;
        q_.parallel_for<InkMail>(sycl::range<1>(n), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const float v = buf[i];
            uint32_t w = sycl::bit_cast<uint32_t>(v);
            if (w == kEmpty) w = 0x7fc00000u;
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>(to[i]).store(w);
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space> in(own[i]);
            uint32_t got = in.load();
            for (int it = 0; got == kEmpty && it < (1 << 20); ++it) got = in.load();
            if (got == kEmpty) {
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>(*err).store(1u);
                got = 0x7fc00000u;
            }
            in.store(kEmpty);
            buf[i] = bfr(v + sycl::bit_cast<float>(got));
        });
        return;
    }
    if (seg_rec_ && n <= size_t(64) * h_) {
        // recording: the copy-out closes this segment; the add opens the next one (no host work now)
        const int k = x.counter[me]++ % TPX::NS;
        static const bool direct = !std::getenv("INK_SEG_HOSTCOPY");   // D2H straight into the peer's IN slot (MEASURED 26.2 -> 25.7 ms/step)
        float* mine = direct ? x.ib[peer][k] : x.hb[me][k];
        q_.memcpy(mine, buf, n * sizeof(float));
        seg_cur_->end_recording(q_);
        seg_dst_->g.push_back(std::make_unique<sycl_ext::command_graph<sycl_ext::graph_state::executable>>(seg_cur_->finalize()));
        seg_dst_->slot.push_back(k);
        seg_dst_->n.push_back(n);
        seg_cur_ = std::make_unique<sycl_ext::command_graph<sycl_ext::graph_state::modifiable>>(q_.get_context(), q_.get_device());
        seg_cur_->begin_recording(q_);
        const float* in = x.ib[me][k];
        if (fold_add_) { peer_in_ = in; return; }
        q_.parallel_for<InkAddRoundS>(sycl::range<1>(n), [=](sycl::id<1> i) { buf[i[0]] = bfr(buf[i[0]] + in[i[0]]); });
        return;
    }
    if (seg_rec_) throw std::runtime_error("segment recording: large exchange");
    const int k = x.counter[me]++ % TPX::NS;
    float* mine = x.hb[me][k];
    const float* theirs = x.hb[peer][k];
    g_tp_stage[me] = 1;  // copying out
    const auto t0 = std::chrono::steady_clock::now();
    q_.memcpy(mine, buf, n * sizeof(float)).wait();
    const auto t1 = std::chrono::steady_clock::now();
    g_tp_wait_ms[me][0] += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ++g_tp_n[me];
    if (n <= size_t(64) * h_) {
        // small (decode / verify): the host thread writes our part straight into the peer's pinned IN slot and the
        // add kernel reads its own IN slot over PCIe -- no copy-in command.  Slot k is reused NS exchanges later,
        // after this card's add kernel ran (its later copy-out waits behind it on the in-order queue).
        std::memcpy(x.ib[peer][k], mine, n * sizeof(float));
        g_tp_stage[me] = 2;
        x.bar.arrive_and_wait();
        g_tp_wait_ms[me][1] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        g_tp_stage[me] = 3;
        g_tp_progress = now_ms();
        const float* in = x.ib[me][k];
        if (fold_add_) { peer_in_ = in; return; }
        q_.parallel_for<InkAddRoundH>(sycl::range<1>(n), [=](sycl::id<1> i) { buf[i[0]] = bfr(buf[i[0]] + in[i[0]]); });
        return;
    }
    g_tp_stage[me] = 2;  // waiting for the other card's thread
    x.bar.arrive_and_wait();
    g_tp_stage[me] = 3;
    g_tp_progress = now_ms();
    float* pr = peer_;
    q_.memcpy(pr, theirs, n * sizeof(float));
    q_.parallel_for<InkAddRound>(sycl::range<1>(n), [=](sycl::id<1> i) { buf[i[0]] = bfr(buf[i[0]] + pr[i[0]]); });
    // the peer must not overwrite our read slot before the copy-in ran: slots rotate over NS exchanges and every
    // exchange waits on its own copy-out, which runs after this card's earlier copy-in (in-order queue)
}

// Both cards hold the best (value, index) of their vocabulary half; the host picks (lowest index on ties).
// n rows' tokens: one card = next_; two cards = per row the better (value, lowest index on ties) of both halves
void Engine::final_tokens(int n, int* out) {
    if (world_ == 1) { q_.memcpy(out, next_, size_t(n) * sizeof(int)).wait(); return; }
    float v[8]; int i[8];
    q_.memcpy(v, bestv_, size_t(n) * sizeof(float));
    q_.memcpy(i, next_, size_t(n) * sizeof(int)).wait();
    g_tp_stage[rank_] = 6;
    g_tp_progress = now_ms();
    TPX& x = *tpx_;
    for (int m = 0; m < n; ++m) { x.bv[rank_][m] = v[m]; x.bi[rank_][m] = i[m]; }
    x.bar.arrive_and_wait();
    for (int m = 0; m < n; ++m) {
        const float v0 = x.bv[0][m], v1 = x.bv[1][m];
        const int i0 = x.bi[0][m], i1 = x.bi[1][m];
        out[m] = (v1 > v0 || (v1 == v0 && i1 < i0)) ? i1 : i0;
    }
    x.bar.arrive_and_wait();
}

int Engine::final_token(int m) {
    float v; int i;
    if (world_ > 1) g_tp_stage[rank_] = 4;  // reading the token
    q_.memcpy(&v, bestv_ + m, sizeof(float));
    q_.memcpy(&i, next_ + m, sizeof(int)).wait();
    if (world_ == 1) return i;
    g_tp_stage[rank_] = 5;
    g_tp_progress = now_ms();
    TPX& x = *tpx_;
    x.bv[rank_][m] = v; x.bi[rank_][m] = i;
    x.bar.arrive_and_wait();
    const float v0 = x.bv[0][m], v1 = x.bv[1][m];
    const int i0 = x.bi[0][m], i1 = x.bi[1][m];
    x.bar.arrive_and_wait();
    return (v1 > v0 || (v1 == v0 && i1 < i0)) ? i1 : i0;
}

void Engine::run_seg(int M) {
    if (world_ == 1 || !seg_on_ || M > 8 || (tpx_ && tpx_->box[0])) { run(M); return; }
    TPX& x = *tpx_;
    const int me = rank_, peer = 1 - rank_;
    SegStep& S = seg_[M];
    auto align = [&] { x.counter[me] = (x.counter[me] + TPX::NS - 1) / TPX::NS * TPX::NS; };
    if (S.g.empty()) {
        align();
        const int c0 = x.counter[me];
        q_.wait();
        seg_dst_ = &S;
        seg_cur_ = std::make_unique<sycl_ext::command_graph<sycl_ext::graph_state::modifiable>>(q_.get_context(), q_.get_device());
        seg_cur_->begin_recording(q_);
        seg_rec_ = true;
        try { run(M); } catch (...) { seg_rec_ = false; seg_cur_->end_recording(q_); throw; }
        seg_rec_ = false;
        seg_cur_->end_recording(q_);
        S.g.push_back(std::make_unique<sycl_ext::command_graph<sycl_ext::graph_state::executable>>(seg_cur_->finalize()));
        seg_cur_.reset();
        x.counter[me] = c0;   // recording exchanged nothing
        if (rank_ == 0) std::printf("two-card step M=%d recorded: %zu segments\n", M, S.g.size());
    }
    align();
    const int c0 = x.counter[me];
    const size_t ns = S.g.size();
    for (size_t j = 0; j < ns; ++j) {
        sycl::event ev = q_.ext_oneapi_graph(*S.g[j]);
        if (j + 1 == ns) break;
        const int k = S.slot[j];
        if ((c0 + int(j)) % TPX::NS != k) throw std::runtime_error("segment replay: slot order");
        g_tp_stage[me] = 1;
        const auto t0 = std::chrono::steady_clock::now();
        static const bool spin = std::getenv("INK_SEG_SPIN") != nullptr;   // poll the segment's event (MEASURED: no gain)
        if (spin) {
            while (ev.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {}
        } else ev.wait();
        const auto t1 = std::chrono::steady_clock::now();
        g_tp_wait_ms[me][0] += std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++g_tp_n[me];
        static const bool direct = !std::getenv("INK_SEG_HOSTCOPY");
        if (!direct) std::memcpy(x.ib[peer][k], x.hb[me][k], S.n[j] * sizeof(float));
        g_tp_stage[me] = 2;
        x.bar.arrive_and_wait();
        g_tp_wait_ms[me][1] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        g_tp_stage[me] = 3;
        g_tp_progress = now_ms();
    }
    x.counter[me] = c0 + int(ns) - 1;
}

int Engine::step(int token, std::vector<float>* logits_out) {
    want_logits_ = logits_out != nullptr;
    if (pos_ >= limit_) throw std::runtime_error("context full");
    q_.memcpy(tok_, &token, sizeof(int));
    const bool taps_saved = taps_on;
    taps_on = false;
    struct Restore { bool& f; bool v; ~Restore() { f = v; } } restore{taps_on, taps_saved};
    if (use_graph && !gexec_ && (world_ == 1 || (tpx_ && tpx_->box[0]))) {
        if (world_ > 1) gpar_ = tpx_->mcount[rank_];
        sycl_ext::command_graph<sycl_ext::graph_state::modifiable> g(q_.get_context(), q_.get_device());
        g.begin_recording(q_);
        run(1);
        g.end_recording(q_);
        gexec_ = std::make_unique<sycl_ext::command_graph<sycl_ext::graph_state::executable>>(g.finalize());
        if (world_ > 1) { gex_ = tpx_->mcount[rank_] - gpar_; tpx_->mcount[rank_] = gpar_; }   // recording ran nothing
    }
    // two cards: the graph's mailbox parities are baked from capture time; replay only on the same parity
    const bool replay = gexec_ && (world_ == 1 || ((tpx_->mcount[rank_] - gpar_) & 1) == 0);
    if (replay) {
        q_.ext_oneapi_graph(*gexec_);
        if (world_ > 1) tpx_->mcount[rank_] += gex_;
    } else if (world_ > 1 && !logits_out) run_seg(1);
    else run(1);
    if (logits_out) {
        logits_out->resize(c_.output_vocab);
        q_.memcpy(logits_out->data(), logits_, size_t(c_.output_vocab) * sizeof(float)).wait();
        if (last_raw_) for (float& v : *logits_out) v = c_.logit_cap * std::tanh(b70::bf16_to_f32(b70::f32_to_bf16(v)) / c_.logit_cap);
    }
    const int next = final_token();
    ++pos_;
    return next;
}

int Engine::prefill(const std::vector<int>& ids, std::vector<float>* logits_out) {
    want_logits_ = logits_out != nullptr;
    if (pos_ + int(ids.size()) > limit_) throw std::runtime_error("context full");
    size_t at = 0;
    while (at < ids.size()) {
        const int M = int(std::min<size_t>(size_t(mmax_), ids.size() - at));
        q_.memcpy(tok_, ids.data() + at, size_t(M) * sizeof(int));
        if (M == 1 && gexec_ && !mtp_) q_.ext_oneapi_graph(*gexec_);
        else run(M);
        if (mtp_) draft_commit(pos_, 0, M);
        pos_ += M;
        at += M;
    }
    if (logits_out) { logits_out->resize(c_.output_vocab); q_.memcpy(logits_out->data(), logits_, size_t(c_.output_vocab) * sizeof(float)); }
    return final_token();
}

// M rows at positions *pos_d .. *pos_d + M - 1; advances *pos_d by M.  M == 1 has no host value that
// depends on the position, so it is recorded once and replayed.
void Engine::run(int M) {
    const int h = h_, MM = mmax_;
    const int* posd = pos_d_;
    float* pool = pool_;
    const size_t RS = rs_;
    const float eps = c_.eps;
    if (pp_first()) {   // hidden = rms(embed[token], embed_norm)
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
    if (!pp_first()) { bank = pp_in_.bank; pre_bank = pp_in_.pre_bank; prev_bank = pp_in_.prev_bank; next_bank = pp_in_.next_bank; }
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
    int vi = -1;
    // INK_SECT=1: host-wall time per section of the prompt path (waits after each section; slows the run)
    static const bool sect_on = std::getenv("INK_SECT") != nullptr;
    auto sect_t = std::chrono::steady_clock::now();
    auto mark = [&](int k) {
        if (!sect_on || M == 1) return;
        q_.wait();
        const auto now = std::chrono::steady_clock::now();
        sect_ms_[k] += std::chrono::duration<double, std::milli>(now - sect_t).count();
        sect_t = now;
    };
    mark(7);
    for (const auto& v : visits_) {
        ++vi;
        if (vi < vbeg_ || (vend_ >= 0 && vi >= vend_)) continue;
        static const int tapmode = std::getenv("INK_TAPMODE") ? std::atoi(std::getenv("INK_TAPMODE")) : 0;
        if (taps_on && tapmode != 2) {
            const int o = tapmode == 1 ? 1 : 0;
            if (vi == 5 + o || vi == 23 + o || vi == 41 + o) copy(HID, vi == 5 + o ? TAP0 : vi == 23 + o ? TAP1 : TAP2);
        }
        const int i = v.physical;
        const Layer& Ly = L_[i];
        const Slot& sl = S_[v.cache_slot];
        const auto& a = v.sliding ? c_.local : c_.full;
        const int H = a.heads / world_, KVH = a.kv_heads / world_, D = a.head_dim, K = KVH * D, R = c_.rel_dim, Q = H * D;
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
        {
            static const int tapmode = std::getenv("INK_TAPMODE") ? std::atoi(std::getenv("INK_TAPMODE")) : 0;
            if (taps_on && tapmode == 2 && (vi == 5 || vi == 23 || vi == 41)) copy(in[0], vi == 5 ? TAP0 : vi == 23 ? TAP1 : TAP2);
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
            const bool gemm = M >= 32 || (gsm_ && M <= 8);
            const bool pad4 = (i4_ && M <= 8 && Ly.Wp4.pay) || (i8v_ && M >= 2 && M <= 8 && Ly.WpR);
            if (v.sliding && !v.has_message) {
                nb.j[0] = pad4 ? NormJob{in[0], Ly.nq, N0, xpad_, kp_} : NormJob{in[0], Ly.nq, N0, gemm ? nb_ : nullptr}; nb.count = 1;
                nrm[0] = nrm[1] = nrm[2] = N0;
                same = true;
            } else {
                const uint16_t* w[3] = {Ly.nq, Ly.nk, Ly.nv};
                for (int j = 0; j < 3; ++j)
                    nb.j[j] = pad4 ? NormJob{in[j], w[j], N0 + j, xpad_ + size_t(j) * 8 * kp_, kp_} : NormJob{in[j], w[j], N0 + j, gemm ? nb_ + size_t(j) * MM * h : nullptr};
                nb.count = 3;
                nrm[0] = N0; nrm[1] = N1; nrm[2] = N2;
            }
            rms(nb, M);
        }
        const int NA = Ly.NA, NB = Ly.NB, NC = Ly.NC;
        static const int i8m_ks = std::getenv("INK_I8M") ? std::atoi(std::getenv("INK_I8M")) : 0;   // MEASURED 11-24: 8-row FMA GEMV slower than b70::launch_fp8_smallm (33.5 -> 38.6 ms/step): opt-in
        if (i8m_ks > 0 && Ly.WpQ && M >= 2 && M <= 8 &&
            ink_es_gemv_i8m(q_, InkI8MArgs{Ly.WpQ, Ly.WpS, NA + NB + NC, h, M, row(nrm[0]), row(nrm[1]), row(nrm[2]), size_t(h), projA_, projB_, projC_,
                                           size_t(NA), size_t(NB), size_t(NC), NA, NA + NB, 0, 0.f}, i8m_ks)) {
        } else if (i8v_ && M >= 2 && M <= 8 && Ly.WpR && same && projAll_ && [&] {
                // one input for all three segments (sliding visits): one launch over NA + NB + NC rows, then the three
                // segments are views of the [M][NA + NB + NC] result
                return i8_small(Ly.WpR, Ly.WpS, 0, NA + NB + NC, kp_, xpad_, projAll_, M);
            }()) {
            split_proj(NA, NB, NC, M);
        } else if (i8v_ && M >= 2 && M <= 8 && Ly.WpR && [&] {
                const uint16_t* xs[3] = {xpad_, same ? xpad_ : xpad_ + size_t(8) * kp_, same ? xpad_ : xpad_ + 2 * size_t(8) * kp_};
                return i8_small(Ly.WpR, Ly.WpS, 0, NA, kp_, xs[0], projA_, M) && i8_small(Ly.WpR, Ly.WpS, NA, NB, kp_, xs[1], projB_, M) &&
                       i8_small(Ly.WpR, Ly.WpS, NA + NB, NC, kp_, xs[2], projC_, M);
            }()) {
        } else if (i4_ && M <= 8 && Ly.Wp4.pay && [&] {
                const uint16_t* xs[3] = {xpad_, same ? xpad_ : xpad_ + size_t(8) * kp_, same ? xpad_ : xpad_ + 2 * size_t(8) * kp_};
                return i4_gemv(Ly.Wp4, 0, NA, xs[0], projA_, M) && i4_gemv(Ly.Wp4, NA, NB, xs[1], projB_, M) && i4_gemv(Ly.Wp4, NA + NB, NC, xs[2], projC_, M);
            }()) {
        } else if (gsm_ && M <= 8 && [&] {
                const uint16_t* xs[3] = {nb_, same ? nb_ : nb_ + size_t(MM) * h, same ? nb_ : nb_ + 2 * size_t(MM) * h};
                return gsm_dense(Ly.Wp, NA, h, xs[0], projA_, M) && gsm_dense(Ly.Wp + size_t(NA) * h, NB, h, xs[1], projB_, M) &&
                       gsm_dense(Ly.Wp + size_t(NA + NB) * h, NC, h, xs[2], projC_, M);
            }()) {
        } else if (M == 1) {
            static const int ksp = std::getenv("INK_KS_WP") ? std::atoi(std::getenv("INK_KS_WP")) : 2;   // MEASURED 11-24: Wp+Wo 235 -> 204 ms (KS 4 -> 2)
            static const int esk = std::getenv("INK_ESI8") ? std::atoi(std::getenv("INK_ESI8")) : 8;   // MEASURED 11-24: KS 8 186.7 ms vs SIMT gemvt8 204   // ESIMD int8 GEMV, value = K split
            if (Ly.WpQ && esk > 0 && ink_es_gemv_i8(q_, InkI8Args{Ly.WpQ, Ly.WpS, NA + NB + NC, h, row(nrm[0]), row(nrm[1]), row(nrm[2]), projA_, projB_, projC_, NA, NA + NB, 0, 0.f}, esk)) {
            } else if (Ly.WpQ) gemvt8<0>(Ly.WpQ, Ly.WpS, NA + NB + NC, h, ksp, row(nrm[0]), projA_, NA, row(nrm[1]), projB_, NA + NB, row(nrm[2]), projC_);
            else gemvt<0>(Ly.WpT, NA + NB + NC, h, 2, row(nrm[0]), projA_, NA, row(nrm[1]), projB_, NA + NB, row(nrm[2]), projC_);
        } else {
            const uint16_t* xb[3] = {nb_, same ? nb_ : nb_ + size_t(MM) * h, same ? nb_ : nb_ + 2 * size_t(MM) * h};
            if (M <= 8)
                gemvtm<0>(Ly.WpT, NA + NB + NC, h, 2, M, row(nrm[0]), row(nrm[1]), row(nrm[2]), h, projA_, NA, NA, projB_, NB, NA + NB, projC_, NC,
                          NA + NB + NC);
            else {
                dense(Ly.Wp, NA, h, row(nrm[0]), h, xb[0], projA_, NA, M, true);
                dense(Ly.Wp + size_t(NA) * h, NB, h, row(nrm[1]), h, xb[1], projB_, NB, M, true);
                dense(Ly.Wp + size_t(NA + NB) * h, NC, h, row(nrm[2]), h, xb[2], projC_, NC, M, true);
            }
        }
        mark(0);
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
            const int taps = c_.conv, G = H + 2 * KVH, koff = Ly.koff, kfull = Ly.kfull;
            // the rel-position bias table of the matrix-unit attention (rows >= 2), built here per (row, q head) from
            // the clipped rel values (was the InkBias launch): bias[m][j][e] = bf(sum_r rel[m][j][r] * table[r][e])
            static const bool post_bias = !std::getenv("INK_SEP_BIAS");
            const bool pb = post_bias && M >= 2 && D == 128 && !std::getenv("INK_ATT_SIMT") && !std::getenv("INK_NO_ATTES") && Ly.E % 1024 == 0;
            if (pb && !bias_) bias_ = dev_.alloc<uint16_t>(size_t(mmax_) * std::max(c_.local.heads, c_.full.heads) * std::max(c_.window, c_.rel_extent));
            uint16_t* bsp = pb ? bias_ : nullptr;
            post_bias_done_ = pb;
            const uint16_t* tbl = Ly.table;
            const int Eb = Ly.E;
            static const bool fold_hv = !std::getenv("INK_SEP_HISTV");   // verify ring inputs stored by the kernels that read them
            const bool hvf = fold_hv && M > 1 && verify_ && !std::getenv("INK_HV_NOK");
            float* vbK = hvf ? vbuf_ + (size_t(v.cache_slot) * 4 + 0) * 8 * size_t(h) : nullptr;
            float* vbV = hvf ? vbuf_ + (size_t(v.cache_slot) * 4 + 1) * 8 * size_t(h) : nullptr;
            q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> rls(sycl::range<1>(16), cgh);
            cgh.parallel_for<InkPost>(sycl::nd_range<1>(size_t(M) * G * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
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
                        const float rv = bfr(bfr(r * f2) * tau);
                        A[orel + j * R + d] = rv;
                        if (bsp) rls[d] = rv;
                    }
                    if (bsp) {   // uniform per work-group
                        sycl::group_barrier(g);
                        const int e0 = d * 8;
                        for (int eb = e0; eb < Eb; eb += 128 * 8) {
                            float b[8];
#pragma unroll
                            for (int e = 0; e < 8; ++e) b[e] = 0.f;
#pragma unroll
                            for (int r = 0; r < 16; ++r) {   // the reference's order: r = 0..15
                                const float rl = rls[r];
                                const sycl::vec<uint16_t, 8> tv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(tbl + size_t(r) * Eb + eb);
#pragma unroll
                                for (int e = 0; e < 8; ++e) b[e] += rl * bfw(tv[e]);
                            }
                            sycl::vec<uint16_t, 8> o;
#pragma unroll
                            for (int e = 0; e < 8; ++e) o[e] = tobf(b[e]);
                            *reinterpret_cast<sycl::vec<uint16_t, 8>*>(bsp + (size_t(m) * H + j) * Eb + eb) = o;
                        }
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
                if (vbK) (isk ? vbK : vbV)[size_t(m) * h + ch] = x;
                const float* dyn = P + size_t(m) * ld + K;
                float sum = 0.f;
#pragma unroll
                for (int lag = 0; lag < 4; ++lag) {   // taps == 4 (geometry check); loads not serialized by a variable bound
                    const bool on = lag <= pos;
                    const float xl = lag == 0 ? x : (m - lag >= 0 ? bfr(P[size_t(m - lag) * ld + ch]) : hist[size_t(((pos - lag) % 3 + 3) % 3) * K + ch]);
                    const float wt = sliding ? bfw(w[ch * 4 + 3 - lag]) : bfw(w[lag * kfull + koff + ch]) + bfr(dyn[kh_i * 4 + lag]);
                    if (on) sum += xl * wt;
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
            });
            if (M > 1 && verify_) { if (!hvf) { hist_update_v(v.cache_slot, 0, projB_, NB, K, M); hist_update_v(v.cache_slot, 1, projC_, NC, K, M); } }
            else if (M > 1) { hist_update(sl.khist, projB_, NB, K, M); hist_update(sl.vhist, projC_, NC, K, M); }
        }
        mark(1);
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
                    sycl::local_accessor<float, 1> wl(sycl::range<1>(128), cgh), pv(sycl::range<1>(8 * 128), cgh);
                    cgh.parallel_for<InkAtt2>(sycl::nd_range<1>(size_t(H) * NCH * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        const int gi = int(it.get_group(0)), j = gi / NCH, c = gi % NCH, d = int(it.get_local_id(0));
                        const int pos = *posd, count = sycl::min(pos + 1, win), first = pos + 1 - count;
                        if (c * 128 >= count) return;
                        auto sg = it.get_sub_group();
                        const int sgid = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
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
                        float acc[8];
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
                        for (int kk = 0; kk < 16; ++kk) {
                            const int k = sgid * 16 + kk;
                            if (k < nk) {
                                const float w = wl[k];
                                const sycl::vec<uint16_t, 8> vv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(vc + size_t((first + c * 128 + k) % cap) * K + hd + lane * 8);
#pragma unroll
                                for (int e = 0; e < 8; ++e) acc[e] = sycl::fma(w, bfw(vv[e]), acc[e]);
                            }
                        }
                        sycl::group_barrier(it.get_group());
#pragma unroll
                        for (int e = 0; e < 8; ++e) pv[sgid * 128 + lane * 8 + e] = acc[e];
                        sycl::group_barrier(it.get_group());
                        float o = 0.f;
                        for (int t = 0; t < 8; ++t) o += pv[t * 128 + d];
                        part[(size_t(j) * NCH + c) * 128 + d] = o;
                    });
                });
                q_.parallel_for<InkAtt3>(sycl::nd_range<1>(size_t(H) * 128, 128), [=](sycl::nd_item<1> it) {
                    const int j = int(it.get_group(0)), d = int(it.get_local_id(0));
                    const int pos = *posd, count = sycl::min(pos + 1, win), nch = (count + 127) / 128;
                    float acc = 0.f;
                    for (int x = 0; x < nch; ++x) acc += part[(size_t(j) * NCH + x) * 128 + d];
                    const float ov = bfr(bfr(acc) * sigm(bfr(pA[og + j * D + d])));
                    aout[j * D + d] = ov;
                    aoutb[j * D + d] = tobf(ov);
                });
            } else if ((M >= 16 || (M >= 2 && D == 128 && (cap % 16 == 0 || !v.sliding) && !std::getenv("INK_OLD_ATTB") && !std::getenv("INK_NO_ATTES"))) &&
                       !std::getenv("INK_ATT_SIMT")) {
                // prompt rows and verify rows (2..15) on the matrix unit (tools/inkling2_moe_gemm.cpp); verify rows used
                // InkAttB (a sub-group per row and head walking every key: 175 us a visit at ~300 keys on two cards)
                if (!bias_) bias_ = dev_.alloc<uint16_t>(size_t(mmax_) * std::max(c_.local.heads, c_.full.heads) * std::max(c_.window, c_.rel_extent));
                uint16_t* bs = bias_;
                const int Hh = H;
                // bias[m][j][d] = bf(sum_r rel[m][j][r] * table[r][d]), the reference's summation order
                if (!post_bias_done_) q_.parallel_for<InkBias>(sycl::range<1>(size_t(M) * Hh * (E / 16)), [=](sycl::id<1> id) {
                    const size_t t = id[0], d0 = (t % (E / 16)) * 16, mj = t / (E / 16);
                    const int m = int(mj / Hh), j = int(mj % Hh);
                    float b[16];
#pragma unroll
                    for (int e = 0; e < 16; ++e) b[e] = 0.f;
#pragma unroll
                    for (int r = 0; r < 16; ++r) {   // same order per output: r = 0..15
                        const float rl = pA[size_t(m) * NA + orel + j * 16 + r];
                        const sycl::vec<uint16_t, 16> tv = *reinterpret_cast<const sycl::vec<uint16_t, 16>*>(table + size_t(r) * E + d0);
#pragma unroll
                        for (int e = 0; e < 16; ++e) b[e] += rl * bfw(tv[e]);
                    }
                    sycl::vec<uint16_t, 16> o;
#pragma unroll
                    for (int e = 0; e < 16; ++e) o[e] = tobf(b[e]);
                    *reinterpret_cast<sycl::vec<uint16_t, 16>*>(bs + mj * E + d0) = o;
                });
                static const bool att_es = !std::getenv("INK_NO_ATTES");   // ESIMD flash (GRIMOIRE flash_esimd design)
                static const bool att_cmp = std::getenv("INK_ATT_CMP") != nullptr;
                if (att_es && att_cmp && D == 128 && (cap % 16 == 0 || !v.sliding)) {   // debug: ESIMD vs the joint_matrix kernel on the same inputs
                    static int ncmp = 0;
                    float* o2 = sycl::malloc_device<float>(size_t(M) * Q, q_);
                    uint16_t* b2 = sycl::malloc_device<uint16_t>(size_t(M) * Q, q_);
                    ink_attn_prompt(q_, InkAttnArgs{pA, NA, oq, orel, osink, og, H, groups, E, table, kc, vc, cap, K, win, scale, o2, b2, Q, posd, M, bs});
                    ink_attn_esimd(q_, InkAttnArgs{pA, NA, oq, orel, osink, og, H, groups, E, table, kc, vc, cap, K, win, scale, aout, aoutb, Q, posd, M, bs});
                    q_.wait();
                    std::vector<float> h1(size_t(M) * Q), h2(size_t(M) * Q);
                    q_.memcpy(h1.data(), o2, h1.size() * 4).wait();
                    q_.memcpy(h2.data(), aout, h2.size() * 4).wait();
                    double md = 0, mx = 0; size_t nd = 0, worst = 0;
                    for (size_t k = 0; k < h1.size(); ++k) { const double d = std::fabs(double(h1[k]) - h2[k]); if (d > md) { md = d; worst = k; } if (d > 0) ++nd; mx = std::max(mx, double(std::fabs(h1[k]))); }
                    if (ncmp++ < 60) std::printf("attcmp visit %d sliding %d M %d: max|d| %.5f (max|ref| %.3f) at row %zu col %zu, %zu of %zu differ\n", vi, int(v.sliding), M, md, mx, worst / Q, worst % Q, nd, h1.size());
                    sycl::free(o2, q_); sycl::free(b2, q_);
                } else if (att_es && D == 128 && (cap % 16 == 0 || !v.sliding)) {
                    InkAttnArgs aa{pA, NA, oq, orel, osink, og, H, groups, E, table, kc, vc, cap, K, win, scale, aout, aoutb, Q, posd, M, bs};
                    static const bool att_split = !std::getenv("INK_NO_ATTSPLIT");
                    if (att_split && M <= 8) {   // verify rows: one thread per (head, 256-key slice), then a merge
                        // slices sized like GRIMOIRE's verify attention (~768 threads, >= 32 keys a slice): MEASURED 11-24
                        static const int chunk_env = std::getenv("INK_ATT_CHUNK") ? std::atoi(std::getenv("INK_ATT_CHUNK")) : 64;
                        const int chunk = chunk_env == 32 ? 32 : (chunk_env / 64) * 64, nsplit = (std::min(win, limit_) + 15 + chunk) / chunk;
                        if (!attpart_) {   // once, for the widest case (graph chains keep the pointer)
                            attpart_n_ = size_t((limit_ + 15 + 32) / 32) * std::max(c_.local.heads, c_.full.heads) * 8 * 130;
                            attpart_ = dev_.alloc<float>(attpart_n_);
                        }
                        if (size_t(nsplit) * H * 8 * 130 > attpart_n_) throw std::runtime_error("attention partials");
                        aa.nsplit = nsplit; aa.chunk = chunk; aa.part = attpart_;
                    }
                    ink_attn_esimd(q_, aa);
                }
                else
                    ink_attn_prompt(q_, InkAttnArgs{pA, NA, oq, orel, osink, og, H, groups, E, table, kc, vc, cap, K, win, scale, aout, aoutb, Q, posd, M, bs});
            } else if (M >= 16) {
                // prompt rows: a work-group = 16 consecutive query rows of one head (one sub-group each); every 16-key
                // tile of K / V goes through SLM once for all 16 rows.  Same per-score arithmetic as the M <= 8 path.
                const int RB = 16, nb = (M + RB - 1) / RB;
                q_.submit([&](sycl::handler& cgh) {
                    sycl::local_accessor<float, 1> qs(sycl::range<1>(RB * 144), cgh);
                    sycl::local_accessor<float, 1> kt(sycl::range<1>(16 * 132), cgh);
                    sycl::local_accessor<float, 1> vt(sycl::range<1>(16 * 128), cgh);
                    cgh.parallel_for<InkAttT>(sycl::nd_range<1>(size_t(nb) * H * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        auto g = it.get_group();
                        auto sg = it.get_sub_group();
                        const int lid = int(it.get_local_id(0)), r = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
                        const int b = int(it.get_group(0)) / H, j = int(it.get_group(0)) % H;
                        const int m = b * RB + r, mlast = sycl::min(M - 1, b * RB + RB - 1);
                        const bool live = m < M;
                        const int base = *posd;
                        const int pos = base + m;
                        const int count = sycl::min(pos + 1, win), first = pos + 1 - count;
                        const int pfirst = base + b * RB, lo = sycl::max(0, pfirst + 1 - sycl::min(pfirst + 1, win)), hi = base + mlast;
                        const float* A = pA + size_t(live ? m : 0) * NA;
#pragma unroll
                        for (int e = 0; e < 8; ++e) qs[r * 144 + lane * 8 + e] = live ? A[oq + j * D + lane * 8 + e] : 0.f;
                        qs[r * 144 + 128 + lane] = live ? A[orel + j * R + lane] : 0.f;
                        const int hd = (j / groups) * D;
                        float* ktp = kt.template get_multi_ptr<sycl::access::decorated::no>().get();
                        float* vtp = vt.template get_multi_ptr<sycl::access::decorated::no>().get();
                        auto load_tile = [&](float* t, const uint16_t* src, int k0, int pitch) {
                            // 16 keys x 128 dims, 256 threads x 8 elements
                            const int kk = lid / 16, d0 = (lid % 16) * 8, kp = k0 + kk;
                            if (kp <= hi) {
                                const sycl::vec<uint16_t, 8> v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(src + size_t(kp % cap) * K + hd + d0);
#pragma unroll
                                for (int e = 0; e < 8; ++e) t[kk * pitch + d0 + e] = bfw(v[e]);
                            }
                        };
                        auto score = [&](int kp) {
                            const int k = kp & 15;
                            float s = 0.f;
#pragma unroll 8
                            for (int d0 = 0; d0 < 128; ++d0) s += qs[r * 144 + d0] * kt[k * 132 + d0];
                            const float qk = bfr(bfr(s) * scale);
                            const int distance = pos - kp;
                            float bias = 0.f;
                            if (distance < E)
                                for (int rr = 0; rr < 16; ++rr) bias += qs[r * 144 + 128 + rr] * bfw(table[size_t(rr) * E + distance]);
                            return bfr(qk + bfr(bias));
                        };
                        float mx = -INFINITY, l = 0.f;
                        for (int k0 = lo; k0 <= hi; k0 += 16) {
                            sycl::group_barrier(g);
                            load_tile(ktp, kc, k0, 132);
                            sycl::group_barrier(g);
                            const int kp = k0 + lane;
                            if (live && kp >= first && kp <= pos) {
                                const float s = score(kp);
                                if (s > mx) { l = l * sycl::exp(mx - s) + 1.f; mx = s; } else l += sycl::exp(s - mx);
                            }
                        }
                        const float gm = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
                        const float lsum = sycl::reduce_over_group(sg, mx == -INFINITY ? 0.f : l * sycl::exp(mx - gm), sycl::plus<float>());
                        const float sink = live ? bfr(A[osink + j]) : 0.f;
                        const float high = sycl::fmax(sink, gm);
                        const float sum = sycl::exp(sink - high) + lsum * sycl::exp(gm - high);
                        float acc[8];
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[e] = 0.f;
                        for (int k0 = lo; k0 <= hi; k0 += 16) {
                            sycl::group_barrier(g);
                            load_tile(ktp, kc, k0, 132);
                            load_tile(vtp, vc, k0, 128);
                            sycl::group_barrier(g);
                            const int kp = k0 + lane;
                            const float w = (live && kp >= first && kp <= pos) ? bfr(sycl::exp(score(kp) - high) / sum) : 0.f;
#pragma unroll
                            for (int kk = 0; kk < 16; ++kk) {
                                const float wk = sycl::select_from_group(sg, w, kk);
#pragma unroll
                                for (int e = 0; e < 8; ++e) acc[e] = sycl::fma(wk, vt[kk * 128 + lane * 8 + e], acc[e]);
                            }
                        }
                        if (live)
#pragma unroll
                            for (int e = 0; e < 8; ++e) {
                                const int c = j * D + lane * 8 + e;
                                const float o = bfr(bfr(acc[e]) * sigm(bfr(A[og + c])));
                                aout[size_t(m) * Q + c] = o;
                                aoutb[size_t(m) * Q + c] = tobf(o);
                            }
                    });
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
        mark(2);
        if (world_ > 1) {
            if (M == 1 && Ly.WoQ) gemvt8<1>(Ly.WoQ, Ly.WoS, h, Q, 8, aout_, row(Z), h, nullptr, nullptr, h, nullptr, nullptr);
            else if (M == 1) gemvt<1>(Ly.WoT, h, Q, 8, aout_, row(Z), h, nullptr, nullptr, h, nullptr, nullptr);
            else if (M <= 8 && i8m_ks > 0 && Ly.WoQ &&
                     ink_es_gemv_i8m(q_, InkI8MArgs{Ly.WoQ, Ly.WoS, h, Q, M, aout_, aout_, aout_, size_t(Q), row(Z), row(Z), row(Z), size_t(h), size_t(h), size_t(h), h, h, 1, 0.f}, i8m_ks)) {}
            else if (M <= 8 && i8v_ && i8_small(Ly.WoR, Ly.WoS, 0, h, Q, aoutb_, row(Z), M)) {}
            else if (M <= 8) gemvtm<1>(Ly.WoT, h, Q, 8, M, aout_, aout_, aout_, Q, row(Z), h, h, nullptr, 0, h, nullptr, 0, h);
            else dense(Ly.Wo, h, Q, aout_, Q, aoutb_, row(Z), h, M, false);
            static const bool fold_env = !std::getenv("INK_SEP_ADD");
            fold_add_ = fold_env && ((M == 1 && !std::getenv("INK_RMN")) || (M > 1 && M <= 8 && verify_ && !std::getenv("INK_OLD_RESIDV")));
            peer_in_ = nullptr;
            allreduce(row(Z), size_t(M) * h);
            fold_add_ = false;
        } else if (M == 1 && i4_ && i4_gemv(Ly.Wo4, 0, h, aoutb_, row(Z), 1)) {
        } else if (M == 1 && gsm_ && gsm_dense(Ly.Wo, h, Q, aoutb_, row(Z), 1)) {
        } else if (M == 1 && Ly.WoQ) {
            static const int kso = std::getenv("INK_KS_WO") ? std::atoi(std::getenv("INK_KS_WO")) : 8;
            static const int esko = std::getenv("INK_ESI8O") ? std::atoi(std::getenv("INK_ESI8O")) : 8;
            static const bool wo2 = std::getenv("INK_WO_TWICE") != nullptr;   // experiment: L2-hot second call
            if (wo2) ink_es_gemv_i8(q_, InkI8Args{Ly.WoQ, Ly.WoS, h, Q, aout_, aout_, aout_, row(Z), nullptr, nullptr, h, h, 0, 0.f}, esko);
            if (!(esko > 0 && ink_es_gemv_i8(q_, InkI8Args{Ly.WoQ, Ly.WoS, h, Q, aout_, aout_, aout_, row(Z), nullptr, nullptr, h, h, 0, 0.f}, esko)))
                gemvt8<0>(Ly.WoQ, Ly.WoS, h, Q, kso, aout_, row(Z), h, nullptr, nullptr, h, nullptr, nullptr);
        }
        else if (M == 1) gemvt<0>(Ly.WoT, h, Q, 8, aout_, row(Z), h, nullptr, nullptr, h, nullptr, nullptr);
        else if (M <= 8 && i8m_ks > 0 && Ly.WoQ &&
                 ink_es_gemv_i8m(q_, InkI8MArgs{Ly.WoQ, Ly.WoS, h, Q, M, aout_, aout_, aout_, size_t(Q), row(Z), row(Z), row(Z), size_t(h), size_t(h), size_t(h), h, h, 0, 0.f}, i8m_ks)) {}
        else if (M <= 8 && i8v_ && i8_small(Ly.WoR, Ly.WoS, 0, h, Q, aoutb_, row(Z), M)) {}
        else if (M <= 8) gemvtm<0>(Ly.WoT, h, Q, 8, M, aout_, aout_, aout_, Q, row(Z), h, h, nullptr, 0, h, nullptr, 0, h);
        else dense(Ly.Wo, h, Q, aout_, Q, aoutb_, row(Z), h, M, true);
        int bank_row = -1;
        if (v.block_boundary) {
            if (next_bank >= kMaxBankRows) throw std::runtime_error("bank rows exhausted");
            bank_row = BANK0 + next_bank++;
        }
        static const bool rmn_off = !std::getenv("INK_RMN");   // fused residual/mix/norm: opt-in (measured slower, not bit-identical)
        if (M == 1 && !rmn_off && h <= 3072 && c_.conv >= 2) {
            // fused residual + MLP mix (+ norm when no message gate follows)
            if (bank_row >= 0) bank.push_back(bank_row);
            resid_mix_norm(Ly.aconv, sl.ahist, bank_row, job(bank, RES, Ly.mq, XM), Ly.nm, v.has_message ? -1 : NM);
            int xm = XM;
            if (v.has_message) {
                GateBatch gb{};
                int ctrl[1] = {3};
                gb.j[0] = GateJob{XM, M3, GM, Ly.cgain[3], Ly.cforget[3]};
                gb.count = 1;
                gates(gb, Ly, ctrl, M);
                xm = GM;
                NormBatch nb{}; nb.j[0] = NormJob{xm, Ly.nm, NM, nullptr}; nb.count = 1; rms(nb, M);
            }
        } else {
        {   // residual = sconv(z); bank push or + hidden
            const uint16_t* w = Ly.aconv;
            float* hist = sl.ahist;
            const int taps = c_.conv;
            const float cs = conv_static;
            const int br = bank_row;
            static const bool fold_hv2 = !std::getenv("INK_SEP_HISTV");
            float* vbZ = fold_hv2 && M > 1 && verify_ && !std::getenv("INK_HV_NOZ") ? vbuf_ + (size_t(v.cache_slot) * 4 + 2) * 8 * size_t(h) : nullptr;
            const float* pz = peer_in_;   // two cards: the peer's partial of row(Z), added here (exchange add folded)
            peer_in_ = nullptr;
            if (M > 1 && M <= 8 && verify_ && !std::getenv("INK_OLD_RESIDV")) {
                // verify rows: a work-item per channel walks the rows with the conv window in registers (each Z value
                // and peer partial read once; the same per-row arithmetic as InkResid)
                q_.parallel_for<InkResidV>(sycl::range<1>(size_t(h)), [=](sycl::id<1> id) {
                    const int ch = int(id[0]);
                    const int pos0 = *posd;
                    const float* zr = pool + size_t(Z) * RS;
                    float win[3];   // inputs at lags 1, 2, 3 of the current row
#pragma unroll
                    for (int l = 1; l <= 3; ++l) win[l - 1] = hist[size_t(((pos0 - l) % 3 + 3) % 3) * h + ch];
                    const float w0 = bfw(w[ch * 4 + 3]), w1 = bfw(w[ch * 4 + 2]), w2 = bfw(w[ch * 4 + 1]), w3 = bfw(w[ch * 4 + 0]);
                    for (int m = 0; m < M; ++m) {
                        const int pos = pos0 + m;
                        const float zv = pz ? bfr(zr[size_t(m) * h + ch] + pz[size_t(m) * h + ch]) : zr[size_t(m) * h + ch];
                        const float x = bfr(zv);
                        if (vbZ) vbZ[size_t(m) * h + ch] = x;
                        float sum = 0.f;
                        if (0 <= pos) sum += x * w0;
                        if (1 <= pos) sum += win[0] * w1;
                        if (2 <= pos) sum += win[1] * w2;
                        if (3 <= pos) sum += win[2] * w3;
                        const float out = bfr((x + sum) * cs);
                        win[2] = win[1]; win[1] = win[0]; win[0] = x;
                        const size_t o = size_t(m) * h + ch;
                        const float hid = pool[size_t(HID) * RS + o];
                        if (br >= 0) { pool[size_t(br) * RS + o] = hid; pool[size_t(RES) * RS + o] = out; }
                        else pool[size_t(RES) * RS + o] = out + hid;
                    }
                });
            } else
            q_.parallel_for<InkResid>(sycl::range<1>(size_t(M) * h), [=](sycl::id<1> id) {
                const int m = int(id[0] / h), ch = int(id[0] % h);
                const int pos = *posd + m;
                const float* zr = pool + size_t(Z) * RS;
                auto zat = [&](int mm) { return pz ? bfr(zr[size_t(mm) * h + ch] + pz[size_t(mm) * h + ch]) : zr[size_t(mm) * h + ch]; };
                const float x = bfr(zat(m));
                if (vbZ) vbZ[size_t(m) * h + ch] = x;
                float sum = 0.f;
#pragma unroll
                for (int lag = 0; lag < 4; ++lag) {
                    const float xl = lag == 0 ? x : (m - lag >= 0 ? bfr(zat(m - lag)) : hist[size_t(((pos - lag) % 3 + 3) % 3) * h + ch]);
                    if (lag <= pos) sum += xl * bfw(w[ch * 4 + 3 - lag]);
                }
                const float out = bfr((x + sum) * cs);
                if (M == 1) hist[size_t(pos % (taps - 1)) * h + ch] = x;
                const size_t o = size_t(m) * h + ch;
                const float hid = pool[size_t(HID) * RS + o];
                if (br >= 0) { pool[size_t(br) * RS + o] = hid; pool[size_t(RES) * RS + o] = out; }
                else pool[size_t(RES) * RS + o] = out + hid;
            });
            if (M > 1 && verify_) { if (!vbZ) hist_update_v(v.cache_slot, 2, row(Z), h, h, M); }
            else if (M > 1) hist_update(sl.ahist, row(Z), h, h, M);
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
        }
        mark(3);
        // router: sigmoid scores, top-k by score + bias, weights = scale * score / sum, sorted by expert id
        static const bool old_route = std::getenv("INK_OLD_ROUTE") != nullptr;
        const bool fused_route = M == 1 && !old_route && c_.experts <= 128;
        if (fused_route) {
            // decode: one 256-thread work-group per expert computes its logit; the last work-group to finish (device
            // counter) does InkTopk's selection -- one launch instead of an 8-work-group GEMV + a top-k kernel
            const uint16_t* Wr = Ly.Wr;
            const float* x = row(NM);
            float* lg = rlog_;
            const float* bias = Ly.rbias;
            int* sel = sel_;
            float* wts = wts_;
            int* cnt = route_cnt_;
            const int En = c_.experts, TK = c_.top_k;
            const float rscale = c_.route_scale;
            q_.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> key(sycl::range<1>(128), cgh), score(sycl::range<1>(128), cgh);
                sycl::local_accessor<int, 1> byrank(sycl::range<1>(16), cgh);
                sycl::local_accessor<float, 1> total(sycl::range<1>(1), cgh);
                sycl::local_accessor<int, 1> last(sycl::range<1>(1), cgh);
                cgh.parallel_for<InkRouteTopk>(sycl::nd_range<1>(size_t(En) * 256, 256), [=](sycl::nd_item<1> it) {
                    auto g = it.get_group();
                    const int ex = int(it.get_group(0)), lid = int(it.get_local_id(0));
                    const uint16_t* wr = Wr + size_t(ex) * h;
                    float acc = 0.f;
                    for (int i = lid; i < h; i += 256) acc += bfw(wr[i]) * x[i];
                    acc = sycl::reduce_over_group(g, acc, sycl::plus<float>());
                    if (lid == 0) {
                        lg[ex] = acc;
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::global_space> c(*cnt);
                        last[0] = c.fetch_add(1) == En - 1 ? 1 : 0;
                    }
                    sycl::group_barrier(g);
                    if (!last[0]) return;
                    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
                    const int e = lid;
                    if (e < En) {
                        const float lv = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(lg[e]).load();
                        const float sc = sigm(lv); score[e] = sc; key[e] = sc + bias[e];
                    }
                    sycl::group_barrier(g);
                    int rank = 1 << 30;
                    if (e < En) {
                        rank = 0;
                        for (int f = 0; f < En; ++f) rank += (key[f] > key[e] || (key[f] == key[e] && f < e)) ? 1 : 0;
                        if (rank < TK) byrank[rank] = e;
                    }
                    sycl::group_barrier(g);
                    if (e == 0) { float s2 = 0.f; for (int r2 = 0; r2 < TK; ++r2) s2 += score[byrank[r2]]; total[0] = s2; *cnt = 0; }
                    sycl::group_barrier(g);
                    if (rank < TK) {
                        int at = 0;
                        for (int r2 = 0; r2 < TK; ++r2) at += byrank[r2] < e ? 1 : 0;
                        sel[at] = e;
                        wts[at] = rscale * (score[e] / (total[0] + 1e-20f));
                    }
                });
            });
        } else if (M == 1) gemvt<1>(Ly.WrT, c_.experts, h, 32, row(NM), rlog_, c_.experts, nullptr, nullptr, c_.experts, nullptr, nullptr);
        else if (M <= 8 && gsmv_ && gsm_dense(Ly.Wr, c_.experts, h, nmb_, rlog_, M)) {}
        else if (M <= 8) gemvtm<1>(Ly.WrT, c_.experts, h, 32, M, row(NM), row(NM), row(NM), h, rlog_, c_.experts, c_.experts, nullptr, 0, c_.experts, nullptr, 0, c_.experts);
        else dense(Ly.Wr, c_.experts, h, row(NM), h, nmb_, rlog_, c_.experts, M, false);
        {
            const float* lg = rlog_;
            const float* bias = Ly.rbias;
            int* sel = sel_;
            float* wts = wts_;
            const int En = c_.experts, TK = c_.top_k;
            const float rscale = c_.route_scale;
            static const bool fuse_rg = std::getenv("INK_ROUTEGRP") != nullptr;   // verify rows: top-k + grouping in one work-group (MEASURED 11-24: slower, 34.8 -> 36.9 ms/step; opt-in)
            route_deferred_ = !fused_route && fuse_rg && M >= 2 && M <= 8 && En <= 128 && M * TK <= 64;
            if (!fused_route && !route_deferred_) q_.submit([&](sycl::handler& cgh) {
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
        mark(4);
        if (M == 1) experts_decode(Ly);
        else if (M <= 8) experts_small(Ly, M);
        else experts_prompt(Ly, M);
        if (world_ > 1) {
            if (M == 1) {   // this card's experts' weighted sum (others' slots hold zeros), unrounded
                const float* dbuf = dbuf_;
                const float* wts = wts_;
                float* yb = row(YB);
                const int TK = c_.top_k;
                q_.parallel_for<InkPartial>(sycl::range<1>(size_t(h)), [=](sycl::id<1> id) {
                    float acc = 0.f;
                    for (int s2 = 0; s2 < TK; ++s2) acc += wts[s2] * dbuf[size_t(s2) * h + id[0]];
                    yb[id[0]] = acc;
                });
            }
            static const bool fold_env2 = !std::getenv("INK_SEP_ADD");
            fold_add_ = fold_env2 && (M == 1 || (M > 1 && M <= 8 && verify_ && !std::getenv("INK_OLD_RESIDV")));
            peer_in_ = nullptr;
            allreduce(row(YB), size_t(M) * h);
            fold_add_ = false;
        }
        mark(5);
        {   // hidden = bf(residual + sconv(y)); y = bf(sum_s w_s d_s) in slot (= ascending expert) order
            const uint16_t* w = Ly.mconv;
            float* hist = sl.mhist;
            const float* dbuf = dbuf_;
            const float* wts = wts_;
            const int TK = c_.top_k, taps = c_.conv;
            const float cs = conv_static;
            const bool tp = world_ > 1;
            static const bool fold_hv3 = !std::getenv("INK_SEP_HISTV");
            float* vbY = fold_hv3 && M > 1 && verify_ && !std::getenv("INK_HV_NOY") ? vbuf_ + (size_t(v.cache_slot) * 4 + 3) * 8 * size_t(h) : nullptr;
            const float* py = peer_in_;   // two cards: the peer's partial of row(YB), added here (exchange add folded)
            peer_in_ = nullptr;
            if (M > 1 && M <= 8 && verify_ && !std::getenv("INK_OLD_RESIDV")) {
                // verify rows: a work-item per channel walks the rows (each YB value and peer partial read once)
                q_.parallel_for<InkMlpConvV>(sycl::range<1>(size_t(h)), [=](sycl::id<1> id) {
                    const int ch = int(id[0]);
                    const int pos0 = *posd;
                    const float* yb = pool + size_t(YB) * RS;
                    float win[3];
#pragma unroll
                    for (int l = 1; l <= 3; ++l) win[l - 1] = hist[size_t(((pos0 - l) % 3 + 3) % 3) * h + ch];
                    const float w0 = bfw(w[ch * 4 + 3]), w1 = bfw(w[ch * 4 + 2]), w2 = bfw(w[ch * 4 + 1]), w3 = bfw(w[ch * 4 + 0]);
                    for (int m = 0; m < M; ++m) {
                        const int pos = pos0 + m;
                        const float x = py ? bfr(yb[size_t(m) * h + ch] + py[size_t(m) * h + ch]) : yb[size_t(m) * h + ch];
                        if (vbY) vbY[size_t(m) * h + ch] = bfr(x);
                        float sum = 0.f;
                        if (0 <= pos) sum += x * w0;
                        if (1 <= pos) sum += win[0] * w1;
                        if (2 <= pos) sum += win[1] * w2;
                        if (3 <= pos) sum += win[2] * w3;
                        const float y = bfr((x + sum) * cs);
                        win[2] = win[1]; win[1] = win[0]; win[0] = x;
                        const size_t o = size_t(m) * h + ch;
                        pool[size_t(HID) * RS + o] = bfr(pool[size_t(RES) * RS + o] + y);
                    }
                });
            } else
            q_.parallel_for<InkMlpConv>(sycl::range<1>(size_t(M) * h), [=](sycl::id<1> id) {
                const int m = int(id[0] / h), ch = int(id[0] % h);
                const int pos = *posd + m;
                const float* yb = pool + size_t(YB) * RS;
                float x;
                if (M == 1 && !tp) {
                    float acc = 0.f;
#pragma unroll
                    for (int s = 0; s < 8; ++s) acc += wts[s] * dbuf[size_t(s) * h + ch];   // top_k == 8 (geometry check)
                    x = bfr(acc);
                } else x = py ? bfr(yb[size_t(m) * h + ch] + py[size_t(m) * h + ch]) : yb[size_t(m) * h + ch];
                if (vbY) vbY[size_t(m) * h + ch] = bfr(x);
                float sum = 0.f;
#pragma unroll
                for (int lag = 0; lag < 4; ++lag) {
                    const float yl = m - lag >= 0 ? (py ? bfr(yb[size_t(m - lag) * h + ch] + py[size_t(m - lag) * h + ch]) : yb[size_t(m - lag) * h + ch]) : 0.f;
                    const float xl = lag == 0 ? x : (m - lag >= 0 ? yl : hist[size_t(((pos - lag) % 3 + 3) % 3) * h + ch]);
                    if (lag <= pos) sum += xl * bfw(w[ch * 4 + 3 - lag]);
                }
                const float y = bfr((x + sum) * cs);
                if (M == 1) hist[size_t(pos % (taps - 1)) * h + ch] = x;
                const size_t o = size_t(m) * h + ch;
                pool[size_t(HID) * RS + o] = bfr(pool[size_t(RES) * RS + o] + y);
            });
            if (M > 1 && verify_) { if (!vbY) hist_update_v(v.cache_slot, 3, row(YB), h, h, M); }
            else if (M > 1) hist_update(sl.mhist, row(YB), h, h, M);
        }
        mark(6);
    }
    if (!pp_last()) {   // a pipeline stage before the last: hand over the state, advance the position
        pp_out_ = PPState{bank, pre_bank, prev_bank, next_bank};
        int* pd = pos_d_;
        q_.single_task<InkPosP>([=]() { *pd += M; });
        return;
    }
    const bool allrows = all_logits_ || taps_on;
    {   // output mix, norm (all rows when the drafter needs the taps or a verify needs every row's logits)
        MixBatch mb{};
        mb.j[0] = job(bank, HID, oquery_, X0); mb.count = 1;
        mix(mb, allrows ? M : 1, allrows ? 0 : M - 1);
        const bool padU = (i4_ && un4_.pay && M <= 8 && !allrows) || (i8v_ && unR_ && M <= 8 && all_logits_);
        NormBatch nb{}; nb.j[0] = NormJob{X0, fnorm_, N0, padU ? xpadU_ : nullptr, kp_}; nb.count = 1;
        rms(nb, allrows ? M : 1, allrows ? 0 : M - 1);
        if (taps_on) copy(N0, TAP3);
    }
    const int V = c_.output_vocab;
    static const int i8m_un = std::getenv("INK_I8M") ? std::atoi(std::getenv("INK_I8M")) : 0;
    if (all_logits_ && logitsR_ && M <= 8 &&
        ((i8m_un > 0 && unembedQ_ &&
          ink_es_gemv_i8m(q_, InkI8MArgs{unembedQ_, unembedS_, vrows_, h, M, row(N0), row(N0), row(N0), size_t(h), logitsR_, logitsR_, logitsR_,
                                         size_t(vrows_), size_t(vrows_), size_t(vrows_), vrows_, vrows_, 1, 0.f}, i8m_un)) ||
         (i8v_ && unR_ && i8_small(unR_, unembedS_, 0, vrows_, kp_, xpadU_, logitsR_, M)))) {
        // INT8 unembed for the verify rows (b70::launch_fp8_smallm, raw), softcap + argmax in one pass
        if (world_ == 1) argmax_rows_cap(logitsR_, vrows_, V, M, next_, nullptr, 0, c_.logit_cap);
        else argmax_rows_cap(logitsR_, vrows_, vvalid_, M, next_, bestv_, v0_, c_.logit_cap);
        int* pd = pos_d_;
        q_.single_task<InkPosV2>([=]() { *pd += M; });
        return;
    }
    if (all_logits_) {
        if (world_ == 1) {
            gemvtm<2>(unembedT_, c_.vocab, h, 1, M, row(N0), row(N0), row(N0), h, logitsM_, V, c_.vocab, nullptr, 0, c_.vocab, nullptr, 0, V, c_.logit_cap);
            argmax_rows(logitsM_, V, M, next_);
        } else {   // this card's vocabulary half; best (value, global index) per row, picked across cards on the host
            gemvtm<2>(unembedT_, vrows_, h, 1, M, row(N0), row(N0), row(N0), h, logitsM_, V, vrows_, nullptr, 0, vrows_, nullptr, 0, vvalid_, c_.logit_cap);
            argmax_rows_v(logitsM_, V, vvalid_, M, next_, bestv_, v0_);
        }
        int* pd = pos_d_;
        q_.single_task<InkPosV>([=]() { *pd += M; });
        return;
    }
    const int VL = vvalid_, vbase = v0_;
    const bool raw_logits = i4_ && un4_.pay && M <= 8 && !all_logits_ && !taps_on && !want_logits_ && i4_gemv(un4_, 0, vrows_, xpadU_ + size_t(M - 1) * kp_, logits_, 1);   // softcap in the argmax
    last_raw_ = raw_logits;
    if (raw_logits) {
    } else if (unembedQ_) gemvt8<2>(unembedQ_, unembedS_, vrows_, h, 2, row(N0, M - 1), logits_, vrows_, nullptr, nullptr, vrows_, nullptr, nullptr, c_.logit_cap);
    else gemv<2>(unembed_, VL, h, row(N0, M - 1), logits_, VL, nullptr, nullptr, VL, nullptr, nullptr, c_.logit_cap);
    {
        const float* lg = logits_;
        float* bv = amx_v_;
        int* bi = amx_i_;
        int* nx = next_;
        float* bestv = bestv_;
        const float rawcap = raw_logits ? c_.logit_cap : 0.f;   // raw INT4 logits: softcap here (same formula as gemv<2>)
        const int per = 256 * 16, G = (VL + per - 1) / per;
        q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lv(sycl::range<1>(256), cgh);
            sycl::local_accessor<int, 1> li(sycl::range<1>(256), cgh);
            cgh.parallel_for<InkArg1>(sycl::nd_range<1>(size_t(G) * 256, 256), [=](sycl::nd_item<1> it) {
                const int lid = int(it.get_local_id(0)), base = int(it.get_group(0)) * per;
                float best = -INFINITY; int bidx = 0x7fffffff;
                for (int t = 0; t < 16; ++t) {
                    const int k = base + t * 256 + lid;
                    if (k >= VL) continue;
                    const float v = rawcap > 0.f ? rawcap * sycl::tanh(bfr(lg[k]) / rawcap) : lg[k];
                    if (v > best || (v == best && k < bidx)) { best = v; bidx = k; }
                }
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
                if (lid == 0) { *nx = li[0] + vbase; *bestv = lv[0]; }
            });
        });
    }
    int* pd = pos_d_;
    q_.single_task<InkPos>([=]() { *pd += M; });
}

#include "inkling2_mtp.inc"

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
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 6) {
        std::fprintf(stderr, "usage: inkling2_gpu MODEL_DIR check IDS CONTEXT NEW_TOKENS [REF_LOGITS|-] [CHUNK]\n"
                             "       inkling2_gpu MODEL_DIR bench IDS CONTEXT PP_LIST TG [CHUNK]\n");
        return 2;
    }
    const std::string mode = argv[2];
    const auto ids = read_ids(argv[3]);
    const int ctx = std::atoi(argv[4]);
    const auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.size() != 1 && mode != "tp" && mode != "tpmtp" && mode != "pp") throw std::runtime_error("expected exactly one visible GPU (set ONEAPI_DEVICE_SELECTOR)");
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
        {   // warm-up: graph capture, GEMM kernels (one full chunk: oneDNN's kernels for M = chunk), expert path
            std::vector<int> w(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), size_t(std::min(chunk, 256))));
            int nx = eng.prefill(w);
            for (int i = 0; i < 4; ++i) nx = eng.step(nx);
            eng.reset();
            if (ids.size() >= size_t(chunk) && chunk + 8 <= ctx) {
                std::vector<int> fc(ids.begin(), ids.begin() + chunk);
                eng.prefill(fc);
                eng.reset();
            }
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
            if (std::getenv("INK_SECT")) {
                static const char* nm[8] = {"pre-attn mix/norm/Wp", "q/k/v prep", "attention", "Wo+residual+mix+norm", "router+gates", "experts", "post", "embed"};
                for (int k = 0; k < 8; ++k) { std::printf("  sect %-22s %8.1f ms\n", nm[k], eng.sect_ms_[k]); eng.sect_ms_[k] = 0; }
            }
            std::fflush(stdout);
        }
        return 0;
    }
    if (mode == "mtp") {
        // inkling2_gpu MODEL mtp IDS CTX PP N [CHUNK]: plain greedy vs DSpark MTP after a PP-token prompt
        const int pp = std::atoi(argv[5]);
        const int n = argc > 6 ? std::atoi(argv[6]) : 128;
        const int chunk = argc > 7 ? std::atoi(argv[7]) : 2048;
        Engine eng(q, argv[1], ctx, chunk);
        eng.load_mtp(std::string(argv[1]) + "/mtp.safetensors");
        std::vector<int> p(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), size_t(pp)));
        // plain greedy
        int nx = eng.prefill(p);
        std::vector<int> plain{nx};
        auto t0 = std::chrono::steady_clock::now();
        for (int g = 1; g < n; ++g) { nx = eng.step(nx); plain.push_back(nx); }
        const double tp = ms_since(t0);
        // MTP
        eng.reset();
        nx = eng.prefill(p);
        std::vector<int> spec{nx};
        int steps = 0;
        t0 = std::chrono::steady_clock::now();
        while (int(spec.size()) < n) {
            const auto got = eng.mtp_step(spec.back());
            spec.insert(spec.end(), got.begin(), got.end());
            ++steps;
        }
        const double ts = ms_since(t0);
        spec.resize(n);
        int same = 0;
        while (same < n && plain[same] == spec[same]) ++same;
        std::printf("prompt %d tokens, %d new tokens\n", int(p.size()), n);
        std::printf("plain head:"); for (int g = 0; g < std::min(n, 20); ++g) std::printf(" %d", plain[g]);
        std::printf("\nmtp head:  "); for (int g = 0; g < std::min(n, 20); ++g) std::printf(" %d", spec[g]);
        std::printf("\n");
        std::printf("plain greedy: %.1f tok/s\n", 1000.0 * (n - 1) / tp);
        std::printf("MTP (DSpark block 8): %.1f tok/s, %d steps, %.2f tokens/step, %.1f ms/step\n", 1000.0 * (n - 1) / ts, steps,
                    double(n - 1) / steps, ts / steps);
        std::printf("MTP output equals plain greedy for the first %d of %d tokens\n", same, n);
        // verify / rollback alone: drafts forced to the plain continuation, then to wrong tokens
        for (int mode2 = 0; mode2 < 2; ++mode2) {
            eng.reset();
            nx = eng.prefill(p);
            std::vector<int> f{nx};
            int st2 = 0;
            while (int(f.size()) < n) {
                int dr[7];
                for (int k = 0; k < 7; ++k) {
                    const size_t at = f.size() + k;
                    dr[k] = mode2 == 0 ? (at < plain.size() ? plain[at] : 0) : 0;
                }
                const auto got = eng.mtp_step(f.back(), dr);
                f.insert(f.end(), got.begin(), got.end());
                ++st2;
            }
            f.resize(n);
            int sm = 0;
            while (sm < n && plain[sm] == f[sm]) ++sm;
            std::printf("forced %s drafts: %d steps, output equals plain for the first %d of %d tokens\n", mode2 == 0 ? "correct" : "wrong", st2, sm, n);
        }
        if (same < n) {
            std::printf("plain:"); for (int g = std::max(0, same - 3); g < std::min(n, same + 8); ++g) std::printf(" %d", plain[g]);
            std::printf("\nmtp:  "); for (int g = std::max(0, same - 3); g < std::min(n, same + 8); ++g) std::printf(" %d", spec[g]);
            std::printf("\n");
        }
        return 0;
    }
    if (mode == "pp") {
        // inkling2_gpu MODEL pp IDS CTX PP_LIST TG [CHUNK]: two cards as a two-stage pipeline (each holds the full
        // weights; card 0 runs the first half of the visits, card 1 the rest).  Prompt chunks overlap across the
        // cards; decoding passes each token through both stages.  First line: greedy check on the first prompt.
        std::vector<int> pps;
        { std::stringstream ss(argv[5]); std::string t; while (std::getline(ss, t, ',')) pps.push_back(std::atoi(t.c_str())); }
        const int tg = argc > 6 ? std::atoi(argv[6]) : 128;
        const int chunk = argc > 7 ? std::atoi(argv[7]) : 2048;
        if (devs.size() != 2) throw std::runtime_error("pp: expected two GPUs");
        sycl::context c0(devs[0]), c1(devs[1]);
        sycl::queue q0(c0, devs[0], sycl::property_list{sycl::property::queue::in_order()});
        sycl::queue q1(c1, devs[1], sycl::property_list{sycl::property::queue::in_order()});
        Engine E0(q0, argv[1], ctx, chunk), E1(q1, argv[1], ctx, chunk);
        const int S = std::getenv("INK_PP_SPLIT") ? std::atoi(std::getenv("INK_PP_SPLIT")) : E0.n_visits() / 2;
        E0.set_visits(0, S);
        E1.set_visits(S, -1);
        const int maxr = 3 + kMaxBankRows;
        E0.pp_alloc(maxr); E1.pp_alloc(maxr);
        const size_t slot_el = size_t(maxr) * chunk * 2304;
        uint16_t* slot[2] = {sycl::malloc_host<uint16_t>(slot_el, q0), sycl::malloc_host<uint16_t>(slot_el, q0)};
        if (!slot[0] || !slot[1]) throw std::runtime_error("pp: pinned slots");
        std::printf("pipeline: card 0 visits [0,%d), card 1 [%d,%d), chunk %d\n", S, S, E0.n_visits(), chunk);
        // one prompt through the pipeline, chunks overlapped (two host slots)
        auto prefill2 = [&](const std::vector<int>& p) -> int {
            const int n = int(p.size());
            std::vector<std::pair<int, int>> ch;
            for (int at = 0; at < n; at += chunk) ch.push_back({at, std::min(chunk, n - at)});
            std::mutex mu; std::condition_variable cv;
            int ready[2] = {-1, -1}; bool busy[2] = {false, false};
            Engine::PPState st[2];
            int out = -1;
            std::exception_ptr err;
            std::thread t1([&] {
                try {
                    for (size_t k = 0; k < ch.size(); ++k) {
                        const int sidx = int(k % 2);
                        { std::unique_lock<std::mutex> l(mu); cv.wait(l, [&] { return ready[sidx] == int(k); }); }
                        E1.pp_import(slot[sidx], ch[k].second, st[sidx]);
                        E1.sync_all();
                        { std::lock_guard<std::mutex> l(mu); ready[sidx] = -1; busy[sidx] = false; }
                        cv.notify_all();
                        out = E1.pp_run(p.data() + ch[k].first, ch[k].second);
                    }
                } catch (...) { err = std::current_exception(); std::lock_guard<std::mutex> l(mu); cv.notify_all(); }
            });
            for (size_t k = 0; k < ch.size(); ++k) {
                const int sidx = int(k % 2);
                E0.pp_run(p.data() + ch[k].first, ch[k].second);
                { std::unique_lock<std::mutex> l(mu); cv.wait(l, [&] { return !busy[sidx] || err; }); }
                if (err) break;
                E0.pp_export(slot[sidx], ch[k].second);
                { std::lock_guard<std::mutex> l(mu); st[sidx] = E0.pp_out_; busy[sidx] = true; ready[sidx] = int(k); }
                cv.notify_all();
            }
            t1.join();
            if (err) std::rethrow_exception(err);
            return out;
        };
        auto step2 = [&](int tok) -> int {
            E0.pp_run(&tok, 1);
            E0.pp_export(slot[0], 1);
            E1.pp_import(slot[0], 1, E0.pp_out_);
            return E1.pp_run(&tok, 1);
        };
        auto reset2 = [&] { E0.reset(); E1.reset(); };
        {   // warm-up + greedy check (compare with a one-card run of the same prompt)
            std::vector<int> w(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), 64));
            int nx = prefill2(w);
            std::printf("two-stage greedy after %zu prompt tokens:", w.size());
            std::printf(" %d", nx);
            for (int g = 1; g < 16; ++g) { nx = step2(nx); std::printf(" %d", nx); }
            std::printf("\n");
            reset2();
            if (ids.size() >= size_t(chunk) && chunk + 8 <= ctx) { std::vector<int> fc(ids.begin(), ids.begin() + chunk); prefill2(fc); reset2(); }
        }
        std::printf("| test | tokens | time (ms) | tok/s |\n|---|---|---|---|\n");
        for (int pp : pps) {
            if (size_t(pp) > ids.size() || pp + tg > ctx) continue;
            reset2();
            std::vector<int> p(ids.begin(), ids.begin() + pp);
            E0.sync_all(); E1.sync_all();
            auto t0 = std::chrono::steady_clock::now();
            int nx = prefill2(p);
            const double pms = ms_since(t0);
            t0 = std::chrono::steady_clock::now();
            for (int g = 0; g < tg; ++g) nx = step2(nx);
            const double tms = ms_since(t0);
            std::printf("| pp%d | %d | %.1f | %.1f |\n| tg%d @ %d | %d | %.1f | %.1f |\n", pp, pp, pms, 1000.0 * pp / pms, tg, pp, tg, tms, 1000.0 * tg / tms);
        }
        sycl::free(slot[0], q0); sycl::free(slot[1], q0);
        return 0;
    }
    if (mode == "tp" || mode == "tpmtp") {
        // tpmtp: inkling2_gpu MODEL tpmtp IDS CTX PP N [CHUNK]: two cards, plain greedy vs DSpark MTP
        // inkling2_gpu MODEL tp IDS CTX PP_LIST TG [CHUNK]: two cards (both visible), greedy check of the first prompt + bench
        std::vector<int> pps;
        { std::stringstream ss(argv[5]); std::string t; while (std::getline(ss, t, ',')) pps.push_back(std::atoi(t.c_str())); }
        const int tg = argc > 6 ? std::atoi(argv[6]) : 128;
        const int chunk = argc > 7 ? std::atoi(argv[7]) : 2048;
        if (devs.size() != 2) throw std::runtime_error("tp: expected two GPUs");
        // one context per card unless INK_TPCTX=shared: no object of one device is ever used by the other's queue
        const bool shared = std::getenv("INK_TPCTX") && std::string(std::getenv("INK_TPCTX")) == "shared";
        std::printf("two cards, %s contexts\n", shared ? "one shared" : "separate");
        sycl::context cs(devs), c0(devs[0]), c1(devs[1]);
        // INK_DEVX=1: device mailboxes in pinned host memory of the two-card context; each card's queue in its own SYCL
        // context over that same native Level Zero context (Xenithium rt/fabric.cpp), so both kernels can touch the boxes
        const bool devx = std::getenv("INK_DEVX") && std::atoi(std::getenv("INK_DEVX")) == 1;
        auto per_card = [&](int r) {
            if (!devx) return r == 0 ? c0 : c1;
            const auto nat = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(cs);
            return sycl::make_context<sycl::backend::ext_oneapi_level_zero>(
                sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context>{nat, {devs[r]}, sycl::ext::oneapi::level_zero::ownership::keep});
        };
        sycl::context k0 = per_card(0), k1 = per_card(1);
        sycl::queue q0(shared ? cs : k0, devs[0], sycl::property_list{sycl::property::queue::in_order()});
        sycl::queue q1(shared ? cs : k1, devs[1], sycl::property_list{sycl::property::queue::in_order()});
        TPX tpx;
        if (devx) {
            for (int r = 0; r < 2; ++r) {
                tpx.box[r] = sycl::malloc_host<uint32_t>(2 * TPX::kBoxWords, cs);
                if (!tpx.box[r]) throw std::runtime_error("mailbox");
                std::fill(tpx.box[r], tpx.box[r] + 2 * TPX::kBoxWords, kEmpty);
            }
            tpx.err = sycl::malloc_host<uint32_t>(2, cs);
            tpx.err[0] = tpx.err[1] = 0;
            std::printf("device mailboxes in pinned host memory (system atomics on host USM: %s / %s)\n",
                        devs[0].has(sycl::aspect::usm_atomic_host_allocations) ? "yes" : "no", devs[1].has(sycl::aspect::usm_atomic_host_allocations) ? "yes" : "no");
        }
        g_tp_progress = now_ms();
        std::thread([&] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                if (now_ms() - g_tp_progress.load() > 60000) {
                    std::printf("WATCHDOG: no progress for 60 s; card 0 stage %d, card 1 stage %d (1 copy-out wait, 2 thread meet, 4 token read)\n",
                                g_tp_stage[0].load(), g_tp_stage[1].load());
                    std::_Exit(3);
                }
            }
        }).detach();
        std::unique_ptr<Engine> eng[2];
        {   // rank 0 allocates the shared host slots: load it first
            eng[0] = std::make_unique<Engine>(q0, argv[1], ctx, chunk, 0, 2, &tpx);
            g_tp_progress = now_ms();
            eng[1] = std::make_unique<Engine>(q1, argv[1], ctx, chunk, 1, 2, &tpx);
            g_tp_progress = now_ms();
        }
        for (auto& e : eng) e->use_graph = false;
        const bool mtp_mode = mode == "tpmtp";
        auto work_mtp = [&](int r) {
            Engine& E = *eng[r];
            E.load_mtp(std::string(argv[1]) + "/mtp.safetensors");
            g_tp_progress = now_ms();
            const int n = tg;
            std::vector<int> p(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), size_t(pps[0])));
            {   // warm-up
                int nx = E.prefill(p);
                for (int i = 0; i < 4; ++i) nx = E.step(nx);
                E.reset();
                nx = E.prefill(p);
                for (int i = 0; i < 2; ++i) { const auto got = E.mtp_step(nx); nx = got.back(); }
                E.reset();
            }
            int nx = E.prefill(p);
            std::vector<int> plain{nx};
            E.sync_all();
            auto t0 = std::chrono::steady_clock::now();
            for (int g = 1; g < n; ++g) { nx = E.step(nx); plain.push_back(nx); }
            const double tpl = ms_since(t0);
            E.reset();
            nx = E.prefill(p);
            std::vector<int> spec{nx};
            int steps = 0;
            E.sync_all();
            g_tp_n[r] = 0; g_tp_wait_ms[r][0] = g_tp_wait_ms[r][1] = 0;
            E.mtp_ms_[0] = E.mtp_ms_[1] = E.mtp_ms_[2] = 0;
            t0 = std::chrono::steady_clock::now();
            while (int(spec.size()) < n) {
                const auto got = E.mtp_step(spec.back());
                spec.insert(spec.end(), got.begin(), got.end());
                ++steps;
            }
            const double ts = ms_since(t0);
            E.sync_all();
            if (r == 0) for (int c = 0; c < 2; ++c) std::printf("  card %d MTP: %ld exchanges (%.1f / step), copy-out wait %.1f ms, thread meet %.1f ms; draft %.1f verify %.1f commit %.1f ms\n", c, g_tp_n[c],
                                                                double(g_tp_n[c]) / steps, g_tp_wait_ms[c][0], g_tp_wait_ms[c][1], eng[c]->mtp_ms_[0], eng[c]->mtp_ms_[1], eng[c]->mtp_ms_[2]);
            spec.resize(n);
            int same = 0;
            while (same < n && plain[same] == spec[same]) ++same;
            if (r == 0) {
                std::printf("two cards, prompt %d tokens, %d new tokens\n", int(p.size()), n);
                std::printf("plain head:"); for (int g = 0; g < std::min(n, 16); ++g) std::printf(" %d", plain[g]);
                std::printf("\nplain greedy: %.1f tok/s\n", 1000.0 * (n - 1) / tpl);
                std::printf("MTP: %.1f tok/s, %d steps, %.2f tokens/step, %.1f ms/step\n", 1000.0 * (n - 1) / ts, steps, double(n - 1) / steps, ts / steps);
                std::printf("MTP output equals plain greedy for the first %d of %d tokens\n", same, n);
            }
        };
        auto work = [&](int r) {
            if (mtp_mode) { work_mtp(r); return; }
            Engine& E = *eng[r];
            {   // warm-up + 5-token greedy check
                std::vector<int> five(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), 5));
                int nx = E.prefill(five);
                std::vector<int> out{nx};
                for (int g = 1; g < 12; ++g) { nx = E.step(nx); out.push_back(nx); }
                if (r == 0) { std::printf("two-card greedy after 5 prompt tokens:"); for (int v : out) std::printf(" %d", v); std::printf("\n"); }
                if (r == 0 && tpx.err) std::printf("  mailbox spin timeouts after check: card 0 %u, card 1 %u\n", tpx.err[0], tpx.err[1]);
                E.reset();
            }
            if (r == 0) std::printf("| test | tokens | time (ms) | tok/s |\n|---|---|---|---|\n");
            for (int pp : pps) {
                if (size_t(pp) > ids.size() || pp + tg > ctx) continue;
                E.reset();
                std::vector<int> p(ids.begin(), ids.begin() + pp);
                auto t0 = std::chrono::steady_clock::now();
                int nx = E.prefill(p);
                const double pms = ms_since(t0);
                E.sync_all();
                g_tp_n[r] = 0; g_tp_wait_ms[r][0] = g_tp_wait_ms[r][1] = 0;
                t0 = std::chrono::steady_clock::now();
                for (int g = 0; g < tg; ++g) nx = E.step(nx);
                const double tms = ms_since(t0);
                if (r == 0) { std::printf("| pp%d | %d | %.1f | %.1f |\n| tg%d @ %d | %d | %.1f | %.1f |\n", pp, pp, pms, 1000.0 * pp / pms, tg, pp, tg, tms, 1000.0 * tg / tms); std::fflush(stdout); }
                if (r == 0) for (int c = 0; c < 2; ++c) std::printf("  card %d: %ld exchanges, copy-out wait %.1f ms, thread meet %.1f ms\n", c, g_tp_n[c], g_tp_wait_ms[c][0], g_tp_wait_ms[c][1]);
                if (r == 0 && tpx.err) std::printf("  mailbox spin timeouts: card 0 %u, card 1 %u\n", tpx.err[0], tpx.err[1]);
            }
        };
        std::thread t1(work, 1);
        work(0);
        t1.join();
        return 0;
    }
    throw std::invalid_argument("mode must be check, bench, mtp or tp");
} catch (const std::exception& e) {
    std::fprintf(stderr, "inkling2_gpu: %s\n", e.what());
    return 1;
}
