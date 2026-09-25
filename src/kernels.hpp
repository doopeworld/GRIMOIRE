// =====================================================================
//  kernels.hpp  --  SYCL side of the engine
// =====================================================================
#ifndef B70_KERNELS_HPP
#define B70_KERNELS_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include "b70/weights.hpp"
#include "b70/b70q4.hpp"

namespace b70 {

using sycl_bf16 = sycl::ext::oneapi::bfloat16;

// Bridge between the SYCL-free header type and the SYCL one. Identical
// layout, so this is a reinterpret, not a conversion.
inline sycl_bf16 to_sycl_bf16(float f) { return sycl_bf16(f); }

// ---------------------------------------------------------------------
// Xe2 (Battlemage) execution geometry.
//
// SG_SIZE 16 is the native SIMD width the XMX DPAS pipe is built around.
// Do not change it without changing the joint_matrix tile shapes: the
// hardware only exposes M in 1..8, N == 16 (at SIMD16), K == 16 for
// bf16 and K == 32 for int8.
// ---------------------------------------------------------------------
constexpr int SG_SIZE = 16;
// Widest head the flash kernels can hold.  Each accumulates a head into a
// PRIVATE array of MAX_DPL floats per lane (dpl = head_dim / SG_SIZE), so a
// wider head writes past the end of device stack memory -- a DEVICE_LOST,
// not a wrong number.  Declared here so attention.cpp, prefill.cpp and the
// engine's capability check cannot drift apart.
constexpr int MAX_DPL      = 16;
constexpr int MAX_HEAD_DIM = MAX_DPL * SG_SIZE;

// WIDE instantiation, for heads the 16-slot accumulator cannot hold.
// gemma-4's full-attention layers are head_dim 512.  Each flash kernel is
// a template on its accumulator width and is instantiated TWICE: every
// head_dim <= MAX_HEAD_DIM keeps the 16-slot kernel it has always used --
// same private array, same register footprint, nothing to re-measure --
// and only a wider head reaches the 32-slot one.  Widening the constant
// itself would have doubled the private array for EVERY model, including
// Qwen at head_dim 128, which is a register-pressure change to the hot
// decode kernel that cannot be judged off the card (rule 8).
constexpr int MAX_DPL_WIDE      = 32;
constexpr int MAX_HEAD_DIM_WIDE = MAX_DPL_WIDE * SG_SIZE;

// Upper bound on FlashDecoding split-K chunks for single-token decode.
// GRAPH_SPLITS (8) sizes the graph-recorded path and never scaled with
// context depth: measured 2026-09-04 on Qwen3.8-27B at 4778 tokens, the 16
// full-attention layers cost 3650 us EACH (54.8 of 82 ms per token) at an
// achieved 4.6 GB/s against a 602 GB/s roofline -- exactly the latency-bound
// regime the flash-decode comment warns about. 8 splits x 40 heads is 320
// sub-groups on a 256-EU part, and each split then walks ~600 keys serially.
// The partial (acc,m,l) workspace is sized for this bound; the merge already
// skips empty splits, so over-splitting a short sequence is harmless.
constexpr int MAX_SPLITS = 128;

constexpr int TM      = 8;    // joint_matrix M
constexpr int TN      = 16;   // joint_matrix N
constexpr int TK_BF16 = 16;   // joint_matrix K, bf16 DPAS
constexpr int TK_INT8 = 32;   // joint_matrix K, int8 DPAS (2x rate)

// Work-group tiling for the prefill GEMM.
constexpr int WG_SUBGROUPS = 8;

// Two-stage argmax: stage-1 work-groups, and the partial buffers they
// write. Owned by the engine so the hot path allocates nothing.
constexpr int kArgmaxGroups = 512;
extern float*   g_argmax_pv;
extern int32_t* g_argmax_pi;

// Register blocking for the prefill GEMM.
//
// Measured at M_PER_SG=1, N_PER_SG=4 the kernel reached 5.5 TFLOP/s --
// 3.1% of the ~180 TFLOP/s bf16 DPAS peak. The limiter is fragment
// reuse: 4 MADs for every 5 fragment loads leaves the matrix pipe idle
// waiting on SLM.
//
// A 4x4 block issues 16 MADs from 8 loads, doubling the arithmetic per
// byte moved out of SLM. Each sub-group then owns a 32x64 output tile
// and holds 16 accumulators, which fits comfortably in the 256-GRF
// register file the build already requests.
// MEASURED by sweep on Arc Pro B70, [8192x2048] x 4096 tokens:
//   M_PER_SG=1   5509 GFLOP/s
//   M_PER_SG=2   8778 GFLOP/s   <- best
//   M_PER_SG=4   6907 GFLOP/s   (register spill)
// Two blocks balances fragment reuse against register pressure; four
// spills and one under-uses the matrix pipe. Neither extreme was
// predictable from first principles -- the sweep found it.
constexpr int N_PER_SG     = 4;                       // TN blocks across N
constexpr int M_PER_SG     = 2;                       // override: GRIMOIRE_MPSG
constexpr int WG_N         = TN * N_PER_SG;                  // 64
constexpr int WG_K_BF16    = 32;
constexpr int WG_K_INT8    = 64;
// The int8 pipeline keeps the original 1-block-per-sub-group shape, so
// it needs its own M tile. Sharing WG_M after the float path grew to
// 4 blocks would leave it computing 64 of every 256 rows and silently
// dropping the rest.
// Each sub-group now owns M_PER_SG_INT blocks of TM rows.  This doubles the
// work-group's M tile, which halves how many times the int4 B tile has to be
// unpacked into SLM -- that unpack, not the DPAS, is what limits this kernel.
constexpr int M_PER_SG_INT = 2;
constexpr int WG_M_INT     = TM * M_PER_SG_INT * WG_SUBGROUPS;   // 128
constexpr int WG_M         = TM * M_PER_SG * WG_SUBGROUPS;   // 128

// Decode GEMV: elements each lane consumes per step. 16 was chosen so a
// lane's chunk never straddles an MX block (32) or an INT4 group (128),
// which keeps the scale load uniform and out of the inner loop.
constexpr int GEMV_EPL  = 16;
constexpr int GEMV_STEP = SG_SIZE * GEMV_EPL;         // 256 elements / sub-group / step

constexpr int ROWS_PER_SG = 4;

// Elements per lane per step, and the unroll depth.
//
// MEASURED on a B70, not derived. Both knobs trade off against each
// other and the optimum is format-specific:
//
//   fmt     EPL=16        EPL=32       EPL=64
//   bf16    502 (83%)     --           --
//   int8    450 (74%)     389 (64%)    --
//   int4    294 (48%)     --           377 (62%)
//   fp8     295 (49%)     257 (42%)    --
//   mxfp8   220 (36%)     200 (33%)    --
//   mxfp4   184 (30%)     137 (23%)    --
//
// Bigger EPL fully unrolls a longer inner loop; past a point the
// register pressure spills and throughput falls. int4 is the exception
// because its 4-bit loads are so narrow that it needs the extra width.
//
// Defaults below are the best measured value per format. Override at
// runtime with B70_EPL / B70_UNROLL to re-sweep on other silicon --
// several variants are compiled in, so no rebuild is needed.
//
// UPPER BOUND IS CORRECTNESS, NOT TUNING: a lane's chunk must not cross
// a scale boundary (MX block = 32, INT4 group = 128) or the hoisted
// scale is wrong.
// Per-format defaults, MEASURED by tools/sweep.sh on a B70 (BMG-G31),
// 16384x16384 GEMV. Every format has a different optimum, which is why a
// single global setting cannot work:
//
//   fmt       EPL/UNROLL   GB/s   %peak
//   bf16         64 / 2    543.3  89.4
//   int8         16 / 8    481.4  79.2
//   int4         16 / 4    386.2  63.5
//   mxfp8        16 / 8    353.9  58.2
//   mxfp4        16 / 2    345.1  56.8
//   fp8_e4m3     16 / 8    342.5  56.3
//
// The tradeoff is register pressure: wider EPL and deeper UNROLL both
// expand the fully-unrolled inner loop, and past a point the spills cost
// more than the extra memory-level parallelism buys. bf16 tolerates the
// widest loads because its elements need no decode at all.
//
// EPL_MAX is a CORRECTNESS bound, not a tuning one: a lane's chunk must
// not cross a scale boundary (MX block = 32, INT4 group = 128) or the
// hoisted scale applies to the wrong elements.
template <Fmt F> struct GemvGeom {
    static constexpr int EPL_DEFAULT = 16;
    static constexpr int UNROLL_DEFAULT = 4;
    static constexpr int EPL_MAX     = 64;
};
template <> struct GemvGeom<Fmt::BF16>     { static constexpr int EPL_DEFAULT = 64; static constexpr int UNROLL_DEFAULT = 2; static constexpr int EPL_MAX = 64; };
template <> struct GemvGeom<Fmt::INT8>     { static constexpr int EPL_DEFAULT = 16; static constexpr int UNROLL_DEFAULT = 8; static constexpr int EPL_MAX = 64; };
template <> struct GemvGeom<Fmt::FP8_E4M3> { static constexpr int EPL_DEFAULT = 16; static constexpr int UNROLL_DEFAULT = 8; static constexpr int EPL_MAX = 64; };
template <> struct GemvGeom<Fmt::FP8_E5M2> { static constexpr int EPL_DEFAULT = 16; static constexpr int UNROLL_DEFAULT = 8; static constexpr int EPL_MAX = 64; };
template <> struct GemvGeom<Fmt::MXFP8>    { static constexpr int EPL_DEFAULT = 16; static constexpr int UNROLL_DEFAULT = 8; static constexpr int EPL_MAX = 32; };
template <> struct GemvGeom<Fmt::MXFP4>    { static constexpr int EPL_DEFAULT = 16; static constexpr int UNROLL_DEFAULT = 2; static constexpr int EPL_MAX = 32; };
template <> struct GemvGeom<Fmt::INT4>     { static constexpr int EPL_DEFAULT = 16; static constexpr int UNROLL_DEFAULT = 4; static constexpr int EPL_MAX = 64; };

// Runtime knobs, read once from the environment.
int gemv_epl_override();      // 0 = use per-format default
int gemv_unroll_override();   // 0 = use default (4)
void set_gemv_tuning(int epl, int unroll, int wide); // runtime autotune

sycl::event launch_gemm_xmx(sycl::queue& q, const QuantWeight& w,
                            const sycl_bf16* x, float* y, int M,
                            const std::vector<sycl::event>& deps = {});
// Large-M prompt GEMM (src/gemm_fast.cpp; bin/libgrimoire_gemm.so in the
// AOT binaries).  launch_gemm_xmx() takes it whenever the shape fits.
bool gemm_fast_supported(const QuantWeight& w, int M);
sycl::event launch_gemm_fast(sycl::queue& q, const QuantWeight& w,
                             const sycl_bf16* x, float* y, int M,
                             const std::vector<sycl::event>& deps = {});
// Matrix-unit causal prefill attention (src/gemm_fast.cpp);
// launch_flash_prefill() takes it whenever the geometry fits.
bool flash_fast_supported(int head_dim, int num_heads, int num_kv_heads);
sycl::event launch_flash_prefill_fast(sycl::queue& q, const float* qv,
    const uint8_t* k_cache, const uint8_t* v_cache, float* out, int tokens,
    int start_pos, int num_heads, int num_kv_heads, int head_dim, int seq_cap,
    float softmax_scale, const std::vector<sycl::event>& deps = {});
sycl::event launch_quantize_rows_int8(sycl::queue& q, const float* x,
                                      int8_t* xq, float* scales, int M, int K,
                                      const std::vector<sycl::event>& deps = {});
sycl::event launch_gemm_xmx_int(sycl::queue& q, const QuantWeight& w,
                                const int8_t* xq, const float* scales,
                                float* y, int M,
                                const std::vector<sycl::event>& deps = {});
sycl::event launch_gemm_b70q4(sycl::queue& q, const B70Q4View& w,
                              const int8_t* xq, const float* x_scales,
                              float* y, int M,
                              const std::vector<sycl::event>& deps = {});

// y[N] = W[N][K] . x[K]             -- decode, bandwidth bound
sycl::event launch_gemv(sycl::queue& q, const QuantWeight& w,
                        const float* x, float* y,
                        const std::vector<sycl::event>& deps = {});

// FlashDecoding over a paged KV cache
struct AttnParams {
    const float* q;          // [num_heads][head_dim]
    const uint8_t* k_cache;  // FP8 E4M3 [num_kv_heads][head_dim][seq_cap]
    const uint8_t* v_cache;  // FP8 E4M3 [num_kv_heads][seq_cap][head_dim]
    float*       out;        // [num_heads][head_dim]
    int seq_len, seq_cap, head_dim, num_heads, num_kv_heads;
    float softmax_scale;

    // Split-K workspace. `splits` chunks per head, each producing an
    // unnormalized partial plus its (m, l) statistics.
    float* partials = nullptr;   // [num_heads][splits][head_dim]
    float* part_m   = nullptr;   // [num_heads][splits]
    float* part_l   = nullptr;   // [num_heads][splits]
    int    splits   = 1;

    // Graph capture: seq_len changes every token, so it is read from
    // device memory and `splits` is held FIXED. Chunks that fall past
    // the current sequence end emit an empty partial (-inf, 0), which
    // the merge skips -- so a constant launch shape stays correct at
    // every context length.
    const int32_t* d_seq_len = nullptr;   // if set, overrides seq_len

    // Sliding-attention window, in keys, counting the current position.
    // <= 0 means the whole history, which is every model here except
    // gemma-4's sliding layers.  Applied by NARROWING the scanned range
    // rather than by masking scores: a key outside the window contributes
    // nothing, so skipping it is both exact and cheaper.  Attending to the
    // full history where a window was meant is completely silent -- it is
    // identical until the context passes the window, then quietly wrong.
    int window_left = 0;
};

// Pick enough chunks to fill the machine without shredding the sequence.
inline int pick_splits(int seq_len, int num_heads, int eus) {
    const int want = (eus * 4) / (num_heads > 0 ? num_heads : 1);
    int s = want < 1 ? 1 : want;
    const int max_by_len = (seq_len + 255) / 256;      // >=256 keys per chunk
    if (s > max_by_len) s = max_by_len;
    if (s < 1) s = 1;
    if (s > 64) s = 64;
    return s;
}

struct DeltaNetParams {
    // NOTE: q and k have n_k_heads, v has n_v_heads, and n_v = 2*n_k for
    // this model. The reference does repeat_interleave(2) on q and k, so
    // v-head h reads q/k head h/2. Indexing q/k by the v-head number
    // instead runs off the end of both buffers for the upper half of the
    // heads -- which produces fluent-looking garbage, not a crash.
    const float* q;        // [n_k_heads][k_dim]  L2-normalized, post-conv
    const float* k;        // [n_k_heads][k_dim]  L2-normalized, post-conv
    const float* v;        // [n_v_heads][v_dim]
    const float* a;        // [n_v_heads] decay, already exp()'d
    const float* beta;     // [n_v_heads] gate, already sigmoid()'d
    float*       state;    // [n_v_heads][v_dim][k_dim]  read AND written
    float*       out;      // [n_v_heads][v_dim]
    int n_heads, k_dim, v_dim;
    int n_k_heads = 0;      // 0 means same as n_heads
};

struct ConvParams {
    const float* x;         // [channels]      this token's projection
    const bf16_t* weight;   // [channels][K]   as stored in the checkpoint
    float*       ring;     // [channels][K-1] previous tokens, rolling
    float*       out;      // [channels]
    int channels, kernel;
};

// Prefill variant: all M tokens in one launch, state resident in SLM.
struct DeltaNetPrefillParams {
    const float* q;        // [M][n_heads][k_dim]  L2-normalized, post-conv
    const float* k;        // [M][n_heads][k_dim]
    const float* v;        // [M][n_heads][v_dim]
    const float* a;        // [M][n_heads]
    const float* beta;     // [M][n_heads]
    float*       state;    // [n_heads][v_dim][k_dim] carried in and out
    float*       out;      // [M][n_heads][v_dim]
    int n_heads, k_dim, v_dim, n_tokens;
    int n_k_heads = 0;
};

sycl::event launch_dequant_bf16(sycl::queue& q, const QuantWeight& w,
                                sycl_bf16* dst,
                                const std::vector<sycl::event>& deps = {});

sycl::event launch_f32_to_bf16(sycl::queue& q, const float* src, sycl_bf16* dst,
                               size_t n, const std::vector<sycl::event>& deps = {});
sycl::event launch_bf16_to_f32(sycl::queue& q, const sycl_bf16* src, float* dst,
                               size_t n, const std::vector<sycl::event>& deps = {});
sycl::event launch_f32_to_bf16_scaled(sycl::queue& q, const float* src,
    sycl_bf16* dst, size_t n, float scale,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_rmsnorm_gate_silu_bf16_io(sycl::queue& q,
    const sycl_bf16* x,const float* z,const bf16_t* w,sycl_bf16* out,
    int n_heads,int dim,float eps,const std::vector<sycl::event>& deps = {});
sycl::event launch_gate_sigmoid_mul_bf16_io(sycl::queue& q,
    const sycl_bf16* x,const float* gate,sycl_bf16* out,size_t n,
    const std::vector<sycl::event>& deps = {});

// ---- Qwen4-Exp PLE n-gram embedding (host ref: b70/qwen4_exp.hpp) ----
// The 20M-row table itself lives in HOST memory (vLLM pins it too); these
// produce the ids to gather and the gate that scales the result.
sycl::event launch_ple_ngram_ids(sycl::queue& q, const int32_t* tokens,
    int64_t* out, int first, int n_tokens, const int64_t* multipliers,
    const int64_t* sizes, const int64_t* offsets, int ngram_context_len,
    int heads_per_ngram, int ngram_heads, int eos_token_id,
    const std::vector<sycl::event>& deps = {});
// Per (token, stream): the two grouped norms have their own (1 + w)
// affine, and BOTH outputs are produced here because conv_in is normed
// from `gated`.  `value` is H wide and gates into every stream.
sycl::event launch_ple_gate(sycl::queue& q, const float* key,
    const float* value, const float* query, const bf16_t* nk,
    const bf16_t* nq, const bf16_t* ncw, float* gated, float* conv_in,
    int rows, int hc_count, int H, float eps,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_ple_conv(sycl::queue& q, const float* conv_in,
    const float* gated, const float* hidden, const bf16_t* w, float* out,
    int first, int rows, int channels, int kernel, int dilation,
    const std::vector<sycl::event>& deps = {});
// `table` is HOST memory: the n-gram table is the one weight in this
// engine that does not live in VRAM, which is the architecture's design.
// BF16, or FP8-E4M3 with one global `scale`.
sycl::event launch_ple_embed_gather(sycl::queue& q, const void* table,
    bool fp8, float scale, const int64_t* ids, float* out, int rows,
    int ngram_heads, int head_dim, int64_t table_rows,
    const std::vector<sycl::event>& deps = {});
// ---- Qwen4-Exp QSA (host reference: b70/qwen4_exp.hpp) ---------------
sycl::event launch_qsa_attention(sycl::queue& q, const float* qv,
    const uint8_t* k_cache, const uint8_t* v_cache, const int32_t* idx,
    float* out, int rows, int n_heads, int kv_heads, int head_dim,
    int seq_cap, int n_idx, float softmax_scale,
    const std::vector<sycl::event>& deps = {});

// Stage 1 scores every compressed key block with the MQA indexer; stage 3
// turns the selected blocks into token indices.  Stage 2 (top-k) reuses
// the engine's existing top-k, and the attention that consumes the index
// list is launch_qsa_attention in attention.cpp.
// `keys` is the sequence's compressed key cache, [n_blocks][head_dim],
// SHARED by every query row; only `qv`, `logits` and `visible` are
// per-row.  `logits` is [rows][n_blocks].
sycl::event launch_qsa_index_logits(sycl::queue& q, const float* qv,
    const float* keys, float* logits, int rows, int n_heads, int head_dim,
    int n_blocks, const int32_t* visible,
    const std::vector<sycl::event>& deps = {});
// Pool a completed group of raw index keys, then norm and RoPE the
// POOLED key at the position of the group's FIRST token.
sycl::event launch_rope_rows(sycl::queue& q, float* x, int rows, int heads,
    int dim, int first, float theta, float partial_factor,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_copy_rows_strided(sycl::queue& q, const float* src,
    float* dst, int rows, int src_stride, int dst_stride, int width,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_qsa_row_meta(sycl::queue& q, int32_t* visible,
    int32_t* seq_len, int32_t* query_pos, int rows, int first, int total,
    int compress_ratio, const std::vector<sycl::event>& deps = {});
sycl::event launch_qsa_pool_blocks(sycl::queue& q, const float* raw,
    float* pooled, int first_block, int n_blocks, int compress_ratio,
    int head_dim, const std::vector<sycl::event>& deps = {});
sycl::event launch_qsa_rope_blocks(sycl::queue& q, float* keys,
    int first_block, int n_blocks, int head_dim, int compress_ratio,
    float theta, float partial_factor,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_qsa_topk_blocks(sycl::queue& q, const float* logits,
    int32_t* out, int rows, int n_blocks, int topk, const int32_t* visible,
    const std::vector<sycl::event>& deps = {});
// `out` is token_topk + compress_ratio - 1 wide: the trailing entries are
// the INCOMPLETE block, which stage 1 cannot see and which contains the
// query's own token.  `query_pos` is per row.
sycl::event launch_qsa_expand_blocks(sycl::queue& q, const int32_t* blocks,
    int32_t* out, int rows, int block_topk, int compress_ratio,
    int token_topk, const int32_t* seq_len, const int32_t* query_pos,
    const std::vector<sycl::event>& deps = {});
// ---- Qwen4-Exp HyperConnections (host reference: b70/qwen4_exp.hpp) ---
// The residual stream is hc_count wide; mix() collapses it for the block
// and combine() injects the block output back into every stream.  The
// two projections use the ordinary GEMV -- only these three have no
// existing kernel.
sycl::event launch_hc_norm(sycl::queue& q, const float* x, const bf16_t* w,
    float* out, int rows, int hc_count, int hidden, float eps,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_hc_gated_mean(sycl::queue& q, const float* up_out,
    const float* normed, float* out, int rows, int hc_count, int hidden,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_hc_combine(sycl::queue& q, const float* hyper,
    const float* inj_out, const float* block, float* out, int rows,
    int hc_count, int hidden, const std::vector<sycl::event>& deps = {});
// silu(down_out / hc_count) -- the divide is INSIDE the nonlinearity.
sycl::event launch_hc_silu(sycl::queue& q, float* x, int n, int hc_count,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_gemm_batched(sycl::queue& q, const QuantWeight& w,
                                const float* x, float* y, int M,
                                const std::vector<sycl::event>& deps = {});
sycl::event launch_deltanet_prefill(sycl::queue& q, const DeltaNetPrefillParams& p,
                                    const std::vector<sycl::event>& deps = {});
sycl::event launch_rmsnorm_residual_batched(
    sycl::queue& q, float* h, const float* r0, const float* r1,
    const bf16_t* weight, float* out, int tokens, int hidden, float eps,
    sycl_bf16* out_bf = nullptr,
    const std::vector<sycl::event>& deps = {}, float weight_offset = 1.0f);
sycl::event launch_rmsnorm_residual_f16_batched(
    sycl::queue& q, float* h, const float* residual, const bf16_t* weight,
    float* out, int tokens, int hidden, float eps,
    const std::vector<sycl::event>& deps = {}, float weight_offset = 0.0f);
// FP16-storage activation variants; see prefill.cpp for why these are
// numerically identical to the f32 versions.
sycl::event launch_embed_f16_h(sycl::queue&, const sycl::half*, const int32_t*,
    sycl::half*, int, int, const std::vector<sycl::event>& = {});
sycl::event launch_add_f16_round_h(sycl::queue&, sycl::half*, const sycl::half*,
    int, const std::vector<sycl::event>& = {});
sycl::event launch_gate_sigmoid_mul_h(sycl::queue&, sycl::half*,
    const sycl::half*, int, int, const std::vector<sycl::event>& = {});
sycl::event launch_swiglu_h(sycl::queue&, const sycl::half*, sycl::half*, int,
    int, const std::vector<sycl::event>& = {});
sycl::event launch_dflash_store_tap_h(sycl::queue&, const sycl::half*, float*,
    int, int, int, int, int, const std::vector<sycl::event>& = {});
sycl::event launch_rmsnorm_residual_f16w_h(sycl::queue&, sycl::half*,
    const sycl::half*, const sycl::half*, sycl::half*, int, int, float,
    const std::vector<sycl::event>& = {}, float weight_offset = 0.0f);
sycl::event launch_muse_post_attn_pre_ff_h(sycl::queue&, sycl::half*,
    sycl::half*, const sycl::half*, const sycl::half*, sycl::half*, int, int,
    float, float, const std::vector<sycl::event>& = {});
sycl::event launch_rmsnorm_residual_f16w_batched(
    sycl::queue& q, float* h, const float* residual, const sycl::half* weight,
    float* out, int tokens, int hidden, float eps,
    const std::vector<sycl::event>& deps = {}, float weight_offset = 0.0f);
sycl::event launch_rmsnorm_residual_batched_quant(
    sycl::queue& q, float* h, const float* r0, const float* r1,
    const bf16_t* weight, float* out, sycl_bf16* out_bf,
    int8_t* out_q, float* out_scale, int tokens, int hidden, float eps,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_rmsnorm_moe_residual_batched(sycl::queue& q,float* h,
    const sycl_bf16* routed,const int32_t* inverse,const float* route_weight,
    const float* shared,const bf16_t* weight,float* out,sycl_bf16* out_bf,
    int tokens,int top_k,int hidden,float eps,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_causal_conv1d_prefill(
    sycl::queue& q, const ConvParams& p, int tokens,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_causal_conv1d_split_prefill(
    sycl::queue& q,const ConvParams& p,int tokens,float* qv,float* kv,float* vv,
    sycl_bf16* vv_bf,
    int qk_size,int v_size,const std::vector<sycl::event>& deps = {});
sycl::event launch_causal_conv1d_split_bf16_prefill(sycl::queue& q,
    const sycl_bf16* x,const bf16_t* weight,float* ring,int channels,int kernel,
    int tokens,sycl_bf16* qv,sycl_bf16* kv,sycl_bf16* vv,int qk_size,int v_size,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_router_topk_batched(
    sycl::queue& q, const float* logits, int tokens, int n_experts, int top_k,
    int32_t* out_expert, float* out_weight, bool normalize,
    const std::vector<sycl::event>& deps = {});
// ---- K2-Horizon ------------------------------------------------------
// zero_centered is FALSE for K2 (weight initialised to ones, applied
// directly); Qwen3.5's norms are zero-centered and apply (1 + w).
// weight_offset is 1.0 for Qwen3.5's zero-centered norms ((1 + w)) and
// 0.0 for K2, whose norm weight is initialised to ones and applied
// directly.  set_norm_convention is called once by Grimoire::build so no
// call site can keep the wrong convention.
void set_norm_convention(int groups, float weight_offset);
// Read back what is live, so a banner cannot drift from the kernels.
void get_norm_convention(int* groups, float* weight_offset);
bool norm_is_grouped(int hidden);
extern int   g_norm_groups;
extern float g_norm_weight_offset;
sycl::event launch_rmsnorm_grouped(sycl::queue& q, float* h,
    const float* r0, const float* r1, const bf16_t* weight, float* out,
    sycl_bf16* out_bf, int tokens, int hidden, int n_groups, float eps,
    float weight_offset, const std::vector<sycl::event>& deps = {});
sycl::event launch_softplus_gate(sycl::queue& q, const float* attn,
    const float* gate, float* out, int64_t n, float beta,
    const std::vector<sycl::event>& deps = {});
// Fused MoVA value projection: routes stay on the device.  `w` is the E
// experts packed expert-major as one [E*N][K] weight; rex/rwt are the
// [M][top_k] routing table the K2 router wrote.  See ops.cpp.
sycl::event launch_mova_value_packed(
    sycl::queue& q, const QuantWeight& w, const float* x,
    const int32_t* rex, const float* rwt, float* y,
    int M, int N, int E, int top_k,
    const std::vector<sycl::event>& deps = {});
// GeGLU -- gelu_pytorch_tanh(gate) * up.  Gemma asks for this and NOT
// silu; substituting one for the other is silent.  See ops.cpp.
sycl::event launch_geglu(sycl::queue& q, const float* gate, const float* up,
    float* out, int n, const std::vector<sycl::event>& deps = {});
sycl::event launch_geglu_batched(sycl::queue& q, const float* gu, float* out,
    int rows, int inter, const std::vector<sycl::event>& deps = {});
// Proportional RoPE -- gemma-4 full-attention layers.  NOT partial_rope:
// the exponent divides by the full head_dim and the pairing is over
// head_dim/2.  See ops.cpp and ref/gemma4_proportional_rope.py.
sycl::event launch_rope_proportional(sycl::queue& q, float* x, int n_heads,
    int head_dim, const int32_t* d_pos, float theta, float partial_factor,
    const std::vector<sycl::event>& deps = {}, float freq_divisor = 1.0f);
sycl::event launch_silu_scale_accum(sycl::queue& q, const float* in, float* out,
    float w, int n, const std::vector<sycl::event>& deps = {});
sycl::event launch_router_topk_k2(
    sycl::queue& q, const float* logits, const bf16_t* bias,
    int tokens, int n_experts, int top_k,
    int32_t* out_expert, float* out_weight, bool normalize, float scaling,
    const std::vector<sycl::event>& deps = {});

sycl::event launch_router_topk_bf16_batched(
    sycl::queue& q, const sycl_bf16* logits, int tokens, int n_experts,
    int top_k, int32_t* out_expert, float* out_weight, bool normalize,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_qk_norm_rope_batched(
    sycl::queue& q, float* qv, float* kv, const bf16_t* qw, const bf16_t* kw,
    int tokens, int q_heads, int k_heads, int dim, int start_pos,
    float theta, float partial_factor, float eps,
    const std::vector<sycl::event>& deps = {}, float weight_offset = 1.0f);
// gemma-4 full-attention layers.  Same norm, a DIFFERENT rotation --
// see the definition in prefill.cpp; not interchangeable with the call
// above by passing a different partial_factor.
sycl::event launch_qk_norm_rope_proportional_batched(
    sycl::queue& q, float* qv, float* kv, const bf16_t* qw, const bf16_t* kw,
    int tokens, int q_heads, int k_heads, int dim, int start_pos,
    float theta, float partial_factor, float eps,
    const std::vector<sycl::event>& deps = {}, float weight_offset = 1.0f,
    float freq_divisor = 1.0f);
sycl::event launch_kv_append_batched(
    sycl::queue& q, const float* k, const float* v, uint8_t* k_cache,
    uint8_t* v_cache, int tokens, int start_pos, int n_kv_heads,
    int head_dim, int seq_cap, const std::vector<sycl::event>& deps = {});
sycl::event launch_qk_norm_rope_f16_batched(
    sycl::queue& q, const float* q_src, const float* k_src, sycl::half* q_dst,
    sycl::half* k_dst, const bf16_t* q_weight, const bf16_t* k_weight,
    int tokens, int q_heads, int k_heads, int head_dim, int start_pos,
    float theta, float eps, const std::vector<sycl::event>& deps = {},
    bool use_rope = true, float query_scale = 1.0f,
    float weight_offset = 0.0f);
sycl::event launch_qkv_norm_rope_f16_fused(
    sycl::queue& q, const sycl::half* qkv_src, sycl::half* q_dst,
    sycl::half* k_dst, sycl::half* v_dst, const bf16_t* q_weight,
    const bf16_t* k_weight, int tokens, int q_heads, int k_heads,
    int head_dim, int start_pos, float theta, float eps,
    const std::vector<sycl::event>& deps = {}, bool use_rope = true,
    float query_scale = 1.0f, float weight_offset = 0.0f);
sycl::event launch_qkv_norm_rope_f16w_fused(
    sycl::queue& q, const sycl::half* qkv_src, sycl::half* q_dst,
    sycl::half* k_dst, sycl::half* v_dst, const sycl::half* q_weight,
    const sycl::half* k_weight, int tokens, int q_heads, int k_heads,
    int head_dim, int start_pos, float theta, float eps,
    const std::vector<sycl::event>& deps = {}, bool use_rope = true,
    float query_scale = 1.0f, float weight_offset = 0.0f);
sycl::event launch_f32_to_f16(sycl::queue& q, const float* src,
    sycl::half* dst, size_t count,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_f16_to_f32(sycl::queue& q, const sycl::half* src,
    float* dst, size_t count,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_kv_append_f16_paged(
    sycl::queue& q, const sycl::half* k, const sycl::half* v,
    sycl::half* k_cache, sycl::half* v_cache, int tokens, int start_pos,
    int n_kv_heads, int head_dim, int block_size,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_dflash_context_kv_f16(
    sycl::queue& q, const float* fused_kv, sycl::half* all_k,
    sycl::half* all_v, const bf16_t* stacked_k_norm, int layers, int tokens,
    int kv_heads, int head_dim, int start_pos, float theta, float eps,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_dflash_context_kv_f16w(
    sycl::queue& q, const float* fused_kv, sycl::half* all_k,
    sycl::half* all_v, const sycl::half* stacked_k_norm, int layers, int tokens,
    int kv_heads, int head_dim, int start_pos, float theta, float eps,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_flash_prefill(
    sycl::queue& q, const float* qv, const uint8_t* k_cache,
    const uint8_t* v_cache, float* out, int tokens, int start_pos,
    int num_heads, int num_kv_heads, int head_dim, int seq_cap,
    float softmax_scale, const std::vector<sycl::event>& deps = {});

// DFlash query attention. Query K/V rows have already been appended at
// [context_len, context_len + tokens). Some trained heads use a non-causal
// block while sliding-attention heads such as Muse are causal.
sycl::event launch_dflash2_block_attention(
    sycl::queue& q, const float* qv, const uint8_t* k_cache,
    const uint8_t* v_cache, float* out, int tokens, int context_len,
    int num_heads, int num_kv_heads, int head_dim, int seq_cap,
    int sliding_window, bool causal, float softmax_scale,
    const std::vector<sycl::event>& deps = {});

// DFlash2-only elementwise and selector kernels.  Activations remain f32 in
// Grimoire; the explicit bf16 round points mirror the reference model.
// Target taps are stored token-major as [position, tap_count, hidden], so
// fc.weight can consume one token's residual streams without a gather.
sycl::event launch_dflash_store_tap(
    sycl::queue& q, const float* src, float* taps, int rows, int hidden,
    int tap_count, int start_pos, int tap,
    const std::vector<sycl::event>& deps = {}, bool fp16_round = false);
sycl::event launch_dflash_store_tap_dev(
    sycl::queue& q, const float* src, float* taps, int hidden,
    int tap_count, const int32_t* position, int tap,
    const std::vector<sycl::event>& deps = {}, bool fp16_round = false);
sycl::event launch_dflash2_grouped_conv(
    sycl::queue& q, const float* x, const float* coefficients,
    const bf16_t* base, float* out, int rows, int hidden, int taps,
    int group_size, int block_size, int side,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_topk16_rows(
    sycl::queue& q, const float* logits, int rows, int vocab,
    int32_t* out_ids, float* out_values,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_dflash2_selector_edges(
    sycl::queue& q, const bf16_t* predecessor, const bf16_t* successor,
    const int32_t* candidate_ids, const float* unary,
    const float* projected_hidden, int32_t anchor_token,
    float* scores, int steps, int top_k, int rank,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_dflash2_path_walk(
    sycl::queue& q, const float* scores, const int32_t* candidate_ids,
    int32_t* tokens, int steps, int top_k,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_embed_batched(sycl::queue& q, const bf16_t* table,
    const int32_t* tokens, float* out, int count, int hidden,
    const std::vector<sycl::event>& deps = {});
// Row-sharded counterpart of launch_embed_batched, for tensor parallel.
// Each rank holds vocabulary rows [begin, begin+rows) of the table; a token
// outside that range contributes ZERO from this rank, and the caller sums
// the ranks with an all-reduce.  Writing zero (rather than skipping the
// row) is what makes that sum exact.
sycl::event launch_embed_batched_shard(sycl::queue& q, const bf16_t* table,
    const int32_t* tokens, float* out, int count, int hidden,
    int begin, int rows, const std::vector<sycl::event>& deps = {});
sycl::event launch_embed_f16_batched(sycl::queue& q, const sycl::half* table,
    const int32_t* tokens, float* out, int count, int hidden,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_split_deltanet_qkv_batched(sycl::queue& q, const float* src,
    float* qv, float* kv, float* vv, int tokens, int qk_size, int v_size,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_split_dn_fused_projections(sycl::queue& q,const float* src,
    float* qkv,float* z,float* ab,int tokens,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_split_qgate_batched(sycl::queue& q, const float* src,
    float* qout, float* gout, int tokens, int heads, int dim,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_split_qgate_bf16(sycl::queue& q,const sycl_bf16* src,
    sycl_bf16* qout,float* gout,int tokens,int heads,int dim,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_qk_norm_rope_bf16_batched(sycl::queue& q,sycl_bf16* qv,
    sycl_bf16* kv,const bf16_t* qw,const bf16_t* kw,int tokens,int q_heads,
    int k_heads,int dim,int start_pos,float theta,float partial_factor,float eps,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_kv_append_bf16_batched(sycl::queue& q,const sycl_bf16* k,
    const sycl_bf16* v,uint8_t* k_cache,uint8_t* v_cache,int tokens,int start_pos,
    int n_kv_heads,int head_dim,int seq_cap,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_deltanet_gates_batched(sycl::queue& q, const float* ab,
    const bf16_t* A_log, const bf16_t* dt_bias, float* alpha, float* beta,
    int tokens, int heads, const std::vector<sycl::event>& deps = {});
sycl::event launch_deltanet_native_gates(sycl::queue& q, const float* ab,
    float* gate_a_head_major, float* beta_head_major, int tokens, int heads, int64_t stride,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_l2norm_heads_pair_bf16(sycl::queue& q,const float* qsrc,
    const float* ksrc,sycl_bf16* qdst,sycl_bf16* kdst,int n_heads,int dim,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_l2norm_heads_pair_bf16_io(sycl::queue& q,sycl_bf16* qv,
    sycl_bf16* kv,int n_heads,int dim,const std::vector<sycl::event>& deps = {});
sycl::event launch_swiglu_batched(sycl::queue& q, const float* gu, float* out,
    int tokens, int inter, const std::vector<sycl::event>& deps = {});
sycl::event launch_swiglu_f16_batched(sycl::queue& q, const float* gu,
    float* out, int tokens, int inter,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_swiglu_bf16_split(sycl::queue& q, const sycl_bf16* gate,
    const sycl_bf16* up, sycl_bf16* out, int tokens, int inter,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_swiglu_bf16(sycl::queue& q, const sycl_bf16* gu,
    sycl_bf16* out, int tokens, int inter,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_swiglu_bf16_quant(sycl::queue& q, const sycl_bf16* gu,
    sycl_bf16* out, int8_t* out_q, float* out_scale, int tokens, int inter,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_scale_by_sigmoid_batched(sycl::queue& q, float* x,
    const float* gate, int tokens, int hidden,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_gate_sigmoid_mul_batched(sycl::queue& q, float* x,
    const float* gate, int tokens, int hidden,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_permute_rows_bf16(sycl::queue& q, const float* src,
    const int32_t* perm_token, sycl_bf16* dst, int rows, int width,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_moe_unpermute(sycl::queue& q, const float* src,
    const int32_t* inverse, const float* route_weight, float* dst,
    int tokens, int top_k, int hidden,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_moe_unpermute_bf16(sycl::queue& q, const sycl_bf16* src,
    const int32_t* inverse, const float* route_weight, float* dst,
    int tokens, int top_k, int hidden,
    const std::vector<sycl::event>& deps = {});

// Device-only vLLM-style MoE routing: count rows per expert, build the
// expert-major permutation, and duplicate BF16 activations without a host
// synchronization. Ornith's native fast path is fixed at top-k 8.
void launch_moe_remap_bf16_top8(sycl::queue& q, const sycl_bf16* hidden,
    const int32_t* topk_ids, sycl_bf16* remapped, int32_t* rows_per_expert,
    int32_t* expert_offsets, int32_t* unpermuted_to_permuted,
    int tokens, int hidden_size,
    int num_experts);

sycl::event launch_deltanet_step(sycl::queue& q, const DeltaNetParams& p,
                                 const std::vector<sycl::event>& deps = {});
sycl::event launch_causal_conv1d(sycl::queue& q, const ConvParams& p,
                                 const std::vector<sycl::event>& deps = {});
sycl::event launch_causal_conv1d_l2norm(sycl::queue& q, const ConvParams& p,
                                        int norm_heads, int head_dim,
                                        const std::vector<sycl::event>& deps = {});

sycl::event launch_flash_decode(sycl::queue& q, const AttnParams& p,
                                const std::vector<sycl::event>& deps = {});
sycl::event launch_flash_merge(sycl::queue& q, const AttnParams& p,
                               const std::vector<sycl::event>& deps = {});
// Small speculative verify batch: one subgroup per query, with all queries in
// a work-group sharing each staged K/V tile. `base_seq_len` is the number of
// cache entries visible to row 0; row r sees base_seq_len+r entries.
sycl::event launch_flash_decode_batched(
    sycl::queue& q, const float* qv, const uint8_t* k_cache,
    const uint8_t* v_cache, float* out, int tokens, int base_seq_len,
    int num_heads, int num_kv_heads, int head_dim, int seq_cap,
    float softmax_scale, float* partials, float* part_m, float* part_l,
    int splits, const std::vector<sycl::event>& deps = {});

} // namespace b70
#endif
