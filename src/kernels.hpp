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
    const std::vector<sycl::event>& deps = {});
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
sycl::event launch_router_topk_bf16_batched(
    sycl::queue& q, const sycl_bf16* logits, int tokens, int n_experts,
    int top_k, int32_t* out_expert, float* out_weight, bool normalize,
    const std::vector<sycl::event>& deps = {});
sycl::event launch_qk_norm_rope_batched(
    sycl::queue& q, float* qv, float* kv, const bf16_t* qw, const bf16_t* kw,
    int tokens, int q_heads, int k_heads, int dim, int start_pos,
    float theta, float partial_factor, float eps,
    const std::vector<sycl::event>& deps = {}, float weight_offset = 1.0f);
sycl::event launch_kv_append_batched(
    sycl::queue& q, const float* k, const float* v, uint8_t* k_cache,
    uint8_t* v_cache, int tokens, int start_pos, int n_kv_heads,
    int head_dim, int seq_cap, const std::vector<sycl::event>& deps = {});
sycl::event launch_qk_norm_rope_f16_batched(
    sycl::queue& q, const float* q_src, const float* k_src, sycl::half* q_dst,
    sycl::half* k_dst, const bf16_t* q_weight, const bf16_t* k_weight,
    int tokens, int q_heads, int k_heads, int head_dim, int start_pos,
    float theta, float eps, const std::vector<sycl::event>& deps = {});
sycl::event launch_qkv_norm_rope_f16_fused(
    sycl::queue& q, const sycl::half* qkv_src, sycl::half* q_dst,
    sycl::half* k_dst, sycl::half* v_dst, const bf16_t* q_weight,
    const bf16_t* k_weight, int tokens, int q_heads, int k_heads,
    int head_dim, int start_pos, float theta, float eps,
    const std::vector<sycl::event>& deps = {});
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
