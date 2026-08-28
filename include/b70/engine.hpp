// =====================================================================
//  b70/engine.hpp  --  the forward pass
//
//  Everything below this line is assembly: the kernels are written and
//  measured, and this sequences them. The design decisions that matter
//  are about residency and launch count, not arithmetic.
//
//  RESIDENCY
//  ---------
//  Weights are uploaded once at load and never touched again. MXFP4
//  experts go to VRAM byte-for-byte from the mmapped shard -- no
//  dequantize/requantize pass, no staging copy. Activations live in a
//  small fixed pool of device scratch that is reused every token.
//
//  LAUNCH BUDGET per decoded token, 40 layers:
//     norms + residual        80   (fused, one per norm)
//     linear attn (30 layers) 30 x 6 = 180
//     full attn   (10 layers) 10 x 7 = 70
//     MoE                     80   (fused: 2 per layer)
//     shared expert           120
//     lm_head + argmax        2
//                            ----
//                            ~530 launches, ~2.7 ms of pure launch
//                            latency at 5 us.
//
//  That is the same order as the memory time, which is why the fused
//  expert path mattered so much: it removed 880 launches per token.
//  Reducing the rest further needs command-list batching, which is the
//  next structural win after this runs.
// =====================================================================
#ifndef B70_ENGINE_HPP
#define B70_ENGINE_HPP

#include "qwen35.hpp"
#include "moe.hpp"

namespace b70 {

// Device-resident weights for one layer. Raw pointers into VRAM.
struct DevLayer {
    LayerKind kind = LayerKind::LINEAR_ATTN;

    const bf16_t* input_norm     = nullptr;
    const bf16_t* post_attn_norm = nullptr;

    // gated deltanet
    const bf16_t* la_qkv = nullptr;   // [2*Hk*Dk + Hv*Dv][H]
    const bf16_t* la_z   = nullptr;   // [Hv*Dv][H]
    const bf16_t* la_a   = nullptr;   // [Hv][H]
    const bf16_t* la_b   = nullptr;   // [Hv][H]
    const bf16_t* la_conv= nullptr;   // [channels][K]
    const bf16_t* la_Alog= nullptr;   // [Hv]
    const bf16_t* la_dtb = nullptr;   // [Hv]
    const bf16_t* la_norm= nullptr;   // [Dv]
    const bf16_t* la_out = nullptr;   // [H][Hv*Dv]

    // full attention
    const bf16_t* q_proj = nullptr;
    const bf16_t* k_proj = nullptr;
    const bf16_t* v_proj = nullptr;
    const bf16_t* o_proj = nullptr;

    // FFN
    const float*  router  = nullptr;
    MoeLayer      moe;                 // packed experts, expert-major
    const bf16_t* sh_gate = nullptr;
    const bf16_t* sh_up   = nullptr;
    const bf16_t* sh_down = nullptr;
    const bf16_t* sh_gate_w = nullptr;

    // per-layer recurrent / cache state
    float* dn_state = nullptr;         // [Hv][Dv][Dk], context-independent
    float* conv_ring= nullptr;         // [channels][K-1]
    uint8_t* k_cache  = nullptr;       // FP8 E4M3, full-attn only, D-major
    uint8_t* v_cache  = nullptr;
};

// Scratch reused every token. Sized once; never reallocated.
struct Scratch {
    float* h        = nullptr;   // hidden state    [H]
    float* h2       = nullptr;   // normed copy     [H]
    float* resid    = nullptr;   // residual        [H]
    float* qkv      = nullptr;   // deltanet qkv    [channels]
    float* zbuf     = nullptr;
    float* abuf     = nullptr;
    float* bbuf     = nullptr;
    float* attn_out = nullptr;
    float* moe_h    = nullptr;   // [top_k][I]
    float* moe_y    = nullptr;   // [H]
    float* logits   = nullptr;   // [vocab]
    float* rlogits  = nullptr;   // [n_experts]
    int32_t* d_expert = nullptr; // [top_k]
    float*   d_weight = nullptr; // [top_k]
    int32_t* d_tok    = nullptr;
    float*   d_val    = nullptr;
    // split-K attention workspace
    float* part = nullptr;
    float* pm   = nullptr;
    float* pl   = nullptr;
    // deltanet gates and shared-expert intermediates
    float* alpha        = nullptr;   // [n_v_heads]
    float* beta         = nullptr;   // [n_v_heads]
    float* sh_g         = nullptr;   // [shared_inter]
    float* sh_u         = nullptr;   // [shared_inter]
    float* sh_out       = nullptr;   // [hidden]
    float* sh_gate_val  = nullptr;   // [1]
    int32_t* d_pos      = nullptr;   // position, device-resident for graphs
    int32_t* d_seq_len  = nullptr;   // pos+1, likewise
    float* qsplit       = nullptr;   // queries after removing the gate
    float* gsplit       = nullptr;   // the attention output gate
};

struct Engine {
    Qwen35Config cfg;
    std::vector<DevLayer> layers;
    Scratch  s;

    const bf16_t* embed   = nullptr;
    const bf16_t* fnorm   = nullptr;
    QuantWeight   lm_head;            // quantized at load: see note below
    bool          lm_head_quant = false;

    int pos = 0;                      // current sequence position

    // Reset the recurrent state and KV cache for a new sequence. The
    // DeltaNet state MUST be zeroed between sequences -- it carries the
    // entire history, so a stale state silently conditions the next
    // conversation on the previous one.
    void reset();
};

// ---------------------------------------------------------------------
// Upload a resolved checkpoint to VRAM.
//
// lm_head is [248320][2048] -- 1.02 GB in bf16, read EVERY token. At
// 542 GB/s that is 1.88 ms, larger than the entire fused MoE block
// (1.58 ms). Quantizing it to INT4 at load time costs a few seconds of
// CPU and takes it to 0.56 ms. It is the single highest-value change in
// the whole decode path, and it is a load-time decision rather than a
// kernel.
struct UploadOptions {
    Fmt  lm_head_fmt   = Fmt::INT4;   // Fmt::BF16 to keep it unquantized
    bool quantize_lm_head = true;
    int  max_seq       = 8192;        // KV cache capacity, full-attn layers
};

} // namespace b70
#endif
