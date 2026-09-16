// =====================================================================
//  b70/qwen35.hpp  --  Qwen3.5 model description and weight resolution
//
//  Covers both variants in one description, since they differ only in
//  the FFN block:
//    Qwen3_5ForConditionalGeneration      dense  (27B, 64 layers)
//    Qwen3_5MoeForConditionalGeneration   MoE    (35B-A3B, 40 layers)
//
//  Layers alternate two attention mechanisms, given by config
//  `layer_types`. For the 35B-A3B that is 30 linear + 10 full:
//
//    linear_attention  Gated DeltaNet. Recurrent state of
//                      [key_head_dim x value_head_dim] per value head,
//                      2 MB per layer, INDEPENDENT of context length.
//                      All 30 layers together are ~63 MB, where a
//                      conventional KV cache at 256k context would be
//                      tens of gigabytes.
//
//    full_attention    Standard GQA with a growing KV cache. Only 10
//                      layers, so the cache stays small.
//
//  Weight loading is ZERO-COPY for MXFP4. compressed-tensors emits
//  exactly the layout the kernels already read:
//      weight_packed  [N][K/2]   4-bit nibbles, row-major
//      weight_scale   [N][K/32]  E8M0 shared exponents
//  so shards are mmapped and DMA'd to VRAM with no conversion pass and
//  no second copy on disk.
// =====================================================================
#ifndef B70_QWEN35_HPP
#define B70_QWEN35_HPP

#include "safetensors.hpp"
#include "weights.hpp"
#include "native_model.hpp"
#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <cstring>
#include <cmath>

namespace b70 {

enum class LayerKind { LINEAR_ATTN, FULL_ATTN };

// Shared by offline conversion and upload validation: never pack these
// Qwen/Ornith weights. A runtime --proj option cannot override this policy.
bool keep_qwen_bf16(const std::string& name);

struct Qwen35Config {
    // core dims
    int hidden          = 0;
    int n_layers        = 0;
    int vocab           = 0;
    int n_heads         = 0;      // full-attention query heads
    int n_kv_heads      = 0;
    int head_dim        = 0;
    float rms_eps       = 1e-6f;
    float post_norm_eps = 1e-6f;
    float rope_theta    = 1e7f;
    float partial_rope  = 1.0f;   // fraction of head_dim that gets RoPE
    bool  attn_out_gate = false;
    bool  is_muse       = false;   // Muse Glimmer: sandwich norms, scaleless qk/embed norm
    // ---- Agnes ------------------------------------------------------
    // Agnes-3.0-Flash (model_type "agnes"): a Qwen3.5-shaped hybrid --
    // gated DeltaNet 3:1 against gated full attention, head_dim 256,
    // partial RoPE 0.25, one MTP head -- with a vision tower and ONE
    // architectural addition this engine does not have: a second, narrow
    // SwiGLU per layer (parallel_ffn_inter wide) whose output is summed
    // with the main MLP's.  Its checkpoint ships custom modeling code
    // (auto_map -> modeling_agnes.py), so it is not any stock
    // architecture and nothing in vLLM's registry covers it.
    bool  is_agnes      = false;
    int   parallel_ffn_inter = 0;  // parallel_ffn_intermediate_size, 0 == none
    bool  mrope         = false;   // rope_parameters.mrope_section present
    std::vector<int> mrope_section;
    float query_prescale = 1.0f;   // Muse scale_query_by (post qk-norm)
    bool  tie_embeddings= false;

    // gated deltanet
    int lin_k_heads     = 0;
    int lin_v_heads     = 0;
    int lin_k_dim       = 0;
    int lin_v_dim       = 0;
    int conv_kernel     = 0;

    // MoE (0 == dense)
    int n_experts       = 0;
    int top_k           = 0;
    int moe_inter       = 0;
    int shared_inter    = 0;
    int dense_inter     = 0;      // dense variant FFN width

    // ---- K2-Horizon ------------------------------------------------
    // Qwen3-MoE geometry with grouped norms, a softplus attention gate,
    // a sigmoid router whose bias steers selection only, and MoVA: the
    // value projection replaced by a routed bank of experts.
    bool  is_k2           = false;
    int   norm_groups     = 1;      // layernorm_num_groups (1 == plain RMSNorm)
    int   mova_experts    = 0;      // mova_num_experts (0 == no MoVA)
    int   mova_top_k      = 0;      // mova_num_experts_per_tok
    bool  moe_gate_bias   = false;  // router bias present
    bool  router_sigmoid  = false;  // router_score_func == "sigmoid"
    bool  norm_topk_prob  = true;
    float router_scale    = 1.0f;   // router_scaling_factor
    int   rope_head_dim   = 0;      // 0 == same as head_dim (no split rope)
    int   sparse_step     = 1;      // decoder_sparse_step
    int   n_shared_expert = 0;      // num_shared_experts
    int   attn_gate       = 0;      // 0 none, 1 silu, 2 softplus
    bool  query_key_norm  = false;
    std::vector<int>  mlp_only_layers;
    std::vector<bool> k2_sparse;    // per layer: MoVA attention + routed MoE

    // ---- Gemma-4 ----------------------------------------------------
    // google/gemma-4-31B-it (model_type "gemma4"/"gemma4_text").  A dense
    // Gemma-sandwich transformer, but with per-LAYER-TYPE geometry that
    // nothing else here has: sliding layers are head_dim 256 over 16 KV
    // heads, full-attention layers head_dim 512 over 4.  See
    // GEMMA4-2026-09-14.md and ref/gemma4.py -- every item below is
    // silent if implemented wrongly.
    // ---- Qwen4-Exp / Qwen3.8-Flash-Next -----------------------------
    // model_type "qwen4_exp" / "qwen4_exp_text".  Its text config INHERITS
    // Qwen3NextConfig (ref/qwen4_exp_config.py:34), so the Gated-DeltaNet +
    // MoE hybrid underneath is the shape this engine already runs.  Three
    // mechanisms are layered on top and NONE of them exists here yet:
    //
    //   QSA   qwen_sparse_attention layers.  An MQA "indexer" mean-pools
    //         keys into blocks, scores sum_h relu(q.k)/sqrt(d), keeps the
    //         top blocks, and attention runs over THAT GATHERED LIST.
    //         ref/qwen4_exp_nvidia_indexer_qsa.py, .../qsa.py
    //   PLE   a per-layer n-gram embedding: a 20M-entry bigram/trigram
    //         table read at the ple_layer_ids layers, through a short
    //         conv.  ref/qwen4_exp_nvidia_ngram_embedding.py, ple_layer.py
    //   HC    HyperConnections: the residual is hc_count STREAMS wide with
    //         a low-rank mix, so the layer's mix/combine is not this
    //         engine's single-stream residual at all.
    //         ref/qwen4_exp_common_hyperconnection.py
    //
    // Parsed so the loader can SAY what the file is (rule 10) and
    // unsupported_reason() can refuse it by name.  Nothing consumes these
    // yet -- adding a field without a path that uses it is exactly the
    // silent-wrong-output trap rule 10 warns about, so the refusal is
    // what keeps them honest.
    bool  is_qwen4_exp    = false;
    int   hc_count        = 0;      // HyperConnection streams (ref default 4)
    int   hc_lowrank      = 0;      // its low-rank width (ref default 320)
    int   ngram_size      = 0;      // 3 == bigrams+trigrams
    int   heads_per_ngram = 0;
    int   ple_embed_dim   = 0;
    int   ple_conv_kernel = 0;
    int64_t ngram_vocab_base = 0;   // 20,000,000 in the reference
    std::vector<int> ple_layer_ids; // 1-BASED in the config (ref:127)
    // QSA indexer.  All five are required together once any is present,
    // indexer_kv_heads must be 1 (the MQA operators assume it), and
    // indexer_budget / indexer_compress_ratio must be 512 or 2048
    // (ref/qwen4_exp_config.py:_validate_qsa_config).
    int   indexer_n_heads = 0;
    int   indexer_kv_heads = 0;
    int   indexer_head_dim = 0;
    int   indexer_budget  = 0;
    int   indexer_compress_ratio = 0;
    std::vector<bool> qsa_attention;  // per layer: is this a QSA layer

    bool  is_gemma4       = false;
    int   global_head_dim = 0;      // full-attention head_dim (0 == same)
    int   n_global_kv_heads = 0;    // full-attention KV heads (0 == same)
    // Gemma-4 keys RoPE by layer type: sliding uses plain RoPE at 1e4,
    // full uses "proportional" RoPE at 1e6 with partial_rotary_factor
    // 0.25.  `rope_theta` above stays the SLIDING value; these two carry
    // the full-attention pair.
    float global_rope_theta = 0.0f;     // 0 == no per-type override
    float global_partial_rope = 1.0f;
    // proportional RoPE divides every inverse frequency by this
    // (ref/gemma4_proportional_rope.py, last line).  1.0 is the identity
    // and is what the 31B checkpoint uses.
    float global_rope_factor = 1.0f;
    bool  global_rope_proportional = false;
    // hidden_states *= layer_scalar as the LAST act of each decoder
    // layer, after the residual add.  Per layer, from the checkpoint.
    bool  layer_scalar    = false;
    // GeGLU instead of SwiGLU: gelu(gate) * up, tanh approximation.
    bool  geglu           = false;
    // attention_k_eq_v: full-attention layers have NO v_proj; V is the
    // PRE-norm, PRE-RoPE k_proj output through a scaleless RMSNorm.
    bool  k_eq_v          = false;
    float logit_softcap   = 0.0f;   // final_logit_softcapping (0 == none)
    float attn_scale      = 0.0f;   // 0 == the usual 1/sqrt(head_dim)
    float embed_scale     = 1.0f;   // embeddings multiplied on lookup
    int   sliding_window  = 0;

    // head_dim / KV heads for one layer, which differ per layer type on
    // Gemma-4 and are uniform everywhere else.  Never read cfg.head_dim
    // directly in a per-layer context.
    int layer_head_dim(int i) const {
        return (global_head_dim > 0 && layer_global(i)) ? global_head_dim
                                                        : head_dim;
    }
    int layer_kv_heads(int i) const {
        return (n_global_kv_heads > 0 && layer_global(i)) ? n_global_kv_heads
                                                          : n_kv_heads;
    }
    // The widest of either, for anything sized once for all layers.
    int max_head_dim() const { return std::max(head_dim, global_head_dim); }
    int max_kv_heads() const { return std::max(n_kv_heads, n_global_kv_heads); }

    // True when layer i is a FULL-attention (global) layer on a model that
    // gives the two layer types different parameters.  Every other
    // architecture answers false for every layer, so each accessor below
    // collapses to the model-wide value it always returned.
    bool layer_global(int i) const {
        return i >= 0 && i < int(layer_types.size()) &&
               !muse_sliding_attention.empty() &&
               !muse_sliding_attention[size_t(i)];
    }
    // RoPE base, rotated fraction and flavour for one layer.  Gemma-4
    // keys rope_parameters by layer type: sliding layers get plain RoPE
    // at 1e4 over the whole head, full-attention layers "proportional"
    // RoPE at 1e6 over a quarter of it.  Picking the wrong one leaves the
    // model fluent and wrong, which is why no forward path may read
    // cfg.rope_theta / cfg.partial_rope in a per-layer context.
    float layer_rope_theta(int i) const {
        return (global_rope_theta > 0.0f && layer_global(i)) ? global_rope_theta
                                                            : rope_theta;
    }
    float layer_partial_rope(int i) const {
        return (global_rope_theta > 0.0f && layer_global(i)) ? global_partial_rope
                                                            : partial_rope;
    }
    bool layer_rope_proportional(int i) const {
        return global_rope_proportional && layer_global(i);
    }
    // Frequency divisor for one layer.  Only proportional RoPE has one.
    float layer_rope_factor(int i) const {
        return layer_rope_proportional(i) ? global_rope_factor : 1.0f;
    }
    // Attention softmax scale for a layer of this head_dim.  Gemma-4 sets
    // `self.scaling = 1.0` flat (ref/gemma4.py) instead of the usual
    // 1/sqrt(head_dim); attn_scale is 0 for every other architecture,
    // which keeps this exactly the expression it replaces.
    float attn_softmax_scale(int hd) const {
        return attn_scale > 0.0f ? attn_scale
                                 : 1.0f / std::sqrt(float(hd));
    }

    std::vector<LayerKind> layer_types;
    // Muse uses sliding RoPE attention for three layers followed by one
    // full NoPE layer. Keep the checkpoint distinction; both map to the
    // generic FULL_ATTN execution kind elsewhere.
    std::vector<bool> muse_sliding_attention;

    bool is_moe() const { return n_experts > 0; }

    // Bytes of recurrent state per linear-attention layer.
    int64_t deltanet_state_bytes() const {
        return int64_t(lin_v_heads) * lin_k_dim * lin_v_dim * 4;
    }
};

// A resolved tensor: which shard holds it, and where.
struct TensorRef {
    int      shard = -1;
    const NativeTensorRecord* native = nullptr;
    uint64_t native_payload_offset = 0;
    uint64_t native_scale_offset = 0;
    STTensor t;
    // GPTQ/AutoRound logical matrix. `t` identifies qweight but its shape
    // is rewritten to logical [out,in]; the auxiliary tensors retain their
    // physical safetensors descriptors.
    bool     gptq = false;
    // compressed-tensors symmetric INT4: physical I32 [N,K/8] payload plus
    // BF16 [N,K/group] scales. Unlike GPTQ this is already output-major and
    // can be passed directly to oneDNN after exposing the logical [N,K].
    bool     compressed_int4 = false;
    bool     row_scaled = false; // FP8 payload with separate [N,1] scale
    int      qzeros_shard = -1, scales_shard = -1;
    STTensor qzeros_t, scales_t;
    int      gptq_group = 0;
    bool     ok() const { return shard >= 0 || native != nullptr; }
};

// One transformer layer's weights, as references into the mmapped
// shards. Nothing is copied at this stage.
struct Qwen35Layer {
    LayerKind kind = LayerKind::LINEAR_ATTN;

    TensorRef input_norm, post_attn_norm;

    // --- gated deltanet -------------------------------------------
    TensorRef la_in_qkv, la_in_z, la_in_a, la_in_b;
    TensorRef la_conv1d, la_A_log, la_dt_bias, la_norm, la_out;

    // --- full attention -------------------------------------------
    TensorRef q_proj, k_proj, v_proj, o_proj, q_norm, k_norm;
    TensorRef attn_gate;                    // Muse: self_attn.gate_proj (output gate)
    TensorRef pre_ff_norm, post_ff_norm;    // Muse/Gemma-4: sandwich FFN norms
    TensorRef layer_scalar;                 // Gemma-4: per-layer residual scale

    // --- FFN ------------------------------------------------------
    TensorRef router;                       // MoE only
    TensorRef sh_gate, sh_up, sh_down;      // shared expert / dense FFN
    // Agnes: mlp.parallel_ffn.{gate,up,down}_proj -- a second, narrower
    // SwiGLU inside the same mlp, summed with the main one.  Empty on
    // every other architecture.
    TensorRef pf_gate, pf_up, pf_down;
    TensorRef sh_gate_w;                    // shared_expert_gate
    // experts[e] -> {packed, scale} for each projection
    std::vector<TensorRef> e_gate_p, e_gate_s;
    std::vector<TensorRef> e_up_p,   e_up_s;
    std::vector<TensorRef> e_down_p, e_down_s;

    // --- K2-Horizon ------------------------------------------------
    // MoVA: on a sparse layer v_proj does not exist.  The value is a
    // routed mixture over v_experts[], each [kv_heads*head_dim, hidden],
    // selected by v_router and put through SiLU before the router weight
    // scales it.  Dense layers keep an ordinary v_proj.
    bool k2_sparse = false;
    TensorRef v_router, v_router_bias;
    std::vector<TensorRef> v_experts;
    TensorRef router_bias;                  // mlp.gate.bias (moe_gate_bias)
};

struct Qwen35Model {
    Qwen35Config cfg;
    std::string  dir;
    std::string  prefix;      // "model.language_model." or "model."
    Fmt          expert_fmt = Fmt::MXFP4;

    std::vector<std::unique_ptr<SafeTensors>> shards;
    std::unique_ptr<NativeModel> native_model;
    std::map<std::string, TensorRef>          index;

    TensorRef embed, final_norm, lm_head;
    std::vector<Qwen35Layer> layers;

    // Open every shard, parse config, resolve all tensor names.
    bool load(const std::string& dir, std::string& err, bool skip_vision = true,
              bool index_only = false);
    bool native_view(const TensorRef& r, QuantWeight& out, std::string& err) const;
    bool read_native_f32(const TensorRef& r, float* dst, std::string& err) const;

    const void* data(const TensorRef& r) const {
        if(r.native)return static_cast<const uint8_t*>(native_model->payload(*r.native))+
                           r.native_payload_offset;
        return r.ok() ? shards[r.shard]->data(r.t) : nullptr;
    }
    // Raw byte read that never dereferences the mapping.
    bool read_raw(const TensorRef& r, void* dst, std::string& err) const {
        if (!r.ok()) { err = "tensor not resolved"; return false; }
        if (r.native) {
            if (r.native->encoding != uint32_t(NativeEncoding::RAW)) {
                err = "packed native tensor requested as raw"; return false;
            }
            const size_t bytes = size_t(r.t.end-r.t.begin);
            if (r.t.end < r.t.begin || r.native_payload_offset > r.native->payload_bytes ||
                bytes > r.native->payload_bytes-r.native_payload_offset) {
                err = "native raw slice out of bounds"; return false;
            }
            std::memcpy(dst, static_cast<const uint8_t*>(native_model->payload(*r.native))+
                        r.native_payload_offset, bytes);
            return true;
        }
        if (r.shard < 0 || size_t(r.shard) >= shards.size()) {
            err = "shard index out of range"; return false;
        }
        return shards[r.shard]->read_raw(r.t, dst, err);
    }

    size_t shard_count() const { return shards.size(); }

    // Call once the header index is built and all reads use read_raw.
    void unmap_all() { for (auto& s : shards) if (s) s->unmap(); }

    int64_t bytes(const TensorRef& r) const {
        return r.ok() ? int64_t(r.t.end - r.t.begin) : 0;
    }

    // Build a QuantWeight over an already-packed MXFP4 tensor pair.
    // No conversion: the pointers go straight to the kernel.
    QuantWeight quant_view(const TensorRef& packed, const TensorRef& scale,
                           int N, int K) const;

    int64_t total_bytes(bool include_experts = true) const;
    void    summary() const;
};

} // namespace b70
#endif
