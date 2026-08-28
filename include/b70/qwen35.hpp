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
#include <vector>
#include <map>
#include <memory>
#include <cstring>

namespace b70 {

enum class LayerKind { LINEAR_ATTN, FULL_ATTN };

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

    std::vector<LayerKind> layer_types;

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
    TensorRef pre_ff_norm, post_ff_norm;    // Muse: sandwich feed-forward norms

    // --- FFN ------------------------------------------------------
    TensorRef router;                       // MoE only
    TensorRef sh_gate, sh_up, sh_down;      // shared expert / dense FFN
    TensorRef sh_gate_w;                    // shared_expert_gate
    // experts[e] -> {packed, scale} for each projection
    std::vector<TensorRef> e_gate_p, e_gate_s;
    std::vector<TensorRef> e_up_p,   e_up_s;
    std::vector<TensorRef> e_down_p, e_down_s;
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
        return r.native ? int64_t(r.native->payload_bytes) :
               (r.ok() ? int64_t(r.t.end - r.t.begin) : 0);
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
