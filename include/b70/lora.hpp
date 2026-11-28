// b70/lora.hpp -- a PEFT LoRA adapter (adapter_config.json + adapter_model.safetensors)
// for the dense non-MoE checkpoint path. The adapter's A/B are read (F32, converted to bf16 at
// load), validated against the base geometry, packed in the dense bf16 layout and held as one
// contiguous host blob whose per-(layer, module) views the engine hands to the delta kernels.
//
// Placement (single card, no sharding): the column / row split with C = 1.
//   column-parallel (q/k/v, gate/up): A [r16][k] replicated, B [n][r16]; the delta lands in the
//     module's own output columns (out_off 0, or `inter` for the up half of the fused gate|up
//     row). No cross-card traffic.
//   row-parallel (o, down): A [r16][k], B [n][r16]; the delta lands in the full output row.
// r is padded to r16 = ceil(r/16) * 16 (a multiple of 16, like the engine's dense bodies);
// scale = lora_alpha / r. The delta is formed in fp32, rounded to bf16 once (the mid), and
// B * bf16(mid) is accumulated into the caller's fp32 buffer -- the same rounding points the
// engine's own activations use.
//
// This header is host-safe: it is compiled by the host-only test suite and by the AOT build
// alike. The device-side upload and the delta kernels live in include/b70/lora_dev.hpp, which
// pulls in SYCL and is not part of the host gate.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "b70/formats.hpp"

namespace b70::lora {

enum class Mod { Q, K, V, O, Gate, Up, Down };

// The base model's geometry the adapter is validated against and packed against.
// Per-layer head_dim / kv_heads / full_attention are indexed by layer -- the gemma-4-style
// variation in shape reaches the module widths without a model-wide average.
struct LoraGeom {
    int layers = 0;
    int hidden = 0;
    int q_heads = 0;
    bool attn_output_gate = false;
    int inter = 0;           // dense intermediate size (both gate and up rows)
    int head_dim = 0;        // model-wide; the per-layer value overrides when set
    int kv_heads = 0;        // model-wide; the per-layer value overrides when set
    std::string prefix = "model.language_model.";  // the base module path prefix
    // Per-layer overrides (indexed 0..layers-1; empty means the model-wide value).
    std::vector<int>  layer_head_dim;
    std::vector<int>  layer_kv_heads;
    std::vector<bool> full_attention;

    int hd(int i) const {
        if (i >= 0 && i < int(layer_head_dim.size()) && layer_head_dim[size_t(i)] > 0)
            return layer_head_dim[size_t(i)];
        return head_dim;
    }
    int kv(int i) const {
        if (i >= 0 && i < int(layer_kv_heads.size()) && layer_kv_heads[size_t(i)] > 0)
            return layer_kv_heads[size_t(i)];
        return kv_heads;
    }
    bool full_attn(int i) const {
        if (i >= 0 && i < int(full_attention.size())) return full_attention[size_t(i)];
        return true;
    }
};

// One (layer, module) LoRA delta: the packed A [r16][k] and B [n][r16] (bf16, row-major,
// pointers into the adapter's blob), the module widths and the offset of B's rows inside the
// target buffer.
struct LoraModule {
    const uint16_t* a = nullptr;
    const uint16_t* b = nullptr;
    int r16 = 0;
    int k = 0;
    int n = 0;
    int out_off = 0;     // column offset of B's rows within the target buffer
    float scale = 1.f;
    bool present() const { return a && b && r16 > 0 && k > 0 && n > 0; }
};

// One layer's LoRA deltas. A module the adapter does not carry keeps nulls; the engine's
// delta launches skip those.
struct LoraLayer {
    LoraModule q, k, v, o;      // GQA: column-parallel q/k/v, row-parallel o
    LoraModule gate, up, down;  // MLP: column-parallel gate/up, row-parallel down
    bool any() const {
        return q.present() || k.present() || v.present() || o.present() ||
               gate.present() || up.present() || down.present();
    }
};

// The loaded adapter: the per-layer bundles and the packed blob (host memory) every layer's
// views point into. Throwing ctor -- any failure throws std::runtime_error with a message
// that names the offending file/tensor/layer.
class LoraAdapter {
public:
    explicit LoraAdapter(const std::string& adapter_dir, const LoraGeom& geom);
    ~LoraAdapter() = default;
    LoraAdapter(const LoraAdapter&) = delete;
    LoraAdapter& operator=(const LoraAdapter&) = delete;

    const LoraLayer& layer(int i) const { return layers_.at(size_t(i)); }
    int layers() const { return int(layers_.size()); }
    int r() const { return r_; }
    int r16() const { return r16_; }
    int n_max() const { return n_max_; }   // the largest module output (sizes the scratch)
    float scale() const { return scale_; }
    const std::string& name() const { return name_; }

    // The packed blob: every module's A and B concatenated, row-major bf16, in
    // per-layer-per-module order. Layer views point into it, so the pointers are stable.
    const uint8_t* blob() const { return blob_.data(); }
    size_t blob_size() const { return blob_.size(); }

private:
    std::string name_;                  // base_model_name_or_path
    int r_ = 0;                          // the adapter's rank
    int r16_ = 0;                       // the rank padded to a multiple of 16
    int n_max_ = 0;                     // the largest module output on the card
    float scale_ = 1.f;                 // lora_alpha / r
    std::vector<uint8_t> blob_;
    std::vector<LoraLayer> layers_;
};

// The host reference: out[r*out_ld + out_off + c] += scale * (B . bf16(A . x)) for r < m.
// The mid is rounded to bf16 once, as the engine's activations; the delta accumulates in
// fp32. x is fp32, row-major [m][k]. out is fp32, row-major [m][out_ld]. A reference for
// the host test only -- the device kernels implement the same rounding points.
void lora_reference_delta(const LoraModule& lm, const float* x, float* out, int m, int out_ld);

// One bf16 fact the loader and the reference share: round-to-nearest-even fp32 -> bf16
// (formats.hpp's f32_to_bf16), exposed as bits.
inline uint16_t lora_f32_to_bf16(float f) { return f32_to_bf16(f).bits; }

}  // namespace b70::lora
