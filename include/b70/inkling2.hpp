// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "b70/inkling2_expert.hpp"
#include "b70/safetensors.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace b70::inkling2 {

// Native HF Mini layout (model.llm.*), not a Qwen alias or a GGUF converter.
struct Attention {
    int heads=0, kv_heads=0, head_dim=0;
};
struct Visit {
    int physical, cache_slot, pass;
    bool sliding, block_boundary, has_message;
    bool operator==(const Visit&) const = default;
};
struct Config {
    int hidden=0, layers=0, vocab=0, output_vocab=0, eos=0, max_context=0;
    int inter=0, experts=0, top_k=0, rel_dim=0, rel_extent=0, window=0, conv=0;
    int residual_block=0, loop_start=0, loop_end=0, passes=0, gate_rank=0, log_floor=0;
    float eps=0, qk_clip=0, rel_clip=0, log_alpha=0, route_scale=0;
    float gate_cap=0, up_cap=0, logit_cap=0;
    Attention full, local;
    std::vector<bool> sliding;
    int cache_slots() const { return layers+(passes-1)*(loop_end-loop_start); }
    std::vector<Visit> visits() const;
};

// Initial port supports fixed-pass text, not variable loops, vision, or MTP.
Config parse_config(std::string_view json);
using TensorMap=std::map<std::string,STTensor>;
struct LayerLayout {
    PackedFormat gate_up, down;
};
std::vector<LayerLayout> validate_tensors(const Config& c,const TensorMap& tensors);
// Parse a genuine safetensors JSON header, not an index. Does not read weight data.
TensorMap read_tensor_header(std::string_view header);

// Exact native weight layout for oracle and future GPU uploads: W13 rows
// alternate gate/up. NVFP4 has a multiplier per expert; MXFP8 uses E8M0 per 32.
struct PackedRows {
    PackedFormat fmt;
    int experts, rows, cols;
    const uint8_t* data;
    const uint8_t* scale;
    const float* amax=nullptr;
    float denominator=0.f;
    void row(int expert,int out,std::span<float> values) const;
};

struct Route { int expert; float weight; };
// BF16 weights/activations are widened for the host oracle. Softcapping and
// routing itself remain FP32. Bias selects experts, but never weights them.
std::vector<Route> route(std::span<const float> logits,std::span<const float> bias,int k,float scale);
float swiglu(float gate,float up,float gate_cap,float up_cap);

struct Checkpoint {
    Config config;
    std::vector<LayerLayout> layout;
    std::shared_ptr<const SafeTensors> file;
    static Checkpoint open(const std::string& dir);
    // Exposes exactly one expert's source bytes; valid while the shard is mapped.
    ExpertMatrix expert_matrix(int layer, int expert, bool down) const;
};

// Slow native text oracle for parity work. One sequence, contiguous token IDs,
// no sampling, parallelism, images, draft head or attempt at serving throughput.
// Production dispatcher must not silently fall back to this implementation.
class TextReference {
public:
    TextReference(Checkpoint weights,int context);
    ~TextReference();
    TextReference(TextReference&&) noexcept;
    TextReference& operator=(TextReference&&) noexcept;
    void reset();
    int position() const;
    const Config& config() const;
    // Process token at current position and return greedy next ID. If supplied,
    // logits must have output_vocab slots (padded vocab is never sampled).
    int step(int token,std::span<float> logits={});
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace b70::inkling2
