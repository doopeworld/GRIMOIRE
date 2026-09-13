// =====================================================================
//  mini_model.hpp -- synthetic checkpoints for the device test tools.
//
//  Every architecture GRIMOIRE claims to support, written to disk as a
//  real config.json + model.safetensors with RANDOM BF16 weights, small
//  enough to load in under a second and big enough that every shape rule
//  (head counts, group divisibility, expert counts) is exercised.
//
//  Random, never zero: a zeroed checkpoint makes a dead code path
//  indistinguishable from a live one, which is exactly the failure these
//  tools exist to catch.
//
//  These are NOT models.  Output from them is noise by construction.
//  What they establish is that a path RUNS, stays finite, stays in
//  vocabulary and is reproducible -- and, for the parallel tests, that
//  two processes produce the same tokens as one.  Rule 8 is untouched:
//  no number from here means anything.
// =====================================================================
#ifndef B70_MINI_MODEL_HPP
#define B70_MINI_MODEL_HPP

#include "b70/formats.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace b70 {
namespace mini {

namespace fs = std::filesystem;

// `fill`, when non-empty, gives the tensor its exact values instead of
// random ones.  Needed for index tables (a DFlash d2t map) where random
// noise is not a weaker test, it is a different tensor.
struct Tn {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> fill = {};
};

// A named architecture and the tensors it needs.
struct Arch {
    const char* name;
    std::string config;
    std::vector<Tn> tensors;
    int vocab;
    int layers;
};

inline void write_model(const fs::path& dir, const Arch& a, uint32_t seed = 20260912) {
    fs::create_directories(dir);
    std::ofstream(dir/"config.json") << a.config;
    std::ostringstream h; h << '{';
    uint64_t off = 0;
    for (size_t i = 0; i < a.tensors.size(); ++i) {
        if (i) h << ',';
        size_t n = 1; for (auto d : a.tensors[i].shape) n *= size_t(d);
        h << '"' << a.tensors[i].name << "\":{\"dtype\":\"BF16\",\"shape\":[";
        for (size_t d = 0; d < a.tensors[i].shape.size(); ++d) {
            if (d) h << ',';
            h << a.tensors[i].shape[d];
        }
        h << "],\"data_offsets\":[" << off << ','; off += n*2; h << off << "]}";
    }
    h << '}';
    std::string hs = h.str(); while (hs.size() % 8) hs += ' ';

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<bf16_t> data(off/2);
    for (auto& v : data) v = f32_to_bf16(nd(rng));
    {   // exact values where a tensor asked for them
        size_t at = 0;
        for (const Tn& t : a.tensors) {
            size_t n = 1; for (auto d : t.shape) n *= size_t(d);
            for (size_t i = 0; i < t.fill.size() && i < n; ++i)
                data[at+i] = f32_to_bf16(t.fill[i]);
            at += n;
        }
    }

    std::ofstream f(dir/"model.safetensors", std::ios::binary);
    const uint64_t n = hs.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f << hs;
    f.write(reinterpret_cast<const char*>(data.data()),
            std::streamsize(data.size()*sizeof(bf16_t)));
}

// ---- K2-Horizon: MoVA + grouped norm + softplus gate + sigmoid router --
// Layers 0-1 dense, 2-3 sparse, so one model carries BOTH shapes.
inline Arch k2() {
    const int H=64, KV=32, Q=64, I=48, MI=32, E=4, MV=3, V=128, L=4;
    Arch a{"k2-horizon", R"JSON({
  "model_type": "k2_horizon", "attention_gate_func": "softplus",
  "hidden_size": 64, "num_hidden_layers": 4, "vocab_size": 128,
  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
  "intermediate_size": 48, "moe_intermediate_size": 32,
  "num_experts": 4, "num_experts_per_tok": 2, "num_shared_experts": 1,
  "mova_num_experts": 3, "mova_num_experts_per_tok": 2,
  "mlp_only_layers": [0, 1], "decoder_sparse_step": 1,
  "layernorm_num_groups": 2, "moe_gate_bias": true,
  "router_score_func": "sigmoid", "router_scaling_factor": 2.5,
  "norm_topk_prob": true, "query_key_norm": false, "rope_head_dim": 16,
  "rms_norm_eps": 1e-06, "tie_word_embeddings": false,
  "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}
})JSON", {}, V, L};
    auto& t = a.tensors;
    t = { {"model.embed_tokens.weight", {V,H}}, {"model.norm.weight", {H}},
          {"lm_head.weight", {V,H}} };
    for (int l = 0; l < L; ++l) {
        const std::string b = "model.layers." + std::to_string(l) + ".";
        const bool sparse = (l >= 2);
        t.push_back({b+"input_layernorm.weight", {H}});
        t.push_back({b+"post_attention_layernorm.weight", {H}});
        const std::string s = b + "self_attn.";
        t.push_back({s+"q_proj.weight", {Q,H}});
        t.push_back({s+"k_proj.weight", {KV,H}});
        t.push_back({s+"o_proj.weight", {H,Q}});
        t.push_back({s+"gate_proj.weight", {Q,H}});
        if (sparse) {
            t.push_back({s+"v_router.weight", {MV,H}});
            t.push_back({s+"v_router.bias", {MV}});
            for (int e = 0; e < MV; ++e)
                t.push_back({s+"v_experts."+std::to_string(e)+".weight", {KV,H}});
        } else {
            t.push_back({s+"v_proj.weight", {KV,H}});
        }
        const std::string m = b + "mlp.";
        if (sparse) {
            t.push_back({m+"gate.weight", {E,H}});
            t.push_back({m+"gate.bias", {E}});
            for (int e = 0; e < E; ++e) {
                const std::string x = m+"experts."+std::to_string(e)+".";
                t.push_back({x+"gate_proj.weight", {MI,H}});
                t.push_back({x+"up_proj.weight",   {MI,H}});
                t.push_back({x+"down_proj.weight", {H,MI}});
            }
            for (const char* n : {"gate_proj","up_proj","down_proj"})
                t.push_back({m+"shared_experts."+std::string(n)+".weight",
                             std::string(n)=="down_proj" ? std::vector<int64_t>{H,MI}
                                                         : std::vector<int64_t>{MI,H}});
        } else {
            t.push_back({m+"gate_proj.weight", {I,H}});
            t.push_back({m+"up_proj.weight",   {I,H}});
            t.push_back({m+"down_proj.weight", {H,I}});
        }
    }
    return a;
}

// ---- MTP head -------------------------------------------------------
// One extra decoder layer plus the fc that mixes the previous hidden
// state with the next token's embedding.  Shapes are exactly what
// Grimoire::build checks; anything else is refused at load, which is the
// point -- a drafter whose geometry does not match its target is a
// silent accuracy loss, not a crash.
inline void add_mtp(std::vector<Tn>& t, int H, int Q, int KV, int HD,
                    int I, int n_experts, int moe_inter) {
    t.push_back({"mtp.fc.weight", {H, 2*H}});
    t.push_back({"mtp.pre_fc_norm_hidden.weight", {H}});
    t.push_back({"mtp.pre_fc_norm_embedding.weight", {H}});
    t.push_back({"mtp.norm.weight", {H}});
    const std::string b = "mtp.layers.0.";
    t.push_back({b+"input_layernorm.weight", {H}});
    t.push_back({b+"post_attention_layernorm.weight", {H}});
    const std::string a = b + "self_attn.";
    t.push_back({a+"q_proj.weight", {Q,H}});
    t.push_back({a+"k_proj.weight", {KV,H}});
    t.push_back({a+"v_proj.weight", {KV,H}});
    t.push_back({a+"o_proj.weight", {H,Q}});
    t.push_back({a+"q_norm.weight", {HD}});
    t.push_back({a+"k_norm.weight", {HD}});
    const std::string m = b + "mlp.";
    if (n_experts > 0) {
        t.push_back({m+"gate.weight", {n_experts,H}});
        for (int e = 0; e < n_experts; ++e) {
            const std::string x = m+"experts."+std::to_string(e)+".";
            t.push_back({x+"gate_proj.weight", {moe_inter,H}});
            t.push_back({x+"up_proj.weight",   {moe_inter,H}});
            t.push_back({x+"down_proj.weight", {H,moe_inter}});
        }
        // A routed MTP head carries its own shared expert.  Deliberately
        // NO shared_expert_gate: the engine documents that one as
        // optional, so leaving it out is the case worth covering.
        t.push_back({m+"shared_expert.gate_proj.weight", {moe_inter,H}});
        t.push_back({m+"shared_expert.up_proj.weight",   {moe_inter,H}});
        t.push_back({m+"shared_expert.down_proj.weight", {H,moe_inter}});
    } else {
        t.push_back({m+"gate_proj.weight", {I,H}});
        t.push_back({m+"up_proj.weight",   {I,H}});
        t.push_back({m+"down_proj.weight", {H,I}});
    }
}

// ---- plain dense transformer (Qwen3 shape, no MoE, no linear attn) ----
// This is the "any model" case: a stock architecture with none of
// GRIMOIRE's own features.  It is also the shape that exposed the
// Hv == 0 prefill bug.
inline Arch dense(int L = 4, bool mtp = false) {
    const int H=64, Q=64, KV=32, I=128, V=128;
    std::ostringstream c;
    c << R"JSON({
  "model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": )JSON" << L
      << R"JSON(, "vocab_size": 128,
  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
  "intermediate_size": 128, "rms_norm_eps": 1e-06,
  "tie_word_embeddings": false, "rope_theta": 1000000.0
})JSON";
    Arch a{mtp ? "dense+mtp" : "dense", c.str(), {}, V, L};
    auto& t = a.tensors;
    t = { {"model.embed_tokens.weight", {V,H}}, {"model.norm.weight", {H}},
          {"lm_head.weight", {V,H}} };
    for (int l = 0; l < L; ++l) {
        const std::string b = "model.layers." + std::to_string(l) + ".";
        t.push_back({b+"input_layernorm.weight", {H}});
        t.push_back({b+"post_attention_layernorm.weight", {H}});
        const std::string s = b + "self_attn.";
        t.push_back({s+"q_proj.weight", {Q,H}});
        t.push_back({s+"k_proj.weight", {KV,H}});
        t.push_back({s+"v_proj.weight", {KV,H}});
        t.push_back({s+"o_proj.weight", {H,Q}});
        t.push_back({s+"q_norm.weight", {16}});
        t.push_back({s+"k_norm.weight", {16}});
        const std::string m = b + "mlp.";
        t.push_back({m+"gate_proj.weight", {I,H}});
        t.push_back({m+"up_proj.weight",   {I,H}});
        t.push_back({m+"down_proj.weight", {H,I}});
    }
    if (mtp) add_mtp(t, H, Q, KV, 16, I, 0, 0);
    return a;
}

// ---- hybrid: DeltaNet linear attention + full attention ---------------
// This is the Qwen3.5 / Ornith shape, and it was the one architecture with
// no end-to-end coverage at all -- which matters, because the linear
// layers are where the recurrent state, the conv ring and the speculative
// ROLLBACK of both live.  Every other model in this file has Hv == 0 and
// exercises none of it.
//
// Layer 0 and 2 are linear_attention, 1 and 3 full attention, so one model
// carries both kinds and the boundary between them.
inline Arch hybrid(int L = 4, bool mtp = false) {
    const int H=64, Q=64, KV=32, I=128, V=128;
    const int LK=2, LV=2, DK=16, DV=16, CONV=4;      // linear-attn geometry
    const int QKV = 2*LK*DK + LV*DV;                 // 2*2*16 + 2*16 = 96
    std::ostringstream c;
    c << R"JSON({
  "model_type": "qwen3_next", "hidden_size": 64, "num_hidden_layers": )JSON" << L
      << R"JSON(, "vocab_size": 128,
  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
  "intermediate_size": 128, "rms_norm_eps": 1e-06,
  "tie_word_embeddings": false, "rope_theta": 1000000.0,
  "linear_num_key_heads": 2, "linear_num_value_heads": 2,
  "linear_key_head_dim": 16, "linear_value_head_dim": 16,
  "linear_conv_kernel_dim": 4,
  "layer_types": [)JSON";
    for (int l = 0; l < L; ++l)
        c << (l ? "," : "") << (l % 2 == 0 ? R"("linear_attention")"
                                           : R"("full_attention")");
    c << "]}";
    Arch a{mtp ? "hybrid+mtp" : "hybrid", c.str(), {}, V, L};
    auto& t = a.tensors;
    t = { {"model.embed_tokens.weight", {V,H}}, {"model.norm.weight", {H}},
          {"lm_head.weight", {V,H}} };
    for (int l = 0; l < L; ++l) {
        const std::string b = "model.layers." + std::to_string(l) + ".";
        t.push_back({b+"input_layernorm.weight", {H}});
        t.push_back({b+"post_attention_layernorm.weight", {H}});
        if (l % 2 == 0) {
            const std::string la = b + "linear_attn.";
            t.push_back({la+"in_proj_qkv.weight", {QKV,H}});
            t.push_back({la+"in_proj_z.weight",   {LV*DV,H}});
            t.push_back({la+"in_proj_a.weight",   {LV,H}});
            t.push_back({la+"in_proj_b.weight",   {LV,H}});
            t.push_back({la+"conv1d.weight",      {QKV,1,CONV}});
            t.push_back({la+"A_log",              {LV}});
            t.push_back({la+"dt_bias",            {LV}});
            t.push_back({la+"norm.weight",        {DV}});
            t.push_back({la+"out_proj.weight",    {H,LV*DV}});
        } else {
            const std::string sa = b + "self_attn.";
            t.push_back({sa+"q_proj.weight", {Q,H}});
            t.push_back({sa+"k_proj.weight", {KV,H}});
            t.push_back({sa+"v_proj.weight", {KV,H}});
            t.push_back({sa+"o_proj.weight", {H,Q}});
            t.push_back({sa+"q_norm.weight", {16}});
            t.push_back({sa+"k_norm.weight", {16}});
        }
        const std::string m = b + "mlp.";
        t.push_back({m+"gate_proj.weight", {I,H}});
        t.push_back({m+"up_proj.weight",   {I,H}});
        t.push_back({m+"down_proj.weight", {H,I}});
    }
    if (mtp) add_mtp(t, H, Q, KV, 16, I, 0, 0);
    return a;
}

// ---- routed MoE with a shared expert (Qwen3-MoE shape) ----------------
// Exercises the routed path, the shared expert and TP expert sharding.
inline Arch moe(int L = 4, bool mtp = false) {
    const int H=64, Q=64, KV=32, MI=32, SI=32, E=8, V=128;
    std::ostringstream c;
    c << R"JSON({
  "model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": )JSON" << L
      << R"JSON(, "vocab_size": 128,
  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
  "intermediate_size": 128, "moe_intermediate_size": 32,
  "shared_expert_intermediate_size": 32,
  "num_experts": 8, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
  "mlp_only_layers": [], "norm_topk_prob": true,
  "rms_norm_eps": 1e-06, "tie_word_embeddings": false, "rope_theta": 1000000.0
})JSON";
    Arch a{mtp ? "moe+mtp" : "moe", c.str(), {}, V, L};
    auto& t = a.tensors;
    t = { {"model.embed_tokens.weight", {V,H}}, {"model.norm.weight", {H}},
          {"lm_head.weight", {V,H}} };
    for (int l = 0; l < L; ++l) {
        const std::string b = "model.layers." + std::to_string(l) + ".";
        t.push_back({b+"input_layernorm.weight", {H}});
        t.push_back({b+"post_attention_layernorm.weight", {H}});
        const std::string s = b + "self_attn.";
        t.push_back({s+"q_proj.weight", {Q,H}});
        t.push_back({s+"k_proj.weight", {KV,H}});
        t.push_back({s+"v_proj.weight", {KV,H}});
        t.push_back({s+"o_proj.weight", {H,Q}});
        t.push_back({s+"q_norm.weight", {16}});
        t.push_back({s+"k_norm.weight", {16}});
        const std::string m = b + "mlp.";
        t.push_back({m+"gate.weight", {E,H}});
        for (int e = 0; e < E; ++e) {
            const std::string x = m+"experts."+std::to_string(e)+".";
            t.push_back({x+"gate_proj.weight", {MI,H}});
            t.push_back({x+"up_proj.weight",   {MI,H}});
            t.push_back({x+"down_proj.weight", {H,MI}});
        }
        t.push_back({m+"shared_expert.gate_proj.weight", {SI,H}});
        t.push_back({m+"shared_expert.up_proj.weight",   {SI,H}});
        t.push_back({m+"shared_expert.down_proj.weight", {H,SI}});
        t.push_back({m+"shared_expert_gate.weight",      {1,H}});
    }
    if (mtp) add_mtp(t, H, Q, KV, 16, 0, E, MI);
    return a;
}

// ---- a DFlash drafter for the dense target above -----------------------
// The drafter is a separate checkpoint in its own directory, selected with
// GRIMOIRE_DFLASH_MODEL.  Its shape has to agree with the TARGET it drafts
// for: same hidden width, and taps that name layers the target actually
// runs.  fc consumes n_taps * target_hidden and the engine feeds it exactly
// target_layers.size() * hidden, so those two must match or the context
// projection reads the wrong amount of memory.
//
// `own_head` gives the drafter its own reduced-vocabulary lm_head plus the
// d2t map that turns its ids back into target ids -- the layout every
// EAGLE3-derived drafter ships and the one that is silent when ignored:
// decoded through the TARGET's head instead, the draft still yields real
// token ids and the verifier simply rejects them.
//
// `forced_target`, when >= 0, collapses the draft head's whole vocabulary
// onto that one target id (d2t[i] = forced_target - i, so every row maps
// there).  Random weights make a drafter that agrees with a random target
// about one time in `vocab`, so every speculative test so far ran at 0%
// acceptance and never once exercised accept-then-continue in the real
// engine.  A drafter that always proposes a token the target actually
// emits does: it is accepted wherever the target repeats that token,
// rejected everywhere else, and both have to come out identical.
inline Arch dflash_draft(int L = 2, bool own_head = false, bool own_embed = false,
                         int forced_target = -1) {
    const int H=64, Q=64, KV=32, HD=16, I=128, V=128, DV=64;
    const int taps[] = {0, 2};                 // target layers 0 and 2
    const int NT = int(sizeof(taps)/sizeof(taps[0]));
    std::ostringstream c;
    c << R"JSON({
  "model_type": "qwen3_dflash", "hidden_size": 64, "num_hidden_layers": )JSON" << L
      << R"JSON(, "vocab_size": 128,
  "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
  "intermediate_size": 128, "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
  "target_layer_ids": [)JSON" << taps[0] << ',' << taps[1] << R"JSON(],
  "dflash_config": {"mask_token_id": 7, "block_size": 16})JSON";
    if (own_head || forced_target >= 0)
        c << ",\n  \"draft_vocab_size\": "
          << (forced_target >= 0 ? 16 : DV);
    c << "\n}";

    Arch a{forced_target >= 0 ? "dflash+forced"
           : own_head ? "dflash+head" : "dflash", c.str(), {}, V, L};
    auto& t = a.tensors;
    t = { {"fc.weight", {H, int64_t(NT)*H}}, {"hidden_norm.weight", {H}},
          {"norm.weight", {H}} };
    for (int l = 0; l < L; ++l) {
        const std::string b = "layers." + std::to_string(l) + ".";
        t.push_back({b+"input_layernorm.weight", {H}});
        t.push_back({b+"post_attention_layernorm.weight", {H}});
        const std::string s = b + "self_attn.";
        t.push_back({s+"q_proj.weight", {Q,H}});
        t.push_back({s+"k_proj.weight", {KV,H}});
        t.push_back({s+"v_proj.weight", {KV,H}});
        t.push_back({s+"o_proj.weight", {H,Q}});
        t.push_back({s+"q_norm.weight", {HD}});
        t.push_back({s+"k_norm.weight", {HD}});
        const std::string m = b + "mlp.";
        t.push_back({m+"gate_proj.weight", {I,H}});
        t.push_back({m+"up_proj.weight",   {I,H}});
        t.push_back({m+"down_proj.weight", {H,I}});
    }
    if (own_embed) t.push_back({"embed_tokens.weight", {V,H}});
    if (own_head || forced_target >= 0) {
        const int rows = forced_target >= 0 ? 16 : DV;
        t.push_back({"lm_head.weight", {rows,H}});
        // d2t is a DELTA per draft id: target = i + d2t[i].  Delta i maps
        // draft id i to target id 2i, which is inside the target vocabulary
        // and is emphatically NOT the identity, so a build that drops the
        // mapping proposes different tokens rather than the same ones.
        // Only HALF the forced vocabulary collapses onto the target.
        //
        // Mapping every draft id to one token made the fixture blind to
        // the thing the parallel cases exist to check: with all ids equal,
        // a wrong tap row changes the logits but not the proposed token,
        // so acceptance could not move and "identical output" proved
        // nothing about tap fidelity.  With the odd ids left alone the
        // proposal depends on which row wins, which depends on the
        // drafter's hidden state, which depends on the forwarded taps --
        // so acceptance becomes tap-sensitive while staying non-zero.
        std::vector<float> d2t(size_t(rows), 0.0f);
        for (int i = 0; i < rows; ++i)
            d2t[size_t(i)] = float(forced_target >= 0
                ? (i % 2 == 0 ? forced_target - i : 0)
                : i);
        t.push_back({"d2t", {rows}, d2t});
    }
    return a;
}

}  // namespace mini
}  // namespace b70

#endif
