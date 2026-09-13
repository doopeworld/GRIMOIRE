#pragma once
// =====================================================================
//  Resolve a DFlash drafter's config.json exactly the way vLLM's
//  qwen3_dflash.py does.
//
//  WHY THIS FILE EXISTS.  Every field resolved here feeds the draft
//  forward pass, and NOT ONE of them fails loudly when it is wrong.  A
//  drafter handed the wrong RoPE theta, the wrong mask token, the wrong
//  norm epsilon or the wrong attention shape still runs, still produces
//  fluent-looking draft tokens, and still yields correct output -- the
//  verifier simply rejects nearly all of them.  The only symptom is an
//  acceptance rate.  vLLM says so itself, about its own RoPE layout:
//  "a mismatch is silent -- acceptance collapses but nothing errors and
//  the output stays correct."
//
//  That is exactly the shape of the open Ornith blocker: 2.21 accepted
//  tokens per step against the reference's 6.1-7.7
//  (HANDOFF-2026-08-28-ORNITH-DFLASH.md).  Before this, the loader read
//  four integers out of the draft config with a substring search and
//  hardcoded the rest -- head_dim 128, eps 1e-6, rope_theta 1e7, and
//  "layers 0..4 slide at 4096".  The reference derives all of them, and
//  its default rope_theta is 1e6, not 1e7.
//
//  The rules below are transcribed from ref/qwen3_dflash.py
//  (_resolve_layer_attention, _dflash_layer_causal, DFlashQwen3Model
//  .__init__, DFlashQwen3DecoderLayer.__init__).  Keep them transcribed:
//  if the reference changes, change this, do not re-derive it.
// =====================================================================
#include "b70/json.hpp"
#include <string>
#include <string_view>
#include <vector>

namespace b70 {

// One draft layer's attention shape.  window == 0 means full attention.
struct DFlashLayerAttn {
    int  window = 0;
    bool causal = false;
};

struct DFlashSettings {
    int   n_layers      = 0;
    int   hidden        = 0;
    int   n_heads       = 0;
    int   n_kv_heads    = 0;
    int   head_dim      = 0;   // resolved: head_dim, else hidden / n_heads
    int   inter         = 0;
    int   vocab         = 0;
    int   draft_vocab   = 0;
    int   mask_token    = -1;  // -1: the config names none
    int   selector_top_k = 0;
    int   selector_rank  = 0;
    float rope_theta    = 1.0e6f;  // vLLM's set_default_rope_theta default
    float rms_eps       = 1.0e-6f;
    bool  attention_bias = false;
    bool  use_aux_hidden_state = true;
    // True when the config actually described the attention shape
    // (layer_types, dflash_config.use_swa, dflash_config.causal or
    // is_causal).  When false, `layers` carries what the reference would
    // compute from a silent config -- full attention, non-causal -- and
    // the caller can see that no checkpoint fact backs it.
    bool  attn_from_config = false;
    // Which key the target taps came from, empty if none was found.
    std::string tap_key;
    std::vector<int>             target_layers;
    std::vector<DFlashLayerAttn> layers;
    std::vector<std::string>     notes;   // printed at load
    std::string                  error;   // non-empty: refuse the drafter
};

namespace dflash_detail {

inline const Json* typed(const Json& j, const char* key, Json::Kind kind) {
    const Json* v = j.find(key);
    return v && v->kind == kind ? v : nullptr;
}

// The reference builds one lookup scope for the drafter's own knobs:
//   drafter_config = dict(eagle_config); drafter_config.update(dflash_config)
// so dflash_config wins over eagle_config, and both win over the top
// level.  Attention shape is deliberately NOT resolved through this --
// _resolve_layer_attention reads config.dflash_config directly.
struct Scopes {
    const Json* dflash = nullptr;
    const Json* eagle  = nullptr;
    const Json* top    = nullptr;

    const Json* lookup(const char* key, Json::Kind kind) const {
        if (dflash) if (const Json* v = typed(*dflash, key, kind)) return v;
        if (eagle)  if (const Json* v = typed(*eagle,  key, kind)) return v;
        if (top)    if (const Json* v = typed(*top,    key, kind)) return v;
        return nullptr;
    }
    int number_or(const char* key, int fallback) const {
        const Json* v = lookup(key, Json::Kind::Number);
        return v ? int(v->number) : fallback;
    }
    bool flag_or(const char* key, bool fallback) const {
        const Json* v = lookup(key, Json::Kind::Boolean);
        return v ? v->boolean : fallback;
    }
};

inline std::vector<int> int_array(const Json& v) {
    std::vector<int> out;
    for (const Json& e : v.array) {
        if (e.kind != Json::Kind::Number) return {};
        out.push_back(int(e.number));
    }
    return out;
}

} // namespace dflash_detail

// Parse a DFlash draft config.json.  Never throws: a config that cannot
// be used comes back with `error` set, because a drafter is optional and
// refusing it must not take the engine down with it.
inline DFlashSettings parse_dflash_config(std::string_view text) {
    using Kind = Json::Kind;
    using namespace dflash_detail;
    DFlashSettings s;

    Json top;
    try {
        top = Json::parse(text);
    } catch (const std::exception& e) {
        s.error = std::string("draft config.json is not valid JSON: ") + e.what();
        return s;
    }
    if (top.kind != Kind::Object) {
        s.error = "draft config.json is not a JSON object";
        return s;
    }

    const Json* dflash_cfg = typed(top, "dflash_config", Kind::Object);
    const Json* eagle_cfg  = typed(top, "eagle_config",  Kind::Object);
    const Scopes sc{dflash_cfg, eagle_cfg, &top};

    // ---- plain shape -------------------------------------------------
    s.n_layers   = sc.number_or("num_hidden_layers", 0);
    s.hidden     = sc.number_or("hidden_size", 0);
    s.n_heads    = sc.number_or("num_attention_heads", 0);
    s.n_kv_heads = sc.number_or("num_key_value_heads", s.n_heads);
    s.inter      = sc.number_or("intermediate_size", 0);
    s.vocab      = sc.number_or("vocab_size", 0);
    // DFlashQwen3ForCausalLM: draft_vocab_size defaults to vocab_size.
    s.draft_vocab = sc.number_or("draft_vocab_size", s.vocab);
    s.attention_bias = sc.flag_or("attention_bias", false);
    s.use_aux_hidden_state = sc.flag_or("use_aux_hidden_state", true);
    s.selector_top_k = sc.number_or("selector_top_k", 0);
    s.selector_rank  = sc.number_or("selector_rank", 0);

    if (s.n_layers <= 0)
        s.error = "draft config.json has no usable num_hidden_layers";

    // head_dim, else hidden_size // total_num_heads (DFlashQwen3Attention).
    s.head_dim = sc.number_or("head_dim", 0);
    if (s.head_dim <= 0 && s.n_heads > 0) s.head_dim = s.hidden / s.n_heads;

    if (const Json* v = sc.lookup("rms_norm_eps", Kind::Number))
        s.rms_eps = float(v->number);

    // ---- RoPE --------------------------------------------------------
    // Current vLLM keeps RoPE under config.rope_parameters; older configs
    // put rope_theta at the top level.  When neither exists the decoder
    // layer calls set_default_rope_theta(config, default_theta=1000000),
    // so 1e6 -- not 1e7 -- is the reference's answer for a silent config.
    // Read it as a double: "rope_theta": 5000000.0 is the usual spelling
    // and an integer parse of it is a coin flip.
    bool rope_seen = false;
    if (const Json* rp = typed(top, "rope_parameters", Kind::Object))
        if (const Json* v = typed(*rp, "rope_theta", Kind::Number)) {
            s.rope_theta = float(v->number); rope_seen = true;
        }
    if (!rope_seen)
        if (const Json* v = typed(top, "rope_theta", Kind::Number)) {
            s.rope_theta = float(v->number); rope_seen = true;
        }
    if (!rope_seen)
        s.notes.push_back("no rope_theta in the draft config; using vLLM's "
                          "default 1000000 (set_default_rope_theta)");

    // ---- mask token --------------------------------------------------
    // drafter_config.get("mask_token_id", getattr(config, "mask_token_id"))
    if (const Json* v = sc.lookup("mask_token_id", Kind::Number))
        s.mask_token = int(v->number);
    else
        s.notes.push_back("no mask_token_id in the draft config -- the mask "
                          "rows are embedded from whatever id the caller "
                          "defaults to, which is a guess");

    // ---- target taps -------------------------------------------------
    // The ecosystem spells this three ways.  Which one it is does not
    // change the meaning, so accept all three and record the winner; the
    // count is cross-checked against fc's input width by the caller,
    // which is the check that actually catches a wrong tap set.
    for (const char* key : {"target_layer_ids", "aux_hidden_state_layer_ids",
                            "eagle_aux_hidden_state_layer_ids"}) {
        const Json* v = sc.lookup(key, Kind::Array);
        if (!v) continue;
        std::vector<int> ids = int_array(*v);
        if (ids.empty()) continue;
        s.target_layers = std::move(ids);
        s.tap_key = key;
        break;
    }
    if (s.target_layers.empty())
        s.notes.push_back("the draft config names no target layer ids; the "
                          "taps below are a built-in default, not a "
                          "checkpoint fact");

    // ---- attention shape, per layer ----------------------------------
    // _resolve_layer_attention + _dflash_layer_causal, transcribed.
    const Json* layer_types = typed(top, "layer_types", Kind::Array);
    const Json* swa_flag =
        dflash_cfg ? typed(*dflash_cfg, "use_swa", Kind::Boolean) : nullptr;
    const bool use_swa = swa_flag && swa_flag->boolean;

    bool any_sliding = false;
    if (layer_types)
        for (const Json& e : layer_types->array)
            if (e.kind == Kind::String && e.string == "sliding_attention")
                any_sliding = true;

    // causal, resolved once: is_causal wins, then dflash_config.causal,
    // then "this layer is a sliding layer".
    const Json* is_causal = typed(top, "is_causal", Kind::Boolean);
    const Json* causal_override =
        dflash_cfg ? typed(*dflash_cfg, "causal", Kind::Boolean) : nullptr;

    s.attn_from_config = layer_types != nullptr || swa_flag != nullptr ||
                         is_causal != nullptr || causal_override != nullptr;

    // layer_types shorter than the model is an IndexError in the reference,
    // so it is a broken config, not a layer to guess at.
    if (layer_types && s.n_layers > 0 &&
        layer_types->array.size() != size_t(s.n_layers) && s.error.empty())
        s.error = "draft config layer_types has " +
                  std::to_string(layer_types->array.size()) + " entries for " +
                  std::to_string(s.n_layers) + " layers";

    // The window, when a layer slides: dflash_config.swa_window_size, else
    // the top-level sliding_window.  The reference raises when a sliding
    // layer has neither, and so does this -- a guessed window is the same
    // class of silent divergence this file exists to remove.
    int window = 0;
    if (dflash_cfg)
        if (const Json* v = typed(*dflash_cfg, "swa_window_size", Kind::Number))
            window = int(v->number);
    if (window <= 0)
        if (const Json* v = typed(top, "sliding_window", Kind::Number))
            window = int(v->number);

    s.layers.resize(size_t(s.n_layers > 0 ? s.n_layers : 0));
    for (size_t i = 0; i < s.layers.size(); ++i) {
        const bool typed_sliding =
            layer_types && i < layer_types->array.size() &&
            layer_types->array[i].kind == Kind::String &&
            layer_types->array[i].string == "sliding_attention";
        const bool is_sliding =
            (!layer_types || (use_swa && !any_sliding)) ? use_swa : typed_sliding;

        s.layers[i].window = is_sliding ? window : 0;
        s.layers[i].causal = is_causal       ? is_causal->boolean
                           : causal_override ? causal_override->boolean
                                             : (layer_types && typed_sliding);
        if (is_sliding && window <= 0 && s.error.empty())
            s.error = "draft layer " + std::to_string(i) + " uses sliding "
                      "attention but the config gives no window "
                      "(dflash_config.swa_window_size or sliding_window)";
    }

    if (!s.attn_from_config)
        s.notes.push_back("the draft config describes no attention shape, so "
                          "every layer is full attention and non-causal -- "
                          "which is what the reference computes for a silent "
                          "config, not an assumption made here");
    if (s.attention_bias)
        s.notes.push_back("attention_bias is set: the reference gives qkv and "
                          "o_proj a bias, and this engine's draft path has no "
                          "bias term -- acceptance from this drafter is not "
                          "comparable to the reference");

    return s;
}

} // namespace b70
