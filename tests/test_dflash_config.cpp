// =====================================================================
//  test_dflash_config.cpp -- pin the DFlash draft-config resolution
//  against ref/qwen3_dflash.py.
//
//  Every value resolved here is one the draft forward pass consumes
//  silently: get it wrong and the drafter still runs, still produces
//  readable output, and simply has its tokens rejected.  There is no
//  crash and no wrong answer to notice, only an acceptance rate -- which
//  is why the Ornith drafter sat at 2.21 accepted per step against a
//  reference reporting 6.1-7.7 while every self-check passed.
//
//  The three configs below are the three shapes vLLM's own
//  _resolve_layer_attention docstring names, so they are the cases the
//  reference was written to handle, not cases invented here.
// =====================================================================
#include "b70/dflash_config.hpp"
#include <cstdio>
#include <cmath>
#include <string>

using namespace b70;
static int g_fail = 0;

#define CHECK(c, ...) do { if(!(c)){ std::printf("  FAIL %s:%d ",__FILE__,__LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while(0)

static void eq_(long long got, long long want, const char* name,
                const char* file, int line) {
    if (got != want) {
        std::printf("  FAIL %s:%d %s: got %lld want %lld\n",
                    file, line, name, got, want);
        ++g_fail;
    }
}
#define EQ(got, want, name) eq_((long long)(got),(long long)(want),name,__FILE__,__LINE__)

// "standard" DFlash: z-lab/Qwen3.5-9B-DFlash -- all full attention,
// non-causal.  This is the shape the Ornith drafter is expected to have.
static const char* kStandard = R"JSON({
  "architectures": ["Qwen3DFlashForCausalLM"],
  "hidden_size": 4096,
  "intermediate_size": 12288,
  "num_hidden_layers": 6,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "rms_norm_eps": 1e-06,
  "vocab_size": 262144,
  "draft_vocab_size": 32768,
  "sliding_window": null,
  "rope_parameters": {"rope_type": "default", "rope_theta": 5000000.0},
  "dflash_config": {"mask_token_id": 248077, "block_size": 16},
  "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37]
})JSON";

// MiMo-V2.5-Pro-FP4-DFlash: use_swa forces sliding on every layer even
// though layer_types says nothing; still non-causal.
static const char* kUseSwa = R"JSON({
  "hidden_size": 4096,
  "num_hidden_layers": 4,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-05,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "dflash_config": {"use_swa": true, "swa_window_size": 2048,
                    "mask_token_id": 201818}
})JSON";

// gemma-4-31B-it-DFlash: mixed layer types; sliding layers are causal,
// full-attention layers are not.
static const char* kMixed = R"JSON({
  "hidden_size": 2048,
  "num_hidden_layers": 4,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "vocab_size": 262144,
  "sliding_window": 1024,
  "layer_types": ["sliding_attention", "full_attention",
                  "sliding_attention", "full_attention"],
  "dflash_config": {"mask_token_id": 7}
})JSON";

int main() {
    // ---- standard DFlash ---------------------------------------------
    {
        const DFlashSettings s = parse_dflash_config(kStandard);
        CHECK(s.error.empty(), "standard config rejected: %s", s.error.c_str());
        EQ(s.n_layers, 6, "standard n_layers");
        EQ(s.head_dim, 128, "standard head_dim");
        EQ(s.mask_token, 248077, "standard mask_token from dflash_config");
        EQ(s.draft_vocab, 32768, "standard draft_vocab");
        EQ(s.target_layers.size(), 8, "standard tap count");
        CHECK(s.tap_key == "target_layer_ids", "standard tap key: %s",
              s.tap_key.c_str());
        // A float rope_theta nested under rope_parameters.  The old
        // substring+strtol reader could not see either.
        CHECK(std::fabs(s.rope_theta - 5.0e6f) < 1.0f,
              "standard rope_theta: got %g want 5e6", double(s.rope_theta));
        CHECK(std::fabs(s.rms_eps - 1.0e-6f) < 1.0e-12f,
              "standard rms_eps: got %g", double(s.rms_eps));
        // THE REGRESSION THIS FILE EXISTS FOR.  The loader used to slide
        // layers 0..4 at a 4096 window for every original DFlash drafter.
        // The reference gives a silent config full attention on every
        // layer, and "sliding_window": null does not change that.
        for (size_t i = 0; i < s.layers.size(); ++i) {
            EQ(s.layers[i].window, 0, "standard layer window");
            CHECK(!s.layers[i].causal, "standard layer %zu must be non-causal", i);
        }
        CHECK(!s.attn_from_config,
              "a config with no layer_types/use_swa/causal describes no "
              "attention shape");
    }

    // ---- use_swa ------------------------------------------------------
    {
        const DFlashSettings s = parse_dflash_config(kUseSwa);
        CHECK(s.error.empty(), "use_swa config rejected: %s", s.error.c_str());
        EQ(s.mask_token, 201818, "use_swa mask_token");
        // head_dim absent -> hidden / heads.
        EQ(s.head_dim, 128, "use_swa derived head_dim");
        EQ(s.draft_vocab, 151936, "draft_vocab defaults to vocab_size");
        CHECK(s.attn_from_config, "use_swa does describe the attention shape");
        for (size_t i = 0; i < s.layers.size(); ++i) {
            EQ(s.layers[i].window, 2048, "use_swa window");
            // use_swa slides but does NOT make a layer causal: causality
            // falls through to layer_types, which is absent.
            CHECK(!s.layers[i].causal, "use_swa layer %zu stays non-causal", i);
        }
        CHECK(std::fabs(s.rms_eps - 1.0e-5f) < 1.0e-12f, "use_swa rms_eps");
    }

    // ---- mixed layer types --------------------------------------------
    {
        const DFlashSettings s = parse_dflash_config(kMixed);
        CHECK(s.error.empty(), "mixed config rejected: %s", s.error.c_str());
        const bool want_sliding[4] = {true, false, true, false};
        for (size_t i = 0; i < s.layers.size(); ++i) {
            EQ(s.layers[i].window, want_sliding[i] ? 1024 : 0, "mixed window");
            CHECK(s.layers[i].causal == want_sliding[i],
                  "mixed layer %zu causal: got %d want %d", i,
                  int(s.layers[i].causal), int(want_sliding[i]));
        }
        // No rms_norm_eps in this config -- the reference's Qwen3Config
        // default is 1e-6 and so is ours.
        CHECK(std::fabs(s.rms_eps - 1.0e-6f) < 1.0e-12f, "mixed default eps");
    }

    // ---- is_causal overrides everything --------------------------------
    {
        const DFlashSettings s = parse_dflash_config(R"JSON({
          "hidden_size": 512, "num_hidden_layers": 2,
          "num_attention_heads": 8, "num_key_value_heads": 8,
          "is_causal": true, "sliding_window": 128,
          "layer_types": ["sliding_attention", "full_attention"]
        })JSON");
        CHECK(s.error.empty(), "is_causal config rejected: %s", s.error.c_str());
        CHECK(s.layers[0].causal && s.layers[1].causal,
              "is_causal:true makes every layer causal, layer_types or not");
        EQ(s.layers[0].window, 128, "is_causal keeps the sliding window");
        EQ(s.layers[1].window, 0, "full layer stays full under is_causal");
    }
    {   // dflash_config.causal is the second-priority override
        const DFlashSettings s = parse_dflash_config(R"JSON({
          "hidden_size": 512, "num_hidden_layers": 2,
          "num_attention_heads": 8, "num_key_value_heads": 8,
          "sliding_window": 64,
          "layer_types": ["sliding_attention", "full_attention"],
          "dflash_config": {"causal": false}
        })JSON");
        CHECK(s.error.empty(), "causal-override config rejected: %s",
              s.error.c_str());
        CHECK(!s.layers[0].causal && !s.layers[1].causal,
              "dflash_config.causal:false wins over layer_types");
        EQ(s.layers[0].window, 64, "override does not change the window");
    }

    // ---- rope_theta: the reference's default is 1e6, not 1e7 -----------
    {
        const DFlashSettings s = parse_dflash_config(R"JSON({
          "hidden_size": 512, "num_hidden_layers": 1,
          "num_attention_heads": 8, "num_key_value_heads": 8
        })JSON");
        CHECK(std::fabs(s.rope_theta - 1.0e6f) < 1.0f,
              "silent config -> set_default_rope_theta 1e6, got %g",
              double(s.rope_theta));
        CHECK(!s.notes.empty(), "a silent rope_theta is worth saying out loud");
        EQ(s.mask_token, -1, "no mask_token_id means none, not a guess");
    }
    {   // top-level rope_theta still wins when rope_parameters is absent
        const DFlashSettings s = parse_dflash_config(R"JSON({
          "hidden_size": 512, "num_hidden_layers": 1,
          "num_attention_heads": 8, "num_key_value_heads": 8,
          "rope_theta": 10000000.0
        })JSON");
        CHECK(std::fabs(s.rope_theta - 1.0e7f) < 1.0f, "top-level rope_theta");
    }

    // ---- alternate tap key spellings -----------------------------------
    for (const char* key : {"aux_hidden_state_layer_ids",
                            "eagle_aux_hidden_state_layer_ids"}) {
        const std::string cfg = std::string(R"JSON({
          "hidden_size": 512, "num_hidden_layers": 1,
          "num_attention_heads": 8, "num_key_value_heads": 8,
          ")JSON") + key + R"JSON(": [2, 5, 9]})JSON";
        const DFlashSettings s = parse_dflash_config(cfg);
        CHECK(s.error.empty(), "%s rejected: %s", key, s.error.c_str());
        EQ(s.target_layers.size(), 3, "tap count from an alternate key");
        CHECK(s.tap_key == key, "tap key recorded: %s", s.tap_key.c_str());
        EQ(s.target_layers[2], 9, "tap ids read in order");
    }

    // ---- refusals ------------------------------------------------------
    {   // sliding with no window anywhere: the reference raises, so do we
        const DFlashSettings s = parse_dflash_config(R"JSON({
          "hidden_size": 512, "num_hidden_layers": 2,
          "num_attention_heads": 8, "num_key_value_heads": 8,
          "dflash_config": {"use_swa": true}
        })JSON");
        CHECK(!s.error.empty(), "a sliding layer with no window must refuse");
    }
    {
        const DFlashSettings s = parse_dflash_config(R"JSON({
          "hidden_size": 512, "num_hidden_layers": 4,
          "num_attention_heads": 8, "num_key_value_heads": 8,
          "layer_types": ["full_attention", "full_attention"]
        })JSON");
        CHECK(!s.error.empty(), "layer_types shorter than the model must refuse");
    }
    {
        const DFlashSettings s = parse_dflash_config("{ not json");
        CHECK(!s.error.empty(), "invalid JSON must refuse, not throw");
    }
    {
        const DFlashSettings s = parse_dflash_config("[]");
        CHECK(!s.error.empty(), "a non-object config must refuse");
    }
    {
        const DFlashSettings s = parse_dflash_config(R"JSON({"hidden_size": 8})JSON");
        CHECK(!s.error.empty(), "no num_hidden_layers must refuse");
    }

    // ---- attention_bias is a divergence, and says so -------------------
    {
        const DFlashSettings s = parse_dflash_config(R"JSON({
          "hidden_size": 512, "num_hidden_layers": 1,
          "num_attention_heads": 8, "num_key_value_heads": 8,
          "attention_bias": true
        })JSON");
        CHECK(s.attention_bias, "attention_bias read");
        bool warned = false;
        for (const std::string& n : s.notes)
            if (n.find("attention_bias") != std::string::npos) warned = true;
        CHECK(warned, "an unimplemented qkv/o bias must be said out loud");
    }

    if (g_fail) { std::printf("dflash-config: %d FAILURES\n", g_fail); return 1; }
    std::printf("dflash-config: reference attention shapes, rope default, "
                "tap keys and refusals PASS\n");
    return 0;
}
