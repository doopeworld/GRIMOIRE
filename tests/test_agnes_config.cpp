// =====================================================================
//  test_agnes_config.cpp -- parse the REAL Agnes-3.0-Flash config.json
//  through the production loader and pin every field the forward pass
//  depends on.
//
//  The config text below is verbatim from the checkpoint
//  (Agnes-AI/Agnes-3.0-Flash), so a loader change that silently drops a
//  key fails here rather than on the card.
//
//  Two things about this architecture are silent when wrong, and both
//  are pinned:
//
//  1. It nests every text dimension under "text_config" and ships a
//     vision tower with its OWN hidden_size (1152).  Scanning the whole
//     file for "hidden_size" returns whichever comes first -- correct
//     here only because of the order the file happens to be written in.
//  2. It spells its layer types "agnes_delta_attention" and
//     "agnes_global_attention".  The old loader mapped anything it did
//     not recognise to FULL_ATTN, so all 54 gated-DeltaNet layers would
//     have run as full attention: a model that loads, generates fluent
//     text, and is not this model.
// =====================================================================
#include "b70/qwen35.hpp"
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace b70;
namespace fs = std::filesystem;
static int g_fail = 0;

#define CHECK(c, ...) do { if(!(c)){ std::printf("  FAIL %s:%d ",__FILE__,__LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while(0)
static void eq_(long long got, long long want, const char* name,
                const char* file, int line) {
    if (got != want) {
        std::printf("  FAIL %s:%d %s: got %lld want %lld\n", file, line, name, got, want);
        ++g_fail;
    }
}
#define EQ(got, want, name) eq_((long long)(got),(long long)(want),name,__FILE__,__LINE__)

// The 72-entry layer_types array, exactly as the checkpoint writes it:
// three agnes_delta_attention then one agnes_global_attention, repeated.
static std::string layer_types_array() {
    std::string a = "[";
    for (int i = 0; i < 72; ++i) {
        if (i) a += ",";
        a += (i % 4 == 3) ? "\"agnes_global_attention\""
                          : "\"agnes_delta_attention\"";
    }
    return a + "]";
}

static std::string config(bool with_parallel_ffn) {
    std::ostringstream c;
    c << R"JSON({
  "model_type": "agnes",
  "architectures": ["AgnesForConditionalGeneration"],
  "auto_map": {
    "AutoConfig": "configuration_agnes.AgnesConfig",
    "AutoModelForCausalLM": "modeling_agnes.AgnesForConditionalGeneration"
  },
  "text_config": {
    "model_type": "agnes_text",
    "hidden_size": 5120,
    "num_hidden_layers": 72,
    "intermediate_size": 17408,)JSON";
    if (with_parallel_ffn) c << R"JSON(
    "parallel_ffn_intermediate_size": 2048,)JSON";
    c << R"JSON(
    "vocab_size": 248320,
    "layer_types": )JSON" << layer_types_array() << R"JSON(,
    "global_attention_interval": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "attn_output_gate": true,
    "partial_rotary_factor": 0.25,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "output_gate_type": "swish",
    "mamba_ssm_dtype": "float32",
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "initializer_range": 0.02,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    },
    "max_position_embeddings": 262144,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "bos_token_id": 248044,
    "eos_token_id": 248044,
    "tie_word_embeddings": false,
    "use_cache": true,
    "dtype": "bfloat16"
  },
  "vision_config": {
    "model_type": "agnes_vision",
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "out_hidden_size": 5120,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "num_position_embeddings": 2304,
    "deepstack_visual_indexes": [],
    "hidden_act": "gelu_pytorch_tanh",
    "initializer_range": 0.02
  },
  "image_token_id": 248056,
  "video_token_id": 248057,
  "vision_start_token_id": 248053,
  "vision_end_token_id": 248054,
  "tie_word_embeddings": false
})JSON";
    return c.str();
}

// One real tensor so the loader's index is non-empty.
static void write_stub(const fs::path& dir, const std::string& cfg) {
    fs::create_directories(dir);
    std::ofstream(dir/"config.json") << cfg;
    const std::vector<uint16_t> vals(256*8, 0);
    std::ostringstream h;
    h << "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[256,8],"
         "\"data_offsets\":[0," << vals.size()*2 << "]}}";
    std::string hs = h.str(); while (hs.size() % 8) hs += ' ';
    std::ofstream f(dir/"model.safetensors", std::ios::binary);
    const uint64_t n = hs.size();
    f.write((const char*)&n, 8); f << hs;
    f.write((const char*)vals.data(), std::streamsize(vals.size()*2));
}

int main() {
    std::printf("=== Agnes-3.0-Flash config parse (real checkpoint config.json) ===\n\n");
    char tmpl[] = "/tmp/grimoire-agnescfg-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;

    // ---- the real config: must REFUSE, and say why --------------------
    {
        const fs::path d = root / "real";
        write_stub(d, config(true));
        Qwen35Model ld; std::string err;
        const bool ok = ld.load(d.string(), err, /*skip_vision=*/true, /*index_only=*/true);
        std::printf("parallel_ffn is refused, not approximated\n");
        CHECK(!ok, "the checkpoint declares parallel_ffn and the loader accepted it");
        CHECK(err.find("parallel_ffn") != std::string::npos,
              "the refusal must name parallel_ffn: %s", err.c_str());
        // 2048 / (17408 + 2048) = 10.5%.  The message quotes the share of
        // every FFN that would go missing, because "some of the model" is
        // not an actionable thing to read at 3am.
        CHECK(err.find("10%") != std::string::npos,
              "the refusal should quantify what would be dropped: %s", err.c_str());
        std::printf("  %s\n\n", err.c_str());
    }

    // ---- everything else, with that one key removed -------------------
    const fs::path d = root / "nopffn";
    write_stub(d, config(false));
    Qwen35Model ld; std::string err;
    if (!ld.load(d.string(), err, true, /*index_only=*/true)) {
        std::printf("  FAIL loader rejected the config: %s\n", err.c_str());
        std::error_code e; fs::remove_all(root, e);
        return 1;
    }
    const auto& c = ld.cfg;

    std::printf("detection and core dims\n");
    CHECK(c.is_agnes, "model_type agnes not detected");
    CHECK(!c.is_muse && !c.is_k2, "must not be mistaken for Muse or K2");
    // THE regression: the vision tower's hidden_size is 1152 and sits in
    // the same file.  Reading the text tower's 5120 is the whole point of
    // extracting text_config instead of scanning the file.
    EQ(c.hidden, 5120, "hidden (NOT the vision tower's 1152)");
    EQ(c.n_layers, 72, "n_layers");
    EQ(c.vocab, 248320, "vocab");
    EQ(c.n_heads, 24, "n_heads");
    EQ(c.n_kv_heads, 4, "n_kv_heads");
    EQ(c.head_dim, 256, "head_dim");
    EQ(c.dense_inter, 17408, "intermediate_size (NOT the vision tower's 4304)");
    CHECK(!c.is_moe(), "Agnes-3.0-Flash is dense, not MoE");
    CHECK(!c.tie_embeddings, "tie_word_embeddings");

    std::printf("attention\n");
    CHECK(c.attn_out_gate, "attn_output_gate");
    CHECK(std::fabs(c.partial_rope - 0.25f) < 1e-6f,
          "partial_rotary_factor: got %g want 0.25", double(c.partial_rope));
    CHECK(std::fabs(c.rope_theta - 1.0e7f) < 1.0f,
          "rope_theta: got %g want 1e7", double(c.rope_theta));
    CHECK(std::fabs(c.rms_eps - 1.0e-6f) < 1e-12f, "rms_norm_eps");
    // head_dim 256 / SG_SIZE 16 = 16 accumulator slots, exactly MAX_DPL.
    EQ(c.head_dim / 16, 16, "head_dim/SG_SIZE must fit MAX_DPL");

    std::printf("gated DeltaNet\n");
    EQ(c.lin_k_heads, 16, "linear_num_key_heads");
    EQ(c.lin_v_heads, 48, "linear_num_value_heads");
    EQ(c.lin_k_dim, 128, "linear_key_head_dim");
    EQ(c.lin_v_dim, 128, "linear_value_head_dim");
    EQ(c.conv_kernel, 4, "linear_conv_kernel_dim");

    std::printf("mRoPE\n");
    CHECK(c.mrope, "mrope_section not detected");
    EQ(c.mrope_section.size(), 3, "mrope_section length");
    if (c.mrope_section.size() == 3) {
        EQ(c.mrope_section[0], 11, "mrope_section[0]");
        EQ(c.mrope_section[1], 11, "mrope_section[1]");
        EQ(c.mrope_section[2], 10, "mrope_section[2]");
        // The sections partition the ROTARY half-dims: head_dim 256 at
        // partial_rotary_factor 0.25 is 64 rotated dims = 32 pairs, and
        // 11+11+10 = 32.  If that ever fails to add up, the text-only
        // equivalence with plain RoPE no longer holds either.
        const int pairs = int(c.head_dim * c.partial_rope) / 2;
        EQ(c.mrope_section[0]+c.mrope_section[1]+c.mrope_section[2], pairs,
           "mrope_section must partition the rotary half-dims");
    }

    std::printf("layer types (3 delta : 1 global, 72 layers)\n");
    EQ(c.layer_types.size(), 72, "layer_types length");
    int linear = 0, full = 0;
    bool pattern_ok = true;
    for (int i = 0; i < int(c.layer_types.size()); ++i) {
        const bool want_full = (i % 4 == 3);
        const bool got_full = c.layer_types[size_t(i)] == LayerKind::FULL_ATTN;
        if (got_full != want_full) pattern_ok = false;
        (got_full ? full : linear)++;
    }
    CHECK(pattern_ok, "layer 3,7,11... must be the global-attention layers");
    EQ(linear, 54, "gated-DeltaNet layers");
    EQ(full, 18, "full-attention layers");
    CHECK(linear > 0, "agnes_delta_attention must NOT fall through to FULL_ATTN");

    // ---- an unrecognised layer type must refuse, not default ----------
    {
        std::string bad = config(false);
        const size_t p = bad.find("\"agnes_delta_attention\"");
        bad.replace(p, std::strlen("\"agnes_delta_attention\""), "\"agnes_mystery\"");
        const fs::path d2 = root / "bad_layer";
        write_stub(d2, bad);
        Qwen35Model l2; std::string e2;
        const bool ok = l2.load(d2.string(), e2, true, true);
        CHECK(!ok, "an unknown layer_types entry must refuse, not default to full");
        CHECK(e2.find("agnes_mystery") != std::string::npos,
              "the refusal must name the value it did not understand: %s", e2.c_str());
    }

    std::error_code ec; fs::remove_all(root, ec);
    if (g_fail) { std::printf("\nagnes-config: %d FAILURES\n", g_fail); return 1; }
    std::printf("\nALL PASS (0 failures)\n");
    return 0;
}
