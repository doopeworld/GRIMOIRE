// =====================================================================
//  test_gemma4_config.cpp -- parse the REAL gemma-4-31B-it config.json
//  through the production loader, and pin every field the forward pass
//  depends on.
//
//  The config text below is verbatim from google/gemma-4-31B-it, and the
//  weight map is built to match the real model.safetensors.index.json.
//  Both were read, not guessed; the forward pass they describe is in
//  ref/gemma4.py and the write-up is GEMMA4-2026-09-14.md.
//
//  Nine things about this architecture are silent when wrong.  The ones
//  the LOADER decides are pinned here:
//
//  1. Per-LAYER-TYPE geometry.  Sliding layers are head_dim 256 over 16
//     KV heads; full-attention layers are head_dim 512 over 4.  Nothing
//     else in this engine has two geometries in one model, and reading
//     the model-wide scalar for a full layer is silent.
//  2. rope_parameters keyed BY LAYER TYPE -- sliding at theta 1e4 plain,
//     full at 1e6 "proportional" with partial_rotary_factor 0.25.  A
//     whole-file scan for "rope_theta" finds whichever comes first.
//  3. attention_k_eq_v: a FULL-attention layer ships NO v_proj.  The
//     real weight map proves it -- layers 5, 11, ... 59 have none and
//     every sliding layer does.
//  4. layer_scalar, which config.json never mentions: the reference
//     multiplies the residual stream by it at the end of every layer.
//  5. The vision tower declares its own hidden_activation
//     ("gelu_pytorch_tanh" as it happens), so the activation check must
//     read the TEXT block and never fall back to the whole file.
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

// The real layer_types: five sliding_attention then one full_attention,
// repeated.  With 60 layers that puts full attention at 5, 11, ... 59.
static std::string layer_types_array(int n) {
    std::string a = "[";
    for (int i = 0; i < n; ++i) {
        if (i) a += ",";
        a += (i % 6 == 5) ? "\"full_attention\"" : "\"sliding_attention\"";
    }
    return a + "]";
}
static bool is_full(int i) { return i % 6 == 5; }

// The checkpoint's own text_config, verbatim except num_hidden_layers,
// which the caller shrinks so a test can build a matching weight map.
struct Opts {
    const char* moe    = "false";
    const char* kvshare= "0";
    const char* perlay = "0";
    const char* dwmlp  = "false";
};
static std::string config(int n_layers, Opts o = Opts{}) {
    std::ostringstream c;
    c << R"JSON({
  "architectures": ["Gemma4ForConditionalGeneration"],
  "model_type": "gemma4",
  "dtype": "bfloat16",
  "eos_token_id": [1, 106],
  "image_token_id": 258880,
  "tie_word_embeddings": true,
  "text_config": {
    "model_type": "gemma4_text",
    "attention_bias": false,
    "attention_k_eq_v": true,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 5376,
"hidden_size_per_layer_input": )JSON" << o.perlay << R"JSON(,
    "intermediate_size": 21504,
"enable_moe_block": )JSON" << o.moe << R"JSON(,
    "num_experts": null,
    "num_attention_heads": 32,
    "num_global_key_value_heads": 4,
    "num_key_value_heads": 16,
"num_kv_shared_layers": )JSON" << o.kvshare << R"JSON(,
    "num_hidden_layers": )JSON" << n_layers << R"JSON(,
    "layer_types": )JSON" << layer_types_array(n_layers) << R"JSON(,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional"
      },
      "sliding_attention": {
        "rope_theta": 10000.0,
        "rope_type": "default"
      }
    },
    "sliding_window": 1024,
    "tie_word_embeddings": true,
"use_double_wide_mlp": )JSON" << o.dwmlp << R"JSON(,
    "vocab_size": 262144
  },
  "vision_config": {
    "model_type": "gemma4_vision",
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "head_dim": 72,
    "global_head_dim": 72,
    "intermediate_size": 4304,
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "num_key_value_heads": 16,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"}
  }
})JSON";
    return c.str();
}

// The tensor NAMES a correctly shaped checkpoint carries: the real
// prefix, no lm_head (tied), a layer_scalar on every layer, and a
// v_proj on sliding layers ONLY.
static std::vector<std::string> tensor_names(int n_layers, int drop_scalar = -1,
                                             int force_v_on_full = -1,
                                             int drop_v = -1) {
    std::vector<std::string> n;
    n.push_back("model.language_model.embed_tokens.weight");
    n.push_back("model.language_model.norm.weight");
    for (int i = 0; i < n_layers; ++i) {
        const std::string b = "model.language_model.layers." + std::to_string(i) + ".";
        n.push_back(b + "input_layernorm.weight");
        n.push_back(b + "post_attention_layernorm.weight");
        n.push_back(b + "pre_feedforward_layernorm.weight");
        n.push_back(b + "post_feedforward_layernorm.weight");
        if (i != drop_scalar) n.push_back(b + "layer_scalar");
        n.push_back(b + "mlp.gate_proj.weight");
        n.push_back(b + "mlp.up_proj.weight");
        n.push_back(b + "mlp.down_proj.weight");
        n.push_back(b + "self_attn.q_proj.weight");
        n.push_back(b + "self_attn.k_proj.weight");
        n.push_back(b + "self_attn.o_proj.weight");
        n.push_back(b + "self_attn.q_norm.weight");
        n.push_back(b + "self_attn.k_norm.weight");
        if ((!is_full(i) || i == force_v_on_full) && i != drop_v)
            n.push_back(b + "self_attn.v_proj.weight");
    }
    return n;
}

// The loader indexes the SAFETENSORS files themselves, not
// model.safetensors.index.json, so a weight-shape test has to write real
// tensors.  Two elements each: enough to be 2-D and resolvable, and the
// checks here are about which tensors EXIST, not their values.
static void write_stub(const fs::path& dir, const std::string& cfg,
                       const std::vector<std::string>& names = {}) {
    fs::create_directories(dir);
    std::ofstream(dir/"config.json") << cfg;
    std::vector<std::string> all = names;
    if (all.empty()) all.push_back("model.embed_tokens.weight");
    std::ostringstream h;
    h << "{";
    size_t off = 0;
    for (size_t i = 0; i < all.size(); ++i) {
        // layer_scalar is nn.Buffer(torch.ones(1)) in the reference
        // (ref/gemma4.py:1367), i.e. exactly ONE element, and the uploader
        // reads it into a single float.  Writing it as [2] here would make
        // this fixture disagree with the checkpoint it stands for, and the
        // loader's shape check -- correctly -- rejects it.
        const bool scalar = all[i].size() > 7 &&
                            all[i].compare(all[i].size()-7, 7, "_scalar") == 0;
        const bool vec = scalar ||
                         all[i].find("norm.weight") != std::string::npos;
        const size_t elems = scalar ? 1 : (vec ? 2 : 4);   // [1], [2] or [2,2]
        if (i) h << ",";
        h << "\"" << all[i] << "\":{\"dtype\":\"BF16\",\"shape\":"
          << (scalar ? "[1]" : vec ? "[2]" : "[2,2]")
          << ",\"data_offsets\":[" << off << "," << off + elems*2 << "]}";
        off += elems * 2;
    }
    h << "}";
    std::string hs = h.str(); while (hs.size() % 8) hs += ' ';
    std::ofstream f(dir/"model.safetensors", std::ios::binary);
    const uint64_t n = hs.size();
    f.write((const char*)&n, 8); f << hs;
    const std::vector<uint16_t> zeros(off/2, 0);
    f.write((const char*)zeros.data(), std::streamsize(off));
}

int main() {
    std::printf("=== gemma-4-31B-it config parse (real checkpoint config.json) ===\n\n");
    char tmpl[] = "/tmp/grimoire-g4cfg-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;

    // ---- the real config, all 60 layers -------------------------------
    {
        const fs::path d = root / "real";
        write_stub(d, config(60));
        Qwen35Model ld; std::string err;
        const bool ok = ld.load(d.string(), err, /*skip_vision=*/true,
                                /*index_only=*/true);
        std::printf("the real config loads\n");
        CHECK(ok, "the checkpoint's own config was rejected: %s", err.c_str());
        if (ok) {
            const auto& c = ld.cfg;
            CHECK(c.is_gemma4, "model_type gemma4 detected");

            std::printf("core dims\n");
            // 5376, not the vision tower's 1152.
            EQ(c.hidden, 5376, "hidden_size (text, not the vision tower)");
            EQ(c.n_layers, 60, "num_hidden_layers");
            EQ(c.vocab, 262144, "vocab_size");
            EQ(c.n_heads, 32, "num_attention_heads");
            EQ(c.dense_inter, 21504, "intermediate_size");
            CHECK(c.tie_embeddings, "tie_word_embeddings");
            CHECK(std::fabs(c.rms_eps - 1e-6f) < 1e-12f, "rms_norm_eps");

            std::printf("per-layer-type geometry\n");
            EQ(c.head_dim, 256, "sliding head_dim");
            EQ(c.n_kv_heads, 16, "sliding kv heads");
            EQ(c.global_head_dim, 512, "full-attention head_dim");
            EQ(c.n_global_kv_heads, 4, "full-attention kv heads");
            EQ(c.max_head_dim(), 512, "max head_dim");
            EQ(c.max_kv_heads(), 16, "max kv heads");
            // The accessors are the whole point: reading the model-wide
            // scalar for a full layer is the silent failure.
            EQ(c.layer_head_dim(0), 256, "layer 0 is sliding");
            EQ(c.layer_kv_heads(0), 16, "layer 0 kv heads");
            EQ(c.layer_head_dim(5), 512, "layer 5 is full attention");
            EQ(c.layer_kv_heads(5), 4, "layer 5 kv heads");
            EQ(c.layer_head_dim(59), 512, "layer 59 is full attention");
            EQ(c.layer_head_dim(58), 256, "layer 58 is sliding");

            std::printf("layer types (5 sliding : 1 full, 60 layers)\n");
            int full = 0;
            for (int i = 0; i < 60; ++i) {
                const bool slide = !c.muse_sliding_attention.empty() &&
                                   c.muse_sliding_attention[size_t(i)];
                CHECK(slide == !is_full(i), "layer %d sliding flag", i);
                if (!slide) ++full;
                // Both map to the generic full-attention execution kind.
                CHECK(c.layer_types[size_t(i)] == LayerKind::FULL_ATTN,
                      "layer %d must not be linear attention", i);
            }
            EQ(full, 10, "full-attention layer count");
            EQ(c.sliding_window, 1024, "sliding_window");

            std::printf("rope, keyed by layer type\n");
            // cfg.rope_theta carries the SLIDING value; a whole-file scan
            // would have taken whichever came first in the file.
            CHECK(std::fabs(c.rope_theta - 10000.0f) < 1e-3f,
                  "sliding rope_theta: got %g want 10000", double(c.rope_theta));
            CHECK(std::fabs(c.global_rope_theta - 1000000.0f) < 1e-1f,
                  "full rope_theta: got %g want 1e6", double(c.global_rope_theta));
            CHECK(std::fabs(c.global_partial_rope - 0.25f) < 1e-6f,
                  "full partial_rotary_factor: got %g want 0.25",
                  double(c.global_partial_rope));
            CHECK(std::fabs(c.partial_rope - 1.0f) < 1e-6f,
                  "sliding layers rotate every dim");
            CHECK(c.global_rope_proportional,
                  "full-attention rope_type is proportional, which is NOT "
                  "the engine's partial_rope -- different frequencies and a "
                  "different dimension pairing");

            std::printf("scalars the reference sets and the config states\n");
            CHECK(c.k_eq_v, "attention_k_eq_v");
            CHECK(std::fabs(c.logit_softcap - 30.0f) < 1e-6f,
                  "final_logit_softcapping: got %g want 30",
                  double(c.logit_softcap));
            // ref/gemma4.py: self.scaling = 1.0, NOT 1/sqrt(head_dim).
            CHECK(std::fabs(c.attn_scale - 1.0f) < 1e-6f,
                  "attention scale is 1.0, not 1/sqrt(head_dim)");
            // Gemma4TextScaledWordEmbedding(embed_scale = hidden ** 0.5)
            CHECK(std::fabs(c.embed_scale - std::sqrt(5376.0f)) < 1e-2f,
                  "embed_scale: got %g want sqrt(5376)=%g",
                  double(c.embed_scale), std::sqrt(5376.0));
            // The activation must come from the TEXT block.  The vision
            // tower in this very config also says gelu_pytorch_tanh, so a
            // whole-file fallback cannot tell them apart -- and here they
            // agree, which is exactly why a wrong reading would pass
            // unnoticed on this checkpoint and fail on another.
            CHECK(c.geglu, "hidden_activation gelu_pytorch_tanh -> GeGLU");
        }
    }

    // ---- the weight map, and what it must enforce ---------------------
    // 12 layers keeps the map small and still covers two full-attention
    // positions (5 and 11).
    {
        const fs::path d = root / "weights";
        write_stub(d, config(12), tensor_names(12));
        Qwen35Model ld; std::string err;
        const bool ok = ld.load(d.string(), err, true, /*index_only=*/false);
        std::printf("weight map: v_proj on sliding layers only\n");
        CHECK(ok, "a correctly shaped checkpoint was rejected: %s", err.c_str());
        if (ok) {
            for (int i = 0; i < 12; ++i) {
                const auto& l = ld.layers[size_t(i)];
                CHECK(l.v_proj.ok() == !is_full(i),
                      "layer %d v_proj present=%d, expected %d", i,
                      int(l.v_proj.ok()), int(!is_full(i)));
                CHECK(l.layer_scalar.ok(), "layer %d layer_scalar", i);
                CHECK(l.pre_ff_norm.ok() && l.post_ff_norm.ok(),
                      "layer %d sandwich norms", i);
            }
        }
    }

    // ---- the refusals -------------------------------------------------
    // Each of these is a checkpoint that does not match its own config.
    // Loading it anyway would produce fluent text from the wrong model.
    struct Bad { const char* what; std::vector<std::string> names; };
    const std::vector<Bad> bad = {
        {"a full-attention layer that HAS a v_proj", tensor_names(12, -1, 5, -1)},
        {"a sliding layer with NO v_proj",           tensor_names(12, -1, -1, 4)},
        {"a layer with no layer_scalar",             tensor_names(12, 3, -1, -1)},
    };
    std::printf("refusals\n");
    for (size_t i = 0; i < bad.size(); ++i) {
        const fs::path d = root / ("bad" + std::to_string(i));
        write_stub(d, config(12), bad[i].names);
        Qwen35Model l; std::string e;
        const bool ok = l.load(d.string(), e, true, false);
        CHECK(!ok, "%s must be refused, but loaded", bad[i].what);
        if (!ok) std::printf("  refused: %s\n", bad[i].what);
    }

    // ---- config fields the REFERENCE reads and this engine does not ---
    // Each is absent from the 31B config, so none of them is exercised by
    // the checkpoint in hand -- which is exactly why they need a test: a
    // sibling that sets one would load here and be wrong with no symptom.
    {
        struct Raw { const char* what; std::string from, to; };
        const std::vector<Raw> raw = {
            {"bidirectional attention",
             "\"attention_k_eq_v\": true,",
             "\"attention_k_eq_v\": true,\n    "
             "\"use_bidirectional_attention\": \"all\","},
            {"an explicit per_layer_config",
             "\"attention_k_eq_v\": true,",
             "\"attention_k_eq_v\": true,\n    "
             "\"per_layer_config\": [{\"head_dim\": 128}],"},
        };
        for (size_t i = 0; i < raw.size(); ++i) {
            std::string c = config(12);
            const size_t at = c.find(raw[i].from);
            CHECK(at != std::string::npos, "%s: anchor not found in the config",
                  raw[i].what);
            if (at == std::string::npos) continue;
            c.replace(at, raw[i].from.size(), raw[i].to);
            const fs::path d = root / ("raw" + std::to_string(i));
            write_stub(d, c, tensor_names(12));
            Qwen35Model l; std::string e;
            const bool ok = l.load(d.string(), e, true, false);
            CHECK(!ok, "%s must be refused, but loaded", raw[i].what);
            if (!ok) std::printf("  refused: %s\n", raw[i].what);
        }
        // And the one that is IMPLEMENTED rather than refused: the
        // proportional-RoPE frequency divisor.  Absent means 1.0.
        {
            std::string c = config(12);
            const std::string from = "\"rope_type\": \"proportional\"";
            const size_t at = c.find(from);
            CHECK(at != std::string::npos, "rope factor: anchor not found");
            if (at != std::string::npos) {
                c.replace(at, from.size(), from + ", \"factor\": 8.0");
                const fs::path d = root / "ropefactor";
                write_stub(d, c, tensor_names(12));
                Qwen35Model l; std::string e;
                const bool ok = l.load(d.string(), e, true, true);
                CHECK(ok, "a rope factor must LOAD, not be refused: %s", e.c_str());
                if (ok) {
                    CHECK(std::fabs(l.cfg.global_rope_factor - 8.0f) < 1e-6f,
                          "rope factor parsed: got %f", double(l.cfg.global_rope_factor));
                    EQ(l.cfg.layer_rope_factor(5) == 8.0f, 1,
                       "a full layer carries the factor");
                    EQ(l.cfg.layer_rope_factor(0) == 1.0f, 1,
                       "a sliding layer does not");
                    std::printf("  parsed: proportional rope factor\n");
                }
            }
        }
    }

    // ---- features that are configured off here, and must be refused ---
    // The 31B is dense with no KV sharing and no per-layer inputs.  A
    // sibling that turns any of them on is a different forward pass, and
    // running it as this one would be silent.
    struct Uns { const char* what; Opts o; };
    Opts o_moe;    o_moe.moe    = "true";
    Opts o_kv;     o_kv.kvshare = "4";
    Opts o_per;    o_per.perlay = "256";
    Opts o_dw;     o_dw.dwmlp   = "true";
    const std::vector<Uns> unsupported = {
        {"MoE", o_moe}, {"KV sharing", o_kv},
        {"per-layer inputs", o_per}, {"double-wide MLP", o_dw},
    };
    for (const auto& u : unsupported) {
        const fs::path d = root / (std::string("uns-") + u.what);
        write_stub(d, config(12, u.o));
        Qwen35Model l; std::string e;
        const bool ok = l.load(d.string(), e, true, true);
        CHECK(!ok, "%s is not implemented and must be refused", u.what);
        if (!ok) std::printf("  refused: %s\n", u.what);
    }

    std::error_code ec; fs::remove_all(root, ec);
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
