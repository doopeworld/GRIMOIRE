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

static std::string config(bool with_parallel_ffn, bool with_attn_gate = false) {
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
    "attn_output_gate": AGNES_GATE,
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
    std::string out = c.str();
    // Both unimplemented features are refusals, so the "everything else"
    // pass has to switch them off to reach the fields underneath.
    const std::string tok = "AGNES_GATE";
    const size_t g = out.find(tok);
    out.replace(g, tok.size(), with_attn_gate ? "true" : "false");
    return out;
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

    // ---- the real config, exactly as the checkpoint ships it ----------
    {
        const fs::path d = root / "real";
        write_stub(d, config(true, true));
        Qwen35Model ld; std::string err;
        const bool ok = ld.load(d.string(), err, /*skip_vision=*/true, /*index_only=*/true);
        std::printf("the real config loads\n");
        CHECK(ok, "the checkpoint's own config was rejected: %s", err.c_str());
        if (ok) {
            // parallel_ffn is FOLDED into the main FFN at upload -- two
            // SwiGLUs summed are one SwiGLU over the concatenated
            // intermediate -- so after loading the feed-forward width is
            // the sum, and every buffer sized from it has to agree.
            EQ(ld.cfg.parallel_ffn_inter, 2048, "parallel_ffn_intermediate_size");
            EQ(ld.cfg.dense_inter, 17408 + 2048, "folded FFN intermediate width");
        }
    }

    // ---- everything else, with that one key removed -------------------
    {   // attn_output_gate needs no special handling and must NOT refuse:
        // the gate rides in a double-width q_proj, and the engine detects
        // that from the tensor width and splits it per head already.
        const fs::path d = root / "gate";
        write_stub(d, config(false, true));
        Qwen35Model l; std::string e;
        const bool ok = l.load(d.string(), e, true, true);
        CHECK(ok, "attn_output_gate must load: the gate is in q_proj and the "
                  "engine detects it by width (%s)", e.c_str());
        CHECK(l.cfg.attn_out_gate, "attn_output_gate parsed");
    }

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
    // No parallel_ffn in this variant, so nothing is folded in.
    EQ(c.dense_inter, 17408, "intermediate_size (NOT the vision tower's 4304)");
    EQ(c.parallel_ffn_inter, 0, "no parallel_ffn in this variant");
    CHECK(!c.is_moe(), "Agnes-3.0-Flash is dense, not MoE");
    CHECK(!c.tie_embeddings, "tie_word_embeddings");

    std::printf("attention\n");
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

    // ---- the fold identity, in arithmetic ------------------------------
    // Everything above is names and numbers.  This is the claim the whole
    // parallel_ffn implementation rests on:
    //
    //   down(silu(gate x) * up x) + pdown(silu(pgate x) * pup x)
    //     == down'(silu(gate' x) * up' x)
    //
    // with gate' = [gate;pgate] and up' = [up;pup] stacked by ROW, and
    // down' = [down|pdown] joined by COLUMN.  It is true because a matmul
    // over a concatenated contraction dimension is the sum of the two
    // partial products -- but the layouts are easy to get backwards, and
    // getting them backwards yields a matrix of exactly the right shape
    // that computes something else.  So compute it both ways.
    {
        std::printf("parallel_ffn fold\n");
        const int H = 6, I = 4, PI = 3, IF = I + PI;
        auto rnd = [](int n, unsigned seed) {
            std::vector<float> v(size_t(n), 0.0f);
            unsigned s2 = seed;
            for (auto& x : v) { s2 = s2 * 1664525u + 1013904223u;
                                x = float(int(s2 >> 16) % 200 - 100) / 100.0f; }
            return v;
        };
        const auto x     = rnd(H, 1);
        const auto gate  = rnd(I  * H, 2),  up   = rnd(I  * H, 3);
        const auto pgate = rnd(PI * H, 4),  pup  = rnd(PI * H, 5);
        const auto down  = rnd(H  * I, 6),  pdown= rnd(H  * PI, 7);
        auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
        auto matvec = [](const std::vector<float>& m, const std::vector<float>& v,
                         int n, int k) {
            std::vector<float> o(size_t(n), 0.0f);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < k; ++j) o[size_t(i)] += m[size_t(i)*k+j] * v[size_t(j)];
            return o;
        };
        // reference: two independent SwiGLUs, summed
        std::vector<float> ref(size_t(H), 0.0f);
        {
            auto g = matvec(gate, x, I, H), u = matvec(up, x, I, H);
            std::vector<float> h(size_t(I), 0.0f);
            for (int i = 0; i < I; ++i) h[size_t(i)] = silu(g[size_t(i)]) * u[size_t(i)];
            auto a = matvec(down, h, H, I);
            auto pg = matvec(pgate, x, PI, H), pu = matvec(pup, x, PI, H);
            std::vector<float> ph(size_t(PI), 0.0f);
            for (int i = 0; i < PI; ++i) ph[size_t(i)] = silu(pg[size_t(i)]) * pu[size_t(i)];
            auto b = matvec(pdown, ph, H, PI);
            for (int i = 0; i < H; ++i) ref[size_t(i)] = a[size_t(i)] + b[size_t(i)];
        }
        // folded: gate_up rows in the order gate, pgate, up, pup -- which
        // is what the SwiGLU kernel's [all gate][all up] split needs --
        // and down joined per row along K.
        std::vector<float> gu;
        gu.insert(gu.end(), gate.begin(),  gate.end());
        gu.insert(gu.end(), pgate.begin(), pgate.end());
        gu.insert(gu.end(), up.begin(),    up.end());
        gu.insert(gu.end(), pup.begin(),   pup.end());
        std::vector<float> dn(size_t(H) * IF);
        for (int n = 0; n < H; ++n) {
            std::copy(down.begin()  + size_t(n)*I,  down.begin()  + size_t(n+1)*I,
                      dn.begin() + size_t(n)*IF);
            std::copy(pdown.begin() + size_t(n)*PI, pdown.begin() + size_t(n+1)*PI,
                      dn.begin() + size_t(n)*IF + I);
        }
        auto gv = matvec(gu, x, 2*IF, H);
        std::vector<float> h(size_t(IF), 0.0f);
        for (int i = 0; i < IF; ++i)
            h[size_t(i)] = silu(gv[size_t(i)]) * gv[size_t(IF + i)];
        const auto got = matvec(dn, h, H, IF);
        double worst = 0.0;
        for (int i = 0; i < H; ++i)
            worst = std::max(worst, double(std::fabs(got[size_t(i)] - ref[size_t(i)])));
        CHECK(worst < 1e-5, "folded FFN != two SwiGLUs summed: max|d| %.3e", worst);
        std::printf("  two SwiGLUs summed == one folded SwiGLU, max|d| %.2e\n", worst);

        // And prove the layout matters: the obvious-looking row order
        // gate, up, pgate, pup has the right SHAPE and the wrong answer.
        std::vector<float> wrong;
        wrong.insert(wrong.end(), gate.begin(),  gate.end());
        wrong.insert(wrong.end(), up.begin(),    up.end());
        wrong.insert(wrong.end(), pgate.begin(), pgate.end());
        wrong.insert(wrong.end(), pup.begin(),   pup.end());
        auto wv = matvec(wrong, x, 2*IF, H);
        std::vector<float> wh(size_t(IF), 0.0f);
        for (int i = 0; i < IF; ++i)
            wh[size_t(i)] = silu(wv[size_t(i)]) * wv[size_t(IF + i)];
        const auto bad = matvec(dn, wh, H, IF);
        double diff = 0.0;
        for (int i = 0; i < H; ++i)
            diff = std::max(diff, double(std::fabs(bad[size_t(i)] - ref[size_t(i)])));
        CHECK(diff > 1e-3, "the wrong row order should NOT agree, but did");
    }

    std::error_code ec; fs::remove_all(root, ec);
    if (g_fail) { std::printf("\nagnes-config: %d FAILURES\n", g_fail); return 1; }
    std::printf("\nALL PASS (0 failures)\n");
    return 0;
}
