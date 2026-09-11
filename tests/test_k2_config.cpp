// =====================================================================
//  test_k2_config.cpp -- parse the REAL K2-Horizon-MoVA-36B-A4B
//  config.json through the production loader and pin every field the
//  forward pass depends on.  The config text below is verbatim from the
//  checkpoint, so a loader change that silently drops a key fails here
//  rather than at 3am on the Tower.
// =====================================================================
#include "b70/qwen35.hpp"
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <unistd.h>

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
#define EQ(got, want, name) eq_((long long)(got), (long long)(want), name, __FILE__, __LINE__)

static const char* kConfig = R"JSON({
  "architectures": ["K2HorizonForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attention_gate_func": "softplus",
  "bos_token_id": 0,
  "decoder_sparse_step": 1,
  "dtype": "bfloat16",
  "eos_token_id": 1,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "intermediate_size": 6144,
  "layernorm_num_groups": 2,
  "max_position_embeddings": 524288,
  "mlp_only_layers": [0, 1, 2],
  "model_type": "k2_horizon",
  "moe_gate_bias": true,
  "moe_intermediate_size": 768,
  "mova_num_experts": 64,
  "mova_num_experts_per_tok": 4,
  "norm_topk_prob": true,
  "num_attention_heads": 32,
  "num_experts": 100,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 48,
  "num_key_value_heads": 8,
  "num_shared_experts": 1,
  "query_key_norm": false,
  "rms_norm_eps": 1e-06,
  "rope_head_dim": 128,
  "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
  "router_aux_loss_coef": 0.001,
  "router_scaling_factor": 2.5,
  "router_score_func": "sigmoid",
  "sliding_window": null,
  "tie_word_embeddings": false,
  "use_cache": true,
  "vocab_size": 250624
})JSON";

// one real tensor so the loader's index is non-empty
static void write_stub(const fs::path& dir, const char* cfg) {
    fs::create_directories(dir);
    std::ofstream(dir/"config.json") << cfg;
    const std::vector<bf16_t> vals(256*8, bf16_t{});
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
    std::printf("=== K2-Horizon config parse (real checkpoint config.json) ===\n\n");
    char tmpl[] = "/tmp/grimoire-k2cfg-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path dir = tmpl;
    write_stub(dir, kConfig);

    Qwen35Model ld; std::string err;
    if (!ld.load(dir.string(), err, true, /*index_only=*/true)) {
        std::printf("  FAIL loader rejected the config: %s\n", err.c_str());
        std::error_code e; fs::remove_all(dir, e);
        return 1;
    }
    const auto& c = ld.cfg;

    std::printf("K2 detection and core dims\n");
    CHECK(c.is_k2, "model_type k2_horizon not detected");
    EQ(c.hidden, 2560, "hidden");
    EQ(c.n_layers, 48, "n_layers");
    EQ(c.vocab, 250624, "vocab");
    EQ(c.n_heads, 32, "n_heads");
    EQ(c.n_kv_heads, 8, "n_kv_heads");
    EQ(c.head_dim, 128, "head_dim");
    EQ(c.dense_inter, 6144, "dense intermediate_size");
    CHECK(!c.tie_embeddings, "tie_word_embeddings must be false");
    CHECK(std::fabs(c.rope_theta - 1e7f) < 1.0f,
          "rope_theta %.1f: must come from rope_parameters", double(c.rope_theta));

    std::printf("\nMoE\n");
    EQ(c.n_experts, 100, "num_experts");
    EQ(c.top_k, 8, "num_experts_per_tok");
    EQ(c.moe_inter, 768, "moe_intermediate_size");
    EQ(c.n_shared_expert, 1, "num_shared_experts");
    // the config has NO shared_expert_intermediate_size key; the width is
    // moe_intermediate_size * num_shared_experts, and defaulting it to 0
    // would silently load a zero-width shared expert.
    EQ(c.shared_inter, 768, "derived shared expert width");

    std::printf("\nMoVA\n");
    EQ(c.mova_experts, 64, "mova_num_experts");
    EQ(c.mova_top_k, 4, "mova_num_experts_per_tok");

    std::printf("\nNew operators\n");
    EQ(c.norm_groups, 2, "layernorm_num_groups");
    EQ(c.attn_gate, 2, "attention_gate_func softplus == 2");
    CHECK(c.attn_out_gate, "softplus gate must set attn_out_gate");
    CHECK(c.router_sigmoid, "router_score_func sigmoid");
    CHECK(c.moe_gate_bias, "moe_gate_bias");
    CHECK(c.norm_topk_prob, "norm_topk_prob");
    CHECK(std::fabs(c.router_scale - 2.5f) < 1e-6f,
          "router_scaling_factor %.3f want 2.5", double(c.router_scale));
    CHECK(!c.query_key_norm, "query_key_norm must be false (no q_norm/k_norm)");
    EQ(c.rope_head_dim, 128, "rope_head_dim");

    std::printf("\nLayer plan\n");
    EQ(c.mlp_only_layers.size(), 3u, "mlp_only_layers count");
    if (c.mlp_only_layers.size() == 3) {
        EQ(c.mlp_only_layers[0], 0, "mlp_only[0]");
        EQ(c.mlp_only_layers[1], 1, "mlp_only[1]");
        EQ(c.mlp_only_layers[2], 2, "mlp_only[2]");
    }
    int dense = 0, sparse = 0;
    for (int i = 0; i < c.n_layers; ++i) (c.k2_sparse[size_t(i)] ? sparse : dense)++;
    std::printf("  %d dense, %d sparse\n", dense, sparse);
    EQ(dense, 3, "dense layers"); EQ(sparse, 45, "sparse layers");
    for (int i = 0; i < 3; ++i) CHECK(!c.k2_sparse[size_t(i)], "layer %d must be dense", i);
    CHECK(c.k2_sparse[3] && c.k2_sparse[47], "layers 3 and 47 must be sparse");

    // a rope_head_dim that differs from head_dim needs the split/interleave
    // path; the loader must refuse rather than quietly run plain rope.
    std::printf("\nRefusals\n");
    {
        std::string bad = kConfig;
        const size_t p = bad.find("\"rope_head_dim\": 128");
        if (p != std::string::npos) bad.replace(p, 20, "\"rope_head_dim\": 64 ");
        const fs::path d2 = dir / "split_rope";
        write_stub(d2, bad.c_str());
        Qwen35Model l2; std::string e2;
        const bool ok = l2.load(d2.string(), e2, true, true);
        std::printf("  rope_head_dim != head_dim -> %s\n", ok ? "ACCEPTED" : e2.c_str());
        CHECK(!ok, "must refuse a checkpoint needing the split rope path");
    }

    // Rule 4: a weight whose N is not a multiple of 256 must never be
    // packed for a W4A8 tile.  K2's two routers are exactly that shape.
    std::printf("\nQuantizer policy (small-N routers stay BF16)\n");
    struct { const char* name; bool keep; } pol[] = {
        {"model.layers.7.self_attn.v_router.weight",      true },  // [64,2560]
        {"model.layers.7.mlp.gate.weight",                true },  // [100,2560]
        {"model.embed_tokens.weight",                     true },
        {"lm_head.weight",                                true },
        {"model.layers.7.self_attn.v_experts.3.weight",   false},  // [1024,2560] -- quantize
        {"model.layers.7.self_attn.q_proj.weight",        false},
        {"model.layers.7.mlp.experts.3.down_proj.weight", false},
    };
    for (const auto& t : pol) {
        const bool got = keep_qwen_bf16(t.name);
        std::printf("  %-46s %s\n", t.name, got ? "BF16" : "quantize");
        CHECK(got == t.keep, "%s: policy says %s", t.name, got ? "BF16" : "quantize");
    }

    std::error_code e; fs::remove_all(dir, e);
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
