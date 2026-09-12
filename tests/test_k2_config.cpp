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
#include <string>
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


// ---------------------------------------------------------------------
// A complete miniature K2 checkpoint: 4 layers, 0-1 dense and 2-3 sparse,
// with the tensor names taken from the module paths in
// modeling_k2_horizon.py.  Resolving every one of these is what proves
// the loader's naming matches the checkpoint's.
// ---------------------------------------------------------------------
static const char* kMiniConfig = R"JSON({
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
})JSON";

struct Tn { std::string name; std::vector<int64_t> shape; };

static void write_model(const fs::path& dir, const char* cfg,
                        const std::vector<Tn>& ts) {
    fs::create_directories(dir);
    std::ofstream(dir/"config.json") << cfg;
    std::ostringstream h; h << '{'; uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        if (i) h << ',';
        size_t n = 1; for (auto d : ts[i].shape) n *= size_t(d);
        h << '"' << ts[i].name << "\":{\"dtype\":\"BF16\",\"shape\":[";
        for (size_t d = 0; d < ts[i].shape.size(); ++d) {
            if (d) h << ',';
            h << ts[i].shape[d];
        }
        h << "],\"data_offsets\":[" << off << ','; off += n*2; h << off << "]}";
    }
    h << '}';
    std::string hs = h.str(); while (hs.size() % 8) hs += ' ';
    std::ofstream f(dir/"model.safetensors", std::ios::binary);
    const uint64_t n = hs.size();
    f.write((const char*)&n, 8); f << hs;
    std::vector<bf16_t> zero(off/2, bf16_t{});
    f.write((const char*)zero.data(), std::streamsize(off));
}

static std::vector<Tn> mini_tensors() {
    const int H=64, KV=32, Q=64, I=48, MI=32, E=4, MV=3, V=128;
    std::vector<Tn> t = {
        {"model.embed_tokens.weight", {V,H}}, {"model.norm.weight", {H}},
        {"lm_head.weight", {V,H}},
    };
    for (int L = 0; L < 4; ++L) {
        const std::string b = "model.layers." + std::to_string(L) + ".";
        const bool sparse = (L >= 2);
        t.push_back({b+"input_layernorm.weight", {H}});
        t.push_back({b+"post_attention_layernorm.weight", {H}});
        const std::string a = b + "self_attn.";
        t.push_back({a+"q_proj.weight", {Q,H}});
        t.push_back({a+"k_proj.weight", {KV,H}});
        t.push_back({a+"o_proj.weight", {H,Q}});
        t.push_back({a+"gate_proj.weight", {Q,H}});     // softplus attn gate
        if (sparse) {
            t.push_back({a+"v_router.weight", {MV,H}});
            t.push_back({a+"v_router.bias", {MV}});
            for (int e = 0; e < MV; ++e)
                t.push_back({a+"v_experts."+std::to_string(e)+".weight", {KV,H}});
        } else {
            t.push_back({a+"v_proj.weight", {KV,H}});   // dense layers keep v_proj
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
                t.push_back({m+"shared_experts."+n+".weight",
                             std::string(n)=="down_proj" ? std::vector<int64_t>{H,MI}
                                                         : std::vector<int64_t>{MI,H}});
        } else {
            t.push_back({m+"gate_proj.weight", {I,H}});
            t.push_back({m+"up_proj.weight",   {I,H}});
            t.push_back({m+"down_proj.weight", {H,I}});
        }
    }
    return t;
}

static void test_tensor_mapping(const fs::path& root) {
    std::printf("\nTensor mapping (miniature K2 checkpoint)\n");
    const fs::path d = root / "mini";
    write_model(d, kMiniConfig, mini_tensors());
    Qwen35Model M; std::string e;
    if (!M.load(d.string(), e, true, false)) {
        std::printf("  FAIL loader rejected the mini model: %s\n", e.c_str());
        ++g_fail; return;
    }
    CHECK(M.embed.ok() && M.lm_head.ok() && M.final_norm.ok(), "top-level tensors");
    int missing = 0;
    for (int L = 0; L < 4; ++L) {
        const auto& l = M.layers[size_t(L)];
        const bool sparse = (L >= 2);
        EQ(l.k2_sparse, sparse, "k2_sparse flag");
        if (!l.input_norm.ok() || !l.post_attn_norm.ok()) ++missing;
        if (!l.q_proj.ok() || !l.k_proj.ok() || !l.o_proj.ok()) ++missing;
        CHECK(l.attn_gate.ok(), "layer %d: softplus attn gate_proj missing", L);
        if (sparse) {
            CHECK(!l.v_proj.ok(), "layer %d: sparse layer must NOT have v_proj", L);
            CHECK(l.v_router.ok(), "layer %d: v_router missing", L);
            CHECK(l.v_router_bias.ok(), "layer %d: v_router.bias missing", L);
            EQ(l.v_experts.size(), 3u, "MoVA expert count");
            for (size_t x = 0; x < l.v_experts.size(); ++x)
                CHECK(l.v_experts[x].ok(), "layer %d: v_experts.%zu missing", L, x);
            CHECK(l.router.ok(), "layer %d: mlp.gate missing", L);
            CHECK(l.router_bias.ok(), "layer %d: mlp.gate.bias missing", L);
            CHECK(l.sh_gate.ok() && l.sh_up.ok() && l.sh_down.ok(),
                  "layer %d: shared_experts (plural) missing", L);
            for (int x = 0; x < 4; ++x)
                CHECK(l.e_gate_p[size_t(x)].ok() && l.e_up_p[size_t(x)].ok() &&
                      l.e_down_p[size_t(x)].ok(), "layer %d: expert %d missing", L, x);
        } else {
            CHECK(l.v_proj.ok(), "layer %d: dense layer needs v_proj", L);
            CHECK(l.v_experts.empty(), "layer %d: dense layer must have no MoVA", L);
            CHECK(!l.router.ok(), "layer %d: dense layer must have no router", L);
            CHECK(l.sh_gate.ok() && l.sh_up.ok() && l.sh_down.ok(),
                  "layer %d: dense mlp.{gate,up,down}_proj missing", L);
        }
    }
    EQ(missing, 0, "core per-layer tensors missing");
    std::printf("  4 layers resolved: 2 dense (v_proj + plain MLP), "
                "2 sparse (MoVA + routed MoE)\n");
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

    test_tensor_mapping(dir);

    std::error_code e; fs::remove_all(dir, e);
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
