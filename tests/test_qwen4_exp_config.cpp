// =====================================================================
//  test_qwen4_exp_config.cpp -- parse a Qwen4-Exp (Qwen3.8-Flash-Next)
//  config through the production loader, and pin what the engine has to
//  know about it.
//
//  The config below follows ref/qwen4_exp_config.py -- the vLLM class,
//  extracted into ref/ so nobody guesses this architecture twice -- and
//  the published dimensions of Qwen3.8-Flash-Next: hidden 2560 over 48
//  layers, 3 of every 4 layers Gated DeltaNet and the 4th Qwen Sparse
//  Attention, MoE with 512 experts and one shared.
//
//  Qwen4ExpTextConfig INHERITS Qwen3NextConfig (ref:34).  That is the
//  whole reason this test exists: the base is a shape GRIMOIRE already
//  runs, so the DANGEROUS failure is not "it does not load" -- it is
//  loading the familiar part and silently skipping the three that are
//  new.  Each of those is fluent when faked:
//
//    QSA  running the sparse layers as dense attention changes what the
//         model attends to and nothing downstream notices
//    PLE  dropping the n-gram embedding removes an addition to the
//         residual stream; the model still produces text
//    HC   collapsing hc_count residual streams to one is a different
//         network that still has the right shapes
//
//  So this pins the PARSE, and asserts the engine REFUSES BY NAME.
//
//  Run:  ./bin/test_qwen4_exp_config
// =====================================================================
#include "b70/qwen35.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace b70;
namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("  FAIL: "); \
    std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
#define EQ(got, want, what) do { const auto g_=(got); const auto w_=(want); \
    if (g_ != w_) { ++fails; std::printf("  FAIL: %s: got %lld want %lld\n", \
    what, (long long)g_, (long long)w_); } } while (0)

// 48 layers: three linear_attention then one qwen_sparse_attention,
// twelve times over -- the published 12 x (3 x GDN + 1 x QSA) layout.
static std::string layer_types_array() {
    std::ostringstream c;
    c << "[";
    for (int i = 0; i < 48; ++i) {
        if (i) c << ", ";
        c << ((i % 4) == 3 ? "\"qwen_sparse_attention\"" : "\"linear_attention\"");
    }
    c << "]";
    return c.str();
}

static std::string config() {
    std::ostringstream c;
    c << R"JSON({
  "model_type": "qwen4_exp",
  "text_config": {
    "model_type": "qwen4_exp_text",
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "vocab_size": 151936,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "intermediate_size": 5120,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
    "partial_rotary_factor": 0.25,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "shared_expert_intermediate_size": 512,
    "moe_intermediate_size": 512,
    "linear_num_value_heads": 48,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "hc_count": 4,
    "hc_lowrank": 320,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ple_embed_dim": 2560,
    "ple_conv_kernel_size": 4,
    "ngram_vocab_size_base": 20000000,
    "ple_layer_ids": [2],
    "indexer_n_heads": 24,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 1,
    "output_gate_type": "sigmoid",
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0},
    "layer_types": )JSON" << layer_types_array() << R"JSON(
  }
})JSON";
    return c.str();
}

static void write_stub(const fs::path& dir, const std::string& cfg) {
    fs::create_directories(dir);
    std::ofstream(dir/"config.json") << cfg;
    const char* name = "model.embed_tokens.weight";
    std::ostringstream h;
    h << "{\"" << name << "\":{\"dtype\":\"BF16\",\"shape\":[2,2],"
      << "\"data_offsets\":[0,8]}}";
    std::string hs = h.str(); while (hs.size() % 8) hs += ' ';
    std::ofstream f(dir/"model.safetensors", std::ios::binary);
    const uint64_t n = hs.size();
    f.write((const char*)&n, 8); f << hs;
    const std::vector<uint16_t> zeros(4, 0);
    f.write((const char*)zeros.data(), 8);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== Qwen4-Exp (Qwen3.8-Flash-Next) config parse ===\n");
    std::printf("reference: ref/qwen4_exp_config.py (vLLM), which inherits\n");
    std::printf("Qwen3NextConfig -- the base IS supported, the rest is not.\n\n");

    char tmpl[] = "/tmp/grimoire-q4x-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;

    const fs::path d = root / "q4x";
    write_stub(d, config());
    Qwen35Model ld; std::string err;
    const bool ok = ld.load(d.string(), err, /*skip_vision=*/true,
                            /*index_only=*/true);
    CHECK(ok, "the config was rejected by the loader: %s", err.c_str());
    if (!ok) { std::printf("\nFAILED (%d)\n", fails ? fails : 1); return 1; }

    const auto& c = ld.cfg;
    std::printf("the architecture is recognised\n");
    CHECK(c.is_qwen4_exp, "model_type qwen4_exp was not detected");

    // The Qwen3-Next base, which this engine already runs.  Pinned so a
    // regression here is not mistaken for a Qwen4-Exp-specific problem.
    std::printf("the Qwen3-Next base parses\n");
    EQ(c.hidden, 2560, "hidden_size");
    EQ(c.n_layers, 48, "num_hidden_layers");
    EQ(c.n_heads, 24, "num_attention_heads");
    EQ(c.n_kv_heads, 2, "num_key_value_heads");
    EQ(c.head_dim, 256, "head_dim");
    EQ(c.n_experts, 512, "num_experts");
    EQ(c.top_k, 10, "num_experts_per_tok");

    // 3 of every 4 layers are Gated DeltaNet.  If this drifts, the model
    // runs 36 recurrent layers as attention and stays fluent.
    std::printf("the 12 x (3 x GDN + 1 x QSA) layout parses\n");
    int linear = 0, attn = 0, qsa = 0;
    for (int i = 0; i < c.n_layers; ++i) {
        if (c.layer_types[size_t(i)] == LayerKind::LINEAR_ATTN) ++linear;
        else ++attn;
        if (!c.qsa_attention.empty() && c.qsa_attention[size_t(i)]) ++qsa;
    }
    EQ(linear, 36, "Gated DeltaNet layers");
    EQ(attn, 12, "attention layers");
    // Every attention layer in this checkpoint is a QSA layer, and the
    // loader must say so per layer -- "some attention layers" is not
    // enough to refuse on, and not enough to implement against later.
    EQ(qsa, 12, "layers marked qwen_sparse_attention");

    std::printf("the QSA indexer fields parse\n");
    EQ(c.indexer_n_heads, 24, "indexer_n_heads");
    EQ(c.indexer_kv_heads, 1, "indexer_kv_heads");
    EQ(c.indexer_head_dim, 128, "indexer_head_dim");
    EQ(c.indexer_budget, 2048, "indexer_budget");
    EQ(c.indexer_compress_ratio, 1, "indexer_compress_ratio");
    // ref/qwen4_exp_config.py:_validate_qsa_config -- the MQA operators
    // assume one KV head, and budget/ratio must land on 512 or 2048.
    CHECK(c.indexer_kv_heads == 1, "the QSA MQA operators require indexer_kv_heads=1");
    const int block_topk = c.indexer_compress_ratio
                         ? c.indexer_budget / c.indexer_compress_ratio : 0;
    CHECK(block_topk == 512 || block_topk == 2048,
          "indexer_budget/indexer_compress_ratio must be 512 or 2048, got %d",
          block_topk);

    std::printf("the PLE n-gram fields parse\n");
    EQ(c.ngram_size, 3, "ngram_size");
    EQ(c.heads_per_ngram, 8, "heads_per_ngram");
    EQ(c.ple_embed_dim, 2560, "ple_embed_dim");
    EQ(c.ple_conv_kernel, 4, "ple_conv_kernel_size");
    EQ(c.ngram_vocab_base, 20000000LL, "ngram_vocab_size_base");
    EQ((int)c.ple_layer_ids.size(), 1, "ple_layer_ids count");
    // 1-BASED in the config (ref:127).  Storing it converted would make
    // this assertion pass while the value meant something else.
    if (!c.ple_layer_ids.empty())
        EQ(c.ple_layer_ids[0], 2, "ple_layer_ids[0] (1-based, verbatim)");
    // ref:_validate_ple_config -- ple_embed_dim must divide by the total
    // ngram heads, or the per-head split is ragged.
    const int ngram_heads = (c.ngram_size - 1) * c.heads_per_ngram;
    CHECK(ngram_heads > 0 && c.ple_embed_dim % ngram_heads == 0,
          "ple_embed_dim %d must be divisible by ngram heads %d",
          c.ple_embed_dim, ngram_heads);

    std::printf("the HyperConnection fields parse\n");
    EQ(c.hc_count, 4, "hc_count");
    EQ(c.hc_lowrank, 320, "hc_lowrank");

    // The REFUSAL is asserted in bin/test_model_matrix, not here: this
    // file is host-only and builds in seconds with g++, while
    // Grimoire::unsupported_reason() needs the SYCL engine.  Splitting
    // them the way gemma-4 does keeps the parse fast to iterate on --
    // and the parse is what goes out of date when a config changes.

    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASS",
                fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
