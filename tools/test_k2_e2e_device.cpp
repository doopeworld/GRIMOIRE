// =====================================================================
//  test_k2_e2e_device.cpp -- drive the WHOLE K2 engine path on a device.
//
//  tests/test_k2_config.cpp proves the loader resolves a K2 checkpoint.
//  bin/test_k2_kernels proves each new kernel matches its reference.
//  Neither one runs the engine, so until this existed nothing had ever
//  executed Grimoire's K2 path: MoVA's routed value projection, the
//  softplus attention gate, the grouped norms and the sigmoid router all
//  sat in a forward() that had never been called.
//
//  So: write a miniature K2 checkpoint with RANDOM weights (zeros would
//  make every path look finite and equal), load it through the real
//  public entry points, and generate.  The weights are noise, so the
//  tokens mean nothing -- what is being checked is that the path runs,
//  stays finite, stays in vocabulary, and is reproducible.  Per rule 8
//  this is not a substitute for reading real output from a real model on
//  the B70; it is the floor below which that is not worth attempting.
//
//  Build:
//    icpx -fsycl -fsycl-targets=spir64 -O2 -std=c++20 -I include -I src \
//      tools/test_k2_e2e_device.cpp src/grimoire.cpp src/qwen35_loader.cpp \
//      src/native_model.cpp src/safetensors.cpp src/quantize.cpp src/gptq.cpp \
//      src/gemv_decode.cpp src/gemm_xmx.cpp src/attention.cpp src/deltanet.cpp \
//      src/moe_kernels.cpp src/moe_ref.cpp src/ops.cpp src/prefill.cpp \
//      src/tokenizer.cpp -o bin/test_k2_e2e
// =====================================================================
#include "b70/grimoire_api.hpp"
#include "b70/formats.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace b70;

static int g_fail = 0;
#define CHECK(c, ...) do { if(!(c)){ std::printf("  FAIL "); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while(0)

// Same geometry as tests/test_k2_config.cpp's miniature checkpoint: 4
// layers, 0-1 dense and 2-3 sparse, so one run covers BOTH shapes --
// ordinary v_proj next to MoVA, and a plain MLP next to the routed MoE.
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
        t.push_back({a+"gate_proj.weight", {Q,H}});
        if (sparse) {
            t.push_back({a+"v_router.weight", {MV,H}});
            t.push_back({a+"v_router.bias", {MV}});
            for (int e = 0; e < MV; ++e)
                t.push_back({a+"v_experts."+std::to_string(e)+".weight", {KV,H}});
        } else {
            t.push_back({a+"v_proj.weight", {KV,H}});
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

// Random BF16, small enough that four layers of residual cannot blow up
// but never zero: a zeroed checkpoint makes a dead path indistinguishable
// from a live one.
static void write_model(const fs::path& dir) {
    fs::create_directories(dir);
    std::ofstream(dir/"config.json") << kMiniConfig;
    const auto ts = mini_tensors();
    std::ostringstream h; h << '{';
    uint64_t off = 0;
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

    std::mt19937 rng(20260912);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<bf16_t> data(off/2);
    for (auto& v : data) v = f32_to_bf16(nd(rng));

    std::ofstream f(dir/"model.safetensors", std::ios::binary);
    const uint64_t n = hs.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f << hs;
    f.write(reinterpret_cast<const char*>(data.data()),
            std::streamsize(data.size()*sizeof(bf16_t)));
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== K2 engine, end to end on a device ===\n\n");

    char tmpl[] = "/tmp/grimoire-k2e2e-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path dir = tmpl;
    write_model(dir);
    std::printf("miniature checkpoint: %s\n", dir.string().c_str());

    // BF16 projections: the point here is the K2 operator path, not the
    // quantizer, and BF16 keeps the comparison free of quantization noise.
    const int max_seq = 128;
    std::string err;
    Grimoire* e = grimoire_new();
    if (!e) { std::printf("  FAIL grimoire_new returned null\n"); return 1; }
    if (!grimoire_load(*e, dir.string(), Fmt::BF16, max_seq, err)) {
        std::printf("\n  FAIL load: %s\n", err.c_str());
        grimoire_delete(e);
        fs::remove_all(dir);
        return 1;
    }
    std::printf("\nloaded.\n");

    const std::vector<int32_t> prompt{7, 11, 3, 42, 5, 90, 1, 64};
    const int want = 12;

    std::vector<int32_t> a, b;
    FinishReason ra{}, rb{};
    int na = 0, nb = 0;
    try {
        na = grimoire_serve_generate(*e, prompt, want, -1, a, -1, {}, &ra);
        nb = grimoire_serve_generate(*e, prompt, want, -1, b, -1, {}, &rb);
    } catch (const std::exception& ex) {
        std::printf("\n  FAIL generate threw: %s\n", ex.what());
        grimoire_delete(e);
        fs::remove_all(dir);
        return 1;
    }

    std::printf("\ngenerated %d tokens:", na);
    for (int t : a) std::printf(" %d", t);
    std::printf("\n");

    CHECK(na == want, "asked for %d tokens, got %d", want, na);
    for (int t : a) CHECK(t >= 0 && t < 128, "token %d is outside the vocabulary", t);

    // Same prompt twice must give the same tokens.  Greedy decode is
    // deterministic; anything else here means state survived reset() --
    // the exact failure mode reset()'s own comment describes.
    CHECK(nb == na && a == b, "two identical prompts produced different output "
                              "-- state leaked across requests");
    std::printf("reproducible across requests: %s\n", (a == b) ? "yes" : "NO");

    // A longer prompt exercises the batched prefill path (the grouped norm
    // batched kernel, the batched router) rather than the M=1 decode path.
    std::vector<int32_t> longer;
    for (int i = 0; i < 48; ++i) longer.push_back(int32_t((i * 17 + 3) % 128));
    std::vector<int32_t> c;
    try {
        const int nc = grimoire_serve_generate(*e, longer, 4, -1, c, -1, {}, &ra);
        CHECK(nc == 4, "48-token prompt: asked for 4, got %d", nc);
        for (int t : c) CHECK(t >= 0 && t < 128, "long prompt produced token %d", t);
        // Say which path actually ran.  Without matrix hardware the engine
        // refuses the batched path and the caller retries sequentially, so
        // calling this "batched prefill verified" would be a lie.
        const bool xmx = sycl::device{sycl::default_selector_v}
                             .has(sycl::aspect::ext_intel_matrix);
        std::printf("48-token prompt: ok (%d tokens) -- %s prefill\n", nc,
                    xmx ? "BATCHED" : "sequential (device has no matrix hardware, "
                                      "the batched path was NOT exercised)");
    } catch (const std::exception& ex) {
        std::printf("  FAIL long prompt threw: %s\n", ex.what());
        ++g_fail;
    }

    grimoire_delete(e);
    fs::remove_all(dir);
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
