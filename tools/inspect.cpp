// =====================================================================
//  inspect.cpp  --  read a HuggingFace model directory and report what
//  the loader will actually have to deal with.
//
//    b70-inspect /models/Ornith-1.0-35B-MXFP4
//
//  Prints the architecture, the quantization convention, and the exact
//  tensor names/shapes/dtypes of layer 0. Every quantizer (gptq,
//  auto-round, compressed-tensors) emits a different set of side
//  tensors, and guessing the convention produces silent garbage rather
//  than an error -- so this reads them off disk instead.
// =====================================================================
#include "b70/safetensors.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

using namespace b70;

static std::string shape_str(const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(s[i]);
    }
    return out + "]";
}

// Collapse a numeric index right after `key` to a single letter, so
// tensors that differ only by index fold into one line. Without this a
// 256-expert MoE prints 1500+ rows and the attention tensors are lost in
// the noise.
static std::string collapse(const std::string& n, const std::string& key,
                            const char* tag) {
    std::string out = n;
    size_t p = 0;
    while ((p = out.find(key, p)) != std::string::npos) {
        size_t a = p + key.size(), b = a;
        while (b < out.size() && isdigit(out[b])) ++b;
        if (b == a) { p = a; continue; }
        out = out.substr(0, a) + tag + out.substr(b);
        p = a + strlen(tag);
    }
    return out;
}

static std::string canon(const std::string& n) {
    std::string out = collapse(n, "layers.", "N");
    out = collapse(out, "experts.", "E");
    out = collapse(out, "blocks.", "B");
    return out;
}

static int layer_of(const std::string& n) {
    const std::string key = "layers.";
    const size_t p = n.find(key);
    if (p == std::string::npos) return -1;
    return std::atoi(n.c_str() + p + key.size());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: b70-inspect <model_dir>\n");
        return 1;
    }
    const std::string dir = argv[1];

    HFModel m;
    std::string err;
    if (!m.discover(dir, err)) {
        std::printf("ERROR: %s\n", err.c_str());
        return 1;
    }

    std::printf("=== %s ===\n", dir.c_str());
    const char* keys[] = {"architectures", "model_type", "num_hidden_layers",
                          "hidden_size", "num_attention_heads", "num_key_value_heads",
                          "head_dim", "intermediate_size", "vocab_size",
                          "rope_theta", "rms_norm_eps", "tie_word_embeddings",
                          "num_experts", "num_experts_per_tok", "moe_intermediate_size",
                          "shared_expert_intermediate_size"};
    for (const char* k : keys) {
        auto it = m.config.find(k);
        if (it != m.config.end())
            std::printf("  %-32s %s\n", k, it->second.c_str());
    }
    std::printf("  %-32s %zu\n", "shards", m.shards.size());

    // ---- scan every shard, but only keep layer 0 + non-layer tensors --
    std::map<std::string, STTensor> layer0;
    std::map<std::string, int> fold_count;
    const bool skip_vision = !(argc > 2 && std::string(argv[2]) == "--all");
    size_t vision_skipped = 0;
    double vision_bytes = 0;
    std::vector<std::string> globals;
    std::set<std::string> dtypes;
    double total_bytes = 0;
    int    max_layer = -1;
    size_t n_tensors = 0;

    for (const std::string& shard : m.shards) {
        SafeTensors st;
        if (!st.open(shard, err)) {
            std::printf("  !! %s\n", err.c_str());
            continue;
        }
        for (const auto& kv : st.tensors()) {
            ++n_tensors;
            const STTensor& t = kv.second;
            total_bytes += double(t.end - t.begin);
            dtypes.insert(st_dtype_name(t.dtype));
            // The vision tower is dead weight for a text-only run.
            if (skip_vision && t.name.find(".visual.") != std::string::npos) {
                ++vision_skipped;
                vision_bytes += double(t.end - t.begin);
                continue;
            }
            const int L = layer_of(t.name);
            max_layer = std::max(max_layer, L);
            if (L == 0) {
                const std::string c = canon(t.name);
                ++fold_count[c];
                layer0[c] = t;
            } else if (L < 0) {
                const std::string c = canon(t.name);
                if (!fold_count.count(c)) globals.push_back(c);
                ++fold_count[c];
            }
        }
    }

    std::printf("  %-32s %zu\n", "tensors", n_tensors);
    std::printf("  %-32s %d\n", "max layer index", max_layer);
    std::printf("  %-32s %.2f GiB\n", "total weight bytes", total_bytes / 1073741824.0);
    if (vision_skipped)
        std::printf("  %-32s %zu tensors, %.2f GiB (use --all to include)\n",
                    "vision tower SKIPPED", vision_skipped,
                    vision_bytes / 1073741824.0);
    std::printf("  %-32s ", "dtypes present");
    for (const std::string& d : dtypes) std::printf("%s ", d.c_str());
    std::printf("\n");

    // ---- infer the quantization convention from the side tensors -----
    bool has_qweight = false, has_qzeros = false, has_gidx = false;
    bool has_scale = false, has_packed = false, has_zero_point = false;
    bool has_wscale = false, has_gscale = false;
    for (const auto& kv : layer0) {
        const std::string& n = kv.first;
        if (n.find("qweight")      != std::string::npos) has_qweight = true;
        if (n.find("qzeros")       != std::string::npos) has_qzeros  = true;
        if (n.find("g_idx")        != std::string::npos) has_gidx    = true;
        if (n.find("weight_packed")!= std::string::npos) has_packed  = true;
        if (n.find("weight_scale") != std::string::npos) has_wscale  = true;
        if (n.find("weight_zero_point") != std::string::npos) has_zero_point = true;
        if (n.find("weight_global_scale") != std::string::npos) has_gscale = true;
        if (n.find("scales")       != std::string::npos) has_scale   = true;
    }
    std::printf("\n  detected packing: ");
    if (has_qweight && has_qzeros)      std::printf("GPTQ/AutoRound (qweight+qzeros%s)\n",
                                                    has_gidx ? "+g_idx" : "");
    else if (has_packed)                std::printf("compressed-tensors (weight_packed%s%s)\n",
                                                    has_wscale ? "+weight_scale" : "",
                                                    has_gscale ? "+global_scale" : "");
    else if (has_wscale)                std::printf("compressed-tensors FP8 (weight_scale)\n");
    else if (has_scale)                 std::printf("scales present, unrecognised layout\n");
    else                                std::printf("unquantized (plain weights)\n");
    (void)has_zero_point;

    // ---- layer 0, the thing the loader is written against ------------
    std::printf("\n  --- layer 0 tensors (%zu) ---\n", layer0.size());
    std::vector<std::string> names;
    for (const auto& kv : layer0) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    for (const std::string& n : names) {
        const STTensor& t = layer0[n];
        const int c = fold_count[n];
        char mult[24] = "";
        if (c > 1) std::snprintf(mult, sizeof mult, "  x%d", c);
        std::printf("  %-62s %-5s %-14s%s\n", n.c_str(),
                    st_dtype_name(t.dtype), shape_str(t.shape).c_str(), mult);
    }

    if (!globals.empty()) {
        std::printf("\n  --- non-layer tensors (first %zu) ---\n", globals.size());
        std::sort(globals.begin(), globals.end());
        for (const std::string& n : globals) {
            const int c = fold_count[n];
            if (c > 1) std::printf("  %-62s x%d\n", n.c_str(), c);
            else       std::printf("  %s\n", n.c_str());
        }
    }
    return 0;
}
