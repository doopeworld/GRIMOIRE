// =====================================================================
//  qwen35_loader.cpp
// =====================================================================
#include "b70/qwen35.hpp"
#include "b70/k2_horizon.hpp"
#include <cctype>
#include "b70/tensor_layout.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

namespace b70 {
namespace {

// config.json flattens text_config into the top level in HFModel::discover
// only for scalars it recognises, so pull nested values by hand.
std::string read_file(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    std::string out;
    char buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

// Minimal scalar extraction: find "key" then the next number/string/bool.
// The safetensors header parser is strict; config.json only needs lookups.
bool find_scalar(const std::string& j, const std::string& key, std::string& out) {
    const std::string pat = "\"" + key + "\"";
    size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    ++p;
    while (p < j.size() && isspace((unsigned char)j[p])) ++p;
    if (p >= j.size()) return false;
    if (j[p] == '"') {
        const size_t e = j.find('"', p + 1);
        if (e == std::string::npos) return false;
        out = j.substr(p + 1, e - p - 1);
        return true;
    }
    size_t e = p;
    while (e < j.size() && j[e] != ',' && j[e] != '}' && j[e] != ']' &&
           !isspace((unsigned char)j[e])) ++e;
    out = j.substr(p, e - p);
    return true;
}

int    cfg_i(const std::string& j, const char* k, int d) {
    std::string v; return find_scalar(j, k, v) ? std::atoi(v.c_str()) : d;
}
float  cfg_f(const std::string& j, const char* k, float d) {
    std::string v; return find_scalar(j, k, v) ? float(std::atof(v.c_str())) : d;
}
bool   cfg_b(const std::string& j, const char* k, bool d) {
    std::string v;
    if (!find_scalar(j, k, v)) return d;
    return v == "true" || v == "1";
}

} // namespace

bool keep_qwen_bf16(const std::string& name) {
    auto ends=[&](const char* suffix){const size_t n=std::strlen(suffix);
        return name.size()>=n && name.compare(name.size()-n,n,suffix)==0;};
    // GRIMOIRE_QUANT_LM_HEAD=1 lets the output projection be packed.  It is
    // the single largest per-token read that is not an expert -- 1.017 GB at
    // BF16 for a 248320x2048 head, against 0.266 at int4, and test_ops puts
    // that at ~1.6 ms per token, more than the whole fused MoE block.  It is
    // read once per plain decode step and once per DRAFT step on top, so a
    // k=6 speculative round pays it several times over.
    //
    // Off by default because it was never measured here, not because it was
    // measured and rejected.  An independent B70 report (Qwen3.8-27B, GPTQ
    // int4 + MTP) quantized this head and found perplexity and GSM8K
    // largely unchanged -- worth reproducing on our own benchmark before
    // making it the default.
    static const bool quant_lm_head = [] {
        const char* e = std::getenv("GRIMOIRE_QUANT_LM_HEAD");
        return e && *e && std::atoi(e) != 0;
    }();
    const bool is_lm_head = name=="lm_head.weight" || ends(".lm_head.weight");
    if (is_lm_head && quant_lm_head) return false;
    return name.rfind("mtp.",0)==0 || name.find("embed_tokens")!=std::string::npos ||
        is_lm_head ||
        ends(".linear_attn.in_proj_a.weight") || ends(".linear_attn.in_proj_b.weight") ||
        ends(".mlp.gate.weight") || ends(".mlp.shared_expert_gate.weight") ||
        // K2-Horizon MoVA value router: [mova_num_experts, hidden] = [64,2560].
        // N=64 is not a multiple of 256, so it must never reach a W4A8 tile
        // (the B 2-D block loads do not clamp), and a router is the last
        // place to spend precision -- it is 0.16 MB of a 36B model.
        ends(".self_attn.v_router.weight");
}

bool Qwen35Model::load(const std::string& d, std::string& err, bool skip_vision,
                       bool index_only) {
    dir = d;
    const std::string j = read_file(dir + "/config.json");
    if (j.empty()) { err = "cannot read " + dir + "/config.json"; return false; }

    // ---- config -----------------------------------------------------
    // Muse Glimmer nests the text dims under "text_config"; parse from there.
    std::string model_type; find_scalar(j, "model_type", model_type);
    const bool muse = model_type.find("muse_glimmer") != std::string::npos
                   || j.find("muse_glimmer_text") != std::string::npos;
    std::string tj = j;
    if (muse) {
        size_t tp = j.find("\"text_config\"");
        if (tp != std::string::npos) {
            size_t b = j.find("{", tp); int depth = 0; size_t e = b;
            for (; e < j.size(); ++e) { if (j[e]=='{') ++depth; else if (j[e]=='}') { if(--depth==0){++e;break;} } }
            tj = j.substr(b, e - b);
        }
    }
    const std::string& cj = muse ? tj : j;
    cfg.hidden        = cfg_i(cj, "hidden_size", 0);
    cfg.n_layers      = cfg_i(cj, "num_hidden_layers", 0);
    cfg.vocab         = cfg_i(cj, "vocab_size", 0);
    cfg.n_heads       = cfg_i(cj, "num_attention_heads", 0);
    cfg.n_kv_heads    = cfg_i(cj, "num_key_value_heads", 0);
    cfg.head_dim      = cfg_i(cj, "head_dim", 0);
    cfg.rms_eps       = cfg_f(cj, "rms_norm_eps", 1e-6f);
    cfg.post_norm_eps = cfg_f(cj, "post_norm_eps", cfg.rms_eps);
    cfg.rope_theta    = cfg_f(cj, "rope_theta", 1e7f);
    cfg.partial_rope  = cfg_f(j, "partial_rotary_factor", 1.0f);
    cfg.attn_out_gate = cfg_b(j, "attn_output_gate", false);
    cfg.tie_embeddings= cfg_b(j, "tie_word_embeddings", false);

    cfg.lin_k_heads   = cfg_i(j, "linear_num_key_heads", 0);
    cfg.lin_v_heads   = cfg_i(j, "linear_num_value_heads", 0);
    cfg.lin_k_dim     = cfg_i(j, "linear_key_head_dim", 0);
    cfg.lin_v_dim     = cfg_i(j, "linear_value_head_dim", 0);
    cfg.conv_kernel   = cfg_i(j, "linear_conv_kernel_dim", 0);

    cfg.n_experts     = cfg_i(j, "num_experts", 0);
    cfg.top_k         = cfg_i(j, "num_experts_per_tok", 0);
    cfg.moe_inter     = cfg_i(j, "moe_intermediate_size", 0);
    cfg.shared_inter  = cfg_i(j, "shared_expert_intermediate_size", 0);
    cfg.dense_inter   = cfg_i(cj, "intermediate_size", 0);

    if (muse) {
        cfg.is_muse = true;
        cfg.attn_out_gate = true;
        float qk = cfg_f(cj, "qk_scale_factor", 0.0f);
        float ex = cfg_f(cj, "scale_query_by", 0.0f);
        cfg.query_prescale = ex > 0.0f ? ex : (qk > 0.0f ? qk : 1.0f);
    }
    // ---- K2-Horizon --------------------------------------------------
    // model_type "k2_horizon".  rope_head_dim == head_dim here, so the
    // split/interleave rope path in the reference never runs; refuse a
    // checkpoint that would need it rather than silently using plain rope.
    cfg.is_k2 = model_type.find("k2_horizon") != std::string::npos;
    if (cfg.is_k2) {
        cfg.norm_groups     = cfg_i(cj, "layernorm_num_groups", 1);
        cfg.mova_experts    = cfg_i(j, "mova_num_experts", 0);
        cfg.mova_top_k      = cfg_i(j, "mova_num_experts_per_tok", 0);
        cfg.moe_gate_bias   = cfg_b(j, "moe_gate_bias", false);
        cfg.norm_topk_prob  = cfg_b(j, "norm_topk_prob", true);
        cfg.router_scale    = cfg_f(j, "router_scaling_factor", 1.0f);
        cfg.rope_head_dim   = cfg_i(cj, "rope_head_dim", 0);
        cfg.sparse_step     = cfg_i(j, "decoder_sparse_step", 1);
        cfg.n_shared_expert = cfg_i(j, "num_shared_experts", 0);
        cfg.query_key_norm  = cfg_b(j, "query_key_norm", false);
        cfg.rope_theta      = cfg_f(j, "rope_theta", cfg.rope_theta);

        std::string score; find_scalar(j, "router_score_func", score);
        cfg.router_sigmoid = score.find("sigmoid") != std::string::npos;

        std::string gate; find_scalar(j, "attention_gate_func", gate);
        cfg.attn_gate = gate.find("softplus") != std::string::npos ? 2
                      : gate.find("silu")     != std::string::npos ? 1 : 0;
        cfg.attn_out_gate = cfg.attn_gate != 0;

        // rope_theta lives under "rope_parameters" in this config, not at
        // the top level where cfg_f above looked for it.
        {
            const size_t rp = j.find("\"rope_parameters\"");
            if (rp != std::string::npos) {
                const size_t b = j.find('{', rp), e = j.find('}', b);
                if (b != std::string::npos && e != std::string::npos)
                    cfg.rope_theta = cfg_f(j.substr(b, e - b), "rope_theta", cfg.rope_theta);
            }
        }

        // The shared expert is one MLP of moe_intermediate_size *
        // num_shared_experts.  The config has no shared_expert_intermediate_size
        // key, so without this the shared expert would load as width 0.
        if (cfg.n_shared_expert > 0 && cfg.shared_inter == 0)
            cfg.shared_inter = cfg.moe_inter * cfg.n_shared_expert;

        // mlp_only_layers is a plain int array
        cfg.mlp_only_layers.clear();
        {
            const size_t p = j.find("\"mlp_only_layers\"");
            if (p != std::string::npos) {
                const size_t a = j.find('[', p), b = j.find(']', a);
                if (a != std::string::npos && b != std::string::npos) {
                    const std::string arr = j.substr(a + 1, b - a - 1);
                    size_t q = 0;
                    while (q < arr.size()) {
                        while (q < arr.size() && !std::isdigit(static_cast<unsigned char>(arr[q]))) ++q;
                        if (q >= arr.size()) break;
                        size_t e2 = q;
                        while (e2 < arr.size() && std::isdigit(static_cast<unsigned char>(arr[e2]))) ++e2;
                        cfg.mlp_only_layers.push_back(std::atoi(arr.substr(q, e2 - q).c_str()));
                        q = e2;
                    }
                }
            }
        }
    }

    if (cfg.hidden <= 0 || cfg.n_layers <= 0) {
        err = "config.json missing hidden_size or num_hidden_layers"; return false;
    }
    if (cfg.is_k2) {
        if (cfg.rope_head_dim && cfg.head_dim && cfg.rope_head_dim != cfg.head_dim) {
            err = "k2_horizon: rope_head_dim != head_dim needs the split/interleave "
                  "rope path, which is not implemented"; return false;
        }
        if (cfg.norm_groups <= 0 || cfg.hidden % cfg.norm_groups) {
            err = "k2_horizon: hidden_size is not divisible by layernorm_num_groups";
            return false;
        }
    }

    // ---- layer_types -------------------------------------------------
    cfg.layer_types.assign(cfg.n_layers, LayerKind::FULL_ATTN);
    cfg.muse_sliding_attention.assign(cfg.n_layers, false);
    {
        const size_t p = j.find("\"layer_types\"");
        if (p != std::string::npos) {
            const size_t a = j.find('[', p), b = j.find(']', a);
            if (a != std::string::npos && b != std::string::npos) {
                const std::string arr = j.substr(a, b - a);
                size_t q = 0;
                int i = 0;
                while (i < cfg.n_layers) {
                    const size_t s = arr.find('"', q);
                    if (s == std::string::npos) break;
                    const size_t e = arr.find('"', s + 1);
                    if (e == std::string::npos) break;
                    const std::string v = arr.substr(s + 1, e - s - 1);
                    cfg.layer_types[i] = (v == "linear_attention")
                                      ? LayerKind::LINEAR_ATTN : LayerKind::FULL_ATTN;
                    cfg.muse_sliding_attention[i] =
                        (v == "sliding_attention");
                    ++i;
                    q = e + 1;
                }
            }
        }
    }

    // ---- K2 per-layer sparsity ---------------------------------------
    // A sparse layer runs MoVA attention and the routed MoE; a dense one
    // runs plain attention and a single intermediate_size MLP.  Both
    // shapes coexist (layers 0-2 dense, 3-47 sparse in the 36B-A4B), so
    // nothing downstream may assume one kind.
    cfg.k2_sparse.assign(cfg.n_layers, false);
    if (cfg.is_k2) {
        for (int i = 0; i < cfg.n_layers; ++i)
            cfg.k2_sparse[size_t(i)] = k2::is_sparse_layer(
                i, cfg.mlp_only_layers.empty() ? nullptr : cfg.mlp_only_layers.data(),
                int(cfg.mlp_only_layers.size()), cfg.n_experts, cfg.sparse_step);
    }

    // ---- shards ------------------------------------------------------
    shards.clear();
    index.clear();
    native_model.reset();
    std::string native_path=dir+"/model-v3.b70";
    if (::access(native_path.c_str(),F_OK)!=0) native_path=dir+"/model-v2.b70";
    if (::access(native_path.c_str(),F_OK)==0) {
        native_model=std::make_unique<NativeModel>();
        if(!native_model->open(native_path,err))return false;
        const auto* rec=native_model->records();
        for(uint64_t i=0;i<native_model->header().tensor_count;++i){
            const auto& nr=rec[i];
            const std::string name(nr.name,strnlen(nr.name,sizeof nr.name));
            if(skip_vision&&(name.find(".visual.")!=std::string::npos||name.find("vision")!=std::string::npos))continue;
            TensorRef r;r.native=&nr;r.t.name=name;
            r.t.dtype=STDtype(nr.source_dtype);
            r.t.shape.assign(nr.shape,nr.shape+nr.rank);
            r.t.begin=0;r.t.end=nr.payload_bytes;
            index[name]=r;
        }
    } else {
        HFModel hf;
        if (!hf.discover(dir, err)) return false;
        for (size_t i = 0; i < hf.shards.size(); ++i) {
            auto st = std::make_unique<SafeTensors>();
            if (!st->open(hf.shards[i], err)) return false;
            for (const auto& kv : st->tensors()) {
                if (skip_vision && (kv.first.find(".visual.")!=std::string::npos||kv.first.find("vision")!=std::string::npos)) continue;
                TensorRef r;
                r.shard = int(i);
                r.t     = kv.second;
                index[kv.first] = r;
            }
            shards.push_back(std::move(st));
        }
    }
    if (index.empty()) { err = "no tensors found in " + dir; return false; }

    // ---- prefix ------------------------------------------------------
    // Multimodal checkpoints nest the text model one level deeper. The
    // MoE 35B uses model.language_model.*, plain text models use model.*
    prefix = index.count("model.language_model.embed_tokens.weight")
           ? "model.language_model." : "model.";

    // Offline format compilers operate over the checkpoint index directly.
    // Some BF16 releases store all experts in fused 3-D tensors rather than
    // the per-expert names expected by the inference compatibility loader.
    if (index_only) return true;

    auto get = [&](const std::string& n) {
        auto it = index.find(n);
        return it == index.end() ? TensorRef{} : it->second;
    };
    auto linear = [&](const std::string& base) {
        TensorRef r = get(base + ".weight");
        if (r.ok()) {
            // compressed-tensors FP8 stores a per-output-channel scale
            // beside every quantized Linear weight.  This is not limited to
            // routed experts: attention, shared experts, routers and the
            // output projection use the same convention.
            TensorRef sc = get(base + ".weight_scale");
            if (sc.ok()) {
                r.row_scaled = true;
                r.scales_shard = sc.shard;
                r.scales_t = sc.t;
            }
            return r;
        }
        TensorRef cp = get(base + ".weight_packed");
        TensorRef cs = get(base + ".weight_scale");
        if (cp.ok() && cs.ok() && cp.t.dtype == STDtype::I32 &&
            cs.t.dtype == STDtype::BF16 && cp.t.shape.size() == 2 &&
            cs.t.shape.size() == 2 && cp.t.shape[0] == cs.t.shape[0]) {
            const int out = int(cp.t.shape[0]);
            const int in = int(cp.t.shape[1]) * 8;
            const int groups = int(cs.t.shape[1]);
            if (groups > 0 && in % groups == 0) {
                cp.compressed_int4 = true;
                cp.scales_shard = cs.shard;
                cp.scales_t = cs.t;
                cp.gptq_group = in / groups;
                cp.t.shape = {out, in};
                return cp;
            }
        }
        TensorRef qw = get(base + ".qweight");
        TensorRef qz = get(base + ".qzeros");
        TensorRef sc = get(base + ".scales");
        if (!qw.ok() || !qz.ok() || !sc.ok() || qw.t.shape.size() != 2 ||
            qz.t.shape.size() != 2 || sc.t.shape.size() != 2) return TensorRef{};
        const int in = int(qw.t.shape[0]) * 8;
        const int out = int(qw.t.shape[1]);
        const int groups = int(qz.t.shape[0]);
        if (groups <= 0 || in % groups) return TensorRef{};
        qw.gptq = true;
        qw.qzeros_shard = qz.shard; qw.qzeros_t = qz.t;
        qw.scales_shard = sc.shard; qw.scales_t = sc.t;
        qw.gptq_group = in / groups;
        qw.t.shape = {out, in};
        return qw;
    };
    // compressed-tensors MXFP4: base.weight_packed [N][K/2] E2M1 nibbles +
    // base.weight_scale [N][K/32] E8M0.  This IS GRIMOIRE's MXFP4 layout, so
    // the upload path copies it straight to VRAM (see quantize_upload_t).
    auto packed = [&](const std::string& base) {
        TensorRef p = get(base + ".weight_packed");
        if (!p.ok()) return TensorRef{};
        TensorRef sc = get(base + ".weight_scale");
        if (!sc.ok()) return TensorRef{};
        p.scales_shard = sc.shard;
        p.scales_t = sc.t;
        const int N = int(p.t.shape[0]);
        const bool int4 = p.t.dtype == STDtype::I32 && sc.t.dtype == STDtype::BF16;
        const int K = int(p.t.shape[1]) * (int4 ? 8 : 2);
        if (int4) {
            const int groups = int(sc.t.shape[1]);
            if (groups <= 0 || K % groups) return TensorRef{};
            p.compressed_int4 = true;
            p.gptq_group = K / groups;
        }
        p.t.shape = {N, K};
        return p;
    };

    embed      = get(prefix + "embed_tokens.weight");
    final_norm = get(prefix + "norm.weight");
    lm_head    = get("lm_head.weight");
    if (!lm_head.ok()) lm_head = get(prefix + "lm_head.weight");
    if (!embed.ok()) { err = "embed_tokens not found (prefix " + prefix + ")"; return false; }

    // ---- layers ------------------------------------------------------
    layers.assign(cfg.n_layers, {});
    for (int L = 0; L < cfg.n_layers; ++L) {
        Qwen35Layer& lay = layers[L];
        lay.kind = cfg.layer_types[L];
        const std::string b = prefix + "layers." + std::to_string(L) + ".";

        lay.input_norm     = get(b + "input_layernorm.weight");
        lay.post_attn_norm = get(b + "post_attention_layernorm.weight");

        if (lay.kind == LayerKind::LINEAR_ATTN) {
            const std::string a = b + "linear_attn.";
            lay.la_in_qkv  = linear(a + "in_proj_qkv");
            lay.la_in_z    = linear(a + "in_proj_z");
            lay.la_in_a    = linear(a + "in_proj_a");
            lay.la_in_b    = linear(a + "in_proj_b");
            lay.la_conv1d  = get(a + "conv1d.weight");
            lay.la_A_log   = get(a + "A_log");
            lay.la_dt_bias = get(a + "dt_bias");
            lay.la_norm    = get(a + "norm.weight");
            lay.la_out     = linear(a + "out_proj");
        } else {
            const std::string a = b + "self_attn.";
            lay.q_proj = linear(a + "q_proj");
            lay.k_proj = linear(a + "k_proj");
            lay.v_proj = linear(a + "v_proj");
            lay.o_proj = linear(a + "o_proj");
            lay.q_norm = get(a + "q_norm.weight");
            lay.k_norm = get(a + "k_norm.weight");
            if (cfg.is_muse) {
                lay.attn_gate    = linear(a + "gate_proj");
                lay.pre_ff_norm  = get(b + "pre_feedforward_layernorm.weight");
                lay.post_ff_norm = get(b + "post_feedforward_layernorm.weight");
            }
            if (cfg.is_k2) {
                // Every K2 layer has the attention output gate; only the
                // SPARSE ones replace v_proj with the MoVA bank.  Layers
                // 0-2 are dense and keep the ordinary v_proj resolved
                // above, so both shapes have to coexist here.
                lay.k2_sparse = cfg.k2_sparse[size_t(L)];
                if (cfg.attn_gate) lay.attn_gate = linear(a + "gate_proj");
                if (lay.k2_sparse && cfg.mova_experts > 0) {
                    lay.v_router      = linear(a + "v_router");
                    lay.v_router_bias = get(a + "v_router.bias");
                    lay.v_experts.resize(size_t(cfg.mova_experts));
                    for (int e = 0; e < cfg.mova_experts; ++e)
                        lay.v_experts[size_t(e)] =
                            linear(a + "v_experts." + std::to_string(e));
                }
            }
        }

        const std::string m = b + "mlp.";
        // K2: a DENSE layer has no router and no experts -- its FFN is one
        // intermediate_size MLP under mlp.{gate,up,down}_proj.  cfg.is_moe()
        // is true for the model as a whole, so without this the three dense
        // layers would be sent looking for experts that do not exist.
        if (cfg.is_k2 && !lay.k2_sparse) {
            lay.sh_gate = linear(m + "gate_proj");
            lay.sh_up   = linear(m + "up_proj");
            lay.sh_down = linear(m + "down_proj");
        } else if (cfg.is_moe()) {
            lay.router    = linear(m + "gate");
            // K2 names the shared expert "shared_experts" (plural) and
            // carries a router bias; Qwen uses "shared_expert" and a
            // separate shared_expert_gate.
            if (cfg.is_k2) {
                lay.router_bias = get(m + "gate.bias");
                lay.sh_gate = linear(m + "shared_experts.gate_proj");
                lay.sh_up   = linear(m + "shared_experts.up_proj");
                lay.sh_down = linear(m + "shared_experts.down_proj");
            } else {
                lay.sh_gate   = linear(m + "shared_expert.gate_proj");
                lay.sh_up     = linear(m + "shared_expert.up_proj");
                lay.sh_down   = linear(m + "shared_expert.down_proj");
                lay.sh_gate_w = linear(m + "shared_expert_gate");
            }

            lay.e_gate_p.resize(cfg.n_experts); lay.e_gate_s.resize(cfg.n_experts);
            lay.e_up_p.resize(cfg.n_experts);   lay.e_up_s.resize(cfg.n_experts);
            lay.e_down_p.resize(cfg.n_experts); lay.e_down_s.resize(cfg.n_experts);
            const TensorRef fused_gu = get(m + "experts.gate_up_proj");
            const TensorRef fused_dn = get(m + "experts.down_proj");
            for (int e = 0; e < cfg.n_experts; ++e) {
                const std::string x = m + "experts." + std::to_string(e) + ".";
                lay.e_gate_p[e] = get(x + "gate_proj.weight_packed");
                lay.e_gate_s[e] = get(x + "gate_proj.weight_scale");
                lay.e_up_p[e]   = get(x + "up_proj.weight_packed");
                lay.e_up_s[e]   = get(x + "up_proj.weight_scale");
                lay.e_down_p[e] = get(x + "down_proj.weight_packed");
                lay.e_down_s[e] = get(x + "down_proj.weight_scale");
                auto fallback = [&](const std::string& base, TensorRef& p, TensorRef& s) {
                    if (p.ok()) return;
                    p = linear(base);
                    if (!p.ok() || p.gptq) return;
                    s = get(base + ".weight_scale");
                    if (s.ok()) {
                        p.row_scaled = true;
                        p.scales_shard = s.shard;
                        p.scales_t = s.t;
                    }
                };
                fallback(x + "gate_proj", lay.e_gate_p[e], lay.e_gate_s[e]);
                fallback(x + "up_proj",   lay.e_up_p[e],   lay.e_up_s[e]);
                fallback(x + "down_proj", lay.e_down_p[e], lay.e_down_s[e]);
                // BF16 Ornith-1.5 stores expert matrices fused as
                // [E,2I,H] and [E,H,I].  Resolve zero-copy logical slices so
                // the B70 upload path can quantize them directly to MXFP4.
                if (!lay.e_gate_p[e].ok() && fused_gu.ok()) {
                    lay.e_gate_p[e] = fused_expert_slice(fused_gu,
                        uint64_t(e) * 2 * cfg.moe_inter * cfg.hidden,
                        cfg.moe_inter, cfg.hidden, ".gate");
                    lay.e_up_p[e] = fused_expert_slice(fused_gu,
                        (uint64_t(e) * 2 * cfg.moe_inter + cfg.moe_inter) * cfg.hidden,
                        cfg.moe_inter, cfg.hidden, ".up");
                }
                if (!lay.e_down_p[e].ok() && fused_dn.ok())
                    lay.e_down_p[e] = fused_expert_slice(fused_dn,
                        uint64_t(e) * cfg.hidden * cfg.moe_inter,
                        cfg.hidden, cfg.moe_inter, ".down");
            }
        } else if (cfg.is_muse) {
            lay.sh_gate = packed(m + "gate_proj");
            lay.sh_up   = packed(m + "up_proj");
            lay.sh_down = packed(m + "down_proj");
        } else {
            lay.sh_gate = linear(m + "gate_proj");
            lay.sh_up   = linear(m + "up_proj");
            lay.sh_down = linear(m + "down_proj");
        }
    }

    // ---- validate ----------------------------------------------------
    // A missing tensor here means the name convention differs and the
    // model would otherwise run on garbage. Fail loudly instead.
    int missing = 0;
    std::string first_missing;
    auto need = [&](const TensorRef& r, const char* what, int L) {
        if (!r.ok()) {
            if (++missing == 1)
                first_missing = std::string(what) + " (layer " + std::to_string(L) + ")";
        }
    };
    for (int L = 0; L < cfg.n_layers; ++L) {
        const Qwen35Layer& lay = layers[L];
        need(lay.input_norm, "input_layernorm", L);
        if (lay.kind == LayerKind::LINEAR_ATTN) {
            need(lay.la_in_qkv, "linear_attn.in_proj_qkv", L);
            need(lay.la_out,    "linear_attn.out_proj", L);
            need(lay.la_conv1d, "linear_attn.conv1d", L);
        } else {
            need(lay.q_proj, "self_attn.q_proj", L);
            need(lay.o_proj, "self_attn.o_proj", L);
        }
        // K2 mixes both FFN shapes in one model: a dense layer has a plain
        // MLP and no router at all, so requiring mlp.gate everywhere would
        // reject a valid checkpoint.
        if (cfg.is_k2 && !lay.k2_sparse) {
            need(lay.sh_gate, "mlp.gate_proj", L);
        } else if (cfg.is_moe()) {
            need(lay.router, "mlp.gate", L);
            need(lay.e_gate_p.empty() ? TensorRef{} : lay.e_gate_p[0], "experts.0.gate_proj", L);
        } else {
            need(lay.sh_gate, "mlp.gate_proj", L);
        }
        if (cfg.is_k2 && lay.k2_sparse && cfg.mova_experts > 0) {
            // MoVA replaces v_proj entirely; if these are missing the value
            // stream is silently absent rather than wrong.
            need(lay.v_router, "self_attn.v_router", L);
            need(lay.v_experts.empty() ? TensorRef{} : lay.v_experts[0],
                 "self_attn.v_experts.0", L);
        }
    }
    if (missing) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "%d expected tensors missing, first: %s. "
                      "The checkpoint uses a different naming convention.",
                      missing, first_missing.c_str());
        err = buf;
        return false;
    }
    return true;
}

bool Qwen35Model::native_view(const TensorRef& r, QuantWeight& out, std::string& err) const {
    if (!r.native || !native_model || r.t.shape.size()!=2) {
        err="expected a native matrix"; return false;
    }
    NativeLayout l;
    if (!native_layout(*r.native,l,err)) return false;
    if (r.t.shape[0]<=0 || r.t.shape[0]>INT32_MAX || r.t.shape[1]!=l.K ||
        r.native_payload_offset%l.row_payload_bytes) {
        err="invalid native matrix slice geometry"; return false;
    }
    return native_model->view(*r.native,r.native_payload_offset/l.row_payload_bytes,
                              int(r.t.shape[0]),out,err);
}

bool Qwen35Model::read_native_f32(const TensorRef& r,float* dst,std::string& err) const {
    if (!r.native || !dst) { err="native tensor or destination missing"; return false; }
    if (r.native->encoding!=uint32_t(NativeEncoding::RAW)) {
        QuantWeight w;
        if (!native_view(r,w,err)) return false;
        for (int n=0;n<w.N;++n) for (int k=0;k<w.K;++k)
            dst[size_t(n)*w.K+k]=w.at(n,k);
        return true;
    }
    const int64_t n=r.t.numel();
    const auto dtype=STDtype(r.native->source_dtype);
    const size_t unit=size_t(st_dtype_size(dtype));
    if (!unit || n<=0 || uint64_t(n)>SIZE_MAX/unit ||
        r.native_payload_offset>r.native->payload_bytes ||
        uint64_t(n)*unit>r.native->payload_bytes-r.native_payload_offset) {
        err="unsupported native RAW dtype or invalid slice"; return false;
    }
    const auto* p=static_cast<const uint8_t*>(data(r));
    st_to_f32(p,dtype,n,dst);
    return true;
}

QuantWeight Qwen35Model::quant_view(const TensorRef& packed, const TensorRef& scale,
                                    int N, int K) const {
    QuantWeight w;
    w.fmt        = expert_fmt;
    w.N          = N;
    w.K          = K;
    w.payload    = static_cast<const uint8_t*>(data(packed));
    w.scales     = data(scale);
    w.zeros      = nullptr;
    w.row_bytes  = bytes_per_row(expert_fmt, K);
    w.row_scales = scales_per_row(expert_fmt, K);
    return w;
}

int64_t Qwen35Model::total_bytes(bool include_experts) const {
    int64_t n = bytes(embed) + bytes(final_norm) + bytes(lm_head);
    for (const Qwen35Layer& L : layers) {
        n += bytes(L.input_norm) + bytes(L.post_attn_norm);
        n += bytes(L.la_in_qkv) + bytes(L.la_in_z) + bytes(L.la_in_a) + bytes(L.la_in_b);
        n += bytes(L.la_conv1d) + bytes(L.la_A_log) + bytes(L.la_dt_bias)
           + bytes(L.la_norm) + bytes(L.la_out);
        n += bytes(L.q_proj) + bytes(L.k_proj) + bytes(L.v_proj) + bytes(L.o_proj);
        n += bytes(L.router) + bytes(L.sh_gate) + bytes(L.sh_up) + bytes(L.sh_down)
           + bytes(L.sh_gate_w);
        if (include_experts)
            for (size_t e = 0; e < L.e_gate_p.size(); ++e)
                n += bytes(L.e_gate_p[e]) + bytes(L.e_gate_s[e])
                   + bytes(L.e_up_p[e])   + bytes(L.e_up_s[e])
                   + bytes(L.e_down_p[e]) + bytes(L.e_down_s[e]);
    }
    return n;
}

void Qwen35Model::summary() const {
    int nlin = 0, nfull = 0;
    for (LayerKind k : cfg.layer_types)
        (k == LayerKind::LINEAR_ATTN ? nlin : nfull)++;

    std::printf("  %-26s %s\n", "prefix", prefix.c_str());
    std::printf("  %-26s %d\n", "hidden", cfg.hidden);
    std::printf("  %-26s %d  (%d linear, %d full)\n", "layers", cfg.n_layers, nlin, nfull);
    std::printf("  %-26s %d\n", "vocab", cfg.vocab);
    if (cfg.is_moe())
        std::printf("  %-26s %d experts, top-%d, inter %d, shared %d\n", "moe",
                    cfg.n_experts, cfg.top_k, cfg.moe_inter, cfg.shared_inter);
    else
        std::printf("  %-26s dense, inter %d\n", "ffn", cfg.dense_inter);
    std::printf("  %-26s %d q / %d kv heads, dim %d, rope %.0f (partial %.2f)\n",
                "full attention", cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                cfg.rope_theta, cfg.partial_rope);
    std::printf("  %-26s %d k-heads x%d, %d v-heads x%d, conv %d\n", "gated deltanet",
                cfg.lin_k_heads, cfg.lin_k_dim, cfg.lin_v_heads, cfg.lin_v_dim,
                cfg.conv_kernel);
    std::printf("  %-26s %.2f MiB/layer, %.1f MiB total (context-independent)\n",
                "deltanet state",
                double(cfg.deltanet_state_bytes()) / 1048576.0,
                double(cfg.deltanet_state_bytes()) * nlin / 1048576.0);
    std::printf("  %-26s %.2f GiB\n", "weights (text only)",
                double(total_bytes()) / 1073741824.0);
}

} // namespace b70
