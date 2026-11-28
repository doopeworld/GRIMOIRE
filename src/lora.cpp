// b70/lora.cpp -- the PEFT LoRA adapter loader. Reads the adapter's A/B (F32, converted to
// bf16 at load), validates them against the base geometry, pads to r16 and packs every module
// into one contiguous host blob. Any failure throws std::runtime_error naming the offending
// file / tensor / layer -- the caller (Grimoire::build()) turns the throw into a named refusal.
#include "b70/lora.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "b70/json.hpp"
#include "b70/safetensors.hpp"

namespace b70::lora {

namespace {

// The module's placement and full dims (single card, C = 1).
struct ModInfo {
    Mod mod;
    bool col;      // column-parallel (q/k/v, gate/up) or row-parallel (o, down)
    int full_n;    // B's output dim (full)
    int full_k;    // A's input dim (full)
    int out_off;   // the offset of B's rows within the target buffer
};

bool is_gqa_module(Mod m) {
    return m == Mod::Q || m == Mod::K || m == Mod::V || m == Mod::O;
}

// Map a module name to its placement at layer i. full_n / full_k depend on the layer's
// head_dim / kv_heads; gate/up's width is the model-wide dense intermediate.
bool mod_info(const std::string& name, const LoraGeom& g, int layer, ModInfo& out) {
    const int hd = g.hd(layer);
    const int kv = g.kv(layer);
    const int q_n = g.q_heads * (g.attn_output_gate ? 2 : 1) * hd;
    const int k_n = kv * hd;
    const int q_hd = g.q_heads * hd;   // the attention output (query-only) width
    if (name == "q_proj")    { out = {Mod::Q,    true,  q_n,     g.hidden, 0}; return true; }
    if (name == "k_proj")    { out = {Mod::K,    true,  k_n,     g.hidden, 0}; return true; }
    if (name == "v_proj")    { out = {Mod::V,    true,  k_n,     g.hidden, 0}; return true; }
    if (name == "o_proj")    { out = {Mod::O,    false, g.hidden, q_hd,    0}; return true; }
    if (name == "gate_proj") { out = {Mod::Gate, true,  g.inter, g.hidden, 0}; return true; }
    if (name == "up_proj")   { out = {Mod::Up,   true,  g.inter, g.hidden, g.inter}; return true; }
    if (name == "down_proj") { out = {Mod::Down, false, g.hidden, g.inter, 0}; return true; }
    return false;
}

// Parse an adapter tensor name -> (layer, module, is_a). False when the name is not a LoRA
// tensor (an auxiliary side table carried along in the file, e.g. the base model's own).
bool parse_name(const std::string& name, const LoraGeom& g, int& layer, std::string& module, bool& is_a) {
    const std::string bb = "base_model.model.";
    if (name.rfind(bb, 0) != 0) return false;
    std::string rest = name.substr(bb.size());
    // The nested text model carries model.language_model.; a plain text model does
    // not.  Accept either (the configured one first) -- the module path that follows
    // is identical, only the nesting differs.
    std::vector<std::string> prefixes = {g.prefix, "model.language_model.", "model."};
    std::sort(prefixes.begin(), prefixes.end());
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
    std::sort(prefixes.begin(), prefixes.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    std::string::size_type start = std::string::npos;
    for (const auto& p : prefixes) {
        if (rest.size() >= p.size() && rest.rfind(p, 0) == 0) { start = p.size(); break; }
    }
    if (start == std::string::npos) {
        // Bare: the checkpoint carries no nesting and the name starts directly
        // with "layers." (a plain-layout adapter, e.g. "base_model.model.layers.0...").
        if (rest.rfind("layers.", 0) == 0) start = 0;
        else return false;
    }
    rest = rest.substr(start);
    const std::string layers_mark = "layers.";
    if (rest.rfind(layers_mark, 0) != 0) return false;
    rest = rest.substr(layers_mark.size());  // "{l}.{module_path}.lora_{A,B}.weight"
    const auto dot = rest.find('.');
    if (dot == std::string::npos || dot == 0) return false;
    int l = 0;
    try { l = std::stoi(rest.substr(0, dot)); } catch (...) { return false; }
    layer = l;
    std::string mid = rest.substr(dot + 1);  // "{module_path}.lora_{A,B}.weight"
    const std::string la = ".lora_A.weight", lb = ".lora_B.weight";
    if (mid.size() > la.size() && mid.compare(mid.size() - la.size(), la.size(), la) == 0) {
        is_a = true;
        module = mid.substr(0, mid.size() - la.size());
    } else if (mid.size() > lb.size() && mid.compare(mid.size() - lb.size(), lb.size(), lb) == 0) {
        is_a = false;
        module = mid.substr(0, mid.size() - lb.size());
    } else {
        return false;
    }
    const auto d2 = module.rfind('.');
    if (d2 != std::string::npos) module = module.substr(d2 + 1);  // "mlp.gate_proj" -> "gate_proj"
    return true;
}

// Read a tensor's F32 elements out of the (pread) bytes and convert row-major to bf16.
std::vector<uint16_t> read_f32_bf16(const SafeTensors& ck, const STTensor& t) {
    if (t.dtype != STDtype::F32)
        throw std::runtime_error("lora: " + t.name + " is not F32 (the adapter's A/B must be float32)");
    const size_t n = size_t(t.numel());
    if (n == 0) throw std::runtime_error("lora: " + t.name + " is empty");
    std::vector<uint8_t> raw(size_t(t.end - t.begin));
    std::string err;
    if (!ck.read_raw(t, raw.data(), err))
        throw std::runtime_error("lora: cannot read " + t.name + ": " + err);
    std::vector<float> f32(n);
    st_to_f32(raw.data(), t.dtype, int64_t(n), f32.data());
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = f32_to_bf16(f32[i]).bits;
    return out;
}

// [rows][cols] bf16 (padded with zeros) -> [rows_pad][cols_pad].
std::vector<uint16_t> pad2(const std::vector<uint16_t>& src, int rows, int cols,
                           int rows_pad, int cols_pad) {
    std::vector<uint16_t> out(size_t(rows_pad) * cols_pad, 0);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[size_t(r) * cols_pad + c] = src[size_t(r) * cols + c];
    return out;
}

LoraModule& layer_module(LoraLayer& ll, Mod m) {
    switch (m) {
        case Mod::Q:    return ll.q;
        case Mod::K:    return ll.k;
        case Mod::V:    return ll.v;
        case Mod::O:    return ll.o;
        case Mod::Gate: return ll.gate;
        case Mod::Up:   return ll.up;
        case Mod::Down: return ll.down;
    }
    return ll.q;
}

struct Config {
    int r = 0;
    int r16 = 0;
    float scale = 1.f;
    std::string name;
};

Config parse_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("lora: cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    const Json cfg = Json::parse(ss.str());
    const Json* pt = cfg.find("peft_type");
    if (!pt || pt->kind != Json::Kind::String || pt->text() != "LORA")
        throw std::runtime_error("lora: peft_type must be LORA");
    const Json* rv = cfg.find("r");
    if (!rv || rv->kind != Json::Kind::Number || !std::isfinite(rv->number) ||
        std::trunc(rv->number) != rv->number || rv->number < 1)
        throw std::runtime_error("lora: adapter_config.json has no positive integer r");
    const Json* av = cfg.find("lora_alpha");
    if (!av || av->kind != Json::Kind::Number || !std::isfinite(av->number) || av->number <= 0)
        throw std::runtime_error("lora: adapter_config.json has no positive lora_alpha");
    if (const Json* d = cfg.find("use_dora"); d && d->kind == Json::Kind::Boolean && d->boolean)
        throw std::runtime_error("lora: use_dora is not supported");
    if (const Json* rs = cfg.find("use_rslora"); rs && rs->kind == Json::Kind::Boolean && rs->boolean)
        throw std::runtime_error("lora: use_rslora is not supported");
    Config cfg_out;
    cfg_out.r = int(rv->number);
    cfg_out.r16 = (cfg_out.r + 15) / 16 * 16;
    cfg_out.scale = float(av->number) / float(cfg_out.r);
    if (const Json* bn = cfg.find("base_model_name_or_path"); bn && bn->kind == Json::Kind::String)
        cfg_out.name = bn->string;
    return cfg_out;
}

}  // namespace

LoraAdapter::LoraAdapter(const std::string& adapter_dir, const LoraGeom& geom) {
    const Config cfg = parse_config(adapter_dir + "/adapter_config.json");
    name_ = cfg.name;
    r_ = cfg.r;
    r16_ = cfg.r16;
    scale_ = cfg.scale;

    // 1. The adapter's tensors.
    SafeTensors ck;
    std::string err;
    const std::string st_path = adapter_dir + "/adapter_model.safetensors";
    if (!ck.open(st_path, err))
        throw std::runtime_error("lora: cannot open " + st_path + ": " + err);

    struct Entry {
        const STTensor* a = nullptr;
        const STTensor* b = nullptr;
        ModInfo info{};
    };
    std::map<int, std::map<std::string, Entry>> entries;
    for (const auto& kv : ck.tensors()) {
        const STTensor& ti = kv.second;
        int layer = -1;
        std::string module;
        bool is_a = false;
        if (!parse_name(ti.name, geom, layer, module, is_a)) continue;  // an auxiliary table, not a LoRA tensor
        if (ti.dtype != STDtype::F32)
            throw std::runtime_error("lora: " + ti.name + " is not F32 (the adapter's A/B must be float32)");
        if (layer < 0 || layer >= geom.layers)
            throw std::runtime_error("lora: " + ti.name + " names layer " + std::to_string(layer) +
                                     " past the model's " + std::to_string(geom.layers));
        ModInfo mi;
        if (!mod_info(module, geom, layer, mi))
            throw std::runtime_error("lora: unsupported module " + module + " (" + ti.name + ")");
        if (is_gqa_module(mi.mod) && !geom.full_attn(layer))
            throw std::runtime_error("lora: " + module + " on layer " + std::to_string(layer) +
                                     " is not a full-attention (GQA) layer");
        Entry& e = entries[layer][module];
        auto shape_desc = [&ti]() {
            std::ostringstream s;
            s << "[";
            for (size_t i = 0; i < ti.shape.size(); ++i) s << (i ? ", " : "") << ti.shape[i];
            s << "]";
            return s.str();
        };
        if (is_a) {
            if (ti.shape.size() != 2 || ti.shape[0] != cfg.r || ti.shape[1] != mi.full_k)
                throw std::runtime_error("lora: " + ti.name + " is " + shape_desc() +
                                         ", expected [r=" + std::to_string(cfg.r) + ", k=" +
                                         std::to_string(mi.full_k) + "]");
            e.a = &ti;
        } else {
            if (ti.shape.size() != 2 || ti.shape[0] != mi.full_n || ti.shape[1] != cfg.r)
                throw std::runtime_error("lora: " + ti.name + " is " + shape_desc() +
                                         ", expected [n=" + std::to_string(mi.full_n) + ", r=" +
                                         std::to_string(cfg.r) + "]");
            e.b = &ti;
        }
        e.info = mi;
    }
    if (entries.empty())
        throw std::runtime_error("lora: no LoRA tensors found in " + adapter_dir);
    for (const auto& kv : entries) {
        for (const auto& m : kv.second) {
            if (!m.second.a || !m.second.b)
                throw std::runtime_error("lora: layer " + std::to_string(kv.first) + " " + m.first +
                                         " has " + (m.second.a ? "an A but not the B" : "a B but not the A"));
        }
    }

    // 2. Pack every module into the blob. One blob, per-layer views, stable pointers.
    layers_.assign(size_t(geom.layers), LoraLayer{});
    n_max_ = 0;
    size_t off = 0;

    struct Slot {
        int layer;
        Mod mod;
        const STTensor* a;
        const STTensor* b;
        int k, n, out_off;
        size_t a_off, b_off;
    };
    std::vector<Slot> slots;
    for (const auto& kv : entries) {
        const int layer = kv.first;
        for (const auto& m : kv.second) {
            const Entry& e = m.second;
            Slot s;
            s.layer = layer;
            s.mod = e.info.mod;
            s.a = e.a;
            s.b = e.b;
            s.k = e.info.full_k;
            s.n = e.info.full_n;
            s.out_off = e.info.out_off;
            s.a_off = off;
            off += size_t(cfg.r16) * size_t(s.k) * sizeof(uint16_t);
            s.b_off = off;
            off += size_t(s.n) * size_t(cfg.r16) * sizeof(uint16_t);
            n_max_ = std::max(n_max_, s.n);
            slots.push_back(s);
        }
    }
    blob_.assign(off, 0);
    for (const Slot& s : slots) {
        const std::vector<uint16_t> a = read_f32_bf16(ck, *s.a);
        const std::vector<uint16_t> b = read_f32_bf16(ck, *s.b);
        const std::vector<uint16_t> a_pad = pad2(a, cfg.r, s.k, cfg.r16, s.k);
        const std::vector<uint16_t> b_pad = pad2(b, s.n, cfg.r, s.n, cfg.r16);
        std::memcpy(blob_.data() + s.a_off, a_pad.data(), a_pad.size() * sizeof(uint16_t));
        std::memcpy(blob_.data() + s.b_off, b_pad.data(), b_pad.size() * sizeof(uint16_t));
        LoraModule& lm = layer_module(layers_[size_t(s.layer)], s.mod);
        lm.a = reinterpret_cast<const uint16_t*>(blob_.data() + s.a_off);
        lm.b = reinterpret_cast<const uint16_t*>(blob_.data() + s.b_off);
        lm.r16 = cfg.r16;
        lm.k = s.k;
        lm.n = s.n;
        lm.out_off = s.out_off;
        lm.scale = cfg.scale;
    }
}

void lora_reference_delta(const LoraModule& lm, const float* x, float* out, int m, int out_ld) {
    if (!lm.present()) return;
    if (out_ld < lm.n + lm.out_off)
        throw std::invalid_argument("lora_reference_delta: the out stride is short of the module's columns");
    std::vector<float> mid_b16(size_t(m) * lm.r16);
    for (int r = 0; r < m; ++r) {
        for (int j = 0; j < lm.r16; ++j) {
            float mid = 0.0f;
            for (int c = 0; c < lm.k; ++c)
                mid += bf16_to_f32(bf16_t{lm.a[size_t(j) * lm.k + c]}) * x[size_t(r) * lm.k + c];
            mid_b16[size_t(r) * lm.r16 + j] = bf16_to_f32(f32_to_bf16(mid));
        }
    }
    for (int r = 0; r < m; ++r) {
        for (int i = 0; i < lm.n; ++i) {
            float d = 0.0f;
            for (int j = 0; j < lm.r16; ++j)
                d += bf16_to_f32(bf16_t{lm.b[size_t(i) * lm.r16 + j]}) * mid_b16[size_t(r) * lm.r16 + j];
            out[size_t(r) * out_ld + lm.out_off + i] += lm.scale * d;
        }
    }
}

}  // namespace b70::lora
