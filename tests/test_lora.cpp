// PEFT LoRA adapter gate (host): the config rejects, the name -> placement mapping and
// shape rejects, the packed A/B bytes, and the reference delta math against an independent
// hand-rolled reference. GPU kernels (lora_apply) are the same arithmetic, verified on the
// B70 in the GPU follow-up -- this file is the executable host half of the gate.
//
// The geometry is deliberately small (2 layers, hidden 64) -- the pack/convert/reference
// logic is exercised at full strength, just at tiny widths. One extra case reproduces the
// hermes adapter's Qwen3.8-27B mapping (hidden 5120, 48 heads, head_dim 128,
// attn_output_gate) to pin the q/k/v/o/gate/up/down widths the real adapter carries.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <unistd.h>

#include "b70/lora.hpp"
#include "b70/safetensors.hpp"
#include "b70/formats.hpp"

namespace fs = std::filesystem;
namespace lo = b70::lora;

namespace {

int g_fail = 0, g_pass = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  ok ? ++g_pass : ++g_fail;
}
bool throws(const std::function<void()> &f, std::string *msg = nullptr) {
  try { f(); }
  catch (const std::exception &e) { if (msg) *msg = e.what(); return true; }
  return false;
}

// ---- a synthetic safetensors writer --------------------------------------------------
// Layout: [u64 header_len][header (padded with spaces)][data]. data_offsets are offsets
// into the data segment.
static void write_shard(const fs::path &p, const std::string &header, const std::string &data, int pad = 0) {
  std::string h = header + std::string(pad, ' ');
  std::ofstream o(p, std::ios::binary);
  uint64_t n = h.size();
  for (int i = 0; i < 8; ++i) o.put(static_cast<char>((n >> (8 * i)) & 0xff));
  o << h << data;
}

// A deterministic value in [-0.5, 0.5).
float val(uint32_t seed, size_t i) {
  uint32_t x = seed * 2654435761u + static_cast<uint32_t>(i) * 40503u;
  x ^= x >> 15;
  x *= 2246822519u;
  x ^= x >> 13;
  return static_cast<float>(x & 0xFFFF) / 65535.f - 0.5f;
}

// ---- a synthetic adapter -----------------------------------------------------------------

struct Mod {
  std::string name;
  int full_n, full_k;
  int mpath;  // 0 self_attn (q/k/v/o), 1 mlp (gate/up/down)
};

struct Geom {
  int layers = 2;
  int hidden = 64;
  int q_heads = 2;
  int kv_heads = 2;
  int head_dim = 16;
  bool attn_output_gate = false;
  int inter = 64;
  std::vector<bool> full_attention = {true, false};  // layer 0 GQA, layer 1 GDN

  int q_n() const { return q_heads * (attn_output_gate ? 2 : 1) * head_dim; }
  int k_n() const { return kv_heads * head_dim; }

  lo::LoraGeom geom() const {
    lo::LoraGeom g;
    g.layers = layers;
    g.hidden = hidden;
    g.q_heads = q_heads;
    g.attn_output_gate = attn_output_gate;
    g.inter = inter;
    g.head_dim = head_dim;
    g.kv_heads = kv_heads;
    g.prefix = "model.language_model.";
    g.full_attention = full_attention;
    return g;
  }
};

static std::vector<Mod> all_mods(const Geom &g) {
  const int hd = g.head_dim, kv = g.kv_heads;
  return {
      {"q_proj",  g.q_heads * (g.attn_output_gate ? 2 : 1) * hd, g.hidden, 0},
      {"k_proj",  kv * hd, g.hidden, 0},
      {"v_proj",  kv * hd, g.hidden, 0},
      {"o_proj",  g.hidden, g.q_heads * hd, 0},
      {"gate_proj", g.inter, g.hidden, 1},
      {"up_proj",   g.inter, g.hidden, 1},
      {"down_proj", g.hidden, g.inter, 1}};
}

static bool is_gqa_mod(const Mod &m) { return m.mpath == 0; }

static uint32_t mod_seed(const Mod &m) {
  uint32_t h = 2166136261u;
  for (const char *p = m.name.c_str(); *p; ++p) h = (h ^ static_cast<uint8_t>(*p)) * 16777619u;
  return h;
}

// Write adapter_config.json + adapter_model.safetensors with real F32 values and the
// geometry's module set. gqa_layers: the layers that carry q/k/v/o (rest = GDN).
static void write_adapter(const fs::path &dir, const Geom &g, int r, int alpha,
                          const std::vector<int> &gqa_layers, bool zero) {
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::ofstream(dir / "adapter_config.json")
      << R"({"peft_type":"LORA","r":)" << r << R"(,"lora_alpha":)" << alpha
      << R"(,"base_model_name_or_path":"test-base","use_dora":false,"use_rslora":false})";
  std::string hdr = "{";
  std::string data;
  size_t off = 0;
  bool first = true;
  auto add = [&](const std::string &name, const std::vector<int64_t> &shape, const std::vector<float> &f32) {
    const size_t bytes = f32.size() * 4;
    if (!first) hdr += ",";
    first = false;
    hdr += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (size_t i = 0; i < shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
    for (float v : f32) { char b[4]; std::memcpy(b, &v, 4); data.append(b, 4); }
    off += bytes;
  };
  for (int l = 0; l < g.layers; ++l) {
    const bool gqa = std::find(gqa_layers.begin(), gqa_layers.end(), l) != gqa_layers.end();
    for (const Mod &m : all_mods(g)) {
      if (is_gqa_mod(m) && !gqa) continue;  // GDN layers carry only the MLP modules
      const std::string mpath = (m.mpath == 0 ? "self_attn." : "mlp.") + m.name;
      const std::string base = "base_model.model.model.language_model.layers." + std::to_string(l) + "." + mpath;
      std::vector<float> a(size_t(r) * m.full_k), b(size_t(m.full_n) * r);
      for (size_t i = 0; i < a.size(); ++i) a[i] = zero ? 0.f : val(mod_seed(m) + static_cast<uint32_t>(l) * 97u, i);
      for (size_t i = 0; i < b.size(); ++i) b[i] = zero ? 0.f : val(mod_seed(m) + static_cast<uint32_t>(l) * 97u + 7u, i);
      add(base + ".lora_A.weight", {r, m.full_k}, a);
      add(base + ".lora_B.weight", {m.full_n, r}, b);
    }
  }
  hdr += "}";
  write_shard(dir / "adapter_model.safetensors", hdr, data, 3);
}

// ---- the reference, written independently of lora_reference_delta -----------------------
// out[r][out_off + c] += scale * sum_j bf16(B[c][j]) * mid[r][j],
// mid[r][j] = bf16(sum_k bf16(A[j][k]) * x[r][k]).
static void hand_reference(const lo::LoraModule &lm, const float *x, float *out, int m, int out_ld) {
  std::vector<float> mid(size_t(m) * lm.r16);
  for (int r = 0; r < m; ++r) {
    for (int j = 0; j < lm.r16; ++j) {
      double s = 0.0;
      for (int c = 0; c < lm.k; ++c)
        s += double(b70::bf16_to_f32(b70::bf16_t{lm.a[size_t(j) * lm.k + c]})) * x[size_t(r) * lm.k + c];
      mid[size_t(r) * lm.r16 + j] = b70::bf16_to_f32(b70::f32_to_bf16(float(s)));
    }
  }
  for (int r = 0; r < m; ++r) {
    for (int i = 0; i < lm.n; ++i) {
      double d = 0.0;
      for (int j = 0; j < lm.r16; ++j)
        d += double(b70::bf16_to_f32(b70::bf16_t{lm.b[size_t(i) * lm.r16 + j]})) * mid[size_t(r) * lm.r16 + j];
      out[size_t(r) * out_ld + lm.out_off + i] += lm.scale * float(d);
    }
  }
}

static int out_ld_for(const Geom &g, const Mod &m) {
  if (m.mpath == 1) return 2 * g.inter;
  if (m.name == "q_proj") return g.q_n();
  if (m.name == "o_proj") return g.hidden;
  return g.k_n();
}

// A plain (non-nested) checkpoint's adapter names the tensors "base_model.model.layers.{L}...",
// not "base_model.model.model.language_model.layers.{L}...".  The loader must accept both --
// only the nesting differs, the module path that follows is identical.
static void plain_prefix_test(const fs::path &base, int &dir_no) {
  const Geom g;
  lo::LoraGeom lg = g.geom();
  lg.prefix = "model.";
  const std::string cfg = R"({"peft_type":"LORA","r":16,"lora_alpha":32,"base_model_name_or_path":"plain-base"})";
  const std::string base_pat = "base_model.model.layers.";
  const std::vector<std::pair<std::string, std::vector<int64_t>>> tensors = {
      {base_pat + "0.self_attn.q_proj.lora_A.weight", {16, 64}},
      {base_pat + "0.self_attn.q_proj.lora_B.weight", {32, 16}}};
  const fs::path d = base / std::to_string(dir_no++);
  fs::remove_all(d);
  fs::create_directories(d);
  std::ofstream(d / "adapter_config.json") << cfg;
  std::string hdr = "{";
  std::string data;
  size_t off = 0;
  bool first = true;
  for (const auto &[name, shape] : tensors) {
    size_t numel = 1;
    for (auto s : shape) numel *= size_t(s);
    const size_t bytes = numel * 4;
    if (!first) hdr += ",";
    first = false;
    hdr += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (size_t i = 0; i < shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
    data.append(bytes, '\0');
    off += bytes;
  }
  hdr += "}";
  write_shard(d / "adapter_model.safetensors", hdr, data, 3);
  lo::LoraAdapter plain(d.string(), lg);
  expect(plain.layer(0).q.present(), "plain prefix: the adapter parses (base_model.model.layers.{L}...)");
  expect(plain.name() == "plain-base", "plain prefix: the name is recorded");
}

// ---- the hermes-adapter mapping check -----------------------------------------------------

static fs::path tmpdir() {
  const char *t = std::getenv("TMPDIR");
  return fs::path(t ? t : "/tmp") / ("grimoire-test-lora-" + std::to_string(::getpid()));
}

}  // namespace

int main() {
  try {
    const fs::path base = tmpdir();
    fs::remove_all(base);
    fs::create_directories(base);
    int dir = 0;

    // ---- config rejects -----------------------------------------------------
    {
      struct Bad { const char *what, *cfg; };
      const Bad bads[] = {
          {"no peft_type", R"({"r":16,"lora_alpha":32})"},
          {"peft_type not LORA", R"({"peft_type":"ADALORA","r":16,"lora_alpha":32})"},
          {"no r", R"({"peft_type":"LORA","lora_alpha":32})"},
          {"r = 0", R"({"peft_type":"LORA","r":0,"lora_alpha":32})"},
          {"r not an integer", R"({"peft_type":"LORA","r":3.5,"lora_alpha":32})"},
          {"no lora_alpha", R"({"peft_type":"LORA","r":16})"},
          {"lora_alpha = 0", R"({"peft_type":"LORA","r":16,"lora_alpha":0})"},
          {"use_dora", R"({"peft_type":"LORA","r":16,"lora_alpha":32,"use_dora":true})"},
          {"use_rslora", R"({"peft_type":"LORA","r":16,"lora_alpha":32,"use_rslora":true})"},
      };
      for (const Bad &b : bads) {
        const fs::path d = base / std::to_string(dir++);
        fs::create_directories(d);
        std::ofstream(d / "adapter_config.json") << b.cfg;
        std::string msg;
        const bool t = throws([&] { lo::LoraAdapter ad(d.string(), Geom{}.geom()); }, &msg);
        expect(t, std::string("config: ") + b.what + " throws (" + msg + ")");
      }
    }

    // ---- mapping / shape rejects ------------------------------------------------
    {
      const Geom g;  // hidden 64, q 2 heads, kv 2, head_dim 16, inter 64
      const std::string P = "base_model.model.model.language_model.layers.";
      const std::string cfg = R"({"peft_type":"LORA","r":16,"lora_alpha":32,"base_model_name_or_path":"t"})";
      auto write_one = [&](const fs::path &d, const std::vector<std::pair<std::string, std::vector<int64_t>>> &tensors) {
        fs::remove_all(d);
        fs::create_directories(d);
        std::ofstream(d / "adapter_config.json") << cfg;
        std::string hdr = "{";
        std::string data;
        size_t off = 0;
        bool first = true;
        for (const auto &[name, shape] : tensors) {
          size_t numel = 1;
          for (auto s : shape) numel *= size_t(s);
          const size_t bytes = numel * 4;
          if (!first) hdr += ",";
          first = false;
          hdr += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[";
          for (size_t i = 0; i < shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(shape[i]);
          hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
          data.append(bytes, '\0');
          off += bytes;
        }
        hdr += "}";
        write_shard(d / "adapter_model.safetensors", hdr, data, 3);
      };
      struct Bad { const char *what; std::vector<std::pair<std::string, std::vector<int64_t>>> t; };
      Bad bads[] = {
          {"A shape [r+1][k]", {{P + "0.self_attn.q_proj.lora_A.weight", {17, 64}}, {P + "0.self_attn.q_proj.lora_B.weight", {32, 16}}}},
          {"B shape [n][r+1]", {{P + "0.self_attn.q_proj.lora_A.weight", {16, 64}}, {P + "0.self_attn.q_proj.lora_B.weight", {32, 17}}}},
          {"unsupported module", {{P + "0.self_attn.router.lora_A.weight", {16, 64}}, {P + "0.self_attn.router.lora_B.weight", {32, 16}}}},
          {"layer past the model", {{P + "99.self_attn.q_proj.lora_A.weight", {16, 64}}, {P + "99.self_attn.q_proj.lora_B.weight", {32, 16}}}},
          {"q_proj on a GDN layer", {{P + "1.self_attn.q_proj.lora_A.weight", {16, 64}}, {P + "1.self_attn.q_proj.lora_B.weight", {32, 16}}}},
          {"A without B", {{P + "0.self_attn.q_proj.lora_A.weight", {16, 64}}}},
          {"B without A", {{P + "0.self_attn.q_proj.lora_B.weight", {32, 16}}}},
      };
      for (const Bad &b : bads) {
        const fs::path d = base / std::to_string(dir++);
        write_one(d, b.t);
        std::string msg;
        const bool t = throws([&] { lo::LoraAdapter ad(d.string(), g.geom()); }, &msg);
        expect(t, std::string("adapter: ") + b.what + " throws (" + msg + ")");
      }
      // an empty adapter file (no tensors)
      {
        const fs::path d = base / std::to_string(dir++);
        fs::remove_all(d);
        fs::create_directories(d);
        std::ofstream(d / "adapter_config.json") << cfg;
        write_shard(d / "adapter_model.safetensors", "{}", "", 3);
        std::string msg;
        const bool t = throws([&] { lo::LoraAdapter ad(d.string(), g.geom()); }, &msg);
        expect(t, std::string("adapter: empty adapter throws (") + msg + ")");
      }
      // a non-F32 tensor
      {
        const fs::path d = base / std::to_string(dir++);
        fs::remove_all(d);
        fs::create_directories(d);
        std::ofstream(d / "adapter_config.json") << cfg;
        std::string hdr = "{\"" + P + "0.self_attn.q_proj.lora_A.weight\":{\"dtype\":\"F16\",\"shape\":[16,64],\"data_offsets\":[0,2048]}}";
        std::string data(2048, '\0');
        write_shard(d / "adapter_model.safetensors", hdr, data, 3);
        std::string msg;
        const bool t = throws([&] { lo::LoraAdapter ad(d.string(), g.geom()); }, &msg);
        expect(t, std::string("adapter: F16 tensor throws (") + msg + ")");
      }
    }

    // ---- the integration: a small synthetic adapter end to end ---------------------------
    {
      const int r = 24, alpha = 48;  // scale 2.0, r16 32
      const Geom g;
      const std::vector<int> gqa_layers = {0};  // layer 0 GQA, layer 1 GDN
      const fs::path d = base / std::to_string(dir++);
      write_adapter(d, g, r, alpha, gqa_layers, false);
      lo::LoraAdapter ad(d.string(), g.geom());
      expect(ad.layers() == g.layers, "adapter: layers");
      expect(ad.r16() == 32, "adapter: r16 = ceil(24/16)*16 = 32");
      expect(ad.scale() == 2.0f, "adapter: scale = lora_alpha / r = 2.0");
      expect(ad.name() == "test-base", "adapter: base_model_name_or_path");
      const lo::LoraLayer &gqa = ad.layer(0);
      expect(gqa.q.present() && gqa.k.present() && gqa.v.present() && gqa.o.present() &&
             gqa.gate.present() && gqa.up.present() && gqa.down.present(),
             "adapter: the GQA layer carries every module");
      const lo::LoraLayer &gdn = ad.layer(1);
      expect(!gdn.q.present() && !gdn.k.present() && !gdn.v.present() && !gdn.o.present(),
             "adapter: the GDN layer carries no GQA module");
      expect(gdn.gate.present() && gdn.up.present() && gdn.down.present(),
             "adapter: the GDN layer carries the MLP modules");
      // the packed A is [r16][k]: rows r..r16-1 are exactly zero (the padding)
      bool pad_ok = true;
      for (int rr = r; rr < 32; ++rr)
        for (int c = 0; c < gqa.gate.k; ++c)
          if (gqa.gate.a[size_t(rr) * gqa.gate.k + c] != 0) { pad_ok = false; break; }
      expect(pad_ok, "adapter: A padding rows are exactly zero");
      // spot-check the pack: A[0][0] is bf16(val(seed(gate),0)) on layer 0
      const uint32_t gate_seed = mod_seed({"gate_proj", g.inter, g.hidden, 1});
      const uint16_t expect_a0 = b70::f32_to_bf16(val(gate_seed, 0)).bits;
      expect(gqa.gate.a[0] == expect_a0, "adapter: the packed A element equals bf16(F32 value)");

      // the reference delta math vs the hand-rolled independent loops
      for (const int m : {1, 32}) {
        std::vector<float> x(size_t(m) * g.hidden);
        for (size_t i = 0; i < x.size(); ++i) x[i] = val(0xABCDu + uint32_t(i % 7) * 97u, i) * 2.f;
        for (int layer = 0; layer < g.layers; ++layer) {
          const lo::LoraLayer &ll = ad.layer(layer);
          for (const Mod &md : all_mods(g)) {
            if (is_gqa_mod(md) && layer != 0) continue;  // GDN layers carry no GQA modules
            const lo::LoraModule *lm = nullptr;
            if (md.name == "q_proj") lm = &ll.q;
            else if (md.name == "k_proj") lm = &ll.k;
            else if (md.name == "v_proj") lm = &ll.v;
            else if (md.name == "o_proj") lm = &ll.o;
            else if (md.name == "gate_proj") lm = &ll.gate;
            else if (md.name == "up_proj") lm = &ll.up;
            else if (md.name == "down_proj") lm = &ll.down;
            if (!lm->present()) continue;
            const int ld = out_ld_for(g, md);
            const std::vector<float> init(size_t(m) * ld, 0.0f);
            std::vector<float> got(size_t(m) * ld), ref(size_t(m) * ld);
            for (size_t i = 0; i < init.size(); ++i) {
              const float v = val(0x1234u + uint32_t(i % 5) * 131u, i) * 0.5f;
              got[i] = v; ref[i] = v;
            }
            lo::lora_reference_delta(*lm, x.data(), got.data(), m, ld);
            hand_reference(*lm, x.data(), ref.data(), m, ld);
            double worst = 0.0;
            for (size_t i = 0; i < ref.size(); ++i)
              worst = std::max(worst, double(std::fabs(got[i] - ref[i])));
            // tolerance 5e-3: the hand-rolled reference accumulates in double while the factory
            // reference accumulates in float -- the gap is float rounding, not layout.
            expect(worst <= 5e-3, std::string("lora ") + md.name + " layer " + std::to_string(layer) +
                                  " m=" + std::to_string(m) + ": reference == hand-rolled (worst " +
                                  std::to_string(worst).substr(0, 6) + ")");
          }
        }
      }

      // zero-adapter invariance: the delta is exactly 0, the output bit-identical
      {
        const fs::path d2 = base / std::to_string(dir++);
        write_adapter(d2, g, r, alpha, gqa_layers, true);  // all-zero A / B
        lo::LoraAdapter z(d2.string(), g.geom());
        const int m = 4;
        const lo::LoraModule &lm = z.layer(0).q;
        std::vector<float> x(size_t(m) * lm.k);
        for (size_t i = 0; i < x.size(); ++i) x[i] = val(0x1234u, i);
        const int ld = out_ld_for(g, all_mods(g)[0]);
        std::vector<float> init(size_t(m) * ld);
        for (size_t i = 0; i < init.size(); ++i) init[i] = val(0x5678u, i);
        std::vector<float> out(init.begin(), init.end());
        lo::lora_reference_delta(lm, x.data(), out.data(), m, ld);
        expect(out == init, "zero adapter: the delta is exactly 0 (bit-identical output)");
      }

      // the hermes mapping: 48 q heads, attn_output_gate, head_dim 128
      {
        lo::LoraGeom lg;
        lg.layers = 64; lg.hidden = 5120; lg.q_heads = 48;
        lg.attn_output_gate = true; lg.inter = 17408;
        lg.head_dim = 128; lg.kv_heads = 8;
        lg.prefix = "model.language_model.";
        for (int i = 0; i < 64; ++i) lg.full_attention.push_back(i % 4 == 0);  // 16 GQA layers
        const fs::path d3 = base / std::to_string(dir++);
        fs::remove_all(d3);
        fs::create_directories(d3);
        std::ofstream(d3 / "adapter_config.json")
            << R"({"peft_type":"LORA","r":32,"lora_alpha":64,"base_model_name_or_path":"Qwen/Qwen3.8-27B","use_dora":false,"use_rslora":false})";
        const int hr = 32;
        const std::string base = "base_model.model.model.language_model.layers.0.";
        struct T { std::string name; std::vector<int64_t> shape; };
        std::vector<T> ts = {
            {base + "self_attn.q_proj.lora_A.weight", {hr, 5120}},
            {base + "self_attn.q_proj.lora_B.weight", {12288, hr}},
            {base + "self_attn.k_proj.lora_A.weight", {hr, 5120}},
            {base + "self_attn.k_proj.lora_B.weight", {1024, hr}},
            {base + "self_attn.v_proj.lora_A.weight", {hr, 5120}},
            {base + "self_attn.v_proj.lora_B.weight", {1024, hr}},
            {base + "self_attn.o_proj.lora_A.weight", {hr, 6144}},
            {base + "self_attn.o_proj.lora_B.weight", {5120, hr}},
            {base + "mlp.gate_proj.lora_A.weight", {hr, 5120}},
            {base + "mlp.gate_proj.lora_B.weight", {17408, hr}},
            {base + "mlp.up_proj.lora_A.weight", {hr, 5120}},
            {base + "mlp.up_proj.lora_B.weight", {17408, hr}},
            {base + "mlp.down_proj.lora_A.weight", {hr, 17408}},
            {base + "mlp.down_proj.lora_B.weight", {5120, hr}},
        };
        std::string hdr = "{";
        std::string data;
        size_t off = 0;
        for (size_t i = 0; i < ts.size(); ++i) {
          size_t numel = 1;
          for (auto s : ts[i].shape) numel *= size_t(s);
          const size_t bytes = numel * 4;
          hdr += (i == 0 ? std::string{} : std::string{","}) + "\"" + ts[i].name + "\":{\"dtype\":\"F32\",\"shape\":[";
          for (size_t j = 0; j < ts[i].shape.size(); ++j) hdr += (j ? "," : "") + std::to_string(ts[i].shape[j]);
          hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
          data.append(bytes, '\0');
          off += bytes;
        }
        hdr += "}";
        write_shard(d3 / "adapter_model.safetensors", hdr, data, 3);

        lo::LoraAdapter ha(d3.string(), lg);
        expect(ha.r16() == 32, "hermes adapter: r16 = 32");
        expect(ha.scale() == 2.0f, "hermes adapter: scale = 64/32 = 2.0");
        expect(ha.name() == "Qwen/Qwen3.8-27B", "hermes adapter: base name");
        const lo::LoraLayer &hl = ha.layer(0);
        expect(hl.q.k == 5120 && hl.q.n == 12288,
               "hermes adapter: q_proj A [32,5120] B [12288,32] (attn_output_gate-packed)");
        expect(hl.k.n == 1024, "hermes adapter: k_proj n = 1024");
        expect(hl.v.n == 1024, "hermes adapter: v_proj n = 1024");
        expect(hl.o.k == 6144 && hl.o.n == 5120,
               "hermes adapter: o_proj A [32,6144] B [5120,32] (attention output)");
        expect(hl.gate.n == 17408 && hl.gate.k == 5120, "hermes adapter: gate n = 17408");
        expect(hl.up.n == 17408 && hl.up.out_off == 17408, "hermes adapter: up offset = inter");
        expect(hl.down.k == 17408 && hl.down.n == 5120, "hermes adapter: down A [32,17408] B [5120,32]");
      }
    }

    // the plain-layout prefix: a non-nested checkpoint's adapter names the tensors
    // base_model.model.layers.{L}... rather than base_model.model.model.language_model.layers.{L}...
    plain_prefix_test(base, dir);

    fs::remove_all(base);
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    std::printf("%s: %d passed, %d failed\n", "FAILED", g_pass, g_fail);
    return 1;
  }
  std::printf("%s: %d passed, %d failed\n", g_fail ? "FAILED" : "OK", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
