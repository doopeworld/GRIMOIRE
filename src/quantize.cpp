// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// =====================================================================
//  quantize.cpp  --  offline fp32 -> {bf16,fp8,int8,int4,mxfp8,mxfp4}
//
//  Runs on the host, once, when the model is converted. Nothing here is
//  on the inference critical path.
// =====================================================================
#include "b70/weights.hpp"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace b70 {
namespace {

inline float absmax(const float* p, int n, int stride = 1) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(p[i * stride]));
    return m;
}

// OCP MX scale selection: place the shared exponent so the block maximum
// lands at the top of the element format's range.
//   X = clamp(floor(log2(amax)) - emax_elem + 127, 0, 254)
uint8_t pick_e8m0(float amax, int emax_elem) {
    if (!(amax > 0.0f) || !std::isfinite(amax)) return 127;   // 2^0
    int e = std::ilogb(amax) - emax_elem;
    e += 127;
    if (e < 0)   e = 0;
    if (e > 254) e = 254;
    return uint8_t(e);
}

} // namespace

// Rows are independent and every loop below writes only its own rows, so
// big matrices are split across threads (bit-identical output).  Loading a
// BF16 checkpoint quantizes every weight here, on the host: K2-Horizon's
// 75 GB took ~6 minutes on one thread.
template <class F>
static void for_rows(int N, size_t work, F&& fn) {
    const int hw = int(std::thread::hardware_concurrency());
    const int T = work < (size_t(1) << 20) ? 1 : std::max(1, std::min(16, std::min(hw, N)));
    if (T <= 1) { fn(0, N); return; }
    std::vector<std::thread> th;
    std::exception_ptr err;
    std::mutex mu;
    for (int t = 0; t < T; ++t)
        th.emplace_back([&, t] {
            const int n0 = int(int64_t(N) * t / T), n1 = int(int64_t(N) * (t + 1) / T);
            try { fn(n0, n1); }
            catch (...) { std::lock_guard<std::mutex> g(mu); if (!err) err = std::current_exception(); }
        });
    for (auto& x : th) x.join();
    if (err) std::rethrow_exception(err);
}

PackedWeight quantize(const float* src, int N, int K, Fmt fmt) {
    return quantize(src, N, K, fmt, nullptr);
}

PackedWeight quantize(const float* src, int N, int K, Fmt fmt, const float* importance) {
    if (!src || N <= 0 || K <= 0 || K > std::numeric_limits<int>::max() / 2)
        throw std::invalid_argument("quantize: invalid matrix dimensions or null input");
    switch (fmt) {
    case Fmt::BF16: case Fmt::FP8_E4M3: case Fmt::FP8_E5M2:
    case Fmt::INT8: case Fmt::INT4: case Fmt::MXFP8: case Fmt::MXFP4: break;
    case Fmt::Q4_K: case Fmt::Q6_K: case Fmt::Q5_K: case Fmt::Q8_0:
    case Fmt::IQ4_XS: case Fmt::IQ4_NL:
        throw std::invalid_argument("quantize: GGUF formats require raw GGUF blocks");
    default: throw std::invalid_argument("quantize: unknown format");
    }
    const int block = fmt == Fmt::INT4 ? kInt4Group :
        (fmt == Fmt::MXFP4 || fmt == Fmt::MXFP8) ? kMXBlock : 1;
    if (K % block)
        throw std::invalid_argument("quantize: row width is not a whole number of groups");
    if (size_t(N) > std::numeric_limits<size_t>::max() / size_t(K) / sizeof(float))
        throw std::overflow_error("quantize: matrix size overflow");
    const size_t count = size_t(N) * size_t(K);
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(src[i]))
            throw std::invalid_argument("quantize: non-finite input weight");
    PackedWeight w;
    w.fmt = fmt; w.N = N; w.K = K;
    w.row_bytes  = bytes_per_row(fmt, K);
    w.row_scales = scales_per_row(fmt, K);
    w.payload.assign(size_t(N) * size_t(w.row_bytes), 0);

    switch (fmt) {

    // -----------------------------------------------------------------
    case Fmt::BF16: {
        for_rows(N, count, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            uint8_t* row = w.payload.data() + int64_t(n) * w.row_bytes;
            for (int k = 0; k < K; ++k) {
                bf16_t b = f32_to_bf16(src[int64_t(n) * K + k]);
                row[2 * k]     = uint8_t(b.bits & 0xFF);
                row[2 * k + 1] = uint8_t(b.bits >> 8);
            }
        }
        });
        break;
    }

    // -----------------------------------------------------------------
    // Per-output-channel symmetric scaling. One fp32 per row.
    case Fmt::INT8:
    case Fmt::FP8_E4M3:
    case Fmt::FP8_E5M2: {
        w.scales_raw.assign(size_t(N) * sizeof(float), 0);
        float* sc = reinterpret_cast<float*>(w.scales_raw.data());
        const float target = (fmt == Fmt::INT8)     ? 127.0f
                           : (fmt == Fmt::FP8_E4M3) ? 448.0f
                                                    : 57344.0f;
        for_rows(N, count, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            const float* in  = src + int64_t(n) * K;
            uint8_t*     row = w.payload.data() + int64_t(n) * w.row_bytes;
            const float  am  = absmax(in, K);
            const float  s   = am > 0.0f ? float(std::max(double(am) / target,
                double(std::numeric_limits<float>::min()))) : 1.0f;
            sc[n] = s;
            for (int k = 0; k < K; ++k) {
                const float v = float(double(in[k]) / s);
                if (fmt == Fmt::INT8) {
                    int q = int(std::nearbyint(v));
                    q = std::max(-127, std::min(127, q));
                    row[k] = uint8_t(int8_t(q));
                } else if (fmt == Fmt::FP8_E4M3) {
                    row[k] = f32_to_e4m3(v);
                } else {
                    row[k] = f32_to_e5m2(v);
                }
            }
        }
        });
        break;
    }

    // -----------------------------------------------------------------
    // Asymmetric 4-bit, group of 128 along K. bf16 scale + uint8 zero.
    // This is the GPTQ/AWQ convention, so converted checkpoints from the
    // usual toolchains drop straight in.
    case Fmt::INT4: {
        // Round-to-nearest from a dense (BF16 / FP32) weight -- what --proj int4 does to every
        // linear a checkpoint does not already store as GPTQ / compressed INT4 (Flash-Next's dense
        // part, a BF16 GGUF, Qwen3.8-27B from its BF16 original).  Two opt-in refinements
        // (2026-11-05, both read only here; the kernels take any group of 64 or 128 from the
        // weight's shape, WeightView::int4_gshift):
        //   GRIMOIRE_INT4_RTN_GROUP=64   groups of 64 along K instead of 128 (+0.19 bits a weight)
        //   GRIMOIRE_INT4_RTN_SEARCH=1   each group's range shrunk by the factor (1.00 .. 0.76) that
        //                                minimises its squared reconstruction error, instead of
        //                                plain min / max (an outlier no longer widens every step)
        // Defaults: 128 and plain min / max, as before.
        // An importance vector (GRIMOIRE_IMATRIX, b70/imatrix.hpp) turns the search on and
        // weights every channel's squared error by it, so the range follows the expected
        // OUTPUT error: channels the layer's input drives hard keep their precision.
        // Measured offline (2026-11-06, Qwen3.8-27B BF16 layers 10-11 with Unsloth's
        // imatrix, output error relative to plain min / max at 128), groups of 128 / 64:
        // q_proj 0.32 / 0.28, in_proj_z 0.56 / 0.45, in_proj_qkv 0.62 / 0.50, k_proj
        // 0.58 / 0.48, v_proj 0.71 / 0.59, o / out_proj 0.80-0.85 / 0.66-0.71, MLP
        // 0.83-0.88 / 0.70-0.74 (the range search alone at 64: 0.74-0.83).
        static const int rtn_group = [] {
            const char* e = std::getenv("GRIMOIRE_INT4_RTN_GROUP");
            return e && std::atoi(e) == 64 ? 64 : kInt4Group; }();
        static const bool rtn_search = [] {
            const char* e = std::getenv("GRIMOIRE_INT4_RTN_SEARCH");
            return e && *e == '1'; }();
        const int GS = (K % rtn_group == 0) ? rtn_group : kInt4Group;
        if (GS != kInt4Group) w.row_scales = K / GS;
        const int G = w.row_scales;
        w.scales_raw.assign(size_t(N) * G * sizeof(bf16_t), 0);
        w.zeros.assign(size_t(N) * G, 0);
        bf16_t* sc = reinterpret_cast<bf16_t*>(w.scales_raw.data());
        // The scale and zero for range [lo, hi] (lo <= 0 <= hi), as the kernel will see them.
        auto pick = [](double lo, double hi, bf16_t& sb, float& s, int& z) {
            s = float((hi - lo) / 15.0);
            if (!(s > 0.0f)) s = 1.0f;
            // Round the scale to bf16 *before* choosing the codes, so
            // the packer quantizes against the value the kernel will
            // actually see. Skipping this is a classic silent 2x
            // error amplifier at group boundaries.
            sb = f32_to_bf16(s);
            if (sb.bits == 0) sb.bits = 1; // smallest positive BF16 scale
            s = bf16_to_f32(sb);
            if (!std::isfinite(s) || !(s > 0.0f))
                throw std::overflow_error("quantize: INT4 scale is not representable");
            z = int(std::nearbyint(-lo / double(s)));
            z = std::max(0, std::min(15, z));
        };
        for_rows(N, count, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            const float* in  = src + int64_t(n) * K;
            uint8_t*     row = w.payload.data() + int64_t(n) * w.row_bytes;
            for (int g = 0; g < G; ++g) {
                const int base = g * GS;
                // The stored zero point is an integer in [0,15], so zero
                // must lie inside the represented range. Clamping only the
                // zero point after using [min,max] collapses one-sided groups.
                float lo = 0.0f, hi = 0.0f;
                for (int i = 0; i < GS; ++i) {
                    lo = std::min(lo, in[base + i]);
                    hi = std::max(hi, in[base + i]);
                }
                bf16_t sb; float s; int z;
                pick(lo, hi, sb, s, z);
                if (rtn_search || importance) {
                    double best = -1.0;
                    for (int c = 0; c <= 12; ++c) {
                        const double f = 1.0 - 0.02 * c;
                        bf16_t cb; float cs; int cz;
                        pick(double(lo) * f, double(hi) * f, cb, cs, cz);
                        double err = 0.0;
                        for (int i = 0; i < GS; ++i) {
                            int q = int(std::nearbyint(double(in[base + i]) / cs)) + cz;
                            q = std::max(0, std::min(15, q));
                            const double d = double(in[base + i]) - (double(q) - cz) * cs;
                            err += importance ? d * d * double(importance[base + i]) : d * d;
                        }
                        if (best < 0.0 || err < best) { best = err; sb = cb; s = cs; z = cz; }
                    }
                }
                sc[int64_t(n) * G + g] = sb;
                w.zeros[int64_t(n) * G + g] = uint8_t(z);
                for (int i = 0; i < GS; ++i) {
                    int q = int(std::nearbyint(double(in[base + i]) / s)) + z;
                    q = std::max(0, std::min(15, q));
                    if (!std::isfinite(float((double(q) - z) * s)))
                        throw std::overflow_error("quantize: INT4 reconstructed value overflows");
                    const int k = base + i;
                    if (k & 1) row[k >> 1] = uint8_t((row[k >> 1] & 0x0F) | (q << 4));
                    else       row[k >> 1] = uint8_t((row[k >> 1] & 0xF0) | q);
                }
            }
        }
        });
        break;
    }

    // -----------------------------------------------------------------
    // OCP Microscaling. Block of 32 along K, one E8M0 byte per block.
    case Fmt::MXFP8:
    case Fmt::MXFP4: {
        const int B      = w.row_scales;
        const int emax   = (fmt == Fmt::MXFP8) ? kEmaxE4M3 : kEmaxE2M1;
        w.scales_raw.assign(size_t(N) * B, 0);
        for_rows(N, count, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            const float* in  = src + int64_t(n) * K;
            uint8_t*     row = w.payload.data() + int64_t(n) * w.row_bytes;
            for (int b = 0; b < B; ++b) {
                const int   base = b * kMXBlock;
                const float am   = absmax(in + base, kMXBlock);
                const uint8_t X  = pick_e8m0(am, emax);
                w.scales_raw[int64_t(n) * B + b] = X;
                const float inv = 1.0f / e8m0_to_f32(X);
                for (int i = 0; i < kMXBlock; ++i) {
                    const float v = in[base + i] * inv;
                    const int   k = base + i;
                    if (fmt == Fmt::MXFP8) {
                        row[k] = f32_to_e4m3(v);
                    } else {
                        const uint8_t q = f32_to_e2m1(v);
                        if (k & 1) row[k >> 1] = uint8_t((row[k >> 1] & 0x0F) | (q << 4));
                        else       row[k >> 1] = uint8_t((row[k >> 1] & 0xF0) | q);
                    }
                }
            }
        }
        });
        break;
    }
    case Fmt::Q4_K: case Fmt::Q6_K: case Fmt::Q5_K: case Fmt::Q8_0:
    case Fmt::IQ4_XS: case Fmt::IQ4_NL:
        throw std::invalid_argument("quantize: GGUF formats require raw GGUF blocks");
    }

    return w;
}

} // namespace b70

// =====================================================================
//  llama.cpp importance matrices (b70/imatrix.hpp).  Here rather than in a
//  file of their own so every binary that links the quantizer has them
//  without a build-script change.
// =====================================================================
#include "b70/imatrix.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace b70 {
namespace {

struct File {
    FILE* f = nullptr;
    ~File() { if (f) std::fclose(f); }
    bool read(void* p, size_t n) { return std::fread(p, 1, n, f) == n; }
    template <class T> bool get(T& v) { return read(&v, sizeof(T)); }
};

bool gguf_string(File& f, std::string& s) {
    uint64_t n = 0;
    if (!f.get(n) || n > (uint64_t(1) << 20)) return false;
    s.resize(size_t(n));
    return n == 0 || f.read(&s[0], size_t(n));
}

// Skip one GGUF metadata value of type t (reading general.alignment on the way).
bool gguf_skip(File& f, uint32_t t, uint64_t* as_u64) {
    static const int width[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (t == 8) { std::string s; return gguf_string(f, s); }
    if (t == 9) {
        uint32_t et = 0; uint64_t n = 0;
        if (!f.get(et) || !f.get(n)) return false;
        for (uint64_t i = 0; i < n; ++i) if (!gguf_skip(f, et, nullptr)) return false;
        return true;
    }
    if (t > 12) return false;
    uint64_t v = 0;
    if (!f.read(&v, size_t(width[t]))) return false;
    if (as_u64) *as_u64 = v;
    return true;
}

}  // namespace

bool ImportanceMatrix::load(const std::string& path, std::string& err) {
    m_.clear();
    File f; f.f = std::fopen(path.c_str(), "rb");
    if (!f.f) { err = "cannot open " + path; return false; }
    char magic[4] = {};
    if (!f.read(magic, 4)) { err = "empty file"; return false; }
    if (std::memcmp(magic, "GGUF", 4) == 0) {
        uint32_t ver = 0; uint64_t nt = 0, nkv = 0;
        if (!f.get(ver) || ver < 2 || !f.get(nt) || !f.get(nkv)) { err = "unsupported GGUF header"; return false; }
        uint64_t align = 32;
        for (uint64_t i = 0; i < nkv; ++i) {
            std::string k; uint32_t t = 0; uint64_t v = 0;
            if (!gguf_string(f, k) || !f.get(t) || !gguf_skip(f, t, &v)) { err = "bad GGUF metadata"; return false; }
            if (k == "general.alignment" && v > 0) align = v;
        }
        struct TI { std::string name; std::vector<uint64_t> dims; uint32_t type = 0; uint64_t off = 0; };
        std::vector<TI> ti(static_cast<size_t>(nt));
        for (auto& t : ti) {
            uint32_t nd = 0;
            if (!gguf_string(f, t.name) || !f.get(nd) || nd > 4) { err = "bad GGUF tensor info"; return false; }
            t.dims.resize(nd);
            for (auto& d : t.dims) if (!f.get(d)) { err = "bad GGUF tensor info"; return false; }
            if (!f.get(t.type) || !f.get(t.off)) { err = "bad GGUF tensor info"; return false; }
        }
        const long here = std::ftell(f.f);
        const uint64_t base = (uint64_t(here) + align - 1) / align * align;
        std::unordered_map<std::string, const TI*> by;
        for (auto& t : ti) by[t.name] = &t;
        auto read_f32 = [&](const TI& t, std::vector<float>& out) {
            uint64_t n = 1; for (auto d : t.dims) n *= d;
            if (t.type != 0 || n > (uint64_t(1) << 32)) return false;  // F32 only
            out.resize(size_t(n));
            return std::fseek(f.f, long(base + t.off), SEEK_SET) == 0 && f.read(out.data(), size_t(n) * 4);
        };
        const std::string sfx = ".in_sum2";
        for (auto& t : ti) {
            if (t.name.size() <= sfx.size() || t.name.compare(t.name.size() - sfx.size(), sfx.size(), sfx)) continue;
            const std::string base_name = t.name.substr(0, t.name.size() - sfx.size());
            auto c = by.find(base_name + ".counts");
            if (c == by.end() || t.dims.empty()) continue;
            std::vector<float> s2, cnt;
            if (!read_f32(t, s2) || !read_f32(*c->second, cnt)) { err = "cannot read " + t.name; return false; }
            Entry e; e.K = int(t.dims[0]); e.n_mat = int(s2.size() / size_t(e.K));
            if (e.K <= 0 || cnt.size() != size_t(e.n_mat)) continue;
            e.v.resize(s2.size());
            for (int m = 0; m < e.n_mat; ++m)
                for (int k = 0; k < e.K; ++k)
                    e.v[size_t(m) * e.K + k] = cnt[size_t(m)] > 0.f ? s2[size_t(m) * e.K + k] / cnt[size_t(m)] : 0.f;
            m_[base_name] = std::move(e);
        }
        return true;
    }
    // imatrix.dat: int32 n; n x { int32 len, name, int32 ncall, int32 nval, f32[nval] }.
    // Each value is mean * ncall; only the relative size within a tensor matters here.
    std::fseek(f.f, 0, SEEK_SET);
    int32_t n = 0;
    if (!f.get(n) || n < 0 || n > (1 << 20)) { err = "neither a GGUF nor an imatrix.dat file"; return false; }
    for (int32_t i = 0; i < n; ++i) {
        int32_t len = 0, ncall = 0, nval = 0;
        if (!f.get(len) || len <= 0 || len > 4096) { err = "bad imatrix.dat entry"; return false; }
        std::string name(size_t(len), ' ');
        if (!f.read(&name[0], size_t(len)) || !f.get(ncall) || !f.get(nval) || nval <= 0) { err = "bad imatrix.dat entry"; return false; }
        Entry e; e.v.resize(size_t(nval));
        if (!f.read(e.v.data(), size_t(nval) * 4)) { err = "short imatrix.dat"; return false; }
        e.K = nval; e.n_mat = 1;  // a merged expert entry's width is resolved by find()
        m_[name] = std::move(e);
    }
    return true;
}

const float* ImportanceMatrix::find(const std::string& tensor_name, int K, bool count) const {
    int expert = -1;
    const std::string g = imatrix_gguf_name(tensor_name, &expert);
    auto it = g.empty() ? m_.end() : m_.find(g);
    if (it == m_.end() || K <= 0) { if (count) ++misses; return nullptr; }
    const Entry& e = it->second;
    const int n_mat = e.K == K ? e.n_mat : (e.n_mat == 1 && int(e.v.size()) % K == 0 ? int(e.v.size()) / K : 0);
    const int row = expert < 0 ? 0 : expert;
    if (n_mat <= 0 || (expert < 0 && n_mat != 1) || row >= n_mat) { if (count) ++misses; return nullptr; }
    if (count) ++hits;
    return e.v.data() + size_t(row) * size_t(K);
}

std::string imatrix_gguf_name(const std::string& name, int* expert) {
    if (expert) *expert = -1;
    if (name.compare(0, 4, "blk.") == 0 || name == "output.weight") return name;
    if (name == "lm_head.weight") return "output.weight";
    // The MTP head's "layers.0" is not the trunk's layer 0 (llama.cpp files it as
    // blk.<n_layer>, which only the config knows): no importance rather than the
    // wrong layer's.
    if (name.compare(0, 4, "mtp.") == 0 || name.find(".mtp.") != std::string::npos) return "";
    // [model.][language_model.]layers.N.rest  (also model.language_model.layers.N.rest)
    const std::string key = "layers.";
    const size_t at = name.find(key);
    if (at == std::string::npos) return "";
    size_t p = at + key.size(), q = p;
    while (q < name.size() && name[q] >= '0' && name[q] <= '9') ++q;
    if (q == p || q >= name.size() || name[q] != '.') return "";
    const std::string layer = name.substr(p, q - p);
    std::string rest = name.substr(q + 1);
    // per-expert HF tensors: mlp.experts.E.{gate,up,down}_proj.weight
    const std::string ex = "mlp.experts.";
    if (rest.compare(0, ex.size(), ex) == 0) {
        size_t a = ex.size(), b = a;
        while (b < rest.size() && rest[b] >= '0' && rest[b] <= '9') ++b;
        if (b == a || b >= rest.size()) return "";
        if (expert) *expert = std::atoi(rest.substr(a, b - a).c_str());
        rest = "mlp.experts." + rest.substr(b + 1);
    }
    static const std::pair<const char*, const char*> map[] = {
        {"self_attn.q_proj.weight", "attn_q.weight"},
        {"self_attn.k_proj.weight", "attn_k.weight"},
        {"self_attn.v_proj.weight", "attn_v.weight"},
        {"self_attn.o_proj.weight", "attn_output.weight"},
        {"linear_attn.in_proj_qkv.weight", "attn_qkv.weight"},
        {"linear_attn.in_proj_qkvz.weight", "attn_qkv.weight"},   // Qwen3-Next: same input
        {"linear_attn.in_proj_z.weight", "attn_gate.weight"},
        {"linear_attn.in_proj_a.weight", "ssm_alpha.weight"},
        {"linear_attn.in_proj_b.weight", "ssm_beta.weight"},
        {"linear_attn.in_proj_ba.weight", "ssm_beta.weight"},     // same input
        {"linear_attn.out_proj.weight", "ssm_out.weight"},
        {"mlp.gate_proj.weight", "ffn_gate.weight"},
        {"mlp.up_proj.weight", "ffn_up.weight"},
        {"mlp.down_proj.weight", "ffn_down.weight"},
        {"mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"},
        {"mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"},
        {"mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight"},
        {"mlp.gate.weight", "ffn_gate_inp.weight"},
        {"mlp.experts.gate_proj.weight", "ffn_gate_exps.weight"},
        {"mlp.experts.up_proj.weight", "ffn_up_exps.weight"},
        {"mlp.experts.down_proj.weight", "ffn_down_exps.weight"},
    };
    for (const auto& m : map)
        if (rest == m.first) return "blk." + layer + "." + m.second;
    return "";
}

}  // namespace b70
