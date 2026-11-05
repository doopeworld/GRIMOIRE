// Host test: llama.cpp importance matrices (b70/imatrix.hpp) and INT4 rounding by them.
//
//   c++ -O2 -std=c++17 -pthread -Iinclude tools/test_imatrix.cpp src/quantize.cpp -o test_imatrix
//   ./test_imatrix                                    synthetic check only
//   ./test_imatrix FILE.imatrix                       + reads a real file, checks the name mapping
//   ./test_imatrix FILE.imatrix SHARD OFF N K NAME    + one real BF16 weight at byte offset OFF:
//                                                       weighted output error, plain vs importance
#include "b70/imatrix.hpp"
#include "b70/weights.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace b70;

static float bf16f(uint16_t h) { uint32_t u = uint32_t(h) << 16; float f; std::memcpy(&f, &u, 4); return f; }

// Dequantize GRIMOIRE's INT4 (row-major payload, bf16 scale + uint8 zero per group).
static std::vector<float> deq(const PackedWeight& p) {
    const int N = p.N, K = p.K, G = p.row_scales, GS = K / G;
    std::vector<float> out(size_t(N) * K);
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(p.scales_raw.data());
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            const uint8_t b = p.payload[size_t(n) * p.row_bytes + size_t(k >> 1)];
            const int q = (k & 1) ? (b >> 4) : (b & 15);
            const int g = k / GS;
            out[size_t(n) * K + k] = (float(q) - float(p.zeros[size_t(n) * G + g])) * bf16f(sc[size_t(n) * G + g]);
        }
    return out;
}

static double werr(const std::vector<float>& W, const std::vector<float>& Q, int N, int K, const float* m) {
    double e = 0, w = 0;
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            const double d = double(W[size_t(n) * K + k]) - Q[size_t(n) * K + k], mk = m ? m[k] : 1.0;
            e += d * d * mk; w += double(W[size_t(n) * K + k]) * W[size_t(n) * K + k] * mk;
        }
    return e / w;
}

int main(int argc, char** argv) {
    int fails = 0;
    // 1. synthetic: 8 loud channels of 1024; the importance-weighted rounding must
    //    lower the weighted error and not raise the plain one by much.
    {
        const int N = 256, K = 1024;
        std::mt19937 rng(7);
        std::student_t_distribution<float> st(4.0f);
        std::vector<float> W(size_t(N) * K), m(size_t(K), 1.0f);
        for (auto& v : W) v = 0.02f * st(rng);
        for (int c = 0; c < 8; ++c) m[size_t(c * 127)] = 400.0f;
        const auto p0 = quantize(W.data(), N, K, Fmt::INT4);
        const auto p1 = quantize(W.data(), N, K, Fmt::INT4, m.data());
        const auto q0 = deq(p0), q1 = deq(p1);
        const double w0 = werr(W, q0, N, K, m.data()), w1 = werr(W, q1, N, K, m.data());
        std::printf("synthetic: weighted error plain %.4e, importance %.4e (%.2fx)\n", w0, w1, w1 / w0);
        if (!(w1 < 0.9 * w0)) { std::printf("FAIL: importance did not lower the weighted error\n"); ++fails; }
        if (p1.row_scales != p0.row_scales || p1.payload.size() != p0.payload.size()) { std::printf("FAIL: layout changed\n"); ++fails; }
    }
    if (argc < 2) { std::printf(fails ? "FAILED\n" : "PASS\n"); return fails ? 1 : 0; }

    // 2. a real file: entries, HF -> GGUF names, refusals.
    ImportanceMatrix im; std::string err;
    if (!im.load(argv[1], err)) { std::printf("FAIL: load: %s\n", err.c_str()); return 1; }
    std::printf("%s: %zu entries\n", argv[1], im.entries());
    struct Case { const char* name; int K; bool want; };
    const Case cases[] = {
        {"model.language_model.layers.10.linear_attn.in_proj_qkv.weight", 5120, true},
        {"model.language_model.layers.10.linear_attn.in_proj_z.weight", 5120, true},
        {"model.language_model.layers.10.linear_attn.out_proj.weight", 6144, true},
        {"model.language_model.layers.11.self_attn.o_proj.weight", 6144, true},
        {"model.language_model.layers.11.mlp.down_proj.weight", 17408, true},
        {"blk.11.attn_q.weight", 5120, true},
        {"model.language_model.layers.11.self_attn.o_proj.weight", 5120, false},  // wrong width
        {"mtp.layers.0.self_attn.q_proj.weight", 5120, false},                   // not trunk layer 0
        {"model.visual.blocks.0.attn.qkv.weight", 1152, false},
    };
    for (const auto& c : cases) {
        const bool got = im.find(c.name, c.K) != nullptr;
        std::printf("  %-66s K=%-6d %s\n", c.name, c.K, got ? "found" : "none");
        if (got != c.want) { std::printf("FAIL: expected %s\n", c.want ? "found" : "none"); ++fails; }
    }

    // 3. a real BF16 weight from a safetensors shard.
    if (argc >= 7) {
        const long long off = std::atoll(argv[3]);
        const int N = std::atoi(argv[4]), K = std::atoi(argv[5]);
        const float* m = im.find(argv[6], K);
        if (!m) { std::printf("FAIL: no importance for %s\n", argv[6]); return 1; }
        FILE* f = std::fopen(argv[2], "rb");
        std::vector<uint16_t> raw(size_t(N) * K);
        if (!f || std::fseek(f, long(off), SEEK_SET) || std::fread(raw.data(), 2, raw.size(), f) != raw.size()) { std::printf("FAIL: read\n"); return 1; }
        std::fclose(f);
        std::vector<float> W(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) W[i] = bf16f(raw[i]);
        const auto q0 = deq(quantize(W.data(), N, K, Fmt::INT4));
        const auto q1 = deq(quantize(W.data(), N, K, Fmt::INT4, m));
        const double w0 = werr(W, q0, N, K, m), w1 = werr(W, q1, N, K, m);
        const double p0 = werr(W, q0, N, K, nullptr), p1 = werr(W, q1, N, K, nullptr);
        std::printf("%s [%d x %d]: weighted output error %.3e -> %.3e (%.2fx), plain %.3e -> %.3e\n",
                    argv[6], N, K, w0, w1, w1 / w0, p0, p1);
    }
    std::printf(fails ? "FAILED\n" : "PASS\n");
    return fails ? 1 : 0;
}
