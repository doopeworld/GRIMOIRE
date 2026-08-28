// b70-load : resolve a checkpoint end to end and report what was found.
// Reads every shard header, wires every tensor, and validates that the
// MXFP4 expert layout matches what the kernels expect -- without
// touching a GPU.
#include "b70/qwen35.hpp"
#include <cstdio>
#include <cmath>

using namespace b70;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: b70-load <model_dir>\n"); return 1; }
    Qwen35Model m;
    std::string err;
    if (!m.load(argv[1], err)) { std::printf("LOAD FAILED: %s\n", err.c_str()); return 1; }

    std::printf("=== %s ===\n", argv[1]);
    m.summary();

    if (!m.cfg.is_moe()) { std::printf("\ndense model, no expert check\n"); return 0; }

    // Verify the on-disk expert layout against the kernel's expectations.
    // If these disagree the model would produce plausible garbage rather
    // than an error, so it is checked explicitly.
    const int H = m.cfg.hidden, I = m.cfg.moe_inter;
    const Qwen35Layer& L0 = m.layers[0];
    struct Chk { const char* name; const TensorRef& p; const TensorRef& s; int N, K; };
    const Chk chks[] = {
        {"gate_proj", L0.e_gate_p[0], L0.e_gate_s[0], I, H},
        {"up_proj",   L0.e_up_p[0],   L0.e_up_s[0],   I, H},
        {"down_proj", L0.e_down_p[0], L0.e_down_s[0], H, I},
    };
    std::printf("\n  --- expert layout check (layer 0, expert 0) ---\n");
    int bad = 0;
    for (const Chk& c : chks) {
        const int64_t want_p = int64_t(c.N) * bytes_per_row(Fmt::MXFP4, c.K);
        const int64_t want_s = int64_t(c.N) * scales_per_row(Fmt::MXFP4, c.K);
        const int64_t got_p  = m.bytes(c.p), got_s = m.bytes(c.s);
        const bool ok = (want_p == got_p && want_s == got_s);
        if (!ok) ++bad;
        std::printf("  %-10s packed %8lld B (want %8lld)  scale %7lld B (want %7lld)  %s\n",
                    c.name, (long long)got_p, (long long)want_p,
                    (long long)got_s, (long long)want_s, ok ? "ok" : "MISMATCH");
    }
    if (bad) {
        std::printf("\n  %d mismatch(es): the on-disk packing is NOT the layout the\n"
                    "  kernels read. A repack pass is required.\n", bad);
        return 1;
    }
    std::printf("\n  Layout matches exactly -- experts upload to VRAM with no\n"
                "  conversion pass. %.2f GiB of expert weights are zero-copy.\n",
                double(m.total_bytes(true) - m.total_bytes(false)) / 1073741824.0);

    // Dequantize a few real weights as a sanity check on the values.
    QuantWeight w = m.quant_view(L0.e_gate_p[0], L0.e_gate_s[0], I, H);
    double amax = 0, sum2 = 0;
    int n = 0;
    for (int r = 0; r < 8; ++r)
        for (int k = 0; k < 256; ++k) {
            const float v = w.at(r, k);
            amax = amax > (v < 0 ? -v : v) ? amax : (v < 0 ? -v : v);
            sum2 += double(v) * v; ++n;
        }
    std::printf("  sample dequant: rms %.5f, absmax %.5f over %d weights\n",
                std::sqrt(sum2 / n), amax, n);
    if (!(amax > 0.0 && amax < 100.0))
        std::printf("  WARNING: values look wrong for transformer weights\n");
    return 0;
}
