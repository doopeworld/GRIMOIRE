// =====================================================================
//  test_engine_plan.cpp  --  the decode-step budget for the real model,
//  computed from MEASURED kernel rates rather than assumed ones.
//
//  Every bandwidth figure below came off the B70 in this project's own
//  benchmark. The point is to find what dominates a token before
//  building the driver, so the driver is built around the right thing.
// =====================================================================
#include "b70/formats.hpp"
#include <cstdio>
#include <cmath>
#include <initializer_list>

using namespace b70;

// measured on Arc Pro B70, 16384^2 GEMV, best of 3 x 50
static double rate(Fmt f) {
    switch (f) {
        case Fmt::BF16:  return 542.0;
        case Fmt::INT8:  return 527.0;
        case Fmt::INT4:  return 478.0;
        case Fmt::MXFP4: return 352.0;
        case Fmt::MXFP8: return 386.0;
        default:         return 380.0;
    }
}
// measured: fused MoE block, one layer, 256 experts top-8
static double fused_moe_ms(Fmt f) { return f == Fmt::INT4 ? 0.0395 : 0.0479; }

static double gemv_ms(double elems, Fmt f) {
    return elems * bits_per_elem(f) / 8.0 / 1e9 / rate(f) * 1e3;
}

int main() {
    const int H = 2048, V = 248320, L = 40, NLIN = 30, NFULL = 10;
    const int HK = 16, DK = 128, HV = 32, DV = 128;
    const int SHARED = 512;
    const double LAUNCH_US = 5.0;

    std::printf("=== decode-step budget: Qwen3.5-MoE 35B-A3B on one B70 ===\n\n");

    for (Fmt ef : { Fmt::MXFP4, Fmt::INT4 }) {
        for (Fmt lf : { Fmt::BF16, Fmt::INT4 }) {
            double ms = 0; int launches = 0;

            const double moe = fused_moe_ms(ef) * L;
            ms += moe; launches += 2 * L;

            // shared expert, bf16 in the checkpoint: 3 x [512][2048]
            const double sh = gemv_ms(3.0 * SHARED * H, Fmt::BF16) * L;
            ms += sh; launches += 3 * L;

            // gated deltanet projections, bf16
            const double lp = gemv_ms(double(2*HK*DK + HV*DV) * H     // qkv
                                    + double(HV*DV) * H               // z
                                    + 2.0 * HV * H                    // a, b
                                    + double(H) * HV * DV,            // out
                                      Fmt::BF16) * NLIN;
            ms += lp; launches += 6 * NLIN;

            // deltanet state: 2 MiB read + written per layer
            const double st = 2.0 * double(HV) * DK * DV * 4 / 1e9 / rate(Fmt::BF16) * 1e3 * NLIN;
            ms += st; launches += NLIN;

            // full attention projections + the attention itself
            const double fp = gemv_ms(4.0 * H * H, Fmt::BF16) * NFULL;
            ms += fp; launches += 7 * NFULL;

            // router: [256][2048] per layer
            const double rt = gemv_ms(256.0 * H, Fmt::BF16) * L;
            ms += rt; launches += L;

            // norms and residuals
            launches += 2 * L;

            const double lm = gemv_ms(double(V) * H, lf);
            ms += lm; launches += 2;

            const double lat = launches * LAUNCH_US / 1000.0;
            const double tot = ms + lat;

            std::printf("experts %-6s lm_head %-6s\n", fmt_name(ef), fmt_name(lf));
            std::printf("  %-22s %7.3f ms\n", "fused MoE (40 layers)", moe);
            std::printf("  %-22s %7.3f ms\n", "shared expert", sh);
            std::printf("  %-22s %7.3f ms\n", "deltanet proj", lp);
            std::printf("  %-22s %7.3f ms\n", "deltanet state", st);
            std::printf("  %-22s %7.3f ms\n", "full attn proj", fp);
            std::printf("  %-22s %7.3f ms\n", "router", rt);
            std::printf("  %-22s %7.3f ms  <-- %s\n", "lm_head", lm,
                        lf == Fmt::BF16 ? "unquantized, dominates" : "quantized");
            std::printf("  %-22s %7.3f ms  (%d launches)\n", "launch latency", lat, launches);
            std::printf("  %-22s %7.3f ms  ->  %.0f tok/s\n\n", "TOTAL", tot, 1000.0 / tot);
        }
    }

    std::printf("Takeaways\n");
    std::printf("  1. Quantizing lm_head is worth more than any remaining kernel work.\n");
    std::printf("  2. The shared expert and deltanet projections are bf16 in the\n");
    std::printf("     checkpoint; quantizing them is the next lever after that.\n");
    std::printf("  3. Launch latency is now comparable to memory time. Batching the\n");
    std::printf("     per-layer sequence into one command list is the structural fix.\n");
    return 0;
}
