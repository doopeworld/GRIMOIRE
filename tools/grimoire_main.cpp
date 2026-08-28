// =====================================================================
//  grimoire  --  CLI entry point
// =====================================================================
#include "b70/formats.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>

namespace b70 {
int grimoire_load_report(const std::string& dir, Fmt proj_fmt, int max_seq);
int grimoire_generate(const std::string& dir, Fmt proj_fmt, int max_seq,
                      const std::string& prompt, int n_predict);
int grimoire_prefill_only(const std::string& dir, Fmt proj_fmt, int max_seq,
                          int n_tokens);
int grimoire_prefix_cache_test(const std::string& dir, Fmt proj_fmt, int max_seq,
                               int n_tokens);
}

int main(int argc, char** argv) {
    std::string model, prompt;
    b70::Fmt pf = b70::Fmt::INT4;
    int max_seq = 8192;
    int n_predict = 128;
    int prefill_only = 0;
    int prefix_cache_test = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model = argv[++i];
        else if (a == "--proj" && i + 1 < argc) {
            const std::string v = argv[++i];
            if      (v == "int4")  pf = b70::Fmt::INT4;
            else if (v == "int8")  pf = b70::Fmt::INT8;
            else if (v == "mxfp4") pf = b70::Fmt::MXFP4;
            else if (v == "mxfp8") pf = b70::Fmt::MXFP8;
            else if (v == "fp8" || v == "fp8_e4m3") pf = b70::Fmt::FP8_E4M3;
            else if (v == "fp8_e5m2") pf = b70::Fmt::FP8_E5M2;
            else if (v == "bf16")  pf = b70::Fmt::BF16;
            else {
                std::fprintf(stderr, "unknown --proj format: %s\n", v.c_str());
                return 1;
            }
        }
        else if (a == "--ctx" && i + 1 < argc) max_seq = std::atoi(argv[++i]);
        else if ((a == "-p" || a == "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if ((a == "-n" || a == "--n-predict") && i + 1 < argc) n_predict = std::atoi(argv[++i]);
        else if (a == "--prefill-only" && i + 1 < argc) prefill_only = std::atoi(argv[++i]);
        else if (a == "--prefix-cache-test" && i + 1 < argc) prefix_cache_test = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::printf(
              "grimoire -- bare-metal inference for Intel Arc Pro (Battlemage)\n\n"
              "  grimoire -m <model_dir> [--proj FORMAT] [--ctx N]\n\n"
              "  FORMAT   int4, int8, mxfp4, mxfp8, fp8_e4m3, fp8_e5m2, bf16\n"
              "  --proj   format for the projections quantized at load.\n"
              "           int4 is the default: it takes the deltanet\n"
              "           projections and lm_head from 3.0 GB to 0.8 GB\n"
              "           per token, the largest single win in decode.\n"
              "           bf16 keeps them as the checkpoint stores them.\n"
              "  --ctx    KV cache capacity for the full-attention layers.\n"
              "           The linear layers do not grow with context.\n"
              "  -p       prompt to generate from. Without it, runs benchmarks.\n"
              "  -n       tokens to generate (default 128).\n");
            return 0;
        }
    }
    if (model.empty()) { std::printf("need -m <model_dir>; see --help\n"); return 1; }
    if (prefix_cache_test > 0)
        return b70::grimoire_prefix_cache_test(model, pf, max_seq, prefix_cache_test);
    if (prefill_only > 0)
        return b70::grimoire_prefill_only(model, pf, max_seq, prefill_only);
    if (!prompt.empty())
        return b70::grimoire_generate(model, pf, max_seq, prompt, n_predict);
    return b70::grimoire_load_report(model, pf, max_seq);
}
