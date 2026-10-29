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
int grimoire_score_dump(const std::string& dir, Fmt proj_fmt, int max_seq,
                        const std::string& ids_file, const std::string& out_file);
}

int main(int argc, char** argv) {
    std::string model, prompt;
    b70::Fmt pf = b70::Fmt::INT4;
    int max_seq = 8192;
    int n_predict = 128;
    int prefill_only = 0;
    int prefix_cache_test = 0;
    std::string score_ids, score_out;

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
        else if (a == "--score-ids" && i + 1 < argc) score_ids = argv[++i];
        else if (a == "--score-out" && i + 1 < argc) score_out = argv[++i];
        else if (a == "-h" || a == "--help") {
            std::printf(
              "grimoire -- bare-metal inference for Intel Arc Pro (Battlemage)\n\n"
              "  grimoire -m <model_dir> [--proj FORMAT] [--ctx N]\n\n"
              "  FORMAT   int4, int8, mxfp4, mxfp8, fp8_e4m3, fp8_e5m2, bf16\n"
              "  --proj   format for projections converted from safetensors at load.\n"
              "           Native v3 artifacts preserve their saved formats.\n"
              "           Use grimoire-quantize to choose formats offline.\n"
              "  --ctx    KV cache capacity for the full-attention layers.\n"
              "           The linear layers do not grow with context.\n"
              "  -p       prompt to generate from. Without it, runs benchmarks.\n"
              "  -n       tokens to generate (default 128).\n"
              "  --score-ids FILE --score-out FILE\n"
              "           feed the token ids in FILE (one per line) one at a time and write\n"
              "           the logits after each (fp32, vocab per position) -- for comparing\n"
              "           with a reference.  GRIMOIRE_LM_HEAD_BF16=1 keeps the checkpoint's\n"
              "           output head instead of converting it to --proj.\n");
#ifdef GRIMOIRE_NATIVE_ONLY
            std::printf("\n  Standalone native build: external framework bridges disabled.\n");
#endif
            return 0;
        }
    }
    if (model.empty()) { std::printf("need -m <model_dir>; see --help\n"); return 1; }
    if (!score_ids.empty() || !score_out.empty()) {
        if (score_ids.empty() || score_out.empty()) {
            std::printf("--score-ids and --score-out go together\n"); return 1;
        }
        return b70::grimoire_score_dump(model, pf, max_seq, score_ids, score_out);
    }
    if (prefix_cache_test > 0)
        return b70::grimoire_prefix_cache_test(model, pf, max_seq, prefix_cache_test);
    if (prefill_only > 0)
        return b70::grimoire_prefill_only(model, pf, max_seq, prefill_only);
    if (!prompt.empty())
        return b70::grimoire_generate(model, pf, max_seq, prompt, n_predict);
    return b70::grimoire_load_report(model, pf, max_seq);
}
