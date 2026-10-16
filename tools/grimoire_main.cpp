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
#include <climits>

namespace b70 {
int grimoire_load_report(const std::string& dir, Fmt proj_fmt, int max_seq);
int grimoire_generate(const std::string& dir, Fmt proj_fmt, int max_seq,
                      const std::string& prompt, int n_predict);
int grimoire_prefill_only(const std::string& dir, Fmt proj_fmt, int max_seq,
                          int n_tokens);
int grimoire_prefix_cache_test(const std::string& dir, Fmt proj_fmt, int max_seq,
                               int n_tokens);
int grimoire_perplexity(const std::string& dir, Fmt proj_fmt,
                       const std::string& path, int ctx, int chunks,
                       const std::string& dump, bool selfcheck,
                       const std::string& kld_base);
}

int main(int argc, char** argv) {
    std::string model, prompt;
    b70::Fmt pf = b70::Fmt::INT4;
    bool proj_explicit = false;
    int max_seq = 8192;
    int n_predict = 128;
    int prefill_only = 0;
    int prefix_cache_test = 0;
    std::string ppl_file, ppl_dump, ppl_kld_base;
    int ppl_ctx = 0, ppl_chunks = 0;
    bool ppl_selfcheck = false;
    bool ppl_option = false, ctx_given = false;
    auto positive = [](const char* value, int& out) {
        char* end = nullptr;
        const long n = std::strtol(value, &end, 10);
        if (!*value || *end || n <= 0 || n > INT_MAX) return false;
        out = int(n);
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model = argv[++i];
        else if (a == "--proj" && i + 1 < argc) {
            const std::string v = argv[++i];
            proj_explicit = true;
            if      (v == "int4")  pf = b70::Fmt::INT4;
            else if (v == "int8")  pf = b70::Fmt::INT8;
            else if (v == "mxfp4") pf = b70::Fmt::MXFP4;
            else if (v == "mxfp8") pf = b70::Fmt::MXFP8;
            else if (v == "fp8" || v == "fp8_e4m3") pf = b70::Fmt::FP8_E4M3;
            else if (v == "fp8_e5m2") pf = b70::Fmt::FP8_E5M2;
            else if (v == "bf16")  pf = b70::Fmt::BF16;
            else if (v == "native") { pf = b70::Fmt::MXFP4; setenv("GRIMOIRE_GGUF_NATIVE","1",1); }
            else {
                std::fprintf(stderr, "unknown --proj format: %s\n", v.c_str());
                return 1;
            }
        }
        else if (a == "--ctx" && i + 1 < argc) { max_seq = std::atoi(argv[++i]); ctx_given = true; }
        else if ((a == "-p" || a == "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if ((a == "-n" || a == "--n-predict") && i + 1 < argc) n_predict = std::atoi(argv[++i]);
        else if (a == "--prefill-only" && i + 1 < argc) prefill_only = std::atoi(argv[++i]);
        else if (a == "--prefix-cache-test" && i + 1 < argc) prefix_cache_test = std::atoi(argv[++i]);
        else if (a == "--ppl" && i + 1 < argc) ppl_file = argv[++i];
        else if (a == "--ppl-kld-base" && i + 1 < argc) { ppl_kld_base = argv[++i]; ppl_option = true; }
        else if (a == "--ppl-ctx" && i + 1 < argc) {
            ppl_option = true;
            if (!positive(argv[++i], ppl_ctx)) { std::fprintf(stderr, "--ppl-ctx must be positive\n"); return 1; }
        }
        else if (a == "--ppl-chunks" && i + 1 < argc) {
            ppl_option = true;
            if (!positive(argv[++i], ppl_chunks)) { std::fprintf(stderr, "--ppl-chunks must be positive\n"); return 1; }
        }
        else if (a == "--ppl-dump" && i + 1 < argc) { ppl_dump = argv[++i]; ppl_option = true; }
        else if (a == "--ppl-selfcheck") { ppl_selfcheck = true; ppl_option = true; }
        else if (a == "-h" || a == "--help") {
            std::printf(
              "grimoire -- bare-metal inference for Intel Arc Pro (Battlemage)\n\n"
              "  grimoire -m <model_dir> [--proj FORMAT] [--ctx N]\n\n"
              "  FORMAT   int4, int8, mxfp4, mxfp8, fp8_e4m3, fp8_e5m2, bf16, native (GGUF)\n"
              "  --proj   format for projections converted from safetensors at load.\n"
              "           native keeps GGUF Q4_K/Q6_K; other projections use MXFP4.\n"
              "           Native v3 artifacts preserve their saved formats.\n"
              "           Use grimoire-quantize to choose formats offline.\n"
              "  --ctx    KV cache capacity for the full-attention layers.\n"
              "           The linear layers do not grow with context.\n"
              "  -p       prompt to generate from. Without it, runs benchmarks.\n"
              "  -n       tokens to generate (default 128).\n");
            std::printf("  --ppl FILE          raw-text perplexity (no chat template).\n"
                        "  --ppl-kld-base FILE compare against llama-perplexity base log-probs; --ppl FILE optional.\n"
                        "  --ppl-ctx N         tokens per chunk (default 512, or base file's n_ctx).\n"
                        "  --ppl-chunks C      score at most C complete chunks.\n"
                        "  --ppl-dump FILE     per-token TSV logits summary.\n"
                        "  --ppl-selfcheck     compare last scored logits with prefill.\n");
#ifdef GRIMOIRE_NATIVE_ONLY
            std::printf("\n  Standalone native build: external framework bridges disabled.\n");
#endif
            return 0;
        }
    }
    const char* gguf_native = std::getenv("GRIMOIRE_GGUF_NATIVE");
    if (gguf_native && std::strcmp(gguf_native, "1") == 0) {
        if (proj_explicit && pf != b70::Fmt::MXFP4) {
            std::fprintf(stderr, "native GGUF requires --proj native or mxfp4 for other tensors\n");
            return 1;
        }
        pf = b70::Fmt::MXFP4;
    }
    if (model.empty()) { std::printf("need -m <model_dir>; see --help\n"); return 1; }
    if (!ppl_file.empty() || !ppl_kld_base.empty()) {
        if ((ppl_ctx && ppl_ctx < 4) || ctx_given || !prompt.empty() || prefill_only || prefix_cache_test ||
            n_predict != 128) {
            std::fprintf(stderr, "--ppl needs --ppl-ctx >= 4 and cannot combine with --ctx, generation, or benchmark modes\n");
            return 1;
        }
        return b70::grimoire_perplexity(model, pf, ppl_file, ppl_ctx ? ppl_ctx : (ppl_kld_base.empty() ? 512 : 0),
                                        ppl_chunks, ppl_dump, ppl_selfcheck, ppl_kld_base);
    }
    if (ppl_option) {
        std::fprintf(stderr, "--ppl-* requires --ppl FILE or --ppl-kld-base FILE\n"); return 1;
    }
    if (prefix_cache_test > 0)
        return b70::grimoire_prefix_cache_test(model, pf, max_seq, prefix_cache_test);
    if (prefill_only > 0)
        return b70::grimoire_prefill_only(model, pf, max_seq, prefill_only);
    if (!prompt.empty())
        return b70::grimoire_generate(model, pf, max_seq, prompt, n_predict);
    return b70::grimoire_load_report(model, pf, max_seq);
}
