// =====================================================================
//  test_qwen4_exp_e2e.cpp -- Qwen4-Exp (Qwen3.8-Flash-Next) end to end.
//
//  The matrix gate proves this architecture LOADS and GENERATES at every
//  projection format.  That is necessary and it is not enough: two of
//  the three mechanisms this model adds are FLUENT WHEN ABSENT.  Run QSA
//  as dense attention, or skip the PLE layer entirely, and the model
//  still emits in-vocabulary, reproducible, input-dependent text.  Every
//  check the matrix makes would pass.
//
//  So the checks here are A/Bs against the SAME WEIGHTS with one
//  mechanism turned off in the config.  The tensor list is byte
//  identical in both arms -- only config.json changes -- so the
//  fixture's RNG stream is identical and any difference in the output is
//  the mechanism, not different random numbers.
//
//  Plus the two ordinary claims: the batched prefill is token-identical
//  to sequential decode, and the output depends on the input.
//
//  Run: ./bin/test_qwen4_exp_e2e
// =====================================================================
#include "b70/grimoire_api.hpp"
#include "b70/formats.hpp"
#include "mini_model.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace b70;

// Counts SUCCESSFUL batched prefills.  Without it a prefill-vs-decode
// comparison passes trivially when the batched path declined and fell
// back to the very path it is being compared against.
namespace b70 { extern long g_qwen4_exp_batched_prefills; }

static int g_fail = 0;
static void CHECK(bool c, const char* why) {
    if (!c) { ++g_fail; std::printf("    FAIL: %s\n", why); }
}

static const std::vector<int32_t> kPrompt{7, 11, 3, 42, 5, 90, 1, 64, 23, 9};
static const std::vector<int32_t> kOther {99, 4, 77, 12, 60, 31, 8, 120, 2, 55};

// Generate with one checkpoint.  Returns false if it would not load.
static bool gen(const fs::path& dir, Fmt fmt, const std::vector<int32_t>& prompt,
                int want, std::vector<int32_t>& out, std::string& err) {
    Grimoire* e = grimoire_new();
    if (!e) { err = "grimoire_new failed"; return false; }
    if (!grimoire_load(*e, dir.string(), fmt, 128, err)) {
        grimoire_delete(e); return false;
    }
    FinishReason r{};
    bool ok = true;
    try {
        grimoire_serve_generate(*e, prompt, want, -1, out, -1, {}, &r);
    } catch (const std::exception& ex) { err = ex.what(); ok = false; }
    grimoire_delete(e);
    return ok;
}

static std::string show(const std::vector<int32_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size() && i < 8; ++i)
        s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== Qwen4-Exp: HyperConnections + QSA + PLE ===\n");
    std::printf("A/Bs are same-weights, config-only: the tensor list does not\n"
                "change, so a difference is the mechanism and not the RNG.\n\n");

    // The batched prefill is XMX from end to end and is unreachable off
    // the card without this: it routes every GEMM through the plain-SYCL
    // batched kernel.  On a B70 device_can_matrix() is true and the flag
    // does nothing, so the Tower path is untouched.  CORRECTNESS ONLY --
    // nothing timed under it means anything (rule 8).
    ::setenv("GRIMOIRE_BATCHED_PREFILL_NOXMX", "1", 1);

    char tmpl[] = "/tmp/grimoire-q4exp-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;

    const mini::Arch base = mini::qwen4_exp();

    // A config edit that keeps the tensor list identical.
    auto variant = [&](const char* name, const char* from, const char* to) {
        mini::Arch v = base;
        v.name = name;
        const size_t at = v.config.find(from);
        if (at == std::string::npos) {
            ++g_fail;
            std::printf("    FAIL: the fixture no longer contains \"%s\", so "
                        "the %s A/B proves nothing\n", from, name);
            return v;
        }
        v.config.replace(at, std::strlen(from), to);
        return v;
    };

    const int kWant = 12;
    std::vector<int32_t> tok_base, tok_other, tok_again;
    {
        const fs::path dir = root / "base";
        mini::write_model(dir, base);
        std::string err;
        if (!gen(dir, Fmt::BF16, kPrompt, kWant, tok_base, err)) {
            std::printf("    FAIL: the base fixture did not run: %s\n", err.c_str());
            std::printf("\nFAILURES (%d)\n", ++g_fail);
            return 1;
        }
        (void)gen(dir, Fmt::BF16, kPrompt, kWant, tok_again, err);
        (void)gen(dir, Fmt::BF16, kOther,  kWant, tok_other, err);
        std::printf("%-22s %s\n", "generated", show(tok_base).c_str());
        CHECK(int(tok_base.size()) == kWant, "did not produce the requested tokens");
        for (int32_t t : tok_base) CHECK(t >= 0 && t < base.vocab,
                                         "a token is outside the vocabulary");
        CHECK(tok_base == tok_again, "two identical requests disagree");
        // A model whose logits never change passes every check above.
        CHECK(tok_base != tok_other,
              "two different prompts gave the same continuation, so the "
              "output does not depend on the input");
    }

    // ---- QSA is not dense attention ----------------------------------
    // Both arms have the indexer tensors; only layer_types changes.  If
    // the engine ran qwen_sparse_attention layers as ordinary full
    // attention -- which it recognises and could -- these would agree.
    {
        mini::Arch v = variant("qsa-off", "\"qwen_sparse_attention\"",
                               "\"full_attention\"      ");
        const fs::path dir = root / "qsa-off";
        mini::write_model(dir, v);
        std::vector<int32_t> t; std::string err;
        if (!gen(dir, Fmt::BF16, kPrompt, kWant, t, err)) {
            ++g_fail;
            std::printf("    FAIL: the dense-attention arm did not run: %s\n",
                        err.c_str());
        } else {
            std::printf("%-22s %s\n", "with dense attention", show(t).c_str());
            CHECK(t != tok_base,
                  "QSA and dense attention give the same tokens, so the "
                  "sparse selection is not reaching attention");
        }
    }

    // ---- the PLE layer runs ------------------------------------------
    // ple_layer_ids -> [] removes the layer from the graph and touches
    // nothing else.  Same tensors, same weights.
    {
        mini::Arch v = variant("ple-off", "\"ple_layer_ids\": [3]",
                               "\"ple_layer_ids\": [] ");
        const fs::path dir = root / "ple-off";
        mini::write_model(dir, v);
        std::vector<int32_t> t; std::string err;
        if (!gen(dir, Fmt::BF16, kPrompt, kWant, t, err)) {
            ++g_fail;
            std::printf("    FAIL: the no-PLE arm did not run: %s\n", err.c_str());
        } else {
            std::printf("%-22s %s\n", "with no PLE layer", show(t).c_str());
            CHECK(t != tok_base,
                  "the model is identical with and without its PLE layer, so "
                  "the n-gram embedding contributes nothing");
        }
    }

    // ---- ple_layer_ids is ONE-BASED ----------------------------------
    // [3] is layer index 2.  Read as 0-based it would land on layer 3 --
    // and the fixture carries PLE weights on BOTH, so that arm loads and
    // runs perfectly well.  That is the point: an off-by-one here is not
    // an error, it is a different model that generates fine.
    {
        mini::Arch v = variant("ple-moved", "\"ple_layer_ids\": [3]",
                               "\"ple_layer_ids\": [4]");
        const fs::path dir = root / "ple-moved";
        mini::write_model(dir, v);
        std::vector<int32_t> t; std::string err;
        if (!gen(dir, Fmt::BF16, kPrompt, kWant, t, err)) {
            ++g_fail;
            std::printf("    FAIL: the moved-PLE arm did not run: %s\n", err.c_str());
        } else {
            std::printf("%-22s %s\n", "PLE moved one layer", show(t).c_str());
            CHECK(t != tok_base,
                  "moving the PLE layer changes nothing, so its placement is "
                  "not being read from ple_layer_ids");
        }
    }

    // ---- the FP8 n-gram table reads the same as the BF16 one ---------
    // This is the format the real checkpoint ships: ~51 B parameters in
    // that table, which is 51 GB of host memory at FP8 and 102 GB at
    // BF16.  A path only ever handed BF16 has not been tested on the
    // format that matters.
    //
    // Both arms round the table's values through E4M3, so the two files
    // hold numerically identical tables and differ only in ENCODING --
    // which turns "it loaded" into "it read the right numbers".  The
    // scale is 1.0 here on purpose: a missing scale is refused rather
    // than defaulted, and that refusal is checked in the matrix gate.
    {
        const fs::path dir = root / "fp8-table";
        mini::write_model(dir, mini::qwen4_exp(4, /*fp8_table=*/true));
        std::vector<int32_t> t; std::string err;
        if (!gen(dir, Fmt::BF16, kPrompt, kWant, t, err)) {
            ++g_fail;
            std::printf("    FAIL: the FP8-table arm did not run: %s\n",
                        err.c_str());
        } else {
            std::printf("%-22s %s\n", "FP8 n-gram table", show(t).c_str());
            CHECK(t == tok_base,
                  "the FP8 and BF16 n-gram tables hold the same numbers and "
                  "must generate the same tokens; they do not, so the FP8 "
                  "read path is wrong");
        }
    }

    // ---- the batched prefill actually ran ----------------------------
    {
        const long n = b70::g_qwen4_exp_batched_prefills;
        std::printf("%-22s %ld\n", "batched prefills", n);
        CHECK(n > 0,
              "no batched prefill ran, so the comparison below would be the "
              "sequential path against itself");
    }

    // ---- batched prefill is token-identical to sequential decode -----
    // The escape hatch the gemma-4 prefill established: one env var
    // forces the token-at-a-time path, so the two graphs can be compared
    // on the same build rather than on two builds.
    {
        const fs::path dir = root / "base";
        std::vector<int32_t> seq; std::string err;
        ::setenv("GRIMOIRE_QWEN4EXP_SEQUENTIAL_PREFILL", "1", 1);
        const bool ok = gen(dir, Fmt::BF16, kPrompt, kWant, seq, err);
        ::unsetenv("GRIMOIRE_QWEN4EXP_SEQUENTIAL_PREFILL");
        if (!ok) {
            ++g_fail;
            std::printf("    FAIL: the sequential arm did not run: %s\n",
                        err.c_str());
        } else {
            std::printf("%-22s %s\n", "sequential prefill", show(seq).c_str());
            CHECK(seq == tok_base,
                  "batched prefill and sequential decode disagree");
        }
    }

    if (!g_fail) fs::remove_all(root);
    else std::printf("\nfixtures kept in %s\n", root.c_str());
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
