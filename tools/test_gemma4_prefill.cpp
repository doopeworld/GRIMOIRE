// =====================================================================
//  test_gemma4_prefill.cpp -- gemma-4 batched prefill must equal
//  sequential decode, token for token.
//
//  gemma-4's residual graph is a SANDWICH (post_attention_layernorm on
//  the attention output before it joins the stream, post_feedforward_
//  layernorm on the FFN output, then a per-layer scalar).  The generic
//  batched prefill is the Qwen graph.  Running a gemma-4 prompt through
//  the wrong one produces FLUENT TEXT that is not the model's output,
//  which no smoke test catches -- so the only useful check is exactness
//  against the path that is already known good.
//
//  Three things this asserts, in order of how quietly they fail:
//
//   1. The batched path ACTUALLY RAN.  g_gemma4_batched_prefills counts
//      successful batched prefills; if the path declines and falls back,
//      both sides of the comparison are the sequential path and the test
//      passes while proving nothing.  That is rule 12's failure shape.
//   2. Batched output == sequential output, for the same prompt.
//   3. It holds at head_dim 512 as well as the small fixture, because
//      the wide head takes a different flash instantiation.
//
//  The batched path is joint_matrix from end to end, so off the card it
//  is reachable only through GRIMOIRE_BATCHED_PREFILL_NOXMX, which routes
//  every projection through the plain-SYCL batched GEMM.  Rule 8: this
//  file makes no claim about speed, here or on a B70.
//
//  Run:  ./bin/test_gemma4_prefill
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

// Defined in src/grimoire.cpp, inside namespace b70.
namespace b70 { extern long g_gemma4_batched_prefills; }
using b70::g_gemma4_batched_prefills;

static const std::vector<int32_t> kPrompt{7, 11, 3, 42, 5, 90, 1, 64, 33, 2};
static const int kWant = 8;

// One generation, with the prefill path selected by environment.
// Returns false if the model would not load or generate at all.
static bool gen(const std::string& dir, Fmt fmt, bool batched,
                std::vector<int32_t>& out, long* ran) {
    if (batched) {
        // NOXMX makes the batched path reachable on a device with no
        // matrix hardware; the second var selects it at all (it is
        // opt-in until a B70 has run this gate).
        ::setenv("GRIMOIRE_BATCHED_PREFILL_NOXMX", "1", 1);
        ::setenv("GRIMOIRE_GEMMA4_BATCHED_PREFILL", "1", 1);
    } else {
        ::unsetenv("GRIMOIRE_GEMMA4_BATCHED_PREFILL");
    }
    const long before = g_gemma4_batched_prefills;
    std::string err;
    Grimoire* e = grimoire_new();
    if (!e) return false;
    if (!grimoire_load(*e, dir, fmt, 128, err)) {
        std::fprintf(stderr, "  load: %s\n", err.c_str());
        grimoire_delete(e);
        return false;
    }
    FinishReason r{};
    try {
        grimoire_serve_generate(*e, kPrompt, kWant, -1, out, -1, {}, &r);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "  generate: %s\n", ex.what());
        grimoire_delete(e);
        return false;
    }
    grimoire_delete(e);
    *ran = g_gemma4_batched_prefills - before;
    return true;
}

static int one(const char* label, mini::Arch a, const fs::path& root,
               Fmt fmt, const char* fname) {
    const fs::path dir = root / (std::string(a.name) + "-" + fname);
    mini::write_model(dir, a);

    std::vector<int32_t> seq, bat;
    long ran_seq = 0, ran_bat = 0;
    if (!gen(dir.string(), fmt, false, seq, &ran_seq)) {
        std::printf("  %-18s %-8s SEQUENTIAL RUN FAILED\n", label, fname);
        return 1;
    }
    if (!gen(dir.string(), fmt, true, bat, &ran_bat)) {
        std::printf("  %-18s %-8s BATCHED RUN FAILED\n", label, fname);
        return 1;
    }
    if (ran_seq != 0) {
        std::printf("  %-18s %-8s the SEQUENTIAL side used the batched "
                    "path (%ld times) -- the comparison is void\n",
                    label, fname, ran_seq);
        return 1;
    }
    if (ran_bat == 0) {
        std::printf("  %-18s %-8s batched prefill NEVER RAN -- it declined "
                    "and fell back, so this cell proves nothing\n",
                    label, fname);
        return 1;
    }
    if (seq != bat) {
        std::printf("  %-18s %-8s MISMATCH after %ld batched prefill(s)\n",
                    label, fname, ran_bat);
        std::printf("      sequential:");
        for (int32_t t : seq) std::printf(" %d", t);
        std::printf("\n      batched   :");
        for (int32_t t : bat) std::printf(" %d", t);
        std::printf("\n");
        return 1;
    }
    std::printf("  %-18s %-8s identical (%d tokens, %ld batched prefill)\n",
                label, fname, kWant, ran_bat);
    return 0;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== gemma-4 batched prefill must equal sequential decode ===\n");
    std::printf("the sandwich graph, k_eq_v, v_norm, GeGLU and the per-layer\n");
    std::printf("sliding window -- each is fluent when wrong.\n\n");

    char tmpl[] = "/tmp/grimoire-g4pf-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;

    struct Cell { const char* name; Fmt fmt; };
    const Cell cells[] = {
        {"bf16", Fmt::BF16},
        {"fp8_e4m3", Fmt::FP8_E4M3},
        {"int8", Fmt::INT8},
        {"mxfp4", Fmt::MXFP4},
    };

    int fails = 0;
    for (const auto& c : cells)
        fails += one("gemma4", mini::gemma4(), root, c.fmt, c.name);
    // The 512-wide head takes the 32-slot flash instantiation, which is a
    // different kernel from the one every cell above exercised.
    for (const auto& c : cells)
        fails += one("gemma4-hd512", mini::gemma4_wide(), root, c.fmt, c.name);

    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASS",
                fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
