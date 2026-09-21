// =====================================================================
//  test_nvfp4_e2e.cpp -- reading an NVFP4 checkpoint.
//
//  NVFP4 is NVIDIA's Blackwell 4-bit format and it wears MXFP4's
//  clothes: the same .weight_packed name, the same [N][K/2] E2M1
//  payload.  What differs is the scale -- E4M3 per 16 against E8M0 per
//  32 -- plus one FP32 scale per tensor that MXFP4 has no counterpart
//  for.  Read one as the other and you get finite numbers, a model that
//  loads, and fluent output.  Nothing in a "does it generate" check can
//  see it.
//
//  So the gate is an EQUALITY, not a smoke test.  mini::to_nvfp4()
//  builds two checkpoints from the same numbers: one storing them as
//  NVFP4, one as plain bf16.  The values are chosen on the E2M1 grid
//  with E4M3 scales and a power-of-two global scale, so both files hold
//  the IDENTICAL number, not an approximation of it -- and the two must
//  therefore generate identical tokens.
//
//  Derivation of the format, and the merged-linear trap that makes
//  per-partition global scales load-bearing: include/b70/nvfp4.hpp.
//
//  Run: ./bin/test_nvfp4_e2e
// =====================================================================
#include "b70/grimoire_api.hpp"
#include "b70/formats.hpp"
#include "b70/nvfp4.hpp"
#include "mini_model.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace b70;

namespace b70 { extern long g_nvfp4_tensors; }

static int g_fail = 0;
static void CHECK(bool c, const char* why) {
    if (!c) { ++g_fail; std::printf("    FAIL: %s\n", why); }
}

static const std::vector<int32_t> kPrompt{7, 11, 3, 42, 5, 90, 1, 64};
static const std::vector<int32_t> kOther {99, 4, 77, 12, 60, 31, 8, 120};
static const int kWant = 10;

static bool gen(const fs::path& dir, Fmt fmt, const std::vector<int32_t>& prompt,
                std::vector<int32_t>& out, std::string& err) {
    Grimoire* e = grimoire_new();
    if (!e) { err = "grimoire_new failed"; return false; }
    if (!grimoire_load(*e, dir.string(), fmt, 128, err)) {
        grimoire_delete(e); return false;
    }
    FinishReason r{};
    bool ok = true;
    try {
        grimoire_serve_generate(*e, prompt, kWant, -1, out, -1, {}, &r);
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
    std::printf("=== NVFP4 checkpoints ===\n");
    std::printf("two files, the same numbers, one stored as NVFP4 and one as\n"
                "bf16: they must generate identically.\n\n");

    char tmpl[] = "/tmp/grimoire-nvfp4-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;

    const mini::NvfpPair pair = mini::to_nvfp4(mini::dense());
    const fs::path dn = root / "nvfp4", db = root / "twin";
    mini::write_model(dn, pair.nvfp4);
    mini::write_model(db, pair.bf16);

    // The host reference, on one block, before anything else runs.  If
    // this disagrees the engine's answer cannot be judged against it.
    //
    // DIVIDES by the global scale (fixed 2026-09-21, external audit).
    // compressed-tensors' own NVFP4 dequantizer computes
    // local_scale / global_scale and passes the stored scale through
    // unchanged -- verified directly against that source, not inferred.
    // The old "* 0.5f" here was not an independent check: it re-derived
    // the same wrong direction the engine used, so the two could only
    // ever agree with each other, never with a real checkpoint.
    {
        // 0x21 = codes 1 (0.5) then 2 (1.0), low nibble first.
        const uint8_t packed[8] = {0x21, 0x43, 0x65, 0x87, 0x21, 0x43, 0x65, 0x87};
        const uint8_t scale[1]  = { f32_to_e4m3(1.5f) };
        float got[16] = {0};
        nvfp4_dequant_row(packed, scale, 0.5f, 16, got);
        const float want0 = e2m1_to_f32(1) * e4m3_to_f32(scale[0]) / 0.5f;
        const float want1 = e2m1_to_f32(2) * e4m3_to_f32(scale[0]) / 0.5f;
        std::printf("%-22s low nibble first: %.4f %.4f\n", "host reference",
                    got[0], got[1]);
        CHECK(got[0] == want0 && got[1] == want1,
              "nvfp4_dequant_row does not put the LOW nibble first; every "
              "pair of weights inside a block is transposed");
    }

    std::vector<int32_t> tn, tb, tother;
    {
        std::string err;
        const long before = b70::g_nvfp4_tensors;
        if (!gen(dn, Fmt::BF16, kPrompt, tn, err)) {
            std::printf("    FAIL: the NVFP4 checkpoint did not run: %s\n",
                        err.c_str());
            std::printf("\nFAILURES (%d)\n", ++g_fail);
            return 1;
        }
        const long decoded = b70::g_nvfp4_tensors - before;
        std::printf("%-22s %ld\n", "NVFP4 tensors decoded", decoded);
        // dense() has 7 projections per layer x 4 layers.
        CHECK(decoded >= 28,
              "too few NVFP4 tensors were decoded -- the checkpoint was read "
              "through some other path and the comparison below is vacuous");
        if (!gen(db, Fmt::BF16, kPrompt, tb, err)) {
            ++g_fail;
            std::printf("    FAIL: the bf16 twin did not run: %s\n", err.c_str());
        }
        (void)gen(dn, Fmt::BF16, kOther, tother, err);
    }

    std::printf("%-22s %s\n", "NVFP4", show(tn).c_str());
    std::printf("%-22s %s\n", "bf16 twin", show(tb).c_str());
    CHECK(tn == tb,
          "the NVFP4 checkpoint and its bf16 twin hold the same numbers and "
          "must generate the same tokens; they do not, so the reader is "
          "decoding the wrong values");
    CHECK(!tn.empty() && tn != tother,
          "two different prompts gave the same continuation, so this "
          "comparison would pass on a model whose logits never change");

    // Requantizing OUT of NVFP4 to each packed format this engine has.
    // That is the claim a dedicated NVFP4->MXFP4 path cannot make: the
    // checkpoint arrives in one format and leaves in whichever --proj
    // asks for, because it goes through f32 on the way.  Lossy, so the
    // check is that the path RUNS and stays in vocabulary, not equality.
    for (auto f : {Fmt::MXFP4, Fmt::INT4, Fmt::FP8_E4M3}) {
        const char* nm = f == Fmt::MXFP4 ? "mxfp4"
                       : f == Fmt::INT4  ? "int4" : "fp8_e4m3";
        std::vector<int32_t> t; std::string err;
        if (!gen(dn, f, kPrompt, t, err)) {
            ++g_fail;
            std::printf("    FAIL: NVFP4 -> %s did not run: %s\n", nm, err.c_str());
            continue;
        }
        bool in_vocab = int(t.size()) == kWant;
        for (int32_t v : t) in_vocab = in_vocab && v >= 0 && v < pair.nvfp4.vocab;
        std::printf("%-22s %s\n", (std::string("NVFP4 -> ") + nm).c_str(),
                    show(t).c_str());
        CHECK(in_vocab, "requantizing out of NVFP4 produced a bad token");
    }

    if (!g_fail) fs::remove_all(root);
    else std::printf("\nfixtures kept in %s\n", root.c_str());
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
