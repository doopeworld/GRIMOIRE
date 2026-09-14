// =====================================================================
//  test_model_matrix.cpp -- every architecture x every projection format.
//
//  "GRIMOIRE should take any model in bf16 and quantize it to whatever
//  suits the card" is a claim with seven formats and several
//  architectures behind it, and until this existed exactly one
//  combination had ever been run end to end.  A format that faults, or
//  a shape a quantizer refuses, is a five-minute discovery here and a
//  power-cycled B70 otherwise.
//
//  For each cell: write a random-weight checkpoint, load it at that
//  projection format, generate, and require that every token is in
//  vocabulary and that a second identical request gives identical
//  tokens.  The weights are noise, so the TEXT means nothing -- what is
//  being checked is that the path runs, does not corrupt memory, and
//  carries no state between requests.  Rule 8 is untouched: nothing
//  here is a speed claim.
//
//  Each cell runs in its OWN process, so one bad combination reports as
//  a failed cell instead of taking the whole run down -- which is also
//  how you find out that it faults rather than errors.
//
//  Run:
//    ./bin/test_model_matrix               # drive every cell
//    ./bin/test_model_matrix <dir> <fmt>   # one cell (used internally)
// =====================================================================
#include "b70/grimoire_api.hpp"
#include "b70/formats.hpp"
#include "mini_model.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace b70;

static const std::vector<int32_t> kPrompt{7, 11, 3, 42, 5, 90, 1, 64};
static const int kWant = 8;

struct FmtName { const char* name; Fmt fmt; };
static const FmtName kFormats[] = {
    {"bf16",     Fmt::BF16},
    {"fp8_e4m3", Fmt::FP8_E4M3},
    {"fp8_e5m2", Fmt::FP8_E5M2},
    {"int8",     Fmt::INT8},
    {"int4",     Fmt::INT4},
    {"mxfp8",    Fmt::MXFP8},
    {"mxfp4",    Fmt::MXFP4},
};

// ---- one cell, in its own process ------------------------------------
// Exit 0 only when the model loaded, generated kWant in-vocabulary
// tokens, and did it identically twice.
static int run_cell(const std::string& dir, const std::string& fname, int vocab) {
    Fmt fmt = Fmt::BF16;
    bool known = false;
    for (const auto& f : kFormats)
        if (fname == f.name) { fmt = f.fmt; known = true; }
    if (!known) return 9;

    std::string err;
    Grimoire* e = grimoire_new();
    if (!e) return 2;
    if (!grimoire_load(*e, dir, fmt, 128, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        grimoire_delete(e); return 3;
    }
    std::vector<int32_t> a, b, c;
    static const std::vector<int32_t> kOther{99, 4, 77, 12, 60, 31, 8, 120};
    FinishReason r{};
    try {
        grimoire_serve_generate(*e, kPrompt, kWant, -1, a, -1, {}, &r);
        grimoire_serve_generate(*e, kPrompt, kWant, -1, b, -1, {}, &r);
        grimoire_serve_generate(*e, kOther,  kWant, -1, c, -1, {}, &r);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "generate: %s\n", ex.what());
        grimoire_delete(e); return 4;
    }
    grimoire_delete(e);
    if (int(a.size()) != kWant) { std::fprintf(stderr, "got %zu tokens\n", a.size()); return 5; }
    for (int32_t t : a) if (t < 0 || t >= vocab) {
        std::fprintf(stderr, "token %d outside vocabulary\n", t); return 6;
    }
    if (a != b) { std::fprintf(stderr, "not reproducible across requests\n"); return 7; }
    // Does the output depend on the INPUT at all?
    //
    // Every check above passes on a model whose logits never change: the
    // same token is in vocabulary, is the right length, and is perfectly
    // reproducible.  That is exactly how a tied-embedding checkpoint
    // behaved for a whole session -- lm_head was never uploaded, gemv
    // wrote nothing, and argmax read stale device memory.  The table said
    // "ok".
    //
    // Two different prompts must not give the same continuation.  With
    // random weights a collision is vanishingly unlikely over 8 tokens,
    // and if it ever does happen it is worth looking at.
    if (a == c) {
        std::fprintf(stderr,
            "output does not depend on the input: two different prompts "
            "gave the same %d tokens\n", kWant);
        return 8;
    }
    return 0;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (argc >= 3) return run_cell(argv[1], argv[2], argc >= 4 ? std::atoi(argv[3]) : 128);

    std::printf("=== architecture x projection format ===\n");
    std::printf("random-weight checkpoints; output is noise by design.\n");
    std::printf("what is checked: loads, generates, in vocabulary, reproducible.\n\n");

    char tmpl[] = "/tmp/grimoire-matrix-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;
    const char* self = argv[0];

    std::vector<mini::Arch> archs = { mini::dense(), mini::moe(),
                                      mini::hybrid(), mini::k2(),
                                      mini::gemma4() };

    std::printf("%-12s", "");
    for (const auto& f : kFormats) std::printf("%-10s", f.name);
    std::printf("\n");

    int fails = 0;
    std::vector<std::string> notes;
    for (auto& a : archs) {
        const fs::path dir = root / a.name;
        mini::write_model(dir, a);
        std::printf("%-12s", a.name);
        for (const auto& f : kFormats) {
            // Both streams go to a per-cell log: the device banner and
            // the bridge-fallback notices are several lines per cell and
            // would bury the table.  The log is printed only on failure.
            const std::string cell_log =
                (dir / (std::string("cell-") + f.name + ".log")).string();
            const pid_t pid = ::fork();
            if (pid == 0) {
                std::freopen(cell_log.c_str(), "w", stdout);
                std::freopen(cell_log.c_str(), "a", stderr);
                const std::string v = std::to_string(a.vocab);
                const char* av[] = { self, dir.c_str(), f.name, v.c_str(), nullptr };
                ::execv(self, const_cast<char* const*>(av));
                ::_exit(127);
            }
            int st = 0;
            ::waitpid(pid, &st, 0);
            const bool exited = WIFEXITED(st);
            const int  code   = exited ? WEXITSTATUS(st) : -1;
            if (exited && code == 0) { std::printf("%-10s", "ok"); continue; }
            ++fails;
            // A SIGNAL is categorically worse than a non-zero exit: it
            // means the path corrupted memory or faulted rather than
            // reporting that it could not do the job.
            const char* tag = !exited ? "CRASH"
                            : code == 3 ? "load"
                            : code == 4 ? "threw"
                            : code == 7 ? "nondet"
                            : code == 8 ? "no-input"
                            : code == 6 ? "vocab"
                                        : "fail";
            std::printf("%-10s", tag);
            std::string why = exited ? ("exit " + std::to_string(code))
                                     : ("SIGNAL " + std::to_string(WTERMSIG(st)));
            // Carry the last line the cell printed: for a load or shape
            // refusal that line IS the diagnosis.
            if (std::FILE* lf = std::fopen(cell_log.c_str(), "r")) {
                char line[512], last[512] = {0};
                while (std::fgets(line, sizeof(line), lf))
                    if (line[0] != '\n') std::strncpy(last, line, sizeof(last)-1);
                std::fclose(lf);
                if (last[0]) {
                    std::string l(last);
                    while (!l.empty() && (l.back()=='\n' || l.back()==' ')) l.pop_back();
                    if (!l.empty()) why += "  -- " + l;
                }
            }
            notes.push_back(std::string(a.name) + " / " + f.name + ": " + why);
        }
        std::printf("\n");
    }

    // ---- architectures the engine must REFUSE, by name ---------------
    // A model this engine cannot execute has exactly two acceptable
    // behaviours: run it correctly, or say so.  The third -- load it on a
    // path with the wrong residual graph and emit fluent text -- cannot be
    // caught by the table above, because that cell would read "ok".
    //
    // gemma-4 itself now RUNS (the row above), but only below head_dim
    // 256: every flash kernel accumulates a head into a private array of
    // MAX_DPL floats per lane, and the real 31B checkpoint's
    // full-attention layers are 512 wide.  Going past that is a
    // DEVICE_LOST, so it must refuse rather than reach the card.  The
    // fixture here is the same model with ONE value changed, which is what
    // makes this a test of the bound and not of gemma-4.
    {
        std::printf("\nmust be refused, not silently run:\n");
        mini::Arch wide = mini::gemma4();
        const std::string from = "\"global_head_dim\": 32";
        const std::string to   = "\"global_head_dim\": 512";
        const size_t at = wide.config.find(from);
        if (at == std::string::npos) {
            ++fails;
            std::printf("  %-14s fixture no longer declares global_head_dim; "
                        "this case proves nothing\n", "head_dim 512");
        } else {
            wide.config.replace(at, from.size(), to);
            const fs::path dir = root / "refuse-wide-head";
            mini::write_model(dir, wide);
            std::string err;
            Grimoire* e = grimoire_new();
            bool loaded = false;
            if (e) {
                loaded = grimoire_load(*e, dir.string(), Fmt::BF16, 128, err);
                grimoire_delete(e);
            }
            if (loaded) {
                ++fails;
                std::printf("  %-14s LOADED -- a 512-wide head overruns a "
                            "device stack array, which is a DEVICE_LOST\n",
                            "head_dim 512");
            } else if (err.find("exceeds what the flash-attention kernels")
                       == std::string::npos) {
                // The fixture is deliberately inconsistent in other ways
                // once the width changes, so require the bound's OWN
                // sentence: any other refusal would pass for the wrong
                // reason and leave the bound untested.
                ++fails;
                std::printf("  %-14s refused, but for the wrong reason: %s\n",
                            "head_dim 512", err.c_str());
            } else {
                std::printf("  %-14s refused by name\n", "head_dim 512");
            }
        }
    }

    if (!notes.empty()) {
        std::printf("\ndetail:\n");
        for (const auto& n : notes) std::printf("  %s\n", n.c_str());
        std::printf("\nre-run one cell with its output visible:\n");
        std::printf("  %s <model dir> <format>\n", self);
    }
    if (!fails) fs::remove_all(root);
    else std::printf("\nlogs kept in %s\n", root.c_str());
    std::printf("\n%s (%d failed cells)\n", fails ? "FAILURES" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
