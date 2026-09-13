// =====================================================================
//  test_parallel_e2e.cpp -- pipeline and tensor parallel must produce
//  EXACTLY the tokens one process produces.
//
//  Greedy decode is deterministic, so a correct split is a bit-identical
//  split.  Anything else -- a layer owned by nobody, a layer owned twice,
//  a hidden state staged with the wrong element count, a rank that runs
//  the LM head when it should not, an expert shard that drops experts --
//  shows up here as different tokens, and nowhere else until someone
//  reads a garbled answer off the card.
//
//  Ranks are separate PROCESSES talking over a UNIX socket, which is how
//  tools/pp2run.sh and tools/tp2run.sh run them on the Tower.  This tool
//  re-execs itself per rank for exactly that reason: fork() alone would
//  share a SYCL runtime that was never meant to be inherited.
//
//  On a host with one device both ranks land on the same device.  That
//  does not test the LINK, and it is not meant to -- it tests the
//  protocol, the layer ownership and the numerics, which is the part
//  that has been wrong before and the part you cannot debug with a
//  power-cycled card.
//
//  Build:
//    icpx -fsycl -fsycl-targets=spir64 -O2 -std=c++20 -I include -I src \
//      tools/test_parallel_e2e.cpp src/grimoire.cpp ... -o bin/test_parallel_e2e
//
//  Run:
//    ./bin/test_parallel_e2e                  # drives every case
//    ./bin/test_parallel_e2e <dir> <role>     # one rank (used internally)
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

static int g_fail = 0;
#define CHECK(c, ...) do { if(!(c)){ std::printf("  FAIL "); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while(0)

static const std::vector<int32_t> kPrompt{7, 11, 3, 42, 5, 90, 1, 64};
static const int kWant = 10;

// ---------------------------------------------------------------------
// One rank: load, generate, write the tokens to `out` (last stage only).
// ---------------------------------------------------------------------
static int run_rank(const std::string& dir, const std::string& out, Fmt fmt) {
    std::string err;
    Grimoire* e = grimoire_new();
    if (!e) { std::fprintf(stderr, "grimoire_new failed\n"); return 2; }
    if (!grimoire_load(*e, dir, fmt, 128, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        grimoire_delete(e); return 3;
    }
    std::vector<int32_t> toks;
    FinishReason r{};
    try {
        grimoire_serve_generate(*e, kPrompt, kWant, -1, toks, -1, {}, &r);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "generate threw: %s\n", ex.what());
        grimoire_delete(e); return 4;
    }
    grimoire_delete(e);
    if (!out.empty()) {
        std::FILE* f = std::fopen(out.c_str(), "w");
        if (!f) return 5;
        for (int32_t t : toks) std::fprintf(f, "%d\n", t);
        std::fclose(f);
    }
    return 0;
}

static std::vector<int32_t> read_tokens(const std::string& path) {
    std::vector<int32_t> v;
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return v;
    int t;
    while (std::fscanf(f, "%d", &t) == 1) v.push_back(t);
    std::fclose(f);
    return v;
}

// Spawn one rank as a fresh process with the given extra environment.
static pid_t spawn(const char* self, const std::string& dir, const std::string& out,
                   const char* fmt, const std::vector<std::string>& env) {
    const pid_t pid = ::fork();
    if (pid != 0) return pid;
    for (const auto& kv : env) {
        const size_t eq = kv.find('=');
        ::setenv(kv.substr(0, eq).c_str(), kv.substr(eq+1).c_str(), 1);
    }
    // Children are noisy (device banner, capability matrix, per-rank
    // progress).  Keep stdout for the driver; a failing rank still gets
    // its stderr through.
    std::freopen("/dev/null", "w", stdout);
    const char* argv[] = { self, dir.c_str(), out.c_str(), fmt, nullptr };
    ::execv(self, const_cast<char* const*>(argv));
    std::fprintf(stderr, "execv failed: %s\n", std::strerror(errno));
    ::_exit(127);
}

static bool wait_ok(const std::vector<pid_t>& pids, const char* what) {
    bool ok = true;
    for (pid_t p : pids) {
        int st = 0;
        if (::waitpid(p, &st, 0) < 0) { ok = false; continue; }
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
            std::printf("  %s: rank pid %d exited %d (signal %d)\n", what, int(p),
                        WIFEXITED(st) ? WEXITSTATUS(st) : -1,
                        WIFSIGNALED(st) ? WTERMSIG(st) : 0);
            ok = false;
        }
    }
    return ok;
}

static std::string join(const std::vector<int32_t>& v) {
    std::string s;
    for (int32_t t : v) { s += ' '; s += std::to_string(t); }
    return s;
}

// ---------------------------------------------------------------------
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // ---- rank mode -------------------------------------------------
    if (argc >= 4) {
        const std::string fs_(argv[3]);
        const Fmt fmt = fs_ == "fp8"   ? Fmt::FP8_E4M3
                      : fs_ == "int4"  ? Fmt::INT4
                      : fs_ == "mxfp4" ? Fmt::MXFP4
                                       : Fmt::BF16;
        return run_rank(argv[1], std::string(argv[2]) == "-" ? "" : argv[2], fmt);
    }

    std::printf("=== pipeline / tensor parallel vs a single process ===\n\n");

    char tmpl[] = "/tmp/grimoire-par-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;
    const char* self = argv[0];

    struct Case { mini::Arch arch; const char* fmt; };
    std::vector<Case> cases = {
        { mini::dense(4), "bf16" },
        { mini::dense(4), "fp8"  },   // the dual-GPU format
        { mini::moe(4),   "bf16" },
        { mini::moe(4),   "fp8"  },
        { mini::k2(),     "bf16" },
        // The hybrid carries DeltaNet layers: a recurrent state and a conv
        // ring that a pipeline split has to keep consistent across ranks.
        { mini::hybrid(4), "bf16" },
        { mini::hybrid(4), "fp8"  },
    };

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        auto& c = cases[ci];
        const fs::path dir = root / (std::string(c.arch.name) + "-" + c.fmt);
        mini::write_model(dir, c.arch);
        std::printf("%-10s %-5s  ", c.arch.name, c.fmt);
        std::fflush(stdout);

        // ---- reference: one process ---------------------------------
        const std::string ref_out = (dir/"ref.txt").string();
        std::vector<std::string> base_env;
        if (!wait_ok({spawn(self, dir.string(), ref_out, c.fmt, base_env)}, "single")) {
            ++g_fail; std::printf("single-process run FAILED\n"); continue;
        }
        const auto ref = read_tokens(ref_out);
        if (int(ref.size()) != kWant) {
            ++g_fail;
            std::printf("single-process produced %zu tokens, wanted %d\n", ref.size(), kWant);
            continue;
        }
        std::printf("single%s", join(ref).c_str());

        // ---- pipeline parallel, N ranks -----------------------------
        // The chain, the collectives and the layer split are all written
        // for N stages, but only TWO had ever been run -- and a 2-rank
        // chain never exercises a MIDDLE stage, the one that both
        // receives and forwards.  GRIMOIRE must work on any number of
        // Battlemage cards, so drive 2, 3 and 4.
        //
        // The mini models have 4 layers, so the splits are 2+2, 2+1+1 and
        // 1+1+1+1 -- the last giving every stage exactly one layer, which
        // is the tightest the boundary code ever gets.
        auto run_pp = [&](int world, const char* layers) {
            const std::string sock = (dir/("pp"+std::to_string(world)+".sock")).string();
            const std::string out  = (dir/("pp"+std::to_string(world)+".txt")).string();
            std::vector<pid_t> pids;
            // Later ranks listen, so start from the back: rank 0 retries
            // the connect for 10 minutes anyway, but starting listeners
            // first keeps the log readable when something goes wrong.
            for (int r = world - 1; r >= 0; --r) {
                std::vector<std::string> env{
                    "GRIMOIRE_PP_RANK=" + std::to_string(r),
                    "GRIMOIRE_PP_WORLD_SIZE=" + std::to_string(world),
                    "GRIMOIRE_PP_LAYERS=" + std::string(layers),
                    "GRIMOIRE_PP_SOCKET=" + sock,
                    "GRIMOIRE_DEVICE_ANY=1"};
                // Only the LAST stage has the head, so only it has tokens.
                pids.push_back(spawn(self, dir.string(),
                                     r == world - 1 ? out : "-", c.fmt, env));
            }
            const std::string what = "pp" + std::to_string(world);
            if (!wait_ok(pids, what.c_str())) {
                ++g_fail; std::printf("   PP%d FAILED", world); return;
            }
            const auto got = read_tokens(out);
            const bool match = (got == ref);
            std::printf("   PP%d %s", world, match ? "match" : "DIFFERS");
            if (!match) std::printf("%s", join(got).c_str());
            CHECK(match, "%s/%s: pipeline parallel over %d ranks changed the output",
                  c.arch.name, c.fmt, world);
        };

        // Every rank runs the head under TP, so rank 0's tokens are the
        // answer.  Uneven shards are the point at 3: 4 query heads over 3
        // ranks is 2/1/1, which no 2-rank run ever produces.
        auto run_tp = [&](int world) {
            const std::string sock = (dir/("tp"+std::to_string(world)+".sock")).string();
            const std::string out  = (dir/("tp"+std::to_string(world)+".txt")).string();
            std::vector<pid_t> pids;
            for (int r = 0; r < world; ++r) {
                std::vector<std::string> env{
                    "GRIMOIRE_TP_RANK=" + std::to_string(r),
                    "GRIMOIRE_TP_WORLD_SIZE=" + std::to_string(world),
                    "GRIMOIRE_TP_SOCKET=" + sock,
                    "GRIMOIRE_DEVICE_ANY=1"};
                pids.push_back(spawn(self, dir.string(),
                                     r == 0 ? out : "-", c.fmt, env));
            }
            const std::string what = "tp" + std::to_string(world);
            if (!wait_ok(pids, what.c_str())) {
                ++g_fail; std::printf("   TP%d FAILED", world); return;
            }
            const auto got = read_tokens(out);
            const bool match = (got == ref);
            std::printf("   TP%d %s", world, match ? "match" : "DIFFERS");
            if (!match) std::printf("%s", join(got).c_str());
            CHECK(match, "%s/%s: tensor parallel over %d ranks changed the output",
                  c.arch.name, c.fmt, world);
        };

        run_pp(2, "2,2");
        run_tp(2);
        // 3 and 4 ranks on every architecture would triple a run that
        // already takes twenty minutes on a CPU device, and the rank
        // COUNT is what is under test here, not the architecture -- the
        // architectures are covered at 2.  Drive the widest counts on the
        // two shapes that stress the boundary most: a plain dense model,
        // and the hybrid, whose DeltaNet layers carry a recurrent state
        // and a conv ring across every split.
        const bool wide = c.arch.name == std::string("dense") ||
                          c.arch.name == std::string("hybrid");
        if (wide) {
            run_pp(3, "2,1,1");
            run_pp(4, "1,1,1,1");
            run_tp(3);
            run_tp(4);
        }
        std::printf("\n");
    }

    fs::remove_all(root);
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
