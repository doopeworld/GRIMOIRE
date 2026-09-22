// =====================================================================
//  test_pp_server.cpp -- a RESIDENT pipeline answers many requests, and
//  the FRONT END has the tokens.
//
//  Pipeline parallel has worked here for a while, but only as a CLI: one
//  prompt, one answer, exit. That hid two things a server needs and a
//  one-shot run can never show.
//
//   * THE PROMPT ONLY EVER REACHED RANK 0.  Every stage runs the same
//     generation loop and they stay in step by exchanging a message per
//     token -- but the CLI gets its prompt from its own command line on
//     every rank, so nobody noticed the request itself was never sent.
//     A server's prompt arrives on rank 0's socket alone.  That, and
//     nothing deeper, is why there was no two-card server.
//
//   * A SECOND REQUEST.  A byte stream cannot be resynchronised: any
//     message whose length the two ends disagree about is invisible on
//     the request that causes it and fatal to the NEXT one.  A CLI exits
//     before that can show.  So this gate sends THREE requests, of
//     DIFFERENT LENGTHS, down one resident pipeline -- and it is the
//     third that would catch a header written but not read.
//
//   * THE ANSWER MUST COME BACK.  Only the last stage owns the head, so
//     only it computes a token; it travels backward for every stage to
//     roll back to the same position.  The existing gate reads its
//     tokens from the LAST rank, which proves the pipeline agrees but
//     not that the front end can answer anybody.  This one reads RANK 0,
//     because that is the rank holding the HTTP socket.
//
//  Run: GRIMOIRE_DEVICE_ANY=1 ./bin/test_pp_server
// =====================================================================
#include "b70/grimoire_api.hpp"
#include "b70/formats.hpp"
#include "mini_model.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <barrier>
#include <exception>
#include <sycl/sycl.hpp>
namespace b70 { extern long g_batch_decode_steps, g_batch_decode_rows; }

namespace fs = std::filesystem;
using namespace b70;

static int g_fail = 0;
static void CHECK(bool c, const char* why) {
    if (!c) { ++g_fail; std::printf("    FAIL: %s\n", why); }
}

// Three requests, deliberately different lengths: a fixed-width header
// read as a variable-length one lines up by accident when every request
// is the same size.
static std::vector<std::vector<int32_t>> requests() {
    std::vector<std::vector<int32_t>> v;
    const int len[3] = {9, 21, 14};
    for (int k = 0; k < 3; ++k) {
        std::vector<int32_t> p;
        uint32_t x = uint32_t(1 + k) * 2654435761u;
        for (int i = 0; i < len[k]; ++i) {
            x = x * 1664525u + 1013904223u;
            p.push_back(int32_t((x >> 9) % 120));
        }
        v.push_back(p);
    }
    return v;
}
static const int kWant = 12;

// One process of the pipeline.  Rank 0 is the front end: it forwards
// each request and then serves it, exactly as the scheduler does inside
// grimoire-server.  Every other rank follows.
static int run_rank(const std::string& dir, const std::string& out, Fmt fmt) {
    std::string err;
    Grimoire* e = grimoire_new();
    if (!e) { std::fprintf(stderr, "grimoire_new failed\n"); return 2; }
    if (!grimoire_load(*e, dir, fmt, 128, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        grimoire_delete(e); return 3;
    }
    if (grimoire_is_pp_worker(*e)) {
        grimoire_pp_worker_loop(*e);
        grimoire_delete(e);
        return 0;
    }
    const bool parallel=std::getenv("GRIMOIRE_PP_WORLD_SIZE")!=nullptr;
    GrimoireScheduler* sc=parallel?grimoire_scheduler_new(*e,3):nullptr;
    if(sc && grimoire_scheduler_width(*sc)<3) {
        std::fprintf(stderr,"pipeline did not enable batching\n");
        grimoire_scheduler_delete(sc); grimoire_pp_shutdown(*e); grimoire_delete(e);
        return 7;
    }
    std::vector<int32_t> all;
    const long steps0=g_batch_decode_steps, rows0=g_batch_decode_rows;
    // A fresh group after a cancelled stream catches protocol misalignment.
    for(int round=0;round<2;++round) {
        const auto prompts=requests();
        std::vector<std::vector<int32_t>> replies(3);
        std::vector<std::exception_ptr> errors(3);
        std::barrier start(3);
        auto client=[&](int k) {
            if(parallel) start.arrive_and_wait();
            try {
                int emitted=0;
                auto emit=[&](int32_t) { return !(round==0 && k==1) || ++emitted<3; };
                if(sc) grimoire_scheduler_generate(*sc,prompts[k],kWant,-1,-1,replies[k],emit);
                else grimoire_serve_generate(*e,prompts[k],kWant,-1,replies[k],-1,emit);
            } catch(...) { errors[k]=std::current_exception(); }
        };
        if(parallel) {
            std::vector<std::thread> threads;
            for(int k=0;k<3;++k) threads.emplace_back(client,k);
            for(auto& thread:threads) thread.join();
        } else for(int k=0;k<3;++k) client(k);
        for(int k=0;k<3;++k) {
            if(errors[k]) {
                try { std::rethrow_exception(errors[k]); }
                catch(const std::exception& ex) { std::fprintf(stderr,"%s\n",ex.what()); }
                if(sc) grimoire_scheduler_delete(sc);
                grimoire_pp_shutdown(*e); grimoire_delete(e); return 4;
            }
            all.insert(all.end(),replies[k].begin(),replies[k].end());
        }
    }
    if(sc) grimoire_scheduler_delete(sc);
    if(parallel && g_batch_decode_rows-rows0<=g_batch_decode_steps-steps0) {
        std::fprintf(stderr,"pipeline never ran a multi-row step\n");
        grimoire_pp_shutdown(*e); grimoire_delete(e); return 8;
    }
    grimoire_pp_shutdown(*e);
    grimoire_delete(e);
    if (!out.empty()) {
        std::FILE* f = std::fopen(out.c_str(), "w");
        if (!f) return 5;
        for (int32_t t : all) std::fprintf(f, "%d\n", t);
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
static std::string join(const std::vector<int32_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size() && i < 12; ++i)
        s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

static pid_t spawn(const char* self, const std::string& dir, const std::string& out,
                   const char* fmt, const std::vector<std::string>& env) {
    const pid_t pid = ::fork();
    if (pid != 0) return pid;
    for (const auto& kv : env) {
        const size_t eq = kv.find('=');
        ::setenv(kv.substr(0, eq).c_str(), kv.substr(eq+1).c_str(), 1);
    }
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

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    ::setenv("GRIMOIRE_SEQ_SLOTS","4",1);
    ::setenv("GRIMOIRE_BATCHED_PREFILL_NOXMX","1",1);
    if (argc >= 4) {                       // rank mode
        const std::string fs_(argv[3]);
        const Fmt fmt = fs_ == "fp8" ? Fmt::FP8_E4M3 : Fmt::BF16;
        return run_rank(argv[1], std::string(argv[2]) == "-" ? "" : argv[2], fmt);
    }

    std::printf("=== a resident pipeline, batching with cancellation ===\n");
    std::printf("three requests down one pipeline, answered at RANK 0,\n"
                "must equal the same three from a single process.\n\n");

    char tmpl[] = "/tmp/grimoire-ppsrv-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;
    const char* self = argv[0];

    struct Case { mini::Arch arch; const char* fmt; };
    std::vector<Case> cases = {
        { mini::dense(4), "bf16" },
        // fp8 is THE dual-GPU format -- DAY-ONE recommends it for two
        // cards, so it is the configuration this gate exists for.
        { mini::dense(4), "fp8"  },
        { mini::moe(4),   "fp8"  },
        // A hybrid carries the recurrent state a pipeline split has to
        // keep straight, and it is what Ornith is.
        { mini::hybrid(4), "fp8" },
        { mini::muse(4), "fp8" },
        { mini::gemma4(4), "fp8" },
    };

    for (auto& c : cases) {
        const fs::path dir = root / (std::string(c.arch.name) + "-" + c.fmt);
        mini::write_model(dir, c.arch);
        std::printf("%-8s %-5s ", c.arch.name, c.fmt);

        // Reference: one process, the same three requests in a row.
        const std::string ref_out = (dir/"single.txt").string();
        if (!wait_ok({spawn(self, dir.string(), ref_out, c.fmt,
                            {"GRIMOIRE_DEVICE_ANY=1"})}, "single")) {
            ++g_fail; std::printf("single-process run FAILED\n"); continue;
        }
        const auto ref = read_tokens(ref_out);
        if (ref.empty()) { ++g_fail; std::printf("no reference tokens\n"); continue; }

        // Two and three stages.  Three is not padding: it is the only
        // shape with a MIDDLE stage, which both receives a request and
        // forwards it, and that is the half of the protocol a two-stage
        // chain never runs.
        for (int world : {2, 3}) {
            const auto devices=sycl::device::get_devices(sycl::info::device_type::gpu);
            if(!devices.empty() && int(devices.size())<world) {
                std::printf("PP%d SKIPPED: insufficient GPUs ",world); continue;
            }
            const std::string sock = (dir/("s"+std::to_string(world)+".sock")).string();
            const std::string out  = (dir/("s"+std::to_string(world)+".txt")).string();
            const char* layers = world == 2 ? "2,2" : "2,1,1";
            std::vector<pid_t> pids;
            for (int r = world - 1; r >= 0; --r) {
                std::vector<std::string> env{
                    "GRIMOIRE_PP_RANK=" + std::to_string(r),
                    "GRIMOIRE_PP_WORLD_SIZE=" + std::to_string(world),
                    "GRIMOIRE_PP_LAYERS=" + std::string(layers),
                    "GRIMOIRE_PP_SOCKET=" + sock,
                    "GRIMOIRE_DEVICE_ANY=1"};
                // RANK 0's tokens, not the last stage's: rank 0 is the
                // one that would be holding an HTTP connection.
                pids.push_back(spawn(self, dir.string(), r == 0 ? out : "-",
                                     c.fmt, env));
            }
            const std::string what = "pp" + std::to_string(world);
            if (!wait_ok(pids, what.c_str())) {
                ++g_fail; std::printf("PP%d FAILED ", world); continue;
            }
            const auto got = read_tokens(out);
            const bool match = (got == ref);
            std::printf("PP%d %s ", world, match ? "match" : "DIFFERS");
            if (!match)
                std::printf("\n    single %s\n    pp%d    %s\n",
                            join(ref).c_str(), world, join(got).c_str());
            CHECK(match,
                  "a resident pipeline answered differently from one process, "
                  "or the front end did not get the tokens back");
            // Three requests of 6 tokens each.  Short output would mean a
            // later request was dropped rather than answered wrongly --
            // which is what a desynchronised header looks like.
            CHECK(got.size() == ref.size(),
                  "the pipeline answered fewer tokens than it was asked for, "
                  "so a later request did not survive the one before it");
        }
        std::printf("\n");
    }

    if (!g_fail) fs::remove_all(root);
    else std::printf("\nfixtures kept in %s\n", root.c_str());
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
