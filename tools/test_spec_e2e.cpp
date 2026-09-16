// =====================================================================
//  test_spec_e2e.cpp -- speculation must not change the answer.
//
//  Speculative decoding is an exactness claim, not an approximation:
//  the target model verifies every drafted token, so greedy decode with
//  a drafter must emit EXACTLY the tokens greedy decode without one
//  emits.  A drafter that is merely bad costs throughput; a verifier
//  that is wrong costs correctness, and nothing else in this tree would
//  notice -- the output stays fluent, it is just not the model's output.
//
//  That property had never been checked.  It is the only test that can
//  tell "the drafter proposes poorly" (fine, a speed problem) from "the
//  verify/rollback path is broken" (not fine, a correctness problem),
//  and it is cheap: same prompt, same model, speculation off then on.
//
//  It also records accepted-per-step via GRIMOIRE_SPEC_STATS, because a
//  drafter that is CORRECT but never accepted is a silent 1.0x -- it
//  looks like a working feature and buys nothing.
//
//  Run:
//    ./bin/test_spec_e2e                     # drive every case
//    ./bin/test_spec_e2e <dir> <out> <fmt>   # one run (used internally)
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
static const int kWant = 16;

static int run_one(const std::string& dir, const std::string& out, const std::string& fname) {
    const Fmt fmt = fname == "fp8"   ? Fmt::FP8_E4M3
                  : fname == "int4"  ? Fmt::INT4
                  : fname == "mxfp4" ? Fmt::MXFP4
                                     : Fmt::BF16;
    std::string err;
    // A drafting refusal must leave the PIPELINE usable, not just fail
    // cleanly on the stage that noticed.  With capacity 32, a valid
    // 25-token prompt and M=8, the drafter cannot prepare (25+8 > 32) --
    // so the last stage refuses, and every earlier stage is already
    // waiting on a reply it will otherwise never get.  The engine and its
    // sockets are then REUSED for a normal request below: closing the
    // model here would hide both a blocked peer and any bytes left on the
    // wire by a wrongly-shaped reply.
    const bool refusal_case =
        std::getenv("GRIMOIRE_TEST_DFLASH_CONTEXT_REFUSAL") != nullptr;
    Grimoire* e = grimoire_new();
    if (!e) return 2;
    if (!grimoire_load(*e, dir, fmt, refusal_case ? 32 : 256, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        grimoire_delete(e); return 3;
    }
    std::vector<int32_t> toks;
    FinishReason r{};
    if (refusal_case) {
        bool refused = false;
        try {
            grimoire_serve_generate(*e, std::vector<int32_t>(25, 7), 1,
                                    -1, toks, -1, {}, &r);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "expected refusal: %s\n", ex.what());
            refused = true;
        }
        if (!refused) {
            std::fprintf(stderr, "the refusal case did not refuse\n");
            grimoire_delete(e); return 6;
        }
        toks.clear();
    }
    try {
        grimoire_serve_generate(*e, kPrompt, kWant, -1, toks, -1, {}, &r);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "generate: %s\n", ex.what());
        grimoire_delete(e); return 4;
    }
    grimoire_delete(e);
    if (!out.empty() && out != "-") {
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

static std::string join(const std::vector<int32_t>& v) {
    std::string s;
    for (int32_t t : v) { s += ' '; s += std::to_string(t); }
    return s;
}

// Run one configuration in its own process and return its exit status.
// Fresh process per configuration: speculation reads its switches through
// function-local statics, so a second configuration in the same process
// would silently reuse the first one's answer.
static int spawn_run(const char* self, const std::string& dir, const std::string& out,
                     const char* fmt, const std::vector<std::string>& env,
                     const std::string& log) {
    const pid_t pid = ::fork();
    if (pid == 0) {
        for (const auto& kv : env) {
            const size_t eq = kv.find('=');
            ::setenv(kv.substr(0, eq).c_str(), kv.substr(eq+1).c_str(), 1);
        }
        std::freopen(log.c_str(), "w", stdout);
        std::freopen(log.c_str(), "a", stderr);
        const char* av[] = { self, dir.c_str(), out.c_str(), fmt, nullptr };
        ::execv(self, const_cast<char* const*>(av));
        ::_exit(127);
    }
    int st = 0;
    ::waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -WTERMSIG(st);
}

// Run one job across two ranks and return {writer rc, peer rc}.  The
// WRITER is the rank that emits the tokens: rank 0 under TP (every rank
// has the full model), the LAST stage under PP (only it owns the final
// hidden state).  The peer's output goes to "-" and is discarded.
static std::pair<int,int> spawn_two(const char* self, const std::string& dir,
        const std::string& out, const char* fmt,
        const std::vector<std::string>& common, const std::string& var,
        int writer, const fs::path& logdir, const char* tag) {
    const int peer_rank = writer == 0 ? 1 : 0;
    auto envfor = [&](int r) {
        auto v = common; v.push_back(var + "=" + std::to_string(r)); return v;
    };
    const pid_t pid = ::fork();
    if (pid == 0) {
        for (const auto& kv : envfor(peer_rank)) {
            const size_t eq = kv.find('=');
            ::setenv(kv.substr(0, eq).c_str(), kv.substr(eq+1).c_str(), 1);
        }
        // Named locals: a temporary's c_str() dangles at the end of the
        // full expression, and execv would receive a freed pointer.
        const std::string d = dir;
        const std::string l =
            (logdir/(std::string(tag)+std::to_string(peer_rank)+".log")).string();
        std::freopen(l.c_str(), "w", stdout);
        std::freopen(l.c_str(), "a", stderr);
        const char* av[] = { self, d.c_str(), "-", fmt, nullptr };
        ::execv(self, const_cast<char* const*>(av));
        ::_exit(127);
    }
    const std::string wlog =
        (logdir/(std::string(tag)+std::to_string(writer)+".log")).string();
    const int rcw = spawn_run(self, dir, out, fmt, envfor(writer), wlog);
    int st = 0; ::waitpid(pid, &st, 0);
    return { rcw, WIFEXITED(st) ? WEXITSTATUS(st) : -WTERMSIG(st) };
}

// Read a GRIMOIRE_DFLASH_DUMP file whole.  The drafter writes its
// concatenated target-tap row as g_01_aux_<pos>.f32 on its first context
// ingest, which is exactly the tensor a pipeline has to reassemble from
// several stages.
static std::vector<char> read_blob(const std::string& path) {
    std::vector<char> v;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n > 0) { v.resize(size_t(n)); if (std::fread(v.data(),1,v.size(),f) != v.size()) v.clear(); }
    std::fclose(f);
    return v;
}

// The "(A of D drafted" counts out of a spec line, so two runs can be
// compared on what the drafter PROPOSED rather than only on the final
// tokens.  Speculation is exact, so the tokens match whether or not the
// drafter saw the right inputs -- the acceptance count is the only thing
// that moves when a draft is computed from the wrong state.
static std::pair<int,int> spec_counts(const std::string& line) {
    const size_t a = line.find('(');
    if (a == std::string::npos) return {-1,-1};
    int acc = -1, drafted = -1;
    if (std::sscanf(line.c_str()+a, "(%d of %d", &acc, &drafted) != 2) return {-1,-1};
    return {acc, drafted};
}

// Pull the "spec:" line GRIMOIRE_SPEC_STATS prints, so a correct-but-
// never-accepted drafter is visible rather than passing quietly.
static std::string spec_line(const std::string& log) {
    std::FILE* f = std::fopen(log.c_str(), "r");
    if (!f) return {};
    char line[512]; std::string found;
    while (std::fgets(line, sizeof(line), f))
        if (std::strstr(line, "spec:")) found = line;
    std::fclose(f);
    while (!found.empty() && (found.back()=='\n' || found.back()==' ')) found.pop_back();
    size_t b = found.find_first_not_of(' ');
    return b == std::string::npos ? found : found.substr(b);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc >= 4) return run_one(argv[1], argv[2], argv[3]);

    std::printf("=== speculation must not change the answer ===\n\n");

    char tmpl[] = "/tmp/grimoire-spec-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;
    const char* self = argv[0];

    struct Case { mini::Arch arch; const char* fmt; };
    std::vector<Case> cases = {
        { mini::dense(4, true), "bf16" },
        { mini::dense(4, true), "fp8"  },
        { mini::moe(4,   true), "bf16" },
        { mini::moe(4,   true), "fp8"  },
        // The one that actually exercises the rollback: a DeltaNet layer
        // carries a recurrent state and a convolution ring, and rejecting
        // a draft has to restore BOTH exactly.  Every other model here has
        // no recurrent state at all, so nothing was testing that code.
        { mini::hybrid(4, true), "bf16" },
        { mini::hybrid(4, true), "fp8"  },
        // Muse: the SANDWICH forward path.  Every case above is the Qwen
        // residual graph, so nothing was checking that speculation stays
        // exact when the layer graph differs -- the drafter chains on the
        // UNNORMALISED hidden state, and forward_muse leaves a different
        // one there than forward() does.  Muse ran nowhere off the card
        // before 2026-09-16; this is the last of its paths to get a gate.
        { mini::muse(6, true), "bf16" },
        { mini::muse(6, true), "fp8"  },
    };

    for (auto& c : cases) {
        const fs::path dir = root / (std::string(c.arch.name) + "-" + c.fmt);
        mini::write_model(dir, c.arch);
        std::printf("%-12s %-5s  ", c.arch.name, c.fmt);

        // ---- plain greedy decode, no drafter ------------------------
        const std::string plain_out = (dir/"plain.txt").string();
        const std::string plain_log = (dir/"plain.log").string();
        int rc = spawn_run(self, dir.string(), plain_out, c.fmt, {}, plain_log);
        if (rc != 0) {
            ++g_fail;
            std::printf("plain decode FAILED (%d)\n", rc);
            std::printf("   %s\n", spec_line(plain_log).c_str());
            continue;
        }
        const auto plain = read_tokens(plain_out);
        if (int(plain.size()) != kWant) {
            ++g_fail;
            std::printf("plain decode gave %zu tokens, wanted %d\n", plain.size(), kWant);
            continue;
        }

        // ---- same thing with MTP speculation on ---------------------
        // Several draft depths: an off-by-one in the accept loop or the
        // rollback usually survives one depth and not the others.
        bool all_ok = true, spec_off = false;
        std::string last_stats;
        for (const char* k : {"1", "2", "3", "5"}) {
            const std::string out = (dir/(std::string("mtp")+k+".txt")).string();
            const std::string log = (dir/(std::string("mtp")+k+".log")).string();
            rc = spawn_run(self, dir.string(), out, c.fmt,
                           {"GRIMOIRE_MTP=1", std::string("GRIMOIRE_MTP_K=")+k,
                            "GRIMOIRE_SPEC_STATS=1", "GRIMOIRE_DEVICE_ANY=1"}, log);
            if (rc != 0) {
                ++g_fail; all_ok = false;
                std::printf("\n   MTP K=%s FAILED (%d): %s", k, rc, spec_line(log).c_str());
                continue;
            }
            const auto spec = read_tokens(out);
            if (spec != plain) {
                ++g_fail; all_ok = false;
                std::printf("\n   MTP K=%s CHANGED THE OUTPUT\n     plain:%s\n     spec :%s",
                            k, join(plain).c_str(), join(spec).c_str());
                continue;
            }
            last_stats = spec_line(log);
            if (last_stats.empty()) spec_off = true;   // no rounds were run
        }
        if (all_ok) {
            // "identical" is trivially true if no draft was ever made.
            // Say which happened -- a silently disabled feature that
            // reports PASS is the failure mode this whole file exists to
            // prevent.
            if (spec_off) {
                std::printf("speculation REFUSED here (see the load banner) "
                            "-- output matches plain decode, nothing drafted\n");
                continue;
            }
            std::printf("identical at K=1,2,3,5");
            if (!last_stats.empty()) std::printf("   [%s]", last_stats.c_str());
        } else {
            std::printf("\n");
            continue;
        }

        // ---- speculation across two processes ----------------------
        // Dual GPU must not be a reason to lose the drafter, and turning
        // it on across ranks must still not change the answer.  Every
        // rank must draft the SAME token or the ranks diverge and the
        // KV caches stop describing the same sequence -- which is silent.
        const std::string tsock = (dir/"tp.sock").string();
        const std::string tp_out = (dir/"tp.txt").string();
        const std::string tp_log = (dir/"tp0.log").string();
        const std::vector<std::string> tp_common{
            "GRIMOIRE_TP_WORLD_SIZE=2", "GRIMOIRE_TP_SOCKET="+tsock,
            "GRIMOIRE_MTP=1", "GRIMOIRE_MTP_K=3",
            "GRIMOIRE_SPEC_STATS=1", "GRIMOIRE_DEVICE_ANY=1"};
        auto with = [&](const char* extra) {
            auto v = tp_common; v.push_back(extra); return v;
        };
        const pid_t peer = ::fork();
        if (peer == 0) {
            for (const auto& kv : with("GRIMOIRE_TP_RANK=1")) {
                const size_t eq = kv.find('=');
                ::setenv(kv.substr(0, eq).c_str(), kv.substr(eq+1).c_str(), 1);
            }
            // Named, not temporaries: dir.string().c_str() dangles the
            // moment the full expression ends, so execv received a freed
            // pointer -- the child tried to open "<garbage>/config.json"
            // and died while rank 0 waited on it forever.
            const std::string d1 = dir.string();
            const std::string l1 = (dir/"tp1.log").string();
            std::freopen(l1.c_str(), "w", stdout);
            std::freopen(l1.c_str(), "a", stderr);
            const char* av[] = { self, d1.c_str(), "-", c.fmt, nullptr };
            ::execv(self, const_cast<char* const*>(av));
            ::_exit(127);
        }
        const int rc0 = spawn_run(self, dir.string(), tp_out, c.fmt,
                                  with("GRIMOIRE_TP_RANK=0"), tp_log);
        int pst = 0; ::waitpid(peer, &pst, 0);
        const int rc1 = WIFEXITED(pst) ? WEXITSTATUS(pst) : -WTERMSIG(pst);
        if (rc0 != 0 || rc1 != 0) {
            ++g_fail;
            std::printf("\n   TP+MTP FAILED (rank0 %d, rank1 %d)\n", rc0, rc1);
            std::printf("     %s\n", spec_line(tp_log).c_str());
            continue;
        }
        const auto tp = read_tokens(tp_out);
        if (tp != plain) {
            ++g_fail;
            std::printf("\n   TP+MTP CHANGED THE OUTPUT\n     plain:%s\n     tp   :%s\n",
                        join(plain).c_str(), join(tp).c_str());
            continue;
        }
        std::printf("   TP+MTP match");

        // ---- the same, pipelined ------------------------------------
        // Under PP only the LAST stage owns the MTP head, so this also
        // checks the two backward hops that keep the stages in step: the
        // draft token, and the verified tokens the head chose.  Rank 0
        // must roll back to exactly the position rank 1 rolled back to.
        const std::string psock = (dir/"pp.sock").string();
        const std::string pp_out = (dir/"pp.txt").string();
        const std::string pp_log = (dir/"pp1.log").string();
        const std::vector<std::string> pp_common{
            "GRIMOIRE_PP_WORLD_SIZE=2", "GRIMOIRE_PP_SPLIT=2",
            "GRIMOIRE_PP_SOCKET="+psock, "GRIMOIRE_MTP=1", "GRIMOIRE_MTP_K=3",
            "GRIMOIRE_SPEC_STATS=1", "GRIMOIRE_DEVICE_ANY=1"};
        auto pwith = [&](const char* extra) {
            auto v = pp_common; v.push_back(extra); return v;
        };
        const pid_t r0 = ::fork();
        if (r0 == 0) {
            for (const auto& kv : pwith("GRIMOIRE_PP_RANK=0")) {
                const size_t eq = kv.find('=');
                ::setenv(kv.substr(0, eq).c_str(), kv.substr(eq+1).c_str(), 1);
            }
            const std::string d0 = dir.string();
            const std::string l0 = (dir/"pp0.log").string();
            std::freopen(l0.c_str(), "w", stdout);
            std::freopen(l0.c_str(), "a", stderr);
            const char* av[] = { self, d0.c_str(), "-", c.fmt, nullptr };
            ::execv(self, const_cast<char* const*>(av));
            ::_exit(127);
        }
        // The last stage writes the tokens: it is the one that has them.
        const int prc1 = spawn_run(self, dir.string(), pp_out, c.fmt,
                                   pwith("GRIMOIRE_PP_RANK=1"), pp_log);
        int p0st = 0; ::waitpid(r0, &p0st, 0);
        const int prc0 = WIFEXITED(p0st) ? WEXITSTATUS(p0st) : -WTERMSIG(p0st);
        if (prc0 != 0 || prc1 != 0) {
            ++g_fail;
            std::printf("\n   PP+MTP FAILED (rank0 %d, rank1 %d)\n", prc0, prc1);
            std::printf("     %s\n", spec_line(pp_log).c_str());
            continue;
        }
        const auto pp = read_tokens(pp_out);
        if (pp != plain) {
            ++g_fail;
            std::printf("\n   PP+MTP CHANGED THE OUTPUT\n     plain:%s\n     pp   :%s\n",
                        join(plain).c_str(), join(pp).c_str());
            continue;
        }
        std::printf("   PP+MTP match\n");
    }

    // ---- DFlash: the other drafter, and the one never executed here ---
    // MTP is a head inside the target checkpoint; DFlash is a SEPARATE
    // model in its own directory, and its entire forward -- the target
    // layer taps, fc, the hidden norm, the context K/V inserted into the
    // draft's own cache, the 1+15 non-causal query block -- runs only when
    // such a directory is present.  No test ever provided one, so none of
    // that had executed anywhere except on the Tower.  The K2 path taught
    // what that costs: four engine bugs in code that had been read many
    // times and never run, one of them a device heap overrun.
    //
    // TENSOR parallel is covered below: the drafter is small, so it loads
    // replicated on every rank and each rank drafts identically.  Two spots
    // had to be fixed for that -- the block embed reads the target's
    // embedding table, which TP shards over the VOCABULARY, and a drafter
    // that shares the target's lm_head reads a head sharded the same way.
    // Both were silent: every rank produced a plausible embedding of the
    // wrong token, drafted a different token from the next one, and the
    // ranks' KV caches quietly stopped describing the same sequence.
    //
    // PIPELINE parallel is covered below too, and it is the harder half:
    // the drafter reads the residual stream tapped at specific TARGET
    // layers, and a split puts those layers on different stages.  Each
    // stage now captures the taps it owns and forwards the block, so the
    // last stage -- which hosts the drafter, as it hosts the MTP head --
    // ends up with the whole concatenated row.  The mini drafter taps
    // layers 0 and 2 of a 4-layer target, so a 2/2 split lands one tap on
    // each stage and the test fails if the forwarding is dropped.
    for (bool own : {false, true}) {
        // own=false: the drafter ships no head or embedding, so it shares
        //            the target's -- what this loader always assumed.
        // own=true : the drafter ships its own reduced-vocabulary lm_head
        //            and a d2t map, the EAGLE3-derived layout.  Ignoring
        //            either is silent: the draft still emits real token
        //            ids, they are just the wrong ones.
        const mini::Arch target = mini::dense(4, false);
        const mini::Arch draft  = mini::dflash_draft(2, own, own);
        for (const char* fmt : {"bf16", "fp8"}) {
            const std::string tag = std::string(draft.name) + "-" + fmt;
            const fs::path tdir = root / tag;
            const fs::path ddir = root / (tag + "-drafter");
            mini::write_model(tdir, target);
            mini::write_model(ddir, draft, 20260913);   // not the target's seed
            std::printf("%-12s %-5s  ", draft.name, fmt);

            const std::string plain_out = (tdir/"plain.txt").string();
            const std::string plain_log = (tdir/"plain.log").string();
            int rc = spawn_run(self, tdir.string(), plain_out, fmt,
                               {"GRIMOIRE_DEVICE_ANY=1"}, plain_log);
            if (rc != 0) {
                ++g_fail;
                std::printf("plain decode FAILED (%d)\n", rc);
                continue;
            }
            const auto plain = read_tokens(plain_out);
            if (int(plain.size()) != kWant) {
                ++g_fail;
                std::printf("plain decode gave %zu tokens, wanted %d\n",
                            plain.size(), kWant);
                continue;
            }

            // Two block widths.  M is the query block INCLUDING the anchor
            // row, so M=4 verifies 3 drafts and M=16 verifies 15; an
            // off-by-one in the accept loop or the rollback usually
            // survives one of them and not the other.
            bool all_ok = true, drafted = false;
            std::string last_stats;
            for (const char* m : {"4", "16"}) {
                const std::string out = (tdir/(std::string("df")+m+".txt")).string();
                const std::string log = (tdir/(std::string("df")+m+".log")).string();
                rc = spawn_run(self, tdir.string(), out, fmt,
                               {"GRIMOIRE_DFLASH_MODEL="+ddir.string(),
                                std::string("GRIMOIRE_DFLASH_M=")+m,
                                "GRIMOIRE_SPEC_STATS=1", "GRIMOIRE_DEVICE_ANY=1"},
                               log);
                if (rc != 0) {
                    ++g_fail; all_ok = false;
                    std::printf("\n   DFlash M=%s FAILED (%d): %s", m, rc,
                                spec_line(log).c_str());
                    continue;
                }
                const auto spec = read_tokens(out);
                if (spec != plain) {
                    ++g_fail; all_ok = false;
                    std::printf("\n   DFlash M=%s CHANGED THE OUTPUT"
                                "\n     plain:%s\n     spec :%s",
                                m, join(plain).c_str(), join(spec).c_str());
                    continue;
                }
                last_stats = spec_line(log);
                if (!last_stats.empty()) drafted = true;
            }
            if (!all_ok) { std::printf("\n"); continue; }
            if (!drafted) {
                // "identical" is free when nothing was ever drafted, and a
                // drafter that silently failed to load would pass that way.
                ++g_fail;
                std::printf("NOTHING WAS DRAFTED -- the drafter did not load "
                            "or speculation was refused\n");
                continue;
            }
            std::printf("identical at M=4,16   [%s]\n", last_stats.c_str());

            // ---- the same, with a drafter that IS accepted ------------
            // Everything above ran at 0% acceptance -- random weights make
            // a drafter that agrees with a random target about once in
            // `vocab` -- so the engine's accept-then-continue path had
            // never run here, only its rollback.  Point a drafter at the
            // token this target repeats most and it is accepted wherever
            // the target emits it and rejected everywhere else, which
            // drives both halves in one generation.  The output must still
            // be the same output.
            int best = -1, best_n = 0;
            for (int32_t t : plain) {
                int n = 0;
                for (int32_t u : plain) if (u == t) ++n;
                if (n > best_n) { best_n = n; best = t; }
            }
            const fs::path fdir = root / (tag + "-forced");
            mini::write_model(fdir, mini::dflash_draft(2, own, own, best), 20260913);
            const std::string fout = (tdir/"forced.txt").string();
            const std::string flog = (tdir/"forced.log").string();
            rc = spawn_run(self, tdir.string(), fout, fmt,
                           {"GRIMOIRE_DFLASH_MODEL="+fdir.string(),
                            "GRIMOIRE_DFLASH_M=8", "GRIMOIRE_SPEC_STATS=1",
                            "GRIMOIRE_DEVICE_ANY=1"}, flog);
            const auto forced = read_tokens(fout);
            const std::string fstats = spec_line(flog);
            if (rc != 0) {
                ++g_fail;
                std::printf("   forced-agreement drafter FAILED (%d): %s\n",
                            rc, fstats.c_str());
            } else if (forced != plain) {
                ++g_fail;
                std::printf("   forced-agreement drafter CHANGED THE OUTPUT"
                            "\n     plain:%s\n     spec :%s\n",
                            join(plain).c_str(), join(forced).c_str());
            } else if (fstats.empty() || fstats.find("(0 of ") != std::string::npos) {
                // The whole point of this case is a non-zero acceptance:
                // at zero it proves nothing and must not pass quietly.
                // Match the drafted COUNT, not the percentage -- "0.0%" is
                // also a substring of "10.0%", so a rate check here would
                // call a healthy 10% acceptance a failure.
                ++g_fail;
                std::printf("   forced-agreement drafter accepted NOTHING "
                            "(token %d appears %d times): %s\n",
                            best, best_n, fstats.c_str());
            } else {
                std::printf("   accept path: identical, token %d x%d   [%s]\n",
                            best, best_n, fstats.c_str());
            }

            // ---- DFlash across two tensor-parallel ranks --------------
            // Run the FORCED drafter, not the random one: at zero
            // acceptance every rank rolls back every draft and the test
            // would pass without either rank ever committing a drafted
            // token.  What has to hold is that both ranks draft the SAME
            // token from sharded weights and commit the same prefix --
            // and that the result is still, exactly, the plain output.
            if (rc == 0) {
                const std::string tsock = (tdir/"df-tp.sock").string();
                const std::string tout  = (tdir/"df-tp.txt").string();
                const std::vector<std::string> common{
                    "GRIMOIRE_TP_WORLD_SIZE=2", "GRIMOIRE_TP_SOCKET="+tsock,
                    "GRIMOIRE_DFLASH_MODEL="+fdir.string(),
                    "GRIMOIRE_DFLASH_M=8", "GRIMOIRE_SPEC_STATS=1",
                    "GRIMOIRE_DEVICE_ANY=1"};
                const auto r = spawn_two(self, tdir.string(), tout, fmt,
                                         common, "GRIMOIRE_TP_RANK", 0,
                                         tdir, "df-tp");
                const std::string tlog = (tdir/"df-tp0.log").string();
                const std::string tstats = spec_line(tlog);
                if (r.first != 0 || r.second != 0) {
                    ++g_fail;
                    std::printf("   TP+DFlash FAILED (rank0 %d, rank1 %d): %s\n",
                                r.first, r.second, tstats.c_str());
                } else if (read_tokens(tout) != plain) {
                    ++g_fail;
                    std::printf("   TP+DFlash CHANGED THE OUTPUT"
                                "\n     plain:%s\n     tp   :%s\n",
                                join(plain).c_str(),
                                join(read_tokens(tout)).c_str());
                } else if (tstats.empty() ||
                           tstats.find("(0 of ") != std::string::npos) {
                    // Same trap as the single-process forced case: an
                    // identical answer is free if nothing was drafted, and
                    // a drafter that failed to load under TP would pass.
                    ++g_fail;
                    std::printf("   TP+DFlash accepted NOTHING -- the drafter "
                                "did not load or drafted nothing: %s\n",
                                tstats.c_str());
                } else if (spec_counts(tstats) != spec_counts(fstats)) {
                    // Exact output is NOT enough on its own: speculation is
                    // exact, so the tokens match even when every draft was
                    // computed from the wrong state and rejected.
                    ++g_fail;
                    std::printf("   TP+DFlash accepted a DIFFERENT number than "
                                "one process -- the drafter saw different state"
                                "\n     1proc:%s\n     tp   :%s\n",
                                fstats.c_str(), tstats.c_str());
                } else {
                    std::printf("   TP+DFlash: identical, same acceptance   [%s]\n",
                                tstats.c_str());
                }

                // ---- DFlash across two PIPELINE stages ---------------
                // The case the tap forwarding exists for.  The mini
                // drafter taps target layers 0 and 2, which a 2/2 split
                // puts on DIFFERENT stages: stage 0 captures the first,
                // stage 1 the second, and only the forwarded block makes
                // the concatenated row the drafter's fc consumes.  Drop
                // the forwarding and the drafter still runs -- on a row
                // that is half zeros -- so the failure this catches is a
                // silent one.
                const std::string psock = (tdir/"df-pp.sock").string();
                const std::string pout  = (tdir/"df-pp.txt").string();
                const std::vector<std::string> pcommon{
                    "GRIMOIRE_PP_WORLD_SIZE=2", "GRIMOIRE_PP_SPLIT=2",
                    "GRIMOIRE_PP_SOCKET="+psock,
                    "GRIMOIRE_DFLASH_MODEL="+fdir.string(),
                    "GRIMOIRE_DFLASH_M=8", "GRIMOIRE_SPEC_STATS=1",
                    "GRIMOIRE_DEVICE_ANY=1"};
                // The LAST stage writes the tokens: it hosts the drafter
                // and the lm_head, so it is the one that has them.
                const auto pr = spawn_two(self, tdir.string(), pout, fmt,
                                          pcommon, "GRIMOIRE_PP_RANK", 1,
                                          tdir, "df-pp");
                const std::string plog2 = (tdir/"df-pp1.log").string();
                const std::string pstats = spec_line(plog2);
                if (pr.first != 0 || pr.second != 0) {
                    ++g_fail;
                    std::printf("   PP+DFlash FAILED (last %d, first %d): %s\n",
                                pr.first, pr.second, pstats.c_str());
                } else if (read_tokens(pout) != plain) {
                    ++g_fail;
                    std::printf("   PP+DFlash CHANGED THE OUTPUT"
                                "\n     plain:%s\n     pp   :%s\n",
                                join(plain).c_str(),
                                join(read_tokens(pout)).c_str());
                } else if (pstats.empty() ||
                           pstats.find("(0 of ") != std::string::npos) {
                    ++g_fail;
                    std::printf("   PP+DFlash accepted NOTHING -- the drafter "
                                "did not load or drafted nothing: %s\n",
                                pstats.c_str());
                } else if (spec_counts(pstats) != spec_counts(fstats)) {
                    // A consistency check, not the tap check: this fixture
                    // decodes every draft id to the same token, so bad taps
                    // move the logits without moving the proposal.  The tap
                    // comparison below is the one that covers forwarding.
                    ++g_fail;
                    std::printf("   PP+DFlash accepted a DIFFERENT number than "
                                "one process -- the forwarded taps do not match"
                                "\n     1proc:%s\n     pp   :%s\n",
                                fstats.c_str(), pstats.c_str());
                } else {
                    std::printf("   PP+DFlash: identical, same acceptance   [%s]\n",
                                pstats.c_str());
                }

                // ---- a refusal must leave the pipeline usable ---------
                if (!own && std::strcmp(fmt, "bf16") == 0) {
                    auto renv = pcommon;
                    renv.push_back("GRIMOIRE_TEST_DFLASH_CONTEXT_REFUSAL=1");
                    renv.push_back("GRIMOIRE_PP_SOCKET="+(tdir/"df-rec.sock").string());
                    const std::string rout = (tdir/"df-rec.txt").string();
                    const auto rr = spawn_two(self, tdir.string(), rout, fmt,
                                              renv, "GRIMOIRE_PP_RANK", 1,
                                              tdir, "df-rec");
                    if (rr.first != 0 || rr.second != 0) {
                        ++g_fail;
                        std::printf("   PP refusal/reuse FAILED (last %d, first %d)"
                                    " -- a peer was left blocked, or the wire was"
                                    " left dirty\n", rr.first, rr.second);
                    } else if (read_tokens(rout) != plain) {
                        ++g_fail;
                        std::printf("   PP refusal/reuse CHANGED THE OUTPUT of the"
                                    " request AFTER the refusal\n     plain:%s\n"
                                    "     after:%s\n", join(plain).c_str(),
                                    join(read_tokens(rout)).c_str());
                    } else {
                        std::printf("   PP refusal: both ranks recovered, next "
                                    "request identical\n");
                    }
                }

                // ---- the taps themselves, byte for byte ---------------
                // The assertion that actually covers the forwarding.  The
                // drafter dumps its concatenated target-tap row on its
                // first context ingest; in one process it builds that row
                // from its own layers, and under PP it has to reassemble
                // it from every stage.  Those must be the same bytes.
                //
                // Nothing else in this file can see a tap error: the
                // verifier makes the OUTPUT identical either way, and
                // this fixture decodes every draft id to the same token,
                // so the proposal does not move either.
                {
                    const fs::path d1 = tdir/"dump-one", d2 = tdir/"dump-pp";
                    fs::create_directories(d1); fs::create_directories(d2);
                    const std::string s1 = (tdir/"dump1.txt").string();
                    const std::string s2 = (tdir/"dump2.txt").string();
                    // Check these runs' exit codes.  Matching dump files
                    // do not mean the runs SUCCEEDED -- the dump is written
                    // during the first context ingest, long before
                    // generation ends, so a later crash in either process
                    // would leave two identical files and a silent pass.
                    const int rc1 = spawn_run(self, tdir.string(), s1, fmt,
                        {"GRIMOIRE_DFLASH_MODEL="+fdir.string(),
                         "GRIMOIRE_DFLASH_M=8", "GRIMOIRE_DEVICE_ANY=1",
                         "GRIMOIRE_DFLASH_DUMP="+d1.string()},
                        (tdir/"dump1.log").string());
                    const std::string psock2 = (tdir/"df-ppd.sock").string();
                    const auto rc2 = spawn_two(self, tdir.string(), s2, fmt,
                        {"GRIMOIRE_PP_WORLD_SIZE=2", "GRIMOIRE_PP_SPLIT=2",
                         "GRIMOIRE_PP_SOCKET="+psock2,
                         "GRIMOIRE_DFLASH_MODEL="+fdir.string(),
                         "GRIMOIRE_DFLASH_M=8", "GRIMOIRE_DEVICE_ANY=1",
                         "GRIMOIRE_DFLASH_DUMP="+d2.string()},
                        "GRIMOIRE_PP_RANK", 1, tdir, "df-ppd");
                    if (rc1 != 0 || rc2.first != 0 || rc2.second != 0) {
                        ++g_fail;
                        std::printf("   tap-dump runs FAILED (one %d, pp last "
                                    "%d, pp first %d)\n",
                                    rc1, rc2.first, rc2.second);
                    }
                    const auto a = read_blob((d1/"g_01_aux_0.f32").string());
                    const auto b = read_blob((d2/"g_01_aux_0.f32").string());
                    if (a.empty() || b.empty()) {
                        ++g_fail;
                        std::printf("   tap dump missing (one %zu bytes, pp %zu) "
                                    "-- the comparison did not run\n",
                                    a.size(), b.size());
                    } else if (a != b) {
                        ++g_fail;
                        size_t diff = 0;
                        for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
                            if (a[i] != b[i]) ++diff;
                        std::printf("   FORWARDED TAPS DIFFER from one process: "
                                    "%zu vs %zu bytes, %zu differing\n",
                                    a.size(), b.size(), diff);
                    } else {
                        std::printf("   PP taps: byte-identical to one process "
                                    "(%zu bytes)\n", a.size());
                    }
                }

                // ---- TP on the SHARED-head drafter ------------------
                // The forced fixture always ships its own lm_head, so the
                // cases above never reach the sharded-target-head branch
                // TP needs.  This drafter shares both the head and the
                // embedding table, which is the configuration where TP
                // shards BOTH out from under it.  Acceptance is zero here
                // (random weights), so the claim is exactness only.
                if (!own) {
                    const std::string ssock = (tdir/"df-tps.sock").string();
                    const std::string sout  = (tdir/"df-tps.txt").string();
                    const auto sr = spawn_two(self, tdir.string(), sout, fmt,
                        {"GRIMOIRE_TP_WORLD_SIZE=2", "GRIMOIRE_TP_SOCKET="+ssock,
                         "GRIMOIRE_DFLASH_MODEL="+ddir.string(),
                         "GRIMOIRE_DFLASH_M=8", "GRIMOIRE_SPEC_STATS=1",
                         "GRIMOIRE_DEVICE_ANY=1"},
                        "GRIMOIRE_TP_RANK", 0, tdir, "df-tps");
                    if (sr.first != 0 || sr.second != 0) {
                        ++g_fail;
                        std::printf("   TP+DFlash(shared head) FAILED "
                                    "(rank0 %d, rank1 %d): %s\n", sr.first,
                                    sr.second,
                                    spec_line((tdir/"df-tps0.log").string()).c_str());
                    } else if (read_tokens(sout) != plain) {
                        ++g_fail;
                        std::printf("   TP+DFlash(shared head) CHANGED THE OUTPUT"
                                    "\n     plain:%s\n     tp   :%s\n",
                                    join(plain).c_str(),
                                    join(read_tokens(sout)).c_str());
                    } else {
                        std::printf("   TP+DFlash(shared head): identical\n");
                    }
                }
            }
        }
    }

    if (!g_fail) fs::remove_all(root);
    else std::printf("\nlogs kept in %s\n", root.c_str());
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
