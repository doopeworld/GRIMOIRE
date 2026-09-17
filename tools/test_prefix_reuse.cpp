// =====================================================================
//  test_prefix_reuse.cpp -- resuming a conversation must change the
//  WORK and not the ANSWER.
//
//  A chat turn is the previous prompt plus the reply plus a new message,
//  so the state left at the end of one request is a strict PREFIX of the
//  next.  The cache used to demand a byte-identical prompt, which a
//  growing conversation never gives, so every turn re-read the entire
//  history.  At concurrency one.  By design.
//
//  This gate runs the SAME three-turn conversation twice against two
//  separate engines -- one resuming, one re-reading -- and requires:
//
//    * identical tokens on every turn.  Reuse is a claim about work, so
//      any difference in the output means the resumed state was wrong,
//      and a wrong resume is FLUENT: it answers from a conversation that
//      is almost this one.
//    * the resuming engine actually resumed.  The counter is read, not
//      assumed; without it this is one engine compared with itself.
//
//  Run: ./bin/test_prefix_reuse
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

namespace b70 {
extern long g_prefix_tokens_reused;
extern long g_prefix_tokens_reused_calls;
}

static int g_fail = 0;
static void CHECK(bool c, const char* why) {
    if (!c) { ++g_fail; std::printf("    FAIL: %s\n", why); }
}

static std::string show(const std::vector<int32_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size() && i < 6; ++i)
        s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== resuming a conversation ===\n");
    std::printf("same three turns, two engines: one resumes, one re-reads.\n"
                "identical tokens required; the resume counter is read.\n\n");

    char tmpl[] = "/tmp/grimoire-prefix-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;
    const fs::path dir = root / "dense";
    mini::write_model(dir, mini::dense());

    // The cache is opt-in.  The COLD arm must run with it off, or it
    // would resume too and there would be nothing to compare against.
    auto engine = [&](bool warm) -> Grimoire* {
        if (warm) ::setenv("GRIMOIRE_PREFIX_CACHE", "1", 1);
        else      ::unsetenv("GRIMOIRE_PREFIX_CACHE");
        Grimoire* e = grimoire_new();
        std::string err;
        if (!e || !grimoire_load(*e, dir.string(), Fmt::BF16, 256, err)) {
            std::printf("    FAIL: load failed: %s\n", err.c_str());
            ++g_fail; return nullptr;
        }
        return e;
    };

    // prefix_cache_enabled() is read ONCE into a static, so the two arms
    // cannot live in one process.  Run the cold arm first, keep its
    // answers, then re-exec logic is unnecessary: build the cold engine
    // while the variable is unset and the warm one after setting it --
    // the static is captured at first use, so do the cold arm's whole
    // conversation before the warm engine is created.
    std::vector<std::vector<int32_t>> cold_turns;
    std::vector<int32_t> convo{7, 11, 3, 42};
    {
        Grimoire* e = engine(false);
        if (!e) { std::printf("\nFAILURES (%d)\n", g_fail); return 1; }
        std::vector<int32_t> c = convo;
        for (int t = 0; t < 3; ++t) {
            std::vector<int32_t> out; FinishReason r{};
            grimoire_serve_generate(*e, c, 5, -1, out, -1, {}, &r);
            cold_turns.push_back(out);
            c.insert(c.end(), out.begin(), out.end());
            c.push_back(int32_t(50 + t));
        }
        grimoire_delete(e);
    }

    const long reused_before = b70::g_prefix_tokens_reused_calls;
    std::vector<std::vector<int32_t>> warm_turns;
    {
        Grimoire* e = engine(true);
        if (!e) { std::printf("\nFAILURES (%d)\n", g_fail); return 1; }
        std::vector<int32_t> c = convo;
        for (int t = 0; t < 3; ++t) {
            std::vector<int32_t> out; FinishReason r{};
            grimoire_serve_generate(*e, c, 5, -1, out, -1, {}, &r);
            warm_turns.push_back(out);
            c.insert(c.end(), out.begin(), out.end());
            c.push_back(int32_t(50 + t));
        }
        grimoire_delete(e);
    }

    const long resumes = b70::g_prefix_tokens_reused_calls - reused_before;
    std::printf("%-22s %ld resumes, %ld tokens not re-read\n", "reuse",
                resumes, b70::g_prefix_tokens_reused);
    CHECK(resumes > 0,
          "no request resumed, so this compared one configuration with "
          "itself and proves nothing");

    for (size_t t = 0; t < cold_turns.size(); ++t) {
        std::printf("turn %zu  re-read %-18s resumed %s\n", t + 1,
                    show(cold_turns[t]).c_str(), show(warm_turns[t]).c_str());
        CHECK(cold_turns[t] == warm_turns[t],
              "a resumed turn answered differently from a re-read one, so "
              "the restored state is not the state it claims to be");
    }

    if (!g_fail) fs::remove_all(root);
    else std::printf("\nfixtures kept in %s\n", root.c_str());
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
