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

    // ---- TWO AGENTS, INTERLEAVED -------------------------------------
    // The case this is actually for.  With one snapshot each request
    // evicts the other's and both fall back to re-reading everything --
    // the very cost reuse exists to remove, reintroduced by the second
    // caller.  With a slot each, both resume.
    //
    // The check is again identity, not speed: agent A resuming must
    // answer exactly what agent A re-reading answers.  A slot that
    // returned agent B's state would be FLUENT -- it is a real
    // conversation, just not this one.
    //
    // It also asserts the invariant the token stream cannot show: SIX
    // turns of TWO conversations must occupy TWO slots.  A conversation
    // that takes a fresh slot per turn answers perfectly and evicts every
    // other agent, so nothing in the output would say so -- the budget is
    // deliberately roomy (8) precisely so the count is free to be wrong.
    //
    // One way to be wrong is only reachable on hardware this container
    // does not have: a cold request saves TWICE where batched prefill
    // runs -- the prompt at start_pos 0 from inside prefill(), then
    // prompt-plus-reply at the end -- and the second save must land in
    // the same slot as the first.  On a CPU device prefill() declines and
    // the caller falls back to forward(), so only the second save ever
    // happens here and this arm CANNOT see that case.  The occupancy
    // assertion is what would catch it on the card; it is stated here
    // rather than only on the Tower because stating it costs nothing.
    {
        ::setenv("GRIMOIRE_PREFIX_CACHE", "1", 1);
        ::setenv("GRIMOIRE_PREFIX_SLOTS", "8", 1);
        Grimoire* e = grimoire_new();
        std::string err;
        if (!e || !grimoire_load(*e, dir.string(), Fmt::BF16, 256, err)) {
            ++g_fail;
            std::printf("    FAIL: two-agent load failed: %s\n", err.c_str());
        } else {
            std::vector<int32_t> a{7, 11, 3, 42}, b{90, 1, 64, 23};
            std::vector<std::vector<int32_t>> got_a, got_b;
            const long before = b70::g_prefix_tokens_reused_calls;
            for (int t = 0; t < 3; ++t) {
                std::vector<int32_t> oa, ob; FinishReason r{};
                grimoire_serve_generate(*e, a, 5, -1, oa, -1, {}, &r);
                got_a.push_back(oa);
                a.insert(a.end(), oa.begin(), oa.end()); a.push_back(int32_t(50 + t));
                // B's request lands BETWEEN A's turns, which is what
                // evicts a single-slot cache.
                grimoire_serve_generate(*e, b, 5, -1, ob, -1, {}, &r);
                got_b.push_back(ob);
                b.insert(b.end(), ob.begin(), ob.end()); b.push_back(int32_t(60 + t));
            }
            const long resumes = b70::g_prefix_tokens_reused_calls - before;
            const int slots = grimoire_prefix_slots_used(*e);
            grimoire_delete(e);
            std::printf("%-22s %ld resumes across two interleaved agents\n",
                        "two agents", resumes);
            std::printf("%-22s %d occupied after 6 turns of 2 conversations\n",
                        "slots", slots);
            CHECK(slots == 2,
                  "two conversations did not settle into two slots, so a "
                  "conversation is consuming a slot it should be extending "
                  "and will evict every other agent as it grows");
            // Four of the six turns can resume (each agent's turns 2 and
            // 3).  With one slot it would be zero, because every request
            // evicts the other's snapshot.
            CHECK(resumes >= 4,
                  "interleaved agents did not both resume, so a second "
                  "caller is still evicting the first's context");

            // And the answers must match the same conversations run
            // WITHOUT any cache at all.
            ::unsetenv("GRIMOIRE_PREFIX_CACHE");
            ::unsetenv("GRIMOIRE_PREFIX_SLOTS");
            Grimoire* c = grimoire_new();
            if (!c || !grimoire_load(*c, dir.string(), Fmt::BF16, 256, err)) {
                ++g_fail;
                std::printf("    FAIL: cold two-agent load failed: %s\n", err.c_str());
            } else {
                std::vector<int32_t> ca{7, 11, 3, 42}, cb{90, 1, 64, 23};
                bool same = true;
                for (int t = 0; t < 3; ++t) {
                    std::vector<int32_t> oa, ob; FinishReason r{};
                    grimoire_serve_generate(*c, ca, 5, -1, oa, -1, {}, &r);
                    same = same && oa == got_a[size_t(t)];
                    ca.insert(ca.end(), oa.begin(), oa.end()); ca.push_back(int32_t(50 + t));
                    grimoire_serve_generate(*c, cb, 5, -1, ob, -1, {}, &r);
                    same = same && ob == got_b[size_t(t)];
                    cb.insert(cb.end(), ob.begin(), ob.end()); cb.push_back(int32_t(60 + t));
                }
                grimoire_delete(c);
                std::printf("%-22s %s\n", "two agents, answers",
                            same ? "identical to no cache" : "DIFFER");
                CHECK(same,
                      "an agent resuming answered differently from the same "
                      "agent re-reading, so a slot returned the wrong "
                      "conversation's state");
            }
        }
    }

    if (!g_fail) fs::remove_all(root);
    else std::printf("\nfixtures kept in %s\n", root.c_str());
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
