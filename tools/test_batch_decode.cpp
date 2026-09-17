// =====================================================================
//  test_batch_decode.cpp -- decoding several conversations in ONE pass
//  must answer exactly what decoding them one at a time answers.
//
//  Decode is weight-bound: a step reads every active weight to produce
//  one token, and it reads the same weights whether it is producing one
//  token or eight.  So eight conversations stepped together cost about
//  what one costs, and a server that steps them one at a time is doing
//  eight times the memory traffic for the same work.  That is the whole
//  reason this path exists.
//
//  It is also an EXACTNESS claim, not an approximation: batching changes
//  which rows a kernel processes, never what any row computes.  So the
//  check is identity, and identity is the only check that can see the
//  failure mode.  Every way of getting per-row state wrong -- a position
//  taken from the wrong row, a key appended to the wrong conversation's
//  cache, an attention length off by one row -- produces finite,
//  plausible numbers and fluent text that belongs to a DIFFERENT
//  conversation.  Nothing weaker than "the same tokens" can see that.
//
//  Two properties this gate insists on, both learned the expensive way:
//
//   * THE PROMPTS ARE DIFFERENT LENGTHS.  Rule 16: at equal lengths
//     every per-row position is the same number, so the whole per-row
//     position machinery is exercised by a test that cannot fail on it.
//     The lengths here are deliberately co-prime-ish and the sequences
//     finish at different times.
//
//   * THE BATCH MUST ACTUALLY HAVE BATCHED.  A path that silently fell
//     back to stepping each sequence alone would pass an identity test
//     perfectly.  The counter is read, not assumed.
//
//  Run: GRIMOIRE_DEVICE_ANY=1 ./bin/test_batch_decode
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
extern long g_batch_decode_steps;    // batched decode steps taken
extern long g_batch_decode_rows;     // rows those steps carried
}

static int g_fail = 0;
static void CHECK(bool c, const char* why) {
    if (!c) { ++g_fail; std::printf("    FAIL: %s\n", why); }
}

static std::string show(const std::vector<int32_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size() && i < 8; ++i)
        s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== batched decode ===\n");
    std::printf("several conversations stepped together must answer exactly\n"
                "what each answers alone.  Prompts are DIFFERENT lengths, so\n"
                "every row sits at its own position (rule 16).\n\n");

    // Enough slots for every sequence in the batch, and on a device with
    // no matrix hardware the plain-SYCL GEMM that makes the batched path
    // runnable at all off the card.  BOTH arms then ingest their prompts
    // through the same batched prefill, so the only thing this gate
    // varies is how the DECODE steps were taken.  CORRECTNESS ONLY:
    // nothing timed here means anything (rule 8).
    ::setenv("GRIMOIRE_SEQ_SLOTS", "8", 1);
    ::setenv("GRIMOIRE_BATCHED_PREFILL_NOXMX", "1", 1);

    char tmpl[] = "/tmp/grimoire-batch-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;

    // Different lengths AND different content.  Same-length prompts make
    // every per-row position identical and the test blind to the bug
    // class it exists for.
    //
    // And LONG ENOUGH TO MATTER.  The first version used prompts of two
    // to six tokens, and on a hybrid model that was not a test: a
    // DeltaNet state shared between all four rows changed NOTHING in the
    // output, and a shared conv ring showed up in one cell out of two.
    // The recurrent state has to accumulate before crossing it is
    // visible, so a short sequence tests the attention half and quietly
    // passes the recurrent half.  Verified by re-running both negative
    // controls at this length, where each fails every hybrid cell.
    const int K = 4;
    std::vector<std::vector<int32_t>> prompts;
    {
        // Deterministic and distinct.  A different multiplier per
        // sequence, so no two share a prefix and their recurrent states
        // diverge from the first token rather than the tenth.
        const int len[K] = {41, 23, 34, 17};
        for (int k = 0; k < K; ++k) {
            std::vector<int32_t> p;
            uint32_t x = uint32_t(1 + k) * 2654435761u;
            for (int i = 0; i < len[k]; ++i) {
                x = x * 1664525u + 1013904223u;
                p.push_back(int32_t((x >> 9) % 120));
            }
            prompts.push_back(p);
        }
    }

    struct Cell { const char* name; mini::Arch arch; };
    std::vector<Cell> cells;
    cells.push_back({"dense", mini::dense()});
    cells.push_back({"moe",   mini::moe()});
    // Hybrid: DeltaNet layers carry the conversation in a recurrent
    // state, not a cache.  Batching it is a stronger claim than batching
    // attention -- a shared state would mix the conversations together
    // token by token and still read as English.
    cells.push_back({"hybrid", mini::hybrid()});

    for (auto& cell : cells) {
        for (Fmt fmt : {Fmt::BF16, Fmt::FP8_E4M3}) {
            const char* fname = (fmt == Fmt::BF16) ? "bf16" : "fp8";
            const fs::path dir = root / (std::string(cell.name) + "-" + fname);
            mini::write_model(dir, cell.arch);

            // ---- arm 1: one conversation at a time -------------------
            std::vector<std::vector<int32_t>> alone; alone.resize(size_t(K));
            {
                std::string err;
                Grimoire* e = grimoire_new();
                if (!e || !grimoire_load(*e, dir.string(), fmt, 256, err)) {
                    ++g_fail;
                    std::printf("    FAIL: %s %s load: %s\n",
                                cell.name, fname, err.c_str());
                    continue;
                }
                for (int k = 0; k < K; ++k) {
                    FinishReason r{};
                    grimoire_serve_generate(*e, prompts[size_t(k)], 12, -1,
                                            alone[size_t(k)], -1, {}, &r);
                }
                grimoire_delete(e);
            }

            // ---- arm 2: all of them in one batched pass --------------
            std::vector<std::vector<int32_t>> together; together.resize(size_t(K));
            long steps = 0, rows = 0;
            {
                std::string err;
                Grimoire* e = grimoire_new();
                if (!e || !grimoire_load(*e, dir.string(), fmt, 256, err)) {
                    ++g_fail;
                    std::printf("    FAIL: %s %s batch load: %s\n",
                                cell.name, fname, err.c_str());
                    continue;
                }
                const long s0 = b70::g_batch_decode_steps;
                const long r0 = b70::g_batch_decode_rows;
                const int n = grimoire_serve_generate_batch(
                    *e, prompts, 12, -1, together, -1);
                steps = b70::g_batch_decode_steps - s0;
                rows  = b70::g_batch_decode_rows  - r0;
                if (n != K) {
                    ++g_fail;
                    std::printf("    FAIL: %s %s batch answered %d of %d\n",
                                cell.name, fname, n, K);
                }
                grimoire_delete(e);
            }

            bool same = true;
            for (int k = 0; k < K; ++k)
                same = same && alone[size_t(k)] == together[size_t(k)];
            std::printf("%-6s %-5s %s\n", cell.name, fname,
                        same ? "identical" : "DIFFER");
            for (int k = 0; k < K; ++k)
                std::printf("    seq %d  prompt %zu  alone %-24s together %s\n",
                            k, prompts[size_t(k)].size(),
                            show(alone[size_t(k)]).c_str(),
                            show(together[size_t(k)]).c_str());
            CHECK(same,
                  "a sequence decoded in a batch answered differently from "
                  "the same sequence decoded alone, so per-row state is "
                  "crossing between conversations");

            // A batched path that quietly stepped each sequence on its own
            // would satisfy every line above.  Require that it batched:
            // more than one row per step, on average, over the steps it
            // took.  The sequences finish at different times, so the last
            // steps legitimately carry fewer rows -- the average is the
            // honest statement, not a per-step minimum.
            std::printf("    %ld batched steps carrying %ld rows (%.2f per step)\n",
                        steps, rows, steps ? double(rows) / double(steps) : 0.0);
            CHECK(steps > 0 && rows > steps,
                  "the batch never carried more than one row per step, so "
                  "this compared the sequential path with itself");

            // ---- and again with sequences that STOP AT DIFFERENT TIMES
            // Above, every sequence ran to the full budget, so the batch
            // was the same width on every step and the narrowing path --
            // dropping a finished row and stepping the rest -- never ran.
            // That is the path a real server is in almost all the time.
            //
            // The stop token is CHOSEN FROM WHAT THIS FIXTURE ACTUALLY
            // EMITS, not hardcoded.  A fixed token happened to suit the
            // dense and MoE cells and appeared nowhere in the hybrid
            // one, so that cell ran every sequence to the full budget and
            // the arm asserted a narrowing that could not happen --
            // a test failing on the test, which is worth avoiding by
            // construction rather than by picking a luckier constant.
            //
            // Pick a token that sequence 0 emits and some OTHER sequence
            // never does: then row 0 retires early and that row does not,
            // so the batch is guaranteed to narrow.
            {
                int kEos = -1;
                for (size_t i = 1; i < alone[0].size() && kEos < 0; ++i) {
                    const int cand = alone[0][i];
                    for (int k = 1; k < K; ++k) {
                        bool seen = false;
                        for (int32_t t : alone[size_t(k)]) if (t == cand) seen = true;
                        if (!seen) { kEos = cand; break; }
                    }
                }
                if (kEos < 0) {
                    std::printf("    ragged: skipped -- every sequence emits "
                                "every token this one does\n");
                    continue;
                }
                std::vector<std::vector<int32_t>> a2; a2.resize(size_t(K));
                std::vector<std::vector<int32_t>> b2; b2.resize(size_t(K));
                std::string err;
                Grimoire* e1 = grimoire_new();
                Grimoire* e2 = grimoire_new();
                if (!e1 || !e2 ||
                    !grimoire_load(*e1, dir.string(), fmt, 256, err) ||
                    !grimoire_load(*e2, dir.string(), fmt, 256, err)) {
                    ++g_fail;
                    std::printf("    FAIL: %s %s ragged load: %s\n",
                                cell.name, fname, err.c_str());
                } else {
                    for (int k = 0; k < K; ++k) {
                        FinishReason r{};
                        grimoire_serve_generate(*e1, prompts[size_t(k)], 12,
                                                kEos, a2[size_t(k)], -1, {}, &r);
                    }
                    const long s0 = b70::g_batch_decode_steps;
                    const long r0 = b70::g_batch_decode_rows;
                    grimoire_serve_generate_batch(*e2, prompts, 12, kEos, b2, -1);
                    const long st = b70::g_batch_decode_steps - s0;
                    const long rw = b70::g_batch_decode_rows  - r0;
                    bool ok2 = true;
                    for (int k = 0; k < K; ++k)
                        ok2 = ok2 && a2[size_t(k)] == b2[size_t(k)];
                    std::printf("    ragged (stop=%d): %s, %ld steps / %ld rows",
                                kEos, ok2 ? "identical" : "DIFFER", st, rw);
                    for (int k = 0; k < K; ++k)
                        std::printf("  [%zu]", b2[size_t(k)].size());
                    std::printf("\n");
                    CHECK(ok2,
                          "sequences that stop at different times answered "
                          "differently in a batch, so retiring a row disturbs "
                          "the rows that remain");
                    // Strictly between 1 and K rows per step on average:
                    // at K it never narrowed, at 1 it never batched.
                    CHECK(st > 0 && rw > st && rw < long(K) * st,
                          "the ragged batch did not actually narrow, so the "
                          "path a real server spends its time in is untested");
                }
                grimoire_delete(e1);
                grimoire_delete(e2);
            }
        }
    }

    if (!g_fail) fs::remove_all(root);
    else std::printf("\nfixtures kept in %s\n", root.c_str());
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
