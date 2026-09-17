// =====================================================================
//  test_scheduler.cpp -- requests that OVERLAP IN TIME must answer
//  exactly what the same requests answer one after another.
//
//  The server no longer holds a mutex for the length of a request.  A
//  resident scheduler owns the engine, every request gets its own
//  sequence slot, and a decode step carries one token for each request
//  in flight.  That is the concurrency that matters for agentic work,
//  and it is also the change with the most ways to be quietly wrong:
//  two requests sharing a slot, a row reading a neighbour's position, a
//  cancelled request leaving its slot marked busy.  Every one of those
//  produces fluent text from the wrong conversation.
//
//  So this gate runs the requests FOR REAL, from several threads at
//  once, released together so they are genuinely simultaneous -- and
//  requires each to come back with exactly what it gets alone.
//
//  It also covers the arm that has to keep working: a model the engine
//  CANNOT batch (linear attention keeps one recurrent state per engine,
//  not per sequence).  There the scheduler must fall back to one request
//  at a time and still answer correctly under concurrent callers, which
//  is the behaviour the server had before any of this.
//
//  Run: GRIMOIRE_DEVICE_ANY=1 ./bin/test_scheduler
// =====================================================================
#include "b70/grimoire_api.hpp"
#include "b70/formats.hpp"
#include "mini_model.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace b70;

namespace b70 {
extern long g_batch_decode_steps;
extern long g_batch_decode_rows;
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

// Release every thread at the same instant.  Without this the first
// request would usually finish before the last one was submitted, the
// scheduler would never hold more than one, and the test would pass
// while proving nothing about concurrency.
struct Gate {
    std::mutex m; std::condition_variable cv; bool open = false;
    void wait() { std::unique_lock<std::mutex> l(m); cv.wait(l, [&]{ return open; }); }
    void release() { { std::lock_guard<std::mutex> l(m); open = true; } cv.notify_all(); }
};

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== scheduler ===\n");
    std::printf("requests issued from several threads at once must answer\n"
                "exactly what they answer one at a time.\n\n");

    ::setenv("GRIMOIRE_SEQ_SLOTS", "8", 1);
    ::setenv("GRIMOIRE_BATCHED_PREFILL_NOXMX", "1", 1);

    char tmpl[] = "/tmp/grimoire-sched-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl;

    const std::vector<std::vector<int32_t>> prompts = {
        {7, 11, 3},
        {90, 1, 64, 23, 5, 18},
        {42, 42},
        {3, 9, 27, 81, 19},
        {5},
        {64, 64, 64, 12},
    };
    const int K = int(prompts.size());

    struct Cell { const char* name; mini::Arch arch; bool expect_batched; };
    std::vector<Cell> cells;
    cells.push_back({"dense",  mini::dense(),  true});
    cells.push_back({"moe",    mini::moe(),    true});
    // Linear attention cannot batch: one recurrent state per ENGINE.  The
    // scheduler must notice and serve one at a time -- and concurrent
    // callers must still get right answers, because that is the path
    // every hybrid checkpoint takes.
    cells.push_back({"hybrid", mini::hybrid(), false});

    for (auto& cell : cells) {
        const fs::path dir = root / cell.name;
        mini::write_model(dir, cell.arch);

        // ---- one at a time ------------------------------------------
        std::vector<std::vector<int32_t>> alone; alone.resize(size_t(K));
        {
            std::string err;
            Grimoire* e = grimoire_new();
            if (!e || !grimoire_load(*e, dir.string(), Fmt::BF16, 256, err)) {
                ++g_fail;
                std::printf("    FAIL: %s load: %s\n", cell.name, err.c_str());
                continue;
            }
            for (int k = 0; k < K; ++k) {
                FinishReason r{};
                grimoire_serve_generate(*e, prompts[size_t(k)], 6, -1,
                                        alone[size_t(k)], -1, {}, &r);
            }
            grimoire_delete(e);
        }

        // ---- all at once, through the scheduler ---------------------
        std::vector<std::vector<int32_t>> conc; conc.resize(size_t(K));
        std::vector<std::string> errs; errs.resize(size_t(K));
        long steps = 0, rows = 0;
        int width = 0;
        {
            std::string err;
            Grimoire* e = grimoire_new();
            if (!e || !grimoire_load(*e, dir.string(), Fmt::BF16, 256, err)) {
                ++g_fail;
                std::printf("    FAIL: %s concurrent load: %s\n",
                            cell.name, err.c_str());
                continue;
            }
            GrimoireScheduler* sc = grimoire_scheduler_new(*e, 8);
            width = grimoire_scheduler_width(*sc);
            const long s0 = b70::g_batch_decode_steps;
            const long r0 = b70::g_batch_decode_rows;
            Gate gate;
            std::vector<std::thread> th;
            for (int k = 0; k < K; ++k)
                th.emplace_back([&, k] {
                    gate.wait();
                    try {
                        FinishReason r{};
                        grimoire_scheduler_generate(*sc, prompts[size_t(k)], 6,
                                                    -1, -1, conc[size_t(k)],
                                                    {}, &r);
                    } catch (const std::exception& ex) {
                        errs[size_t(k)] = ex.what();
                    }
                });
            gate.release();
            for (auto& t : th) t.join();
            steps = b70::g_batch_decode_steps - s0;
            rows  = b70::g_batch_decode_rows  - r0;
            grimoire_scheduler_delete(sc);
            grimoire_delete(e);
        }

        bool same = true;
        for (int k = 0; k < K; ++k) {
            if (!errs[size_t(k)].empty()) {
                ++g_fail;
                std::printf("    FAIL: %s request %d threw: %s\n",
                            cell.name, k, errs[size_t(k)].c_str());
                same = false;
            }
            same = same && alone[size_t(k)] == conc[size_t(k)];
        }
        std::printf("%-7s width %d  %s\n", cell.name, width,
                    same ? "identical" : "DIFFER");
        for (int k = 0; k < K; ++k)
            std::printf("    req %d  prompt %zu  alone %-24s concurrent %s\n",
                        k, prompts[size_t(k)].size(),
                        show(alone[size_t(k)]).c_str(),
                        show(conc[size_t(k)]).c_str());
        CHECK(same,
              "a request answered differently when it overlapped with "
              "others, so concurrent requests are sharing state");

        if (cell.expect_batched) {
            std::printf("    %ld batched steps carrying %ld rows (%.2f per step)\n",
                        steps, rows, steps ? double(rows) / double(steps) : 0.0);
            // The whole claim is that overlapping requests are STEPPED
            // TOGETHER.  A scheduler that queued them would answer
            // identically and prove nothing, so read the counter.
            CHECK(steps > 0 && rows > steps,
                  "the scheduler never carried more than one row per step, "
                  "so overlapping requests were still being queued");
            CHECK(width > 1, "the scheduler refused to batch a model it can batch");
        } else {
            std::printf("    not batchable: %ld batched steps (expected 0)\n", steps);
            CHECK(width == 1,
                  "the scheduler claimed a batch width for a model whose "
                  "recurrent state is per engine, not per sequence");
            CHECK(steps == 0,
                  "a model that cannot be batched took batched decode steps");
        }
    }

    if (!g_fail) fs::remove_all(root);
    else std::printf("\nfixtures kept in %s\n", root.c_str());
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
