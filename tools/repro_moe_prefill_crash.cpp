// =====================================================================
//  repro_moe_prefill_crash.cpp -- a MoE prompt of 32+ tokens through the
//  BATCHED prefill faults intermittently inside a kernel.
//
//  NOT a gate.  A reproducer, kept because the bug is not fixed and the
//  next person to see a random SIGSEGV in an off-card MoE run should be
//  able to confirm it in one command instead of re-deriving it.
//
//  What it does: writes a miniature checkpoint, loads it, and answers
//  four identical requests at two projection formats.  Nothing else --
//  no batching, no sequence slots, no scheduler, no prefix cache.
//
//  WHAT IS ESTABLISHED (2026-09-17, off the card, Xeon @ 2.10GHz with
//  the conda-forge OpenCL CPU runtime):
//
//    * it needs the BATCHED prefill.  With GRIMOIRE_BATCHED_PREFILL_NOXMX
//      unset the engine runs the sequential fallback and 6 runs of 6 are
//      clean; with it set, 4 of 6 die.
//    * it needs a MoE model.  The dense fixture is clean at the same
//      prompt length; so is the hybrid one.
//    * it needs a PROMPT OF ABOUT 32 TOKENS OR MORE.  4, 8, 16, 17 and
//      24 are clean over three runs each; 32, 33 and 41 fault.
//    * it is NOT new.  The same reproducer built against acadf5f -- before
//      any of the prefix-cache, batched-decode or scheduler work -- faults
//      6 times out of 6.
//    * the routes are fine.  GRIMOIRE_MOE_ROUTE_CHECK=0 reports
//      `bad=0 duplicate=0` and the process still dies, so it is not an
//      expert index out of range.  (That flag also makes it deterministic,
//      because the readback it does serialises the queue.)
//    * the fault is INSIDE A KERNEL.  Under gdb the crashing thread is in
//      JIT-generated code with no symbols while the main thread waits in
//      sched_yield.  On a CPU device USM is host malloc, so an
//      out-of-bounds access usually returns garbage and occasionally hits
//      an unmapped page -- which is exactly the two symptoms seen: a
//      SIGSEGV, or "engine returned an invalid token" from an argmax over
//      a buffer that was never written.
//
//  WHAT IS NOT ESTABLISHED, and matters most: whether the B70 is
//  affected.  On the card `noxmx_gemm` is false and mm() uses the XMX
//  tile instead of launch_gemm_batched, so the failing configuration may
//  be the fallback only -- or may not.  An Ornith prompt is far longer
//  than 32 tokens, so if it is not the fallback, this is live on the
//  Tower.  Run this there, both ways, before trusting a long MoE prompt.
//
//  Usage:
//    repro_moe_prefill_crash [dense|moe|hybrid] [prompt_len] [n_tokens]
//
//    GRIMOIRE_BATCHED_PREFILL_NOXMX=1 GRIMOIRE_DEVICE_ANY=1 \
//      ./bin/repro_moe_prefill_crash moe 41 4
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

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string arch = argc > 1 ? argv[1] : "moe";
    const int plen = argc > 2 ? std::atoi(argv[2]) : 41;
    const int ntok = argc > 3 ? std::atoi(argv[3]) : 4;

    char tmpl[] = "/tmp/grimoire-repro-XXXXXX";
    if (!::mkdtemp(tmpl)) { std::printf("mkdtemp failed\n"); return 1; }
    const fs::path root = tmpl, dir = root / arch;
    mini::write_model(dir, arch == "moe"    ? mini::moe()
                         : arch == "hybrid" ? mini::hybrid()
                                            : mini::dense());

    // Deterministic prompt; the CONTENT does not matter, the LENGTH does.
    std::vector<int32_t> p;
    uint32_t x = 2654435761u;
    for (int i = 0; i < plen; ++i) {
        x = x * 1664525u + 1013904223u;
        p.push_back(int32_t((x >> 9) % 120));
    }

    for (Fmt fmt : {Fmt::BF16, Fmt::FP8_E4M3}) {
        std::string err;
        Grimoire* e = grimoire_new();
        if (!e || !grimoire_load(*e, dir.string(), fmt, 256, err)) {
            std::printf("load failed: %s\n", err.c_str());
            return 1;
        }
        for (int k = 0; k < 4; ++k) {
            std::vector<int32_t> out; FinishReason r{};
            grimoire_serve_generate(*e, p, ntok, -1, out, -1, {}, &r);
            std::printf("  %s fmt=%d request %d -> %zu tokens\n",
                        arch.c_str(), int(fmt), k, out.size());
        }
        grimoire_delete(e);
    }
    fs::remove_all(root);
    std::printf("OK -- no fault in this run (it is intermittent; repeat it)\n");
    return 0;
}
