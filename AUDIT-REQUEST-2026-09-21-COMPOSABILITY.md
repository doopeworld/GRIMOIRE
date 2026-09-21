# Audit request: what does NOT work, and what is NOT verified

Written after the 2026-09-21 external audit (9 defects, all fixed — see
`CLAUDE.md` rule 21 and commit `dd04198`) was itself fixed and pushed.
This is a **different, narrower ask**: not "was the fix correct" but
"what is still known-missing or known-unverified in GRIMOIRE, as of
right now." Point the next agent here first.

Repo: `doopeworld/GRIMOIRE`, branch `claude/new-session-pro26b`,
commit `dd04198` at time of writing.

---

## 1. Missing features — not bugs, just not built

These are refused BY NAME in the engine (`batch_unsupported_reason()`
in `src/grimoire.cpp`), not silent gaps — but they are real limits on
"does everything work everywhere."

| Gap | Detail |
| --- | --- |
| No batched decode for Muse, gemma-4, Qwen4-Exp | Only dense/MoE/hybrid can serve several sequences in one step. Each of the other three has its own residual graph and would need the same per-row attention pattern ported to it — mechanical, not designed, not started. |
| No batching under TP or PP | The two-card server (`tools/serve_pp2.sh`) is strictly one request at a time. A pipeline is already parallel over tokens, not sequences; nothing negotiates a batch width across ranks. |
| Batching and speculation (MTP/DFlash) are mutually exclusive | A loaded drafter turns batching off entirely. Nobody has measured which wins at any actual concurrency level. |
| Prefix cache (`GRIMOIRE_PREFIX_CACHE=1`) does not compose with batching | Consequence of this session's F3 fix (`CLAUDE.md` rule 21): the batchable scheduler path never calls `prefix_reuse()`/`restore_prefix_upto()` on admission and never snapshots on completion. A growing conversation is re-read in full every turn once batching is engaged, full stop. |

## 2. Never touched real hardware — CPU-only verification so far

Every device gate in this repo has only ever run on an Intel Xeon CPU
via `GRIMOIRE_DEVICE_ANY=1` (`ext_intel_matrix` correctly refused per
rule 15, so it takes the plain-SYCL fallback path, not the XMX tile).
That means:

- **Every speed/timing number in every doc in this repo is
  unverified.** Rule 8 says never trust one without generated text;
  none of these have generated text on the actual card at all.
- The OCuLink Gen4 x4 link itself — never exercised.
- Every XMX tile / `joint_matrix` kernel — never executed, only
  compiled (AOT `bmg_g31` needs `ocloc`, not available in this
  container).
- `GRIMOIRE_SEQ_SLOTS` / `GRIMOIRE_MAX_BATCH` VRAM budget against a
  real checkpoint at real `--ctx` — sized in theory (N slots = N KV
  caches), never checked against actual VRAM.
- `tools/preflight_b70.sh`'s new `run_multigpu_gate()` (this session's
  F9 fix) — reuses `pp2run.sh`'s proven container flags **by
  reasoning**, never actually run against real GPUs. First real
  multi-rank Tower run is the actual test of this fix, not this
  sandbox.

## 3. Open, unresolved question from this session — needs follow-up

While re-verifying the 9 audit fixes (commit `dd04198`), two gates hit
the preflight's own 900-second per-gate timeout that had **completed
successfully within that same budget earlier in this same session, on
this same container** (`Xeon @ 2.80GHz`, 4 cores):

- `test_model_matrix`: got through 19 of 70 cells (dense complete,
  moe complete, hybrid 5/7) before being killed at 900s. Every
  completed cell reported `ok`. Earlier run this session:
  `ALL PASS (0 failed cells)`, all 56 cells (8 architectures × 7
  formats then), inside budget.
- `test_parallel_e2e`: got through dense-bf16 (single + PP2/PP3/PP4 +
  TP2/TP3/TP4, all `match`) and most of dense-fp8 before being killed
  at 900s. No mismatches, no crashes — just slow. This gate normally
  covers 6 architectures × 2 formats, ~50 matches total.

**No wrong output was observed in either gate** — every cell that
finished, finished correctly. But both gates are running roughly 3x+
slower per cell than earlier in the session, with no commit between
the fast run and the slow one *except* the 9 audit fixes (`dd04198`).

This needs to be run down, not dismissed:

1. Re-run both gates **uncontended** (nothing else competing for CPU)
   with **no timeout cap** — does it eventually finish clean, just
   slowly? If yes, this is environment noise (shared-host contention),
   not a code defect, and the finding is "preflight's 900s budget is
   too tight for a loaded container," not "the fixes broke something."
2. If it's genuinely slower even uncontended, `git bisect` across the
   9 fixes in `dd04198` — **F2 is the prime suspect**, since it
   directly touches the PP decode loop (`src/grimoire.cpp`, the
   rank-0-keeps-matching-peers'-step-count change) that
   `test_parallel_e2e` exercises most heavily. Confirm F2 didn't add
   extra synchronization overhead per token.
3. Either way, update `CLAUDE.md` / the relevant handoff doc with
   the actual finding — do not leave this unresolved silently (rule 14).

## 4. Not yet re-confirmed against the fully-patched tree

As of the last check before this doc was written, these gates had not
yet reported a result in the current full-suite re-run (background,
still in progress):

`test_gemma4_prefill`, `test_qwen4_exp_e2e`, `test_nvfp4_e2e`,
`test_prefix_reuse`, `test_batch_decode`, `test_scheduler`,
`test_pp_server`.

Check `git log` / the latest handoff doc for whether a follow-up
commit landed with their results before starting the audit — if not,
run them yourself first.

## What to actually check, concretely

- `src/grimoire.cpp`: `batch_unsupported_reason()` for the exact,
  current refusal list (section 1 above should match it verbatim —
  if it doesn't, the code changed and this doc is stale).
  build_b70.sh` and `tools/preflight_b70.sh` for how gates are wired.
- `CLAUDE.md` rule 21 for the shape of bug this project keeps finding:
  two features that share a call site, each gated separately, never
  gated interrupting each other. Assume the next real bug looks like
  that, not like a crash.
- Section 3 above before trusting ANY timing claim made about this
  commit or the next one — it is currently an open question, not a
  settled fact in either direction.
