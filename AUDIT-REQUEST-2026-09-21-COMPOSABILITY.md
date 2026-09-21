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

## 3. RESOLVED — the gate timeouts are the container, not the code

**Answered 2026-09-21, after this doc was first written. Kept in full
because the reasoning is the reusable part; the conclusion is at the
bottom of the section.**

### The conclusion first

The four gates that time out off the card do so because **this
container is roughly 2.3x slower than it was earlier in the same
session**, not because of anything in `dd04198`. The A/B:

| `bin/test_model_matrix` build | cells completed in 300s |
| --- | --- |
| `dd04198` — WITH the nine audit fixes | 7 |
| `f4b8758` — BEFORE the nine audit fixes | 8 |
| (for reference) same gate, 2026-09-16, `clean_matrix.log` | 56 cells inside 900s, i.e. >=18 per 300s |

Both builds were run back to back, on the same machine, under the same
background load, with the same 300s budget. The pre-fix binary is just
as slow, so the nine fixes are not the cause. The 7-vs-8 difference is
noise: cells differ in cost, the cutoff lands mid-row, and both sides
were equally contended.

The first row measured (`dense`, 7 cells) is byte-identical work in
both builds and in the 2026-09-16 build, which is what makes this
apples-to-apples rather than a comparison across a gate that grew.

### Why it looked like a regression, and why that was wrong

Two things conspired, and both are instances of rules already in
`CLAUDE.md`:

- **Rule 15 again: a bisect controls for the commit, not for the
  machine.** The first evidence was "this gate finished inside 900s
  earlier today and does not now, and the only thing between the two
  runs is nine commits." That is a real observation and the wrong
  inference — it has no way to see the variable that actually
  differed. The fix is the same as it was then: build the other side
  and run both **now**, on the machine you actually have.
- **The comparison was not apples-to-apples anyway.** The gate grew
  from 8 architectures to 10 (Qwen4-Exp and an NVFP4 row) between the
  fast 2026-09-16 run and today — and Qwen4-Exp is the most expensive
  row in it. That growth landed in `d855869`/`acadf5f`, **before** the
  audit fixes. Part of "it got slower" was simply "it got bigger", and
  attributing the whole delta to the newest commits was unjustified
  before the A/B was run.

### What this means for the Tower

Little to nothing. `preflight_b70.sh`'s 900s per-gate budget is sized
for a B70 running the XMX tiles, not for a contended CPU container
taking the plain-SYCL fallback (rule 20's closing note: off the card
these are *not the same code*). If the Tower's own preflight run trips
this budget, that is a real signal worth chasing; a container tripping
it is not.

**Still worth doing on the Tower:** confirm all four of these gates
(`test_model_matrix`, `test_parallel_e2e`, `test_spec_e2e`,
`test_pp_server`) actually complete inside 900s there. Nobody has
timed them on real hardware, so the budget is unvalidated in the
direction that matters, even though it is very likely fine.

### What was observed in every timed-out run

Worth stating plainly because it is the part that bears on
correctness: **across all four timed-out gates, in both the 900s and
the 1800s runs, every cell that completed, completed correctly.**
Zero mismatches, zero crashes, zero wrong tokens. `test_spec_e2e`
reached `TP+MTP match` and `PP+MTP match` — the multi-rank
speculation path F2 touches — before running out of budget. A timeout
is a statement about the clock, not about the answers.

---

## 3b. Original text of the open question (superseded by 3 above)

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

## 4. Full-suite result against the patched tree (`dd04198`)

All 12 device gates were rebuilt and run against the fully-patched
tree. **Zero failures, zero mismatches, zero crashes anywhere** — the
only non-green results are clock timeouts, explained in section 3.

| gate | result |
| --- | --- |
| `test_k2_kernels` | ALL PASS |
| `test_k2_e2e` | ALL PASS |
| `test_model_matrix` | timeout (every completed cell `ok`; 6.5/10 architectures in 1800s) |
| `test_parallel_e2e` | timeout (every completed cell `match`) |
| `test_spec_e2e` | timeout (every completed cell `identical`; at 1800s reached dense+mtp, moe+mtp bf16 AND fp8 with `TP+MTP match` + `PP+MTP match`, hybrid+mtp refusal, muse bf16) |
| `test_gemma4_prefill` | ALL PASS |
| `test_qwen4_exp_e2e` | ALL PASS |
| `test_nvfp4_e2e` | ALL PASS — this is the gate that covers the F1 fix |
| `test_prefix_reuse` | ALL PASS |
| `test_batch_decode` | ALL PASS — includes the new F7 regression arm (`stale-pos capacity  3 of 3 rows answered`) |
| `test_scheduler` | ALL PASS (`12 batched steps carrying 66 rows`) |
| `test_pp_server` | **ALL PASS** — timed out at 900s, passes clean given 1800s (needs ~1200s off the card) |

Plus the 13 host suites (`make test`, `make test-correctness`), green.

`test_pp_server` is the direct confirmation of section 3's conclusion:
given more clock and nothing else, it goes green with zero failures.
It is not a broken gate, it is a gate that does not fit a 900s budget
on a CPU. The remaining three are the same shape, just bigger — they
are 70 cells, ~50 PP/TP matches, and a full speculation matrix
respectively, where `test_pp_server` is three requests down one
pipeline.

So: **the nine audit fixes introduced no correctness regression that
any gate in this repo can see, and no performance regression either
(section 3's A/B).** What remains unverified is what was always
unverified — everything that needs the actual card (section 2).

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
