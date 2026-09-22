# Audit, 2026-09-22: the branch state, the nine new commits, and Tower readiness

Asked: *audit the "everything is committed and pushed" report
(`codex/composable-serving` -> `bf961c3`, `claude/new-session-pro26b` ->
`fd41d9e`), then say whether GRIMOIRE is finished and ready to test on the
Tower and the B70s.*

## Verdict

- **The report is accurate.** Both remote branches sit exactly on
  `bf961c3` and `fd41d9e`. They are not two separate lines of work:
  `codex/composable-serving` already contains `claude/new-session-pro26b`
  plus 9 newer commits. **So one branch holds everything.**
- **Ready to START testing on the Tower: yes.** At `bf961c3` every source
  compiles cleanly (0 warnings), `bin/grimoire` and `bin/grimoire-server`
  link, and every off-card gate that finished passed (table below).
- **"Finished": no.** It has never run on a B70. No kernel has run on XMX
  hardware, no AOT image or bridge build has been tested, OCuLink has not
  been exercised, and there is no speed number anywhere. The docs also
  describe the code as it was *before* the 9 newest commits (finding F1).
  The Tower run is the first real test, not a formality.
- **Use this branch on the Tower** (`claude/grimoire-audit-testing-qa0v98`
  = `bf961c3` + the fixes below), or `codex/composable-serving` if you
  want the fixes left out. **Do not use `main`.** It is at `dcc549c`
  (2026-08-29), 229 commits behind. `DAY-ONE.md` only says `git pull`,
  which updates whichever branch the Tower checkout is on.

## 1. The report in the screenshot

| claim | checked | result |
| --- | --- | --- |
| `codex/composable-serving` -> `bf961c3` | `git log -1 origin/codex/composable-serving` | true |
| `claude/new-session-pro26b` -> `fd41d9e` | `git log -1 origin/claude/new-session-pro26b` | true |
| "all pushed" | remote tips match | true (the other session's local disk can't be checked from here) |
| `bf961c3` is "my batch-budget fix" | read the diff | true, and correct (section 3) |

Context the report left out:

- `fd41d9e` is an ancestor of `bf961c3`. The PR between them (#1,
  `codex/composable-serving` -> `claude/new-session-pro26b`) is still a
  **draft**. Its description ("the four requested serving combinations
  remain unfinished ... no engine fixes are included") was written for
  its first commit and is now wrong: the 9 commits implement all four.
- Nothing has been merged to `main` since 2026-08-29.
- Four branches each carry one commit that is **not** on the testing
  branch. None of them block Tower testing:
  `tp-weight-sharding` (6c959e0, arbitrary TP/PP weight sharding, +553
  lines of `grimoire.cpp`, never merged),
  `research/paiton-b70-optimizations` (f02e9c4, opt-in compact SYCL
  speculative replay), `vllm-dflash-build-reference` (a doc), and
  `qwen-container-recipes` (a doc: "the server never builds the decode
  graph").
- CI (`.github/workflows/composable-serving-cpu.yml`) is green on `bf961c3`
  across all 7 shards (run 35746755557). It only runs on PRs targeting
  `claude/new-session-pro26b`, and it runs **6 of the 15** device gates:
  batch_spec, batch_prefix, batch_decode, batch_parallel, scheduler and
  prefix_reuse. The other 9 had not been run against the 9 new commits
  until this audit.

## 2. What the 9 new commits are

They close all four "missing features" rows in
`AUDIT-REQUEST-2026-09-21-COMPOSABILITY.md` section 1:

| was refused | now |
| --- | --- |
| prefix cache does not compose with batching | `admit_sequence()` resumes a cached conversation into an idle slot (live slots excluded), and `cache_sequence()` snapshots on retire |
| no batched decode for Muse, gemma-4, Qwen4-Exp | `prefill_gemma4` -> `prefill_sandwich` (Muse + gemma-4 with per-row position/slot). Qwen4-Exp's QSA key caches, PLE history and `q4_tok` are now allocated per slot |
| no batching under TP or PP | rank 0 drives the workers with control messages: kind 2 = admit, 3 = batch step, 4 = TP prefix commit. `gemm_tp()` does the row-batched sharded projections, and the MoE expert remap runs under TP |
| batching and speculation are exclusive | `decode_spec_batch()` verifies per-sequence MTP/DFlash drafts in one target batch, single process only. Distributed + drafter is still refused |

`bf961c3` is the per-row budget fix: `generation_budget()` was computed
per row and then thrown away. The loop capped every row at `n_predict`
and relied on `pos >= max_seq`, which fires one token too late.

`batch_unsupported_reason()` now refuses only two things: TP/PP together
with a drafter, and a single sequence slot. **The default is still
`GRIMOIRE_SEQ_SLOTS=1`, so a plain launch behaves as before (one request at
a time).**

## 3. What was verified in this session (off the card)

Toolchain: icpx 2026.1.1 + OpenCL CPU device, exactly as the CI workflow
and `TOOLCHAIN-IN-A-CONTAINER.md` describe. CPU only, correctness only,
and **no number here says anything about speed** (rule 8).

- All 15 engine sources compile at `bf961c3` with the CI flags plus
  `-Wmisleading-indentation`: **0 warnings, 0 errors.**
- `bin/grimoire`, `bin/grimoire-server` and all 15 gate binaries link.
- `make test` (15 suites) and `make test-correctness` (2): all 17 host
  binaries pass. The docs still say "13 host suites".

Device gates at `bf961c3` (unpatched tip):

| gate | in CI? | result here |
| --- | --- | --- |
| test_k2_kernels | no | ALL PASS |
| test_k2_e2e | no | ALL PASS |
| test_gemma4_prefill | no | ALL PASS |
| test_qwen4_exp_e2e | no | ALL PASS |
| test_nvfp4_e2e | no | ALL PASS |
| test_batch_spec | yes | ALL PASS (forced-DFlash accepted: dense 21, hybrid 9) |
| test_batch_prefix | yes | ALL PASS |
| test_prefix_reuse | yes | ALL PASS |
| test_scheduler | yes | ALL PASS |
| test_batch_decode | yes | ALL PASS |
| test_batch_parallel | yes | green in CI, all 6 shards (not re-run here) |
| test_model_matrix | no | IN PROGRESS: every completed cell `ok` so far |
| test_parallel_e2e | no | IN PROGRESS: every completed cell `match` so far |
| test_spec_e2e | no | IN PROGRESS: every completed cell `identical` / `match` so far |
| test_pp_server | no | queued behind test_spec_e2e |

The last four are slow off the card: the repo's own notes put them at
15-30+ min each on a 4-core container, and 4 ran at once here. This
table is updated when they finish.

Gates re-run on this branch with the fixes below: test_batch_spec,
test_batch_prefix, test_prefix_reuse, test_scheduler, all ALL PASS.

Reviewed by reading: the full 1,729-line `grimoire.cpp` diff
`fd41d9e..bf961c3`, plus the launch scripts, the preflight and the CI
workflow. Specific things checked and found **correct**:

- the budget fix, including the draft-depth bound `remaining - 1` it
  feeds into `decode_spec_batch()`; the scheduler path already used
  `generation_budget()`
- `cache_sequence()` refuses to label a snapshot with tokens the engine
  never processed (a stop or cancel partway through an accepted block
  gives `position > processed.size()`, and the slot is invalidated)
- `save_prefix()` writes to the live slot, so snapshot and KV stay together
- per-slot drafter state: `clear_seq_slot()` zeroes the bound slot's MTP
  KV and `dflash2.context_pos`, and `bind_seq_slot()` saves/loads it
- the per-slot drafter buffer sizes match the build-time allocations
  byte for byte (an undersized copy would be silent on CPU and an
  out-of-bounds write on the card)
- rule 1 in `gemm_tp()`: `has_i4()` is tested before the (possibly freed)
  MXFP4 payload is touched
- TP MoE: non-owned experts become `-1`, and both batched MoE kernels skip
  them (`grow < 0` in gate_up, `e < 0` in down), so this is safe on the
  card, not just masked by CPU host memory
- prefix cache under PP is still refused outright, so the TP-only kind-4
  commit is not a PP hole
- Qwen4-Exp under TP/PP or with a drafter is refused at load by
  `unsupported_reason()`, so its PP/TP-less batched path cannot be reached
  there

## 4. Findings

Ranked by what they cost on the Tower. "Fixed" means fixed on this branch
and re-verified as stated. Nothing here was found by a failing gate. Every
gate that ran passed.

**F1. The docs describe the code before the 9 commits. (docs, not fixed
beyond a pointer)** `CLAUDE.md` (the "does NOT yet compose" and "Concurrency does NOT apply
there" bullets),
`HANDOFF-2026-09-21-DAY-ONE-READY.md` (lines 28, 172-179), `DAY-ONE.md`
(286, 336, 472-478), `CONCURRENCY-2026-09-17.md` (81-86, 165, 182) and
`AUDIT-REQUEST-2026-09-21-COMPOSABILITY.md` section 1 all say batching is
off under TP/PP, off with a drafter, off for Muse/gemma-4/Qwen4-Exp, and
does not compose with the prefix cache. None of that is true at
`bf961c3`. The gate count ("11 device gates") is also stale: there are
15. The next assistant to read `CLAUDE.md` will plan against the old
behaviour. This branch adds a pointer at the top of `CLAUDE.md`. The
per-file rewrite is left for whoever owns those docs.

**F2. Per-slot drafter caches are big, and the load-time VRAM figure
leaves them out. (sizing; banner added)** With a drafter loaded and
`GRIMOIRE_SEQ_SLOTS=N`, `init_draft_slots()` (called from `build()`'s
closing `reset()`) allocates N-1 extra copies of the drafter KV and, for
DFlash, of the **fp32 tap buffer: `max_seq * taps * hidden * 4` bytes per
slot**. For Ornith with the z-lab drafter (8 taps x 2048) that is
**64 KiB per token of `--ctx`, per slot**: 0.5 GiB at 8K, 2 GiB at 32K. At
`GRIMOIRE_SEQ_SLOTS=8` and 32K that is about 14 GiB on top of the model,
roughly 6x the target model's own KV per slot, and none of it goes
through `acct()`. So the "GiB resident" line never shows it. **Fixed:**
the engine now prints `drafter: per-sequence caches for N slots, X GiB
beyond the load-time budget`. **Not fixed (design):** storing taps in
bf16 would halve it, and allocating only for live slots would bound it.

**F3. A failed allocation left half-built state that the next call writes
through. (defensive, fixed)** `init_draft_slots()` marked itself done
(`draft_slots` non-empty) before allocating. A throw partway left later
slots with null cache pointers, which `bind_seq_slot()` then installed as
the live drafter caches. That is a device write through null, a
DEVICE_LOST on the card. `save_prefix()` (older code) had the same shape:
a failed allocation left `layers` non-empty with null entries, and the
next save wrote through them. Realistic reachability is low: the first
drafter allocation happens at load, where a throw ends the process. The
fix is still all-or-nothing in both functions. Verified by compile and
the four gates above. Not verified by forcing an allocation failure.

**F4. `serve_pp2_worker.sh` reported sed's exit status and signalled sed.
(launcher, fixed, tested with a stand-in)** `cmd | sed &` makes `$!` the
PID of sed. So a crashed front end exited 0 (the same bug class as the
`b70run.sh` one in rule 21), and "take the worker down" killed rank 1's
log filter instead of rank 1. There was also no trap: `docker stop` sends
TERM to this bash script, which died and orphaned both servers. They are
then SIGKILLed when the container's PID 1 exits. **Fixed** with process
substitution plus a TERM/INT trap that forwards to both ranks and waits
for them. Tested with a stand-in server. Old script, rank 0 exiting 3:
`bash -x` shows `R0=0`, then `kill` hits sed and `wait` blocks on the job
rank 1 is still in. A real rank 1 would notice the closed socket, and the
container would exit 0. Old script, TERM: the script dies and 2 servers
are orphaned. New script: exits 3 on a rank-0 crash, both ranks drain on
TERM, 0 left behind. Note: `grimoire-server` itself has no
SIGTERM handler, so TERM still ends it abruptly, exactly as `serve.sh`
always has.

**F5. No launcher could turn the new serving features on. (launcher,
fixed)** `serve.sh` and `serve_pp2.sh` pass a fixed `-e` list, so
`GRIMOIRE_SEQ_SLOTS`, `GRIMOIRE_MAX_BATCH`, `GRIMOIRE_PREFIX_CACHE`,
`GRIMOIRE_MTP`, `GRIMOIRE_DFLASH_MODEL` and similar never reached the
container. **Fixed** with bare `-e NAME` pass-through: it copies the host
value only when it is exported, so a default launch is unchanged. Also
corrected `serve_pp2.sh`'s closing message, which quoted a refusal the
engine no longer prints.

**F6. TP batched decode will be slow as written. (performance, not fixed)**
`gemm_tp()` calls `malloc_device` + `free` for two buffers, a full
`wait_and_throw`, and one socket all-gather per row, for every sharded
projection of every layer, every step. Treat the first TP-batching
number on the card as a measurement of that, not of TP batching.

**F7. Batched speculation depth shrinks as concurrency grows. (performance,
unmeasured)** `max_depth = min(requested, 16/n - 1)`: 3 rows allow depth
4, and 8 rows allow depth 1. DFlash still runs its whole 16-token block
per row to get those 1-2 verified tokens. At high concurrency, batching
plus DFlash is likely slower than batching alone. Measure it before
turning both on for Ornith.

**F8. Batched MTP has never accepted a draft in any gate. (coverage)**
In `test_batch_spec`, `dense-mtp` and `hybrid-mtp` both report
`accepted=0`, so the MTP branch of `decode_spec_batch()` that commits an
accepted block (`mtp_warm` over `accepted > 1`, state restored from a
step past the first) never runs. The forced-DFlash rows do accept
(dense 21, hybrid 9), so the DFlash and recurrent-restore paths are
covered. Batched speculation is gated on dense and hybrid only: not MoE,
not Muse (its fp16 drafter-cache path), not gemma-4. `test_spec_e2e`'s
MTP rows also show 0% acceptance, which predates this work.

**F9. Latent, unreachable today.** (a) The TP MoE expert remap exists only
in the plain `M < 32` branch. The grouped `M >= 32` path has none. TP
never batches 32+ rows today, but enabling TP prompt prefill would reach
it. (b) TP admission assumes every rank's prefix cache is identical. A
`save_prefix()` that fails on one rank only would make ranks prefill
different lengths and hang: the same class the PP refusal comment in
`prefix_cache_unusable_reason()` describes. (c)
`grimoire_serve_generate_batch()` under PP/TP sends no control messages
(`serving_control` is set only by the scheduler), so it would hang. Only
tests call it. (d) TP serving is wired in the engine (TP workers run
`grimoire_pp_worker_loop`), but there is no `serve_tp2.sh`. Only
`test_batch_parallel` exercises it.

**F10. On the card, the batched == serial gates compare different
arithmetic for the first time. (expectation)** `GRIMOIRE_BATCHED_PREFILL_NOXMX`
is a no-op on a GPU (`device_can_matrix()`), so on the B70
`test_batch_decode`, `test_scheduler`, `test_batch_*` and
`test_gemma4_prefill` compare XMX-tile batched rows against GEMV decode.
Off the card, both sides used plain SYCL. They are exactness claims and
should hold. If one fails on the Tower, first check whether the diverging
step was a near-tie in the logits, then look for a logic bug.

## 5. What only the Tower can settle

Unchanged from the day-one handoff, plus what the 9 commits added:

- the build itself: `my-vllm-xpu:latest`, the vllm-xpu-kernels checkout,
  `ocloc`/AOT `bmg_g31`, and the bridges (rule 3)
- every XMX / `joint_matrix` tile and every bridge path, none of which
  has ever executed
- `run_multigpu_gate` (full `/dev/dri`, `ZE_AFFINITY_MASK=0,1`), which
  has never run against real GPUs and now also carries
  `test_batch_parallel`
- the 900 s per-gate budget, which is untimed on real hardware
- OCuLink, and the PP/TP control-message traffic over it
- VRAM: `GRIMOIRE_SEQ_SLOTS` x (target KV + recurrent state + drafter
  copies, F2) at real `--ctx`
- every speed number, without exception

## 6. Day one, in order

```bash
ssh root@192.168.8.225
cd /mnt/storage/isos/grimoire-fuse
git fetch origin
git checkout claude/grimoire-audit-testing-qa0v98   # NOT main
git log -1 --oneline                                  # confirm the tip
tools/preflight_b70.sh /models/<model-dir>            # read the generated text
```

Then, one step at a time, reading the banner each time (rule 15):

1. `tools/serve.sh <model>`, one card, defaults: must behave as before.
2. `GRIMOIRE_SEQ_SLOTS=4 GRIMOIRE_MAX_BATCH=4 tools/serve.sh <model>`
   with 2-4 concurrent clients. The banner must say
   `batching up to 4 requests per step`.
3. Add `GRIMOIRE_PREFIX_CACHE=1`: multi-turn resume.
4. `tools/serve_pp2.sh`, first without slots, then with
   `GRIMOIRE_SEQ_SLOTS`.
5. Only then a drafter with slots. Watch the new `drafter:
   per-sequence caches ... GiB` line against free VRAM (F2).
