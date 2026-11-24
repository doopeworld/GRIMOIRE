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
  link, and **all 15 device gates and all 17 host suites pass off the
  card** (CPU OpenCL device; table in section 3).
- **"Finished": no.** It has never run on a B70. No kernel has run on XMX
  hardware, no AOT image or bridge build has been tested, OCuLink has not
  been exercised, and there is no speed number anywhere. The Tower run
  is the first real test, not a formality. Everything that CAN be done
  off the card is done (section 4b): every finding below that code can
  fix is fixed, and the docs match the code again.
- **Everything is on `main`** (fast-forwarded 2026-09-22 from
  `claude/grimoire-audit-testing-qa0v98` = `bf961c3` + the fixes below).
  On the Tower: `git checkout main && git pull`.

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
| test_model_matrix | no | ALL PASS (0 failed cells), 10 architectures x 7 formats, 49 min |
| test_parallel_e2e | no | ALL PASS, 50 PP/TP matches, 0 failures, 55 min |
| test_spec_e2e | no | ALL PASS (0 failures), 58 min |
| test_pp_server | no | ALL PASS, 16 min |

**All 15 device gates and all 17 host suites are green at `bf961c3`.**
That includes the four that timed out in the 2026-09-21 session. Given
up to 90 min each, they all finish. (The earlier timeouts were the
clock, as `AUDIT-REQUEST-2026-09-21-COMPOSABILITY.md` section 3
concluded.)

Gates re-run on this branch with the fixes below: test_batch_spec,
test_batch_prefix, test_prefix_reuse, test_scheduler, all ALL PASS.
`test_spec_e2e` on the patched build (it loads MTP and DFlash drafters
single-process, under TP and under PP, so it goes through the changed
`init_draft_slots()`): **ALL PASS, and identical to the tip run cell
for cell** on a clean solo run. An earlier concurrent run lost one cell
to the OOM killer, described below.

One trap while doing this, worth knowing before running gates in
parallel off the card: with four gates at once, the container's memory
cgroup OOM-killed rank 0 of a `test_spec_e2e` TP cell (`dmesg`: "Memory
cgroup out of memory: Killed process ... test_spec_e2e", 1.8 GB
resident). The gate reports that as `TP+DFlash FAILED (rank0 -9, rank1
4)`, which looks exactly like a code failure. `-9` means SIGKILL: check
`dmesg` before debugging a multi-rank cell that died that way.

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
and re-verified as stated. None of these came from a failing gate: every
gate passes. They came from reading the diff, the launchers and the
coverage.

Status after the second round (section 4b):

| finding | status |
| --- | --- |
| F1 stale docs | **fixed**: DAY-ONE, CLAUDE.md (plus rule 22), and banners on the three superseded docs |
| F2 drafter VRAM per slot | banner added; the size itself is a design choice (bf16 taps would halve it), left for after the Tower measures real VRAM |
| F3 half-built allocations | **fixed** |
| F4 PP-serve worker script | **fixed**; now also serves TP |
| F5 launcher env | **fixed**, plus `GRIMOIRE_DECODE_GRAPH` |
| F6 TP GEMM allocation + per-row exchange | **fixed**: pooled scratch, one exchange per projection |
| F7 spec depth vs batch width | open: needs a measurement, not a guess |
| F8 batched MTP never accepted | **fixed**: forced-acceptance MTP rows + MoE/Muse rows, negative control |
| F9a TP on the grouped MoE path | **fixed** (guarded) |
| F9b TP ranks could disagree on prefix reuse | **fixed** (`tp_agree`), negative control |
| F9c batch driver under PP/TP | **withdrawn**: called by every rank with the same inputs (SPMD) it stays in lockstep; only the server needs the scheduler, and uses it |
| F9d no TP launcher | **fixed**: `tools/serve_tp2.sh` |
| F10 XMX vs GEMV on the card | expectation, nothing to fix off the card |

**F1. The docs describe the code before the 9 commits. (docs; fixed in
round two, see 4b)** `CLAUDE.md` (the "does NOT yet compose" and "Concurrency does NOT apply
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

**F6. TP batched decode will be slow as written. (performance; fixed in
round two)**
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

## 4b. Second round: everything that could be done off the card

Commits on `claude/grimoire-audit-testing-qa0v98` after the first
round, each verified as its message says:

- `dbe5387` **TP serving**: `tp_agree()` makes TP ranks agree on the
  prefix-snapshot save outcome and on the admission reuse length (F9b).
  `gemm_tp()` uses pooled scratch and one exchange for all rows (F6). TP
  never takes the grouped MoE path (F9a). Worker logs name TP vs PP.
  `test_batch_parallel` gained a TP2 + prefix-cache arm over growing
  conversations, the first gate to reach the TP resume or the TP commit.
  **Negative control**, one rank's save forced to fail: the old code
  desynchronises (`TP projection all-gather failed`), the new code
  completes all 9 requests and the gate flags the lost resume.
- `0bde126` **batched MTP acceptance**: `forced_mtp()` builds a head
  whose answer is known (the target's own prediction), accepted 34x on
  dense and 35x on MoE. New MoE and Muse rows (Muse accepts 3 by chance).
  **Negative control**, each row's draft state zeroed: the old gate
  passes it, the new one fails. The hybrid and Muse fixtures never repeat
  a token back to back, so a repeat-guessing head cannot be accepted
  there; hybrid acceptance stays covered by the forced-DFlash row.
- `cae4f07` **`serve_tp2.sh`**, the TP server launcher, and the worker
  script in PP or TP mode (tested with a stand-in in both).
- `9587389` merged the two doc-only branches, and `b1ed424` marked the
  2026-09-04 decode-graph finding as wired (the server builds the graph
  under `GRIMOIRE_DECODE_GRAPH=1`; still unmeasured).

The two remaining unmerged branches, decided on evidence:

- `tp-weight-sharding` (6c959e0): **superseded**. 318 of its 340
  non-trivial added lines are already in the tree, via `c771464`
  (2026-09-11); the rest were reworked since. Nothing to merge.
- `research/paiton-b70-optimizations` (f02e9c4): **held on purpose**. It
  calls itself "separate from the Tower-ready branch". It is opt-in
  (`GRIMOIRE_SPEC_REPLAY=1`), its own two-stage tests never ran, and it
  conflicts with the new batched-speculation state capture in
  `grimoire.cpp` and `preflight_b70.sh`. Merge it after the Tower has a
  baseline, so its effect can be measured against one.

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
git checkout main && git pull
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
   `GRIMOIRE_SEQ_SLOTS`. Then `tools/serve_tp2.sh`, the same way, plus
   `GRIMOIRE_PREFIX_CACHE=1` (TP keeps the cache; PP does not).
5. Only then a drafter with slots. Watch the new `drafter:
   per-sequence caches ... GiB` line against free VRAM (F2).
6. The first measurements, none of which exist yet: tok/s at 1/2/4/8
   concurrent; batching vs drafter vs both (F7); plain decode with
   `GRIMOIRE_DECODE_GRAPH=1` vs without; PP vs TP on OCuLink; how many
   slots fit at the `--ctx` you actually use.

## Final confirmation, on `main` (2026-09-23)

The full suite was re-run on the merged tree (engine as of `dbe5387`, the
last code change): **all 15 device gates and all 17 host suites pass**.
`test_batch_parallel` passed 30 of 30 arms (PP2, PP3, TP2, TP3 and
TP2+cache, over 6 architecture/format cases), `test_model_matrix` had 0
failed cells, and `test_parallel_e2e`, `test_spec_e2e` and
`test_pp_server` were ALL PASS.  GitHub CI (run 35788810884, all 7
shards) is green on the same code.  CPU device only; the B70 run is next.
