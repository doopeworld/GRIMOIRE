# Handoff: everything off-card is done. Day one starts at the Tower.

> **Superseded 2026-09-22 by `AUDIT-2026-09-22-TOWER-READINESS.md`.**
> Nine later commits made batching compose with the prefix cache, TP, PP,
> MTP/DFlash (one card) and Muse/gemma-4/Qwen4-Exp, so this file's "what is
> refused" section and its "11 device gates" are out of date (there are 15).
> The Tower should run branch `claude/grimoire-audit-testing-qa0v98`, not
> `main`.  The rest of this file is still an accurate history.

Read this first if you are picking up GRIMOIRE now. It supersedes
`CONCURRENCY-2026-09-17.md` as the newest status doc (that file is still
correct, just no longer current — read it for the full concurrency design
rationale, this file for what shipped after it and what to do next).

**UPDATE, same day: this handoff was audited, and the audit was right.**
An external review of exactly the commit this file describes found NINE
real defects, five P1 — correctness bugs in code that was each
individually gated and green. All nine are fixed, independently
re-verified (against the audit's own reproductions where given, and
against the upstream compressed-tensors source directly for NVFP4, not
against this repo's own fixture), and every device gate + host suite
rebuilt and rerun clean afterward. **Read `CLAUDE.md` rule 21** — it is
the fuller account, including the pattern behind all nine (features
tested separately, never tested interrupting each other) and is worth
reading before extending any of this further. The short version of what
changed, so this file's earlier claims below are read correctly:

- NVFP4's dequant formula was inverted (multiplied by the global scale,
  where the format's own convention divides) — fixed, verified against
  the upstream source line by line, not against this repo's own twin.
- Two concurrently-admitted requests sharing a prompt could be silently
  routed onto the same physical KV slot, or rewind a live request's own
  recurrent state — fixed by giving the scheduler a way to refuse the
  cache's exact-match rebind, at the cost of: **`GRIMOIRE_PREFIX_CACHE`
  does NOT yet compose with batching** — the batchable path never
  resumed a growing conversation to begin with, and now provably cannot
  by accident either. See `DAY-ONE.md` section 2c for the corrected
  claim.
- A cancelled streaming request under PP could desynchronise the
  resident pipeline for the NEXT request — fixed by keeping every rank's
  step count matched even after a client disconnects.
- A speculative round that hit EOS or was cancelled PARTWAY through an
  accepted block could save a prefix snapshot whose token list didn't
  match the recurrent state it actually held — fixed by declining to
  save in that specific case, now a permanent arm in
  `tests/test_generation.cpp`, not a one-off script.
- A stale, unrelated scalar could refuse a batch every row of which had
  real room — fixed, now a permanent arm in `tools/test_batch_decode.cpp`.
- `tools/b70run.sh` swallowed a failed container's exit status — fixed;
  every Tower gate and real-model run goes through this launcher.
- `tools/preflight_b70.sh`'s multi-rank gates (test_parallel_e2e,
  test_spec_e2e, the new test_pp_server) were wired through a
  single-device launcher, so they would throw on rank 1 the moment real
  GPUs are visible — off the card this was invisible, since no GPU at
  all falls back uniformly for every rank. Given a dedicated multi-device
  runner reusing `pp2run.sh`'s own proven container flags. **This one
  could not be tested here — there is no GPU in this container — verify
  it on the actual first Tower run, not just trust the diff.**

Everything else below this point is as it was, current and correct as
of the fixes above.

Ian is expected back at the Tower around now. **The job stated in
`CLAUDE.md`'s current priority is: the box works the day he powers it
on.** Everything below was built and verified off the card, in a
container, against that goal. Nothing in this file is a benchmark claim
(rule 8) — every number here is a token count or a pass/fail, never a
timing.

## Start here

```bash
ssh root@192.168.8.225
cd /mnt/storage/isos/grimoire-fuse && git pull
tools/preflight_b70.sh /models/<dir>
```

It builds bridges first (rule 3), then the engine, then runs **every**
gate below, in order, and stops at the first required failure. Read the
generated text at the end (rule 8) before trusting anything.

## What shipped since the last full-suite-green commit (`5825caf`)

14 commits, ~3400 lines. In the order they landed:

1. **NVFP4 reading** (`acadf5f`) — NVIDIA Blackwell 4-bit checkpoints
   decode to f32 at load and re-quantize through the ordinary path.
   `test_nvfp4_e2e`.
2. **Honest off-card measurement** (`aea2e1e`) — `PERF-2026-09-17.md`:
   the server was serial, confirmed by comparison against an AMD RDNA4
   stack on the same model family. This is the finding that started
   everything below.
3. **Conversations resume instead of re-reading** (`b572d66`,
   `533753a`, `c7708e1`) — a prefix cache that tolerates a GROWING
   conversation (not just a byte-identical repeat), sized to N slots so
   multiple agents each keep their own context, and rewritten so a
   resume is an O(layers) pointer move instead of a copy proportional to
   `max_seq`. `test_prefix_reuse`.
4. **Several conversations decoded in one pass** (`9aa975d`,
   `fdd3136`) — batched decode: one step, several sequences, one token
   each. Dense, MoE, and HYBRID (Ornith's shape — this needed per-slot
   recurrent state, not just per-slot KV). `test_batch_decode`.
5. **The server mutex is gone** (`6a6bbde`) — a resident scheduler
   steps every overlapping request together instead of queuing them
   behind a lock held for the whole request. `test_scheduler`.
6. **Docs + a real bug found by strengthening a test** (`a054ce3`) —
   `CONCURRENCY-2026-09-17.md`, and CLAUDE.md rule 19: a fixture too
   short to let a recurrent-state bug show, which was later found by
   lengthening it and turned out to also be hiding a real defect
   (`clear_seq_slot()` — reset-then-bind was backwards).
7. **A conversation may resume with a drafter loaded** (`b9f998b`) —
   the prefix-cache-vs-speculation refusal was inherited from an
   unrelated PP deadlock fix and was never actually about MTP. Lifted.
   Also fixed a second env-caching bug of the exact shape rule 19 warns
   about (`mtp_enabled()` was a `static`).
8. **A pre-existing MoE crash, found and fixed** (`171e972`,
   `3b9389d`) — a device-capability guard existed on `mm()` but two
   direct `launch_gemm_xmx` call sites in the MoE `M>=32` branch
   bypassed it. Reproduced against a commit from BEFORE any of this
   session's work (6/6 crashes), so it predates all of it. **A B70 was
   never affected** — it has the matrix hardware the guard exists to
   protect against not having. Fixed anyway because it was blocking
   verification of MoE batched prefill off the card. CLAUDE.md rule 20.
9. **Serving across TWO cards** (`d855869`) — pipeline parallel was
   CLI-only (`pp2run.sh` runs one prompt and exits). The HTTP front end
   now forwards each request down the pipe and the other stages follow
   it. `tools/serve_pp2.sh`. `test_pp_server`.

## The full gate list now (`tools/preflight_b70.sh`, in order)

```
make test / make test-correctness      13 host suites
bin/test_k2_kernels                    every kernel vs its host reference
bin/test_k2_e2e                        K2 path loads and generates
bin/test_model_matrix                  10 architectures x 7 formats
bin/test_parallel_e2e                  PP/TP == single process
bin/test_spec_e2e                      speculation == plain decode
bin/test_gemma4_prefill                batched == sequential
bin/test_qwen4_exp_e2e                 Qwen3.8-Flash-Next mechanisms live
bin/test_nvfp4_e2e                     NVFP4 == bf16 twin
bin/test_prefix_reuse                  resume == re-read, and cheaper
bin/test_batch_decode                  batched decode == serial decode
bin/test_scheduler                     concurrent requests == serial requests
bin/test_pp_server                     a resident pipeline serves, not just runs
generate (real model, real prompt)     YOU READ THE OUTPUT
```

All 11 device gates plus 13 host suites are green off the card as of
`d855869`, rebuilt and re-run against that exact commit before it was
pushed (not just "was green once earlier").

## What is verified and what is not — be precise about this

**Verified off the card (correctness only, no timing means anything):**
- every gate above is a token-identity or byte-count check, run for
  real, with negative controls proven to catch the bug they exist for
  (see CLAUDE.md rules 15–20 for the specific ones worth reading before
  trusting a green gate blindly)
- the two-card server sends three requests of different lengths down one
  resident pipeline (not spawned fresh per request) and reads the
  answer back at the stage that would actually hold the HTTP socket

**NOT verified, and only the Tower can settle it:**
- **every number.** Batched decode, the scheduler, conversation resume,
  and the two-card server are all claimed to be FASTER. None of that is
  measured. The off-card MoE batched-GEMM path is a plain-SYCL fallback,
  not the XMX tile the B70 runs — a green MoE gate here is not a
  statement about the tile (rule 20's closing note).
- the OCuLink link itself, every XMX tile, the AOT image
- whether `GRIMOIRE_SEQ_SLOTS` / `GRIMOIRE_MAX_BATCH` actually fit in
  VRAM alongside a real model at real `--ctx` — this was sized in theory
  (N slots = N KV caches) and never checked against a real checkpoint

## What is refused, on purpose, and still refused

Read `batch_unsupported_reason()` in `src/grimoire.cpp` for the exact
list, but the two that matter most for how Ian actually runs models:

- **batching + speculation are mutually exclusive.** A drafter turns
  batching off; the scheduler falls back to one request at a time with
  MTP/DFlash intact. Nobody has measured which wins at his usual
  concurrency (`CONCURRENCY-2026-09-17.md` names this explicitly as the
  first thing to measure).
- **no batching under TP or PP.** The two-card server (`serve_pp2.sh`)
  is serial by design — a pipeline is already parallel over tokens, not
  sequences. Only Muse, gemma-4, and Qwen3.8-Flash-Next still have no
  batched-decode path at all (each has its own residual graph; the
  pattern from dense/MoE/hybrid would need porting to each one
  separately — mechanical, not designed yet).

## What to send another assistant for an audit

The diff range is `5825caf..d855869` on `claude/new-session-pro26b` —
14 commits, listed above with what each one is for. Point them at:

1. **This file** for the map.
2. **`CLAUDE.md` rules 15–20** — every rule added this session records
   a real bug and, more usefully, *how it was found* (a fixture too
   short to contain the failure, a guard that didn't cover every call
   site, a `static` that cached an environment variable across test
   arms). An auditor should treat a green gate as a claim to verify, not
   evidence on its own — these rules are what to check for.
3. **`CONCURRENCY-2026-09-17.md`** for the full design rationale behind
   the prefix cache / batching / scheduler work, if they want the "why"
   behind commits 3–5 above.
4. Ask them to specifically check: the `PPRequest` wire protocol in
   `pp_send_request`/`pp_recv_request` (fixed-width header, a
   desynchronised length would corrupt the NEXT request silently — this
   is exactly the class of bug rule 7c/7d in `CLAUDE.md` already warns
   about for the existing PP code), and whether `clear_seq_slot()`'s
   bind-then-clear ordering (rule 19) is actually followed at every call
   site that admits a new sequence into a slot.

## Immediate next step once the Tower confirms green

`CONCURRENCY-2026-09-17.md`'s measurement list, in the order it gives:
tok/s at 1/2/4/8 concurrent, batching vs. speculation, real VRAM budget
for `GRIMOIRE_SEQ_SLOTS`, and prefix-reuse time-to-first-token on a real
multi-turn conversation. None of those four have a number yet.
