# Working on GRIMOIRE — read this first

## >>> CURRENT PRIORITY (read this before anything else) <<<

**As of 2026-09-12: Ian is away until roughly 2026-09-19 and comes back
to a Tower with two B70s on OCuLink. The job is that the box WORKS the
day he powers it on — dual GPU, FP8, pipeline parallel.**

Start at `DAY-ONE.md`, and run `tools/preflight_b70.sh /models/<dir>`
before anything else. It builds in the right order (bridges first, rule
3), runs every gate, and finishes by generating text for a human to read
(rule 8). Do not touch performance work until it is green.

What is already verified, off the card, in a container with a real
oneAPI toolchain (`TOOLCHAIN-IN-A-CONTAINER.md`, and rule 9 below):

- every SYCL source compiles; `bin/grimoire` and `bin/grimoire-server` link
- the new kernels RUN and match their host references (`test_k2_kernels`)
- 3 architectures x 7 projection formats all load and generate
  (`test_model_matrix`)
- PP and TP produce token-identical output to a single process, in BF16
  and FP8, for dense, MoE and K2 (`test_parallel_e2e`)

What is NOT verified and only the Tower can settle: the OCuLink link
itself, every XMX tile, the AOT image, and every number.

Below this line is the previous priority. It is still real work and
still wanted -- but it is second until the box is known good.

---

Previously (2026-08-28), the top-of-list task was:

**Implement DFlash speculative decoding for Ornith-1.5-35B-A3B.**
Reference: ultimatechris/Ornith-1.5-35B-A3B-DFlash-SGLang — block-level
drafting (whole 16-token block in one pass), 6-8 tokens accepted per step,
proven 2.32x speedup (202 -> 470 tok/s) on vLLM/SGLang. Full ranked plan,
math, and the 5 other links that were evaluated and rejected are in
`ORNITH-2026-08-27.md` under "ORNITH TG OPTIMIZATION". Read that section
in full before starting.

**Latest implementation handoff:** `HANDOFF-2026-08-28-ORNITH-DFLASH.md`.
The original z-lab DFlash drafter now loads; resume at context ingestion and
the 16-query block forward described in that file.

**RETRACTION (2026-09-12): "the lower-acceptance DFlash2 sidecar" was
measured with its candidate selector DEAD, so that verdict is void.**
The selector kernels (`launch_dflash2_selector_edges`,
`launch_dflash2_path_walk`) have existed in `ops.cpp` since the initial
commit and were NEVER CALLED until 2026-09-12. Both judgements --
`8e2fbc8` (2026-09-01, "97.6 TG, still LOSES to 125 TG plain decode") and
`f677add` (2026-09-02, "LOSES to MTP -- drafter too expensive") -- predate
any call site. What they measured was DFlash2 drafting by per-position
argmax, which ignores the predecessor/successor codebooks entirely: the
weights loaded, the scoring never ran. That is not a slower DFlash2, it is
a different and structurally weaker algorithm.

DFlash2's acceptance comes from scoring the EDGE between the previous
choice and each candidate and walking a path from the verified anchor.
With that removed, low acceptance was the expected result, not evidence
about DFlash2. Ian's position (2026-09-12) is that DFlash2 beats MTP; the
repo holds no valid measurement either way, because the only DFlash2
numbers on record are from the selector-less build. Re-measure both on the
same prompt before trusting any ranking, and read the output (rule 8).

**DFlash config fidelity (2026-09-13).** The drafter's `config.json` was
read with a substring search and `strtol`, and everything else was
hardcoded -- rope_theta defaulting to 1e7 where the reference defaults to
1e6, a fixed head_dim and norm epsilon, "layers 0-4 slide at 4096", and
always the TARGET's lm_head and embedding even for a drafter that ships
its own. All of it now resolves from the draft config the way
`ref/qwen3_dflash.py` resolves it
(`include/b70/dflash_config.hpp`, pinned by `tests/test_dflash_config.cpp`),
and the loader PRINTS what it resolved. Read those `dflash` banner lines
first: every value in them is silent when wrong -- the drafter still runs
and the output stays correct, only acceptance moves. Whether any of them
actually differed for the z-lab checkpoint is still unknown; huggingface
was unreachable from the work container, so one load on the Tower is what
settles it. Details: `HANDOFF-2026-09-13-DFLASH-CONFIG.md`.

This supersedes the Muse Glimmer prefill work mentioned later in this file
and in HANDOFF-2026-08-27-PP-TP-COMPLETE.md, which is real but lower
priority right now. If you are unsure which task is current, ASK IAN rather
than picking one — priorities have been shifting fast in this project and
a stale doc has caused confusion before.

---


This file is for any AI assistant (Claude, or otherwise) picking up this
project. Ian runs multiple assistants on GRIMOIRE at different times — this
doc exists so none of them re-derive or re-break what's already settled.

## Where things run

All real work happens on **the Tower** (`root@192.168.8.225`), a Unraid box
with 2x Intel Arc Pro B70 (Battlemage), reachable over SSH. The repo lives at
`/mnt/storage/isos/grimoire-fuse`. This GitHub repo is the SOURCE OF TRUTH for
code; the Tower is where it's built and run. If the Tower rebooted and
`/mnt/storage/...` is missing, the Unraid ARRAY may not have auto-started —
check before assuming data loss.

Models live under `/mnt/storage/Models/` on the Tower, never in git.

## Backup workflow — GIT, NOT TARBALLS

As of 2026-08-28 this project backs up via `git add && git commit && git push`
to `git@github.com:doopeworld/GRIMOIRE.git`, run from the Tower. **Do not
create `.tar.gz` backups** — any tarballs you find on the Desktop or Tower
predate this and are stale, safe to ignore.

Commit at real checkpoints: a verified working state, right before a risky
kernel/architecture change, and end of session — not only when explicitly
asked. Only committed work survives a lost session; there is no tarball
safety net anymore, so commit discipline matters more, not less.

`.gitignore` excludes all generated content (`bin/`, `*.so`, `runtime-b70/`,
logs, `.bak*`, model files). Keep it that way — this repo is source only.

Auth: an SSH deploy key lives on the Tower (`~/.ssh/id_ed25519`), registered
to Ian's GitHub account. The `known_hosts` warning on push
(`hostfile_replace_entries: ... Operation not permitted`) is a harmless
Unraid mount quirk — push succeeds anyway, ignore it.

## Hard-won rules — do not re-learn these the expensive way

**1. `GRIMOIRE_W4A8=1` frees the MXFP4 payload of every converted weight.**
Any code path that still dereferences `w.payload` after conversion reads a
freed pointer ON DEVICE — this is a `DEVICE_LOST`, not a clean error, and it
drops a B70 off the PCI bus (`forcewake register returns 0xFFFFFFFF`,
needs a POWER CYCLE to recover, no software fix). Also: a converted weight
still reports `fmt == MXFP4` — checking format is NOT a safety check, check
the pointer. After touching weight conversion, `grep -n "\.w\.payload"` and
confirm every hit is guarded or routed through `mm`/`mmb`/`mmbb`/`gemv_any`.

**2. `GRIMOIRE_DAG=1` (out-of-order queue) produces GARBAGE, not a speedup.**
It looks like a free +10% PP because the out-of-order queue overlaps kernels
— but it doesn't express all real dependencies, so it's a race and the
output is wrong. `FULL E2E PP: PASS` does NOT catch this (the self-check
isn't sensitive to ordering bugs) — only generating text and reading it did.
Never trust a queue-ordering change without generating and reading text.

**3. `build_b70.sh` does NOT rebuild the cutlass bridges** (the `.so` files
in `src/`). After touching a bridge (`xe2_grouped_bridge.cpp` etc.), you MUST
run `tools/build_bridges_b70.sh`, which rebuilds bridges then calls
`build_b70.sh`. Running `build_b70.sh` alone silently leaves a stale `.so` —
this caused a DEVICE_LOST that looked like a kernel bug but was a 5-minute-
old bridge.

**4. Any W4A8 GEMM tile is 256 wide in N; the B 2-D block loads do NOT clamp
to the tensor.** A weight with `N % 256 != 0` (e.g. Ornith's deltanet `a/b`
projection, N=96) reads hundreds of KB past the end of the tensor ->
DEVICE_LOST. Guard `N % 256` before routing a small-N weight through a W4A8
tile; use the GEMV instead.

**5. Small-M batches need a SMALL TILE, not the production 128-row tile.**
The 128-row W4A8/MXFP4 tile fetches 128 rows of A per tile regardless of how
few tokens are given — at M=4 that's ~89 MB/layer of pure padding. A 16-row
tile (`m16x256`) is weight-bound and flat from M=1 to M=16. This is what
makes a speculative verify batch or short prefill affordable; forcing the
wide/128 tile on a small batch is a measured DISASTER (batch cost went UP,
not down).

**6. Never `docker run` the B70 directly with a bare `timeout` wrapper.**
`timeout N docker run` orphans the container and wedges the GPU; a bash-as-
PID-1 entrypoint also swallows SIGTERM and escalates to SIGKILL, which tears
down mid-submission and drops the card off the bus. Always launch through
`tools/tune.sh` / `tools/vtune.sh` / `tools/pp2run.sh` / `tools/tp2run.sh`,
which wrap `b70run.sh`'s safe `--init` + in-container `timeout` + detached
`docker wait` pattern.

**7. Multi-GPU: OCuLink Gen4 x4 IS NOW INSTALLED (confirmed by Ian,
2026-09-12).** GPU1 is no longer on the USB4/Thunderbolt link. The old rule
said not to re-attempt single-process multi-device until OCuLink landed —
that condition is now met, so single-process cross-device IS worth
re-attempting. The original failure (raw cross-device allocation fails once
the process holds a large host mmap — measured, see `ORNITH-2026-08-27.md`)
was a link-related limitation and has NOT been retested on OCuLink.

Multiprocess PP/TP still works and is still the proven path —
`tools/pp2run.sh`, `tools/tp2run.sh`,
`HANDOFF-2026-08-27-PP-TP-COMPLETE.md`. The pipeline boundary stages the
hidden state through pinned host memory and a file descriptor
(`pp_send_hidden` / `pp_recv_hidden`): device -> host -> pipe -> host ->
device, synchronously, at every layer-group boundary.

EXPECT THAT TO BE MUCH CHEAPER NOW, AND RE-MEASURE BEFORE REDESIGNING.
USB4 carries PCIe by TUNNELING it, which added large latency to every
device/host transfer on GPU1 — and the staging above is two such transfers
per boundary. OCuLink Gen4 x4 is native PCIe with no tunneling, so the link
change attacks exactly the dominant cost. The host round trip remains in
the code, but its price was mostly the tunnel, not the copy.

So the first move is a measurement, not a rewrite: re-run the multiprocess
pipeline on OCuLink and see where it actually lands. Single-process
device-to-device is still worth trying (see above), but it is no longer
obviously the bigger win.

**MTP speculation now works under BOTH TP and PP (2026-09-12).** The old
note here said speculation was hard-disabled whenever TP or PP was on, and
that dual-GPU meant no drafter. That is no longer true:

- **TP**: the MTP head is small, so it loads REPLICATED on every rank and
  each rank drafts identically with no extra collective. Two spots needed
  fixing — the embedding table is row-sharded under TP (the head must go
  through `embed_one`, not `launch_embed`), and the reduced-draft-vocab
  lm_head shortcut skipped the all-gather.
- **PP**: only the last stage owns the final hidden state, so it hosts the
  head, drafts, and pushes each draft token backward on the same channel
  `argmax_token` already uses. The verified tokens the head chose go back
  the same way (`pp_sync_tokens`), so every stage computes the same
  acceptance count and rolls back to the same position. The stages agree
  on whether speculation is on at all via a one-hop handshake in
  `pp_connect` — deciding locally would let one rank draft while another
  did not, and they would deadlock on the next collective.
  The last stage also loads the embedding table (the head embeds every
  token it drafts), which costs one extra copy on one card.

Verified by `bin/test_spec_e2e`: speculative output is IDENTICAL to plain
greedy output, single process and across two processes, TP and PP, dense
and MoE, BF16 and FP8. That is the right test because speculation is an
exactness claim, not an approximation.

**DFlash now runs under BOTH TP and PP too (2026-09-13).** The old note
here said DFlash was single-GPU only. Both reasons it gave have been
removed:

- **TP**: the drafter is small, so it loads REPLICATED on every rank and
  each rank drafts identically with no extra collective -- the same
  argument that made the MTP head work. Two spots had to be fixed, and
  both were silent: the block embed read the target's embedding table,
  which TP shards over the VOCABULARY (so every rank embedded a plausible
  WRONG token), and a drafter sharing the target's `lm_head` read a head
  sharded the same way.
- **PP**: the drafter consumes the residual stream tapped at several
  TARGET layers, and a split puts those layers on different stages. Each
  stage now captures the taps it owns and forwards the block to the next
  one, which overwrites its copy, fills in its own, and forwards again --
  so the last stage, which hosts the drafter exactly as it hosts the MTP
  head, ends up with the whole concatenated row. The drafted block goes
  backward on the same channel `argmax_token` already uses, so every
  stage speculates on identical candidates and rolls back to the same
  position. Stages agree on the tap count and the draft width through
  two more hops in `pp_connect`, because a stage that disagreed would
  read a different number of floats off the socket than the previous one
  wrote and desynchronise every later message.

The extra PP traffic is `n_taps * hidden` floats per token per boundary,
on top of the hidden state -- for an 8-tap drafter that is 8x the
boundary bytes. Nobody has measured what that costs on OCuLink. Do that
before preferring DFlash to MTP on two cards.

Verified by `bin/test_spec_e2e`, which now drives DFlash single-process,
under TP and under PP, with a drafter forced to non-zero acceptance so
both the accept and the rollback paths run. The mini drafter taps target
layers 0 and 2 of a 4-layer target, so a 2/2 split lands one tap on each
stage: drop the forwarding and the test fails.

The capability matrix still says which one is live -- read it.

**7b. A POINTER CHECK IS NOT A WRITTEN CHECK (learned 2026-09-14).**
`commit_spec_prefix` restored the drafter's hidden state from
`spec_hidden_steps` under `if (spec_hidden_steps)`. That buffer is
allocated the moment MTP loads and written ONLY by the batched verify, so
on any sequential-verify fallback the drafter ran on uninitialised device
memory. The comment above the line already said "skip rather than read a
buffer that was never written" -- the condition just did not implement
it. Symptom: `MTP draft failed`, intermittently, because stale memory is
usually finite and occasionally is not.

Two lessons, both cheap:

- When a buffer has conditional writers, track WRITTEN, not ALLOCATED.
- POISON such buffers with NaN at allocation. It converts this whole
  class from "fails one run in three" to "fails every time, loudly, the
  first time anyone reads it" -- which is what made the A/B that proved
  the fix possible at all.

**7c. UNDER PP, THE LAST STAGE ALWAYS OWES A REPLY -- AND ITS SHAPE
DEPENDS ON THE PHASE (learned 2026-09-14).** Earlier stages are never
idle after calling into the drafter. During DRAFTING they sit in
`pp_sync_tokens()` waiting for M-1 integers. After a CONTEXT-ONLY call
they return at once and go straight to `argmax_token()`, which waits for
exactly ONE. So:

    drafting, failed      -> M-1 negative tokens
    context-only, failed  -> ONE negative token
    context-only, ok      -> nothing; argmax_token sends the scalar

Both wrong answers have been shipped here and both were silent. Sending a
block during a context-only call put the first -1 where the scalar was
expected and left the rest for the NEXT message to misread -- fatal to a
resident server, harmless to a CLI that exits. Sending nothing left the
peers blocked forever on a scalar.

The notifier is an RAII destructor, not a lambda called at each exit,
because the paths that need it most have no exit to annotate: a bridge, an
allocation, a kernel submission or a wait can throw straight out of the
function. It uses a fixed, clamped buffer so the failure path allocates
nothing even after a `bad_alloc`.

`bin/test_spec_e2e` proves it end to end rather than with a mock: capacity
32, a valid 25-token prompt and M=8, so the drafter cannot prepare; both
ranks must fail and then the SAME engine and SAME sockets must serve a
normal request identically. Tearing the model down after the refusal would
hide both a blocked peer and a dirty wire.

**7d. ASK THE COMPILER (learned 2026-09-14).** The fix for 7b was itself
broken by a braceless `if`:

    if (capture_spec)
        q.memcpy(spec_hidden_steps, ...);   // conditional
        spec_hidden_valid = true;           // NOT conditional

so a verify batch wider than `kSpecBatch` marked a buffer valid that
nothing had written -- reachable ON THE CARD, where batched prefill works.
`build_b70.sh` now passes `-Wmisleading-indentation`. Keep it. A warning
costs nothing and this cost three audits.

**8. Never quote a benchmark number from a config that never generated text.**
Multiple false leads in this project (BesTLA, DAG, several kernel "wins")
were caught only because someone bothered to run `-p "prompt" -n N` and read
the output. A PASS on a numeric self-check is not sufficient.

**9. You CAN compile in a work container. Do it before handing off.**
For weeks this project handed off code that had never been through a
compiler, because "there is no SYCL toolchain here" was assumed. It is
not true: `icpx` plus an OpenCL CPU device install from conda-forge in
about ten minutes, no GPU needed. Procedure:
`TOOLCHAIN-IN-A-CONTAINER.md`.

The first run of it (2026-09-12) found that `src/grimoire.cpp` did not
compile at all, that the K2 softplus gate was 2.5e-4 off torch, and --
once the kernels were actually EXECUTED -- four engine bugs in a K2 path
nobody had ever called, including a 46 KB device heap overrun that is a
DEVICE_LOST on the card. None of that was findable by reading.

Two gates now exist and take seconds: `bin/test_k2_kernels` (kernels vs
their host references, on a device) and `bin/test_k2_e2e` (loads a
miniature checkpoint and generates). Run them everywhere, and run them
again on the Tower, where they additionally cover the XMX tiles and the
batched prefill the CPU device cannot touch.

Compile for `-fsycl-targets=spir64` in a container (AOT `bmg_g31` needs
`ocloc`), and use `-fsycl-device-code-split=per_kernel` for anything you
intend to RUN there — `off` puts every kernel in one image and the CPU
runtime dies on the `joint_matrix` ones even when nothing calls them.
`off` stays correct for the B70.

This does NOT weaken rule 8. Nothing here produces a number, and a CPU
device is not a B70. It is the floor, not the ceiling.

**10. The loader says what the file IS. The engine says what it can RUN.
Keep those in different places.**
`src/qwen35_loader.cpp` is a reader: it resolves a checkpoint's config and
refuses only things it cannot PARSE or that contradict the weight map. It
must not refuse a model because no forward path implements it — those two
facts go out of date independently, and the config tests
(`tests/test_gemma4_config.cpp` and friends) need the parse to SUCCEED in
order to pin anything at all. Put "this engine cannot execute that" in
`Grimoire::unsupported_reason()`, which `build()` calls before uploading a
single byte, and name the missing feature in the message.

Recognising a feature is not implementing it. Reading `hidden_act` and
setting `cfg.geglu` made a gelu checkpoint LOAD where it used to be
refused — while every FFN dispatch in the engine was still swiglu. Adding
a config field without a path that consumes it converts a clean refusal
into silent wrong output, which is strictly worse than not supporting the
model. When you add a flag, either wire it the same day or refuse on it
the same day.

A refusal deserves a test, and the test has to be specific. Requiring only
that the message contains the architecture's name passes on the loader's
own shape refusals — so a malformed fixture would look like a working
capability check. Match the engine's sentence.

**11. GEMMA-4 DOES NOT USE `(1 + w)` IN ITS NORMS. Gemma-2 and Gemma-3
DO (learned 2026-09-14, from an external audit).**
`Gemma4RMSNorm.forward` is `normed * self.weight` — no offset
(`ref/gemma4.py:211`). The whole family used the zero-centered convention
until this one, so the wrong answer is exactly what the family name leads
you to. Applying `(1 + w)` shifts every normalised activation in the
model and leaves the output fluent. This engine now sets the convention
per model in one place (`set_norm_convention`, called once in `build()`):
Qwen/Muse whole-row `(1 + w)`, K2 grouped `w`, gemma-4 whole-row `w`.

The knock-on is worth stating because it is not obvious: a SCALELESS norm
(gemma-4's `v_norm`, `with_scale=False`) is reproduced by a weight vector
of ZEROS under `(1 + w)` and by a vector of ONES under `w`. Get the
convention right and the placeholder wrong and the model multiplies its
values by zero. Name such a buffer after the norm it serves, not after
its contents — `gemma_zero` was a name that had to be wrong half the time.

More generally: when reading a reference for a new family member, read
the norm's `forward`, do not infer it from the sibling you already
implemented.

**12. A CONSTANT OUTPUT PASSES EVERY CHECK A GENERATION TEST MAKES
(learned 2026-09-14).** `tie_word_embeddings` was parsed into
`cfg.tie_embeddings` and then read by NOTHING. A tied checkpoint ships no
`lm_head` tensor, so the upload block was skipped, the loader printed
`ok`, and `lm_head` stayed a default `DevQuant` with `N == 0`. `gemv_any`
wrote nothing into `s.logits` and argmax read stale device memory.

The result: the same token every step, for every prompt. In vocabulary.
Correct length. Byte-identical across requests. `bin/test_model_matrix`
passed it seven times out of seven, because every property it checked was
TRUE of a model whose logits never change.

So a generation gate must assert that the output DEPENDS ON THE INPUT --
two different prompts, two different continuations. `test_model_matrix`
does that now, and it is the only check in it that a dead output
projection cannot satisfy. If you add a generation test, add that line.

The second lesson is about the banner again (rule 10): printing `ok` was
unconditional, outside the `if` that did the work. A status line that
cannot say "no" is not a status line.

Qwen, Ornith, K2 and Muse all ship an explicit `lm_head`, which is why a
missing feature survived this long. The first model that needed it was
gemma-4, and it was found by asking one question the test suite never
asked.

## Where to look for current status

Read the newest-dated `.md` at the repo root first (sort by date in the
filename or `git log --oneline -- '*.md'`) — these are running handoff logs
per session, most recent state and next-steps at the bottom of each file:

- `GRIMOIRE-2026-08-26-TG.md` — Qwen TG/PP work, W4A8 kernel details
- `ORNITH-2026-08-27.md` — Ornith TG work, speculative-decoding link reviews,
  ranked next-steps plan
- `HANDOFF-2026-08-27-PP-TP-COMPLETE.md` — multiprocess PP/TP, prefix cache,
  Muse Glimmer integration, and the CURRENT next-steps (Muse batched prefill)

`ref/` holds extracted vLLM reference implementations (DFlash2, Muse Glimmer
modeling code) — pulled from the vLLM nightly image specifically so nobody
has to guess an architecture's exact forward pass again. Read the actual
reference before implementing a new model or speculative decoding scheme;
several sessions were burned guessing norm order / concat order instead of
extracting the real code first.
