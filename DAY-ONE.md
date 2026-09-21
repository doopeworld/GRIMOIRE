# Day one on the Tower

Read this when you sit down at the box. It is the short version; the
reasoning behind every step is in `CLAUDE.md` and the dated handoffs.

## 0. Before anything

```bash
ssh root@192.168.8.225
ls /mnt/storage/isos/grimoire-fuse    # if this is missing, the Unraid
                                      # ARRAY did not auto-start.  Start it.
                                      # It is NOT data loss.
cd /mnt/storage/isos/grimoire-fuse
git pull
```

## 1. One command

```bash
tools/preflight_b70.sh /models/<your-model-dir>
```

It needs the vllm-xpu-kernels checkout for the bridge build and finds it
at `/mnt/user/appdata/vllm-xpu-kernels` or `/mnt/cache/appdata/...`; if
yours is elsewhere, `KERNELS=/path/to/it tools/preflight_b70.sh ...`.
`SKIP_BUILD=1` re-runs only the gates against the build you already have
-- use it to re-check, never to skip past a broken build.

That builds in the right order (bridges first — rule 3), then runs, in
order, and stops at the first required failure:

| stage | what it proves |
| --- | --- |
| build | bridges + engine actually compile on this box |
| no-Torch check | the bridges are still pure SYCL/Level Zero |
| `make test`, `make test-correctness` | 13 host suites |
| `bin/test_k2_kernels` | every new kernel matches its host reference |
| `bin/test_k2_e2e` | the K2 engine path loads and generates |
| `bin/test_model_matrix` | 10 architectures × 7 projection formats, plus the refusals |
| `bin/test_parallel_e2e` | PP and TP give the SAME tokens as one process |
| `bin/test_spec_e2e` | speculation gives the SAME tokens as plain decode, incl. dual GPU, MTP and DFlash |
| `bin/test_gemma4_prefill` | gemma-4 batched prefill == sequential decode |
| `bin/test_qwen4_exp_e2e` | Qwen3.8-Flash-Next: each mechanism is LIVE, and batched prefill == sequential decode |
| `bin/test_nvfp4_e2e` | an NVFP4 checkpoint decodes to the same numbers as a bf16 twin |
| `bin/test_prefix_reuse` | resuming a conversation answers what re-reading it answers, and copies no cache |
| `bin/test_batch_decode` | conversations stepped TOGETHER answer what each answers alone |
| `bin/test_scheduler` | requests from several threads at once answer what they answer one at a time |
| `bin/test_pp_server` | a resident pipeline answers several requests in a row, and rank 0 gets the tokens |
| generate | real model, real prompt — **you read the output** |

**Read the `hybrid` rows of `test_spec_e2e` first.** Off the card they say
REFUSED (no batched verify there, and the engine will not speculate
without one). On your B70 they run for real and must say `identical at
K=1,2,3,5`. If either says `CHANGED THE OUTPUT`, stop: that is
speculation on a Qwen3.5/Ornith-shaped model not reproducing the model's
own output, which is the path the verified 49.8 TG MTP recipe uses.

**A failure that used to live here is fixed** (2026-09-14):
`test_spec_e2e`'s `moe+mtp fp8` PIPELINE case failed intermittently
because the drafter restored its hidden state from a buffer only the
BATCHED verify writes. It is an engine bug, not a CPU-device artifact,
and the same fallback is taken on the card whenever a speculative scratch
allocation fails. Fixed and A/B-proved; see
`HANDOFF-2026-09-13-DFLASH-MULTIGPU.md`.

Still worth running the gates more than once here -- a single green run
on new hardware is weak evidence either way.

Green preflight means the box is sane. It does not mean anything is
fast: no stage above produces a number, deliberately.

**One path in that list gets its first ever execution on your card.** The
container used for all the verification above has no XMX, so the BATCHED
PREFILL path could not run there — every gate above exercised the
sequential decode path instead. Batched prefill is what prompt processing
actually uses, so treat the first `-n 64` on a long prompt as the real
first run: if something is going to be wrong, it is most likely there.
`test_k2_e2e` prints which prefill path it took, by name, so you can see
on the card that the batched one was finally exercised.

## 1b. Single GPU

Once preflight is green, this is the whole thing:

```bash
GPU=gpu0 LIM=600 tools/tune.sh run \
  /grimoire/bin/grimoire -m /models/<dir> --proj int4 \
  -p "Explain in two sentences why the sky is blue." -n 64
```

Always through `tools/tune.sh` — never `timeout ... docker run` (rule 6:
that orphans the container and wedges the card).

Read the `capabilities:` block it prints before you read the tokens. It
names what is actually live — parallel mode, which drafter, whether
prefill is batched or fell back — because every one of those degrades
SILENTLY and a lost feature looks like a slow model.

**Engine switches do NOT pass through your shell.** `tune.sh` builds the
container's environment itself, so `GRIMOIRE_MTP=1 tools/tune.sh ...`
sets the variable on the HOST and the container never sees it. Use
`EXTRA_ENV`, newline-separated `K=V`, which `tune.sh` appends to the env
it hands the container.

Speculation on one card, if the checkpoint has an MTP head:

```bash
EXTRA_ENV="GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_SPEC_STATS=1" GPU=gpu0 LIM=600 tools/tune.sh run \
  /grimoire/bin/grimoire -m /models/<dir> --proj int4 -p "..." -n 64
```

With a separate DFlash drafter instead:

```bash
EXTRA_ENV="GRIMOIRE_DFLASH_MODEL=/models/<drafter>
GRIMOIRE_SPEC_STATS=1" GPU=gpu0 LIM=600 tools/tune.sh run \
  /grimoire/bin/grimoire -m /models/<dir> --proj int4 -p "..." -n 64
```

Always set `GRIMOIRE_SPEC_STATS=1`: it prints accepted-per-step, and a
drafter that is correct but never accepted is a silent 1.0x -- it looks
like a working feature and buys nothing. It also tells you the switch
ARRIVED: if it did not, the capability matrix says `speculation none` and
there is no `spec:` line at all.

## 2. Dual GPU

GPU1 is now on OCuLink Gen4 x4, not USB4. The USB4 link tunnelled PCIe
and added large latency to every host/device transfer, and the pipeline
boundary is two of those per layer group. **Re-measure before redesigning
anything** — the thing that was slow may simply not be slow any more.

Pipeline parallel (the proven path, and what to use for a model that
does not fit on one card):

```bash
GRIMOIRE_PP_SPLIT=24 tools/pp2run.sh renderD129 renderD130 1800 pp \
  /grimoire/bin/grimoire -m /models/<dir> --proj fp8 -p "..." -n 64
```

Tensor parallel:

```bash
tools/tp2run.sh renderD129 renderD130 1800 tp \
  /grimoire/bin/grimoire -m /models/<dir> --proj fp8 -p "..." -n 64
```

Verified in a container, two processes, before you ever powered the box
on: PP and TP both reproduce the single-process tokens exactly, for a
dense model, a routed-MoE model and K2, in BF16 and FP8. What was NOT
verified there is the link itself — that is what these two commands
measure and nothing else can.

**Speculation now works on dual GPU.** The two-rank launchers take the
same newline-separated form but through `GRIM_ENV`, not `EXTRA_ENV` --
they call `b70run.sh`'s env path directly:

```bash
GRIM_ENV="GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_SPEC_STATS=1" GRIMOIRE_PP_SPLIT=24 tools/pp2run.sh \
  renderD129 renderD130 1800 pp \
  /grimoire/bin/grimoire -m /models/<dir> --proj fp8 -p "..." -n 64
```

Under PP the last stage hosts the
MTP head and drives the drafting for the whole pipeline; under TP the head
is replicated and every rank drafts identically. Verified token-for-token
against single-process plain decode — speculation is exact, so if it ever
changes the output that is a bug, not a trade.

DFlash now runs on dual GPU as well, under both TP and PP. Add
`GRIMOIRE_DFLASH_MODEL=/models/<drafter>` to either launcher — every rank
needs the same value, and the stages check that they agree.

Under TP the drafter is replicated on every rank. Under PP the LAST stage
hosts it, and every earlier stage captures the target taps it owns and
forwards them up the pipe. That forwarding costs `n_taps * hidden` floats
per token per boundary on top of the hidden state — for an 8-tap drafter,
8x the boundary traffic. **Nobody has measured that on OCuLink.** MTP
sends nothing extra, so do not prefer DFlash on two cards until you have
compared them on this box, same prompt, accepted-per-step and tok/s.

The load banner prints which drafter is live — read it.

**If you load a DFlash drafter, read its four banner lines before you read
anything else:**

```
  dflash config: 6 layers, taps [...] from target_layer_ids, mask N,
                 rope_theta ..., eps ..., head_dim ...
  dflash attention: K of 6 layers sliding (window W), C causal
  dflash head:  the drafter's own lm_head / the TARGET's lm_head
  dflash embed: the drafter's own embed_tokens / the TARGET's
```

Every one of those is silent when wrong: the drafter still runs, still
proposes real tokens, and the verified output is still correct — only
acceptance moves. Check each against the drafter's own `config.json`.
That is the cheapest possible first step on the 2.21-accepted-per-step
blocker, and it costs one load.
`GRIMOIRE_DFLASH_LEGACY_SLIDING=1` restores the old "layers 0-4 slide at
4096" assumption if you want to A/B it — on accepted-per-step, not tok/s.

One rule the engine now enforces for you: a model with linear-attention
(DeltaNet) layers — Qwen3.5, Ornith — can only speculate when the BATCHED
verify is available, because a rejected draft replays the recurrent state
from buffers only that path captures. On the B70 it is available, so this
never bites in normal use. Where it does bite is **TP**, which always
falls back to token-at-a-time prefill: TP + a hybrid model + speculation
comes up with speculation OFF and says so. PP is unaffected. This was
found by testing, not reasoning — before the check, that configuration
produced fluent output that was quietly not the model's output.

Prompt processing differs between the two modes: **PP has batched prefill,
TP does not** (it falls back to token-at-a-time and says so in the banner).
For anything prompt-heavy on two cards, use PP.

## 2b. Recommended launches

Single card, model fits — the verified Qwen recipe (`QWEN-RECIPE.md`, and
do not mix in the Ornith flags it warns about):

```bash
GPU=gpu0 LIM=900 EXTRA_ENV='GRIMOIRE_W4A8=1
GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_DRAFT_VOCAB=131072'   bash tools/tune.sh qwen /grimoire/bin/grimoire   -m /models/<dir> --proj mxfp4 --ctx 8192 -p 'Explain why the sky is blue.' -n 256
```

Two cards, model does not fit — pipeline, with speculation:

```bash
GRIM_ENV='GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_DRAFT_VOCAB=131072' GRIMOIRE_PP_SPLIT=24 tools/pp2run.sh renderD129 renderD130 1800 pp   /grimoire/bin/grimoire -m /models/<dir> --proj fp8 -p '...' -n 64
```

One thing to know before comparing TG across the two modes:
`GRIMOIRE_MTP_DRAFT_VOCAB` is the flag QWEN-RECIPE says acceptance depends
on. It still applies under **PP** (the head and the lm_head both live on
the last stage, unsharded). Under **TP** the lm_head is row-sharded, so
the reduced-vocabulary shortcut would have each rank argmax over its own
slice — it is routed through the full all-gathered projection instead.
Correct, but the draft-vocab speedup is not available there. Another
reason PP is the default choice for two cards.

## 2c. Several agents at once

New on 2026-09-17. The server used to hold one lock for the whole of a
request, so the second caller waited for the first to FINISH. It now
steps every overlapping request together: a decode step reads all the
active weights to produce one token and reads the SAME weights to
produce eight, so eight agents cost about what one costs.

```bash
GRIMOIRE_SEQ_SLOTS=8   # how many conversations the KV cache holds
GRIMOIRE_MAX_BATCH=8   # how many are stepped together
```

`GRIMOIRE_SEQ_SLOTS` multiplies the KV cache: eight slots is eight times
the cache, so size `--ctx` and the slot count against the VRAM the model
leaves you. The banner prints what it decided —

```
    prefix cache  8 conversations resident
    batching      up to 8 sequences per step
```

— and if it says `one at a time` it names the reason. Read that line
before concluding the flags did nothing.

Also worth setting:

```bash
GRIMOIRE_PREFIX_CACHE=1   # a chat turn resumes instead of re-reading
```

Without it every turn of a conversation re-reads the whole history. With
it the turn processes only the new tokens. For agentic work, where the
prompt is long and the reply is short, that is the larger of the two
wins on a single card.

**CORRECTION (2026-09-21, external audit F8): this does NOT compose
with batching yet, and an earlier version of this doc said it did.**
When the scheduler admits a request into the batchable path (multiple
sequence slots, a matrix-capable device, no drafter), it always clears
the chosen slot and prefills the WHOLE prompt from scratch -- it never
calls `prefix_reuse()`/`restore_prefix_upto()` first, and its completion
path never performs the end-of-request snapshot save either. So with
both flags set, a growing conversation served through the batchable
path is read in full every turn regardless of `GRIMOIRE_PREFIX_CACHE`;
the resume only happens on the SERIAL path (a drafter loaded, one slot,
or a device that cannot batch). This was found, not assumed: fixing a
worse bug in the same admission code (two concurrently-admitted
requests with an identical prompt could be silently rebound onto the
SAME physical slot -- see CLAUDE.md rule 21) required disabling the one
mechanism that could have made batched requests resume at all, and nothing
in the existing gates exercised the combination to notice it was already
broken. Making both compose safely needs cache reuse to participate in
the scheduler's OWN slot ownership, not just the engine's -- real work,
not done here.

**What is NOT covered by any of this**: speculation. A loaded drafter
turns batching off (the banner says so) and the server falls back to one
request at a time, which is what it did before. MTP or DFlash versus
eight-way batching is a real trade nobody has measured -- that is a
Tower measurement and it is in the open list below.

## 2d. Serving on TWO cards

New on 2026-09-18. Until now pipeline parallel was CLI-only —
`pp2run.sh` runs `bin/grimoire` with one prompt and exits — while
`serve.sh` opens exactly one render node. So a model that needs two
cards (Ornith at fp8, anything at bf16) could be RUN but not SERVED,
which is the configuration agentic work actually uses.

```bash
tools/serve_pp2.sh /models/<dir> 8099 24     # model, port, layers on rank 0
PROJ=fp8 CTX=8192 tools/serve_pp2.sh /models/<dir>
```

Rank 0 binds the HTTP port and forwards each request down the pipe; rank
1 loads its layers and follows. Give the SLOWER card fewer layers —
pipeline throughput is set by the slowest stage.

**What was missing was small**: every stage already ran the same
generation loop and stayed in step by exchanging a message per token,
but the PROMPT only ever reached rank 0. The CLI never noticed, because
every rank is launched with the same `-p` and reads it off its own
command line.

**Concurrency does NOT apply here.** Under pipeline parallel the
scheduler falls back to one request at a time, and the prefix cache is
off. Both say so in the banner. On ONE card at int4/mxfp4 you get
batching and resuming; this is the two-card path and it is serial. That
is still the difference between serving the model and not.

Checked by `bin/test_pp_server`: three requests of different lengths
down one resident pipeline, at 2 and 3 stages, answered at RANK 0, must
equal the same three from a single process. The second and third
requests are the point — a byte stream cannot be resynchronised, so a
header written but not read is invisible on the request that causes it
and fatal to the next.

## 2e. More than two cards, or cards that are not the same

GRIMOIRE runs on any Battlemage part, and a box can mix them. Two things
make that work:

**One binary covers both dies.** B70/B65 are `bmg_g31`; B580/B60/B50 are
`bmg_g21`. An AOT image built for one die does not run on the other at
all. `build_b70.sh` now builds both by default, so the same binary drives
every card. `B70_TARGET=intel_gpu_bmg_g31` builds a single die when you
are iterating and want the compile to be quick.

**Device selection is no longer by product name.** It used to match the
literal string "B70", which made a B580 invisible: on a mixed box the
extra rank fell through to the default selector, and on a B580-only box
nothing was selectable. Ranks now take the discrete Arc GPUs in
enumeration order, and each rank prints the card it got:

```
  rank 2 -> GPU 2/3: Intel(R) Arc(TM) B580 Graphics
```

Read that line **before believing any multi-GPU number**. It carries the
card's PCI address where the driver exposes one, and which physical card
a rank gets was changed on 2026-09-13 (audit finding F5) and has been
reasoned about, not run. A rank whose device does not exist now FAILS
rather than quietly falling back to the default selector -- which used to
put every rank on the same card while the collectives still connected,
i.e. a run that looked like it worked.

On a mixed box the layer split depends on which rank landed where. `GRIMOIRE_DEVICES=0,2,1` remaps rank to device index when
the driver's order is not the one you want.

**N cards, uneven split.** `tools/nrun.sh` replaces the two-card
launchers for anything other than exactly two:

```bash
GRIMOIRE_PP_LAYERS=30,30,12 tools/nrun.sh pp   renderD128,renderD129,renderD130 1800 pp3   /grimoire/bin/grimoire -m /models/<dir> --proj fp8 -p '...' -n 64
```

`GRIMOIRE_PP_LAYERS` is one positive count per GPU, summing to the
model's layer count, and it is REQUIRED beyond two cards — the engine
refuses to guess. **Give a smaller card fewer layers**: pipeline
throughput is set by the slowest stage, so an equal split makes the
small card the ceiling for everything.

Tensor parallel across unequal cards is the wrong tool: TP shards evenly
and synchronises every layer, so the smallest card throttles all of them.
Use PP when the cards differ, TP when they match.

`tools/nrun.sh tp <nodes> ...` exists for the matched case.

## 3. Which format

One card, model fits: `--proj int4` (the W4A8 path, int8 XMX, ~367 TOPS).
Two cards, or precision matters: `--proj fp8` (fp8_e4m3; `mxfp8` measured
slower — 214 GB/s vs 320 in `gemv_decode.cpp`).

All seven formats load and generate on all three architectures
(`bin/test_model_matrix`), so a format that fails on a real checkpoint is
about that checkpoint's shapes, not about the format.

## 4. Before you believe any number

Rule 8, and it has caught every false lead in this project: **a config
that never generated readable text has not been verified.** A PASS on a
numeric self-check is not enough. Run `-p "<prompt>" -n 64` and read it.

`tools/benchy_qwen.sh` now stamps the sha256, mtime and git commit of the
binary it is about to measure, and takes `BENCHY_VERSION` to pin the
harness. Use it, so a number can be traced to a build.

## 5. If a card falls off the bus

`forcewake register returns 0xFFFFFFFF` needs a POWER CYCLE. There is no
software recovery. The known causes, all avoidable:

- a stale bridge `.so` (rule 3 — always `tools/build_bridges_b70.sh`)
- `timeout N docker run` (rule 6 — always go through `tools/tune.sh`)
- a W4A8 tile on a weight with `N % 256 != 0` (rule 4)
- reading `w.payload` after W4A8 freed it (rule 1)

## 6. What is still open

- **An intermittent in-kernel fault on a long MoE prompt. FIXED
  2026-09-17, and it never touched the B70.** Off the card,
  a MoE prompt of 32 tokens or more through the BATCHED prefill faults
  about half the time -- either a SIGSEGV inside JIT-generated code, or
  an argmax over a buffer nothing wrote, which surfaces as "engine
  returned an invalid token". Dense and hybrid fixtures are clean at the
  same length; the routes are clean (`GRIMOIRE_MOE_ROUTE_CHECK=0` says
  `bad=0`); and it is NOT new -- the same reproducer built against
  acadf5f, before any of this week's work, faults 6 times out of 6.

  The cause: prefill()'s MoE branch has an `if (M >= 32)` arm whose every
  path ends in `launch_gemm_xmx`, called DIRECTLY rather than through
  `mm()`. `mm()` has carried the no-matrix fallback for a while; those
  two call sites bypassed it. The branch is now gated on
  `device_can_matrix(q)`. A B70 can run the tile, so nothing on the box
  changes — the only thing the bug broke was the ability to CHECK MoE
  batched prefill off the card, which is why it lasted.

  `bin/repro_moe_prefill_crash` is kept as the regression guard; its
  header has the full evidence and the measured threshold (clean at 31
  tokens, fails at 32).

  Worth knowing anyway: off the card, M >= 32 now takes the plain-SYCL
  pair while the B70 takes the grouped/XMX path. A green MoE run here is
  not a statement about the tile.

- **Concurrency: measure it. 2026-09-17.** Several agents are now served
  in one pass instead of one after another, and every claim about it that
  can be checked off the card has been (`bin/test_batch_decode`,
  `bin/test_scheduler`). What CANNOT be checked here is the only thing
  the change is for. Off-card the batched GEMM runs through a plain-SYCL
  fallback that is far slower than the tile it replaces, so no number
  from this container means anything (rule 8). On the Tower:

  * tok/s aggregate at 1, 2, 4 and 8 concurrent requests, same prompt,
    same model. The claim being tested is that 8 costs about what 1
    costs, and the honest failure mode is that it does not -- attention
    is looped per row, so a long context could make attention the
    bottleneck before the weights are.
  * where that breaks down. Find the width at which aggregate tok/s
    stops rising; that is the real `GRIMOIRE_MAX_BATCH` for this box.
  * **batching versus speculation.** A drafter turns batching off today.
    MTP at ~2 accepted/step against 8-way batching is a genuine trade and
    the repo holds no measurement either way. Do not assume; run both.
  * VRAM. Eight slots is eight KV caches. Check what `--ctx` actually
    fits alongside the model before recommending a slot count.

- **Batching plus speculation, together.** They are mutually exclusive
  right now and the refusal is explicit. A verify batch and a
  multi-sequence batch both want the same M rows of the same prefill
  path, so combining them is a scheduling question -- how many draft
  tokens for how many sequences -- not a kernel one. Worth doing only if
  the measurement above says speculation wins at the concurrency Ian
  actually runs.

- **Coverage, 2026-09-16.** `bin/test_model_matrix` now drives EIGHT
  architectures x 7 formats: dense, moe, hybrid, k2-horizon, **muse**,
  **parallel-ffn**, gemma4 and gemma4 at head_dim 512.  Muse and the
  Agnes parallel-FFN fold had never been executed off the card; adding
  the fixtures found three Muse format assumptions and confirmed the fold
  was correct.  See CLAUDE.md rule 14.

- **Batched prefill under TENSOR parallel.** TP declines the batched path
  outright, so a TP run processes its prompt a token at a time. The
  capability matrix says so, and says not to benchmark it as prompt
  processing. Making it real means all-gathering every projection over
  the whole token batch, not just one row, and an expert-parallel
  all-reduce for the routed MoE half. None of it can be executed off the
  card: the batched path is XMX from end to end, and a device without
  matrix hardware cannot run a single one of its kernels, so this is
  work to do ON the Tower, not before it.
- **F07**: single-process `GRIMOIRE_PIPELINE` puts every kernel on
  device 0's queue. It is numerically correct, but GPU1's weights cross
  the link every token. Making it real means threading a per-layer queue
  through every launcher — architectural, not a patch. Use the
  multiprocess launchers in section 2 instead.
- **Single-process cross-device.** Rule 7: the original failure was
  measured over USB4 and has not been retried on OCuLink. That is a
  hardware experiment, not a code change.
- **Everything off-card is green as of 2026-09-17.** All SEVEN gates pass
  on an OpenCL CPU device: test_k2_kernels, test_k2_e2e,
  test_gemma4_prefill (8/8 batched == sequential), test_qwen4_exp_e2e,
  test_model_matrix (9 architectures x 7 formats), test_spec_e2e, and
  test_parallel_e2e (50 matches, 0 failures).  The k2-horizon cell this
  file previously called a Tower-only Intel-runtime crash was the
  over-reporting ext_intel_matrix aspect -- see CLAUDE.md rule 15 -- and
  passes now.

  Two of those gates were BUILT and never RUN by `preflight_b70.sh`:
  test_gemma4_prefill and test_qwen4_exp_e2e are in its list now.  A gate
  that nothing runs is the same gap as no gate (rule 14).

- **Qwen4-Exp / Qwen3.8-Flash-Next runs. DONE 2026-09-17, off-card.**
  HyperConnections (a residual stream hc_count streams wide), QSA (an
  MQA indexer that scores mean-pooled key blocks and attends to the
  top-k plus the incomplete block) and the PLE n-gram embedding (a
  20-million-row table that lives in HOST memory by design -- the one
  exception to VRAM-only in this engine) all execute, single process,
  with a batched prefill that is token-identical to sequential decode.

  `bin/test_qwen4_exp_e2e` proves each mechanism is LIVE rather than
  merely present, by an A/B against the SAME WEIGHTS with one thing
  turned off in config.json only.  Read `QWEN4-EXP-2026-09-16.md`'s
  bottom section before touching any of it: three things the first pass
  got wrong are written down there, and each of them was fluent.

  Still refused BY NAME, not approximated: TP, PP, and an MTP drafter.
  On the Tower this needs the real checkpoint, which no one here has
  been able to download (huggingface is blocked from the work
  container), and it has never produced a number.

- **gemma-4 at its real head_dim. DONE 2026-09-15, off-card.**
  `google/gemma-4-31B-it` loads at head_dim 512.  The four flash kernels
  are templates on their accumulator width, instantiated at 16 and 32:
  every head_dim <= 256 keeps the exact 16-slot kernel it always used, so
  no model that ran before needs re-measuring, and only a wider head
  reaches the 32-slot one.  `bin/test_model_matrix` carries `gemma4-hd512`
  as a full row (all 7 formats, ALL PASS) and moves the refusal case to
  head_dim 1024.

  An earlier note here said the blocker was that `attention.cpp`'s loops
  run to `MAX_DPL` unconditionally while `prefill.cpp` guards its slots.
  That was never true -- both files guard, and the guards predate the
  commit that claimed otherwise.  The real cost was the private array
  `float acc[MAX_DPL]`, sized by the constant rather than by the head.

  What the Tower still owes: register pressure and throughput at dpl 32
  (rule 8 -- no number came from a CPU device), and SLM headroom, which is
  a hard limit.  The batched kernels take `head_dim * KT` floats of local
  memory twice, i.e. 64 KB per work-group at head_dim 512; the CPU device
  reports 256 KB and cannot fail it, a Xe work-group can.  Tile `KT` for
  wide heads if a batched launch is refused for local memory.
- **gemma-4 batched prefill. DONE 2026-09-15, off-card, ON by default.**
  `prefill_gemma4()` is `forward_gemma4()`'s sandwich graph batched --
  `k_eq_v`, the always-on `v_norm`, GeGLU, both rotations, and the
  per-layer sliding window through `launch_dflash2_block_attention`,
  which computes each query's own window bound. Projections take the same
  tuned dispatch as the generic prefill: W4A8 tile (m16 below 17 tokens,
  rule 5; guarded on `N % 256`, rule 4), MXFP4 dense bridge, and
  `gemv_any` for a payload-free converted weight (rule 1).

  `bin/test_gemma4_prefill` asserts batched == sequential token for token
  at bf16/fp8_e4m3/int8/mxfp4 and at head_dim 512, and asserts the batched
  path actually RAN before it compares anything -- without that a path
  that declined would make both sides the same sequential run. It caught
  three bugs that were all FLUENT when wrong: a missing logits tail (the
  caller takes the first generated token from `s.logits`), a GeGLU call
  that aliased its input, and a qk-norm `weight_offset` defaulting to
  Qwen's `(1 + w)`. `GRIMOIRE_GEMMA4_SEQUENTIAL_PREFILL=1` forces the old
  path for an A/B.

  **First thing to run on the card**, because the CPU device could not:
  the W4A8 and MXFP4 branches are BRIDGES, null off-card, so the gate
  above exercised the plain-SYCL fallback and not one line of them. Run
  `bin/test_gemma4_prefill` on the B70, where they load. A verify batch,
  PP and TP still decline and fall back -- unwired, not broken.
