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
| `bin/test_model_matrix` | 3 architectures × 7 projection formats |
| `bin/test_parallel_e2e` | PP and TP give the SAME tokens as one process |
| `bin/test_spec_e2e` | speculation gives the SAME tokens as plain decode, incl. dual GPU, MTP and DFlash |

**Read the `hybrid` rows of `test_spec_e2e` first.** Off the card they say
REFUSED (no batched verify there, and the engine will not speculate
without one). On your B70 they run for real and must say `identical at
K=1,2,3,5`. If either says `CHANGED THE OUTPUT`, stop: that is
speculation on a Qwen3.5/Ornith-shaped model not reproducing the model's
own output, which is the path the verified 49.8 TG MTP recipe uses.
| generate | real model, real prompt — **you read the output** |

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

**Speculation now works on dual GPU.** Add `GRIMOIRE_MTP=1` (and
`GRIMOIRE_MTP_K=3`) to either launcher. Under PP the last stage hosts the
MTP head and drives the drafting for the whole pipeline; under TP the head
is replicated and every rank drafts identically. Verified token-for-token
against single-process plain decode — speculation is exact, so if it ever
changes the output that is a bug, not a trade.

DFlash is still single-GPU only: its drafter needs aux hidden states from
target layers that PP puts on different ranks. So DFlash on one card, MTP
on two. The load banner prints which is live — read it.

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

- **F07**: single-process `GRIMOIRE_PIPELINE` puts every kernel on
  device 0's queue. It is numerically correct but GPU1's weights cross
  the link every token. Use the multiprocess launchers above instead.
  Making it real means threading a per-layer queue through every
  launcher — architectural, not a patch.
- Speculation under TP/PP (section 2).
- MoVA's value projection reads its routing table back to the host once
  per layer. Fine at M=1, useless at 4096 tokens. Pack the experts
  expert-major before quoting a K2 prefill number.
