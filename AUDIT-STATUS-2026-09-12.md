# Audit status — branch `codex/b70-audit-fixes-20260911`

The 2026-09-11 audit inspected `qwen-tg-restored` (390c9d5), `pp-muse`
(7ff09dc), `tp-weight-sharding` (6c959e0) and `main` (dcc549c). It did not
inspect this branch, and said so:

> "The separate local quantizer work at `c99d9b1...` is not the remote code
> above. Its changes must not be attributed to these branches."

`c99d9b1` is one of Codex's commits on this branch. Several findings are
therefore already closed here. Each row below was checked against the code
at HEAD, not assumed. Line numbers are this branch's.

**Nothing on this branch has been through a compiler.** There is no SYCL
toolchain and no B70 in the container the work was done in. Host tests pass
(`make test`, 10 suites); every GPU path is unbuilt.

## Status

| # | P | Finding | State here |
| --- | --- | --- | --- |
| F01 | 1 | decode cursor after batched prefill | **fixed** (Codex) |
| F02 | 1 | context capacity not enforced | **fixed** (Codex) |
| F03 | 1 | Ornith BF16 MTP experts vs loader | **fixed** (Codex) |
| F04 | 1 | DFlash2 selector never executed | **fixed 2026-09-12** |
| F05 | 1 | draft architectures treated as interchangeable | **fixed 2026-09-12** |
| F06 | 1 | Muse ignores pipeline layer ownership | **fixed 2026-09-12** |
| F07 | 1/2 | single-process pipeline does not execute on GPU1 | **open, documented** |
| F08 | 1 | DAG switch drops in-order everywhere | **fixed** (Codex) |
| F09 | 2 | TP/PP capability shortfall | **reported 2026-09-12** |
| F10 | 1/2 | server corrupts valid prompts | **fixed** (Codex) |
| F11 | 2 | MTP context and precision | **fixed** (Codex) |
| F12 | 2 | Muse drafter copies layer 0's K-norm everywhere | **fixed** (Codex) + A/B instrumented 2026-09-12 |
| F13 | 2 | failed speculation looks like an empty success | **fixed** (Codex) |
| F14 | 2 | build/benchmark evidence not tied to a binary | **fixed 2026-09-12** |

## Evidence for the closed items

**F01** — `set_cursor(int p)` writes `*d_pos = p` and `*d_seq_len = p + 1`
as a device command, never a copy from a host temporary. Used at all 11
transitions including the end of batched prefill, `restore_prefix` and
`commit_spec_prefix`. The audit's repair asked for exactly this: one
definition of the next-step cursor invariant.

**F02** — `check_token` throws `"context capacity exhausted"` when
`pos >= max_seq` and `"token outside model vocabulary"` for a bad id.

**F03** — `mtp_fmt` returns the native format only for a native,
non-RAW tensor and BF16 for everything else, so a raw BF16 safetensor
keeps BF16 instead of inheriting the target's projection format. Every MTP
sub-weight goes through it; the router and `shared_expert_gate` are pinned
to BF16 separately. The MXFP4-only check the audit cites at line 995 is
`concat4_native_mxfp4_t`, a different helper — not the MTP expert path.

**F08** — `GRIMOIRE_DAG` no longer exists as a queue-property switch.
`dag` is hardcoded false with "All model paths require in-order
submission", and `queue_props()` always returns `in_order()`.

**F10** — `test_http` covers strict JSON, `\uXXXX`, text-content arrays,
token-aware Harmony and split UTF-8.

**F11** — the lifetime bug is gone: `mtp_warm` uses `set_cursor` and the
device cursor pointer, not stack locals. Prompt context IS filled, by a
batched pass in prefill that runs the `pre_e`/`pre_h`/`fc`/`k_proj`/
`v_proj` chain over the whole prompt and appends M positions with
`launch_kv_append_batched`. `GRIMOIRE_MTP_WARM` no longer exists because
the fill is no longer optional.

## What was fixed on 2026-09-12

**F04 — the selector now runs.** Both kernels
(`launch_dflash2_selector_edges`, `launch_dflash2_path_walk`) had existed
in `ops.cpp` since the initial commit and were never called. They were
checked against vLLM's own source before being wired — `_score_edges` in
`vllm/model_executor/models/qwen3_dflash2.py` and `_selector_walk_kernel`
in `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`. The walk is a
sequential greedy step conditioned only on the previous pick, NOT Viterbi
or beam. `tests/test_dflash2_selector.cpp` replays both kernels lane for
lane: 1536 scores, max diff 0.000e+00.

Three things that would have been silent: `topk16_rows` assumes a packed
row while the draft logits can be padded; with a reduced draft head the
candidate ids index that head, not the vocabulary the codebooks use; and
`selector_top_k` is read from the draft config and refused above 16 rather
than silently running a wider selector at 16.

**F05 — draft architecture validated before use.** `dr()` returns an empty
`TensorRef` for a missing name and `.t.shape[0]` was read straight off it,
so a drafter with a different layout hit out-of-bounds host indexing
instead of a compatibility error. The DaoCloud MoE draft has no
`layers.0.mlp.gate_proj.weight` at all. Every tensor the geometry derives
from is now checked for existence and rank, head counts are checked for
divisibility, and the error names the capability. A config carrying
`aux_hidden_state_layer_ids` instead of `target_layer_ids` is reported
rather than silently falling back to defaults.

**F06 — Muse honours pipeline stages.** `forward_muse` embedded, walked
all `cfg.n_layers` and ran the head unconditionally while `build()` loads
only `[pp_begin, pp_end)`. It now mirrors `forward()`: rank 0 embeds and
applies the scaleless input norm, later ranks receive the residual stream,
the loop runs the local range, non-final ranks send on. `forward_dag` has
the same all-layers shape and now refuses under pipeline.

**F09 — capability matrix printed at load.** Speculation silently does not
load under TP or PP; TP prefill silently falls back to token-at-a-time.
`build()` now states the parallel mode and layer range, whether speculation
is active and which kind (including whether the DFlash2 selector is
running), whether prefill is batched or the sequential fallback, and the
norm convention. The TP prefill line says outright not to benchmark it as
prompt-processing throughput.

**F12 — the norm default was already right; the A/B was missing.**
Codex's `c771464` had already changed the Muse context-K norm from layer
0's weight for all five draft layers to each layer's own checkpoint
weight, which is the default the audit asked for. What that commit also
did was delete the measurement that motivated the old behaviour, and that
measurement is not a guess: against the running Fusion reference, the
effective context-K weight is identical across all five layers
(pairwise cos 1.0000, rms 1.08547) and equals
`layers.0.self_attn.k_norm.weight`, while the checkpoint's five tensors
genuinely differ (rms 1.085, 1.349, 0.895, 1.381, 0.955). Per-layer
weights put Grimoire's context K at cos 0.93-0.97 against Fusion;
layer 0's everywhere raises every layer to ~0.99.

So either Fusion collapses the norm and matching it buys acceptance
against a reference that is itself wrong, or Grimoire's context-K path
is fed something the per-layer weights then expose. Nobody has run the
measurement that separates those. The evidence is now back in the code,
the faithful weights stay the default, and the reference-matching
behaviour is `GRIMOIRE_MUSE_KNORM_LAYER0=1` and named in the capability
matrix (`draft ctx-K`), so no acceptance number can be quoted without
saying which produced it.

The audit also asked for an A/B acceptance comparison, and there was
nothing to measure it with: `generate_tokens` computed `accepted` per
round and threw it away. `SpecStats` (steps / drafted / accepted) is now
an optional out-parameter on `GenerationOptions`, filled before the emit
loop so a cancelled or stop-terminated round still counts — otherwise
the rate biases upward on exactly the requests that end early.
`GRIMOIRE_SPEC_STATS=1` prints accepted-per-step per request. This is
the number tok/s cannot give: a cheaper draft and a more accurate draft
both move tok/s, and only one of them moves this. It applies to every
speculation A/B, not just F12 — MTP vs DFlash2 and selector on/off are
the same question. `tests/test_generation.cpp` pins the arithmetic over
every depth and rejection prefix (anchor never counted as a draft, MTP
totals cross-checked against the engine's own call count).

**F14 — a failed build now fails.** `build_b70.sh` caught every failure
with `|| { echo ...; }`, the one shape `set -e` does not fire on: it
printed "GRIMOIRE BUILD FAILED", continued, and exited 0. Anything
downstream then measured whatever older binary was still in `bin/` and
attributed the result to the source tree. Both required targets now set
`REQUIRED_FAILED` and the script exits 1 saying `bin/` may be stale.
Auxiliary targets stay warn-only. `benchy_qwen.sh` had the same class of
problem on the measurement side: no `pipefail`, so a crashed benchy read
as "no output" rather than as a failure, and an unpinned
`uvx llama-benchy`. It now fails loudly, takes `BENCHY_VERSION` to pin
the harness, and stamps the sha256, mtime and commit of the server
binary it is about to measure.

## The A/B this leaves to run on the Tower

```
GRIMOIRE_SPEC_STATS=1 tools/tune.sh ... -p "<fixed prompt>" -n 128
GRIMOIRE_SPEC_STATS=1 GRIMOIRE_MUSE_KNORM_LAYER0=1 tools/tune.sh ... (same)
```

Same prompt, same depth, both modes. Compare accepted/step, and read the
text in both — per rule 8 a number from a config that never generated
text is not evidence. If per-layer wins or ties, delete the compatibility
mode. If layer 0 wins materially, the finding is in Grimoire's context-K
path, not in the norm choice, and the flag is a marker for where to look.

## Still open

**F07** — the audit says the single-process pipeline "does not execute
there". True, with one correction: the context is SHARED across both
devices, so the device-1 pointers are legal and this runs and is
numerically correct. What it costs is that GPU1's weights are read across
the link every token rather than out of local VRAM. Making it real means
threading a per-layer queue through every launcher (`gemv_any`, `mm`,
`mmb`, all of `ops.cpp`), which take the engine's single member queue
implicitly. That is architectural. The load path now prints what it
actually does and points at `tools/pp2run.sh`.

That is the only finding still open.

## Retraction the audit did not catch

Both commits that judged DFlash2 — `8e2fbc8` (2026-09-01, "97.6 TG, still
LOSES to 125 TG plain decode") and `f677add` (2026-09-02, "LOSES to MTP --
drafter too expensive") — predate any call site for the selector kernels.
What they measured was DFlash2 drafting by per-position argmax with the
codebooks loaded and unused. Those numbers are void as evidence about
DFlash2, and CLAUDE.md's "lower-acceptance DFlash2 sidecar" steer derived
from them. See the retraction in CLAUDE.md.

Note the part of the old finding that may still survive: MTP is one extra
head, DFlash2 runs a five- or six-layer drafter. The selector changes the
acceptance side of that trade, not the draft cost. The measurement that
settles it is accepted-tokens-per-step against draft-cost-per-step, with
the selector live, on the same prompt.
