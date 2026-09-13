# Session handoff — 2026-09-13, the DFlash drafter's config

Branch `codex/b70-audit-fixes-20260911` (mirrored to
`claude/codex-b70-audit-fixes-u1adcv`).

This is priority #1 work: `HANDOFF-2026-08-28-ORNITH-DFLASH.md` says the
Ornith drafter commits **2.21 tokens per step against a reference that
reports 6.1–7.7**, that kernel tuning cannot create the missing tokens,
and that the next step is to find the first divergence from the Python
reference.

## The one sentence that matters

**Read the new `dflash config:` / `dflash attention:` / `dflash head:` /
`dflash embed:` lines the loader prints, before anything else.** They
name, in one run, every value the drafter resolved to. Until now none of
them could be seen at all, and every one of them is silent when wrong:
a drafter on the wrong RoPE theta, the wrong mask token, the wrong norm
epsilon, the wrong attention shape or the wrong output head still runs,
still drafts real tokens, and still leaves the verified output correct.
Only the acceptance rate moves. vLLM says so about its own RoPE layout:
*"a mismatch is silent — acceptance collapses but nothing errors and the
output stays correct."*

## What was actually wrong

The draft `config.json` was read with a substring search and `strtol`,
and the rest was hardcoded. Against `ref/qwen3_dflash.py`:

| value | was | reference |
| --- | --- | --- |
| `rope_theta` | `strtol`, top level only, default **1e7** | double, `rope_parameters.rope_theta` or top level, default **1e6** (`set_default_rope_theta`) |
| `rms_norm_eps` | hardcoded 1e-6 (1e-5 for Muse) | the draft config's own |
| `head_dim` | hardcoded 128 | `head_dim`, else `hidden_size / num_attention_heads` |
| `mask_token_id` | top level only | `dflash_config` ∪ `eagle_config`, then top level |
| target taps | `target_layer_ids` only; other spellings warned and ignored | whichever the checkpoint uses |
| sliding window | "layers 0–4 slide at 4096; DFlash2 slides everywhere" | `layer_types` / `dflash_config.use_swa`; a config that says nothing means **full attention** |
| causality | always non-causal | `is_causal`, then `dflash_config.causal`, then "this layer slides" |
| `lm_head` | always the TARGET's | the drafter's own when it ships one, with `d2t` |
| `embed_tokens` | always the TARGET's | the drafter's own when it ships one |

A `strtol` on `"rope_theta": 5000000.0` is a coin flip on the spelling,
and it could not see `rope_parameters` at all. Note the default: where
the old code guessed 1e7, the reference's answer is 1e6 — a 10x error in
every RoPE frequency in the drafter, with no symptom but acceptance.

The rules now live in `include/b70/dflash_config.hpp`, transcribed from
the reference rather than re-derived, and `tests/test_dflash_config.cpp`
pins them against the three checkpoint shapes vLLM's own
`_resolve_layer_attention` docstring names (standard z-lab DFlash, MiMo's
`use_swa`, gemma-4's mixed `layer_types`).

## Things that were silent and are now refusals

Each of these produced a running drafter with bad acceptance:

- the draft's `hidden_size` differs from the target's
- `use_aux_hidden_state: false` (the drafter wants the target's LAST
  hidden state, not the layer taps — a path this engine does not have)
- a sliding layer with no window anywhere in the config
- `layer_types` shorter than `num_hidden_layers`
- **a tap id with no capture point.** Taps are taken at entry to layer
  `id + 1`, so a tap naming the last layer is never written and `fc`
  reads that slice of `target_aux` as it was allocated — uninitialised
  DEVICE memory, straight into the context projection.
- **`fc`'s input width not equal to taps × hidden.** The reference
  raises here; on this engine it is worse than wrong numbers, because
  too many taps reads past `target_aux` on device, which is a
  DEVICE_LOST and a power cycle rather than an exception.
- a reduced draft `lm_head` with no `d2t` to interpret its ids

## A finding that was written down and never applied

`DFlash2Head::block_size` had a twelve-line comment above it recording a
measurement against the running Fusion reference — its draft context
slots for positions 0–63 are 368–431 and its query slots 432–447, so base
368 = block 23 x **16**; the drafter's own `config.json` declares
`block_size: 16`; and at 64 the 80-key draft sequence is one whole page
plus a 16-key partial page that the paged kernel **reads back as zeros**,
so the query rows "see only the 64 context keys and never the bonus
token".

The field was still initialised to 64. The conclusion was recorded and
the value was not changed, so the Muse draft ran with its anchor row
invisible — which is a total acceptance killer, and exactly the class of
silent failure the rest of this work is about.

It now comes from `dflash_config.block_size` (16 when the config names
none), is printed at load as `draft page N`, and `GRIMOIRE_DFLASH_BLOCK`
overrides it so 16 vs 64 is an A/B on the card rather than an argument.
`block_table` is shared with the target's own paged attention, which
pages at 64 and reads only the first `(max_seq+63)/64` entries: a smaller
draft page makes that table longer, never shorter, so this stays in
bounds. **This is the one change here that touches a Muse path with real
measurements behind it, and it is compile-checked only — no Muse
checkpoint runs in this container.**

## What is new

- **The drafter's own `lm_head`, `d2t` and `embed_tokens` are used when
  the checkpoint ships them.** `d2t` is a delta (`target = i + d2t[i]`),
  folded into absolute ids once at load; the selector codebooks, the
  verifier and the caller all speak target ids. Sharing the target's
  head is correct only for a drafter that ships none, which is exactly
  what `_should_share` decides in `ref/dflash_speculator.py`.
- **`GRIMOIRE_DFLASH_LEGACY_SLIDING=1`** restores the old attention
  assumption, so the two can be A/B'd on accepted-tokens-per-step on the
  card. That measurement is the only thing that settles it.
- **The draft forward runs without matrix hardware.** It used to submit a
  `joint_matrix` kernel unconditionally, which the runtime refuses on a
  CPU device — so the entire DFlash forward was unreachable anywhere but
  a B70, which is why none of it had ever executed off the card. On a
  B70 the predicate is false and nothing changes.

## What is now tested, and what that test found

`bin/test_spec_e2e` grew a DFlash section: a miniature drafter directory
(`mini::dflash_draft`) against a miniature dense target, driven through
`GRIMOIRE_DFLASH_MODEL`. Before this, MTP was the only drafter any test
had ever run — DFlash's whole forward (taps, `fc`, hidden norm, context
K/V into the draft's own cache, the 1+N non-causal query block) had
executed only on the Tower.

Running it immediately found the missing no-matrix fallback above.

It also closes a gap that applied to **every** speculative test in this
tree, MTP included: random weights make a drafter that agrees with a
random target about once in `vocab`, so every case ran at **0%
acceptance** and exercised only the rollback, never accept-then-continue.
The new `dflash+forced` case points a drafter at the token the target
repeats most, so it is accepted where the target emits it and rejected
everywhere else:

```
spec: DFlash depth 15 -- 0.60 accepted/step (6 of 64 drafted, 9.4%)
output identical to plain decode
```

That is the first non-zero speculative acceptance any test here has
measured, and the output still has to match exactly.

## Two bugs in the comparison harness itself

`tools/dflash_compare.py` and `GRIMOIRE_DFLASH_DUMP` are what step 3
below depends on. Running them for the first time found that neither
half worked on the model they were written for:

- **The draft stages were never dumped at all.** `generate_tokens()`
  opens with a context-only `dflash_draft(..., context_only=true)` call,
  and the dump used a single "first call" latch — which that call spent
  entirely. The block embedding, per-layer intermediates, final norm,
  logits and draft ids were never written. The two halves latch
  independently now, each set only once its own work has happened.
- **Stages 04–06 were dumped only on the Muse path.** The Ornith path
  projects context K/V per layer instead of through one fused GEMM, and
  dumped nothing. It now stacks the per-layer results into the
  reference's own layouts (`04` = `all_kv_flat`, `[rows][L,2,nkv,hd]`;
  `05`/`06` = `[L][rows][nkv*hd]`), and `tools/dflash_reference_dump.py`
  was corrected to emit `04` in that layout rather than a third one.

Verified end to end here: a miniature drafter writes all 17 tensors,
both halves, and the comparator names the first divergence correctly on
an injected one.

## Verified in a container (OpenCL CPU device, no XMX)

```
make test              13 host suites, incl. the new dflash-config
make test-correctness  2 suites
test_k2_kernels        ALL PASS
test_k2_e2e            ALL PASS
test_model_matrix      (see the run log)
test_parallel_e2e      (see the run log)
test_spec_e2e          DFlash identical at M=4,8,16, shared head and own
                       head + d2t; accept path identical at 9.4%
src/grimoire.cpp       compiles clean
```

## NOT verified — and be precise about this

**Nothing here says the Ornith drafter's numbers were wrong.**
huggingface.co is blocked from this container, so
`z-lab/Qwen3.5-35B-A3B-DFlash/config.json` could not be read: it is
unknown which of the values in the table above actually differed for
that checkpoint. What changed is that the engine now derives them from
the reference's rules instead of guessing, refuses the cases it cannot
interpret, and **prints what it resolved**. One run on the Tower turns
"unknown" into "known", and that run costs nothing.

No acceptance number here is real: a miniature model with random weights
produces noise by construction (rule 8 is untouched).

## Next, in order

1. **Load the real pair on the card and read the four `dflash` banner
   lines.** If any disagrees with the drafter's `config.json`, that is
   the first divergence and it is free to find.
2. **Re-measure acceptance**, and A/B `GRIMOIRE_DFLASH_LEGACY_SLIDING=1`
   against the default. Accepted-per-step, not tok/s — two things move at
   once in tok/s.
3. If acceptance is still far from 6.1–7.7, run the tensor comparison the
   2026-08-28 handoff asks for: `GRIMOIRE_DFLASH_DUMP=/tmp/dfl` writes
   GRIMOIRE's half, `tools/dflash_reference_dump.py` the reference's, and
   `tools/dflash_compare.py` names the first stage that disagrees.
4. The remaining untested hypothesis from that handoff is unchanged and
   needs two cards: the drafter was distilled against a **BF16** target
   and GRIMOIRE runs the target at 4 bits, because Ornith at FP8 is
   ~36.5 GiB and does not fit one 31.9 GiB card. Testing it means DFlash
   under PP, which is item 3 of the still-open list in
   `HANDOFF-2026-09-12-VERIFICATION.md`.
