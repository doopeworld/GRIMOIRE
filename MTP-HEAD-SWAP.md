# Ornith MTP head swap: shisa-ai distilled head is WORSE than native (2026-09-01)

Tested the hypothesis that GRIMOIRE is using the "wrong" Ornith model,
because NInfer serves a special `Ornith-1.5-35B-A3B-NInfer` artifact whose
base is `shisa-ai/Ornith-1.5-35B-A3B-MTP` rather than the stock
`ornith-ai/Ornith-1.5-35B-A3B` we convert from.

## Result: the swap does not help. Keep the native head.

Same engine, same flags (W4A8 + MTP k=3 + EXACT_VERIFY), n=128, GPU0:

| head | acceptance | committed/step | TG |
|---|---|---|---:|
| native (ornith-ai, what we ship) | 79/120 = 65.8% | 2.55 | **106.9** |
| shisa-ai distilled | 72/124 = 58.1% | 2.26 | 93.6 |

The distilled head is 12% SLOWER end to end. Generation stays coherent
either way, which proves nothing -- speculation is always verified by the
target model, so only acceptance is a signal. 58.1% (vs ~0% for a corrupted
head) confirms the un-stacking below is correct.

## The target weights are IDENTICAL -- we are not using a wrong model

shisa-ai's `mtp-merge-manifest.json`: `linked_files: 16`, `rewritten_shards: 1`,
`native_mtp_removed: 785`, `mtp_count: 19`. Only the MTP head is replaced;
the transformer, embeddings, lm_head, tokenizer and vision tower come from
the same official BF16 Ornith-1.5 checkpoint we already use. Both compiled
.b70 files come out byte-identical in size (19,649,167,040) and tensor count
(1478) -- the swap changes values, not layout.

## Layout difference (why a converter step is needed)

- native: 785 tensors, one per expert
  (`mtp.layers.0.mlp.experts.<e>.{gate,up,down}_proj.weight`)
- shisa: 19 tensors, experts stacked `[E,2I,H]` and `[E,H,I]`

`qwen35_loader.cpp` handles stacked experts for the MAIN model
(`experts.gate_up_proj`, line 325), but the MTP head is loaded separately in
`grimoire.cpp:2300` by explicit per-expert name and has NO fused fallback.
`tools/unstack_mtp_head.py` converts shisa's stacked export into the native
per-expert layout (gate = first `moe_inter` rows, up = next, per
qwen35_loader.cpp:371-375). Output verified: 785 tensors, names and shapes an
exact set-match to the native head.

Reproduce:

    python3 tools/unstack_mtp_head.py     # writes model-mtp-native.safetensors
    # merged dir: relative symlinks to the base shards + rewritten index.json
    # remapping all 785 mtp.* names to model-mtp-native.safetensors
    bin/b70-compile-model /models/Ornith-1.5-35B-A3B-MTPFIX \
                          /models/Ornith-1.5-35B-A3B-MTPFIX/model-v2.b70

## Unresolved discrepancy -- do not trust the published acceptance numbers

shisa-ai reports the NATIVE head at 21.7-32.19% acceptance and their
replacement at 60.51% / 3.078 tokens per round. We measure the opposite
ordering, and our measurement of THEIR head (58.1%) lands close to their
published 60.51% while our measurement of the NATIVE head (65.8%) is double
their published native figure. Either the metrics differ (ours is per-draft-slot
over one prompt; theirs is an eval suite), or the ornith-ai checkpoint we
pulled on 2026-08-23 already ships a better native head than the one they
evaluated. Not resolved. Treat both published figures as non-comparable to
ours.

The NInfer card's "~80%, ~3.4 tokens/round" is worse than non-comparable --
it is explicitly the OFFICIAL Qwen3.6-35B-A3B head's number, quoted as an
expectation for Ornith, not a measurement of it.

## Consequence for the ~200 TG target

The head is not the lever. At 2.55 committed/step we are already near
shisa's claimed 3.078 for their own head, and the measured verify slope
dominates: see [[grimoire-verify-wall]] -- each extra verified token costs
2.95 ms (MoE expert fan-out), so even a perfect drafter caps out near
130 TG at the current verify cost. Fix MoE verify batching first.
