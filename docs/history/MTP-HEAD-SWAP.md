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


## Qwen: opposite bottleneck (measured 2026-09-01, n=128)

    decode(1)          33.2 ms  (30.1 tok/s, no MTP)
    verify(M=4) k=3    47.5 ms/step, 2.74 committed, 51.5 TG, accept 82/113
    verify(M=8) k=7    54.0 ms/step, 3.05 committed, 46.9 TG, accept 86/128
    fit: verify(M) ~= 42.6 + 1.63*(M-1) ms

Qwen slope 1.63 ms = 4.9% of its decode -> ACCEPTANCE-limited.
Ornith slope 2.95 ms = 34.9% of its decode -> VERIFY-limited.

On Qwen a better drafter pays off directly: at k=3 (step 52.9 ms fixed),
2.74 committed = 51.5 TG, 3.4 -> 64 TG, 4.0 perfect -> 76 TG. k=7 is
currently worse than k=3 because acceptance decays with depth faster than
the cheap extra verify is repaid.

But no published artifact supplies a better head. NInfer Build 1
(Qwen3.8-27B) uses the head included in the base checkpoint -- ours.
mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF is a GGUF requantization of the same
ornith-ai checkpoint. ALWAYS check base_model before pulling one.

And vLLM reaches 58.1 TG on this model with this same native head vs our
51.5 -- speculation multipliers are close (1.71x vs 1.78x), the gap is mostly
baseline decode (30.1 vs 32.7). Fix that first; it needs no new model.


## CORRECTION: Ornith speculation is 6% below break-even, not doomed

Measured GRIMOIRE_MTP_ROUTE_DIAG at M=4: ~860-950 unique of 1280 selections
across 40 layers (1.49x reuse, ~22 unique experts/layer vs 8 in decode).
Union batching ALREADY EXISTS (grimoire.cpp:6224+, the M<32 branch). There is
no batching win; 22 of 32 is the model's routing.

GRIMOIRE_MTP_DRAFT_VOCAB=131072 was never applied to Ornith. It is free:
draft 2.53 -> 1.85 ms/step, acceptance unchanged (79/120), TG 106.9 -> 110.2.
65536 is worse (2.24 committed, 96.3 TG).

    decode(1)                      8.45 ms
    step = verify 19.34 + draft 1.85 + snapshot .33 + commit .51 = 22.8 ms
    break-even = 22.8 / 8.45      = 2.70 accepted/step
    we accept                       2.55       <- short by 6%

Above break-even: 2.70 -> 118 TG, 3.5 -> 154, 4.0 -> 175. DFlash2 reports
4.02 accepted/round -> ~150-190 TG, which beats vLLM 138.7 on this card.
THE DRAFTER IS THE LEVER. An earlier note in this file said otherwise; wrong.

vLLM Ornith baseline is 33.7 TG, ~4x slower than our 125 -- that is why their
speculation multiplier is 4.12x and ours cannot be. Their 138.7 speculative is
11% above our 125 non-speculative. NInfer 593 on a 1792 GB/s 5090 is 199
B70-equivalent vs our 125 = 1.6x, not 3x.
