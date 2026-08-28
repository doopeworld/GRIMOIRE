# GRIMOIRE handoff — Ornith original DFlash

Date: 2026-08-28

## Current status

Original `z-lab/Qwen3.5-35B-A3B-DFlash` loads and runs end-to-end with
`ornith-ai/Ornith-1.5-35B-A3B` on Tower GPU0 (`renderD128`). Output is coherent
and exits cleanly, but performance is unacceptable: the latest result is only
22.6 tok/s and 2.21 committed tokens per verifier step versus the approximately
125.5 tok/s non-speculative baseline.

Do not claim this as a successful DFlash result. Do not run more guessed tuning
variants. The next session must find the first numerical divergence from the
Python reference before doing more performance work.

GPU1 was not used. All tests used `tools/tune.sh` with `GPU=gpu0`, resolving to
`renderD128`. No containers remain running.

## Updates and confirmed fixes

- Original DFlash and DFlash2 are detected separately; the original six-layer
  forward no longer requires DFlash2 convolution/selector weights.
- Original DFlash weights now remain BF16 as shipped. Previously target
  `--proj mxfp4` incorrectly quantized the draft to 0.24 GiB. Correct draft
  device size is 0.77 GiB.
- DFlash RMSNorm and Q/K norm use raw Qwen weights (`weight_offset=0`). Target
  MuseGlimmer keeps its existing baked `1 + weight` convention.
- Fixed invalid B70 launches: flattened batched embedding and SwiGLU, tiled
  batched flash decode to a maximum 512-item workgroup, and changed wide
  verifier SwiGLU/sigmoid kernels to padded 2-D launches with local width 256.
- Added opt-in `GRIMOIRE_DFLASH_TRACE` and
  `GRIMOIRE_PREFILL_HOST_PROGRESS` diagnostics.
- Fixed a confirmed vocabulary-projection throughput bug. Draft and verifier
  MXFP4 `lm_head` now use one M-row GEMM instead of 15 and 16 separate full
  vocabulary GEMVs per speculative step.
- Full AOT SYCL build passes.

Modified files: `src/attention.cpp`, `src/grimoire.cpp`, `src/kernels.hpp`,
and `src/prefill.cpp`.

## Exact results

Clean run before batched `lm_head`:

```text
dflash ok (DFlashDraftModel, 0.77 GiB device)
pp 21 tokens in 1983 ms -> 10.6 tok/s
DFlash(k=15) tg 64 tokens in 3267 ms -> 19.6 tok/s
DFlash steps 30, 2.17 committed/step, draft accepts 35/65, rollbacks 30
```

Latest clean run after batched draft and verifier `lm_head`:

```text
dflash ok (DFlashDraftModel, 0.77 GiB device)
pp 21 tokens in 1955 ms -> 10.7 tok/s
DFlash(k=15) tg 64 tokens in 2827 ms -> 22.6 tok/s
DFlash steps 29, 2.21 committed/step, draft accepts 35/64, rollbacks 29
```

Latest log: `/tmp/grim-orn-dflash-batched-result.log`.

The projection correction improved 19.6 to 22.6 tok/s but did not materially
change acceptance. It was a real throughput defect, not the main blocker.

## Reference facts already checked

- Correct pairing is plain Ornith plus `z-lab/Qwen3.5-35B-A3B-DFlash`.
- The ultimatechris reference reports block 16, accepted length 6.1 on GSM8K /
  7.7 on HumanEval, and 202.5 to 470.2 tok/s on two A100s with a BF16 target.
- Original DFlash samples the 15 mask rows, not the anchor row. GRIMOIRE uses
  rows 1 through 15, matching `ref/dflash_speculator.py`.
- Query positions are sequential after accepted context.
- Target taps `[1,6,11,16,22,27,32,37]`, FC then hidden RMSNorm, residual flow,
  and NeoX RoPE were reviewed against `ref/qwen3_dflash.py`.

## Unresolved findings

- Acceptance is 2.21 instead of the reference 6.1 to 7.7. This is the main
  blocker. Kernel tuning cannot create the missing accepted tokens.
- The reference uses a BF16 target; GRIMOIRE uses aggressive target MXFP4
  (about 17.9 GiB versus roughly 73 GiB). The draft was trained on BF16 target
  states/logits. This is plausible but unproven.
- DFlash KV caches are FP8 E4M3; the reference likely uses BF16 KV. This
  precision difference is also unisolated.
- The startup W4A16 grouped-GEMM warning is from the legacy loader. The MXFP4
  grouped library and its M16 symbol exist. Do not call the warning the cause
  without timing evidence.

## Next steps — no blind benchmarks

1. Create a one-step comparison using one fixed prompt and checkpoint. Dump the
   eight target taps, projected/normalized context, first-layer Q/K/V, final
   draft hidden rows, and 15 draft token IDs from GRIMOIRE and Python.
2. Identify the first diverging tensor and fix only that divergence. Repeat the
   same one-step comparison; do not launch full-generation tuning variants.
3. If BF16 forward intermediates match, isolate DFlash KV BF16 versus FP8, then
   target BF16/MXFP4 compatibility as separate experiments.
4. Only once acceptance approaches the reference range, profile draft versus
   verifier and optimize the measured dominant stage.
5. Final validation must use GPU0 and report exact speed and acceptance, then
   update this handoff, commit, and push GitHub.
