# GRIMOIRE handoff — Ornith original-DFlash implementation

Date: 2026-08-28

## Current priority

Implement block speculative decoding for Ornith-1.5-35B-A3B using the
**original DFlashDraftModel** from `z-lab/Qwen3.5-35B-A3B-DFlash`.

This is the drafter used by `ultimatechris/Ornith-1.5-35B-A3B-DFlash-SGLang`:
block size 16, 6.1 accepted tokens on GSM8K and 7.7 on HumanEval, with measured
202.5 -> 470.2 tok/s (2.32x) on 2x A100. It is a better first target than the
already-downloaded `jzinno/...DFlash2`, which measured 4.02 accepted tokens.

## Updates completed

- Downloaded the exact original drafter to the Tower:
  `/mnt/storage/Models/Qwen3.5-35B-A3B-DFlash` (771,819,674-byte safetensors).
- Added `GRIMOIRE_DFLASH_MODEL`; `GRIMOIRE_DFLASH2_MODEL` remains a compatibility
  alias.
- The loader now detects original `DFlashDraftModel` versus
  `DFlash2DraftModel` from the tensor set.
- DFlash2-only grouped-conv and candidate-selector tensors are optional, so the
  original z-lab model loads instead of failing on absent tensors.
- Allocated independent FP8 K/V caches for all six draft layers.
- Recorded the original model's attention pattern: layers 0-4 use a 4096-token
  sliding window; layer 5 uses full attention. DFlash2 keeps all six sliding.
- Release logic now frees every draft K/V cache.
- Expanded target-tap storage to token-major `[max_seq, 8, hidden]` and added
  graph-safe kernels for recording all eight residual taps during both decode
  and batched target verification. The full SYCL build passes.
- Added a WIP original-DFlash forward in `src/grimoire.cpp`: persistent draft
  scratch, target-context ingestion into six draft KV caches, a 16-row
  `[bonus + 15 masks]` forward, batched `lm_head`/argmax, and integration with
  the existing verifier/rollback loop. `kSpecBatch` is now 16.
- Fixed the first WIP runtime failure: speculative rollback buffers were only
  allocated for MTP. They are now allocated for either MTP or DFlash.
- Fixed the next identified failure in `launch_embed_batched`: its 2-D range
  used a 2048-wide dimension, exceeding B70's 1024 per-dimension limit. The
  equivalent token/hidden indexing is now flattened into a 1-D range.

## Important status: runtime retest required

The original DFlash model loads, but speculative decoding is **not yet proven
working**. The latest full run exposed the invalid embed launch described
above. That launch is fixed and the full build passes, but per user instruction
the fix has not been run on a GPU. There is no valid DFlash acceptance or TG
result yet. Treat the committed forward as unfinished WIP.

## Validation and findings

Build command passed:

```bash
docker run --rm --entrypoint bash \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire \
  my-vllm-xpu:latest -lc \
  'bash /grimoire/tools/build_grimoire_only_b70.sh /grimoire'
```

Original-DFlash load test passed on GPU0 and printed:

```text
dflash       ok (DFlashDraftModel, 0.28 GiB device, target taps + draft KV ready)
prompt: 11 tokens
tg 1 tokens in 9 ms -> 110.6 tok/s
```

Latest WIP result on GPU0 (`renderD128`), using the safe runner and MXFP4:

```text
dflash       ok (DFlashDraftModel, 0.24 GiB device, target taps + draft KV ready)
prompt: 21 tokens
Xe2 grouped GEMM unavailable; using exact fallback
terminate called after throwing an instance of 'sycl::_V1::exception'
what(): The number of work-items in each dimension of a work-group cannot exceed
        {1024, 1024, 1024} for this device
```

The failure occurs on the first speculative step. The process exited `134`;
there was no generated text, acceptance count, or TG measurement. Full log:
`/tmp/grim-orn-dflash-mxfp4-result.log`.

Root cause found after the run: `launch_embed_batched(16, 2048)` used a 2-D
`sycl::range` whose hidden dimension exceeded the B70 limit. It is now a
flattened 1-D launch with identical indexing. The full build passed after this
fix. Runtime validation is deliberately deferred to the next session.

An earlier WIP run reached the same point but failed in `snapshot_recurrent`
with a null-copy argument. That allocation bug is fixed. Its log is
`/tmp/grim-orn-dflash-e2e16.log`.

Verifier graph experiment (do not repeat):

- `verify(1)` direct: 184.7 ms / 16 = 11.54 ms.
- Graph replay itself: about 9.42 ms, confirming roughly 2.1 ms submission cost.
- Rebuilding/finalizing the graph every step adds about 1.2 ms at M=1.
- On MTP k=7, per-call graph capture regressed 62.7 -> 55.5 tok/s with identical
  2.37 committed/step. `GRIMOIRE_PREFILL_GRAPH` is not a production fix; a
  reusable graph would require persistent scratch and device-dynamic positions.

Logs:

- `/tmp/grim-orn-verify1-base.log`
- `/tmp/grim-orn-verify1-graph.log`
- `/tmp/grim-orn-mtp7-base64.log`
- `/tmp/grim-orn-mtp7-graph64.log`
- `/tmp/grim-orn-dflash-original-load.log`
- `/tmp/grim-orn-dflash-e2e16.log`
- `/tmp/grim-orn-dflash-mxfp4-result.log`

## Next steps — do in this order

1. Rerun the exact 32-token MXFP4 test on GPU0 through `tools/tune.sh`. Confirm
   the flattened embed launch passes and execution enters the six draft layers.
2. If another stage fails, add narrow stage markers and fix only that stage;
   do not infer runtime success from model load or build success.
3. Validate the WIP context ingestion and six-layer 16-query forward against
   `ref/qwen3_dflash.py`, including draft-cache positions and the non-causal
   block-attention mask. Original DFlash does **not** use the DFlash2 grouped
   convolution or selector path.
4. Validate coherent generated text and exact target distribution, then report
   committed tokens/step, draft time, verify time, and sustained TG on the same
   prompts used for the 125.5 TG baseline.

The exact reference forward and speculator input layout are already extracted
in `ref/qwen3_dflash.py` and `ref/dflash_speculator.py`. Do not infer the
original algorithm from the DFlash2 selector path.
