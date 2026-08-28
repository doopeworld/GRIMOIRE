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

## Next steps — do in this order

1. Implement original-DFlash context ingestion.
   - Preserve all eight target residual taps for every newly verified token,
     not only the last token.
   - Apply `fc.weight` to concatenated taps, then `hidden_norm`.
   - For each draft layer, project/norm/RoPE K plus project V and append them to
     that layer's draft cache at the target positions.
2. Implement one 16-query draft forward.
   - Row 0 is the known bonus token; rows 1-15 use mask token 248077.
   - Six dense Qwen3 layers, hidden 2048, intermediate 6144, 32 Q heads, 8 KV
     heads, head_dim 128, theta 1e7.
   - Attention is non-causal across the whole query block plus context. Use
     `launch_dflash2_block_attention`; window 4096 for layers 0-4 and unlimited
     for layer 5.
   - Original DFlash does **not** use DFlash2 grouped-conv or selector kernels.
3. Apply target `lm_head` to mask rows 1-15 in one batched projection and choose
   one token per row. This is the proposed 15-token block.
4. Feed `[bonus + 15 drafts]` through the existing verifier, acceptance,
   recurrent snapshot, rollback, and replay machinery. Increase
   `kSpecBatch` from 8 to 16 first.
5. Validate coherent generated text and exact target distribution, then report
   committed tokens/step, draft time, verify time, and sustained TG on the same
   prompts used for the 125.5 TG baseline.

The exact reference forward and speculator input layout are already extracted
in `ref/qwen3_dflash.py` and `ref/dflash_speculator.py`. Do not infer the
original algorithm from the DFlash2 selector path.
