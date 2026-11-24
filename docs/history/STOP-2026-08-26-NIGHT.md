# GRIMOIRE STOP — 2026-08-26 NIGHT

## Current measured state

Qwen3.8-27B MTP k=3, 256 generated tokens, short context:

- **51.7 TG sustained** (256 tokens / 4950 ms)
- 94 verifier steps, 2.76 committed tokens/step
- draft accepts 165/222, 57 rollbacks
- profile: snapshot 65.2 ms, draft 385.4 ms, verify 4438.9 ms,
  commit 59.3 ms
- output coherent
- **60 TG target is not yet met**

Log on Tower: `/tmp/grim-mtpfp8-fuseq-256.log`.

## Current implementation

- FP8 E4M3 KV cache throughout decode, prefill, MTP verification, and the MTP
  draft layer. K remains D-major and V D-minor.
- Small-batch split-K attention shares each staged K/V tile across all MTP
  rows and both GQA query heads belonging to a KV head.
- W4A8 activation quantization is reused for repeated projections of the same
  normalized row.
- Dense FFN post-norm and SwiGLU fuse production of int8 activations/scales,
  eliminating their standalone activation-quantize launches.
- `GRIMOIRE_MTP_DRAFT_VOCAB=131072` is the best tested prefix shortlist. It
  preserves acceptance on the controlled sweep and saves draft bandwidth.
  32K and 64K reduce acceptance too much.

## Reproduce the 51.7 TG run

```bash
cd /mnt/storage/isos/grimoire-fuse
GPU=gpu0 LIM=900 EXTRA_ENV='GRIMOIRE_W4A8=1
GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_PROFILE=1
GRIMOIRE_MTP_DRAFT_VOCAB=131072' \
  bash tools/tune.sh mtpfp8-fuseq-256 /grimoire/bin/grimoire \
  -m /models/Qwen3.8-27B-MXFP4-GRIMOIRE \
  --proj mxfp4 --ctx 8192 -p 'Explain why the sky is blue.' -n 256
```

## Important comparison flags

The vLLM reference must retain:

```text
--block-size 64 --dtype bfloat16 --max-num-batched-tokens 4096
--trust-remote-code --async-scheduling
```

Use the established FP8 KV-cache setting as well. These are vLLM flags; their
semantics are implemented internally rather than passed to Grimoire.

## Measurements from tonight

| configuration | sustained TG |
|---|---:|
| prior FP32-KV production verifier | 49.4 |
| FP8 KV, full draft vocabulary | 47.3 |
| FP8 KV, 128K draft vocabulary | 48.9 |
| + projection activation reuse | 49.5 |
| + GQA-shared KV + fused FFN quantization | **51.7** |

FP8 KV is required for longer-context scaling, although its conversion cost
does not pay back at the tiny 18–274-token context used for this gate.

## Do not repeat

- The reported 62.9 TG run was one four-token step with 100% draft acceptance;
  it was not sustained throughput.
- Existing k=3..7 sustained sweep: 46.9, 46.5, 44.7, 45.2, 45.7 TG;
  increasing k alone does not reach 60.
- Compact recurrence-input replay was tried and reverted: state validation
  failed and rejection-heavy traffic became slower.
- First-32K shortlist: 44.4 TG. First-64K: 48.0 TG. Do not use them.

## Preserved fallback

Before the final fused-quantization experiment, the Tower source was copied to:

- `src/grimoire.cpp.pre-fuseq-20260826`
- `src/prefill.cpp.pre-fuseq-20260826`
- `src/kernels.hpp.pre-fuseq-20260826`

The final experiment passed coherence and was faster, so the active source is
the preferred state.
