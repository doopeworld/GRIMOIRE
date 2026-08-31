# Muse Glimmer / DFlash handoff

Branch: `pp-muse`. Speculative decoding WORKS as of 2026-08-31.

## Current measured state

| prompt | accepts | committed/step | decode | prefill |
|---|---|---|---|---|
| 64 tok | 9/11 (82%) | 5.50 | 55.5 tok/s | 363.8 tok/s |
| 3672 tok | 18/64 (28%) | 1.39 | 20.3 tok/s | 1384.5 tok/s |

Was 0/8 accepts, 1.00 committed/step, 16.8 tok/s before this session.
Verifier now reproduces Fusion's token stream exactly: verify@64 emits
19669 200023 10064 262 ... against Fusion's OUTPUT TOKENS
328 19669 200023 10064 262 2818 17001 1509.

vLLM baseline for reference is ~2016 tok/s prefill on one B70, so Grimoire
prefill is at ~69% of vLLM.

## Fixed this session (all confirmed by measurement)

1. `linear_out_f16` undersized in the drafter (bd2422f). The shared lm_head
   wrote 15*202048 halves into a 16*39936 buffer. Draft logits were inf/nan on
   10 of 15 rows.
2. `page_stride_elements` passed in ELEMENTS, not sequence positions
   (0f93472). vLLM computes it as key_cache.stride(0)/key_cache.stride(1) =
   block_size. Grimoire passed block_size*kv_heads*head_dim, inflating every
   page address by 1024x, so page 0 read correctly and every later page
   returned zeros. This broke the TARGET as well as the drafter: with it, the
   target generated garbage past position 64; without the drafter loaded the
   target took the sequential path and looked fine, which masked it.
3. `VW` / `W` undersized in the verify path (87c673e). Same class as (1):
   prefill_muse projects the shared lm_head through xb/yb, whose widths
   enumerate only the layer projections. 16*202048 halves into 16*39936 - a
   5.06x overrun smashing ~5.2 MB of device memory every verify step. THIS was
   the 0/8 cause.
4. Stale `cu_q` (prefill_muse left {0,64}, draft needs {0,M}) plus four memcpys
   sourcing host stack values into async device copies. Real UB, verified NOT
   causal for 0/8. Kept as hygiene.

## Known-open

- **Acceptance collapses with context length: 82% at 64 tokens, 28% at 3672.**
  Prime suspect is the draft sliding window. Grimoire passes
  window_left = window_right = 2047 for the draft layers; at 64 tokens the
  window never binds, at 3672 it does. Same shape as every bug above: correct
  below a boundary, wrong above it. THIS IS THE NEXT THING TO FIX.
- `k_norm` bug-compatibility is in but NOT A/B tested. vLLM stacks
  `_k_norm_weights` as [num_layers, head_dim] and hands it to ops.rms_norm,
  whose weight must be [head_dim], so the kernel applies layer 0's weight to
  all five layers despite its comment claiming per-layer selection. Verified by
  recovering the effective weight from Fusion's own pre-RoPE context K:
  identical across layers (pairwise cos 1.0000, rms 1.08547) and equal to
  layers.0.self_attn.k_norm.weight, while the checkpoint's five k_norm tensors
  genuinely differ (rms 1.085, 1.349, 0.895, 1.381, 0.955). Revert and re-measure
  now that the verifier is correct; it may be unnecessary.
- Target aux states still differ from Fusion at rel_l2 0.032 (cos 0.99947).
  That is INT4 target numerics and bounds achievable acceptance.
- llama-benchy (`--pp 4096`) cannot drive Grimoire: it needs an
  OpenAI-compatible HTTP endpoint and Grimoire is CLI-only. A single-stream
  /v1/chat/completions wrapper around grimoire_generate would make Grimoire and
  Fusion benchmarkable with the identical tool.

## Lesson

All four bugs were width/stride errors that only manifest past a boundary -
page 0, or a buffer sized for layer widths rather than vocab. A 64-token prompt
hid every one of them. Test at the size you benchmark at.

## Tooling built this session

- `GRIMOIRE_DFLASH_DUMP=<dir>` dumps the first DFlash cycle's tensors as raw f32
  (`g_*.f32`) and prints the first two verify steps' candidate/verified vectors.
- `tools/fusion_dump.py` mirrors those dump points into Fusion (`f_*.f32`),
  plus slot mappings and per-layer context K/V. Needs `ENFORCE_EAGER=1` only for
  in-model tensor capture; everything else runs graph-enabled.
- `tools/compare_dumps.py` pairs `g_*`/`f_*` and reports the first divergent
  stage.
- `tools/fusion_verify_ref.py` runs Fusion's target over Grimoire's exact
  candidate sequence and reports its argmax per position - ground truth for
  `verified[]`.

## Runtime

- Tower `root@192.168.8.225`. Fusion on GPU1 = 35:00.0 = renderD130, port 3559.
- Grimoire on the free B70 = 03:00.0. **Render node numbers change on reboot** -
  resolve by PCI via `/dev/dri/by-path`. A run on the iGPU loads fine then dies
  with "No kernel named ... was found"; check the device line says 256 EUs /
  31.9 GiB.
- Never use `enforce_eager` for performance numbers.

## Build

```bash
docker run --rm --entrypoint bash \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire -w /grimoire \
  my-vllm-xpu:latest -lc 'bash build_b70.sh'
```

Attention bridge only (needs the kernel sources mounted at /src):

```bash
docker run --rm --entrypoint bash -e GRIMOIRE_BRIDGE_ONLY=attention \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire \
  -v /mnt/user/appdata/vllm-xpu-kernels:/src \
  -w /grimoire my-vllm-xpu:latest -lc 'bash tools/build_bridges_b70.sh /grimoire'
```

Keep image `my-vllm-xpu:latest`.
