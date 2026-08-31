# Muse Glimmer / DFlash handoff

Branch: `pp-muse`

## Runtime reference

- Tower: `root@192.168.8.225`
- Fusion reference: container `VLLM-XPU-FUSION`, image `my-vllm-xpu:latest`
- Fusion runs on GPU1 (`ZE_AFFINITY_MASK=1`) at port 3559. Do not modify or restart it.
- Grimoire tests run on GPU0 (`/dev/dri/renderD129`).
- Target: `/models/Muse-Glimmer-30B-INT4-W4A16`
- Assistant: `/models/Muse-Glimmer-assistant`, DFlash K=15
- Fusion version: vLLM `0.28.1.dev8+ga4f0b85eb.d20260831`, V2 runner enabled.

## Confirmed Fusion semantics copied into Grimoire

- Harmony prompt tokenization matches 64/64 token IDs.
- DFlash uses one bonus token plus 15 mask tokens, mask ID 201818.
- Query positions are `[last_valid_position + 1, ..., +16]`; sampling uses the 15 mask rows.
- Target auxiliary source IDs are `[1, 13, 25, 37, 49]`, captured after target layers `[1, 13, 25, 37, 49]` as Fusion reports layers `(2, 14, 26, 38, 50)`.
- Assistant uses five non-causal sliding layers, window 2048, block size 64, NeoX RoPE, theta 500000.
- Assistant parameters and shared target embedding/lm-head run as FP16.
- Fusion's `F.linear` oneDNN trace is FP16 src/weights/dst with `ab`/`ba`/`ab` descriptors and user scratchpad; it has no explicit FP16 fast-math attribute. The extra Grimoire attribute was removed.
- Muse RMSNorm adds the weight offset in FP32 and casts only the final result. Grimoire's premature FP16 cast was removed.

## Current measured state (not a successful result)

Identical 64-token prompt: `Write a haiku about the ocean.`

- Fusion first 8 tokens: ` to`, `=self`, `<|message|>`, `Write`, ` a`, ` ha`, `iku`, ` about`.
- Grimoire after the latest two parity corrections: ` to shakes shakes shakes shakes sää smell smell`.
- Both produce the same first target-prefill token (` to`). The first demonstrated divergence is token 2, inside the initial DFlash proposal/verification cycle.
- Grimoire check: 8 tokens, 17.0 tok/s, 0/8 draft accepts. This is invalid for performance evaluation.
- Prior 64-token check before the oneDNN descriptor correction: 16.9 tok/s, 0/8 accepts. The descriptor correction changed draft output but did not restore parity.
- GPU0 is idle after each exited test. GPU1 Fusion remains running.

## Required next step

Do not run another full benchmark or tune parameters. Instrument a temporary Fusion clone on GPU0 and Grimoire to dump the first DFlash-cycle tensors, then compare in this order:

1. concatenated target auxiliary states `[64, 5 * 6656]`;
2. `fc` output and hidden RMSNorm output;
3. fused context K/V projection, K norm, and RoPE output;
4. bonus/mask embeddings;
5. each assistant layer's input norm, Q/K/V, attention output, residual, and MLP output;
6. final norm and 15 draft logits/argmax IDs.

Patch only the first unequal stage. The active Fusion source is under `/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/dflash/` and `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py`.

## Build

```bash
docker run --rm --entrypoint bash \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire -w /grimoire \
  my-vllm-xpu:latest -lc 'bash build_b70.sh'
```

Build only the oneDNN bridge:

```bash
docker run --rm --entrypoint bash -e GRIMOIRE_BRIDGE_ONLY=onednn \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire -w /grimoire \
  my-vllm-xpu:latest -lc 'bash tools/build_bridges_b70.sh /grimoire'
```

Keep image `my-vllm-xpu:latest` (ID `a5044994417a`). It contains the working Fusion reference and the compiler/runtime used for Grimoire builds.
