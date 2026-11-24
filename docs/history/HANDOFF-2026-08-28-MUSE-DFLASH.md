# GRIMOIRE handoff — Muse Glimmer original DFlash

Date: 2026-08-28

## Current result

Muse Glimmer target generation and original DFlash are now coherent on GPU0.
This is a correctness checkpoint, not the final performance result: the owner
requires at least **50 TG**, and the latest DFlash run is **10.3 TG**.

GPU policy is strict: use only Tower GPU0, currently `/dev/dri/renderD128`
(PCI `03:00.0`). Do not inspect, stop, restart, or allocate GPU1.

Latest target-only result:

```text
We need to write a haiku. Traditional haiku is 5-7
pp 30 tokens in 182 ms -> 164.4 tok/s
tg 16 tokens in 517 ms -> 31.0 tok/s
```

Latest DFlash result after both correctness fixes:

```text
We need to write a haiku about the ocean. Probably just output haiku. No constraints. Should be fine.
Make sure it's a haiku:
pp 30 tokens in 164 ms -> 182.4 tok/s
DFlash(k=15) tg 32 tokens in 3109 ms -> 10.3 tok/s
DFlash steps 12, 3.17 committed/step, draft accepts 26/38, rollbacks 12
```

Profiled 16-token run:

```text
DFlash(k=15) tg 16 tokens in 1561 ms -> 10.3 tok/s
DFlash steps 6, 2.83 committed/step, draft accepts 11/17, rollbacks 6
DFlash profile snapshot 0.0 ms, draft 823.7 ms, verify 732.8 ms, commit 4.1 ms
```

That is approximately 137 ms/step in the BF16 drafter and 122 ms/step in the
Muse verifier. Both must be accelerated; acceptance alone cannot reach 50 TG.

## Correctness fixes in this checkpoint

- Generalized original DFlash from Ornith's fixed dimensions to the Muse
  assistant checkpoint: H=6656, I=19968, five layers, QH=32, KVH=8, HD=128,
  target taps `[1,13,25,37,49]`, mask token 201818, window 2048.
- Added Muse's dense batched target prefill/verifier with Gemma sandwich norms,
  attention output gate, dense MLP, target-tap capture, and batched logits.
- Parsed and applied `post_norm_eps`; final Muse norm uses raw checkpoint weight.
- Fixed target-tap storage stride to use the checkpoint's actual tap count.
- Fixed an out-of-bounds/alias error in the draft MLP: SwiGLU has M*I storage
  and down projection writes to a separate M*H buffer.
- Fixed Muse attention gating. The gate projection is `[M,QW]` and must apply
  `sigmoid(gate[t,d])`; the previous batched helper incorrectly used one scalar
  `gate[t]` for every channel. A separate elementwise helper preserves scalar
  gating behavior for other architectures.
- Fixed Muse draft attention semantics. Its config has five
  `sliding_attention` layers, which the vLLM reference resolves as causal.
  GRIMOIRE had evaluated the whole 16-token block non-causally. The block kernel
  now supports causal per-query bounds, enabled for Muse. This increased the
  measured committed tokens/step from 2.67 to 3.17 on the fixed prompt.

Full AOT builds pass after these changes.

## Exact model pairing and run command

- Target: `/mnt/storage/Models/Muse-Glimmer-30B-MXFP4`
- Assistant: `/mnt/storage/Models/Muse-Glimmer-assistant`
- Container: `my-vllm-xpu:latest`
- Repository: `/mnt/storage/isos/grimoire-fuse`

Use `--proj mxfp4`; the default int4 load is not the tested path.

```bash
docker run --rm --device /dev/dri/renderD128 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  -e GRIMOIRE_DFLASH_MODEL=/models/Muse-Glimmer-assistant \
  -v /mnt/storage/isos/grimoire-fuse:/workspace \
  -v /mnt/storage/Models:/models -w /workspace \
  --entrypoint bash my-vllm-xpu:latest -lc \
  "./bin/grimoire -m /models/Muse-Glimmer-30B-MXFP4 --proj mxfp4 --ctx 512 \
   -p 'Write a haiku about the ocean.' -n 32"
```

Add `-e GRIMOIRE_MTP_PROFILE=1` for phase timing.

## Next steps to reach 50 TG

1. Remove per-call Muse verifier allocation/free. `prefill_muse` currently
   allocates all `[M,*]` buffers, logits, BF16 staging, and token buffers every
   speculative step. Add persistent M<=16 verifier scratch (or safely reuse the
   original-DFlash scratch sequentially) and retain it until `release()`.
2. Accelerate the drafter. It remains BF16 and costs ~137 ms/step. For Muse,
   test load-time MXFP4 copies of assistant projections with the existing
   `grimoire_xe2_dense_mxfp4_f32` path, preserving BF16 norms/activations, then
   require coherent output and equal-or-better acceptance.
3. Confirm the Xe2 dense bridge is actually loaded in both `dflash_draft` and
   `prefill_muse`. The 122 ms verifier suggests fallback or poor M=16 dispatch.
   The loader searches `GRIMOIRE_XE2_GROUPED_LIB`, `./bin/libgrimoire_xe2_grouped.so`,
   and `/workspace/bin/libgrimoire_xe2_grouped.so`.
4. Optimize Muse verification only after phase timing shows the exact costly
   kernels. Target weight traffic implies a batched M=16 verifier should be far
   below 122 ms on B70. Reuse the optimized BF16-output/f32-output dense paths
   already used by generic `prefill` rather than maintaining a slow parallel path.
5. Re-run at n=64 or n=128 and report success only when coherent output is at
   least 50 TG. The target-only baseline is ~31 TG.

## Repository note

The Tower checkout had no `.git/index`, which made every tracked file appear
both deleted and untracked. The index was safely reconstructed from HEAD with
`git read-tree HEAD`; no working files were changed by that operation.
