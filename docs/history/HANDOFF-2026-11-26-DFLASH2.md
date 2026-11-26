# Handoff 2026-11-26 — Inkling 2 Mini: DFlash2 drafter (after the power cut)

Branch `inkling-df2-1126` (GitHub). Nothing was lost in the 11-26 power cut: the 11-25 work (e0458f7) was on GitHub
and on the Tower, nothing was running. Tower back 16:54, array started, both B70s at 03:00.0 / 07:00.0, B580 at 0b:00.0.

## What was done

Ian: "wrapzii said that he uses RADIANCE to make inkling faster", then "anything you want to make inkling the fastest
possible". Radiance 2.3.0 (11-26) serves Inkling 2 Mini with **incoai/Inkling-2-Mini-DFlash2** (block-diffusion
drafter + candidate selector) instead of the release's DSpark head: 2x R9700 c1 essays 190 / code 240 / reasoning 358.

Ported that drafter into the Inkling engine (`tools/inkling2_df2.inc`), transcribed from SGLang `dflash.py` /
`dflash_worker_v2.py` and the checkpoint's `sglang_inkling_dflash2.py`:
- 5 INT8 layers over [anchor, mask x 7]: pre-norm attention (32 q / 8 KV heads of 128, q/k RMSNorm, NeoX RoPE 1e4,
  non-causal block, window 2048) and SwiGLU MLP, each wrapped in the grouped dynamic 2-tap convolution;
- context K/V = k_norm(rope(k_proj(hidden_norm(fc([TAP0..TAP3]))))) — the taps are DSpark's;
- candidates: top-16 of the target's INT8 head (raw logits), edge scores
  `30 tanh((logit + <A[pred] * P fin, B[cand]>) / 30)` (softcap after the correction), greedy or sampled walk; the
  walk's softmax is q of the exact p/q verify. Verify / commit / sampling paths are DSpark's, unchanged.
- Drafter weights: `/mnt/storage/Models/Inkling-2-Mini-DFlash2` (HF @4f9088e, sha256 2179...4ee).
- Default: used when the drafter folder sits next to the model (`<models>/Inkling-2-Mini-DFlash2`) or in
  `<model>/dflash2` (needs INK_I8, the server default); `INK_DFLASH2=<dir>` names it, `INK_DFLASH2=0` keeps DSpark.

## Measured (Inkling-2-Mini-Uncensored)

| setup | DSpark | DFlash2 |
|---|---|---|
| BetterBench quick, two B70, weighted decode | 139.7 (2.90 tok/step, 22.5 ms/step) | **179.2** (3.62 tok/step, 22.0 ms/step) |
| two B70 greedy, code2 (code answer) | 251 | **303.8** (7.37 tok/step) |
| two B70 greedy, chat1 | 136.5 (3.27) | **236.0** (5.63) |
| one B70 greedy, code2 | 177 | **212.8** (7.08) |
| one B70 greedy, chat1 | 88.3 (2.90) | **148.5** (4.79) |
| 262144 context, two B70: memory a card | 30.26 GiB | 26.95 GiB |

Greedy output equals plain decoding exactly where DSpark's does (two-card verify rounding makes both diverge from
plain decoding near token 40 on chat1 — pre-existing). Original Inkling-2-Mini: chat1 4.79, code2 7.66 tok/step.

Tried, not kept (all measured): draft stop 0.1 / 0.5 (169.3 / 177.7 vs 179.2 at 0.25); q with the request's top-k /
top-p (`INK_F2_QTRUNC=1`, 178.3); argmax drafts (153.2); INT4 drafter (`INK_F2_I4=1`: chat1 5.63 -> 4.61 tok/step);
BF16 drafter (`INK_F2_BF16=1`: same acceptance, +1 ms/step); verify-expert K splits (`INK_SM_KSG` / `KSD`,
bench 142.6 vs 148.2 us a visit but 1 ms/step slower in the engine).

## Where the time goes now (two B70, 8-row verify ~21.6 ms + draft ~2.1 ms)

- verify experts ~9.6 ms a card (ablation `INK_ABL=1`; 25 distinct experts a visit, this card's 13): the ESIMD
  kernels run ~390 GB/s (`tools/bench_ink_sm.cpp`). The weights are read as 64 B of 16 rows per 2-D load at a
  1152 B row stride; a tiled expert layout is the likely road to ~550 GB/s, but decode, prompt and verify kernels
  all read the same copy (no room for a second one at 262K).
- INT8 small-M dense (Wp / Wo / unembed / drafter) ~2.2 ms; the rest (~1000 small kernels, 96 exchanges) ~9 ms.

## Image

`grimoire-b70:v1.10.14-rc1` (f92f54d5) = v1.10.13 + inkling2_gpu + libink_xexp.so of 2fd3605 for G31 + G21
(`grimoire-runs/ink-rel-1126`, Dockerfile `grimoire-runs/image-ink-1126`). Validation:
`ink-1123/validate_ink_image2.sh` (Ian's template layout: /mnt/storage/Models at /models, 262144 context, two B70).
NOT published — waits for Ian's yes. Ian's GRIMOIRE-DUAL-INKLING2 template needs no change besides the image tag.

## Next

1. Publish v1.10.14 on Ian's yes (save + gzip -6, upload_release.py, README row).
2. Verify expert bandwidth (tiled layout), then INT8 small-M dense; batching several conversations for c2-c8.
