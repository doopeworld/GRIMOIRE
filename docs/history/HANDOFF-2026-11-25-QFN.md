# Handoff 2026-11-25 evening — paused by Ian, resume 11-26

Branch `qwen38-spec-1125` (GitHub; `main` = 50be181 = v1.10.13 source). Nothing running on the Tower.

## Done 11-25
- **v1.10.13 published** (G31 + G21, image b167b2a2, `ENV GRIMOIRE_DFLASH_TREE=7`). Measured on the image, Qwen3.8-27B
  Swift INT4 + DFlash2, BetterBench quick: one B70 110.4, two B70 **142.8** (was 136.6).
- DFlash2 drafter attention (c4f2cc5): drafter 3.9–4.6 → 3.2 ms/step, flat in context.
- Draft trees (50be181, `GRIMOIRE_DFLASH_TREE`): one card, one conversation; +5% tokens/step, +0.9 ms/step.
- Coupled (Gumbel) drafts: opt-in, no gain. Inkling copy drafts (d2c3d9c, `INK_COPY_MIN`): opt-in, slower on
  copy-heavy text (181/175 → 170/168 tok/s one B70) — the 8-row verify cap limits copying.

## QFN (Qwen3.8-Flash-Next) — measured 11-25, two B70, Ian's own containers, BetterBench quick --greedy
| engine | tok/s |
|---|---|
| GRIMOIRE-DUAL-FLASH-NEXT (MTP off, NVFP4 as int8, 170 experts/layer) | **54.8** (18.5 ms/token) |
| GRIMOIRE same, MTP on (v1.10.13) | 39.0 |
| xenithium-qfn (1 B70, 07:00.0) | 8.4 first run → 18.1 with its learned hot-expert ranking |
| urakozz README (PCIe 3.0, DDR4-2133, P2P, 128 GB RAM) | plain tp code 59–66, **MTP tp code 84–92**, pp32K 3,415 |

Gap to urakozz = MTP (ours loses 29%, his gains ~40%) and prompt (ours pp32K 1,637).

## Next (Ian decides the order)
1. QFN MTP verify: our 4-row step costs ~2x a plain token. Cold experts are computed on the CPU per row (up to 4x
   more per step), two-card exchanges per row, idle GPU. Direction: cold experts for all verify rows on the GPU
   (weights read from host once per step, like urakozz), then the queue-ahead two-card step (est. −2 ms/token).
2. QFN prompt: pp32K 1,637 → urakozz 3,415.
3. Inkling copy speed (davetha 328 tok/s on copy edits): needs 16–32-row verify blocks for copy steps.
