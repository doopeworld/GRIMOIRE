# Qwen3.8-Flash-Next (qwen4_exp) on one B70 + system RAM + SSD — design

2026-09-26. Checkpoint: `nvidia/Qwen3.8-Flash-Next-NVFP4` (132.7 GB) in
`/mnt/storage/Models/Qwen3.8-Flash-Next-NVFP4`. Pure GRIMOIRE (SYCL/L0/C++), gpu0 only.

## What the checkpoint really contains (read from its index, not assumed)

| part | format | size | per decoded token |
|---|---|---|---|
| routed experts: 48 layers x 512 x {gate,up,down}, I=640, H=2560 | NVFP4, **modelopt names** (`weight` U8, `weight_scale` E4M3 /16, `weight_scale_2` F32 multiply, `input_scale`) | 67.9 GB | top-10 x 48 = 480 experts = 1.33 GB |
| everything else (DeltaNet x36, full attn x12 + QSA indexer, HC mixers, shared experts, routers, embed, lm_head) | BF16 (excluded from quantization by NVIDIA) | ~11 GB | ~8 GB |
| PLE n-gram table, layer 1 only | FP8 E4M3, **128 row-shards** `ngram_embedding.shard_{i}.weight`, 160 B rows, one F32 scale | ~51 GB | 16 rows = 2.5 KB |
| hash constants (`layer_multipliers`, `ngram_heads_vocab_sizes`, `ngram_heads_offsets`) | int64 tensors **stored in the checkpoint** | tiny | - |
| MTP layer (1) | experts FP8 block (`weight_scale_inv`), rest BF16 | ~2.6 GB | (later) |
| vision tower | - | skipped | - |

GRIMOIRE's qwen4_exp loader was written against synthetic fixtures: it looks for
compressed-tensors NVFP4 names, a single `ngram_embedding.weight`, and derives the hash
constants from the seed. None of those match the real file. Ornith-1.5-35B-A3B-NVFP4 uses the
same modelopt names, so it is the small (fits in VRAM) test for the NVFP4 part.

## Measured links (gpu0, tools/tier_probe.cpp, 2026-09-26)

| link | bandwidth |
|---|---|
| VRAM read (kernel) | 600 GB/s |
| host -> VRAM copy, pinned or pageable, 2.76 MB chunks | 12.4 GB/s (one expert = 227 us) |
| kernel reading pinned host memory directly (zero copy) | 12.8 GB/s |
| CPU DRAM read, 16 / 6 threads | 67 / 56 GB/s |
| SSD (ZFS, 128K records), random 2.76 MB reads | 0.2-0.35 GB/s *during a 132 GB download* — re-measure |

CPU DRAM is 5x PCIe. So RAM-resident experts should be *computed on the CPU* in decode
(ktransformers-style), not pulled over PCIe. Prefill touches every expert every layer, so
there the GPU streams them (PCIe-bound: ~(1-f_vram) x 1.41 GB/layer / 12.4 GB/s).

## Placement

- **VRAM (32 GB):** all BF16 non-expert weights (~11 GB), KV/state/scratch (~2-3 GB), and a
  hot-expert cache filling the rest (~17-19 GB = ~6-7K of the 24,576 experts).
- **Pinned RAM (~49 GB):** every other expert, NVFP4 as stored. Nothing duplicated.
- **SSD:** the PLE table, read by row on demand (16 rows/token); MTP; any expert that does
  not fit the RAM budget (SSD expert tier only if needed — it is 40x slower than RAM).
- Hot/cold: per-(layer, expert) routing counts are recorded at run time and saved next to the
  model; the next load puts the hottest experts in VRAM. First load: uniform.

## Expert block (GRIMOIRE-native, one per layer x expert, 4 KB aligned)

`gate_up payload [2I][H/2] | gate_up E4M3 scales [2I][H/16] | down payload [H][I/2] |
down scales [H][I/16] | f32 global scales {gate, up, down}` = 2,764,864 B. One pointer per
expert (`eptr[layer][e]`, VRAM or host USM) is all a kernel needs, so the same kernels serve
every tier, and a tier change is a pointer swap plus one 2.76 MB copy.

NVFP4 stays native (e2m1 x E4M3 x global, 16-wide blocks). Re-quantizing to MXFP4 would
double-quantize every expert weight (coarser pow-2 scales, 32-wide blocks).

## Compute

- **Decode, phase 1:** GPU NVFP4 MoE GEMV reads every selected expert through `eptr`
  (VRAM, or host over PCIe zero-copy). No host sync, correct first.
- **Decode, phase 2:** RAM-resident experts computed by a CPU thread pool (AVX2, NVFP4 LUT
  decode) while the GPU does the VRAM-resident ones; one small D2H/H2D per layer.
- **Prefill:** route on GPU, then per layer: dequant each expert's NVFP4 (from `eptr`) to a
  bf16 VNNI staging buffer and run the existing fast bf16 GEMM on that expert's rows, SwiGLU,
  down GEMM, weighted scatter-add. PCIe-bound, so the prompt should go in one big chunk.
- **PLE:** CPU computes the 16 hashed row ids per token from the stored constants, preads
  the rows (they are 160 B; the file offsets come from the shard index), uploads.

## Estimates (to check against measurements)

- Decode phase 1 at ~50% VRAM hit rate: ~52 ms PCIe + ~13 ms dense -> ~15 tok/s.
- Decode phase 2: ~12 ms CPU (overlapped) + ~13 ms dense + ~2.5 ms syncs -> ~35-40 tok/s.
- Prefill: ~3.9 s per pass of the whole stack with 28% of experts in VRAM, nearly
  independent of prompt length until compute catches up (4K tokens ~1,000 tok/s,
  16K ~3,900 tok/s).

## Order of work (each step verified before the next)

1. Loader: modelopt NVFP4 (multiply by `weight_scale_2`). Verify on Ornith-1.5-35B-A3B-NVFP4.
2. Loader: PLE shards + stored hash constants; skip vision/MTP.
3. Tiered expert store + NVFP4 MoE decode kernels via `eptr`.
4. Prefill MoE through staging + bf16 GEMM.
5. PLE rows from SSD.
6. Flash-Next end to end: coherent text; then a layer-by-layer reference check.
7. Speed: CPU expert compute, hit-count placement, PLE on a small-recordsize dataset.
