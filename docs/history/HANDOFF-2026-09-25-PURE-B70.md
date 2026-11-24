# HANDOFF 2026-09-25 — pure GRIMOIRE on the B70: correct, and 7x faster prefill

**Read this first.** Supersedes the blocker in `HANDOFF-2026-09-24-FIRST-B70-RUN.md`.

## TL;DR

- **Root cause of yesterday's garbage: the vLLM plug-ins.** The launch scripts
  dlopen'd optional `src/libgrimoire_xe2_*.so` / `libgrimoire_onednn.so`
  plug-ins (vLLM kernels + libtorch). With them, long prompts gave different
  garbage every run and Ornith hit DEVICE_LOST. Without them GRIMOIRE is
  correct and repeatable. `bin/grimoire` itself links no torch/vLLM (ldd).
- **Every launcher now runs GRIMOIRE pure** (only `bin/` + `tools/` mounted,
  `/opt/grimoire/lib` hidden, no plug-in variables, no torch library paths).
  The plug-in .so files were moved to
  `/mnt/storage/isos/grimoire-plugins-removed-20260925/` (nothing deleted).
- **Three of GRIMOIRE's own kernels rewritten** (all SYCL, all verified):

| Qwen3.8-27B, 5987-token prompt, one B70 | 09-25 morning (pure) | now |
|---|---|---|
| dense FFN (GEMM) | 15,489 ms | 2,658 ms |
| DeltaNet recurrence | 7,741 ms | 555 ms |
| flash attention | 5,215 ms | 416 ms |
| **prefill total** | **35.0 s (171 tok/s)** | **4.98 s (1,202 tok/s)** |

  Same generated text in every run and with each old kernel switched back
  on (A/B). Ornith-1.5 long prompt: 9.7 s -> 3.4 s total, coherent.
  (For scale: the plug-in path timed 1,375 tok/s with garbage output; vLLM
  measured 2,015 on one B70.)

## What changed

1. `src/gemm_fast.cpp` (new) -> `bin/libgrimoire_gemm.so`, linked (not
   dlopen'd) into `bin/grimoire` / `bin/grimoire-server`, built with
   `-cl-intel-256-GRF-per-thread`:
   - **large-M GEMM**: dequantize W once into a bf16 VNNI scratch, then a
     joint_matrix GEMM loading straight from global memory (32x64 sub-group
     tile, 256x256 work-group). `tools/bench_gemm_bf16.cpp`: 110 TFLOP/s at
     6144x17408x5120 (old kernel ~13). Taken by `launch_gemm_xmx` when
     N%256==0, K%32==0, M>=32. Kill switch `GRIMOIRE_NO_FAST_GEMM=1`.
   - **matrix-unit flash prefill**: e4m3 cache packed once to bf16, each
     sub-group owns 8 query rows end to end, no work-group barrier in the key
     loop. `tools/bench_flash_prefill.cpp`: 24.7 ms/layer (was ~329),
     checked vs fp64 incl. ragged and resumed (start>0). Taken by
     `launch_flash_prefill` for head_dim 128/256. Kill switch
     `GRIMOIRE_NO_FAST_FLASH=1`.
   - Why a separate library: the per-kernel `grf_size<256>` property gives
     70 TFLOP/s vs 110 with the flag, and the flag cannot go on the engine
     image (gemm_flt and others use 1024-thread work-groups).
2. `src/prefill.cpp`: **row-parallel DeltaNet prefill** -- every state row
   evolves independently, so one sub-group owns 2 rows in registers; no
   barriers. Old kernel kept: `GRIMOIRE_DN_PREFILL_OLD=1`.
3. Launchers (tune/b70run/pp2run/tp2run/nrun/serve*/preflight): pure; image
   `my-vllm-xpu:latest` used ONLY as the oneAPI/Level Zero runtime.
   Preflight builds the engine only (no plug-ins) and checks with ldd that
   nothing needs torch/vLLM.
4. Build: `build_b70.sh`, `tools/build_*_only_b70.sh`, the CI workflow and
   `Dockerfile.b70-native` know about `gemm_fast.cpp` / the library.

## Findings (with evidence)

- The torch-free image `grimoire:b70-native` (built 09-06) segfaults the
  **JIT-built** gate binaries (exit 139, 4/4) while `my-vllm-xpu` runs them
  clean; AOT `bin/grimoire` runs fine in both. The gates are built for
  spir64 (JIT) by design. Not the default image until rebuilt + retested.
- Pure-mode gates on the card: not yet re-run with the rebuilt gates (the
  pre-rebuild run crashed in `test_model_matrix` because it ran in the
  torch-free image -- see above).
- gdb-oneapi needs `INTELGT_AUTO_ATTACH_DISABLE=1` or it quits trying to
  attach the GPU debugger.
- The B70 #1 link is still PCIe Gen4 x2 (x4-capable root port).

## Next steps

1. Full `build_b70.sh` (gates include `gemm_fast.cpp` now), then
   `SKIP_BUILD=1 PROJ=mxfp4 tools/preflight_b70.sh /models/Qwen3.8-27B-MXFP4-GRIMOIRE`
   from grimoire-fuse -- the real correctness picture in pure mode.
2. Decode speed (pure Qwen ~23 tok/s): profile `forward()` with
   `GRIMOIRE_TIMELINE=1`; the oneDNN plug-in used to give 23 -> 30.
3. MTP speculation in pure mode: exact vs plain? acceptance? speed?
4. Dense FFN is now 53% of prefill: tail-row waste (5987 = 23*256+99),
   down-projection tile, INT4/int8 path at 2x the bf16 rate.
5. Two-card PP/TP pure runs (Ornith), then llama-benchy.
