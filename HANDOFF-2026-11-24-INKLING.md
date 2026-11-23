# Inkling 2 Mini on GRIMOIRE — handoff 2026-11-24 (paused by Ian)

Branch `inkling-gpu-1123`. Engine: `tools/inkling2_gpu.cpp` (+ `inkling2_mtp.inc`, `inkling2_moe_gemm.cpp`,
`inkling2_xexp.cpp`). Tower: `/mnt/storage/isos/grimoire-runs/ink-1123` (`build_ink.sh`, `run_ink.sh`, `run_tp.sh`).

## Where it stands (measured, Inkling-2-Mini-Uncensored)

| setup | value |
|---|---|
| 1 B70 prompt pp512 / pp4096 / pp16384 | 1,296 / 2,689 / 2,282 tok/s (start 11-23: 355 / 954 / 636) |
| 1 B70 generation, BF16 weights (exact) | 57.5 tok/s |
| 1 B70 generation, `INK_I8=1` | 66.5 tok/s (12 tokens = CPU reference) |
| 1 B70 generation, `INK_I4=1` (GRIMOIRE INT4 RTN + `launch_int4_smallm_wide`) | 72.1 tok/s (max/mean dlogit 2.54/0.42) |
| 2 B70 generation, `INK_I8=1` | 80.4 tok/s @512, 78.4 @4K |
| 1 B70 MTP (DSpark) | 37.6 (BF16), 44.4 (`INK_I4=1`, 2.76 tok/step) |
| 2 B70 MTP | 47.4 |

Per token (int8 dense, one B70, ~15 ms): experts ~5.5 ms, ~1000 tiny kernels ~4.3 ms, dense ~3 ms, unembed 0.8 ms.

## The wall, and the way around it (READ FIRST)

Four expert-decode designs (SIMT SLM tables; ESIMD DPAS with nibble widening; DPAS one-wave; DPAS with SLM byte
tables) ALL ran at ~300 GB/s (gate_up ~75-79 us, down ~41-44 us per visit). Do not write a fifth.

GRIMOIRE already solved this for Ornith: `src/moe_kernels.cpp` `moe_gu_esimd` / `moe_dn_esimd` ("ESIMD decode MoE,
the llm-scaler shape"), MEASURED 266/248 GB/s (SIMT) -> **477/484 GB/s**:
- a thread streams WHOLE ROWS with contiguous 64-byte block loads (`mx4_row_step`), R row pairs per thread,
  KS threads split H, partial sums meet in SLM only at the end (no tables, no barrier in the loop);
- the activation slice lives in REGISTERS, split even/odd K (= low/high nibble of each byte);
- E2M1 decode in the ALU: nibble magnitude bits -> fp16 bits 9..11 + sign bit 15 = value x 2^-14
  (`e2m1_split`), the 2^14 folded into the block scale; one FMA pair per 16 per scale block.
- down kernel: work-group = TK+1 slots, slot partials summed in slot order (routed weights applied there).

**Plan 1 (first thing):** port `moe_gu_esimd` / `moe_dn_esimd` to Inkling NVFP4: scale = E4M3 per 16 elements x
outer (amax[e]/den), apply per 16-block (8 per 128-step instead of 4), Inkling's softcapped SwiGLU in the epilogue,
gate/up rows interleaved (gate i = row 2i, up i = row 2i+1). Drop the per-weight bf16 rounding (vLLM-style dequant;
check vs CPU ref: expect mean dlogit ~0.05-0.08). Target: experts 5.5 -> ~3.2 ms/token.

**Plan 2:** verify (M=8) experts the same way as Ornith (`launch_moe_mxfp4_smallm` pattern, grouped small-M, every
expert read once per step) -- MTP is only worth it once verify reads at ~480 GB/s.

**Plan 3:** cut the ~1000 tiny kernels per token (fuse norm into mix, router+topk, attention 3->2, gate cat+blend,
post+resid+mlpconv); Xenithium/urakozz fuse exactly these. Target ~-2 ms/token.

**Plan 4 (prompt):** port GRIMOIRE's Ornith prompt MoE `moe_mxfp4_esimd` (gemm_fast.cpp, 52-68 TFLOP/s, SLM 1 KB
byte table, 128-K steps, 2 DPAS per 32 K) to NVFP4 instead of the own `InkMoeG` (~32 TFLOP/s).

**Check once (read-only sysfs, during a decode run):** GPU act_freq on gpu0 while decoding -- short kernels may run
at a lower clock than the long unembed kernel (571 GB/s).

## Do not repeat (all measured 11-24)
- `SYCL_UR_USE_LEVEL_ZERO_V2=0` -> DEVICE_LOST on gpu0 (bcs reset + coredump). Default adapter is V2.
- `INK_DEVX=1` device mailboxes in pinned host memory: no `usm_atomic_host_allocations` on B70 -> spin never sees
  the peer's writes, watchdog exit, engine resets. No P2P on the stock Tower kernel.
- 256-thread mix kernel for decode rows (tg 57 -> 54); kept only for prompt rows.
- A-prefetch in `InkMoeG` (experts 782 -> 888 ms).
- Decoding all experts up front before the oneDNN GEMMs (loses L2 locality).

## Opt-in flags (env, passed by run_ink.sh / run_tp.sh)
`INK_I8=1` int8 Wp/Wo/unembed decode; `INK_I4=1` (+`INK_I4_PARTS` bits 1 Wp, 2 Wo, 4 unembed,
`GRIMOIRE_INT4_RTN_GROUP=64`, `GRIMOIRE_INT4_RTN_SEARCH=1`); `INK_GSM=1` GRIMOIRE bf16_smallm for decode/verify dense
(MTP == plain 128/128); `INK_XEXP=1` DPAS expert decode (no gain); `INK_SECT=1` per-section prompt timing;
`INK_DENSE_ONEDNN=1`, `INK_ATT_SIMT=1`, `INK_EXP_ONEDNN=1` = older paths.
Two cards: `run_tp.sh TAG tp IDS CTX PP_LIST TG CHUNK`, `run_tp.sh TAG tpmtp CHAT CTX PP N CHUNK` (watchdog 60 s).

## Research notes 11-24
- Xenithium (github.com/Niluthon/Xenithium, Apache-2.0, SYCL/ESIMD): clone in the session scratchpad; MoE decode
  `src/kernels/moe/moe.cpp`, verify rows `moe_rows.cpp`, W4 tile `include/xenithium/kernels/tile/w4_tile.hpp`.
  Flint INT4 (MoE 256 top-6) 1 card 278 tok/s with DFlash 2, pp8K 8,166.
- urakozz (OpenCL C + Level Zero command-list replay): Ornith 1 card pp4K 11,223, MTP 312; verify M=8 costs 1.57x one
  plain step; decode dispatch 0.73 us in a replayed list.
- Reference Inkling speeds elsewhere: vLLM + DSpark RTX 5090 ~416 tok/s code / pp8K ~17.5K; llama.cpp IQ4 ~43-63 tok/s.
