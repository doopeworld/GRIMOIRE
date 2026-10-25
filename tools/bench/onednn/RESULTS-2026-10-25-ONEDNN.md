# oneDNN on the Arc Pro B70 vs GRIMOIRE's prompt GEMM (2026-10-25)

Why: two developers suggested adding Intel oneDNN to GRIMOIRE (Ian, 10-25). Measured before
deciding, all on gpu0 (one B70, 03:00.0), weights read cold from memory (rotated copies >= 512 MiB),
oneDNN v3.13.4 built from source with GRIMOIRE's compiler (icpx 2026.1.1) -- no PyTorch, no vLLM.

Files: `build_onednn.sh` + `Dockerfile.dnnlbuild` (build), `mm_b70.cases` / `mm_b70_2.cases`
(benchdnn configurations), `bench_loop.sh` + `run_benchdnn_g0.sh` (benchdnn on gpu0 under gpu0.lock),
`gemm_fast_probe.cpp` (GRIMOIRE's own launch_gemm_fast at the same shapes, host-checked) and
`probe_onednn_g0.sh`. Raw logs: /mnt/storage/isos/onednn/results/ on the Tower.

## 1. One GEMM, y = x W^T, Qwen3.8-27B shapes (ms, lower is better)

GRIMOIRE today = the dequantize pass (`dequant_any`, FP8 -> bf16 VNNI at ~200 GB/s, BF16 re-layout
at ~255 GB/s) + `gemm_bf16_vnni`. TP prefill's chunked FFN pays the dequantize once per layer
(`predq`); every other projection pays it per call. Radiance = libr4d `gemm_fp8a8_*` on one AMD
R9700 (its docs/KERNELS.md, FP8 weights AND FP8 activations).

| M x N x K | weights | GRIMOIRE dequant + GEMM | oneDNN | Radiance R9700 |
|---|---|---:|---:|---:|
| 512 x 34816 x 5120 (gate_up) | FP8 row scale | 2.64 + 1.45 = **4.09** | **1.39** | 1.22 |
| 512 x 5120 x 17408 (down) | FP8 | 1.30 + 0.95 = **2.25** | **0.72** | 0.59 |
| 512 x 16384 x 5120 | FP8 | 1.18 + 0.61 = 1.79 | 0.63 | 0.53 |
| 512 x 12288 x 5120 | FP8 | 0.93 + 0.43 = 1.36 | 0.47 | 0.40 |
| 512 x 5120 x 6144 (o_proj) | FP8 | 0.45 + 0.32 = 0.77 | 0.27 | 0.20 |
| 2048 x 34816 x 5120 | FP8 | 2.64 + 5.92 = 8.56 | 5.28 | - |
| 2048 x 5120 x 17408 | FP8 | 1.35 + 2.84 = 4.19 | 2.66 | - |
| 4096 x 17408 x 5120 (TP rank gate_up) | FP8 | 1.34 + 5.88 = 7.22 | 5.30 | - |
| 4096 x 5120 x 8704 (TP rank down) | FP8 | 0.70 + 2.88 = 3.58 | 2.68 | - |
| 64 x 34816 x 5120 | FP8 | 2.64 + 0.74 = 3.38 | 0.37 | 0.31 |
| 512 x 34816 x 5120 | BF16 | 2.79 + 1.44 = **4.23** | **1.42** | - |
| 512 x 5120 x 17408 | BF16 | 1.39 + 0.93 = 2.32 | 0.76 | - |
| 2048 x 34816 x 5120 | BF16 | 2.79 + 5.85 = 8.64 | 5.73 | - |

GRIMOIRE's bf16 GEMM alone is as good as oneDNN's (512 x 34816: 1.24-1.45 vs 1.42 ms; 2048-4096
rows: equal); the dequantize pass is the whole gap. GRIMOIRE's GEMM is weaker on narrow N at 512
rows (N = 5120: 0.94 vs 0.72 ms -- too few 256x256 work-group tiles for 32 Xe cores).

## 2. oneDNN only (same harness)

| configuration | 512 x 34816 x 5120 | 2048 x 34816 x 5120 | 4096 x 17408 x 5120 | 1 x 34816 x 5120 |
|---|---:|---:|---:|---:|
| bf16 x FP8, one scale per 128 x 128 block (the official Qwen FP8 checkpoint) | 1.41 | 5.49 | 5.49 | 0.304 (586 GB/s) |
| bf16 x int8 weights, row scales (W8A16) | 1.38 | 5.26 | - | 0.305 |
| bf16 x int4, bf16 scale per 128 (W4A16) | 1.46 | 6.13 | - | 0.157 (567 GB/s) |
| bf16 x uint4 + u8 zero per 128 (GPTQ-like) | 1.79 | 7.28 | - | 0.161 |
| int8 x int8, per-token src + per-row weight scales (W8A8, LOSSY) | **0.74** (248 TOPS) | **2.67** (273) | **2.68** (273) | - |
| int8 x int4 (W4A8, LOSSY) | 0.77 | 3.03 | - | - |
| FP8 x FP8 (the B70 has no FP8 matrix unit: emulated) | 2.79 | 11.04 | - | - |
| `--attr-deterministic=true` (FP8 and BF16) | same as without | same | same | - |

## 3. Wired into GRIMOIRE (branch onednn-1025, opt-in GRIMOIRE_ONEDNN=1)

`src/onednn_gemm.{hpp,cpp}`, called from `gemm_fast_run` (and the TP chunked FFN's dequant/predq
pair) for FP8 row-scaled and BF16 weights. gemm_fast_probe through launch_gemm_fast, host check
PASS on every shape (FP8 1.1-1.9e-4 of sum|x*w| -- the reference rounds w*scale to bf16 like the old
path; oneDNN applies the scale to the fp32 sum), times = benchdnn's: FP8 512 x 34816 1.35 ms,
512 x 5120 x 17408 0.62, 4096 x 17408 5.48, 64 rows 0.37; BF16 512 x 34816 1.34.

## 4. Whole model, two B70s (Qwen3.8-27B-FP8, TP, no MTP, llama-benchy c1, --runs 2)

The first attempt (08:06) was stopped by dual.sh's link guard during the load: gpu1's root port
00:06.0 has logged card->host BadDLLP/BadTLP/RxErr ~1.3/s since 07:40 even idle (0 per run on
10-24). Ian chose to run anyway (09:0x, RP_WIN/RP_RUN raised; every other guard kept). Same new
binary (onednn-1025 6cbd37c), one row per setting, back to back (tools/bench/onednn/tp_onednn_ab.sh):

| | pp2048 | pp4096 | tg128 | code probe (2,153 tokens): TTFT cold / repeat, decode, tests |
|---|---:|---:|---:|---|
| GRIMOIRE_ONEDNN=1 | **2,717 +- 3** | **2,677 +- 2** | 31.5 / 30.9 | 1.39 / 0.83 s, 31.4 tok/s, 2/2 PASS |
| off (same binary) | 2,460 +- 1 | 2,537 +- 17 | 31.5 / 30.9 | 0.98 / 0.90 s, 31.5 tok/s, 2/2 PASS |
| 10-24 default (17c5d1f) | 2,494 | 2,494 | 31.5 / 30.9 | 0.98 / 0.90 s, 31.5 tok/s, 2/2 PASS |

+10.5% at pp2048, +5.5% at pp4096; decode unchanged. Link during the two rounds: root port rx
4,208 -> 8,933 -> 20,008 corrected (replay Timeout 5 -> 38), 0 uncorrectable, no drop.

## 5. The cold-TTFT cost, and what does not fix it

One primitive per (format, M, N, K): the FIRST prompt of each new length creates its
primitives (code probe TTFT 1.39 s cold vs 0.83 s repeated). Served prompts vary in length, so
this has to go before GRIMOIRE_ONEDNN=1 can be the default.
- Run-time M (DNNL_RUNTIME_DIM_VAL, one primitive per weight): REJECTED -- oneDNN serves it with
  `ocl:ref:any` (benchdnn mm4: FP8 512 x 34816 1,001 ms vs 1.39; BF16 466 ms vs 1.42).
- To try: primitives for row buckets (multiples of 256, created at load) run over padded rows, or
  oneDNN's (experimental) persistent kernel cache for Level Zero.
