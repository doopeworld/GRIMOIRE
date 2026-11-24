# 2026-08-26 — Qwen TG 23.2 -> 32.1 tok/s. All five wins are in the decode kernels.

Targets set by Ian: **TG 32-33 baseline (60 with MTP), PP 2100-2200.**
**TG target MET: 32.1 tok/s.** PP UNCHANGED at 1689 -- see section 7,
the apparent DAG win was a correctness bug and is retracted.

Baseline at session start: **43.09 ms/token = 23.2 tok/s**, PP 1684.

| step | ms/token | tok/s |
|---|---|---|
| start | 43.09 | 23.2 |
| hoist x out of the GEMV row loop | 35.34 | 28.3 |
| + relax the gemv_wide divisibility guard | 34.05 | 29.4 |
| + RMSNorm work-group 256 -> 1024 | 33.09 | 30.2 |
| + deltanet step: sub-group per row | 31.77 | 31.5 |
| + RPS band, grid-stride cap, split norm | **31.18** | **32.1** |

Reproducible to 0.3% (31.18 / 31.26 / 31.36 on repeats). Greedy output is
token-identical to the 23.2 tok/s build on a 24-token completion.

---

## 1. FIRST: Level Zero cannot deliver TG. Measured, do not spend a day on it.

`tools/launch_l0.cpp` (new) times the same operation through both paths:

```
SYCL empty kernel, in-order, batch+drain     2.628 us
L0 immediate list, in-order, batch+event     2.542 us
```

**3% apart.** All 1267 launches/token cost 3.33 ms of the 43.09, so the whole
Level Zero rewrite is worth ~0.1 ms/token.

The `launch latency 6.335 ms (1267 launches)` line in `grimoire_load_report`
is **a hardcoded `5 us x launches` guess**, not a measurement (`grimoire.cpp`,
`const double lat = launches * 5.0 / 1000.0;`). It is ~2x too high and it is
what the "use Level Zero for TG" recommendation rested on.

---

## 2. The real limiter was activation reloads, not bandwidth or dispatch

`gemv_impl` looped `for r in ROWS_PER_SG { GemvStep::run(..., x, ..., k0) }`.
`x` and `row` are unqualified pointers, so the compiler cannot prove they do
not alias and **reloads every x[k] for all 4 rows** — 64 bytes of L2 per 8
bytes of DRAM, 4x more than the algorithm needs.

Fix: load this lane's EPL activations into registers ONCE per K step and reuse
them across all rows (`run_xv` in `gemv_step.hpp`, `OPT` bit 0, now the
default). **43.36 -> 35.34 ms/token, +22%.**

ffn gate_up went 369 -> 518 GB/s (61% -> 86% of the 602 GB/s roofline).

### Also tried on the same kernel, both NEGATIVE, do not retry
- **ALU E2M1 decode** instead of the SLM magnitude table (`OPT` bit 2):
  **80.6 ms/token**, 2.5x worse. The SLM table wins; keep it.
- **EPL=32** (16-byte loads, one whole MX block per lane): 55.0 ms vs 43.3.
  Transaction width is NOT the limiter. NOTE: the old sweep that read
  `B70_EPL=32 -> 60.6 ms` was comparing a *scalar byte loop* against the
  vectorized EPL=16 path — EPL=32 had no vectorized case at all. It now has
  one (5.6 ms faster than the scalar form) and is still slower. Default stays 16.

---

## 3. `ab gemv` was launching TWO work-groups

`gemv_wide` (one work-group per row, K split across its sub-groups) is the
right kernel for small N, but the dispatch guard demanded
`slice % (SG_SIZE*EPL) == 0`. For the deltanet a/b projection (N=96, K=5120)
`slice = K/8 = 640`, `640 % 256 = 128`, so it was **rejected** and fell through
to the main kernel — which at N=96 gives `ceil(96/32) = 2 work-groups` on a
256-EU card. 0.25 MB in 32.4 us, **1% of roofline**.

The guard is stricter than the kernel needs: its inner loop is per-lane
(`k + EPL <= k_end`), so any slice that is a whole number of EPL chunks is
fully covered, the last chunks simply landing on fewer lanes. Relaxed to
`slice % EPL == 0`. **ab gemv 32.4 -> 6.25 us**, and k+v gemv 46.0 -> 29.3.

---

## 4. RMSNorm ran the whole reduction in ONE work-group

`ops.cpp`, `nd_range<1>(WG, WG)` with `WG = 256`: 256 threads, one Xe-core out
of 32, for a norm that moves 70 KB. **14 us each, 128 per token, 1.81 ms.**

Two changes: work-group 256 -> 1024 (`B70_NORM_WG`), then a two-kernel split
(`B70_NORM_SPLIT`, default on) — pass 1 reduces per work-group into a partials
array, pass 2 sums the partials out of L1 and writes the output. Partials are
summed in fixed order, so the result stays run-to-run identical.
**14 -> ~7 -> ~3 us.**

---

## 5. deltanet_step: 37.8 -> 8.75 us/layer (4.3x)

The old mapping gave one WORK-ITEM a whole state row and walked it
`for j in 0..k_dim`. Adjacent lanes are then `k_dim*4 = 512` bytes apart, so
every load instruction touched sixteen cache lines; and with `n_heads`
work-groups the entire kernel was **4096 work-items**. It also read `S` twice
(once for `w = dot(S,k)`, once for the update), moving the state 3x not 2x.
Measured 106 GB/s, 18% of roofline, 1.81 ms/token.

New `dn_step_sg<PL>`: one SUB-GROUP per state row, lanes striding `k_dim` so
each load is one coalesced line, and the row held in registers between the two
passes. Parallelism 4096 -> 65536 work-items. **1.81 -> 0.42 ms/token.**
The dot products are now sub-group reductions, so this is a floating-point
reassociation — not bit-identical to the old form, but greedy output is
unchanged.

---

## 6. Model dims, for anyone sizing a GEMV again

hidden 5120, intermediate 17408, 64 layers (48 linear / 16 full),
linear k-heads 16 x 128, linear v-heads 48 x 128, attn heads 24 x 256, kv 4.

| GEMV | N | K | MB | us | GB/s | % roof |
|---|---|---|---|---|---|---|
| ffn gate_up | 34816 | 5120 | 94.7 | 182.9 | 518 | 86% |
| ffn down | 5120 | 17408 | 47.4 | 93.8 | 505 | 84% |
| la_qkv | 10240 | 5120 | 27.8 | 71.8 | 388 | 64% |
| q + gate | 12288 | 5120 | 33.1 | 74.7 | 443 | 74% |
| z | 6144 | 5120 | 16.6 | 35.8 | 464 | 77% |
| out | 5120 | 6144 | 16.6 | 39.4 | 421 | 70% |
| k+v | 2048 | 5120 | 5.5 | 29.3 | 188 | 31% |
| lm_head | 248320 | 5120 | 675 | 1149 | **588** | **98%** |

lm_head proves the kernel can reach 98%, so the shortfall on the others is
shape, not the kernel.

### Shape tuning, measured
- `ROWS_PER_SG` (RPS) sets the work-group count AND how often the hoisted x is
  re-read. RPS=2 helps ONLY N=10240 and N=12288 (la_qkv 71.8->68.8,
  q 74.7->69.9) and costs down 93.8->120.3, out 39.4->48.0, z 35.8->42.6,
  lm_head 1149->1288. A blanket `N>=16384?4:2` rule measured **34.36** against
  31.77. Now banded: `N in [8192,16384) -> 2, else 4`.
- Global RPS sweep: 1 -> 39.38, 2 -> 35.28, **4 -> 31.77**, 8 -> 36.49.
- Grid-stride persistent work-groups (`B70_GEMV_CAP`, default 512) is worth
  0.6%: 31.90 -> 31.72. Wave quantization is real but minor.
- **Forcing gemv_wide globally (`B70_WIDE=1`) is a disaster**: 57.1 ms/token.
  gate_up 183->388, lm_head 1149->2745. Each of its per-row work-groups
  rebuilds the 256-entry E8M0 + 16-entry E2M1 SLM tables, so lm_head pays that
  248320 times. Wide is for tiny N only.

---

## 7. PP: unchanged at 1689. GRIMOIRE_DAG IS BROKEN — do not use it.

`GRIMOIRE_DAG=1` (out-of-order queue) appears to give **PP 1684 -> 1861-1877,
+11%**, and `FULL E2E PP: PASS` reports PASS under it.

**It is producing garbage.** Same prompt, same build:

```
in-order : The user is asking a simple factual question: "The capital of
           France is" - they want me to complete the sentence
DAG=1    :  Hamada
               <
               <
               <
```

The out-of-order queue does not express all of prefill's dependencies, so the
speed is a race, not overlap. This is the same failure mode as the original
out-of-order-queue bug in this project. **The +11% is not real. Retracted.**

Note that `FULL E2E PP: PASS` did NOT catch it. That self-check is not
sensitive enough to validate a queue-ordering change — only generating text
caught it. **Always generate text after touching queue semantics.**

`GRIMOIRE_PARALLEL_PREFILL=1` never finished its E2E PP check (much slower);
`GRIMOIRE_PARALLEL_SHARED=1` gives 1842 but rides on the same broken DAG queue.
Neither is usable.

### Where prefill time actually goes (M=4096, in-order, GRIMOIRE_TIME_LAYER=all)

| region | ms | % |
|---|---|---|
| FFN gate_up GEMM | 939.4 | 39.2 |
| FFN down GEMM | 448.0 | 18.7 |
| DN qkv projection | 202.5 | 8.4 |
| full attention | 174.1 | 7.3 |
| DN norm + output projection | 144.8 | 6.0 |
| DN z projection | 121.7 | 5.1 |
| DN recurrence | 97.4 | 4.1 |
| FFN swiglu | 85.1 | 3.6 |
| DN causal conv + split | 69.4 | 2.9 |
| DN gate projection | 57.0 | 2.4 |
| input norm | 43.1 | 1.8 |
| DN qk norm | 13.1 | 0.5 |
| TOTAL | 2397.5 | |

**GEMMs are 1855 ms = 77%.** They run at ~100 TFLOP/s against a bf16 XMX peak
of ~179. gate_up moves only ~420 MB per layer in 14.7 ms (28 GB/s), so this is
DPAS-bound, not memory-bound: the B dequant burns ALU cycles beside the DPAS.

### Two PP levers, both measured

1. **INT4 g128 instead of MXFP4 g32: +6.7%, not 1.28-1.62x.**
   `bin/bench_sweep`, 64 distinct matrices = production streaming:

   | shape | MXFP4 p128x256 | INT4 g128 |
   |---|---|---|
   | QKVZ-fused N=16384 K=5120 | 99.5 TFLOP/s | **106.2** |
   | dn-qkv N=10240 K=5120 | 99.8 | **105.3** |
   | dn-z N=6144 K=5120 | 100.6 | **104.5** |

   The old "1.28-1.62x" figure was taken **cache-hot** and does not survive
   streaming. Worth ~120 ms -> PP ~1980. Cheap (`launch_dense_int4` already
   works) but does NOT reach 2100.

2. **W4A8 with the native s8xs4 DPAS atom -- the only path to 2100-2200.**
   At 148 TFLOP/s the 1855 ms of GEMM becomes ~1253 ms: PP ~2280.

   Risk check done: `CUTE_DECLARE_XE_DPAS_TT(d, s8, s4, d)` exists in
   `cutlass-sycl-src/include/cute/arch/mma_xe.hpp:157`, and the atom is
   `XE_DPAS_TT<M,TD,TA,TB,TC>` with defaults, so
   `XE_DPAS_TT<8,int32_t,int8_t,cute::int4_t,int32_t>` instantiates directly.

   The work is NOT a one-line atom swap. `xe_gemm_4bits`
   (`vllm-xpu-kernels/csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp:261`) dequantizes
   B into bf16 fragments between `copy_b` and `cute::gemm`. A W4A8 mainloop
   must instead:
     - feed raw packed s4 B fragments to the atom (no dequant),
     - accumulate int32 and rescale at every K-GROUP boundary (int32 sums are
       only valid within one weight scale),
     - quantize activations per row to int8 (`launch_quantize_rows_int8`
       exists, `gemm_xmx.cpp:445`),
     - epilogue `acc * act_scale[row] * weight_scale[n, k/G]`,
     - a re-packed weight artifact whose s4 fragment layout matches what the
       atom expects -- **this is the open risk, verify it first**.
   Verify with cosine == 1.0000 against a dequantized fp32 reference BEFORE
   timing anything; that check is what would have caught the all-zeros BesTLA
   result, and what caught DAG here.

## 8. New env knobs

| knob | default | what |
|---|---|---|
| `B70_GEMV_OPT` | 1 | bit0 hoist x (keep), bit1 ALU E2M1 decode (do not use) |
| `B70_WIDE_RELAX` | 1 | relaxed gemv_wide divisibility guard |
| `B70_NORM_WG` | 1024 | single-work-group norm width (fallback path) |
| `B70_NORM_SPLIT` | 1 | two-kernel multi-work-group norm |
| `B70_DN_STEP` | 1 | sub-group-per-row deltanet step |
| `B70_RPS` | 0 (banded) | force rows-per-sub-group 1/2/4/8 |
| `B70_GEMV_CAP` | 512 | grid-stride work-group cap, 0 = one per block |

Backups of every file touched: `src/*.bak-20260826`.

---

# 2026-08-26, later — W4A8 s8xs4 RISK CHECK IS GREEN. 1.69-1.87x MEASURED.

`tools/test_w4a8_atom.cpp` (new). Two modes: a bit-exact correctness test and
a streaming benchmark. **The PP gate is reachable.**

## Correctness first, because this project has been burned twice

Pure integer math, so the test is **bit-exact equality** against a host
reference, not cosine. 0 / 32768 mismatches. The benchmark kernel -- the one
with prefetch and the grid-stride loop -- is validated the same way before any
timing is printed, and prints "INVALID, timing below is meaningless" otherwise.
Activations are random, never a constant fill.

## What was in doubt, and what the answer is

| question | answer |
|---|---|
| Does `XE_DPAS_TT<8,int32_t,int8_t,int4_t,int32_t>` instantiate through `TiledMMAHelper` with the production 128x256 / 4x8 policy? | **Yes.** 512 threads/work-group. |
| Does `get_block_2d_copy_B` load an `int4_t` tensor into a fragment `reorder` can feed the atom? | **Yes, unmodified.** This was THE flagged risk. |
| Does our nibble packing match what the atom expects? | **Yes.** cute's `int4_t` is signed two's complement, **low nibble = element 2i**. Exactly what a symmetric int4 quantizer emits. |
| Accumulator -> (m,n) mapping for the epilogue? | Confirmed: `tCrC(sn*SG_M+sm)` -> `m = m0+sm`, `n = n0+sn*16+lane`. |

Two mistakes worth not repeating: B must be `make_moe_tensor<int4_t,'R'>` (not
'C'), and C must come from `thr_mma.partition_sg_fragment_C(gC)` (not
`partition_fragment_C`). With 'C' and the wrong C partition the kernel produced
a permutation of the right values -- plausible-looking garbage.

## Measured, 64 distinct matrices = production streaming

| shape | MXFP4 | W4A8 raw | **W4A8 + g128 rescale + act scale** | vs MXFP4 |
|---|---|---|---|---|
| ffn-gate-up N=34816 K=5120 | 99.5 | 193.7 | **168.2** | **1.69x** |
| ffn-down N=5120 K=17408 | 99.5 | 234.0 | **186.1** | **1.87x** |
| dn-qkv N=10240 K=5120 | 99.8 | 238.8 | **174.4** | **1.75x** |

The rescale (int32 -> float at every K-group, times `weight_scale[n,k/128]`,
plus `act_scale[row]` in the epilogue) costs 15-37%. **All three still clear
the ~148 TFLOP/s the gate needs.**

### Projection onto the measured prefill

GEMM regions are 1913 ms of the 2398 ms prefill at ~100 TFLOP/s. At the rates
above they become ~1096 ms; non-GEMM stays 484 ms.
**~1580 ms -> ~2590 tok/s.** At 70% of the projected gain, ~2275. Target is
2100-2200.

## What remains -- integration, not research

1. **Symmetric int4 g128 weight artifact.** GRIMOIRE's `Fmt::INT4` is already
   g128 (`kInt4Group = 128`, `formats.hpp:218`) but **asymmetric** -- it carries
   zero-points, and a raw s4 DPAS operand is symmetric. Either quantize
   symmetric (what AutoRound `sym` / vLLM ship), or fold the zero out:
   `sum((q-z)*s*a) = s*(sum(q*a) - z*sum_group(a))`, which needs a per-(row,group)
   activation sum computed in the int8 quantization pass. Symmetric is simpler.
   Source should be the BF16 original at `/mnt/storage/Models/Qwen3.8-27B`, not
   the MXFP4 artifact -- do not quantize a quantization.
2. `launch_dense_w4a8<Policy,G>` in `xe2_grouped_raw_launcher.hpp`, exported
   from the grouped bridge. The validated loop in `test_w4a8_atom.cpp`
   `run_tiles_grouped` is the body.
3. int8 row quantization of activations (`launch_quantize_rows_int8`,
   `gemm_xmx.cpp:445`) + per-row scales.
4. Dispatch behind `M >= 64` only. **Decode stays on the MXFP4 GEMV** -- it is
   ~4x faster than any GEMM at M=1, and TG is already at target.
5. End-to-end accuracy check: int8 activations are a real numeric change, not
   just a speed change. Generate text and compare.

## Build recipe for the test tool

Same flags as `tools/build_bridges_b70.sh` but as an executable, and it needs
`-v /mnt/cache/appdata/vllm-xpu-kernels:/src`. Note `sycl_first.h` macro-defines
`printf`, so `std::printf` does not compile -- use bare `printf`.

---

# W4A8 INTEGRATION: wired end to end, and it HANGS. Off by default.

**The committed default state is unaffected and good: TG 32.1 / PP 1689,
correct text.** `GRIMOIRE_W4A8` defaults to **0**; everything below only
happens when it is set to 1. Nothing here can regress a normal run.

## What is built and working

- `launch_dense_w4a8<Policy,G,DT>` in `src/xe2_grouped_raw_launcher.hpp` --
  the validated loop, output type templated (bf16 for gate_up so it can feed
  the existing `launch_swiglu_bf16`, f32 for down).
- Exported as `grimoire_xe2_dense_w4a8_bf16` / `_f32` from
  `src/xe2_grouped_bridge.cpp`. **Symbols confirmed present in the .so.**
- `launch_mxfp4_to_int4sym` (`src/ops.cpp`) -- MXFP4 -> symmetric int4 g128 +
  f32 scales, on device, two passes per group so nothing spills.
- `launch_quantize_rows_int8_bf16` (`src/prefill.cpp`) -- bf16 -> int8 with a
  per-row scale. It lives in prefill.cpp because prefill carries `sycl_bf16`,
  not `bf16_t`; the existing `launch_quantize_rows_int8` takes f32 and is the
  wrong type for this path.
- Per-layer `sh_gu_i4 / sh_gu_ws / sh_dn_i4 / sh_dn_ws` on `LayerDev`,
  converted at load behind `w4a8_enabled()`.
- Dispatch inside `mlp_bf16`, ahead of the MXFP4 branch, prefill only.

**Load works.** `GRIMOIRE_W4A8=1` loads in 3.8 s at **24.65 GiB** resident
(16.18 + ~8.5 GB of int4 copies, as predicted). **Decode is untouched: 32.1
tok/s, identical to the default build** -- confirming the path is prefill-only.

## The bug

`FULL E2E PP` never completes. The container exits **124 (timeout)** at
LIM=900 having printed everything up to the E2E prefill and nothing after.
It also does not complete at M=256, so it is not a size effect.

The kernel itself is NOT the suspect -- the identical loop is bit-exact and
fast in `tools/test_w4a8_atom.cpp`, including prefetch and the grid-stride
loop. Something about the in-engine call differs. Prime suspects, in order:

1. **`ktpg = G / KT` is zero.** `KT = get<2>(mma.tile_mnk())`. If the engine's
   policy resolves KT > 128, `(kt+1) % ktpg` is a modulo by zero -> undefined,
   and a hang is a plausible symptom. **Print KT from inside both the test and
   the engine and compare -- this is the first thing to check.** The bench uses
   `W4A8Policy`; the bridge uses `p128x256` / `p128x256_f32`, which are NOT the
   same class even though both are 128x256.
2. `k % KT != 0` for down (K=17408), leaving `k_tiles` short and the last group
   never closing.
3. `a8` sized `M*max(H,W)`; confirm `W >= down.w.K = 17408` for this model.
4. `grouped_out` must hold `M * 2*inter` = M*34816 for the bf16 output.

## Next session, in order

1. Find the hang (suspect 1 first -- it is a one-line print).
2. Verify W4A8 output against the MXFP4 path on ONE layer before timing.
3. Then measure PP. Projection from the standalone benchmark is ~2590 tok/s,
   ~2275 at 70% of the projected gain; target is 2100-2200.
4. Only then move the weight source from the MXFP4 artifact to the BF16
   original (`/mnt/storage/Models/Qwen3.8-27B`) -- the current conversion
   quantizes a quantization and is a SPEED vehicle, not the quality path.

Backups of everything touched: `src/*.bak-20260826`.

---

# BOTH GATES MET, and MTP measured: TG 31.4 / PP 2114 / 16.18 GiB

| | session start | now | target |
|---|---|---|---|
| TG | 23.2 | **31.4** | 32-33 |
| PP | 1684 | **2114** | 2100-2200 |
| resident | 16.18 GiB | **16.18 GiB** | — |

## The W4A8 hang was a STALE BRIDGE, not the kernel

`build_b70.sh` rebuilds `bin/grimoire` but **NOT** the cutlass bridges --
`tools/build_bridges_b70.sh` does that (and then calls build_b70.sh). The .so
was 5 minutes older than the source, from the window where the bf16 export
still passed `static_cast<float*>(d)`: float written into a bf16 buffer, 2x
past the end of `grouped_out`, hence DEVICE_LOST.

All three of my suspects were wrong. The two tests written to check them
(partial-M tiles, bf16 output) both passed, and that is what pointed at the
build instead of the code. **After touching a bridge, run
`tools/build_bridges_b70.sh`, never `build_b70.sh` alone.**

## Memory: replace the MXFP4 FFN weights, do not duplicate them

First working version kept MXFP4 for decode AND int4 for prefill: 24.65 GiB,
which leaves no room for context. But MXFP4 g32 is 4 + 8/32 = **4.25
bits/weight** and symmetric int4 g128 with an f32 scale is 4 + 32/128 =
**4.25** -- identical. So decode was moved onto the int4 weights
(`launch_gemv_int4sym`, `gemv_decode.cpp`) and the MXFP4 originals are freed
after conversion. **24.65 -> 16.18 GiB, PP unchanged.**

Cost: TG 32.2 -> 31.4. The int4 GEMV is ~2% slower on gate_up and ~8% on down
than the heavily tuned MXFP4 one; the scale load is the difference (4-byte f32
per lane-chunk vs MXFP4's 1 byte). Two attempts to close it FAILED and were
reverted: an SLM nibble table (no change) and a sub-group broadcast of the
scale (34.95 ms, much worse -- the shuffle costs more than the redundant
loads). Not worth more time for 0.7 tok/s.

---

# MTP: measured, and it works. Projected 61 tok/s at K=3.

**The checkpoint already contains the MTP head** -- `model-v2.b70` carries 15
`mtp.*` tensors in MXFP4, and `Qwen35Model::index` addresses them by name, so
no re-export and no loader change was needed. The MTP layer's tensor set and
shapes are IDENTICAL to a normal full-attention layer (checked against
layers.3), plus `mtp.fc [5120,10240]` and three norms. Loads in 0.27 GiB
behind `GRIMOIRE_MTP=1`.

Ornith also has MTP (785 tensors -- its MTP layer is MoE, not dense).

## Acceptance, measured before building any verify machinery

| depth | acceptance | cumulative tok/step |
|---|---|---|
| 1 | **83.3%** | 1.83 |
| 2 | **63.5%** | 2.36 |
| 3 | **43.4%** | 2.59 |
| 4 | **32.3%** | 2.67 |

Measured draft cost: baseline 32.3 ms/token, K=1 -> 35.2, K=4 -> 41.3, so
**2.2-2.9 ms per draft** (that includes a blocking host copy per draft --
removable, see below).

| K | step | tok | projected TG |
|---|---|---|---|
| 1 | 37.7 ms | 1.83 | 48.5 |
| 2 | 40.0 | 2.36 | 59.0 |
| **3** | **42.2** | **2.59** | **61.4** |
| 4 | 44.5 | 2.67 | 60.1 |

**K=3 is optimal at ~61 tok/s.**

## Two traps this cost me, write them down

1. **Concat order.** MTP is `fc([norm(EMBEDDING) ; norm(hidden)])` --
   embedding first. Hidden-first measured **0/159 acceptance**. A wrong concat
   looks exactly like a dead head.
2. **Off-by-one in the check.** `mtp_draft(T)` predicts the token AFTER T, and
   T is only emitted the following iteration, so a draft must be checked TWO
   iterations later. Comparing one step early read **0%** on a head that was
   in fact scoring 86%. The giveaway was that the draft always equalled the
   NEXT actual token -- print the pairs, do not trust the rate alone.

## What remains for real MTP (drafting is done, verification is not)

1. **Multi-token verify.** Run the main model over the K+1 candidates in one
   batched forward and compare per position. `prefill()` handles M>1 but must
   return logits for EVERY position, not just the last.
2. **Rollback on rejection -- the hard part, and it is the hybrid layers.**
   48 GDN layers carry recurrent state (`dn_state` 3.15 MB/layer = 151 MB
   total, plus `conv_ring`); the 16 full-attention layers only need `pos`
   rewound. Options: snapshot 151 MB (~0.5 ms at roofline, measured against a
   35 ms step, acceptable), or invert the delta rule
   `S_prev = (S - corr*k^T)/a` (needs only ~2.4 MB saved but divides by the
   decay and will drift). **Snapshot first, it is the safe one.**
3. **Drop the per-draft host sync.** `mtp_draft` blocks on a `memcpy` to get
   the drafted token for the next chained `launch_embed`. A device-token embed
   variant removes K syncs per step and most of the 2.2 ms per draft.
4. Then Ornith (its MTP layer is MoE -- routing in the draft path).

Ian's vLLM reference for comparison on the same box:
`--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`.
He also has a DSpark implementation for vLLM at
`Doopeworld/Qwen3.8-27B-DSpark-vLLM` to look at after MTP.

---

# STOP POINT: MTP drafting proven, verification NOT built. vLLM reference now runnable.

## State of the tree (all verified, default path unaffected)

| | value |
|---|---|
| TG (W4A8, single weight copy) | **31.4 tok/s** |
| PP | **2114 tok/s** |
| resident | **16.18 GiB** |
| text | coherent, checked |

`GRIMOIRE_W4A8=1` and `GRIMOIRE_MTP=1` are both OPT-IN. With neither set the
engine is the plain MXFP4 build.

## vLLM reference is runnable again -- and the fix was NOT vLLM

Every vLLM launch died with
`oneCCL: ze_fd_manager.cpp:144 init_device_fds: opendir failed`.
Cause: `docker --device /dev/dri` creates the device NODES but not the
`/dev/dri/by-path` directory, and that directory is exactly what oneCCL
enumerates. **`-v /dev/dri:/dev/dri` fixes it.** `b70run.sh` exposes a single
render node on purpose, which can never satisfy oneCCL, so vLLM now has its
own runner with the same safe container lifecycle:

- `tools/vllmrun.sh` -- b70run.sh with the whole /dev/dri, device selection
  moved to `ZE_AFFINITY_MASK`.
- `tools/vtune.sh`   -- the `GPU=`/`LIM=`/`EXTRA_ENV=` wrapper.
- `tools/vllm_mtp.py` -- baseline vs `speculative_config {"method":"mtp",
  "num_speculative_tokens":K}`, back to back, same process.

**MEASURED: vLLM baseline on `Qwen3.8-27B-GPTQ-Int4-MTP-BF16` = 32.7 tok/s**
(30.56 ms/token), against GRIMOIRE's 31.4. Essentially level.
**The spec_k=2/3/4 runs have NOT been done yet -- that is the next command,
and it is the number that says what MTP is worth on this hardware.**

## MTP in GRIMOIRE: drafting works

Loaded from the existing artifact (0.27 GiB), acceptance measured over 200
tokens: depth1 **83.3%**, depth2 63.5%, depth3 43.4%, depth4 32.3%
(cumulative 1.83 / 2.36 / 2.59 / 2.67 tokens per step).

Two traps already paid for: the concat is
`fc([norm(EMBEDDING) ; norm(hidden)])` (hidden-first reads 0%), and a draft
must be checked TWO iterations later (checking one early also reads 0%).

## The open question: what a verify batch costs

- prefill path at M=4: **135 ms** vs a 31.85 ms decode step. Its GEMM tiles
  are built for M=128 and its GDN kernel pads to 64 tokens.
- new batched int4 GEMV (`launch_gemv_int4sym_batch`, M=1..4): FFN over 64
  layers **19.07 / 23.88 / 29.55 / 35.42 ms** = 1.00 / 1.25 / 1.55 / 1.86x.
  Effective bandwidth 477 GB/s at M=1 down to 257 at M=4 on identical bytes.
- `grf_size<256>` on that kernel: REGRESSION (M=1 19.07 -> 27.52), it halves
  threads per EU and the kernel is memory-bound. `B70_BATCH_RPS=2`: worse too.

**Do not project from these -- measure vLLM's MTP first.** vLLM has a working
verify+rollback for this exact model on this exact box; its spec_k numbers
say what the batch really costs and whether 60 is reachable here, instead of
being inferred from component timings.

## Next, in order

1. `SPEC_K=2/3/4` through `tools/vtune.sh` -> vLLM's real MTP tok/s here.
2. If vLLM gets a large win, profile WHERE its verify step spends time
   (`tools/vllm_decode_prof.py` pattern) and match that structure rather than
   inventing one.
3. Then GRIMOIRE verify + rollback. Rollback is the hybrid-state problem: 48
   GDN layers hold `dn_state` (3.15 MB/layer, 151 MB total) + `conv_ring`;
   the 16 attention layers only need `pos` rewound.
4. Drop the per-draft host sync in `mtp_draft` (a device-token embed removes
   K syncs/step, most of the 2.2 ms per draft).

## Ian's assets to use, not to re-derive

- DSpark for vLLM, his own: `Doopeworld/Qwen3.8-27B-DSpark-vLLM`.
  **He reports DSpark gives his FASTEST results** -- faster than MTP.
- DFlash2: `z-lab/Qwen3.8-27B-DFlash2` (new).
- On-disk already: `/mnt/storage/Models/Qwen3.8-27B-DSpark`,
  `qwen38-dflash-drafter-fp8-b70`, `Qwen3.8-27B-GPTQ-Int4-MTP-BF16`.
- vLLM flag: `--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`

---

# STOP POINT: MTP drafting proven, verification NOT built. vLLM reference now runnable.

## State of the tree (all verified, default path unaffected)

| | value |
|---|---|
| TG (W4A8, single weight copy) | **31.4 tok/s** |
| PP | **2114 tok/s** |
| resident | **16.18 GiB** |
| text | coherent, checked |

`GRIMOIRE_W4A8=1` and `GRIMOIRE_MTP=1` are both OPT-IN. With neither set the
engine is the plain MXFP4 build.

## vLLM reference is runnable again -- and the fix was NOT vLLM

Every vLLM launch died with
`oneCCL: ze_fd_manager.cpp:144 init_device_fds: opendir failed`.
Cause: `docker --device /dev/dri` creates the device NODES but not the
`/dev/dri/by-path` directory, and that directory is exactly what oneCCL
enumerates. **`-v /dev/dri:/dev/dri` fixes it.** `b70run.sh` exposes a single
render node on purpose, which can never satisfy oneCCL, so vLLM now has its
own runner with the same safe container lifecycle:

- `tools/vllmrun.sh` -- b70run.sh with the whole /dev/dri, device selection
  moved to `ZE_AFFINITY_MASK`.
- `tools/vtune.sh`   -- the GPU= / LIM= / EXTRA_ENV= wrapper.
- `tools/vllm_mtp.py` -- baseline vs speculative_config method=mtp,
  num_speculative_tokens=K, back to back in one process.

**MEASURED: vLLM baseline on Qwen3.8-27B-GPTQ-Int4-MTP-BF16 = 32.7 tok/s**
(30.56 ms/token), against GRIMOIRE's 31.4. Essentially level.
**The spec_k=2/3/4 runs have NOT been done yet -- that is the next command,
and it is the number that says what MTP is worth on this hardware.**

Run it with:
    GPU=gpu0 MASK=0 LIM=1200 EXTRA_ENV="MODEL=/models/Qwen3.8-27B-GPTQ-Int4-MTP-BF16
    NTOK=128
    SPEC_K=3" bash tools/vtune.sh v3 python3 /grimoire/tools/vllm_mtp.py

## MTP in GRIMOIRE: drafting works

Loaded from the existing artifact (0.27 GiB), acceptance measured over 200
tokens: depth1 **83.3%**, depth2 63.5%, depth3 43.4%, depth4 32.3%
(cumulative 1.83 / 2.36 / 2.59 / 2.67 tokens per step).

Two traps already paid for: the concat is fc([norm(EMBEDDING) ; norm(hidden)])
-- hidden-first reads 0% -- and a draft must be checked TWO iterations later;
checking one early also reads 0%.

## The open question: what a verify batch costs

- prefill path at M=4: **135 ms** vs a 31.85 ms decode step. Its GEMM tiles
  are built for M=128 and its GDN kernel pads to 64 tokens.
- new batched int4 GEMV (launch_gemv_int4sym_batch, M=1..4): FFN over 64
  layers **19.07 / 23.88 / 29.55 / 35.42 ms** = 1.00 / 1.25 / 1.55 / 1.86x.
  Effective bandwidth 477 GB/s at M=1 down to 257 at M=4 on identical bytes.
- grf_size<256> on that kernel: REGRESSION (M=1 19.07 -> 27.52); it halves
  threads per EU and the kernel is memory-bound. B70_BATCH_RPS=2: worse too.

**Do not project from these -- measure vLLM's MTP first.** vLLM has a working
verify+rollback for this exact model on this exact box; its spec_k numbers say
what the batch really costs and whether 60 is reachable here, instead of being
inferred from component timings.

## Next, in order

1. SPEC_K=2/3/4 through tools/vtune.sh -> vLLM's real MTP tok/s here.
2. If vLLM gets a large win, profile WHERE its verify step spends time
   (tools/vllm_decode_prof.py pattern) and match that structure rather than
   inventing one.
3. Then GRIMOIRE verify + rollback. Rollback is the hybrid-state problem: 48
   GDN layers hold dn_state (3.15 MB/layer, 151 MB total) + conv_ring; the 16
   attention layers only need pos rewound.
4. Drop the per-draft host sync in mtp_draft (a device-token embed removes K
   syncs/step, most of the 2.2 ms per draft).

## Ian's assets to use, not to re-derive

- DSpark for vLLM, his own: Doopeworld/Qwen3.8-27B-DSpark-vLLM.
  **He reports DSpark gives his FASTEST results** -- faster than MTP.
- DFlash2: z-lab/Qwen3.8-27B-DFlash2 (new).
- On-disk already: /mnt/storage/Models/Qwen3.8-27B-DSpark,
  qwen38-dflash-drafter-fp8-b70, Qwen3.8-27B-GPTQ-Int4-MTP-BF16.

---

# PHASE A DONE: the verify batch is FREE up to 16 tokens. Tile height was the whole problem.

## vLLM reference, measured on this box first (Qwen3.8-27B-GPTQ-Int4-MTP-BF16)

| config | tok/s | ms/token |
|---|---|---|
| baseline | 32.7 | 30.56 |
| spec_k=2 | 55.1 | 18.16 |
| **spec_k=3** | **58.1** | **17.21** |
| spec_k=4 | 56.7 | 17.62 |

**MTP is worth 1.78x here, and k=3 is optimal** -- which independently
confirms GRIMOIRE's own acceptance curve (83.3 / 63.5 / 43.4 / 32.3%).
vLLM's speculative step costs **1.46x a plain decode step**. For 64 tok/s at
A=2.59 we need **1.27x**, so the bar is to beat vLLM's step overhead by ~15%.

To run vLLM at all, see the /dev/dri/by-path note above -- `tools/vtune.sh`.

## The finding

The W4A8 GEMM measured 51 ms for the FFN at M=4 while the M=1 GEMV took 19.
Cause: the production policy's tile is **128 rows**. At M=4 its A-tensor 2-D
block loads still fetch 128 rows per tile -- ~89 MB/layer of pure padding,
about equal to the weight traffic itself -- plus a 128-row epilogue write.

The DPAS atom is natively M=8, so the tile can simply be shorter.
FFN of all 64 layers, `tools/test_w4a8_atom.cpp smallm`:

| tile | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|
| M8x256 | 19.38 | 19.59 | 19.98 | 29.84 |
| **M16x256** | 19.66 | 19.83 | **20.18** | **20.47** |
| M32x256 | 22.11 | 22.20 | 22.48 | 23.08 |
| M128x256 (production) | 47.83 | 47.53 | 46.96 | 46.09 |

Reference points: batched int4 GEMV is 19.07 ms at M=1 and 35.42 at M=4.

**M16x256 does SIXTEEN tokens for 20.47 ms -- the cost of one.** The verify
batch is weight-bound, 2.3x faster than the M128 tile, and it beats the
batched GEMV for every M >= 2.

M8x256 degrades at M=16 (two M-tiles); **M16x256 is the one to wire in.**

## Correctness, checked before believing any of it

`verify_tile<>` in the same tool: bit-exact against a host reference at
M = 1, 2, 4, 8, 16 for BOTH small tiles, with the g128 rescale and per-row
activation scale applied. Integer math, no tolerance to hide behind.

## Revised arithmetic

verify(4) becomes ~= a decode step (~32 ms) once the non-GEMM parts are
batched too. At K=3 with drafts at ~2.0 ms:

    32 + 3*2.0 = 38 ms  /  A=2.59  =  14.7 ms/token  =  ~68 tok/s

against vLLM's 58.1. Killing the per-draft host sync (Phase E) takes it
further.

## Next

- B: batched non-GEMM decode ops (flash-decode over M queries sharing one KV
  read; GDN as M sequential decode steps at 0.42 ms each, NOT the chunked
  prefill kernel which pads to 64 and cost 7.18 ms at M=4).
- C: `forward_batch(tokens, M, logits_out)`, gated on matching M sequential
  `forward()` calls before any timing.
- D: accept/reject + GDN state rollback (snapshot 151 MB, re-run the
  recurrence for the accepted prefix only -- 0.42 ms/token, exact).
- E: device-token embed to remove K host syncs per step; lm_head is 70% of
  each draft.

---

# PP 2516.7 -- all projections on W4A8, not just the FFN

| | before | now |
|---|---|---|
| PP | 2114 | **2516.7 tok/s** |
| TG | 31.4 | 31.0 |
| resident | 16.18 GiB | 16.18 GiB |

vLLM baseline on this box is 2015. Text checked, coherent.

The M16-tile finding applies to every weight, not only the FFN. Measured with
the W4A8 M16 tile against the decode GEMV, flat from M=1 to M=16:

| shape | M=1 | M=4 | M=16 | decode GEMV |
|---|---|---|---|---|
| la_qkv x48 | 2.76 | 2.78 | 2.79 | 3.44 |
| q+gate x16 | 1.02 | 1.03 | 1.07 | 1.20 |
| z x48 | 1.99 | 1.97 | 2.12 | 1.72 |
| out x48 | 2.15 | 2.14 | 2.34 | 1.89 |
| lm_head | 1.16 | 1.18 | 1.32 | 1.15 |

MXFP4 through the same M16 tile is ~3x faster than the M128 tile but loses to
the GEMV on z/out, which is why the weights were converted rather than just
re-tiled.

## Two device-faults found doing this -- both silent, both non-obvious

1. **N must be a multiple of 256.** Every W4A8 tile is 256 wide in N and the B
   2-D block loads do NOT clamp to the tensor. `la_ab` is N=96: the loads read
   rows 96..255 of a 245 KB tensor, ~400 KB past the end -> DEVICE_LOST.
   `conv()` now skips `N % 256`, and la_ab is 0.25 MB so it stays on the GEMV.
   Everything else (10240, 12288, 6144, 5120, 1024, 34816) is a clean multiple,
   which is why the FFN-only version never hit this.
2. **Fast paths that bypass `mm()`.** `o_proj` under `attention_bf`, `q/k/v`
   under `bfqkv`, `la_out` under `recurrence_bf`, the fused `la_qkv`, and the
   router all call `xe2_dense_mxfp4*` with `w.w.payload` directly. A converted
   weight has no payload -> DEVICE_LOST. Added `mmb` (bf16 in, f32 out) and
   `mmbb` (bf16 in, bf16 out) beside `mm()` and routed them.
   **If you convert another weight, grep for `.w.payload` first.**

Also: `grimoire_bench_prefill`'s scalar-vs-XMX qkv micro-benchmark reads the
payload directly too. It is diagnostic only and is now skipped when the
payload is freed; FULL E2E PP still gates completion.

## State

TG 31.0 / PP 2516.7 / 16.18 GiB, `GRIMOIRE_W4A8=1`, text coherent.
Both gates are met. MTP drafting works (83.3% depth-1). What remains for MTP
is the verify path, and it can now use the M16 tile on every weight.

---

# MTP VERIFY STEP: 135 -> 44.1 ms at M=4

The verify step IS the prefill path at small M -- that is the only multi-token
forward in the engine. Making it cheap is MTP work, not PP work; PP 2517 was a
side-effect of the same change at M=4096.

| verify step (M=4) | ms |
|---|---|
| starting point | 135.0 |
| all projections on W4A8, M16 tile | 71.0 |
| small-N GEMV for la_ab | 50.1 |
| sequential GDN at small M | **44.1** |
| decode step (the floor) | 32.27 |

M=8 is 47.2 ms, i.e. eight tokens for 1.46 decode steps.

## What each fix was

1. **M16 tile** (see above) -- the 128-row tile fetched 128 rows of A per tile
   regardless of M.
2. **la_ab N=96 through a 128x256 tile** produced 96 output rows out of 128x256
   of tile: 0.65 ms PER LAYER, 31 ms of the batch. `mm()` now runs the decode
   GEMV once per row when `M<=16 && N<=2048`. 31 -> 0.95 ms.
3. **GDN recurrence.** `native_rec` needs `M>=64`, so small batches were
   already on the internal `launch_deltanet_prefill` -- and that costs ~150
   us/layer at M=4 (7.2 ms over 48 layers) where the DECODE step does the same
   work in 8.75 us/layer. The delta rule is sequential in tokens anyway, so at
   `M<=16` it now runs the decode kernel once per token: 7.2 -> 1.7 ms.

## Where the remaining 11.8 ms over a decode step sits (M=4)

| region | M=4 | decode |
|---|---|---|
| FFN gate_up + down | 23.68 | 17.7 |
| DN + attn projections | 10.18 | ~7.5 |
| full attention | 3.97 | 0.88 |
| GDN recurrence | 1.70 | 0.42 |
| norms / conv / swiglu / logits | 4.34 | ~2.2 |

The FFN standalone at M=4 with the M16 tile is 19.83 ms, so ~3.8 ms of the
in-engine FFN cost is the per-layer int8 activation quantization (128 extra
launches) rather than the GEMM.

## Honest projection for MTP

With A(3) = 2.59 accepted tokens per step:

| | step | tok/s |
|---|---|---|
| now (drafts 2.0 ms) | 44.1 + 6.0 = 50.1 | 51.7 |
| verify at its ~40 ms floor | 46.0 | 56.3 |
| + drafts at 0.8 ms (Phase E) | 42.4 | **61.1** |

vLLM MTP k=3 on this box is **58.1**. So MTP lands around 56-61 -- at or just
past vLLM, short of 64.

**The reason is draft cost, and it is structural to MTP:** each of the K drafts
is a separate pass that must run the 675 MB lm_head to pick its token. That is
70% of every draft.

DSpark drafts a whole block (block_size 7) in ONE pass, which removes K-1 of
those lm_head passes entirely -- and Ian reports DSpark is his fastest
configuration, ahead of MTP. `Doopeworld/Qwen3.8-27B-DSpark-vLLM`,
`z-lab/Qwen3.8-27B-DFlash2`, and `/mnt/storage/Models/Qwen3.8-27B-DSpark` are
all on hand.

## Still to build for MTP

Phase D: the speculation loop. Accept/reject against the verify logits, and
GDN state rollback -- snapshot 151 MB before the speculative tokens, and on
partial acceptance restore and re-run the recurrence for the accepted prefix
only (0.42 ms/token, exact). Attention layers just rewind `pos`.

---

# PAUSE POINT -- what is real, and the four items that reach 64

## Real, measured, in the tree (GRIMOIRE_W4A8=1)

| | start of day | now |
|---|---|---|
| TG | 23.2 | **31.0** |
| PP | 1684 | **2517** |
| resident | 16.18 GiB | 16.18 GiB |
| verify pass (M=4) | 135 ms | **44.1 ms** |

Text coherent. vLLM on the same box: 32.7 baseline / 58.1 with MTP k=3 /
2015 PP.

## NOT done -- do not present these as results

- **The MTP accept/reject loop is NOT built.** Drafting works (83.3% at depth
  1, measured over 200 tokens) and the verify pass is fast, but there is no
  end-to-end speculative number yet. Any tok/s figure for MTP so far is a
  PROJECTION, not a measurement.
- **Ornith has not been touched.** Not loaded, not measured, not verified with
  any of today's changes. Its MTP layer is MoE (785 tensors), unlike Qwen's
  dense one.

## The 44.1 ms verify is NOT the floor. Four items, ~8.7 ms.

| piece | now (M=4) | floor | what to do |
|---|---|---|---|
| FFN | 23.7 | 19.8 | 128 per-layer int8 quantize launches; fold the quantize into the preceding norm / swiglu kernel |
| full attention | 4.0 | 1.2 | 4 separate KV reads; write a flash-decode that takes M queries against ONE KV read |
| norms / conv / swiglu | 4.3 | 2.5 | batched variants never tuned for small M |
| projections + GDN + logits | 11.9 | 11.9 | already at floor |
| **total** | **44.1** | **35.4** | |

Plus drafts: 2.0 ms each, of which ~1.4 is the 675 MB lm_head. A shortlist
vocabulary for DRAFTING ONLY (top ~32k of 248k) takes a draft to ~0.8 ms.
The draft only has to be right often enough to survive verification, so a
shortlist miss costs one rejected token, not a wrong output.

    35.4 + 3 x 0.8 = 37.8 ms / 2.59 accepted = 14.6 ms/token = 68 tok/s

## Order of work when resuming

1. Batched flash-decode: M queries, one KV read.  (~2.8 ms)
2. Fold the int8 activation quantize into the norm/swiglu kernels. (~3.8 ms)
3. Tune the batched norm/conv/swiglu at small M. (~1.8 ms)
4. Draft shortlist vocabulary. (~1.2 ms x K)
5. THEN the accept/reject loop + GDN state rollback -> the first real number.
6. THEN Ornith end to end.

## Reminder of the traps already paid for

- Any W4A8 shape needs `N % 256 == 0`; the B block loads do not clamp.
- Grep `.w.payload` before converting another weight -- several fast paths
  bypass `mm()` and will fault the device.
- `build_b70.sh` does NOT rebuild the cutlass bridges; use
  `tools/build_bridges_b70.sh`.
- vLLM needs `-v /dev/dri:/dev/dri` (oneCCL reads /dev/dri/by-path);
  `tools/vtune.sh`.
- MTP concat is fc([norm(EMBEDDING) ; norm(hidden)]); a draft must be checked
  TWO iterations later. Either mistake reads 0% acceptance on a working head.
