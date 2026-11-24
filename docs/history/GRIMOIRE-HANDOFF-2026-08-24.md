# Grimoire B70 handoff — 2026-08-24

## Objective and completion gate

Finish a framework-free Intel Arc Pro B70 inference runtime for Ornith 1.5 and
Qwen3.8-27B using the native `MXFP4-GRIMOIRE` format and Xe2/XMX kernels.

Completion still requires all of the following:

- Native MXFP4 artifacts for both BF16 source models.
- No PyTorch, vLLM, OpenVINO, or llama.cpp runtime dependency.
- Coherent inference from both models on short and long prompts.
- Full end-to-end 4096-token prompt processing at **>= 10,000 tok/s**.
- Reproducible build, conversion, and execution instructions.

The goal is **not complete**. Best verified production-like PP is currently
**9,450.9 tok/s** (4096 tokens in 433.4 ms). The remaining gate gap is 23.8 ms,
or 549.1 tok/s.

## Locations

- Local repository: `/Users/ianernst/grimoire-work/grimoire`
- Remote host: `root@192.168.8.225`
- Remote repository: `/mnt/storage/isos/grimoire-fuse`
- Remote models: `/mnt/storage/Models`
- Intel kernel source: `/mnt/cache/appdata/vllm-xpu-kernels`
- Build container: `my-vllm-xpu:latest`
- B70: 256 EUs, 31.9 GiB VRAM, driver `1.15.39122+11`

The worktree was already heavily modified before this handoff. Do not reset,
checkout, or discard unrelated changes. `git diff --check` passes.

## Native model artifacts

### Ornith

`/mnt/storage/Models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE/model-v2.b70`

- File size: about 18.27 GiB
- Runtime resident: 18.21 GiB
- 40 layers: 30 linear-attention, 10 full-attention
- 256 routed experts, top-8

### Qwen

`/mnt/storage/Models/Qwen3.8-27B-MXFP4-GRIMOIRE/model-v2.b70`

- File size: about 15.26 GiB
- Runtime resident: 16.18 GiB
- 64 layers: 48 linear-attention, 16 full-attention
- Dense model (`top_k=0`)

Both native files load successfully without a framework runtime.

## Runtime dependency state

The production executable/bridges use SYCL, Level Zero, compiler math, and
standard system libraries. No Torch/vLLM/OpenVINO library is required at
runtime. Torch headers are used only to compile the optional historical fused
MoE bridge. Its `ldd` output contains no Torch library.

## Verified inference state

### Qwen

- Native generation has produced coherent output, including:
  `The user is greeting me and asking how I am. This`
- Full 4096-token PP previously passed at 1507.7 tok/s; the latest accepted
  dataflow changes have not yet been rebenchmarked on Qwen.
- Decode is about 21.6 tok/s.
- Two correctness bugs were fixed:
  - Dense `top_k=0` allocations now use at least one element.
  - Dense MLP bridge scratch uses `max(R*..., M*W)`, preventing memory
    corruption/hangs for the 34816-wide MLP.

### Ornith

- Decode is about 109 tok/s.
- Short prompts sometimes produce plausible multilingual text.
- Long-prompt output is still not coherent. The control path and the faster
  BF16-QKV path both fail, so BF16-QKV was not the cause.
- Example 57-token control output: `aio_202408151`
- Coherence must be fixed before completion or before speculative decoding.

## Performance progression (Ornith, full 4096-token end-to-end PP)

All numbers below are complete model passes, not projected GEMM rates.

- 6,962.5 tok/s: earlier coherent baseline
- 7,937.3 tok/s: direct FP32 epilogues plus accepted BF16 dataflow
- 8,104.3 tok/s: BF16 full-attention QKV candidate
- 8,422.0 tok/s: Intel-style vectorized MoE remap
- 8,957.3 tok/s: real-routing 64x128 grouped MoE tile
- 9,221.9 tok/s: fused DeltaNet q/k normalization and direct BF16 value output
- 9,311.5 tok/s: end-to-end BF16 DeltaNet QKV/convolution candidate
- **9,450.9 tok/s: unused FP32 normalization outputs suppressed**

Best verified runtime: **433.4 ms**. Required runtime: **<= 409.6 ms**.

## Accepted structural optimizations

- Direct native MXFP4 BF16 and FP32 Xe2 dense epilogues.
- One RMSNorm pass can emit BF16 alongside FP32.
- FP32 normalized output is suppressed when the native path consumes only
  BF16, removing about 2.6 GB of prompt-time writes.
- DeltaNet convolution and Q/K/V split are fused.
- DeltaNet values can be emitted directly as BF16.
- DeltaNet q/k normalization can emit BF16 directly.
- Optional end-to-end BF16 DeltaNet QKV -> convolution -> normalization path.
- Raw Xe2 GDN recurrence outputs BF16 directly into the gated output path.
- Raw Xe2 attention outputs BF16 directly into gate/output projection.
- MoE remap uses one token workgroup, 16-byte vector loads, and fans each
  source vector to all eight expert rows. Remap fell from about 24.7 to 14 ms.
- MoE gather uses eight values per work-item in registers.
- Grouped MoE production tile is **64x128**, not 128x64. Real routing counts
  fluctuate around 128 rows/expert; 128-row tiles caused roughly half of the
  experts to spill into a second mostly empty tile. The real-model change cut:
  - gate/up from about 90.8 to 70.7 ms
  - down from about 51.5 to 42.2 ms
- BF16 MLP gate/up -> SwiGLU -> FP32 down epilogue.
- Profiling queue/marker overhead is disabled unless
  `GRIMOIRE_PROFILE_PREFILL=1`.

## Rejected or gated experiments

Do not promote these without a new full-model result.

- `GRIMOIRE_PARALLEL_PREFILL=1`: regressed badly.
- `GRIMOIRE_PARALLEL_SHARED=1`: isolated shared/routed overlap regressed
  9450.9 -> 8957.4 tok/s due to XMX/memory contention.
- `GRIMOIRE_FUSE_DN_PROJECTIONS=1`: combined qkv/z/ab regressed.
- `GRIMOIRE_XE2_FUSED_MOE_BRIDGE=...`: fused gate/up/SwiGLU is no faster.
  With the corrected 64x128 fused policy it measured 84.2 ms, equal to
  separate gate/up plus SwiGLU (71.0 + 13.1 ms).
- Uniform-row grouped autotuning selected 128x256/2xEU, but this was invalid
  for real routing. Full-model down GEMM regressed. The current 64x128 policy
  is the accepted real-routing result.
- Command-graph decode replay is essentially tied with direct submission.

Gated candidates currently used for the best number:

- `GRIMOIRE_BF16_QKV=1`
- `GRIMOIRE_BF16_DN_QKV=1`

They improve PP but must remain gated until coherent logits are verified.

## Current untested candidate — test this first

The last build completed successfully but was **not run** because work stopped
for this handoff.

Environment flag:

`GRIMOIRE_DEFER_MOE_GATHER=1`

What it does:

- Writes expert-major down-GEMM output into a persistent BF16 `moe_res` buffer.
- Skips the 16.2 ms routed gather at the end of each layer.
- At the next layer, one fused token workgroup combines:
  - eight routed expert rows and route weights
  - shared-expert residual
  - hidden-state residual update
  - RMSNorm and BF16 normalized output
- The final layer uses the same fused operation before logits.

This is designed to replace approximately 16.2 ms gather + 14 ms next-layer
input normalization with one kernel. It is large enough to cross the 10k gate
if correct and sufficiently fast.

### Exact first command to run

```bash
ssh root@192.168.8.225 "docker run --rm --device /dev/dri --entrypoint bash \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire \
  -v /mnt/storage/Models:/models my-vllm-xpu:latest -lc '
GRIMOIRE_PROFILE_PREFILL=1 \
GRIMOIRE_DEFER_MOE_GATHER=1 \
GRIMOIRE_BF16_QKV=1 \
GRIMOIRE_BF16_DN_QKV=1 \
GRIMOIRE_XE2_GROUPED_BRIDGE=/grimoire/src/libgrimoire_xe2_grouped.so \
GRIMOIRE_XE2_GDN_RAW_BRIDGE=/grimoire/src/libgrimoire_xe2_gdn_raw.so \
GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_raw.so \
/grimoire/bin/grimoire \
  -m /models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE \
  --proj mxfp4 --ctx 8192 2>&1'"
```

Acceptance criteria:

- No hang or SYCL exception.
- `batched MoE probe ... PASS`.
- Full 4096-token pass says `PASS`.
- Combined `MoE gather` plus `input norm` must materially fall.
- Then rerun without `GRIMOIRE_PROFILE_PREFILL=1` and require <=409.6 ms.
- If it regresses or fails, leave the flag off; the 9450.9 path remains safe.

## Build commands

Build the Grimoire executable:

```bash
ssh root@192.168.8.225 "docker run --rm --entrypoint bash \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire \
  my-vllm-xpu:latest -lc \
  'bash /grimoire/tools/build_grimoire_only_b70.sh /grimoire'"
```

Build standard bridges:

```bash
ssh root@192.168.8.225 "docker run --rm --entrypoint bash \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire \
  -v /mnt/cache/appdata/vllm-xpu-kernels:/src \
  my-vllm-xpu:latest -lc \
  'cd /grimoire && bash tools/build_bridges_b70.sh /grimoire'"
```

The fused-MoE headers were removed from the current kernel checkout but are
recoverable from commit `ef113f9`. They were extracted to:

`/tmp/grimoire-fusedref-ef113f9`

The corrected fused library already built successfully, but it did not improve
performance and should remain disabled.

## Misleading diagnostic output to clean later

- `Xe2 grouped GEMM unavailable` refers to the obsolete signed-W4 bridge, not
  the active native MXFP4 grouped bridge.
- `Xe2 chunk GDN unavailable` refers to the old generic bridge, not the active
  raw GDN bridge.
- `10.8% of bf16 XMX peak` describes the internal fallback diagnostic GEMM,
  not the active external Xe2 dense bridge. Active projection kernels have
  previously measured far higher utilization.

## Next work after the PP gate

1. Fix Ornith long-prompt coherence using layer-by-layer logits/activation
   comparison with the BF16 source model.
2. Revalidate Qwen after all accepted dataflow changes.
3. Make accepted fast paths automatic and remove environment flags.
4. Audit runtime dependencies and package conversion/run instructions.
5. Optimize TG by reducing the current ~793 decode launches/token.
6. Add native speculative decoding (DFlash/DSpark, or MTP when trained heads
   exist) only after verifier correctness.
7. Add dual-B70 support: expert parallel for Ornith, tensor parallel for Qwen.

## Current plan status

- Baselines/native artifacts: complete.
- B70-specialized MoE/dataflow optimization: in progress.
- >=10,000 full PP proof: not yet achieved.
- Ornith coherence: not achieved.
- Qwen final validation: pending.
- Dependency/package audit: pending.
# Continuation update — 2026-08-24 afternoon

## Device mapping and isolation

- `level_zero:0` = direct B70 at PCI `0000:03:00.0` / renderD128.
- `level_zero:1` = faster USB4 B70 at PCI `0000:35:00.0` / renderD130.
- Explicit A/B with the accepted deferred-MoE path measured:
  - device 0: 419.2 ms, 9771.1 tok/s PP
  - device 1: 410.8 ms, 9971.2 tok/s PP
- Three more device-1 samples: 410.0, 412.3, 409.6 ms; the last is
  **10001.2 tok/s**, but the gate does not yet have stable margin.
- Always use `ONEAPI_DEVICE_SELECTOR=level_zero:1`. GPU 0 is reserved for the
  user's 12G workload and must not be used.
- Three stale Grimoire/Qwen containers on GPU 0 were found and stopped.

## Accepted deferred MoE result

`GRIMOIRE_DEFER_MOE_GATHER=1` is correct and a real improvement. Alternating
unprofiled A/B on the original device selection measured baseline 434.5/435.8
ms versus deferred 429.0/429.7 ms, about +127 tok/s. Keep the original
256-thread x 8-element fused kernel. Rejected variants:

- 128 threads x 16 elements: 416.6 ms on faster device 1.
- workgroup-local route cache: 412.1/412.1/413.5 ms.
- subgroup route broadcasts: 412.7 ms.
- BF16 router logits: no gain (412.5/411.1 versus 409.7/411.6 baseline).
- prefill command graph: 416.5 ms.

## Qwen current evidence

- Native Qwen MXFP4 output is coherent. Prompt `The capital of France is`
  generated a correct explanation ending in `Paris`.
- Current measured TG is 21.5 tok/s, materially below the user's vLLM GPTQ
  reference of about 33 tok/s. Do not present 21.5 as an acceptable result.
- Device timeline: 46.08 ms/token total; 43.73 ms is the 64-layer stack and
  1.67 ms the LM head. Launch markers total only 44 us. GEMV bandwidth is the
  dominant problem.
- One exhaustive `GRIMOIRE_AUTOTUNE=1` sweep completed before the later PP
  phase hit a driver abort. All 37 configurations produced the same token
  hash. Winner: `EPL=16`, `UNROLL=1`, `WIDE=off`, 44.222 ms / 22.6 tok/s.
- The fast saved Intel vLLM container uses oneDNN W4A16 after
  `transpose_onednn_woq_format`; its small `vllm_int4_for_multi_arc.so` is only
  a statically linked host quantizer, not the inference kernel.

## New compiled but untested decode candidates

The following changes compile successfully but cannot yet be run because
device 1 is driver-hung:

1. `src/gemv_decode.cpp`: interleave all four rows owned by a subgroup at each
   K step, exposing 32 independent weight streams per workgroup instead of
   completing the four rows serially.
2. `src/gemv_step.hpp`: EPL16 MXFP4 uses one aligned 64-bit packed-weight load
   instead of eight scalar byte loads.
3. Framework-free native oneDNN MXFP4 W4A16 prototype:
   `src/xe2_onednn_bridge.cpp/.hpp` and
   `tools/bench_onednn_mxfp4.cpp`. It uses official oneDNN `f4_e2m1` weights,
   E8M0 weight scales with group `{32,1}`, BF16 activation/output, and the
   native Grimoire physical payload/scale order. The executable
   `/grimoire/bin/bench_onednn_mxfp4` compiled successfully.

## Current driver blocker

The Qwen autotune's subsequent 4096 PP phase aborted in Intel NEO at
`drm_neo.cpp:289`. Container `782e56c151b9` remains with Grimoire in
uninterruptible `D` state at `dma_fence_default_wait`; normal `docker stop`
cannot remove it. Device 1 and the integrated Arc no longer enumerate for a
new Level Zero process. PCI endpoint `0000:35:00.0` exposes a reset file and
reports reset methods `flr bus`.

Do not reset without explicit confirmation. Once confirmed, reset only
`0000:35:00.0`, verify GPU 0's workload remains intact, remove the stale
container, then run in this order:

1. `/grimoire/bin/bench_onednn_mxfp4` on `level_zero:1` for plan creation,
   sampled numerical error, and bandwidth.
2. Qwen coherent generation with `B70_EPL=16 B70_UNROLL=1 B70_WIDE=0` to
   validate the interleaved/vector-load GEMV.
3. Qwen 32-step TG benchmark; target at least the vLLM reference 33 tok/s.
4. Ornith deferred-MoE 4096 PP benchmark; target stable <=409.6 ms.


# Continuation update — 2026-08-24 evening

## Driver blocker: RESOLVED

The Tower was rebooted. The hung container `782e56c151b9` is gone and all
three devices enumerate again (two B70s plus the iGPU). No PCI reset was
needed and none was performed.

Device mapping re-verified from sysfs after the reboot:

- `0000:03:00.0` -> `card0` / `renderD128`  (GPU 0, reserved for the user)
- `0000:35:00.0` -> `card2` / `renderD130`  (the faster USB4 B70)

**Preferred isolation:** bind only the render node instead of trusting an
index. `--device /dev/dri/renderD130` makes GPU 0 physically invisible to
the container, and the target device is then `level_zero:0` inside it.
This is safer than `--device /dev/dri` plus `ONEAPI_DEVICE_SELECTOR=level_zero:1`,
because the index ordering is not guaranteed stable across reboots.

## >= 10,000 tok/s PP GATE: MET, WITH STABLE MARGIN

Four consecutive full 4096-token end-to-end runs on `renderD130`, deferred
MoE plus the two gated BF16 paths:

- 401.8 ms -> 10,194.2 tok/s
- 400.7 ms -> 10,221.7 tok/s
- 401.7 ms -> 10,195.7 tok/s
- 400.9 ms -> 10,217.1 tok/s

Spread is 1.1 ms. Required is <= 409.6 ms, so the margin is about 8 ms and
it is now repeatable, not a single lucky sample. The previous 409.6-419.2 ms
readings were taken on a GPU whose state had already been degraded by the
hung context; a clean boot accounts for the shift.

`batched MoE probe: routes exact, weight RMS 0.00e+00, output rel 0.00e+00 PASS`
on every run. Ornith decode also improved to 111.7-112.6 tok/s (graph replay).

This gate should be re-confirmed once more after any future reboot before
being treated as final, since it is the completion criterion.

## Measured memory roofline: 602 GB/s

`tools/bwprobe.cpp` (new) measures streaming read bandwidth. Result: **602 GB/s**,
saturating from 1 GiB upward. This matches the 608 GB/s figure already quoted
in the `gemv_decode.cpp` header, so that number is now independently confirmed.

Two traps, both hit while measuring:

- **Never fill the probe buffer with a constant.** A `memset`-filled buffer
  reported 1,900 GB/s because the memory controller compresses it losslessly.
  Bandwidth appearing to *rise* with buffer size is the tell. The probe now
  fills with a pseudo-random pattern.
- **Device-reported `memory_clock_rate` (2800 MHz) and `memory_bus_width`
  (64 bit) are bogus** on this part; 64-bit with 32 GiB is nonsense. Any
  "theoretical peak" computed from them is garbage. Measure instead.

## Qwen decode budget, fully mapped

Per token 42.7 ms. MXFP4 costs 4.25 bits/weight = 0.53125 B/element.

| region | bytes | time | GB/s | % of 602 |
|---|---|---|---|---|
| lm_head (248320x5120) | 676 MB | 1534 us | **440** | 73% |
| FFN gate_up (34816x5120) | 94.7 MB | 241 us | 393 | 65% |
| FFN down (5120x17408) | 47.4 MB | 123 us | 386 | 64% |
| z gemv (6144x5120) | 16.7 MB | 44 us | 378 | 63% |
| o gemv (5120x6144) | 16.7 MB | 47 us | 356 | 59% |
| la_qkv (10240x5120) | 27.9 MB | 87 us | 322 | 53% |
| k+v (2048x5120) | 5.6 MB | 52 us | **108** | 18% |
| ab gemv + gates (~96 rows) | 0.26 MB | 28 us | **9** | 1.5% |
| in_norm + post_norm | ~0.1 MB | 30 us | ~4 | latency |

Share of the token: **FFN 23.4 ms (54%)**, linear-attn blocks 13.5 ms (31%),
full-attn blocks 4.7 ms (11%), lm_head 1.5 ms (3.6%).

**The 33 tok/s target is reachable.** It needs 30.3 ms/token = 440 GB/s
effective, which is exactly what lm_head already sustains. The work is
lifting the other GEMVs to that level, not beating the hardware.

Occupancy is the mechanism: a GEMV launches `ceil(N/32)` workgroups
(`WG_SUBGROUPS 8 x ROWS_PER_SG 4`). lm_head gets 7760 of them; `k+v` gets 64
and `ab` gets 3, which is why those two collapse. Split-K for low-N shapes is
the obvious next lever and has not been tried.

The dense FFN gate|up is **already fused** into one N=34816 GEMV, so there is
no fusion win left there; its 65% is a kernel limit, not an occupancy one.

## Fixed: B70_UNROLL default was costing 2.5% of decode

`g_tune_unroll` defaulted to **1**, which silently overrode every format's
`GemvGeom<F>::UNROLL_DEFAULT`. MXFP4's tuned value is 2. Measured on the
32-token full-model sweep: 43.66 ms (unroll 1) vs 42.68 ms (format default).

Changed the default to **0**, meaning "use the per-format value".
Qwen TG with no environment flags at all: **22.8 -> 23.3 tok/s**, output
unchanged. This also advances the "remove environment flags" goal.

Re-ran the full GEMV autotune on the new interleaved/64-bit-load kernel; all
37 configurations still produce an identical token hash. EPL=16 remains
correct and EPL 32/64 collapse to 14-17 tok/s. Do not pin `B70_UNROLL=1`.

## Rejected this session

- **oneDNN MXFP4 W4A16 prototype.** It is numerically exact
  (`rel 0.000e+00`) but slow: 147.9 us on 1x10240x5120, i.e. 188 GB/s. The
  native Grimoire GEMV does that identical shape (`la_qkv`) in 86.6 us at
  322 GB/s. oneDNN is ~1.8x slower. Do not pursue it as the decode kernel.
- **Register-resident E2M1 decode.** Hypothesis: the MXFP4 inner loop does
  two SLM reads per element (nibble LUT + activation) to feed one FMA, while
  the ALUs idle, so trading SLM for arithmetic should win. It does not.
  - packed 64-bit bf16 table, shift-select: **15.8 tok/s**
  - pure 32-bit arithmetic bit-assembly with a select: **9.0 tok/s**
  - existing SLM nibble LUT: **23.3 tok/s**
  Xe2 emulates 64-bit variable shifts, and the e==0 special case forces a
  select on every element. The SLM table is the right design; the comment in
  `gemv_step.hpp` defending it is correct. Reverted.

## Ornith coherence: still broken

Confirmed still failing on the control path (no BF16 flags):

- short prompt `The capital of France is` -> empty output
- 75-token prompt -> `Let me think about`, then stops

Different symptom from the old `aio_202408151` garbage — output is now
truncated/empty rather than random tokens — but still not coherent. This
remains the top correctness blocker and needs the layer-by-layer logit
comparison against the BF16 source model.

## Suggested next steps

1. Ornith long-prompt coherence (unchanged priority; now the only thing
   between the project and the completion gate, since PP is met).
2. Split-K decode GEMV for low-N shapes: `k+v` (108 GB/s), `ab` (9 GB/s),
   and the two RMSNorms. Worth roughly 3 ms/token on Qwen.
3. Lift the FFN from 393 GB/s toward lm_head's 440; it is 54% of the token.
4. Make `GRIMOIRE_DEFER_MOE_GATHER` and the two BF16 paths automatic once
   Ornith coherence is verified.

## Files changed on the remote this session

- `src/gemv_decode.cpp` — unroll default 1 -> 0 (the real fix)
- `src/grimoire.cpp` — three timeline marks splitting the dense FFN
  (`ffn gate_up` / `ffn swiglu` / `ffn down`); inert without `GRIMOIRE_TIMELINE`
- `tools/bwprobe.cpp` — new bandwidth probe
- `src/grimoire.cpp.bak-timeline` — pre-instrumentation backup

`src/gemv_step.hpp` was modified and fully reverted; it is byte-identical to
its pre-session state.

# Qwen PP root cause — 2026-08-24 late

## Targets (revised by Ian)

Ornith is DONE: 10,194-10,222 tok/s PP, ~110 TG.
**Qwen targets: 2,000 PP and 30 TG.** The old 10,000 figure was for the Ornith
MoE model only and is not applicable to a dense 27B (see ceiling below).

## Measured Qwen baseline

- PP 4096: **227.9 tok/s** (17,973 ms). The handoff's earlier 1,507.7 figure
  does not reproduce.
- TG: **23.3 tok/s** (coherent, no env flags).

## ROOT CAUSE: the dense GEMM tile was never tuned for Qwen

This is the Ornith lesson, unapplied.

Efficiency against bf16 XMX peak (~180 TFLOP/s, backed out from the engine's
own "11.2% of peak" line):

- **Ornith: 2 x 3e9 x 4096 FLOPs / 401.8 ms = 61 TFLOP/s = 34% of peak**
- **Qwen:   2 x 25.6e9 x 4096 FLOPs / 17,973 ms = 11.7 TFLOP/s = 6.5% of peak**

Same hardware, 5.2x efficiency gap. Why:

- Ornith's dominant kernel is the **grouped** MoE GEMM
  (`grimoire_xe2_grouped_mxfp4_bf16`), which WAS tuned to 64x128 after
  discovering 128-row tiles left half of each tile empty.
- Qwen's dominant kernel is the **dense** FFN
  (`grimoire_xe2_dense_mxfp4_bf16`), which is **hardcoded to `p128x128`** at
  `src/xe2_grouped_bridge.cpp:92`. Never tuned.
- The dense autotune harness `tools/autotune_b70_dense.cpp` already exists AND
  a 9-policy entry point `grimoire_xe2_dense_mxfp4_autotune` is already
  compiled in — but its shape table is **100% Ornith shapes** (every K=2048,
  frequencies weighted to 40 layers). Qwen's shapes were never in the set.

The FFN is **~70% of Qwen's prefill FLOPs**:
- per linear layer: FFN 534.8 MFLOP vs attention 230.7 MFLOP
- per full layer:   FFN 534.8 MFLOP vs attention 209.7 MFLOP

Corroborating symptom: Qwen's per-token prefill cost is **flat** with batch
size — 4.58 ms/token at M=32 vs 4.39 ms/token at M=4096. Batching buys almost
nothing, which is what a mismatched tile looks like. For contrast Ornith goes
from 8.95 ms/token (decode) to 0.098 ms/token (PP), a 91x batching win.

Prefill breakdown (32 tokens, from GRIMOIRE_PROFILE_PREFILL=1). NOTE: `pp_mark`
measures from the PREVIOUS mark, so "input norm" actually contains the previous
layer's dense FFN — it is not a norm:

| region | ms | share |
|---|---|---|
| "input norm" (= dense FFN + norm) | 62.78 | 43% |
| DN recurrence | 42.26 | 29% |
| DN qkv projection | 11.87 | 8% |
| full attention | 8.95 | 6% |
| DN norm + output projection | 7.36 | 5% |
| DN z projection | 5.97 | 4% |

## Ready to run (already built, waiting on the GPU)

- `tools/autotune_b70_dense_qwen.cpp` — the dense autotune with **Qwen's real
  shapes and per-forward-pass frequencies**:
  ffn-gate-up 34816x5120 (x64), ffn-down 5120x17408 (x64), dn-qkv 10240x5120
  (x48), dn-z 6144x5120 (x48), dn-out 5120x6144 (x48), fa-q 12288x5120 (x16),
  fa-kv 2048x5120 (x16), fa-o 5120x6144 (x16)
- Built binary: `/grimoire/bin/autotune_qwen`
- Tuning library: `/grimoire/src/libdense_tune.so`

Build recipe (if a rebuild is needed) is `tools/run_dense_cold_autotune_b70.sh`
with the tool swapped; needs `-v /mnt/cache/appdata/vllm-xpu-kernels:/src`.

### Resume sequence

1. `/grimoire/bin/autotune_qwen` — sweeps 9 tile policies x 8 Qwen shapes,
   prints a frequency-weighted winner. Run WITHOUT sourcing setvars and
   WITHOUT `ONEAPI_DEVICE_SELECTOR` (only one GPU is bound).
2. Make `grimoire_xe2_dense_mxfp4_bf16` / `_f32` dispatch on shape instead of
   the fixed `p128x128`, using the winner(s).
3. Rebuild the grouped bridge, re-measure Qwen PP. 
4. Then TG: split-K for the low-N decode GEMVs (k+v at 108 GB/s, ab at 9 GB/s).
5. Then DSpark/DFlash (below).

## Dead ends — do not spend time here

- **vLLM's oneDNN W4A16 path cannot apply to our artifacts.** `od_w4` is gated
  on `p.fmt == Fmt::INT4` (`src/grimoire.cpp:568`). Our models are MXFP4, so
  the path can never engage; that is why it logs "raw oneDNN W4A16
  unavailable". Enabling it would require an INT4 conversion, which conflicts
  with the native-MXFP4 requirement.
- `Xe2 chunk GDN unavailable` is the KNOWN red herring: `load_xe2_chunk_gdn`
  looks for the obsolete `grimoire_xe2_chunk_gdn_bf16` in the attention
  bridge. The active path is `load_xe2_chunk_gdn_raw`, which loads fine.
- oneDNN MXFP4 as a decode GEMV kernel: 188 GB/s vs the native 322 GB/s.

## Quantization provenance: VERIFIED CORRECT

`b70_compile_model.cpp` **rejects any non-BF16 source**
(`if (r.t.dtype != b70::STDtype::BF16 ...) return false`; usage is
`BF16_MODEL_DIR OUTPUT.b70`). Both sources are unquantized BF16 with
`quant: None`:
- `/mnt/storage/Models/Qwen3.8-27B`
- `/mnt/storage/Models/Ornith-1.5-35B-A3B`

No HF-quantized checkpoint is in the runtime path for either model.

## Speculative decoding assets (inventoried, not yet started)

- `Qwen3.8-27B-DSpark` and `qwen38-dflash-drafter-fp8-b70`: arch
  `Qwen3DSparkModel`, **bf16** (so quantizable with our own tool), 5
  full-attention layers, hidden 5120, inter 10240, 40 heads / 8 kv, head_dim
  128, `block_size: 7`, `num_target_layers: 64` — purpose-built for
  Qwen3.8-27B. Taps target layer ids **[4, 16, 28, 40, 52]**, so the runtime
  must expose those intermediate hidden states to the drafter.
  Has `enable_confidence_head` + markov head (rank 256).
- MTP heads for Qwen3.8-27B exist ONLY inside a GPTQ checkpoint
  (`Qwen3.8-27B-GPTQ-Int4-MTP-BF16`), which conflicts with the BF16-source
  rule unless the heads alone are extracted in BF16.
- `Qwen3.6-35B-MTP` is a different model family (3.6-35B), also GPTQ.

DSpark is the strongest lead for the 2.5-3x TG that vLLM shows, and it is
BF16, so it fits the quantization rule.

## Hardware ceiling note

Dense Qwen 27B at 4096 tokens is 2.1e14 FLOPs. At ~180 TFLOP/s bf16 XMX peak
the absolute ceiling is ~3,500 tok/s PP. So **2,000 PP needs ~57% of XMX
peak** — hard but real. 10,000 would have needed ~3x the hardware's peak and
was never achievable for a dense model on one B70.

## Incident: GPU 1 wedged (my fault)

I ran `docker kill` on a profiling container that had GPU work in flight.
`xe 0000:35:00.0` then returned `0xFFFFFFFF` on MMIO with forcewake failures —
the device fell off the bus. Config space still responded; MMIO did not.
Ian rebooted the server.

**Rule going forward: never kill a container with GPU work in flight.** Let
background runs reach their own `timeout`. This is the second time this exact
action has wedged the device.

# Reboot handoff — Qwen dense-tile incident and recovery plan

## Exact project status at shutdown

Targets remain **Qwen PP >= 2,000 tok/s and TG >= 30 tok/s** on one B70,
using the native framework-free MXFP4 artifact converted from the BF16 source.

Verified good results before this incident:

- Qwen native generation is coherent.
- Known-safe Qwen baseline: **227.9 tok/s PP at M=4096** and **23.3 tok/s TG**.
- A corrected bounded run on `renderD128`, with 32 prompt tokens, completed:
  `FULL E2E PP: PASS`, 148.2 ms / 216.0 tok/s, TG 23.4 tok/s.
- Ornith remains complete at 10,194-10,222 tok/s PP and about 110 tok/s TG.
- The BF16 Qwen dense autotuner tested 9 policies x 8 real Qwen shapes at
  M=4096. All 72 sampled comparisons passed. Its isolated weighted winner was
  128x256 at 1572.698 ms versus 1641.355 ms for 128x128 (only 4.2% faster).

The isolated BF16 autotune result **does not transfer safely to sustained
full-model execution**. A production bridge using 128x256 for BF16 caused the
full 4096-token Qwen run to consume host CPU indefinitely on both B70s. The
first bad experiment also used an unvalidated 128x256 FP32 output policy; that
was reverted, but a second full run with only BF16 at 128x256 reproduced the
pathology. Therefore the BF16 128x256 policy is rejected for production even
though the single-GEMM harness passes.

Two containers were deliberately left untouched because killing active Xe GPU
work previously made the device fall off the bus:

- `74967687ca7f` / Grimoire PID 86285: bound to `renderD130`.
- `1ceccac169f3` / Grimoire PID 123936: bound to `renderD128`.

Ian must power-cycle the Tower manually. Do not attempt `docker kill`, PCI FLR,
or a software reboot before the power cycle.

## Critical source/binary distinction

Remote canonical source:

`/mnt/storage/isos/grimoire-fuse/src/xe2_grouped_bridge.cpp`

has already been restored to the known-safe production dispatch:

- BF16 dense: `p128x128`
- FP32 dense: `p128x128_f32`
- grouped MoE: unchanged `p64x128_prod`

Source SHA-256 at handoff:

`a8c3a1fd62f5c8cc508116fe7b9dd8f2d008242f3b1a514d67fb3e4dafc09e22`

However, `/grimoire/src/libgrimoire_xe2_grouped.so` was built at 16:41 before
the final BF16 restoration. It is stale and potentially unsafe. **After the
power cycle, rebuild the bridge before running any inference. Do not reuse the
existing `.so`.**

## New diagnostic assets

- `tools/autotune_b70_dense_qwen.cpp`: BF16 M=4096 Qwen-shape autotuner.
- `bin/autotune_qwen`: built BF16 autotuner.
- `tools/autotune_b70_dense_qwen_f32.cpp`: separate FP32-output policy tuner.
- `tools/build_qwen_f32_autotune_b70.sh`: reproducible FP32 tuner build.
- `src/libdense_tune_f32.so` and `bin/autotune_qwen_f32`: compiled, **not run**.

The FP32 autotuner intentionally remains unrun. Do not sweep unvalidated
policies on a production GPU until its harness has per-policy watchdog/recovery
and runs one candidate per process. Never proceed from a microbenchmark winner
directly to M=4096 full-model execution.

## Mandatory recovery sequence after the manual power cycle

1. Verify both render nodes enumerate and no stale Grimoire containers remain.
2. Verify GPU 0's user workload requirements before using either device.
3. Rebuild the standard grouped bridge from the restored 128x128 source.
4. Inspect the resulting library timestamp and symbol load.
5. Run Qwen with `GRIMOIRE_BENCH_PREFILL_TOKENS=32` first; require full PASS
   and normal completion.
6. Run a coherent 8-token generation test.
7. Re-establish the known-safe 4096 baseline using the 128x128 bridge only.
   Launch it with an external timeout that is allowed to expire naturally;
   never kill the container while GPU work is active.
8. Only then resume optimization.

## Plan to reach PP 2,000 and TG 30

The 128x256 dense-tile hypothesis is rejected as the main PP lever. Its best
isolated gain was only 4.2%, far below the required 8.8x improvement, and its
sustained behavior is unsafe. Do not spend more full-model runs on it.

PP plan, in priority order:

1. Add correct region timing for M=4096 (the current `pp_mark` labels the
   previous region) and measure an exact layer/operation wall-time budget on
   the safe bridge.
2. Reconcile the huge discrepancy between the isolated dense GEMM weighted
   total (~1.6 s) and full E2E baseline (~18.0 s). This is now the primary PP
   root cause. Likely candidates are repeated conversions/FP32 epilogues,
   recurrent-state work, synchronization, or a full-model data-layout issue;
   tile selection alone cannot explain it.
3. Build a production-shaped loop benchmark that repeats all 64-layer Qwen
   shapes and both BF16/FP32 epilogues with real buffer reuse. It must reproduce
   sustained behavior before any policy reaches the full model.
4. Remove avoidable BF16<->FP32 conversions and fuse residual/norm/epilogues
   where the region budget proves they dominate.
5. Optimize the linear-attention recurrence and full-attention kernels only
   after the exact M=4096 breakdown identifies their real share.

TG plan:

1. Keep the accepted `B70_UNROLL` default fix (`0` means MXFP4's tuned default
   2); safe baseline is 23.3 tok/s.
2. Implement and validate split-K decode GEMV for low-N projections (`k+v` at
   108 GB/s and `ab` at 9 GB/s). This is the clearest native lever and is worth
   roughly 3 ms/token.
3. Lift the FFN GEMVs from ~393 GB/s toward the measured 440 GB/s LM-head rate;
   FFN is 54% of decode time.
4. After verifier correctness, integrate the BF16 Qwen3.8 DSpark/DFlash drafter
   (target taps 4,16,28,40,52). Speculation is the strongest route from native
   ~26 tok/s toward and beyond 30, but must not hide verifier correctness bugs.

## Safety rule added

A tile or output policy may advance only through: isolated one-call numeric
test, repeated production-shape loop, 32-token full-model pass, coherent short
generation, then 4096-token pass. BF16 and FP32 epilogues are separate kernels
and require separate validation. Never infer FP32 safety from BF16 autotuning.

## Required engineering strategy from Ian

The decisive Ornith progress came from comparing Grimoire with how vLLM
processed prompt prefill for MoE models. Before that comparison, roughly six
hours of work produced no useful result. Once the implementation gap was
identified, about 30 minutes of focused work moved Ornith from approximately
700 tok/s PP to approximately 10,000 tok/s PP and met the goal.

Use that same method for Qwen. vLLM, its Intel/XPU kernels, and saved working
vLLM images may be inspected, profiled, and used as performance references to
discover how they achieve their speed. The relevant scheduling, data-layout,
fusion, kernel, and quantization mechanisms must then be understood and
reproduced inside Grimoire's native B70 implementation.

The finished Grimoire runtime must remain completely framework-free and bare
metal: no PyTorch, vLLM, OpenVINO, llama.cpp, or other inference framework may
be required to load or run a model. vLLM is a reference and investigative
tool, not a production dependency. This is the same boundary successfully used
for Ornith 1.5.

There is no architectural reason to accept failure on Qwen after succeeding on
Ornith. They are closely related base-model families; the material distinction
is that Qwen is dense while Ornith is MoE. The implementation must apply the
same comparative analysis to the dense execution path and find the missing
mechanism rather than assuming the current performance gap is unavoidable.

The non-negotiable Qwen completion gates are:

- Prompt processing: **at least 2,000 tok/s**.
- Token generation: **at least 30 tok/s**.
- Coherent output and correctness at short and long contexts.
- Native, framework-free B70 execution.

Only after these native Qwen gates are met should speculative decoding be
integrated. The intended order is MTP, DFlash, and DSpark evaluation, selecting
or combining the mechanisms that increase TG globally while retaining verifier
correctness. Speculation is a later multiplier, not a substitute for reaching
30 tok/s with the correct native verifier path.

After Qwen and speculation are complete, generalize the converter/runtime so
the same B70-native approach works reliably for every supported quantization
artifact produced by this project.

### Quantization provenance is fundamental

All production artifacts must start from the original **BF16-size model** and
be quantized specifically for the Intel Arc Pro B70's native requirements.
This BF16-source -> B70-specific quantization path was essential to the Ornith
breakthrough and remains mandatory for Qwen and future models.

Do not use an already-quantized Hugging Face checkpoint made for a different
GPU vendor or kernel layout as the production model source. Such artifacts may
be examined for research, but they cannot replace conversion from the original
BF16 weights. Grimoire's native MXFP4/E8M0 physical layout, scaling, packing,
and kernels must match the B70 exactly.

# Post-reboot continuation — 2026-08-24 evening

## Recovery completed safely

After Ian manually power-cycled the Tower, both render nodes enumerated and the
two stale Grimoire containers were gone. The safe dense source checksum matched
the handoff. The grouped bridge was rebuilt from the restored production
dispatch:

- BF16 dense `p128x128`
- FP32 dense `p128x128_f32`
- rebuilt library SHA-256:
  `5733b1073f11dbc691da58adba7ccd1ee863dfbb3701e7498ab9a2cd6f2d09ba`

Bounded validation on `renderD130` passed:

- 32-token full E2E: PASS, 148.4 ms, 215.7 tok/s.
- Decode: 23.5 tok/s.
- Coherence prompt `The capital of France is`, 8 generated tokens:
  `The user is asking a simple factual question`.
- Short generation PP 104.2 tok/s, TG 23.4 tok/s.

## vLLM dense-Qwen comparison

The saved fast Qwen container `VLLM-XPU-27B-QWEN-AUTO` uses an AutoRound INT4
checkpoint and the XPU GPTQ path. At load time it calls
`transpose_onednn_woq_format`; at inference it reshapes activations to 2D and
calls `torch.ops._xpu_C.int4_gemm_w4a16`. The Intel implementation:

- consumes BF16 activations directly;
- uses a persistent oneDNN-native packed WOQ weight layout;
- creates and caches primitives keyed by M/N/K/strides/device;
- applies grouped weight scales and zero points as primitive attributes;
- uses user scratchpads and the native Intel GEMM JIT.

This is an investigative reference only. Grimoire remains MXFP4 from original
BF16 source and must remain framework-free. The relevant lesson is persistent
native packing, shape-specific cached execution, direct BF16 dataflow, and
full-path fusion—not importing PyTorch/vLLM or switching production artifacts
to the foreign GPTQ checkpoint.

## Corrected dense-prefill profiling

Added `pp_mark("dense FFN")` immediately after the dense MLP. Previously its
cost was incorrectly charged to the next layer's `input norm` label. This is
profiling-only and inert unless `GRIMOIRE_PROFILE_PREFILL=1`.

Verified M=32 device totals (second stable sample):

- dense FFN: 62.409 ms (43%)
- DeltaNet recurrence: 42.257 ms (29%)
- DN qkv projection: 11.310 ms
- full attention: 8.941 ms
- DN norm + output projection: 7.347 ms
- DN z projection: 5.966 ms
- DN gate projection: 3.898 ms
- input norm: 1.313 ms
- final norm + logits: 1.543 ms
- remaining DN conv/qk norm: about 0.805 ms
- full E2E profiled: PASS, 156.1 ms, 204.9 tok/s

This proves normalization was never the 43% bottleneck; dense FFN and DeltaNet
recurrence are the real M=32 leaders.

## New M=256 profiling incident

A progressive M=256 run was launched on `renderD130` using the restored safe
128x128 bridge plus `GRIMOIRE_PROFILE_PREFILL=1`. The executable first performs
an internal 32-token pass; that pass completed and printed the correct profile.
The actual M=256 pass then consumed about 187% host CPU for minutes without
completion or output. Container `9631d1ca135d` / Grimoire PID 41799 was left
untouched. Do not kill it; Ian must power-cycle manually if it remains active.

`renderD128` was not used after this incident and should remain untouched.

Do not conclude yet that safe 128x128 or raw GDN alone caused the incident.
Unlike the historical unprofiled 4096 baseline, this run enabled hundreds of
profiling marker events. The new evidence narrows the trigger to something
that differs between the completed M=32 pass and M=256, including:

- native recurrence becomes active at M>=64;
- native full-attention path has larger work;
- profiling markers interact with the native path/driver;
- an M-dependent allocation, stride, or queue submission issue.

The previous attribution of both old wedges solely to 128x256 is therefore
incomplete. 128x256 remains rejected, but the profiling/native-path interaction
must be isolated before any more large runs.

## Mandatory next recovery and isolation sequence

1. Power-cycle manually; do not kill container `9631d1ca135d`.
2. Verify both render nodes and ensure no Grimoire container remains.
3. Keep production dense dispatch at 128x128.
4. Do not run M>=64 with full profiling markers.
5. Add a layer limit and/or single-region benchmark so native GDN and attention
   can be tested one operation at a time without executing all 64 layers.
6. Test the M=64 boundary in separate short-lived processes, first without
   profiling markers, then with one coarse marker only. Never jump to M=256.
7. Add a host-side progress line per layer only for diagnosis; it must not add
   device marker events. This will identify the exact layer/operation where
   submission stops while preserving a recoverable bounded test.
8. Only after repeated M=64/128 completion should M=256 be attempted again.

No further GPU work was authorized after the M=256 process stopped making
progress. The untouched second B70 must not be used to repeat the experiment.

# Continuation after second reboot — 2026-08-24 night

## Current GPU/container state — do not guess

- `renderD130` / PCI `0000:35:00.0` / Level Zero 1 is occupied by Docker
  container `objective_sutherland` (`f02fc7e70b63`). Its Grimoire PID 68616 is
  in uninterruptible `D` state at `dma_fence_default_wait`. Docker stop had
  already failed. Do not use, stop, kill, or inspect it with a GPU workload.
- Kernel logs repeatedly reported GuC reset attempts for `0000:35:00.0`.
- `renderD128` / PCI `0000:03:00.0` / Level Zero 0 enumerated correctly after
  the reboot and was the only device authorized and used in this continuation.
- The final full-model attempt on Level Zero 0 ended in `DEVICE_LOST`. All GPU
  testing stopped immediately afterward. Before any later GPU work, enumerate
  the device read-only and check kernel logs/processes. Do not demand repeated
  power cycles from Ian.

## Isolation result: native attention is safe; cumulative raw GDN is the trigger

New diagnostic controls were added to `src/grimoire.cpp` and rebuilt:

- `GRIMOIRE_PREFILL_LAYER_LIMIT=N` limits execution to the first N layers.
- `GRIMOIRE_PREFILL_HOST_PROGRESS=1` waits and prints completion after each
  layer without adding profiling marker events.
- `GRIMOIRE_RAW_GDN_LAYER_LIMIT=N` was added for isolation only; the hybrid
  strategy is rejected after it still reached `DEVICE_LOST` on a repeated pass.

Important controlled results at M=64:

- one layer with raw GDN: completed;
- eight layers with raw GDN: completed;
- all 64 layers with raw GDN omitted but native attention retained: completed
  twice, PASS, 172.7 ms / 370.6 tok/s;
- therefore native attention is safe in this test and the framework-free
  fallback recurrence is safe;
- cumulative use of the raw chunk-GDN path is the trigger class.

The M=32 corrected device profile remains the optimization map: dense FFN is
43% and DeltaNet recurrence is 29%. The fallback recurrence cannot meet the PP
goal, so a stable chunk-GDN path or replacement is still required.

## Structural raw-GDN comparison and correction

The Grimoire bridge was diffed directly against vLLM XPU's pristine source:
`csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp`.

The copied Grimoire header was not computationally equivalent. It had three
coupled experimental changes:

1. a custom forward-output tile policy instead of upstream `64x64x32_4x2`;
2. prepare work redistributed from upstream's subgroup/value-head loop into
   separate `(value-head, chunk)` workgroups;
3. forward value dimension split across multiple workgroups
   (`GRIMOIRE_GDN_DV_PARTS=2`) that can touch shared output/state.

All three computational changes were removed. The corrected
`src/xe2_gdn_profile_header.hpp` now matches upstream kernel semantics. The
only remaining differences are required for a bare-metal extraction: no Torch
adapter/dependency, adjusted include paths, fixed BMG selection, and optional
event timing. The bridge and complete Grimoire binaries rebuilt successfully.

The raw entry point was conclusively shown active using
`GRIMOIRE_GDN_PROFILE=1`. That diagnostic currently throws after submission
because Grimoire's queue lacks `enable_profiling`; do not enable this variable
until the diagnostic checks the queue property. This is a host diagnostic bug,
not evidence of a kernel hang.

## Safe ladder with corrected upstream-semantics raw GDN

All tests used only `renderD128`, M=64, a short-lived container, host progress,
and two passes (warmup plus measurement):

- 8 layers: PASS, 22.3 ms, diagnostic 2868.0 tok/s;
- 16 layers: PASS, 36.0 ms, diagnostic 1777.7 tok/s;
- 32 layers: PASS, 63.3 ms, diagnostic 1011.2 tok/s;
- 64 layers: warmup completed every layer; measured pass completed through
  layer 52, then returned Level Zero `UR_RESULT_ERROR_DEVICE_LOST`.

Thus restoring upstream semantics improves confidence and survives at least
192 raw GDN launches in one process, but it does **not** solve the cumulative
full-model/repeated-pass failure. Do not claim the 2868/1778/1011 numbers as
full-model PP; they are deliberately layer-limited diagnostics. The current
verified full-model Qwen baseline remains about 228 PP and 23.3–23.5 TG.

## Interpretation and direct next plan

The repeatable shape is now much narrower: one complete 64-layer M=64 pass is
possible, but a second pass loses the device near layer 52. A 32-layer process
completes both passes. This suggests cumulative queue/kernel/driver resource
state or cross-device reset interference, not a simple tensor-size OOB in one
specific layer. GPU 1's ongoing GuC reset storm may also poison shared driver
state; this cannot be cleanly separated until `objective_sutherland` is gone
after a deliberate shutdown/power cycle.

Next work should prioritize large structural progress:

1. No more GPU workloads in the present device state. Preserve this source.
2. After a deliberate clean boot, verify no stale containers and both devices.
3. Fix profiling-property detection so raw pass timing cannot throw.
4. Run one full 64-layer M=64 process on an otherwise clean system, then exit;
   separately run another process. This distinguishes per-process cumulative
   state from global driver state without doing two prefills in one process.
5. Compare vLLM lifecycle around this kernel: queue/stream, temporary tensor
   lifetime, event dependencies, scratch/state allocation and reuse. Reproduce
   those semantics bare-metal; never add PyTorch or vLLM runtime dependencies.
6. Once stability is proven, optimize the two real PP leaders together: native
   MXFP4 dense FFN and stable chunk GDN. The target remains full-model PP >=2000
   tok/s and coherent TG >=30 tok/s.
7. Only after PP/TG targets are met: evaluate MTP, DFLASH and DSPARK for global
   TG, then validate every future B70-specific quantization.

The core project requirements remain absolute: pure bare metal, no PyTorch or
vLLM runtime; use vLLM only as an implementation reference. Production model
artifacts must start from original BF16 weights and be quantized specifically
into the B70-native MXFP4/AutoRound layout. Never substitute a pre-quantized HF
artifact made for another GPU vendor.

# Final stop-for-night update — direct-host attempt

Ian identified the final `DEVICE_LOST` as an OOM condition. Treat OOM as the
leading diagnosis, not a proven raw-kernel hang. The source supports this:
`Grimoire::prefill()` creates a large temporary USM arena on every call and
frees it at return. The benchmark first runs a 32-token warmup and then creates
a second, larger M-token arena. SYCL/Level Zero may retain the first generation
in its allocation pool, causing cumulative VRAM pressure despite explicit
`sycl::free` calls. The 16.18 GiB resident model alone fits in 31.9 GiB VRAM.

Added `GRIMOIRE_SKIP_PREFILL_WARMUP=1` to `src/grimoire.cpp`. In this mode the
official full-E2E PP benchmark performs only the requested measured prefill,
avoiding the preliminary 32-token allocation generation. The B70 AOT binary
rebuilt successfully.

## Direct-host bare-metal runtime preparation

The server host did not have `libsycl.so.9`; `bin/grimoire` could not initially
run outside the compiler image. A self-contained runtime directory was created:

`/mnt/storage/isos/grimoire-fuse/runtime-b70/lib`

It contains only the required Intel SYCL/Unified Runtime/Level Zero/math shared
libraries copied from the toolchain image. It contains no Python, PyTorch, or
vLLM runtime. With `LD_LIBRARY_PATH` set to this directory, the host executable
links and `./bin/grimoire --help` runs normally.

The single authorized direct-host benchmark was then attempted with Level Zero
0, native bridges, M=64, and warmup skipped. It aborted before model loading and
before any GPU submission with:

`No device of requested type 'info::device_type::gpu' available`

Therefore the shared libraries alone are insufficient: the host is missing or
cannot discover the Intel Level Zero ICD/device-registration userspace that is
present inside the image. No fallback container benchmark and no additional GPU
test was run after this result.

## Next session — exact first actions

1. Start from a deliberate clean shutdown/power cycle so the stuck
   `objective_sutherland` container and GPU 1 GuC reset state are gone.
2. Install or extract the matching Level Zero ICD configuration and required
   userspace components on the host. Validate direct-host enumeration before
   loading a model. Do not use Docker for inference.
3. Run the official benchmark directly on Level Zero 0 with
   `GRIMOIRE_SKIP_PREFILL_WARMUP=1`; accept only `FULL E2E PP` and measured
   decode/graph replay as PP and TG results.
4. Replace per-call `prefill()` USM allocation with one persistent maximum-size
   scratch arena allocated once and reused. This is the production OOM fix;
   skip-warmup is only a diagnostic safeguard.
5. Add reliable B70 VRAM accounting through Level Zero Sysman or explicit
   allocation accounting, logging model bytes plus scratch bytes before any
   submission. The installed host `intel_gpu_top` cannot detect Xe engines and
   exposes no usable memory telemetry.

No GPU work was left running by the direct-host attempt. The old PID 68616 in
`objective_sutherland` on GPU 1 remained untouched as explicitly required.
