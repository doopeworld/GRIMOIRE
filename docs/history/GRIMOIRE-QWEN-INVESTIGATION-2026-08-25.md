# Qwen optimization investigation — 2026-08-25

## Verified baseline

- Canonical MXFP4 + raw attention: 1690.7 PP, 23.2 TG.
- PP4096 region total: 2382.9 ms; E2E 2422.7 ms.
- Full attention is 173.9 ms, not 229 ms.
- GDN recurrence is 99.8 ms. The full seven DN regions total about 706 ms,
  mostly projections rather than recurrence.

## PP4096 region breakdown

- FFN gate/up: 936.2 ms
- FFN down: 446.4 ms
- Full attention: 173.9 ms
- DN qkv: 202.0 ms
- DN z: 121.5 ms
- DN output region: 143.8 ms
- DN recurrence: 99.8 ms
- DN conv/split: 69.3 ms
- DN gate: 56.8 ms

## vLLM comparison

vLLM's Qwen3.5 loader packs `in_proj_qkv` + `in_proj_z` into
`in_proj_qkvz`, and `in_proj_b` + `in_proj_a` into `in_proj_ba`.
The actual Grimoire QKVZ shape is `[16384 x 5120]`, not the old hard-coded
`12352` assumption.

An experimental MXFP4 QKVZ concatenation was implemented and tested. It hangs
the current dense bridge at M=4096; the experiment was removed from source.
This is not a usable fix without adding/tuning a 16384-output dense policy.

## INT4 result

Directly loading `/models/Qwen3.8-27B-int4-AutoRound` with `--proj int4`
produced 1574.2 PP and 14.8 TG. The existing asymmetric INT4 decoder is much
slower than MXFP4, so group-128 conversion alone is not sufficient.

## TG profile

One-token device timeline: 42.68 ms total. Representative per-layer costs:

- FFN gate/up: 241 us
- FFN down: 115 us
- linear QKV: 110 us
- linear output: 44 us
- linear z: 41 us
- full q projection: 103 us
- LM head: 1.526 ms total

The existing wide/split-K GEMV path was already present and its full-model
autotune was run. Best result remained the default at about 43.25 ms/token;
forcing wide/split-K regressed performance. Repeating this sweep is not useful.

## Next high-value work

1. Port or wrap vLLM's exact AutoRound W4A16 decode path, including its packed
   layout, rather than using Grimoire's current INT4 GEMV.
2. Add a tuned 16384x5120 MXFP4 prefill policy, then retry QKVZ fusion. The
   existing bridge does not support this shape safely/performance-portably.
3. The remaining PP target needs about 375 ms, so attention/GDN alone cannot
   close it; the projection/FFN format and fused-shape work is required.

Canonical source was restored after the failed experiments. The deployed
binary behaves canonically when `GRIMOIRE_FUSE_DN_PROJECTIONS` is unset.

---

## QKVZ fusion — MEASURED, and it is not worth building (2026-08-25, later session)

The prior investigation's next-step #2 was "add a tuned 16384x5120 MXFP4 prefill
policy, then retry QKVZ fusion". Both halves of that are now measured with
`tools/bench_sweep.cpp` (64 distinct weight matrices = production memory conditions,
M=4096, no model load).

### The shape needs no new policy, and it does not hang

| policy | QKVZ-fused 16384x5120 | TFLOP/s |
|---|---|---|
| **128x256** | **6.900 ms** | 99.6 |
| 256x128 | 7.099 | 96.8 |
| 128x128 | 7.367 | 93.3 |
| 64x128 | 8.365 | 82.2 |
| 128x64 | 8.471 | 81.1 |
| 64x64 | 10.405 | 66.0 |
| INT4 g128 | 6.492 | 105.9 |

`p128x256` — the existing production policy — handles N=16384 cleanly at the same
~100 TFLOP/s it gets on every other shape. **There is no missing policy, and no hang.**
The hang seen when the concatenation was wired in is therefore a bug in that
experimental code (buffer sizing / stride / scale layout), not a kernel limitation.

### Fusion is slower than the two separate GEMMs

| | ms/layer |
|---|---|
| QKVZ fused (16384x5120) | 6.900 |
| dn-qkv (10240x5120) | 4.281 |
| dn-z (6144x5120) | 2.538 |
| **sum of the two parts** | **6.819** |

**Fusing costs 0.081 ms/layer.** The GEMM work is identical, and the wider N does not
improve efficiency (99.6 vs 100.3 / 101.5 TFLOP/s for the parts). The only real saving
left is one avoided read of the 42 MB activation and one fewer launch per layer —
generously ~0.1-0.2 ms/layer, i.e. **6-13 ms total**, against a 370 ms gap.

**Recommendation: drop QKVZ fusion.** It is not where the missing time is.

### INT4 g128 is consistently ~5% ahead on every shape

105.9 TFLOP/s vs ~100 for MXFP4, on QKVZ, dn-qkv and dn-z alike — matching the earlier
FFN result. But the end-to-end INT4 run measured **1574 PP / 14.8 TG**, a regression,
because GRIMOIRE's existing INT4 path is an **asymmetric** decoder with a different
layout, not the symmetric g128 one benchmarked here. **The ~5% is only reachable by
implementing the symmetric g128 layout**, not by pointing the current INT4 path at an
AutoRound checkpoint.

### What this leaves

Of the ~370 ms still needed for the 2000 gate, these are now excluded by measurement:
QKVZ fusion (6-13 ms), GDN recurrence (99.8 ms total, already efficient), split-K on
the wide GEMVs (regressed), dense tile choice (exhausted), int4 as a drop-in (regressed).

Region budget, corrected: FFN gate/up 936.2, FFN down 446.4, DN qkv 202.0, full
attention 173.9, DN output 143.8, DN z 121.5, DN recurrence 99.8, DN conv/split 69.3,
DN gate 56.8. **The two FFN GEMMs are 1382.6 ms — 58% of the 2382.9 ms total.** They
run at ~100 TFLOP/s, the same as every other shape, and at parity with vLLM's kernel
under identical conditions. Closing a 370 ms gap therefore requires either a
fundamentally faster 4-bit GEMM than either project currently has, or removing work
rather than speeding it up.

---

## PREFILL PIPELINE DIFF — vLLM vs GRIMOIRE, step by step (2026-08-25)

Read from vLLM's own source in the running image:
`vllm/model_executor/layers/qwen_gdn_linear_attn.py` (`_forward_core`,
`_forward_core_fused_norm_packed`, `prepare_gdn_attention_core_inputs`).

### The two pipelines, per layer

| # | vLLM | GRIMOIRE | ms (grim) |
|---|---|---|---|
| 1 | `in_proj_qkvz` — **one** GEMM, N=16384 | `DN qkv projection` — GEMM N=10240 | 202.0 |
| 2 | *(same GEMM)* | `DN z projection` — **separate** GEMM N=6144 | 121.5 |
| 3 | `in_proj_ba` — one GEMM (b and a fused) | `DN gate projection` — GEMM | 56.8 |
| 4 | `prepare_gdn_attention_core_inputs` — **one** cat+slice, deliberately a single Triton kernel | split/rearrange inside `DN causal conv + split` | *(in 69.3)* |
| 5 | `causal_conv1d_fn(..., apply_l2norm=True)` — **conv AND l2norm in one kernel** | `DN causal conv + split`, then **separate** `DN qk norm` | 69.3 + 13.3 |
| 6 | `fused_gdn_gating` — one Triton kernel for alpha/beta | `launch_deltanet_native_gates` | *(small)* |
| 7 | `chunk_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)` — **l2norm fused into the recurrence** | `DN recurrence` | 99.8 |
| 8 | `_rms_norm_gated_cuda` — **RMSNorm AND the z output-gate in one kernel** | `DN norm + output projection` (norm, gate and GEMM in one region) | 143.8 |
| 9 | `out_proj` GEMM | *(included in 8)* | |
| 10 | dense FFN | `dense FFN` | 1382.6 |
| 11 | full attention (only on the layers in `get_layer_pattern`) | `full attention` | 173.9 |

### Where they actually differ

**1. vLLM fuses aggressively; GRIMOIRE runs discrete kernels.** Three explicit fusions,
each with a source comment saying why:
- conv + l2norm in one kernel (GRIMOIRE: two regions, 69.3 + 13.3 ms)
- l2norm inside the recurrence kernel (GRIMOIRE: separate `qk norm`)
- RMSNorm + output gate in one kernel
- and `prepare_gdn_attention_core_inputs` does one `cat`+slice explicitly so
  "torch.compile emits one Triton copy kernel instead of three separate
  `contiguous()` calls".

**2. QKVZ: fused in vLLM, two GEMMs in GRIMOIRE — but this is NOT worth copying.**
Measured directly: fused 16384x5120 = 6.900 ms/layer vs 4.281 + 2.538 = 6.819 ms for
the two parts. Fusing is *slower*. Only the avoided activation re-read (~0.1 ms/layer)
is real. **Excluded.**

**3. The FFN is identical in structure and speed.** 1382.6 ms, 58% of prefill, at
~100 TFLOP/s — the same rate GRIMOIRE achieves on every other shape, and at parity
with vLLM's kernel measured under identical streaming conditions.

### The honest arithmetic

Total GRIMOIRE elementwise/fusible DN work: `conv+split 69.3 + qk norm 13.3` = 82.6 ms,
plus whatever of the 143.8 ms `norm + output` region is norm rather than GEMM.
**Even fusing all of it perfectly recovers well under 100 ms of the ~370 ms needed.**

So the fusion differences are real, documented, and **too small to close the gap**.
They are worth doing (they are also the cheapest remaining wins) but they are not the
answer on their own.

### The one structural question not yet answered

`Qwen3_5ForCausalLM.get_layer_pattern()` returns `(2, num_layers // 2, num_layers - 3)`
— vLLM runs **full attention on only a subset of layers**, the rest being GDN-only.
GRIMOIRE spends 173.9 ms in `full attention`. **Nobody has verified that GRIMOIRE
applies full attention to the same layer subset vLLM does.** If GRIMOIRE runs it on
more layers than the architecture requires, that is pure wasted work and it is
directly measurable.

**This is the next thing to check** — it is cheap, it is structural rather than a
micro-optimisation, and it is the one place where the two pipelines might be doing a
*different amount of work* rather than the same work at different speeds. Everything
else in this document is the same work.

---

## THE PP GAP, LOCATED (2026-08-25)

### It is entirely outside the FFN

| | GRIMOIRE | vLLM |
|---|---|---|
| total prefill @4096 | 2382.9 ms | ~2032 ms (2015.6 tok/s) |
| dense FFN | 1382.6 ms | ~1382 ms — same kernel, measured at parity |
| **everything else** | **1000.3 ms** | **~650 ms** |

**The gap is 350 ms of non-FFN work.** The FFN is 58% of prefill, runs at
101.4 TFLOP/s, and is at parity with vLLM's kernel under identical streaming
conditions (14.65 vs 14.67 ms/layer). **Stop optimising it.**

### The single biggest inefficiency: full attention at 19 TFLOP/s

Causal attention FLOPs, M=4096, 24 heads, head_dim 256, 16 full-attention layers
(`full_attention_interval=4`, 48 GDN + 16 full — GRIMOIRE does respect this, verified
at `qwen35_loader.cpp:115`):

```
2 x (M^2/2 x heads x head_dim x 2) x 16 layers = 3.30e12 FLOP
measured 173.9 ms  ->  19.0 TFLOP/s
```

| kernel | TFLOP/s |
|---|---|
| dense FFN GEMM | **101.4** |
| DN projections | ~100 |
| **full attention** | **19.0** |

**Our attention kernel is 5.3x less efficient than our own GEMMs on the same card.**
10.87 ms/layer. vLLM runs FlashAttention here (`xpu.py:198 Using Flash Attention
backend`). At even 60 TFLOP/s this region becomes ~55 ms — **119 ms recovered from one
kernel**, and it is the largest single win identified in this project.

### The rest: unfused DN elementwise work, ~150 ms

vLLM fuses three things GRIMOIRE runs as separate kernels (source-verified in
`qwen_gdn_linear_attn.py`):
- `causal_conv1d_fn(..., apply_l2norm=True)` — conv + l2norm in one kernel
  (GRIMOIRE: `conv+split` 69.3 ms **and** a separate `qk norm` 13.3 ms)
- `chunk_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)` — l2norm inside the
  recurrence
- `_rms_norm_gated_cuda` — RMSNorm + z output-gate in one kernel

**119 + 150 = ~270 of the 350 ms.** That accounts for the gap.

### Priority order for the remaining work

1. **Full attention kernel — 119 ms.** Biggest single win, 5x headroom, and vLLM's
   FlashAttention XPU backend is in the image to compare against. Start here.
2. **Fuse the DN elementwise work — ~150 ms.** conv+l2norm, l2norm into the
   recurrence, rmsnorm+gate. Mechanical, follows vLLM's structure exactly.
3. Everything else is measured and excluded: FFN tile choice, int4, QKVZ fusion,
   split-K, GDN recurrence, GPU choice, prompt length, TP.

### What was wrong before

Sections above this one chased the FFN and the dense GEMM for most of a day. The FFN
was never the problem — it is 58% of the time but it is already at vLLM's speed. The
attention kernel's efficiency was never measured until now; it had been reported only
as a raw millisecond figure (173.9 ms), which looks small next to 1382 ms of FFN and
therefore never attracted attention. **Efficiency, not absolute time, is what
identifies a bad kernel.**

---

## Post-nap session: three more leads tested, all negative

### 1. Attention — GRIMOIRE's own kernel is FASTER than vLLM's FMHA bridge

Both bridges load cleanly (`chunk prefill unavailable = 0` in both runs):

| `GRIMOIRE_XE2_ATTN_BRIDGE` | full attention | E2E |
|---|---|---|
| **`libgrimoire_xe2_attention_raw.so`** (GRIMOIRE's own, self-contained) | **173.9 ms** | **2421.9 ms** |
| `libgrimoire_xe2_attention_bridge.so` (links vLLM's real `libattn_kernels_xe_2.so`) | 225.2 ms | 2475.5 ms |

**The earlier claim that attention's 19 TFLOP/s represented 119 ms of headroom was
wrong.** vLLM's own FMHA kernel, called through this bridge, is 30% *slower* on our
shapes. 19 TFLOP/s may simply be what causal attention with head_dim=256 and GQA 24/4
costs here. **Keep `attention_raw`. No win available.**

### 2. 2-D L2 block swizzle — no effect at all

Implemented (`GRIMOIRE_GEMM_SWIZZLE_BN`, blocks of BM x BN tiles, m fastest inside a
block) and swept under streaming conditions:

| GM x BN | gate-up | down |
|---|---|---|
| 4 x 0 (production) | **14.627** | 7.449 |
| 4 x 8 | 14.689 | 7.380 |
| 8 x 8 | 14.649 | 7.445 |
| 4 x 16 | 14.677 | 7.394 |
| 8 x 16 | 14.672 | 7.451 |
| 16 x 8 | 14.721 | 7.625 |
| 8 x 32 | 14.643 | 7.450 |
| 16 x 16 | 14.822 | 7.617 |

**Every combination is within noise of the existing GM=4.** The traffic model that
predicted a 5.5x reduction in B re-reads was wrong — the hardware L2 already captures
that reuse. The code is left in place (BN=0 default = disabled) but there is nothing
to gain. **Do not revisit tile scheduling.**

### 3. The FFN streaming penalty is inherent, not ours

GRIMOIRE 14.63 ms/layer streaming vs 11.87 cache-hot. vLLM's int4 kernel measured
under the identical harness: 14.67 vs 11.73. **Both kernels lose the same ~2.8 ms to
streaming 89 MB of weights per layer with no cross-layer reuse.** This is a property
of the shape and the memory system, not of either implementation.

### Cumulative list of dead ends (all measured, do not retry)

dense tile choice (13 policies, both cache-hot and streaming) | int4 as a drop-in
(regressed to 1574 PP) | int4 g128 (~3%, needs a converter) | QKVZ fusion (slower than
the two parts) | GM swizzle beyond 4 | 2-D block swizzle | oneDNN W4A16 | split-K wide
GEMV | M scaling | GPU choice | tensor parallelism | vLLM's FMHA bridge | GDN
recurrence | layer-type pattern (already correct).

### Honest state

Qwen 1694 PP / 23.0 TG against gates of 2000 / 28. **Everything cheap is exhausted.**
The remaining identified work is the DN elementwise fusion (~150 ms estimated, from
the pipeline diff above): conv+l2norm in one kernel, l2norm inside the recurrence,
rmsnorm+gate together. That is real kernel engineering, not a flag or a policy, and it
is worth roughly 40% of the remaining gap on its own.

---

## THE DIFFERENCE, MEASURED KERNEL-BY-KERNEL (vLLM torch profiler, prefill 4096)

Profiled vLLM in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `enable_prefix_caching=
False`, distinct warm-up and measured prompts so nothing is served from cache).
`tools/vllm_prof.py`.

**Methodology caveat, stated up front:** this run used `enforce_eager=True`, which
disables inductor fusion and inflates the surrounding `aten::copy_` / elementwise rows
(Ian: "I never use enforce eager, it makes results 30% slower"). **It does NOT change
the dedicated kernels**, so the two comparisons below — both dedicated kernels — are
valid. The elementwise/copy rows in that profile are not.

### The two kernels that differ

| kernel | vLLM | GRIMOIRE | delta |
|---|---|---|---|
| **full attention** — `_vllm_fa2_C::varlen_fwd` (16 layers) | **45.5 ms** (2.84 ms/layer) | **173.9 ms** (10.87 ms/layer) | **-128 ms** |
| **GDN core** — `_xpu_C::gdn_attention` (48 layers) | **112.8 ms** | 182.4 ms (conv 69.3 + qk norm 13.3 + recurrence 99.8) | **-70 ms** |

**198 ms of the ~370 ms gap, in two kernels.**

### Why this was missed for a day

GRIMOIRE's `libgrimoire_xe2_attention_bridge.so` links **`libattn_kernels_xe_2.so`**.
**vLLM's fast attention path does not use that library at all** — it uses
**`_vllm_fa2_C.abi3.so`** (FlashAttention-2). Benchmarking the attention bridge
therefore measured the *wrong vLLM kernel*, came out slower than GRIMOIRE's own
(225.2 vs 173.9 ms), and produced the false conclusion "no win available in attention".

### The target

```
/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_vllm_fa2_C.abi3.so   (135 KB)
  T FLASH_NAMESPACE::mha_varlen_fwd(at::Tensor const&, ...)
```
A thin wrapper over a cutlass kernel (the profile's
`compat::experimental::detail::KernelFunctor<&(void cutla...>` row, 45.45 ms / 16 calls,
matches `varlen_fwd` exactly).

It takes `at::Tensor` arguments. **GRIMOIRE's attention bridge already links libtorch,
`libc10`, `libc10_xpu`** (that is why it needs those on `LD_LIBRARY_PATH`), so the
existing bridge pattern reaches this symbol — construct `at::Tensor` views over the
existing USM q/k/v buffers and call `mha_varlen_fwd`. No new dependency beyond what the
bridge already pulls in.

**Estimated: 128 ms, the single largest remaining win, and now precisely located.**

### Next after that

`_xpu_C::gdn_attention` is vLLM's **fused** GDN core — one kernel replacing GRIMOIRE's
conv, qk-norm and recurrence (70 ms). Same approach: it is a registered torch op in
`_C`/`_xpu_C`, reachable through the same libtorch-linked bridge.

### Reproduce

```bash
docker run -d --name vp --init --device /dev/dri -v /dev/dri:/dev/dri \
  -e ZE_AFFINITY_MASK=0 -e CCL_ZE_IPC_EXCHANGE=sockets \
  -v /mnt/storage/Models:/models -v /mnt/storage/isos/grimoire-fuse:/grimoire \
  --entrypoint bash my-vllm-xpu:latest -c "python3 -u /grimoire/tools/vllm_prof.py"
```
`-v /dev/dri:/dev/dri` is required — oneCCL calls `opendir` on the device directory and
`--device` alone is not enough. Drop `enforce_eager` before trusting any elementwise row.

### Ian's exact vLLM launch config (the one that produces 2015.6 PP / 27.95 TG)

```
/models/Qwen3.8-27B-int4-AutoRound --dtype bfloat16 --port 3551 --host 0.0.0.0
  --max-model-len 32768 --max-num-seqs 16 --kv-cache-dtype fp8
  --max-num-batched-tokens 16384 --block-size 64 --language-model-only
  --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
env: VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Differences from what the profiling run above used, which may shift the numbers:
- **`--kv-cache-dtype fp8`** — halves KV cache traffic in attention. Not used in the
  profile; the measured attention advantage may be partly attributable to it, so
  **re-profile with fp8 before finalising the 128 ms estimate**.
- `--language-model-only` — skips the visual tower entirely.
- `--max-num-seqs 16` vs 1, `--max-model-len 32768` vs 8192.
- Never `enforce_eager`.

---

## FA2 BRIDGE — BUILT AND WORKING, blocked on stream integration

### What was built

`src/xe2_fa2_bridge.cpp` -> `src/libgrimoire_xe2_fa2.so`. Calls vLLM's
**`_vllm_fa2_C::varlen_fwd`** (FlashAttention-2) through the torch dispatcher.
Wired into `load_xe2_chunk_prefill()` in `grimoire.cpp`, **opt-in via
`GRIMOIRE_ENABLE_FA2=1`**. Same signature as the existing chunk-prefill entry, so it
is a drop-in.

Three things had to be solved to reach the op at all, all now handled in the bridge:
1. **Call boxed, not by symbol.** `mha_varlen_fwd` has 28 parameters; binding the raw
   C++ symbol risks stack corruption. The bridge builds an `IValue` stack matching the
   registered schema exactly (dumped and reproduced in the source comment).
2. **The op only registers when `_vllm_fa2_C.abi3.so` is loaded** — its `TORCH_LIBRARY`
   static initialiser. The bridge `dlopen`s it (`RTLD_GLOBAL`).
3. **It is a CPython extension** — needs `PyModule_Create2`, so `libpython3.12.so.1.0`
   must be `dlopen`ed `RTLD_GLOBAL` first. Without this: `undefined symbol`.

Build (note: NOT `-DC10_USE_GLOG=0`, which still *defines* the macro and pulls in
missing glog headers; and the SYCL test must be built in `grimoire:queuefix` because
`my-vllm-xpu` has a libsycl/UR mismatch):
```bash
docker run --rm --entrypoint bash -v /mnt/storage/isos/grimoire-fuse:/grimoire my-vllm-xpu:latest -c '
cd /grimoire; T=/opt/venv/lib/python3.12/site-packages/torch
icpx -std=gnu++17 -O3 -fPIC -shared -fsycl -I$T/include -I$T/include/torch/csrc/api/include \
  src/xe2_fa2_bridge.cpp -o src/libgrimoire_xe2_fa2.so \
  -L$T/lib -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -ldl -Wl,-rpath,$T/lib'
```

### Standalone: 4x faster, confirmed

`bin/test_fa2` (Qwen shape M=4096, QH=24, KH=4, D=256, causal):

| | ms/layer | TFLOP/s |
|---|---|---|
| **FA2 (vLLM)** | **2.713** | **76.0** |
| GRIMOIRE in-model | 10.87 | 19.0 |

Matches the profiled 2.84 ms/layer. Output fully populated (25,165,824/25,165,824
non-zero). **Over 16 full-attention layers that is ~130 ms.**

### In-model: SLOWER. 226.1 ms vs 173.9 ms.

Wired in and measured: `full attention 226.066 ms` against `attention_raw`'s 173.9 ms,
E2E 1657.2 vs 1696.7 tok/s. **A regression, so it is opt-in and OFF by default.**

**Cause:** the op runs on **torch's** XPU stream; GRIMOIRE runs its own `sycl::queue`.
The bridge must therefore `queue->wait()` before the call and
`c10::xpu::getCurrentXPUStream().synchronize()` after — **32 full pipeline stalls per
forward pass** across 16 layers, costing more than the 8 ms/layer the kernel saves.
Standalone the harness pays those same syncs but has nothing to overlap, so they are
free there and expensive in the model. That is the entire discrepancy.

### The fix

Make the op execute on GRIMOIRE's queue instead of torch's, so no cross-stream sync is
needed. torch-xpu exposes stream adoption (`c10::xpu::getStreamFromExternal` /
`XPUStreamGuard`); wrapping GRIMOIRE's `sycl::queue` as the current XPU stream for the
duration of the call should remove both syncs. **If that works the ~130 ms is real and
the same technique then applies to `_xpu_C::gdn_attention` (a further ~70 ms), which
has exactly the same cross-stream problem.**

### Status

Qwen **1696.7 PP / 23.0 TG** (baseline intact, FA2 off). The bridge, the test
(`bin/test_fa2`), and the integration are committed and working; only the stream
plumbing stands between this and the 130 ms.

---

## RETRACTION: the "128 ms in attention" was a measurement error. FA2 does not help.

### The error

The `full attention` region does **not** measure the attention kernel. Between its
bounding `pp_mark`s (`grimoire.cpp:2710`-`2786`) it contains **the q/k/v/o projections
(4 GEMMs)**, rope, kv-append, three f32->bf16 conversions and the output gating.

The "19.0 TFLOP/s, 5.3x worse than our GEMMs" figure came from dividing
**attention-only FLOPs** by the **whole region's time**. Wrong denominator.

Done correctly for 16 full-attention layers, M=4096, hidden 5120:
- projections: `2*4096*(6144*5120 + 2*1024*5120 + 5120*6144) * 16` = **9.62e12 FLOP**
- attention: **3.30e12 FLOP**
- region total **1.29e13 FLOP / 173.9 ms = 74.2 TFLOP/s**

**That is healthy and in line with every other region.** GRIMOIRE's attention was never
slow, and there was never 128 ms to recover.

### FA2 measured in-model, twice

| | full attention region | E2E |
|---|---|---|
| `attention_raw` (GRIMOIRE's own) | **173.9 ms** | **1696.7 tok/s** |
| FA2, cross-stream (wait + synchronize) | 226.1 ms | 1657.2 |
| FA2, stream adopted (both syncs removed) | 225.3 ms | 1655.4 |

**Stream adoption changed nothing** (226.1 -> 225.3), which disproves the
"32 pipeline stalls" explanation as well. vLLM's FlashAttention-2 is simply slower than
GRIMOIRE's own attention kernel on these shapes, in this pipeline.

The earlier standalone result (FA2 2.713 ms vs "attention_raw 0.000 ms") was not a
valid comparison: the baseline returned without computing — its preconditions were not
met in the harness — so it measured nothing. **A 0.000 ms result should have been
treated as a broken measurement, not a baseline.**

### Status of the FA2 work

`src/xe2_fa2_bridge.cpp`, `src/libgrimoire_xe2_fa2.so`, `bin/test_fa2` and the
`GRIMOIRE_ENABLE_FA2` hook are all kept: the bridge correctly reaches
`_vllm_fa2_C::varlen_fwd` and the three obstacles to calling any vLLM torch op from
GRIMOIRE are solved and documented (boxed dispatcher call, extension autoload,
libpython preload, plus stream adoption via `getStreamFromExternal`). **That machinery
is reusable for `_xpu_C::gdn_attention` and any other vLLM op.** The FA2 kernel itself
is off by default because it is slower.

### Where that leaves the gap

The ~370 ms is still not located. What is now excluded by measurement: the dense FFN
(at parity with vLLM), attention (74 TFLOP/s, healthy, and vLLM's kernel is slower),
tile choice, int4, QKVZ fusion, 2-D swizzle, oneDNN, split-K, GPU choice, TP, M
scaling, GDN recurrence, layer pattern.

**Method note for whoever continues: never attribute a region's time to one kernel
inside it.** Two wrong conclusions today came from exactly that — first "443 ms of
non-GEMM FFN overhead", now "128 ms in attention". Put a `pp_mark` around the single
kernel before computing its efficiency.

---

## vLLM's linear layers run on BesTLA, not cutlass — and that is where its time goes

Clean profile (no `enforce_eager`, prefix caching off, distinct prompts), Qwen3.8-27B
int4-AutoRound, prefill 4096, in-process:

| op | ms | count | share |
|---|---|---|---|
| `vllm::inc_ark_woq_linear` | **1204.10** | 256 | 32.9% |
| `bestla::sycl_gemm::xmx::IKblockGemmDQCore` | **1100.57** | 256 | 30.0% |
| `aten::copy_` | 287.38 | 580 | 7.8% |
| `_xpu_C::gdn_attention` | 114.42 | 48 | 3.1% |
| `_vllm_fa2_C::varlen_fwd` | 46.38 | 16 | 1.3% |
| `triton_poi_fused_mul_silu_slice_3` | 39.98 | 48 | 1.1% |
| `triton_red_fused__to_copy_add_fused_add_rms_norm_4` | 19.47 | 48 | 0.5% |

**The linear layers are ~63% of vLLM's device time** (the op row and its kernel row
are the same work counted at two levels). Everything else — GDN, attention, the fused
Triton norms — is small.

### The kernel is BesTLA

`bestla::sycl_gemm::xmx::IKblockGemmDQCore` is Intel Neural Compressor's **BesTLA**
GEMM, reached through `vllm::inc_ark_woq_linear` -> `auto_round_kernel.woqgemm_linear`.

**This is a third, distinct kernel** — not GRIMOIRE's cutlass MXFP4, and **not**
`torch.ops._xpu_C.int4_gemm_w4a16`, which is the INC **wna16** path and is what an
earlier comparison in this document benchmarked. That earlier "int4 and MXFP4 are at
parity" result therefore compared against a kernel this model never runs.
**AutoRound checkpoints dispatch to `INCARKLinearMethod` -> ARK -> BesTLA.**

### What is NOT yet established

A direct BesTLA-vs-cutlass number on our shapes. `ark.repack_quantized_weight` rejects
synthetic packed weights (`Corrupt packed weight: blob size ... less than expected`);
the expected blob is `N * (K/2 + K/groupsize * 2)` bytes and it appears to derive K
from a tensor other than the one assumed. **The absolute ms in the table above cannot
be compared directly against GRIMOIRE's region budget either** — profiling inflates
them (total device time 3665 ms against a ~2032 ms real prefill), so only the
*proportions* within the profile are meaningful.

### Next step, concrete

Get a real packed weight instead of a synthetic one: load one `mlp.gate_proj` tensor
group (`qweight`/`scales`/`qzeros`) straight out of
`/models/Qwen3.8-27B-int4-AutoRound`, repack it with the same call
`INCARKLinearMethod` uses, and time `woqgemm_linear` against
`launch_dense_mxfp4` on the identical shape. That yields the one number that decides
whether porting BesTLA is worth it — and it is the same mistake-proof approach that
should have been used from the start: **use the real artifact, not a synthetic one.**

If BesTLA is meaningfully faster on 34816x5120 and 5120x17408, it is the answer to the
whole PP gap, and the bridge machinery to call it already exists
(`src/xe2_fa2_bridge.cpp` shows the pattern: boxed dispatcher call, extension
autoload, libpython preload, stream adoption).

---

## CONFIRMED: BesTLA is 1.37-1.47x faster than our GEMM. This is the PP gap.

Measured with **real weights from `/models/Qwen3.8-27B-int4-AutoRound`** (not
synthetic), M=4096, group_size 128, symmetric, `tools/bestla_real.py`:

| shape | BesTLA (`ark.woqgemm_linear`) | GRIMOIRE (`launch_dense_mxfp4`) | ratio |
|---|---|---|---|
| ffn-gate N=17408 K=5120 | **4.982 ms — 146.6 TFLOP/s** | 7.314 ms — 99.8 TFLOP/s | **1.47x** |
| ffn-down N=5120 K=17408 | **5.073 ms — 143.9 TFLOP/s** | 6.975 ms — 104.7 TFLOP/s | **1.37x** |

**BesTLA sustains ~145 TFLOP/s; our cutlass MXFP4 path sustains ~100.** Same GPU, same
shapes, same M. This is the answer to the whole PP gap.

### Projected

| region | now | at BesTLA rate | saving |
|---|---|---|---|
| FFN gate_up | 936.2 | ~637 | 299 |
| FFN down | 446.4 | ~326 | 120 |
| DN projections | ~480 | ~343 | 137 |
| attention projections | ~130 | ~93 | 37 |
| **total** | | | **~593 ms** |

2417 - 419 (FFN only) = **1998 ms = 2050 tok/s — the gate is met on the FFN alone.**
With all projections: 1824 ms = **~2245 tok/s.**

*Caveat: the 1.47x was measured on N=17408 (`gate_proj`); Qwen's fused gate_up is
N=34816. Verify the ratio holds at that width before banking the full 299 ms.*

### The packed-weight format (this cost several attempts — write it down)

`ark.repack_quantized_weight(QB, scaleB, zp, groupsize, compute_type, weight_type,
scale_type, asym)`:
- **`QB` must be UNPACKED int8 `[K, N]`**, values 0..15. It reads K from `QB.shape[0]`.
  Passing the checkpoint's GPTQ int32 `[K/8, N]` silently produces a blob 8x too small
  and `woqgemm_linear` then fails with
  `Corrupt packed weight: blob size ... less than expected`.
  Unpack with: `((qw.unsqueeze(1) >> (arange(8)*4).view(1,8,1)) & 0xF).reshape(K,N).to(int8)`
- `scaleB` bf16 `[K/groupsize, N]`; `zp` None when symmetric.
- `weight_type` is **`"int4"`** for `woqgemm_linear` (it rejects `"int4_clip"`, which
  `repack` accepts — they use different vocabularies).
- Expected blob size is exactly `N * (K/2 + K/groupsize * 2)` bytes.

### How to port it

The bridge pattern is already proven by `src/xe2_fa2_bridge.cpp`: boxed dispatcher
call, `dlopen` the extension to trigger `TORCH_LIBRARY` registration, preload
`libpython3.12.so.1.0`, and adopt GRIMOIRE's `sycl::queue` as the current XPU stream via
`c10::xpu::getStreamFromExternal`. Same four steps for `ark.woqgemm_linear`.

Two pieces of work:
1. **Bridge** — `grimoire_bestla_linear_bf16(queue, x, packed, out, M, N, K, group)`.
   Repack once at model load, not per call (`repack_quantized_weight` is host-side).
2. **Converter** — `b70_compile_model.cpp` must emit **int4 g128 symmetric** for the
   linear weights (unpacked int8 `[K,N]` + bf16 scales `[K/G,N]`), keeping the
   `keeps_bf16()` exclusions. It currently emits MXFP4 only.

Unlike every previous lead in this document, this one is measured end-to-end on the
real artifact and the arithmetic reaches the gate.

---

## PORTING BesTLA: what is possible, and what is not

### It is not reachable from C++

`auto_round_kernel_xpu.cpython-312-x86_64-linux-gnu.so` (30 MB):
- registers **zero torch ops** (verified by diffing the schema registry before/after
  `CDLL`),
- exports **one** symbol: `PyInit_auto_round_kernel_xpu`. Everything else is hidden.

It is a pure pybind11 module. There is no C ABI, no dispatcher entry — the technique
that worked for FlashAttention-2 (`src/xe2_fa2_bridge.cpp`) **cannot** work here.

### The source is not public

`github.com/intel/neural-speed` -> `bestla/bestla/sycl/` contains the framework
(`sycl_gemm.h`, `sycl_prologue_b.h` with `WeightS4`/`WeightS4Trans`, matching the
`bestla::sycl_prologue_b::WeightS4T` seen in the profile) — **but its GEMM cores are
`SGemmCoreSharedB` / `HGemmCoreSharedB`, fp32 and fp16 only, and there is no
xmx/dpas/joint_matrix code anywhere in the SYCL tree.**

`IKblockGemmDQCore` — the int4-dequant XMX kernel that vLLM actually runs — is newer
code that ships only in the wheel. `github.com/intel/bestla` requires authentication.

### Therefore, three options and no fourth

**A. Embed CPython in GRIMOIRE and call `ark.woqgemm_linear`.**
Works today. Repack once at load; 256 calls per prefill, each a Python call over
tensors wrapping existing USM pointers — call overhead is microseconds against ~1100 ms
of compute, so **it would deliver the measured 1.37-1.47x and reach ~2050-2245 tok/s**.
Cost: GRIMOIRE links `libpython`, holds the GIL around each call, and stops being pure
C++/SYCL. **This is the only option that reaches the gate with known effort.**

**B. Write our own int4 g128 XMX GEMM.**
Keeps GRIMOIRE pure C++/SYCL and B70-specific, which is the stated design goal. The
target is known and measured (~145 TFLOP/s vs our ~100), the shapes are known, and the
public BesTLA prologues show the dequant structure. But it is real kernel engineering
with an uncertain schedule, and our existing cutlass path has already resisted every
tuning attempt in this document.

**C. Ship ~1695 tok/s** and treat 2000 as unreachable without one of the above.

### Recommendation

**A to prove the number end-to-end, then B to remove Python.** A is a few hours and
converts the measurement into a real 2000+ PP result; if it lands, B becomes a
well-specified optimisation with a known target rather than a speculative rewrite.
Doing B first risks weeks against an unvalidated integration.

### Also required either way

`b70_compile_model.cpp` must emit **int4 g128 symmetric** for linear weights (unpacked
int8 `[K,N]` + bf16 scales `[K/G,N]`, `keeps_bf16()` exclusions preserved). It emits
MXFP4 only today. This is needed for A and B alike.

---

## STEP 3 PROVEN: GRIMOIRE can call BesTLA with its own USM buffers, at zero overhead

`src/bestla_bridge.cpp` -> `src/libgrimoire_bestla.so`, driven by
`tools/drive_bestla.cpp` -> `bin/drive_bestla`:

```
device: Intel(R) Arc(TM) Pro B70 Graphics
  packed real checkpoint weight: N=17408 K=5120 G=128
  GRIMOIRE USM buffer 4096 x 5120 handed to BesTLA

  BesTLA via embedded CPython:    4.976 ms   146.7 TFLOP/s
  GRIMOIRE cutlass MXFP4     :    7.314 ms    99.8 TFLOP/s
  speedup: 1.47x
  STEP-3 FEASIBILITY: PASS
```

**The embedded-CPython overhead is zero.** 4.976 ms here vs 4.982 ms from the pure
Python benchmark — identical. The Python call is microseconds against milliseconds of
GPU work, and `c10::xpu::getStreamFromExternal` puts the kernel on **GRIMOIRE's own
queue**, so there is no cross-stream sync either.

### The working chain

```
sycl::malloc_device (GRIMOIRE USM)
  -> at::from_blob(ptr,{M,K},kBFloat16/kXPU)
  -> THPVariable_Wrap  (libtorch_python)
  -> auto_round_kernel.woqgemm_linear(x, packed, None, N, K, G, "bf16","int4","bf16", False)
  -> THPVariable_Unpack -> at::Tensor -> USM
```
`grimoire_bestla_init()` brings up CPython, imports the module and repacks the weight
**once**; `grimoire_bestla_linear()` is the per-call path and allocates nothing but the
tensor wrappers.

### Build recipe (two images, deliberately)

- **Bridge (.so)** in `my-vllm-xpu:latest` — it has torch, `libtorch_python`,
  `Python.h`. Needs `-ltorch_python -lpython3.12` and `-I/usr/include/python3.12`.
- **SYCL executable** in `grimoire:queuefix` — `my-vllm-xpu` cannot *link* SYCL
  executables (`urGraphGetIdExp@LIBUR_LOADER_0.12` undefined), though it *runs* them
  fine. So: build the exe in queuefix, run it in the vLLM image.

### What remains for a full end-to-end result

1. **Converter** — `b70_compile_model.cpp` to emit int4 g128 symmetric (unpacked int8
   `[K,N]` + bf16 scales `[K/G,N]`), `keeps_bf16()` preserved.
2. **Loader** — new encoding -> the int4 path.
3. **Routing** — send dense linears through the bridge; MoE grouped GEMM stays on
   cutlass (BesTLA exposes no grouped variant), so Ornith's expert path is untouched.
4. Re-convert, measure, verify output coherence.

Projected: Qwen ~2050-2245 tok/s; Ornith ~25-30% of its 409 ms is dense GEMM, so
~33 ms -> **~10,400 tok/s, past its gate**.

**This is the go/no-go that was outstanding, and it is a go.** Python is the
scaffolding to prove the number; the eventual pure-C++/SYCL port (option B) now has a
validated target instead of a speculative one.

---

## BesTLA is a PREFILL kernel only — it must be dispatched by M

Measured with `bin/drive_bestla <M>` on the real `gate_proj` weight (N=17408, K=5120,
g128), GRIMOIRE USM buffers, embedded CPython:

| M | BesTLA | weight bandwidth | regime |
|---|---|---|---|
| **1** (decode) | 0.464 ms | **99.1 GB/s** | memory-bound |
| 8 | 0.794 ms | 57.9 GB/s | memory-bound |
| 32 | 0.384 ms | 119.7 GB/s | memory-bound |
| **4096** (prefill) | 4.988 ms | **146.4 TFLOP/s** | compute-bound |

The B70's measured memory roofline is **602 GB/s**. GRIMOIRE's own GEMV reaches
**386-393 GB/s** on FFN shapes at M=1. **BesTLA reaches 99 GB/s — roughly 4x slower
than what GRIMOIRE already has for decode.**

### This explains the earlier INT4 TG collapse

The end-to-end AutoRound INT4 run measured **1574 PP / 14.8 TG** against MXFP4's
1691 / 23.2. The TG half was never an integration bug: **BesTLA is a prefill kernel**
and is simply bad at M=1. Anyone routing decode through it will see the same collapse.

### Consequence for the design

**Dispatch on M.** BesTLA above a threshold (~64, where the chunked/native paths already
switch), GRIMOIRE's existing GEMV below it. This is what vLLM does as well — hence its
`w4a16_policy_m_8 / m_16 / m_32` policy variants and a separate wna16 decode path
alongside the ARK prefill path.

| | gain from BesTLA |
|---|---|
| **Qwen PP** | **+47% on every linear** -> ~2050-2245 tok/s |
| **Ornith PP** | +47% on non-expert linears only (~25% of its time) -> ~10,400 tok/s |
| **Qwen TG** | **none** — would be a 4x regression if used |
| **Ornith TG** | **none** — same |

**TG remains an entirely separate problem for both models** (23.2 vs 28, and vLLM's
27.95). BesTLA contributes nothing to it. Whatever closes the TG gap will be a decode
kernel, not this.

### Also note

Ornith is **MoE**, not dense. BesTLA exposes no grouped variant, so its two largest
GEMM regions — `MoE fused gate+up+SwiGLU` 72.9 ms and `MoE down GEMM` 34.3 ms, 107 ms
of expert work — are **untouched**. Only the ~95-120 ms of ordinary linears benefit.
Qwen, being dense, benefits everywhere; that is where the value is.

---

## ROADMAP ITEM: tensor parallelism is required for anything above 4-bit

**Ian's point, 2026-08-25:** the B70's 32 GB is the largest consumer VRAM available, and
FP8 / INT8 / BF16 weights do not fit a single card for these model sizes. TP is
therefore not optional for those formats — it is the gate to supporting them at all.

| Qwen3.8-27B | weights | one 32 GB B70? |
|---|---|---|
| BF16 | ~54 GB | **no** — 2-3 cards |
| FP8 / INT8 | ~27 GB | **no** — weights fit, nothing left for KV or activations |
| INT4 / MXFP4 | ~14-16 GB | yes |

Ornith-1.5-35B-A3B: ~70 GB BF16, ~35 GB FP8 — **neither fits one card.**

**Formats are already supported; multi-GPU is not.** GRIMOIRE's kernels specialise on
`BF16, INT8, FP8_E4M3, FP8_E5M2, INT4, MXFP4, MXFP8` (see the `GemvGeom` specialisations
in `src/kernels.hpp` and the matching `gemm_xmx.cpp` dispatch). What is missing is
**tensor parallelism**: `src/grimoire.cpp:728` is a single `sycl::queue`, and there is no
`tp_size` / sharding / all-reduce anywhere.

**Work required (deferred to after the PP/TG gates):**
1. Megatron sharding — column-split `gate_up`, row-split `down`, head-split QKV.
2. One all-reduce per layer, after `down` and after the attention output projection.
   Check Level Zero peer access between `0000:03:00.0` and `0000:35:00.0`; stage through
   host if unavailable.
3. Multi-device queue/context management.

**Note this is orthogonal to the BesTLA work.** BesTLA is tied to the AutoRound int4
checkpoint format, so an FP8/BF16 user falls back to GRIMOIRE's own kernels — which is
precisely why the M-dispatch design keeps both paths alive rather than replacing the
existing GEMM.

**It also will not speed up the current 4-bit single-card case:** both B70s measure
equivalent, and vLLM reaches 2015 PP on one card. TP is about *capability* (running
FP8/BF16 at all), not about the current benchmark.

---

## BesTLA WIRED IN: 2089.8 PP achieved, but OUTPUT IS WRONG. Off by default.

### The speed is real

`GRIMOIRE_BESTLA_CKPT=/models/Qwen3.8-27B-int4-AutoRound`, M-dispatch at 64:

| | PP @4096 | TG |
|---|---|---|
| BesTLA ON | **2089.8 tok/s** | 23.2 |
| baseline (off) | 1685.0 | 23.1 |

Dense FFN region 1382.6 -> 1129.7 ms. **The 2000 PP gate is reachable and the
integration works end to end** — 64 layers pack in ~11 s, all three FFN GEMMs run
through BesTLA on GRIMOIRE's own queue.

### But the FFN computes the wrong thing

Same 143-token prompt, `-n 32`:

- **BesTLA off:** *"You have identified the core engineering bottleneck that defines
  the current state of running Large Language Models on consumer hardware..."* — correct.
- **BesTLA on:** *"Wait, the user's message appears to be a corrupted or garbled
  text..."* — the model is reading garbage.

**The 2089.8 number is therefore not yet a real result.** BesTLA is **opt-in** and
**off** by default; the baseline is verified intact at 1685.0 / 23.1 with correct output.

### Ruled out

- **Result dtype/layout** — instrumented: `sizes=[143,17408] dtype=BFloat16 contig=1`,
  exactly what GRIMOIRE's buffers expect. A conversion guard was added anyway.
- **Architecture mismatch** — both checkpoints report `num_hidden_layers 64`,
  `hidden_size 5120`, `intermediate_size 17408`, `vocab_size 248320`, same class.
- **Symmetric zero-point offset** — `GRIMOIRE_BESTLA_ZP` tested at 8 and 0; output is
  **byte-identical garbage** either way, which suggests the packed weight is not
  responding to nibble values as expected.
- **gate/up ordering and buffer layout** — gate -> `grouped_out[0..M*inter]`,
  up -> `+M*inter`, split-SwiGLU over the two, down -> f32. Shapes all check out.

### Most likely remaining cause, and the fix

Borrowing weights from a *different* checkpoint was the shortcut, and it is the part
that is failing. The int4-AutoRound file may be a different revision of the base model
than the BF16 source GRIMOIRE's MXFP4 artifact was built from, or its packed layout
differs from the plain GPTQ nibble order assumed here.

**The correct route is the one originally specified: emit int4 g128 symmetric from
GRIMOIRE's OWN BF16 source in `b70_compile_model.cpp`.** Then the weights are
guaranteed to correspond, there is no second checkpoint to mismatch, and the packing
is under our control. That is converter work, not integration work — the integration
is now proven.

### Also fixed along the way

- `build_b70.sh`: `-fsycl-device-code-split=per_kernel` -> `off`. With per-kernel
  images a single kernel that fails AOT is silently dropped, and the generation path
  died with `No kernel named ...launch_embed_batched... was found` while the benchmark
  path (which never calls it) ran fine.
- `tools/bare.sh` was missing `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`; without it the
  runtime selects a device with no matching AOT image and reports the same
  "no kernel named" error. **Use `tools/tune.sh`.**
- The 11-kernel 2-D range conversion was reverted (it was worth +3 tok/s and is not
  worth the risk); only the split-SwiGLU that BesTLA needs was kept.

---

## !! RETRACTION: the entire BesTLA speed result was measured on a kernel producing ZEROS

### What happened

Every BesTLA number in this document above — 4.976 ms, 146.7 TFLOP/s, "1.47x faster",
143 TFLOP/s across 64 layers, and the 2089.8 tok/s end-to-end — was obtained with
`scale_type="bf16"` and bf16 scales. **In that configuration `woqgemm_linear` returns
an all-zero tensor.** Verified directly against a reference matmul:

```
ref |mean|=0.5784
  ZP=8 w.  s.  -> cosine=+0.0000  |out|=0.0000     (all 8 layout permutations)
```

It was still "fast" because it ran a kernel; it simply did not compute the product.
That is why the zero-point made no difference, and why the model emitted garbage.

### The correct call

Scales must be **fp16** with `scale_type="fp16"` (the wrapper converts activations to
fp16 on XPU: `target_dtype = torch.float16 if A.device.type == "xpu"`):

| compute | scale_type | scale dtype | cosine | |out| |
|---|---|---|---|---|
| bf16 | **fp16** | **fp16** | **+1.0000** | 0.5784 = ref |
| bf16 | bf16 | bf16 | +0.9890 | 155.6 (269x too large) |

With that fix the model produces correct, on-topic text.

### And with correct math BesTLA is SLOWER than GRIMOIRE

| Qwen PP @4096 | tok/s |
|---|---|
| **GRIMOIRE baseline (MXFP4)** | **1685.0** |
| BesTLA, correct fp16 scales | **1390.4** |
| BesTLA, broken bf16 scales (zeros) | 2089.8 — meaningless |

**BesTLA is now OFF by default and should stay off.** Baseline verified intact:
1685.0 PP / 23.1 TG with correct output.

### Why it is slower, and the only remaining salvage

`auto_round_kernel/__init__.py:woqgemm_linear` does
`A_2d = A.to(target_dtype)` on entry and `out.to(raw_input_dtype)` on exit — a bf16
to fp16 conversion of the activations and a conversion back, **on every call**. That is
6 conversions per layer over 64 layers, on tensors of M x 5120 and M x 17408.

A salvage would be to call the lower-level `ark.woqgemm` directly with activations
already in fp16, keeping the whole FFN in fp16 so no conversion is needed. Whether the
underlying kernel then beats GRIMOIRE's cutlass path is **unmeasured** — and given that
every previous estimate here proved wrong, it should be measured against a
cosine-verified reference before any further integration work.

### Lesson

**A speed measurement of a kernel whose output was never checked is not a measurement.**
The correctness check that took ten minutes at the end would have invalidated the entire
line of work at the start. Verify numerics first, then time it.

---

## FINAL: BesTLA kernel is faster, but not reachable without losing the gain

### The kernel, measured correctly (cosine +1.0000 vs a reference matmul)

vLLM's exact XPU recipe (`auto_round_kernel/qlinear.py:207-220`): `sdt="fp16"`,
`cdt="int8"` (W4A8), scales as **float16**, symmetric weights `-= 8` to int8, zeros an
**empty** int8 tensor.

| gate_proj 17408x5120, M=4096 | ms | TFLOP/s |
|---|---|---|
| **BesTLA cdt=int8** | **4.917** | **148.5** |
| BesTLA cdt=bf16 | 5.750 | 127.0 |
| GRIMOIRE (fused, 2x the width) | 14.63 | ~100 |

Two halves = 9.83 ms against GRIMOIRE's 14.63: **the kernel really is ~1.49x faster,
with verified-correct numerics.**

### But end-to-end it is slower, and worse the faster the kernel gets

| Qwen PP @4096 | tok/s |
|---|---|
| **GRIMOIRE baseline** | **1683.3** |
| BesTLA cdt=bf16 | 1390.4 |
| BesTLA cdt=int8 | 1161.6 |

`woqgemm_linear` does `A_2d = A.to(torch.float16)` on entry and `out.to(bf16)` on exit.
At M=4096 that is a 21M-element conversion per call, **192 calls per prefill**. The
faster the kernel, the more the conversions dominate — which is exactly the inversion
observed.

**There is no way around it from C++:** the extension exports only
`PyInit_auto_round_kernel_xpu` (every other symbol hidden), registers no torch op, and
the public BesTLA source has no XMX path. The wrapper is the only entry point, and the
wrapper is the bottleneck.

### Conclusion

**BesTLA is off by default and should stay off.** The 1.49x is real but unreachable
through the only available interface. Reaching it needs one of:
1. Intel exporting a C ABI / torch op for the ARK GEMM, or publishing the XMX source;
2. running GRIMOIRE's FFN activations in fp16 end-to-end so no conversion is needed
   (a large change with its own accuracy implications, and still Python-in-the-loop);
3. writing our own int4 g128 W4A8 XMX kernel — now with a firm, verified target of
   **148.5 TFLOP/s** and a known-correct reference to check against.

Option 3 is the honest path to Ian's stated goal of a pure C++/SYCL engine, and this
session has established exactly what it must beat and how to verify it.

### Verified final state

| | PP @4096 | TG |
|---|---|---|
| **Qwen** (MXFP4, BesTLA off) | **1683.3** | 23.1 |
| **Ornith** (MXFP4) | **9798.3** | ~111 |

Both produce correct, coherent output on real prompts.

---

## THE RECIPE, AND WHAT IT COSTS TO IMPLEMENT IN PURE C++/SYCL

### vLLM's exact configuration (auto_round_kernel/qlinear.py:207-220)

```
sdt = "fp16"                    # scale dtype -- fp16, NOT bf16
cdt = "int8"                    # compute dtype -- W4A8, int8 activations
wdt = BITS_DTYPE_MAPPING[4]     # "int4"
scales -> torch.float16
symmetric: intweight -= 8  ->  int8   (signed -8..7)
zeros = torch.empty(0, dtype=int8)    # EMPTY, not None
```

Verified against a reference matmul: **cosine +1.0000**. With `scale_type="bf16"` the
kernel returns **all zeros** (cosine 0.0000) — that is what invalidated every earlier
measurement in this document.

Kernel speed with the correct recipe, gate_proj 17408x5120 @ M=4096:

| | ms | TFLOP/s |
|---|---|---|
| BesTLA cdt=int8 | **4.917** | **148.5** |
| BesTLA cdt=bf16 | 5.750 | 127.0 |
| GRIMOIRE cutlass MXFP4 (fused, 2x width) | 14.63 | ~100 |

### Why calling their library cannot work

| Qwen PP @4096 | tok/s |
|---|---|
| GRIMOIRE baseline | **1683.3** |
| BesTLA cdt=bf16 | 1390.4 |
| BesTLA cdt=int8 | 1161.6 |
| BesTLA cdt=int8, result copy removed | 1188.8 |

`woqgemm_linear` converts activations bf16->fp16 on entry and the result back on exit.
Removing GRIMOIRE's own copy of the result (27 GB/prefill) changed nothing, so the
wrapper's own conversions dominate. The extension exports only
`PyInit_auto_round_kernel_xpu`, registers no torch op, and the public BesTLA source has
no XMX path — so the wrapper is the only entry point.

### GRIMOIRE already has a W4A8 path — and it is 34x too slow

`src/gemm_xmx.cpp` contains the complete pipeline, **and nothing in the engine calls it**:
- `launch_quantize_rows_int8` (line 445) -- bf16/f32 -> int8 with per-row scales
- `gemm_int<Fmt::INT4>` (line 232) -- int4 weights x int8 activations
- `launch_gemm_xmx_int` (line 475) -- dispatch

Measured (`tools/bench_w4a8.cpp`, M=4096):

| shape | GRIMOIRE W4A8 | TFLOP/s | vs its own MXFP4 |
|---|---|---|---|
| ffn-gate-up 34816x5120 | 330.003 ms | **4.4** | 0.04x |
| ffn-down 5120x17408 | 172.380 ms | **4.2** | 0.04x |

It is a reference implementation (SLM staging, scalar dequant), not a DPAS/XMX kernel.
That is why it was never wired in.

### So the work is: write the kernel

Not integration — a real int8-XMX GEMM with:
- `XE_DPAS` int8 accumulate-to-int32 (the B70 runs int8 XMX at ~2x bf16),
- int4 -> int8 dequant in registers, per-group-of-128 fp16 scales,
- 128x256 workgroup tile, 4x8 subgroup layout (the shape already proven optimal here),
- final scale by `act_scale[row] * weight_scale[group]`.

**Target: 148.5 TFLOP/s, verified reachable on this exact hardware and these exact
shapes. Correctness reference available: cosine must be 1.0000 against the fp32
dequantised matmul in `tools/verify3.py`.**

That is a bounded, well-specified kernel task with a known target and a known test —
which is a materially better position than this document started from.

---

## W4A8 kernel work: 4.4 -> 5.5 TFLOP/s (+25%), and what actually limits it

`gemm_int<Fmt::INT4>` in `src/gemm_xmx.cpp`. Two changes, both kept:

1. **Accumulate across the quantization group.** The int32 accumulator was reset and
   spilled through SLM every KT=64 columns, but the INT4 scale is constant across a
   128-column group — so half the flushes were waste, each costing a
   `joint_matrix_store` plus two barriers per N block. Now it accumulates across the
   whole group and dequantizes once. **4.4 -> 4.8 TFLOP/s.**
2. **Register blocking (`M_PER_SG_INT = 2`).** Each sub-group now owns two TM-row
   blocks, doubling the work-group M tile 64 -> 128. This halves how many times the
   int4 B tile is unpacked into SLM. **4.8 -> 5.5 TFLOP/s.**

Sweep of the blocking factor (gate-up 34816x5120, M=4096):

| M_PER_SG_INT | ms | TFLOP/s |
|---|---|---|
| 1 (original) | 330.0 | 4.4 |
| **2** | **265.9** | **5.5** |
| 4 | 413.4 | 3.5 (register spill) |
| 8 | 604.9 | 2.4 (heavy spill) |

Also tried and reverted: `WG_K_INT8` 64 -> 128 (3.2 TFLOP/s — the extra SLM costs more
occupancy than the saved flushes gain).

### What still limits it: the B dequant, not the DPAS

Per K tile the kernel unpacks `WG_N * KT` int4 weights into SLM, calling
`unpack_int4_raw` and `zero_for(n, k)` **per element**. With `WG_M_INT=128` that whole
tile is re-unpacked for each of the 32 M-tile rows: on gate-up that is
`34816 x 5120 x 32` ~= **5.7e9 unpacks per GEMM**, against a DPAS budget that is orders
of magnitude smaller. The kernel is a dequant loop with a matrix multiply attached.

Larger M tiles cannot fix it — 4 and 8 spill. **The structural fix is to stop staging B
through SLM per tile**: 2-D block loads straight from global memory with in-register
dequant and double buffering, which is exactly what the cutlass MXFP4 path does to
reach ~100 TFLOP/s. That is a rewrite of the loop, not a constant.

### Position

| | TFLOP/s |
|---|---|
| GRIMOIRE W4A8 (now) | **5.5** |
| GRIMOIRE MXFP4 cutlass | ~100 |
| BesTLA W4A8 (target) | **148.5** |

The W4A8 path must beat 100 before it is worth wiring in at all, and reach ~148 to
deliver 2000 PP. It is at 5.5. **Engine untouched and verified: Qwen 1684.2 PP.**

---

## START HERE TOMORROW: cutlass has a NATIVE int8 x int4 DPAS

`cutlass-sycl-src/include/cute/arch/mma_xe.hpp` declares, among others:

```
CUTE_DECLARE_XE_DPAS_TT(d,  s8,  s8,  d)   // int32 = int8 x int8
CUTE_DECLARE_XE_DPAS_TT(d,  s8,  s4,  d)   // int32 = int8 x int4   <-- W4A8, NATIVE
CUTE_DECLARE_XE_DPAS_TT(d,  u8,  s4,  d)
CUTE_DECLARE_XE_DPAS_TT(d,  s4,  s8,  d)
```
with `dpas_type::s4 = int4_t`, `s8 = int8_t`, `d = int32_t`.

**The B70's DPAS takes 4-bit weight operands directly.** No unpack to int8 is needed —
and that unpack is exactly what pins `gemm_int<Fmt::INT4>` at 5.5 TFLOP/s (~5.7e9
per-element `unpack_int4_raw` + `zero_for()` calls per GEMM, staged through SLM).

### The change

`src/xe2_grouped_raw_launcher.hpp` currently has:
```cpp
using Op  = XE_DPAS_TT<8, float, A>;              // A = cutlass::bfloat16_t
using MMA = TiledMMAHelper<MMA_Atom<Op>, Layout<Policy::WGTile>, Policy::SGLayout>::TiledMMA;
```
Swap the atom to `XE_DPAS_TT<8, int32_t, int8_t, int4_t, int32_t>`, feed int8
activations, and keep **everything else** — the 128x256 tile, the 4x8 sub-group layout,
the 2-D block loads, the prefetch, the L2 swizzle. All of that already delivers
~100 TFLOP/s with bf16 activations and 4-bit weights; int8 XMX runs at ~2x the bf16
rate, which is the path to ~148.

### Order of work

1. Instantiate the s8xs4 atom in a new `launch_dense_w4a8<Policy>` beside
   `launch_dense_mxfp4` / `launch_dense_int4`. Weights stay packed int4 (no unpack);
   activations come from the existing `launch_quantize_rows_int8`.
2. Epilogue: scale by `act_scale[row] * weight_scale[n, k/128]`. The MXFP4 path already
   applies a per-group weight scale; the per-row activation scale is the new part.
3. **Verify before timing** — `tools/verify3.py`, cosine must be 1.0000 against the
   dequantised fp32 reference. This is the check that would have caught the all-zeros
   BesTLA result on day one.
4. Benchmark with `tools/bench_stream.cpp` (64 distinct matrices = production memory
   conditions). Gate: must beat 106 TFLOP/s (our int4 cutlass path). Target 148.5.
5. Wire into `mlp_bf16` behind the existing `M >= 64` dispatch; decode stays on the
   MXFP4 GEMV, which is 4x faster than any of this at M=1.

### Why this and not the hand-written kernel

`gemm_int<Fmt::INT4>` needs its whole memory pipeline rebuilt to compete — 2-D block
loads, in-register dequant, double buffering. The cutlass template already has all of
it, tuned, and the only thing it was missing turns out not to be missing at all: the
int4 DPAS atom is right there.

**Risk to check first:** whether `TiledMMAHelper` and the `xe_gemm_4bits` epilogue
accept a mixed s8xs4 atom cleanly, and whether the B fragment layout for `s4` matches
the packed layout our artifacts already use. That is the first hour's work and it
decides the approach.

---

## TG: the decode target, measured — 13.9 ms available, and the stack is behind

### Where the 43.09 ms/token goes (GRIMOIRE_TIMELINE=1)

Per-region weight traffic against the B70's measured 602 GB/s roofline:

| region | MB | us | GB/s | % roof | ms if at 90% |
|---|---|---|---|---|---|
| **ffn gate_up** | 89.1 | 241.4 | 369 | 61% | **4.92** |
| **la_qkv** | 26.2 | 102.5 | 256 | **42%** | **2.60** |
| ffn down | 44.6 | 114.8 | 388 | 64% | 2.08 |
| **ab gemv + gates** | 0.2 | 32.9 | **7.5** | **1%** | **1.56** |
| out gemv | 15.7 | 43.6 | 360 | 60% | 0.70 |
| q gemv + split | 31.5 | 102.0 | 309 | 51% | 0.70 |
| k+v gemv | 5.2 | 45.0 | 117 | 19% | 0.57 |
| z gemv | 15.7 | 40.4 | 389 | 65% | 0.55 |

**Total 13.9 ms -> 29.2 ms/token -> 34.2 tok/s.** Ian's contact reports 31.5, i.e. ~85%
of roofline, so the target is real and slightly conservative.

**Two distinct problems, not one:**
- `ab gemv` moves 0.25 MB in 32.9 us — 1% of roofline. That is launch/occupancy
  latency, not bandwidth. This is the part where **raw Level Zero command lists** help.
- `ffn gate_up` (89 MB) and `la_qkv` (26 MB) at 61% and 42% are bandwidth-efficiency
  problems. Launch overhead is irrelevant at that size; they need kernel work.

### The existing tuning knobs are exhausted

Swept `B70_EPL` / `B70_UNROLL` / `B70_WIDE` on the full model:

| EPL | UNROLL | WIDE | ms/token |
|---|---|---|---|
| **default (16, per-format)** | | | **43.355** |
| 16 | 2 | - | 43.530 |
| 16 | 4 | - | 43.567 |
| 16 | 2 | 1 | 44.039 |
| 32 | 2 | - | 60.644 |
| 32 | 4 | - | 57.968 |
| 64 | 2 | - | 60.705 |

Defaults win; every override is neutral or much worse. Consistent with the earlier
wide/split-K autotune finding. **61% -> 90% is a kernel change, not a parameter.**

### The stack on this box is behind

| | Tower | Ian's contact |
|---|---|---|
| vLLM | 0.26.1rc1.dev799 | **0.27.1** |
| PyTorch | 2.13.0+xpu | **2.14** |
| xpu-kernels | (bundled) | **0.1.13.2** |

He measures **2100 PP / 31.5 TG**; vLLM *on this box* measured **2015 / 27.95**. Same
hardware, so part of his lead is the newer stack — and the newer `vllm-xpu-kernels` is
where the faster kernels live. **Upgrading and re-measuring vLLM here is cheap and
tells us how much is stack versus kernel.** It also gives a newer, better reference to
study.

Model he used: `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` (W4A16 GPTQ, which may map
onto GRIMOIRE's INT4 path more directly than MXFP4 does).

### His recommended order of work

1. **Level Zero for TG** — cuts launch latency; the fix for `ab gemv`-class shapes.
2. **sycl-tla for prefill** — the current name for cutlass-sycl, which GRIMOIRE already
   builds against. See the s8xs4 native DPAS atom note above; that is the concrete lever.
3. **MTP** for generation — `Qwen3.8-27B-GPTQ-Int4-MTP-BF16` and the DSpark/DFlash
   drafters are already on the Tower.
4. **Prefix caching** last.

---

## REFRAME: GRIMOIRE's PP already beats the reference baseline. TG is a SPEC-DECODE problem.

Ian's contact's full results (py3.14, torch 2.14.0, kernels built from source):

| spec | pp4096 | tg256 |
|---|---|---|
| **none (baseline)** | **1973** | **31.50** |
| dspark, 4 draft tok | 2084 | 43.76 +/- 4.18 |
| dspark, 7 draft tok | 2025 | **50.19 +/- 5.17** |
| mtp, 1 draft tok | 1931 | 42.56 +/- 0.30 |
| mtp, 2 draft tok | 1918 | 45.23 +/- 2.68 |

### What this changes

**1. The "2100 PP" was speculative decoding, not the baseline.** His no-spec PP is
**1973**. vLLM measured on this Tower is **2015.6** on `Qwen3.8-27B-int4-AutoRound` —
**already ahead of his baseline**, on an older torch. So the torch 2.14 question is
largely moot for prefill, and chasing torch/vLLM versions for PP is not worth further
time.

**2. TG 31.5 -> 50 comes from speculative decoding, not from kernel bandwidth.**
The GEMV bandwidth analysis (13.9 ms available, 61% of roofline) is still valid and
still worth ~34 tok/s eventually, but it is the *hard* path. Spec decode gets 1.6x on
top of whatever the base decode rate is, and it is an architectural feature rather than
a kernel rewrite.

**3. GRIMOIRE has no speculative decoding at all.** Its base decode is 23.1 tok/s.
With DSpark-class acceptance (his 31.5 -> 50.19 = 1.59x) that would be ~37 tok/s,
comfortably past the 28 gate, without touching a single GEMV.

### Assets already on the Tower

- `Qwen3.8-27B-DSpark` and `qwen38-dflash-drafter-fp8-b70` — bf16 `Qwen3DSparkModel`
  drafters purpose-built for this model: 5 layers, block_size 7, tapping target layers
  [4,16,28,40,52]. Being bf16 they satisfy the quantize-from-BF16-ourselves rule.
- `Qwen3.8-27B-GPTQ-Int4-MTP-BF16` — MTP head, but inside a GPTQ checkpoint.

His numbers say **DSpark at 7 draft tokens is the best TG configuration** (50.19), with
MTP at 2 draft tokens second (45.23). DSpark also costs the least PP (2025 vs 1918).

### Revised priority for GRIMOIRE

1. **Speculative decoding (DSpark, 7 draft tokens)** — the single biggest TG lever,
   ~1.6x, and the drafter is already on disk.
2. GEMV bandwidth work (61% -> 90% of roofline, ~13.9 ms) — compounds with spec decode.
3. PP: **deprioritise**. 1684 vs a same-box vLLM 2015 is a real gap, but the reference
   baseline is 1973 and the remaining work there is a kernel rewrite with the
   s8xs4 DPAS atom. It is the expensive path and no longer the one blocking a gate.

### Version findings (settled, no further work needed)

- torch **2.14 is not obtainable**: pruned from nightly (now `2.15.0.dev2026082x`),
  and stable XPU is `2.13.0+xpu`. Installing `2.15.0.dev` works and vLLM still imports,
  but it breaks vLLM's pins (`torch==2.13.0`, `triton==3.7.2+xpu`).
- The Tower's `vllm-xpu-kernels 0.1.14.dev0` is **Ian's own custom build**, not an
  upstream release; his `0.1.13.2` is the official wheel.
- `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` is config-identical to the Tower's
  existing `/models/Qwen3.8-27B-W4A16` (gptq, 4-bit, **group_size 64**, sym,
  autoround 0.15.0). **No need to download it.** Measured here: 1714 PP / 30.4 TG.

---

## DECODE, STEP BY STEP: GRIMOIRE vs vLLM, same box, same day

**vLLM: 37.10 ms/token = 27.0 tok/s** (48 tokens, int4-AutoRound, fp8 KV, measured
directly; matches its 27.95 via llama-benchy). **GRIMOIRE: 43.09 ms/token = 23.2 tok/s.**
**Gap 5.99 ms/token, 16%.**

### vLLM decode composition (share of device time; absolute us inflated by the profiler)

| op | % |
|---|---|
| `vllm::inc_ark_woq_linear` | 35.9 |
| `bestla::sycl_prologue_b::WeightS4T::dequantS8` | **33.1** |
| `gemm_kernel` / `aten::mm` | 12.2 |
| `aten::copy_` | 3.1 |
| `bestla::...IKblockGemmDQCore` | 2.0 |
| `_xpu_C::gdn_attention` | 2.0 |
| elementwise / norms | ~4 |
| `_vllm_fa2_C::varlen_fwd` | 0.4 |

**Linear layers + their dequant ~= 71%.**

### GRIMOIRE decode composition (GRIMOIRE_TIMELINE=1)

| bucket | ms/token | % |
|---|---|---|
| FFN (gate_up + swiglu + down, 64 layers) | 22.9 | 53 |
| linear-attn projections (qkv/ab/z/out, 48) | 10.5 | 24 |
| full attention (16) | 4.5 | 10 |
| GDN core (conv/l2norm/deltanet, 48) | 2.2 | 5 |
| norms | 1.4 | 3 |
| lm_head | 1.5 | 3 |

**Linear layers ~= 77%.**

### What the step-by-step says

**No single divergent step.** Both engines are the same shape — roughly three quarters
of decode is weight-loading linear layers. There is no op vLLM runs that GRIMOIRE skips,
and none missing on either side. The 16% is spread across the linears.

vLLM in fact does *more* work in principle: **33% of its decode is a separate
`dequantS8` pass** that materialises weights into a buffer before the GEMM, where
GRIMOIRE dequantizes inside the GEMV. It still comes out ahead — which points at raw
throughput per weight byte, not at algorithmic structure or a missing stage.

### Per-region bandwidth (GRIMOIRE, vs the 602 GB/s roofline)

| region | GB/s | % roof | ms recoverable at 90% |
|---|---|---|---|
| ffn gate_up | 369 | 61% | 4.92 |
| la_qkv | 256 | 42% | 2.60 |
| ffn down | 388 | 64% | 2.08 |
| ab gemv + gates | 7.5 | **1%** | 1.56 |
| out gemv | 360 | 60% | 0.70 |
| q gemv + split | 309 | 51% | 0.70 |
| k+v gemv | 117 | 19% | 0.57 |
| z gemv | 389 | 65% | 0.55 |

**13.9 ms total -> 29.2 ms/token -> 34.2 tok/s.** `ab gemv` (0.25 MB in 32.9 us) is
latency-bound, not bandwidth-bound — that is the one place raw Level Zero / fewer
launches would pay. The rest is kernel throughput.

### Tuning state: every knob measured and already optimal

| knob | best | alternatives tried |
|---|---|---|
| `B70_EPL` | 16 (default) | 32 -> 60.6 ms, 64 -> 60.7 ms |
| `B70_UNROLL` | per-format default | 2 -> 43.53, 4 -> 43.57 |
| `B70_WIDE` | default | forced -> 44.04 |
| `ROWS_PER_SG` | **4** | 8 -> 50.8 ms, 16 -> 74.1 ms |

**Closing the 16% needs a kernel change, not a parameter.**

### Traps for the next session

- **Do NOT tune against `tools/bench_gemv2.cpp`.** It runs, but reads 133 GB/s on
  gate-up where production reads 369 — it is not representative. Use end-to-end decode
  ms/token from a real model run.
- Changing a constant in `kernels.hpp` requires rebuilding **`bin/grimoire`**, not just
  a bench tool. A stale binary produced a 13.5 tok/s "result" that was simply the
  `ROWS_PER_SG=16` build still in place.
- Current committed constants: `ROWS_PER_SG = 4`, `M_PER_SG_INT = 2`.
