# Decode is attention-bound — handoff 2026-09-04

## The one thing that matters

GRIMOIRE's problem at 4k context was never MTP, the draft head, or the model.
It is **single-token decode speed**, and it is **attention**.

Measured with `GRIMOIRE_TIMELINE=1`, one request, prompt=4778, per token:

```
                        BEFORE          AFTER (committed)
full attention layer    3,650 us   →    1,197 us     (x16 layers)
sum full               54,836 us   →   19,152 us     67% -> 44% of token
sum linear             21,832 us   →   21,820 us     unchanged
TOTAL per token        82,012 us   →   43,930 us     1.87x
base decode              11.74 t/s →     22.8 t/s
```

Root cause: `launch_flash_decode` split-K width was hardcoded to
`GRAPH_SPLITS = 8` and never scaled with context depth. At 4778 tokens each
split walked ~600 keys serially, achieving **4.6 GB/s against the measured
602 GB/s roofline**. The kernel's own comment describes exactly this
latency-bound failure; the constant defeated it.

## Benchmarks (llama-benchy 0.4.0, Sherlock corpus, pp4096/tg32, runs=3)

| build | pp4096 | tg32 |
|---|---:|---:|
| start of day (MXFP4, 4-bit MTP head) | 2484 | 15.8 |
| + BF16 MTP head | 2396 | 22.7 |
| + adaptive split-K (committed) | 2100-2500 | **~28** |

Reference on the same box, same benchy, same cell:

| engine | pp4096 | tg32 |
|---|---:|---:|
| vLLM XPU MTP k=4 (GPTQ-Int4-MTP-BF16) | 1666 | 44.2 / 46.1 |
| vLLM XPU MTP k=3 | 1691 | 46.6 |
| vLLM XPU DSpark k=7 | 1716 | 39.6 |

Published reference (MikeCaldera/intel-arc-pro-b70-qwen38-vllm) is
**84.65 tok/s but at p512/g128**, with two vLLM patches applied, metric
`(completion_tokens - 1) / (request_end - first_generated_token)`.
It is NOT comparable to pp4096/tg32. Their own no-MTP baseline is 32.75.

## MTP multiplier shrinks as base decode improves

```
base 11.74 -> MTP 22.72   = 1.94x
base 22.80 -> MTP 28.0    = 1.23x
```
Verify is a fixed 4-token forward pass (57-60 ms/round, unchanged by any
attention work so far). As base decode gets cheaper, speculation overhead
takes a larger share. **Do not assume 1.8-1.9x survives.** The base number
has to do the work.

## Committed changes

- `src/attention.cpp` — `decode_splits()`; split-K scales with `seq_len` at
  ~128 keys/split, floor `GRAPH_SPLITS`, cap `MAX_SPLITS`. Applied
  identically in `launch_flash_decode` and `launch_flash_merge` (a mismatch
  reads the wrong partial lanes).
- `src/kernels.hpp` — `MAX_SPLITS = 64`.
- `src/grimoire.cpp` — partials workspace sized `MAX_SPLITS`; opt-in
  `GRIMOIRE_DECODE_BATCHED_ATTN` path (see NEXT STEP 1 — currently WRONG).
- `tools/b70_compile_model.cpp` — `keeps_bf16()` now keeps all `mtp.*` at
  BF16.
- `src/grimoire.cpp` (loader) — MTP head format follows the checkpoint:
  RAW -> BF16, packed -> `PF`. Backward compatible with old artifacts.
- New model: `/models/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16` (15.87 GiB,
  866 tensors, `mtp RAW=15 MXFP4=0`, head 0.81 GiB vs 0.23 GiB).

## NEXT STEPS (in order)

### 1. Fix batched-decode masking — worth ~1.2x, it is a BUG not a design issue
`GRIMOIRE_DECODE_BATCHED_ATTN=1` routes single-token decode through
`launch_flash_decode_batched`. Geometry is 24 query heads over 4 KV heads
(`q_per_kv = 6`), so `launch_flash_decode` fetches **every KV byte 6 times**;
the batched kernel gives a workgroup one KV head and stages K/V in SLM.

MEASURED with the flag on: full attention 19,152 -> **12,836 us**, token
43,930 -> **37,154 us (26.9 t/s base)**.
**But output is garbage**: `'roppelnd\n== oppelnd (Deutsch) =='`.

NARROWED (evening of 2026-09-04) -- these are ruled OUT, do not re-test:

- **Not the split count.** Fails identically with
  `GRIMOIRE_DECODE_BATCHED_SPLITS=8` (the value the working verify path uses)
  and with the adaptive value.
- **Not the `base_seq_len` convention.** Fails with
  `GRIMOIRE_DECODE_BATCHED_DELTA=0` (`base_seq_len = pos+1`, live-decode
  semantics) and `-1` (`= pos`, the verify path's semantics).
- **Not the kernel itself.** It is CORRECT at short context: a 6-token prompt
  returns `" Paris."`. It is also correct at depth for M>1, which is what the
  MTP verify path exercises every round.

So the failure is specific to **M=1 at depth**. Static review found nothing:
K/V base pointers, strides, the `head = kvh*q_per_kv + q_in_kv` mapping, the
`s < s_end && s < row_seq` mask, the partials index
`(row*num_heads + head)*splits + part`, and the SLM staging loop all match
`launch_flash_decode` or are internally consistent. Note `rows_per_wg`
collapses to 1 at M=1, so `wg = 1*q_per_kv*SG_SIZE = 96` -- the one geometry
the verify path never produces.

NEXT ACTION: stop reading and diff numerically. Dump `s.attn_out` for a single
`(layer, pos)` from both kernels on the same ~2000-token prompt and compare
element-wise. `probe()` at `probe_layer` already exists at the decode site
(`probe("FA attn core", s.attn_out, qheads*cfg.head_dim)`). Suspect the M=1
workgroup geometry (wg=96) first, since that is the only untested shape.

Both override env vars (`GRIMOIRE_DECODE_BATCHED_SPLITS`,
`GRIMOIRE_DECODE_BATCHED_DELTA`) are committed and still in place for this.

### 2. Attention is still ~1197 us/layer against a ~16 us roofline
KV per layer at 4778 tokens is 4 kv_heads x 256 dim x 4778 x 1 byte x 2
(K+V) = 9.8 MB = 16 us at 602 GB/s. Remaining loss after the GQA fix is
cache-line efficiency: KV is fp8, and 16 lanes read 16 consecutive bytes of a
64-byte line, both for K (`kh[d*seq_cap + s]`) and V (`vrow[lane+d*SG_SIZE]`).

### 3. Explain the pp4096 drift before quoting any PP number
Identical binaries produced 1969, 2029, 2103, 2213, 2230, 2500 tok/s.
`throttle/status = 0`, `max_freq = 2800`. Cause unknown. Until this is
understood, PP deltas under ~20% are not measurable.

### 4. Only then revisit speculation
Acceptance with the BF16 head is 43-73% at short prompts, 21-48% at 4k --
already at or above vLLM's 27-50%. Acceptance is NOT the bottleneck now.

## THINGS ALREADY TRIED THAT ARE WORSE — do not repeat

- **Wider split-K** (`MAX_SPLITS=128`, 64 keys/split): TG 22.1 vs 28.7, and
  acceptance fell to 15-26%. More partials -> more merge rounding -> worse
  draft agreement.
- **Vectorized K loads** (4 keys/lane as one 32-bit load, to fill the cache
  line): TG 22.2. Correct (coherence passed) but slower.
- **Adaptive split-K on the verify path**: no gain, 60 -> 57 ms/round.
  Verify is weight-bound, not attention-bound. Reverted to `GRAPH_SPLITS`.
- **CUTLASS paged attention** (`grimoire_xe2_dflash_paged_f16`): NOT usable,
  it hard-requires `head_dim == 128` and this model is 256.

## Gotchas confirmed today

- `GRIMOIRE_MTP_EXACT_VERIFY` is **presence-only** (`getenv != nullptr`,
  `src/grimoire.cpp` ~5899). Setting it to `0` still enables it.
- `--proj mxfp4` on a GPTQ checkpoint re-quantizes all 64 layers at load and
  is very slow. Use `--proj mxfp4` only with the `.b70` artifact.
- The container runs the image binary, NOT `bin/grimoire-server` from the
  source mount. After patching, either rebuild the image
  (`tools/build_b70_native_image.sh`, ~12 min) or run the source binary via
  `my-vllm-xpu:latest` with `LD_LIBRARY_PATH` and bridges under
  `/grimoire/src` (as `tools/tune.sh` does).
- `decode_splits()` makes launch geometry depend on `seq_len`, which breaks
  the recorded-graph assumption noted at the decode site. `GRIMOIRE_DECODE_GRAPH`
  is off by default and already known-broken, so this is safe today.
- A B70 dropped off the PCI bus during a container swap and oopsed the `xe`
  driver (`cdclk_divider` divide-by-zero under `rpm_resume`, MMIO returning
  0xFFFFFFFF). Recovered only by reboot. Pin power to avoid the resume path:
  `echo on > /sys/bus/pci/devices/0000:{03,35}:00.0/power/control`.
