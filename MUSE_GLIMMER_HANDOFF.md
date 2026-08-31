# Muse Glimmer / DFlash handoff

Branch: `pp-muse`. Speculative decoding WORKS as of 2026-08-31.

## PREFILL: the GEMMs are NOT the gap (settled 2026-08-31, do not re-derive)

Ran the exact five Muse projection shapes at M=3672 through Fusion's own op
(torch.ops._xpu_C.int4_gemm_w4a16) and compared against Grimoire's split
profile, summed over 52 layers:

    Grimoire pure oneDNN GEMM : 1345.5 ms
    Fusion   pure oneDNN GEMM : 1309.4 ms   <- within 3%

Both are the same dnnl::matmul with the same configuration, verified at source
level: vLLM's trans_type_t::nt gives wei_strides = {1, ldb} with
ldb = mat2.strides()[last]*8 = K, identical to Grimoire's
memory::desc wei({k,n}, u4, dims{1,k}); same scales mask 3 with {group_size,1},
same s8 zero-points mask 0, same fpmath_mode::f16(true), same
scratchpad_mode::user. See vllm-xpu-kernels csrc/xpu/onednn/int4_gemm_w4a16.h
and onednn_ext.h.

**Do not spend time tuning the GEMMs or the oneDNN descriptors.** The gap is:

    Fusion  : 1702 total - 1309 GEMM =  393 ms non-GEMM
    Grimoire: 2412 total - 1345 GEMM = 1067 ms non-GEMM   <- ~670 ms excess

Grimoire dispatches ~11 kernels per layer x 52 layers with no overlap:
in_norm, qkv GEMM, qkv-norm-rope, o_gate GEMM, kv-append, attention, attn copy,
gate-sigmoid-mul, o_proj GEMM, post_norm, add, pre_ff_norm, gate_up GEMM,
swiglu, down GEMM, post_ff_norm, add. vLLM fuses most of that elementwise work
into its compiled graph. **The remaining lever is kernel fusion and dispatch
count.** Candidates: merge post_norm+add into one pass, fold swiglu into the
gate_up epilogue, fold gate-sigmoid-mul into the attention epilogue.

Prefill progress this session, all correctness-verified (output stays
byte-identical to Fusion's " to=selfWrite a haiku about"):

| build | prefill 3672 tok | tok/s | vs Fusion |
|---|---|---|---|
| session start | 2676 ms | 1372 | 65% |
| FP16 activations (a3ac518) | 2470 ms | 1486 | 70% |
| GEMM direct-to-dst (d13d8d7) | 2412 ms | 1523 | 72% |

Fusion reference: 1702.5 ms / 2129.3 tok/s, measured against the live server on
GPU1 with a UNIQUE PREFIX PER REQUEST. Without cache-busting the server returns
a cached prefill and reports ~28,000 tok/s, which is meaningless. That number
matches llama-benchy pp4096 (2144 +/- 28), so the method is sound.

## Current measured state (llama-benchy-comparable, 3 runs each)

| test | Fusion (llama-benchy) | Grimoire | ratio |
|---|---|---|---|
| tg32  | 44.23 ± 7.57 (peak 48.97) | 83.5 ± 7.7 (peak 92.4) | **1.9x faster** |
| tg128 | 44.49 ± 4.69 (peak 54.00) | 51.5 avg (49.0/54.6/51.0) | on par |
| pp4096 | 2144-2195 | 1384.5 @3672 tok | ~65% |

Was 0/8 accepts, 1.00 committed/step, 16.8 tok/s at session start.
Verifier now reproduces Fusion's token stream exactly: verify@64 emits
19669 200023 10064 262 ... against Fusion's OUTPUT TOKENS
328 19669 200023 10064 262 2818 17001 1509.

**Grimoire's decode is now AHEAD of Fusion; prefill is the real gap (~65%),
and it is a GEMM-kernel-throughput gap, not a correctness bug.** Do not
re-litigate decode without a llama-benchy-comparable measurement (matched
prompt length AND repeat count) - see "Retracted" below for how this went
wrong once already this session.

## Fixed this session (all confirmed by measurement)

1. `linear_out_f16` undersized in the drafter (bd2422f). The shared lm_head
   wrote 15*202048 halves into a 16*39936 buffer. Draft logits were inf/nan on
   10 of 15 rows.
2. `page_stride_elements` passed in ELEMENTS, not sequence positions
   (0f93472). vLLM computes it as key_cache.stride(0)/key_cache.stride(1) =
   block_size. Grimoire passed block_size*kv_heads*head_dim, inflating every
   page address by 1024x, so page 0 read correctly and every later page
   returned zeros. This broke the TARGET as well as the drafter: with it, the
   target generated garbage past position 64; without the drafter loaded the
   target took the sequential path and looked fine, which masked it.
3. `VW` / `W` undersized in the verify path (87c673e). Same class as (1):
   prefill_muse projects the shared lm_head through xb/yb, whose widths
   enumerate only the layer projections. 16*202048 halves into 16*39936 - a
   5.06x overrun smashing ~5.2 MB of device memory every verify step. THIS was
   the 0/8 cause.
4. Stale `cu_q` (prefill_muse left {0,64}, draft needs {0,M}) plus four memcpys
   sourcing host stack values into async device copies. Real UB, verified NOT
   causal for 0/8. Kept as hygiene.

## Profiling notes

- `GRIMOIRE_MUSE_TIME_LAYER=all` (and `=<n>`) profiles prefill_muse per stage.
  It drains the queue at each mark, so it INFLATES small frequently-called
  stages: it attributed 546 ms to the fp32<->fp16 conversions but removing them
  recovered only ~230 ms. Use it for RANKING stages, not absolute cost.
- Any Grimoire Muse run WITHOUT GRIMOIRE_DFLASH_MODEL silently takes the
  sequential prefill fallback (exact_attn is false without the dflash2 buffers)
  and is 10x slower. It is not representative of anything. Watch for
  "batched prefill unavailable, using sequential fallback".
- Never time Grimoire while another container holds GPU0. Fusion on GPU1 is
  fine and does not contend.

## Retracted this session - read before re-deriving

- **"Acceptance collapses with context length" / sliding-window theory: WRONG,
  retracted.** An early measurement (random-word prompts) showed 82% accepts
  at 64 tokens vs 28% at 3672 and pointed at the draft window (2047) binding
  above ~2048 tokens. A coherent-prose control at the same lengths killed it:
  59% @2662, 53% @3662, no cliff at 2048, and 3210-token random-word landed at
  44% (recovered, not worse than 2420's 25%). The random-word prompts were
  measuring prompt-entropy / drafter-predictability, not a Grimoire defect.
  There is no known long-context acceptance bug. Do not re-open this without a
  coherent-prompt sweep.
- **"Grimoire is at ~40% of vLLM decode": WRONG, retracted.** That number
  compared a random-word-prompt speculative run against vLLM's short-context
  tg32 figure - not the same workload. The correct comparison (tg32-vs-tg32,
  3 repeats each) has Grimoire at 1.9x Fusion. See "Current measured state".
- Do not trust a decode/prefill number produced while more than one container
  holds the GPU. Confirmed this session: three concurrent runs on the same
  B70 produced garbage timings (a build even crashed under contention). Always
  `docker ps` and confirm only Fusion (GPU1) is running before timing Grimoire
  runs on GPU0.

## Known-open

- `k_norm` bug-compatibility is in but NOT A/B tested. vLLM stacks
  `_k_norm_weights` as [num_layers, head_dim] and hands it to ops.rms_norm,
  whose weight must be [head_dim], so the kernel applies layer 0's weight to
  all five layers despite its comment claiming per-layer selection. Verified by
  recovering the effective weight from Fusion's own pre-RoPE context K:
  identical across layers (pairwise cos 1.0000, rms 1.08547) and equal to
  layers.0.self_attn.k_norm.weight, while the checkpoint's five k_norm tensors
  genuinely differ (rms 1.085, 1.349, 0.895, 1.381, 0.955). Revert and re-measure
  now that the verifier is correct; it may be unnecessary.
- Target aux states still differ from Fusion at rel_l2 0.032 (cos 0.99947).
  That is INT4 target numerics; may bound achievable acceptance somewhat but
  acceptance is already comparable to Fusion so this is low priority.
- **Prefill is the real remaining gap: ~65% of Fusion (1384.5 vs 2144-2195
  tok/s at ~3700-4096 tokens).** This is a GEMM-kernel-throughput question
  (CUTLASS/oneDNN vs Grimoire's hand-written kernels), consistent with prior
  project findings that the PP gap is "kernel+format, not tuning". This is
  where effort should go next, NOT decode.
- llama-benchy (`--pp 4096`) cannot drive Grimoire: it needs an
  OpenAI-compatible HTTP endpoint and Grimoire is CLI-only. A single-stream
  /v1/chat/completions wrapper around grimoire_generate would make Grimoire and
  Fusion benchmarkable with the identical tool and remove the need for manual
  CLI-vs-llama-benchy comparisons like the ones in this handoff.

## Lesson

All four bugs fixed this session were width/stride errors that only manifest
past a boundary - page 0, or a buffer sized for layer widths rather than
vocab. A 64-token prompt hid every one of them. Test at the size you
benchmark at - and when comparing to a benchmark tool's numbers, match its
exact test (prompt length, repeat count), not an approximation of it.

## Tooling built this session

- `GRIMOIRE_DFLASH_DUMP=<dir>` dumps the first DFlash cycle's tensors as raw f32
  (`g_*.f32`) and prints the first two verify steps' candidate/verified vectors.
- `tools/fusion_dump.py` mirrors those dump points into Fusion (`f_*.f32`),
  plus slot mappings and per-layer context K/V. Needs `ENFORCE_EAGER=1` only for
  in-model tensor capture; everything else runs graph-enabled.
- `tools/compare_dumps.py` pairs `g_*`/`f_*` and reports the first divergent
  stage.
- `tools/fusion_verify_ref.py` runs Fusion's target over Grimoire's exact
  candidate sequence and reports its argmax per position - ground truth for
  `verified[]`.

## Runtime

- Tower `root@192.168.8.225`. Fusion on GPU1 = 35:00.0 = renderD130, port 3559.
- Grimoire on the free B70 = 03:00.0. **Render node numbers change on reboot** -
  resolve by PCI via `/dev/dri/by-path`. A run on the iGPU loads fine then dies
  with "No kernel named ... was found"; check the device line says 256 EUs /
  31.9 GiB.
- Never use `enforce_eager` for performance numbers.

## Build

```bash
docker run --rm --entrypoint bash \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire -w /grimoire \
  my-vllm-xpu:latest -lc 'bash build_b70.sh'
```

Attention bridge only (needs the kernel sources mounted at /src):

```bash
docker run --rm --entrypoint bash -e GRIMOIRE_BRIDGE_ONLY=attention \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire \
  -v /mnt/user/appdata/vllm-xpu-kernels:/src \
  -w /grimoire my-vllm-xpu:latest -lc 'bash tools/build_bridges_b70.sh /grimoire'
```

Keep image `my-vllm-xpu:latest`.
