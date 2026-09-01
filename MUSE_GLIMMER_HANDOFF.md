# Muse Glimmer / DFlash handoff

Branch: `pp-muse`. Speculative decoding WORKS as of 2026-08-31.

## PREFILL: at parity with Fusion (settled 2026-09-01)

Measured with BOTH engines treated identically - one discarded call to build the
oneDNN plans for that shape, then time the second:

| tokens | Fusion | Grimoire warm |
|---|---|---|
| ~400  |  199.4 ms |  204 ms (2138 tok/s) |
| ~800  |  389.5 ms |  418 ms (2001 tok/s) |
| ~1650 |  771.4 ms |  808 ms (2058 tok/s) |
| ~3150 | 1446.9 ms | 1531 ms (2066 tok/s) |
| ~3650 | 1701.2 ms | 1782 ms (2061 tok/s) |

**Grimoire 2061 tok/s vs Fusion 2129 tok/s = 97%**, within ~4% of llama-benchy
pp4096 (2144 +/- 28). Run Grimoire with `GRIMOIRE_PREFILL_WARMUP=1` for any
comparison against a warm server.

Scaling fits:

    Fusion          0.4607 ms/token +  24.4 ms fixed
    Grimoire cold   0.486  ms/token + 620   ms fixed
    Grimoire warm   ~0.47  ms/token + ~15   ms fixed

**Per-token compute was always within 5% of Fusion.** The GEMMs match too:
1345.5 ms (Grimoire) vs 1309.4 ms (Fusion) over 52 layers at the same shapes,
with descriptors verified identical in vllm-xpu-kernels source
(csrc/xpu/onednn/int4_gemm_w4a16.h, onednn_ext.h): trans_type_t::nt gives
wei_strides {1, ldb} with ldb = strides[last]*8 = K, same as Grimoire's
memory::desc wei({k,n}, u4, dims{1,k}); same scales mask 3 with {group_size,1},
same s8 zero-points mask 0, same fpmath_mode::f16(true), same user scratchpad.

**Do NOT tune the GEMMs and do NOT chase per-stage kernel fusion for prefill.**
Per-token compute is already at parity.

### The cold-start cost (real, but a server never pays it)

oneDNN keys matmul primitives on (m,n,k), so a fresh process JIT-compiles a new
plan set for every distinct prompt length. Measured in layer 0 at M=3672:
33.8 ms for the qkv shape, 147.0 ms for the FFN-down shape, versus ~0.1 ms for
the same marks in layers 1-51. Total ~600 ms per fresh invocation. That is the
whole of the apparent prefill gap, and it amortises to nothing in any
long-running process.

Prefill work that DID help this session (all correctness-verified, output stays
byte-identical to Fusion's " to=selfWrite a haiku about"):

| change | cold prefill 3672 | tok/s |
|---|---|---|
| session start | 2676 ms | 1372 |
| FP16 activations (a3ac518) | 2470 ms | 1486 |
| GEMM direct-to-dst (d13d8d7) | 2412 ms | 1523 |

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

## Retracted - read before re-deriving

- **"Grimoire prefill is 65% of Fusion": WRONG, retracted 2026-09-01.** Cold
  Grimoire process vs warm Fusion server. Matched properly it is 97%. Always
  use GRIMOIRE_PREFILL_WARMUP=1, and remember my Fusion measurements warmed
  each shape with a throwaway request first.
- **"Grimoire GEMMs are 1.85x slower on FFN down / o_proj": WRONG, retracted.**
  Compared instrumented Grimoire stages (GEMM + conversions + profiler drain)
  against Fusion's pure GEMM. Measured properly they match within 3%.

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

## Qwen MTP + W4A8: the EXACT_VERIFY requirement (2026-09-01)

Verified working recipe, measured on one B70:

    -m /models/Qwen3.8-27B-MXFP4-GRIMOIRE --proj mxfp4 --ctx 8192
    GRIMOIRE_MTP=1
    GRIMOIRE_W4A8=1
    GRIMOIRE_MTP_EXACT_VERIFY=1
    GRIMOIRE_XE2_ATTN_BRIDGE=<...>/libgrimoire_xe2_attention_raw.so

    pp 4070 tokens in 1628 ms -> 2499.4 tok/s
    MTP(k=3) tg 128 tokens in 3420 ms -> 37.4 tok/s
    MTP steps 37, 3.51 committed/step, draft accepts 93/101 (92%), rollbacks 8

vs the 2026-08-26 record of TG 31.0 / PP 2516.7: TG +21%, PP unchanged.

**GRIMOIRE_W4A8=1 WITHOUT GRIMOIRE_MTP_EXACT_VERIFY=1 SILENTLY DESTROYS MTP.**
Measured, same prompt and build:

| flags | PP | TG | accepts |
|---|---|---|---|
| MTP only | 1685 | 22.1 | 93/102 (91%) |
| MTP + W4A8 | 2711 | 13.4 | **0/128 (0%)** |
| MTP + W4A8 + EXACT_VERIFY | 2499 | **37.4** | 93/101 (92%) |

Cause (src/grimoire.cpp, the `if (!exact_verify && xe2_w4a8_f32 && a8 && a8s)`
branch in the generic prefill's `next_tokens` block): with W4A8 the VERIFY pass
quantizes activations to int8 before the lm_head projection. That perturbs the
logits enough to flip argmax, so the verifier disagrees with the drafter on
every token -- 0 accepts, 1.00 committed/step, a rollback every step. The
symptom looks exactly like a broken drafter but the drafter is fine.
GRIMOIRE_MTP_EXACT_VERIFY=1 forces the accurate lm_head path for verification
only; W4A8 still accelerates everything else. This pairing was never documented.

**GRIMOIRE_PREFILL_WARMUP=1 HANGS QWEN.** That flag (added 2026-08-31 for the
Muse cold-start measurement) runs a throwaway prefill then reset() before the
timed one. On Qwen it stalls at 100% CPU on one core with the GPU idle, right
after "lm_head mxfp4 ... ok" and before the 64 layers load. It is Muse-safe but
NOT Qwen-safe -- Qwen's prefill carries DeltaNet recurrent state that the extra
prefill+reset leaves inconsistent. Do not use it on Qwen or Ornith. It is not
deterministic: one earlier Qwen run with the flag did complete (PP 2711).
