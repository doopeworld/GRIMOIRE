# Decode rewrite — bandwidth first

## The standard

**Anything under 550 GB/s sustained is not worth building.** The B70 is a
608 GB/s part. Every decode change is judged against achieved bandwidth
first and tok/s second. If a change does not move GB/s, it is not a change
worth making.

Judge with bytes, not benchmarks: `achieved GB/s = active_bytes / token_time`.
`llama-benchy` pp4096 drifts 1969-2500 on identical binaries, so it cannot
referee anything below a ~20% delta. The device timeline can.

## The budget (Qwen3.8-27B, DENSE — no MoE, every token reads every weight)

```
active bytes / token      ~14.5 GB   (15.87 GiB artifact - embed_tokens)
  FFN  3 x 17408 x 5120 MXFP4         142 MB/layer  x64 = 9.1 GB   <- dominant
  attention / GDN projections + lm_head              ~5.4 GB
  KV cache, all 16 full-attn layers @4778 tok        0.157 GB      <- trivial

@ 608 GB/s -> 23.8 ms/token -> 42 t/s base
@ 550 GB/s -> 26.4 ms/token -> 38 t/s base   <- TARGET
measured      43.9 ms/token -> 330 GB/s      <- 54% of the card
```

## Where the 330 GB/s is lost

### Lane 1 — attention mixer: ~11.4 ms of pure waste
```
full-attention layer   1197 us
linear (GDN) layer      463 us
delta = attention mixer 734 us   x16 layers = 11.7 ms
KV bytes moved, all 16 layers: 157 MB = 0.29 ms @ 550 GB/s
```
Nearly the entire 11.7 ms moves no bytes. It is occupancy and access-pattern
loss in the hand-written scalar kernel (16 lanes reading 16 bytes of a 64-byte
line, 256 times per key).

**Fix: delete the hand-written decode kernel. Use `cutlass_paged_decode_xe2`.**
- `csrc/xpu/attn/xe_2/paged_decode_xe2.h` in vllm-xpu-kernels
- `decode_policy_q8_h256_p16` — head_size 256 is built
- qgroup is a packed-Q TILE size, not a cap: "Ratios <= 8 use qgroup=8".
  Ours is 24/4 = 6, so q8 is correct. Nothing needs inventing.
- split-K (`num_kv_splits`, `temp_out`, `exp_sums`, `max_logits`), GQA tiling
  and fp8 `k_scale`/`v_scale` are all kernel parameters. No hand-rolled
  softmax, no hand-rolled split merge, no `GRAPH_SPLITS` constant.

Blocker is LAYOUT, not dtype. vLLM runs `--kv-cache-dtype fp8` on this same
kernel. It needs `key_cache [num_block, block_size, heads, head_size]` paged;
GRIMOIRE stores `K [kv_head][head_dim][seq_cap]` D-major and
`V [kv_head][seq_cap][head_dim]`. That private format is the entire reason
decode cannot use the path prefill already wins on (2500 vs vLLM 1691).

Ceiling for this lane alone: 43.9 -> ~32 ms = 450 GB/s = 31 t/s. **Short of
550. Necessary but not sufficient.**

### Lane 2 — FFN GEMV: the term that actually decides it
```
142 MB per layer in 463 us = 307 GB/s
needs 142 MB in 258 us     = 550 GB/s
9.1 GB of the 14.5 GB budget
```
This was never examined on 2026-09-04. It is the dominant term and the one
that decides whether the target is met. Measure with `tools/bench_stream.cpp`
before touching it.

## Order of work

1. **Measure GEMV achieved GB/s** in isolation. Dominant term, unexamined.
2. **Repave attention**: KV cache -> paged blocks, `launch_kv_append_dev`
   writes blocks, decode calls `cutlass_paged_decode_xe2`.
3. **Unify**: decode (`seq_q=1`), MTP verify (`seq_q=M`) and prefill all on
   the one cutlass path. Today they are three implementations, two
   hand-written — the source of every bug hit on 2026-09-04.
4. Delete `launch_flash_decode`, `launch_flash_merge`,
   `launch_flash_decode_batched`, `decode_splits`, `GRAPH_SPLITS`,
   `MAX_SPLITS`, and the `GRIMOIRE_DECODE_BATCHED_*` flags.

## What this supersedes

The 2026-09-04 split-K work (`5964dff`) took base decode 11.7 -> 22.8 t/s and
the token 82.0 -> 43.9 ms. It stays only as a floor while lane 1 is built,
then it is deleted with the rest of the hand-written kernel. It was patching
the wrong road: even perfect, it tops out around 450 GB/s.

Do NOT resume the open M=1 batched-attention bug
(`GRIMOIRE_DECODE_BATCHED_ATTN`, see DECODE-ATTENTION-2026-09-04.md). It is
worth ~1.2x on a code path scheduled for deletion.

---

# MEASURED: the bridge exists — 625.7 GB/s (2026-09-04, tools/bench_bridge.cpp)

Bare SYCL, 8 GB VRAM arena, no model, no cutlass, no torch. Same card, same
bytes, same allocation — only the access pattern differs:

```
scalar1      35.6 GB/s   <- what launch_flash_decode does today
vec4        137.1 GB/s
vec16       411.2 GB/s   <- wide loads alone
vec16x4     625.7 GB/s   <- wide loads + 4 streams in flight
```

**17.6x between the worst and best pattern on identical hardware**, and the
worst one is the one decode uses.

Conclusions that supersede everything above:

1. **The B70 is not the limit.** 625.7 GB/s beats the 608 spec sheet.
2. **cutlass is NOT required.** Raw SYCL reaches full width. cutlass is fast
   because it emits wide loads, not because it is cutlass. The earlier plan to
   route decode through `cutlass_paged_decode_xe2` is optional, not necessary —
   prefer owning the kernel.
3. **Width alone is not enough.** vec16 = 411 GB/s; four independent streams
   take it to 625. The load pipeline needs several requests in flight to hide
   latency. Any rewrite must carry >= 4 concurrent streams per sub-group.
4. **Sub-group-contiguous addressing is mandatory.** Lanes must read adjacent
   16 B chunks so each step retires one 256 B run.

## Budget against the measured bridge

```
14.5 GB/token @ 625.7 GB/s ->  23.2 ms -> 43 t/s base   <- achievable today
                 measured  ->  43.9 ms -> 330 GB/s      <- 53% of the road
```

43 t/s base on ONE card, no speculation, no TP. With MTP at the conservative
1.25x measured on 2026-09-04 that is ~54 TG, past the 44 bar, with the second
B70 untouched.

## Revised order of work

1. Port the `vec16x4` pattern into the FFN GEMV (9.1 GB of the 14.5 GB budget,
   currently 307 GB/s). Largest single win.
2. Same pattern for the attention mixer (currently ~35 GB/s scalar).
3. Lay the `.b70` arena out in execution order so a token is one linear sweep.
   The format is ours -- safetensors cannot do this, and it is the one
   structural advantage GRIMOIRE has over every other engine on this card.
4. Only then consider TP.

Re-run the bar at any time: `bin/bench_bridge <GB> <iters>`.

---

# Format: `.b70` reads every quantized tensor TWICE

Every serious engine owns its format because the format is where the bandwidth
lives: OpenVINO has IR, llama.cpp has GGUF, TensorRT has engine files. We have
`.b70`. It is currently laid out for convenience, not for streaming.

```c
struct NativeTensorRecord {
    uint64_t payload_offset, payload_bytes;   // weights here
    uint64_t scales_offset,  scales_bytes;    // scales far away
};
```

Two regions per tensor means **two streams from two distant addresses** for
every MXFP4 weight. GGUF's block quants interleave the scale INTO the block so
one contiguous read returns both. That is very likely part of why the FFN GEMV
measures 307 GB/s while `bench_bridge`'s single contiguous stream reaches 625.7.

Converter changes (`tools/b70_compile_model.cpp` -- ours, no retraining):

1. **Interleave scales into the payload.** One block = its weights and its
   scale, adjacent. One stream, not two.
2. **Order tensors by execution** so a token is one linear sweep from byte 0.
3. **Align blocks to the 256 B sub-group run** `bench_bridge` showed is optimal.

Note this changes the artifact, so the loader must accept both layouts (as it
already does for the RAW-vs-packed MTP head) or the artifact version bumps.

**Keep it format-agnostic: int4 and MXFP4 both.** Under streaming int4 g128 is
only ~6.7% off MXFP4 (the old 1.28-1.62x figure was cache-hot, see
grimoire-int4-path). Set the priority accordingly: choosing the quant format is
a ~7% decision, the access pattern is a 17.6x decision (35.6 -> 625.7 GB/s).
Do not spend time picking a format until every load is wide. The block layout
-- interleaved scale, 256 B alignment, execution order -- must work for both.

# START HERE TOMORROW

```
bin/bench_bridge 8 5          # re-establish the bar: expect ~625 GB/s
```

Then, in order, measuring GB/s after each step and stopping if it does not move:

1. **FFN GEMV** -- 9.1 GB of the 14.5 GB budget, currently 307 GB/s. Port the
   `vec16x4` pattern: 16 B per lane, lanes sub-group-contiguous, >= 4
   independent streams in flight. Largest single win.
2. **Attention mixer** -- currently ~35 GB/s (scalar byte loads). Same pattern.
   Do NOT resume the M=1 batched-attention bug; that code is for deletion.
3. **Converter** -- interleave scales, order by execution, align to 256 B.
4. TP across both B70s only after one card runs at width.

Standing rule: judge every change on achieved GB/s from the device timeline,
not on llama-benchy, which drifts 1969-2500 on identical binaries.
