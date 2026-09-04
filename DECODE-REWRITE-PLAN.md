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
