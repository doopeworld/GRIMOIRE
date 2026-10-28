# Qwen3.8 context comparison — 2026-10-10

GRIMOIRE: one Arc Pro B70, Qwen3.8-27B-GPTQ-Int4-MTP-BF16, MTP K=6,
draft vocabulary 65536, one sequence slot, context capacity 135168, one user,
thinking disabled, greedy, natural EOS. Auto prompt chunk budget. Fresh prompt
speed is actual prompt tokens divided by first positive token latency. Decode
uses the repeat request. All ten answers pass syntax and interval behavior tests.
These are individual observations, not a statistical estimate.

Wrapzii: published coding results from two Arc Pro B60s, TP2, Swift 1.5 baked
checkpoint, MTP6 and batch cap 4224. Hardware, weights, prompt lengths and sample
methods differ. This is a published deployment comparison. The 8K/64K pair uses
his October 2 sweep (warm median decode); 128K uses the later corrected W4A8 path
(single fresh confirmation), including **1254.9 prompt / 78.02 decode**. This is
not a uniform post-fix sweep. The wiring report also has a warmed 8K sample of
139.53 decode tok/s, without a paired fresh prompt speed in its summary table.

| Context | GRIMOIRE prompt tok/s | GRIMOIRE decode tok/s | Wrapzii prompt tok/s | Wrapzii decode tok/s |
|---|---:|---:|---:|---:|
| 4K | 1,975 | 119.4 | — | — |
| 8K | 2,011 | 116.1 | 1,724 | 136.4 |
| 16K | 1,868 | 114.2 | — | — |
| 64K | 1,267 | 88.6 | 1,367 | 97.9 |
| 128K | 834 | 65.8 | 1,255 | 78.0 |

Sources: [Wrapzii October 2 sweep](https://github.com/Wrapzii/k8v4-xpu/blob/main/docs/batched-prefill-2026-10-02.md)
and [W4A8 wiring correction](https://github.com/Wrapzii/k8v4-xpu/blob/main/docs/w4a8-wiring-2026-10-02.md),
rechecked October 10. Comparable 4K/16K coding rows are not published there.
Historical forced 96-token fox results are a different workload.

GRIMOIRE now processes the 8K prompt faster than this published sweep. Its decode
is slower at every length with a comparable published row. Long prompt speed is
still slower at 64K and 128K. Prior claims of universal wins are unsupported.

## Raw measurements

| Target | Actual prompt tokens | Fresh TTFT s | Repeat TTFT s | Fresh decode tok/s | Repeat decode tok/s | Passing answers |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 4,279 | 2.166 | 2.090 | 119.0 | 119.4 | 2/2 |
| 8K | 8,374 | 4.165 | 4.112 | 116.2 | 116.1 | 2/2 |
| 16K | 16,565 | 8.870 | 8.844 | 114.1 | 114.2 | 2/2 |
| 64K | 65,718 | 51.861 | 52.061 | 88.6 | 88.6 | 2/2 |
| 128K | 131,252 | 157.405 | 160.494 | 65.8 | 65.8 | 2/2 |

Receipts: Tower `/mnt/storage/isos/grimoire-runs/bench-1003/cmp-k8v4/grim-pf1010-final-solo-1010-1503.jsonl`;
wrapper `/mnt/storage/isos/grimoire-runs/bench-1010/serve-final-solo.out`.
Mac copies in `~/grimoire-work/results-20261010/`.

## Controlled prefill comparison

Same prompt prefix, context capacity 135168, two slots, K6, vocab65536, chunk2048:

| Context | Committed TTFT s | Candidate TTFT s | Prompt throughput gain |
|---|---:|---:|---:|
| 4K | 2.53 | 2.47 | 2.4% |
| 16K | 10.41 | 9.80 | 6.2% |
| 64K | 65.39 | 56.08 | 16.6% |
| 128K | 199.01 | 163.82 | 21.5% |

All corresponding coding answers PASS. The controlled comparison isolates the
prefill library change. The five-context table also changes slot count and auto
chunk budget; its difference from October 9 cannot be attributed solely to the
attention kernel. Decode depends on generated answer and speculative acceptance.

## Retained changes

- Correct GRF allocation for ESIMD and joint_matrix in the same library.
- Eight query rows per ESIMD thread, 128-key blocks by default for validated
  Qwen geometry H24/KVH4/D256. Other shapes keep joint_matrix.
- Single-slot generation can use the existing fast DPAS verify kernel.
- Correct inclusive cache lengths in older single-slot verify alternatives.
- Propagate launcher failures; numerical probes link the production library.

Four-row flash, packed temporary keys, interleaved output accumulators and blocked
BF16 weight panels were rejected. The measured FFN cost remains substantial.
The published v1.9.0 image is unchanged and still requires its existing warning.

Ornith controls prevented a broad prefill default: the candidate ESIMD 4K answer
missed empty-list handling with and without MTP. Keeping its original joint_matrix
prompt kernel restores 4K/8K behavior PASS with fast one-slot verify enabled.
This is why the default is limited to the validated Qwen geometry.

## Validation checkpoint

Source7250ac3, Tower longctx-attn-wip. Final default dispatch numeric probes pass
at one token, ragged/resumed/long Qwen shapes, and Ornith H16/KVH2. The latter's
whole output hash equals explicit joint_matrix. Scoped default K4 served checks
pass all eight Qwen/Ornith4K/8K answers. Actual 96-token GPTQ exact and legacy
exact outputs match plain; the old binary fails the same engine fixture.
Full model regression completed44/44 with ALL DONE and no GPU0 guard trip.
Sherlock reference and hot-expert text are identical. Existing speculative prose
comparisons still differ from plain; these44 smoke runs are not universal token
parity or44 behavioral passes. General greedy parity is not established by the
96-token GPTQ fixture.

All agent-owned GPU containers exited; Ian resumed GPU use with his ComfyUI
container. Source7250ac3 and validation handoff are saved on longctx-attn-wip.
Intel cloud remains pending review after28hours; no cloud workload was run.
