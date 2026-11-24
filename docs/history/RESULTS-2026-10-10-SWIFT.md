# Swift 1.5 comparison — 2026-10-10

Both columns now use Wrapzii's exact
`Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8` checkpoint.
GRIMOIRE uses one Arc Pro B70; Wrapzii's published deployment uses two B60s, TP2.
This is a comparison of deployments with different hardware and engine settings.

GRIMOIRE: native INT4 body/head/MTP, resident INT8 embedding, MTP K=6,
draft vocabulary 65536, one sequence slot, context capacity 135168, auto prefill
chunk budget, concurrency one, thinking off, greedy and natural EOS. Fresh prompt
speed = actual prompt tokens / first positive token latency. Decode uses one
repeat request. All ten answers pass syntax and interval behavior checks.
These are individual observations, not statistical estimates.

Wrapzii: batch cap 4224, MTP6, K8/V4 cache. The 8K/64K paired numbers are his
October 2 sweep; 128K uses the later corrected W4A8 confirmation. Warm median
decode at 8K/64K and single fresh decode at 128K differ from our sample method.
The wiring report also has a newer warm 8K decode sample of 139.53 tok/s with
no paired fresh prompt rate in its summary table. These are published results,
not a local Wrapzii run on the Tower. Comparable 4K/16K coding rows are absent.

| Context | GRIMOIRE prompt tok/s | GRIMOIRE decode tok/s | Wrapzii prompt tok/s | Wrapzii decode tok/s |
|---|---:|---:|---:|---:|
| 4K | 1,967 | 113.9 | — | — |
| 8K | 2,000 | 111.0 | 1,724 | 136.4 |
| 16K | 1,853 | 107.2 | — | — |
| 64K | 1,265 | 82.1 | 1,367 | 97.9 |
| 128K | 827 | 61.3 | 1,255 | 78.0 |

Sources: [October 2 paired sweep](https://github.com/Wrapzii/k8v4-xpu/blob/main/docs/batched-prefill-2026-10-02.md)
and [corrected W4A8 measurement](https://github.com/Wrapzii/k8v4-xpu/blob/main/docs/w4a8-wiring-2026-10-02.md).
Rechecked October 10. The latest 128K value is **1254.9 prompt / 78.02 decode**,
not the superseded 1111/74.48 pair.

Matching Swift did not close the gap. GRIMOIRE prompt speed exceeds this published
8K sweep, but decode trails at all comparable lengths; 64K/128K prompt speed also
trails. Hardware, KV format, draft configuration and sample methods still differ.

## Exact receipts

| Target | Actual prompt tokens | Fresh TTFT s | Repeat TTFT s | Fresh decode tok/s | Repeat decode tok/s | Passing answers |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 4,278 | 2.175 | 2.097 | 113.80 | 113.87 | 2/2 |
| 8K | 8,373 | 4.186 | 4.135 | 111.01 | 110.96 | 2/2 |
| 16K | 16,564 | 8.941 | 8.879 | 107.16 | 107.22 | 2/2 |
| 64K | 65,717 | 51.964 | 52.673 | 81.99 | 82.05 | 2/2 |
| 128K | 131,251 | 158.770 | 157.430 | 61.36 | 61.33 | 2/2 |

Tower JSONL: `/mnt/storage/isos/grimoire-runs/bench-1003/cmp-k8v4/grim-swift-final-1010-1836.jsonl`.
Logs/scripts: `/mnt/storage/isos/grimoire-runs/bench-swift-1010/`.
Mac backup: `~/grimoire-work/results-20261010-swift/`.

## Loader fix and validation

The old directory scan overwrote the dense embedding with INT8 side-file codes;
the BF16 upload converted raw integers without applying scales. The new loader
selects the manifest's side file explicitly and attaches its row scales. It
rejects unscaled INT8 and malformed recipes before weight upload.

The lookup matches the reference's rounding order: scale FP16 -> BF16, integer
code -> BF16 (exact), multiply -> BF16. Codes and 16-bit scales stay resident;
the embedding uses 1.18 GiB instead of the dense table's 2.37 GiB. No dense
embedding expansion or framework dependency is used in GRIMOIRE.

CPU: 51,200 independent rational oracle values pass, including signed zero.
Negative controls mismatch for 50,490 unscaled values and 11,058 late-rounding
values. Loader rejection/selection cases pass; real Swift resolves the INT8
source/scales and all nine baked GPTQ linears. Existing safetensors suite passes.
GPU0: production single, batch, shard and dynamic graph lookup at widths 256 and
5120 matches the oracle bit for bit. All ten actual Swift coding answers pass.

Already packed HF INT4 MTP projections retain their saved precision by default.
Explicit GRIMOIRE_MTP_HEAD_FMT overrides remain supported.

| Same 4K/8K pilot prompts | 4K repeat decode | 8K repeat decode | Behavior |
|---|---:|---:|---|
| Native INT4 MTP, draft vocab 65536 | 114.0 | 111.1 | 4/4 PASS |
| BF16 MTP, draft vocab 65536 | 95.2 | 93.4 | 4/4 PASS |
| Native INT4 MTP, full draft vocab | 110.7 | 102.3 | 4/4 PASS |

Native INT4 / 65536 was retained. BF16 produces the same answers and acceptance
on these samples but costs more time. Full vocabulary improves acceptance, with
added projection traffic outweighing the gain. No fine-tune speed gain is assumed.

The broad 44-run model sweep completed before this Swift patch. Post-patch CPU,
embedding GPU, actual Swift and targeted original-model checks are separate evidence;
a full new 44-model sweep is not claimed. Multi-GPU execution and a new release
image are not validated by this work. Published v1.9.0 image and warning are unchanged.
