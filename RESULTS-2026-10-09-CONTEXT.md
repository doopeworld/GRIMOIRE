# Qwen3.8 context comparison — 2026-10-09

GRIMOIRE: measured on one Arc Pro B70, commit 031a87b, Qwen3.8-27B-GPTQ-Int4-MTP-BF16,
MTP K=6, draft vocab 65536, 2 sequence slots, context capacity 135168, one user,
thinking disabled, greedy, natural EOS. Fresh prompt speed = actual prompt tokens / TTFT;
decode below is the repeated request's rate. All 10/10 generated functions pass behavior checks.

Wrapzii: latest available published coding measurements per context, batch-token cap
4224, Swift 1.5 Qwen3.8 baked checkpoint, two Arc Pro B60s, TP2, MTP6. The 8K/64K
rows are from the October 2 isolated sweep; the 128K row uses the subsequent W4A8
wiring fix, 1254.9 prompt / 78.02 decode tok/s. These are published results, not
measurements on Ian's Tower. Hardware, checkpoint and context counts differ.
Comparable 4K/16K coding rows are not reported. 8K/64K decode is a warmed median;
128K decode is a single fresh confirmation. The latest fix was not retested at
8K/64K in that report, so this is not one uniform runtime sweep.

| Context target | GRIMOIRE prompt tok/s | GRIMOIRE decode tok/s | Wrapzii prompt tok/s | Wrapzii decode tok/s |
|---|---:|---:|---:|---:|
| 4K | 1,928 | 120.0 | — | — |
| 8K | 1,872 | 121.2 | 1,724 | 136.4 |
| 16K | 1,691 | 115.0 | — | — |
| 64K | 1,052 | 77.1 | 1,367 | 97.9 |
| 128K | 678 | 66.8 | 1,255 | 78.0 |

Source: [Wrapzii October 2 coding sweep](https://github.com/Wrapzii/k8v4-xpu/blob/main/docs/batched-prefill-2026-10-02.md).
The 8K/64K sweep predates the corrected W4A8 installer. The table's updated 128K
row comes from his [W4A8 wiring fix](https://github.com/Wrapzii/k8v4-xpu/blob/main/docs/w4a8-wiring-2026-10-02.md):
1,254.9 prompt tok/s and 78.02 decode tok/s at 128,197 tokens, a single fresh confirmation.
The older 128K sweep row was 1,111 prompt / 74.48 decode; it is superseded in this table.
Historical 16K fox data exists (2,030.25 prompt / 98.88 decode tok/s), but uses forced
96-token completions and a different workload; excluded from this coding comparison.

Actual prompt tokens / fresh TTFT on GRIMOIRE:

| Target | Actual prompt tokens | Fresh TTFT s | Answers passing |
|---|---:|---:|---:|
| 4K | 4,284 | 2.22 | 2/2 |
| 8K | 8,379 | 4.48 | 2/2 |
| 16K | 16,570 | 9.80 | 2/2 |
| 64K | 65,723 | 62.47 | 2/2 |
| 128K | 131,257 | 193.68 | 2/2 |

The Tower's attempted local k8v4 launch exited during model initialization because
its installed attention library is compiled for TP2 heads 12/2, while TP1 needs 24/4.
No local k8v4 benchmark numbers were produced. Ian then explicitly requested GitHub
results; no retry was run. All GPU containers are stopped, gpu0 VRAM is 0.02 GiB,
and all monitored gpu0 AER counters remain zero.

One-slot bug: fixed in dd286b3; first failing candidate 1aa80fd. Full regression sweep
completed 44/44 without a GPU guard trip. The selected attention change (031a87b)
preserves committed short-context layout and improves long-context attention.
v1.9.0 release notes now warn about the unchanged release binary's one-slot bug.
Intel cloud: key exists; How to Connect SSH line requested, inspection pending.
