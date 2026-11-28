# Native B60 decode: what helped, what to investigate, and what to park

These are screening results from two Intel Arc Pro B60s running the native
`llama-bench` adapter, not official HTTP serving benchmarks. The baseline was
built from Grimoire v1.10.10 source (`acc75ea3`), with the native adapter and
bench-only probes, not the published executable. Use these numbers to decide
what to measure next, not as general throughput claims. The default engine
behavior is unchanged by this note.

## Baseline and clocks

Unless noted otherwise: Swift INT4, TP2, one slot, 8,192 identical input token
IDs, 128 exact output commits, context 8,336, greedy MTP depth 6, oneDNN on,
FP8 KV, and graph replay off. A fresh load includes one discarded complete
warmup and three measured requests. Counterbalanced results compare both run
orders for a prompt. Prefill is the host-call duration around the native
prompt prefill. Decode is first commit to both ranks drained (127 subsequent
commits). Neither is a GPU-event span or HTTP delivery time. Do not sum the two
ranks or add nested timings to their parent spans.

## oneDNN: 2,114 → 2,506 input tok/s (+18.54%)

On the same warmed native counting workload, enabling oneDNN saved **0.606 s**
of prompt prefill: **3.875262 → 3.269224 s (−15.64%)** on rank 0. Prefill
throughput is `8192 / prompt_prefill_seconds`, not `8192 / time_to_first_token`
and not the sum of two TP ranks. All 128 ordered output IDs matched on both
ranks, including the excluded warmup. The decode result was **126.49 → 127.55
output tok/s** (127 tokens after first commit). This one off-first pair does
not establish a decode speedup or generalize to other prompts.

| oneDNN / rep | Prefill r0, ms | Prefill r1, ms | Input tok/s r0 | First commit, ms | Decode, ms | Output tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Off 1 | 3873.801 | 3873.801 | 2114.72 | 3874.617 | 1004.048 | 126.488 |
| Off 2 | 3875.262 | 3875.273 | 2113.92 | 3875.951 | 1003.593 | 126.545 |
| Off 3 | 3878.294 | 3878.346 | 2112.27 | 3878.959 | 1008.870 | 125.883 |
| On 1 | 3269.224 | 3269.197 | 2505.79 | 3269.952 | 995.729 | 127.545 |
| On 2 | 3275.551 | 3275.513 | 2500.95 | 3276.198 | 996.720 | 127.418 |
| On 3 | 3269.189 | 3269.168 | 2505.82 | 3269.838 | 988.491 | 128.479 |

Both arms issued one 8,192-row prompt call per rank. Warmup dispatch receipts
show 1,024 BF16 VNNI fallback `gemm_predq` calls per rank with oneDNN off;
with it on, that counter went to zero and the logs show 512 fused rotated-INT8
oneDNN + SwiGLU calls and 384 ordinary oneDNN calls per rank. These are host
dispatch counts, not GPU durations. Keep oneDNN on for the decode baseline.

[Evidence and reproduction](benchmarks/b60-native-v11010/README.md) ·
[All 12 measured rows with unrounded clocks](benchmarks/b60-native-v11010/native-measurements.csv) ·
[oneDNN off](benchmarks/b60-native-v11010/onednn-0.verified.json) /
[on](benchmarks/b60-native-v11010/onednn-1.verified.json)

## Where to spend the next measurement window

| Lead | Observed on this setup | Decision |
|---|---|---|
| Target verification | In a native counting trace, decode was ~1,005 ms, of which draft, verify and commit were ~266, ~717 and ~12 ms on rank 0. | Investigate first. Begin with a specific verifier producer or attention reader, not a whole-stack change. |
| BF16 BAR peer rendezvous | **10,112 → 2,560 waits per rank** (−74.7%), at the same 2,560 reductions and with exact committed IDs. Decode elapsed **1,004.54 → 995.23 ms** (−0.93%) in one off-first pair. | Worth preserving as an opt-in experiment. Test slower interconnects independently; no established throughput gain there or reason to change the default. |
| All-visible verify tiles | Exact numerical/ID checks passed; native decode was **+4.00% / +0.15% elapsed** in the two orders. | Park this shortcut. It does not rule out a better attention or verifier kernel. |
| Device-only draft attention graph | Captured the device producer before the first host-visible reduction; native decode was **−0.94% / +0.07% elapsed**. | Park this partial graph. It did not remove the whole draft's host dependencies. |
| Gathered 65K draft vocabulary | Smaller head was ~72% cheaper per proposal on the host completion clock. Hybrid decode elapsed was **−10.75% / −11.09% on prose**, **−16.35% / −18.03% on code**, but **+90.00% / +91.33% on multilingual**; exact IDs in all 24 paired comparisons. | Do not promote the hybrid. Acceptance, not just head time, determines cost per commit. |
| Adaptive hybrid draft head | A predeclared 15 ms/commit budget over four complete hybrid rounds sent multilingual back to full at round 5 and warmed prose at round 9. Code never switched. Decode elapsed was **−2.22% / −2.78%** on prose, **−16.52% / −16.85%** on code, and **−0.46% / −0.59%** on multilingual, all vs. that binary's full head. Exact IDs matched. | Mitigated this multilingual regression; near-neutral there, with the hybrid's 167.5 MiB per-rank allocation still resident. An opt-in screen, not a general policy or default. |
| Packed INT8 K / INT4 V | At a matched 16K retrieval point, target KV allocation was **275,025,920 → 209,442,816 bytes per rank** (−23.85%). The packed arm had **+18.4–18.7% prefill elapsed** and **+5.3–5.5% decode elapsed**. | Capacity lead, not a speed win. Qualify retrieval, rollback and usable context separately from throughput. |

### What the rendezvous number means

The measured path was synchronous Megatron TP2 with BF16 peer partials through
BAR. A whole partial was published once instead of once per piece; it still
advanced the shared counter by the original number of pieces. This changed
neither bytes transferred nor the number or arithmetic of reductions. Host
publication/peer wait/flush declined by about **10.7 ms** on rank 0. Most of
the remaining nested reduction time was in the wait for the producer, which
also includes already queued GPU work. It is not evidence that the fabric was
saturated. The paired prefill clocks drifted **3.247 → 3.361 s** (+3.51%, **2,523 →
2,437 input tok/s**), even though the change was in verify. This ordered pair
is not a repeatability result. [All rank spans, including warmup](benchmarks/b60-native-v11010/rendezvous-rank-spans.csv)
show the counts and elapsed ns per request; these are host spans, not GPU
kernel times.

For PCIe with fewer lanes, OCuLink, or another topology, first record the actual
link and verify that this BF16 BAR path dispatches. Counterbalance same-binary
on/off, verify exact outputs on both ranks, and compare whole-request decode
and host publication separately. Fewer rendezvous is not a 75% reduction in
transfer time. Keep the existing per-piece path available as a control. [Draft experiment #5](https://github.com/doopeworld/GRIMOIRE/pull/5) separates the opt-in protocol from the original measurement probes.

### Why the draft result does not generalize

The initial 65K numeric prefix passed a counting compute screen because the
largest stock draft ID was **220**. It had no realistic coverage pressure.
The later hybrid replaced 153 unused high prefix IDs with out-of-prefix IDs
observed on six independently authored calibration requests. Neither held-out
inputs nor their outputs selected those IDs. Despite the cheaper head, the
multilingual regression required **62 rounds instead of 28** and accepted
**1.05 instead of 3.54 drafts per round**. The three fixed prose, code and
multilingual prompts should now be treated as regression prompts, not fresh
samples for generalization. Use new independent data for any future coverage
claim; don't tune membership or thresholds on these three results.

### Kernel and KV directions

- **Verify attention:** compare only the candidate reader/kernel against the
  current B60 dispatch. Use the same KV layout and visible spans at first.
  Check HD256 GQA, unequal sequence positions, partial tiles, causal masking,
  and rejected-tail rollback with exact committed outputs before timing. Keep
  attention cost, target FFN/GDN producer completion, and BAR publication as
  separate measurements. A benchmark across repositories with several switches
  applied together is useful provenance, not attribution to one kernel.
### Packed V4 at 16K: less target KV, slower at this operating point

Both arms used INT8 K, with only target V switching between FP8 and packed
scalar affine INT4; the drafter retained FP8 V. Context capacity was 16,528
with one slot, 16,384 input IDs and 128 exact output commits. Each fresh load
had one discarded warmup and three measured requests. Here are the block
medians in actual run order:

| Block | Target V | Prefill r0, s | Input tok/s | Decode, s | Output tok/s |
|---|---|---:|---:|---:|---:|
| 0 | FP8 | 20.708186 | 791.18 | 1.408032 | 90.197 |
| 1 | INT4 | 24.580613 | 666.54 | 1.484708 | 85.539 |
| 2 | INT4 | 24.585128 | 666.42 | 1.488070 | 85.345 |
| 3 | FP8 | 20.760738 | 789.18 | 1.413092 | 89.874 |

[All 12 measured rows](benchmarks/b60-native-v11010/packed-v4-16k.native-rows.csv)
include both rank prefill clocks, first-commit latency, draft counters and
retrieval flags. All requests recovered the three markers in order and had
the same 128 committed IDs. Draft work changed (FP8: 101 accepted / 162
judged; INT4: 102 / 156), so native decode does not isolate the attention
reader. The target KV allocation change is **−65,583,104 bytes per rank**
(−23.85%); that is not the change in whole-engine residency or evidence of
maximum usable context. Improve the read/dequant path at fixed cache layout
first. Qualify longer contexts, rollback and independent slot histories
separately before claiming a capacity or serving gain.
