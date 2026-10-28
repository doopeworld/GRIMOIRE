# GRIMOIRE — paused 2026-10-09, resume tomorrow

Ian: finish the current diagnostic, save work, and resume tomorrow. No further
GPU jobs are authorized by this pause. Resume only when Ian asks.

## Findings

- One-slot MTP corruption fixed in dd286b3: fused DeltaNet verification left live
  state untouched; fully accepted blocks were never committed by generation.hpp.
  Qwen K=4/K=6 and Ornith exposed the old bug. Repaired coding requests pass.
- Full gpu0 regression completed 44/44, ALL DONE, with no guard trip. Sherlock
  reference and hot-expert text are identical. Existing speculative prose
  differences also appeared in the earlier regression; no blanket parity claim.
- Attention work-group A/B was completed at identical cache capacities. The
  unconditional grouping regressed short contexts. Selected layout in 031a87b
  retains committed short-context mapping and groups when the long-context split
  floor exceeds the thread target. 64K/128K attention improves 26%/22% in the
  isolated kernel comparison. Prefill was not improved by this verify-kernel change.
- Full context table completed with 10/10 GRIMOIRE behavior checks passing.
  One B70, two slots, ctx 135168, MTP6, draft vocab 65536, thinking off:
  prompt/decode tok/s 4K 1928/120.0; 8K 1872/121.2; 16K 1691/115.0;
  64K 1052/77.1; 128K 678/66.8. See RESULTS-2026-10-09-CONTEXT.md.
- Ian requested Wrapzii's published GitHub results. His latest 128K row is
  1254.9 prompt / 78.02 decode (W4A8 wiring confirmation), not 1111/74.48.
  Table corrected in 9b76b0a. His 8K/64K rows are earlier published samples.
  His hardware is two B60s; checkpoint is Swift, not our base Qwen checkpoint.
- The old saved comparison only established a short-context decode win on equal
  hardware: GRIMOIRE 8K 122.5 vs Wrap's TP1 one-B70 118.2, both PASS. Today GRIMOIRE
  8K is 121.2, not a large regression. The current published baseline is two B60s
  at 136.38. "GRIMOIRE beats Wrapzii everywhere" was never established; long-context
  correctness was broken and no completed long-context comparison existed then.
- The attempted local k8v4 launch exited during model initialization: TP2-only
  attention library expects heads 12/2; TP1 needs 24/4. No local numbers generated.
  No retry after Ian switched the requested comparison to published GitHub data.
- New diagnostic: profile16k-baseline.log, host region timing (TIME_LAYER=all;
  no profiling queue or retained-event mode). CLI, same model and ctx/slot/MTP
  settings, 16556 prompt tokens, one output token. Chunks: 5120/5120/5120/1196.
  Timed region total 9685.500 ms; generation elapsed 9.756 s. These are diagnostic
  timings, not replacement production benchmark rates.
  Feed-forward gate/up plus down = 4490.451 ms (46.4%). Flash attention =
  2315.382 ms (23.9%). The three equal 5120-token chunks spend 225.285 / 656.592 /
  1084.187 ms in flash attention as context grows. Final 1196-token chunk: 349.318 ms.
  This identifies costs at 16K, not a measured attribution of the entire 128K run.
- The existing flash prefill ESIMD implementation is OFF by default in gemm_fast.cpp;
  the default is joint_matrix. Its ESIMD alternative has reported register spilling
  under the current library build. Verify compiler output and measure it with proper
  -doubleGRF before changing defaults. No performance patch was attempted after profiling.
- Saved 128K decode timing: ~19.73 ms drafting, 60.13 ms verification, 1.58 ms commit;
  ~81.44 ms/step, 5.41 emitted tokens/step over the sampled block. Current library's
  long-context attention still accounts for much of the increased step cost.
- v1.9.0 release notes now warn about the one-slot bug; release image/asset unchanged.
- All GPU containers stopped naturally after tests. gpu0 VRAM 0.02 GiB; all monitored
  correctable/nonfatal/fatal AER totals zero. GPU1 untouched; no queued/background jobs.
- Intel cloud key exists. Asked for How to Connect SSH line; none supplied. No cloud
  connection or heavy work performed.

## Next steps

1. Resume performance work on gpu0 only when Ian asks. Start with the measured
   prefill costs: feed-forward GEMMs and growing attention time. Prove each change
   with a controlled A/B and passing generated answers; preserve pure SYCL/Level Zero.
2. Check the existing ESIMD prefill candidate and its actual register allocation,
   then test a separate library built with -doubleGRF. Compare with default at the
   same cache and prompt/chunk sizes. Never use diagnostic TIMING/DEBUG rates as wins.
3. Inspect large-M GPTQ feed-forward GEMM efficiency and whether a pure SYCL MLP-only
   W4A8 path is justified; do not enable a weight conversion before auditing payload
   lifetime, shape bounds and correctness. Keep one change at a time.
4. Revalidate any retained change with actual coding answers at 4K/8K/16K/64K/128K.
   Compare only matching hardware/workload/settings when claiming a win. Published
   two-B60 results are a separate reference, not an equal-hardware benchmark.
5. Intel cloud: when SSH line arrives, inspect hardware only with the designated key.

Artifacts: Tower bench-1003/profile16k-baseline.log and prompt-profile16k.txt;
Mac ~/grimoire-work/results-20261009/profile16k-baseline.log and context report.
Code and handoff live on longctx-attn-wip. Do not resume jobs from old watcher scripts.
