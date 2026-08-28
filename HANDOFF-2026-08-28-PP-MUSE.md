# GRIMOIRE handoff -- Muse Glimmer batched prefill (PP), parallel track

Date: 2026-08-28. Isolated worktree/branch `pp-muse`, built from the coherent
DFlash checkpoint (c3f8d28). Runs on GPU1 (renderD130, PCI 35:00.0) only, kept
separate from the GPU0/DFlash track intentionally.

## Goal
Owner wants >=2000 pp on Muse Glimmer 30B. Roofline (dense 6656/19968, 52
layers, 39 sliding + 13 full-attn, GQA 32/2): ~47.5 GFLOP/token, compute-bound
above ~43 tokens. At the ~101 TFLOP/s the B70 hit on Ornith/Qwen dense W4A16
GEMM, ceiling is ~1780-2100 pp depending on prompt length. 2000 is close to
the ceiling, not comfortable headroom -- getting there requires the dense FFN
GEMM (87% of FLOPs/token) at ~95%+ of that efficiency. The FFN-is-DRAM-bound
issue found on Ornith/Qwen (weights re-read ~19x/layer) likely applies here
too and is probably the main gap.

## Status: FIRST BASELINE ATTEMPT CRASHED (host-side, not GPU)

Command:
`b70_native_inference -m /models/Muse-Glimmer-30B-MXFP4 --proj mxfp4 --prefill-only 64 --ctx 4096`
on GPU1 via the safe detached+timeout launch pattern (renderD130 only exposed,
--init, in-container timeout, graceful TERM before KILL).

Result: container exited 139 (SIGSEGV). dmesg:
`segfault at 9908b13e ip 00000000004de5c3 ... error 6` (write to unmapped
user address) on the host process -- NOT a SYCL/device exception, NOT
DEVICE_LOST. GPU1 PCI runtime_status confirmed `active` immediately after --
the card is fully healthy, this did not touch the USB4 hazard at all.

No log output was captured before the crash (stdout unbuffered via
setvbuf(_IONBF), so the crash happened before the first printf, i.e. during
`e.build()` or very early in `e.prefill()`, before the FULL E2E PP ONLY
line -- possibly even before the PASS/FAIL print).

## Next steps (whoever picks this up)

1. Get an actual stack trace: run once under `gdb --args` or with
   `ulimit -c unlimited` inside the container (still on GPU1, still with the
   safe launcher -- host segfault is a code bug, not a hazard, but keep the
   pattern anyway) to find which line in `build()` or `prefill_muse()`
   dereferences a bad pointer for the MXFP4 checkpoint specifically.
2. Prime suspect: something in `prefill_muse`'s new buffer sizing
   (`W=std::max(...)` over q/k/v/o/gate/sh_gu/sh_down dims, or the `dense`
   MXFP4 kernel dispatch `load_xe2_dense_mxfp4_f32()`) not matching this
   model's actual MXFP4 tensor layout/shapes -- this path was only exercised
   with n=16 (DFlash draft block) and n=30 (their target-only smoke test) so
   far, never with a larger batched M like 64. Check whether `W` or an MXFP4
   scale-buffer stride assumption breaks at this M.
3. Once it runs, re-attempt the 64-token probe, then step 128 -> 256 -> 512
   to build the real pp-vs-length curve and compare to the roofline above.
4. Note: this crash is unrelated to the DFlash GPU0 track's 50 TG goal, but
   the batched Muse forward is shared code (their DFlash verify step is
   this same kernel) -- once PP is fast and correct, verify should get
   faster too.

## Safety notes carried over

- GPU1 (35:00.0) is behind the CPU's Thunderbolt/USB4 root complex -- fragile
  by construction. GPU0 (03:00.0) is direct PCIe.
- Never SIGKILL a container with GPU work in flight -- always use the
  b70run.sh-style detached + in-container timeout + graceful TERM pattern.
- This crash was confirmed host-side (SIGSEGV, error 6, write to unmapped
  address) -- GPU1 was verified `active` after, no bus drop occurred.

## UPDATE: first real baseline (correct binary)

Root cause of the earlier crash: wrong binary. `--prefill-only` is parsed by
`tools/grimoire_main.cpp` (-> `bin/grimoire`), not `src/main.cpp`
(-> `b70_native_inference`). The latter has no flag parsing at all -- the
earlier SIGSEGV at both M=16 and M=64 was garbage argv handling, unrelated to
Muse/prefill_muse correctness. Not a real bug in the batched Muse path.

Built `bin/grimoire` via `tools/build_grimoire_only_b70.sh` and reran on
GPU1 (renderD130), Muse-Glimmer-30B-MXFP4:

```
FULL E2E PP ONLY: PASS, 64  tokens in 1399.7 ms ->  45.7 tok/s
FULL E2E PP ONLY: PASS, 256 tokens in 1718.8 ms -> 148.9 tok/s
```

GPU1 confirmed `active`/healthy after both runs.

Both points fit `time = 1.293s + N * 1.662ms`:
- ~1.29s fixed one-time overhead per run (cold-queue first kernel launch,
  not part of steady throughput -- amortizes away at longer prompts).
- **~602 tok/s marginal (steady-state) rate.**

vs. the ~1780-2100 tok/s roofline: **current implementation runs at roughly
1/3.5 of the theoretical ceiling.** This is the real number to chase, not the
raw 64-token or 256-token figures above, which are still overhead-dominated.

### Next steps
1. Get a longer-prompt point (512, 1024) to confirm the marginal rate holds
   and drop the fixed-overhead term further -- should approach the true
   steady-state pp more closely.
2. Given the ~3.5x gap and the FFN-is-DRAM-bound precedent from Ornith/Qwen
   (weights re-read ~19x/layer), profile the dense FFN GEMM
   (`load_xe2_dense_mxfp4_f32` dispatch in `prefill_muse`'s `mm` lambda)
   first -- it's 87% of FLOPs/token, so it is almost certainly the lever.
3. Compare against `GRIMOIRE_MUSE_PREFILL_GEMV=1` and
   `GRIMOIRE_MUSE_PREFILL_EXACT_BF16=1` (env flags already in
   `prefill_muse`) to isolate whether the MXFP4 dense-GEMM dispatch itself
   is the bottleneck vs. the surrounding per-layer glue kernels.
