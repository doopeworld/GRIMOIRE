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
