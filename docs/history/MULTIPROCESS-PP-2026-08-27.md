# Multiprocess PP status — 2026-08-27

## Settled result

Two-process pipeline parallelism is functional on the direct B70 plus the
USB4-attached B70. The production split is 24,16: rank 0 owns transformer
layers `[0,24)` and rank 1 owns `[24,40)`. Each rank has its own Level Zero
process/context. A Unix socket transfers the materialized hidden stream once
at the stage boundary and returns the selected token to rank 0.

The default single-GPU path remains opt-out and unchanged. Multiprocess PP is
enabled only by `GRIMOIRE_PP_RANK=0|1`; `tools/pp2run.sh` launches both ranks.

## Required container/device topology

Use the exact known-good vLLM topology in `tools/pp2run.sh`:

- `--ipc=host --privileged --shm-size=10g`
- `--device /dev/dri:/dev/dri`
- `-v /dev/dri/by-path:/dev/dri/by-path`
- common `ZE_AFFINITY_MASK=0,1`
- spawn-style separate rank processes; rank selects local device 0 or 1
- CCL/OFI and vLLM compatibility environment retained for the future TP path

Do **not** run `ZE_AFFINITY_MASK=1` alone on the USB4 card. It caused a Level
Zero segmentation fault and the external B70 required a server reboot. Also do
not use the old single-process `GRIMOIRE_PIPELINE`; its shared-context device-1
allocations fail on USB4.

## Correctness and capacity evidence

- Both ranks load and connect: rank 0 = 24 layers / 11.22 GiB, rank 1 = 16
  layers / 7.88 GiB.
- Prompt `Hello`, four greedy tokens: PP and single GPU both emit
  `The user just said`.
- Production bridge environment plus deferred-MoE boundary: same text on both
  ranks, both exit 0.
- Safe staged prefill-only mode:
  - 128 tokens: PASS on both ranks.
  - 1024 tokens: PASS on both ranks.
  - 4096 tokens: PASS on both ranks; rank-1 end-to-end completion 1577.5 ms.
- After every accepted test, exact vLLM device-count probe still reports two
  B70s.

Current USB4 speed is not an optimization target. The external link will be
replaced by OCuLink PCIe Gen4 x4. Functional correctness and safe process/device
ownership are the acceptance criteria until then.

## Safe benchmark path

Use `--prefill-only N` for a single real full-model prefill. It skips the
standalone projection/MoE diagnostics in `grimoire_load_report()`.

The old no-prompt benchmark must not be used for dual-rank USB4 validation: its
independent M=4096 projection diagnostic produced `UR_RESULT_ERROR_DEVICE_LOST`
on rank 1 before staged prefill began. The GPU remained visible afterward, but
the diagnostic is irrelevant to PP and unnecessarily risky.

## Implementation notes

- `Grimoire::rank_device()` mirrors vLLM local-rank device selection.
- Decode rank 0 materializes the final routed/shared FFN residual and sends H
  floats; rank 1 receives it, zeros pending residual buffers, and runs its
  local layers plus final norm/head.
- Batched prefill sends M×H floats once. Pinned host staging grows on demand.
- Deferred MoE gather is materialized at the boundary; rank 1 treats its first
  local layer as a new stage, not as layer 24 following an in-process layer 23.
- Rank 1 sends greedy token IDs back to rank 0 so both generation loops remain
  lockstep.
- Decode and prefill graph capture are disabled in PP because socket I/O cannot
  be captured in a SYCL graph.

## Rollback

Unset `GRIMOIRE_PP_RANK` and run the normal single-GPU launcher. The PP code is
fully gated. The pre-work archive remains:
`grimoire-FULL-BACKUP-20260827-1510.tar.gz`
with SHA256
`7f6f23e12f2a352f3c01ddf32f3e5670bb2f6576b2ce54df19f33a99d113c32b`.
