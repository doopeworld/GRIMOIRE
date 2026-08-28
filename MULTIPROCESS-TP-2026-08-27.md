# Multiprocess TP status — 2026-08-27

## Functional milestone

Two-rank projection tensor parallelism works on the direct plus USB4 B70 pair.
Both processes keep the complete checkpoint for this correctness-first version.
For every decode projection with an even output-row count:

1. rank 0 computes the first half of output rows;
2. rank 1 computes the second half;
3. the ranks exchange their halves over the rank socket;
4. both reconstruct the exact full output tensor before the next operator.

This is real tensor computation sharding, not pipeline or data parallelism. It
is not yet storage-sharded: model weights and routed experts remain replicated,
and batched prefill still uses replicated computation. Those are capacity and
performance follow-ups, not blockers for functional TP.

## Evidence

- `GRIMOIRE_TP_RANK=0|1`, separate Level Zero processes and devices.
- Both ranks load, connect, and exit 0.
- Prompt `Hello`, four greedy tokens: both TP ranks and the single-GPU baseline
  emit `The user just said`.
- Four-token USB4 smoke result: 60.9/61.0 tok/s. This is not a performance
  target; the external link will move to OCuLink PCIe Gen4 x4.
- Post-test vLLM-style device-count check reports both B70s healthy.
- SYCL decode graph is disabled because host socket collectives cannot be
  captured.

## Launch

Use `tools/tp2run.sh`. It carries the same proven privileged `/dev/dri`, host
IPC, 10 GiB SHM, by-path mount, `ZE_AFFINITY_MASK=0,1`, spawn, CCL/OFI, and
bridge-library environment as the working dual-XPU setup.

## Remaining TP optimization (not required for functional acceptance)

- Load only each rank's projection rows instead of retaining full weights.
- Shard routed/shared expert tensors and use reduce-scatter/all-reduce at the
  mathematically appropriate row-parallel boundaries.
- Extend tensor sharding to batched prefill.
- Replace the correctness socket collective with oneCCL/XCCL after OCuLink.
