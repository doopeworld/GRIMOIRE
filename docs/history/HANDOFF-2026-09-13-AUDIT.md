# External audit of this branch, and the eight fixes

Date: 2026-09-13. Branch `codex/b70-audit-fixes-20260911`
(mirrored to `claude/codex-b70-audit-fixes-u1adcv`).

An independent audit reviewed `1af8b7e~1..e844fe2` (48 commits) and
returned eight actionable findings, four of them P1. **Its verdict was
"do not certify this branch for B70 use yet", and that verdict still
stands** — every fix below was made and verified on an OpenCL CPU
device. Nothing here has touched a GPU.

All eight are fixed in `02aebae`. Each was reproduced or read back
independently before being changed; none was taken on the auditor's word.

## The four P1s

**F1 — packed INT4 read as if unpacked, overrunning its payload.**
The loader marks compressed-tensors INT4 and rewrites the tensor's
LOGICAL shape to `[N][K]` while the payload still holds `K/8` int32
words per row. `read_matrix_f32` handled `native` and `gptq` but had no
`compressed_int4` branch, so it asked `read_f32` for `N*K` values from a
buffer holding `N*K/8` — a heap overread, with the group scales ignored
on top. Dormant until `c6976ac`: every earlier caller happened to hold a
format that never reached it, and the Agnes FFN fold reaches it. Now
decoded through `read_compressed_int4_ref` and `QuantWeight::at()`, the
reference dequant the GPU tiles must match bit for bit.

**F2 — a declared parallel FFN could silently vanish.** The uploader
folds only when all three parallel projections resolve, and otherwise
takes the ordinary gate/up/down branch. `dense_inter` had already been
widened to the folded size, so nothing downstream would notice: the
engine would simply run a narrower FFN than the checkpoint has. A
declared `parallel_ffn` now requires all three tensors in every layer and
checks their widths against `parallel_ffn_intermediate_size`, naming the
layer and the tensor.

**F3 — deferred MoE gathering assumed an all-routed model.** It was
gated on `cfg.is_moe()`, which is a property of the MODEL: K2 is a MoE
model whose first layers are DENSE. With the bridges present, `M>=32`
and `GRIMOIRE_DEFER_MOE_GATHER=1` — which `tools/nrun.sh` sets — layer 0
writes `r0` and layer 1 un-permutes routing buffers the dense layer never
filled, dropping layer 0's real output and reading stale route indices.
The fused norm it uses is also whole-width `(1 + weight)`, not K2's
grouped direct-weight convention. Now requires every layer this process
executes to be routed, and excludes K2 outright.

**F4 — the speculative fallback could commit a prefix from images
nothing wrote.** `spec_verify_available()` answers for the DEVICE, not
for the call: a per-call scratch failure still lands in the sequential
fallback, where `commit_spec_prefix` replays `spec_dn_steps` /
`spec_conv_inputs` that only the batched verify fills. The comment in
`generation.hpp` asserting this was unreachable was wrong. A recurrent
model whose batched verify declines now degrades that round to a plain
step — nothing has been submitted at that point — and the round is not
counted, because its draft was never judged. `tests/test_generation.cpp`
covers it, and reverting the fix trips the new assertion.

## The four P2s

**F5 — the launcher validated render node names and then ignored them.**
`ZE_AFFINITY_MASK` was built from the node COUNT, so asking for
`renderD129,renderD130` on a three-card box silently ran on
`renderD128,renderD129`. The mask now comes from each node's position
among the present `renderD*` nodes, duplicates are rejected, and it is
printed. On the engine side, an explicitly ranked process whose device
does not exist no longer falls through to the default GPU selector —
every rank would have picked the same card while the collectives still
connected, which looks like a working run. It fails where the cause is
visible, and the rank banner carries the PCI address where the driver
exposes one.

**F6 — preflight could finish clean with gates that were never built.**
Device-test compile failures are warning-only in `build_b70.sh`, and a
missing binary was reported as a skip. Required gates now fail.

**F7 — a failed container reported shell success.** Ending on the `else`
branch returns that `echo`'s status. Fixed in `nrun.sh` and in
`pp2run.sh` / `tp2run.sh`, which had the same block.

**F8 — the shared block table was sized from the draft page alone.** It
is shared with the target's paged attention, which pages at 64 and
indexes `ceil(max_seq/64)`; safe only while the draft page is <= 64, and
a configured 128 undersized it. Sized for both consumers now.

## What the audit settled that this session could not

huggingface.co is blocked from the work container. The auditor fetched
both files and checked them against the code:

- `modeling_agnes.py` (`AgnesMLP`) confirms two bias-free FFNs taking the
  same input with their outputs summed — which is what makes the fold
  valid — and their fixture drove the REAL helpers, confirming the
  gate/pgate/up/pup row order and the per-row `down` interleave.
- The real `z-lab/Qwen3.5-35B-A3B-DFlash/config.json` passes the
  production resolver for all asserted fields. That checkpoint's explicit
  `rope_theta` is 1e7, so the 1e6 default is not used for it.

They also corrected a number quoted here: the K2 scratch shortfall is
**42 KiB**, not the "46 KB" in the comment at `src/grimoire.cpp`.

## Verified after the fixes (still CPU device only)

```
make test / make test-correctness   13 + 2 suites
test_k2_e2e        ALL PASS      (the path F3 touches)
test_model_matrix  ALL PASS      28/28
test_parallel_e2e  ALL PASS      14 two-rank + 16 three/four-rank matches
test_spec_e2e      ALL PASS      MTP exact single/TP/PP; DFlash exact at
                                 M=4,16 with shared and own head;
                                 hybrid correctly refuses
```

## Unchanged, and still the whole job

No GPU, no XMX, no batched prefill, no AOT image, no bridge execution, no
OCuLink. No real model loaded and no generated text read (rule 8). No
throughput number for anything. Ornith DFlash acceptance is still 2.21
against the reference's 6.1-7.7. Agnes has never been loaded against the
real checkpoint and its MTP head is unvalidated. `nrun.sh` has never run
Docker.

**First thing to watch on the box:** F5 changed which physical GPU each
rank gets, and that change has been reasoned about, not run. The rank
banner now prints the card and its PCI address — read it before
believing any multi-GPU result.

The auditor's recommended order was: fix F1-F4, repair failure
propagation and card selection, then run preflight on the Tower. The
first two are done.
