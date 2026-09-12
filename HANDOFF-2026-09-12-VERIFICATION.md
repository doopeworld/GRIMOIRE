# Session handoff — 2026-09-12, verification and dual-GPU speculation

Branch `codex/b70-audit-fixes-20260911` (mirrored to
`claude/codex-b70-audit-fixes-u1adcv`).

Read `DAY-ONE.md` first if you are at the Tower. This file is the record
of what changed and why, and what is still open.

## The thing that made the rest possible

This project had been handing off uncompiled code for weeks on the belief
that a work container cannot build it. That belief was wrong: a full
oneAPI DPC++ toolchain plus an OpenCL CPU device installs from
conda-forge in about ten minutes, no GPU involved.
`TOOLCHAIN-IN-A-CONTAINER.md` is the procedure; CLAUDE.md rule 9 is the
rule. Every finding below came from actually compiling and running the
code, and none of them were findable by reading it.

## Bugs found by compiling

1. **`src/grimoire.cpp` did not compile.** A redefinition of `derr` in
   the draft-architecture validation.
2. **`launch_softplus_gate` was 2.5e-4 off torch** — `log(1+exp x)`
   where the reference uses `log1p(exp x)`; the `1.0f +` discards the
   low bits for a negative gate. Fixed to 1.98e-7. Same pattern fixed in
   the DeltaNet gates in both `ops.cpp` and `prefill.cpp`.

## Bugs found by RUNNING the kernels

3. Nothing else — the four K2 kernels matched their references once the
   softplus was fixed. `bin/test_k2_kernels` is the standing gate.

## Bugs found by running the ENGINE (the K2 path had never executed)

4. **K2 dense layers were sent looking for a MoE router.** `cfg.is_moe()`
   is a property of the model; being routed is a property of the layer,
   and K2 is the first architecture where those differ.
   `LayerDev::moe_layer` now decides per layer.
5. **A 46 KB device heap overrun, three times per token.** `s.sh_g` was
   sized from the shared-expert width while K2's dense layers put a full
   `intermediate_size` MLP in `sh_gu`. On a B70 that is DEVICE_LOST.
6. **Batched prefill was silently off for every model without DeltaNet
   layers.** `alpha`/`beta` are sized `gdn_tokens * Hv`, and `Hv == 0`
   makes the zero-byte allocation return null, which the null sweep read
   as failure. Correct output, ~50x slower prompt processing, behind one
   line on stderr. Not K2-specific.
7. **The allocation-failure diagnostic named the wrong buffer** (the
   names table was missing `bn_bf`, so everything from index 11 on was
   off by one). Now `static_assert`ed against the vector it describes.

## Bugs found by running two PROCESSES

8. **Tensor parallel wrote past its own scratch.** The fix for (5) used
   `sh_gu.w.N`, which under TP is this rank's SHARD, while `gemv_any`
   all-gathers the FULL row into that buffer. Every FFN overran by
   `(world-1)/world`. Dense models survived by luck.
9. **MoVA under TP** cleared and accumulated the shard width while the
   gather wrote the full row, leaving the top of every value vector
   stale. Silent — every shape still lines up.

## Bugs found by running SPECULATION

10. **A MoE checkpoint with an MTP head could not load.**
    `shared_expert_gate` is optional (the next line says so) but was
    uploaded unconditionally through the missing-tensor path.
11. **A failed batched verify killed the request** instead of degrading.
    Now redone sequentially — `prefill()` returns false only before
    submitting work, an invariant now stated at the function and modelled
    by the host test's fake engine.
12. **Speculation changed the output on DeltaNet models.** Found the
    first time the hybrid architecture was covered, at every draft depth
    including K=1: ten tokens identical, then permanently divergent.
    A rejection replays the recurrent state from buffers only the
    BATCHED verify captures, so the sequential fallback restored state
    that was never saved. Speculation is an exactness claim, so the fix
    is to REFUSE: `spec_verify_available()` requires the batched verify
    whenever the model has linear-attention layers, and the load banner
    says when it is off.

## What was added

- **Speculation on dual GPU.** It used to be hard-disabled under TP and
  PP. TP: the MTP head loads replicated, and two TP-specific bugs were
  fixed (the sharded embedding table, and the reduced-draft-vocab lm_head
  shortcut skipping the all-gather). PP: the last stage hosts the head,
  drafts, and pushes draft and verified tokens backward; the stages agree
  via a handshake in `pp_connect`, and the last stage loads the embedding
  table because the head embeds what it drafts. See CLAUDE.md rule 7.
- **K2 prefill no longer stalls per token.** `mova_value_batched` routes
  the whole batch and reads the table back once instead of two host round
  trips per token — ~368,000 stalls removed from a 4096-token prompt.
- **Five gates that need no model and run in seconds**, all wired into
  `build_b70.sh` and `tools/preflight_b70.sh`:

  | gate | what it pins |
  | --- | --- |
  | `test_k2_kernels` | each new kernel vs its host reference, on a device |
  | `test_k2_e2e` | the K2 engine path loads and generates |
  | `test_model_matrix` | 4 architectures x 7 projection formats |
  | `test_parallel_e2e` | PP and TP equal one process, token for token |
  | `test_spec_e2e` | speculation equals plain decode, incl. TP and PP |

- **`tools/preflight_b70.sh`** — one command for the Tower, in the right
  order, refusing to continue past a failed build, ending by generating
  text for a human to read.

## Verified here (OpenCL CPU device, no XMX)

```
test_k2_kernels     ALL PASS   grouped norm, softplus, router, MoVA,
                               topk16 over a padded stride, selector
                               edges 0.000e+00, path walk exact,
                               prefill vs decode norm convention 0.000e+00
test_model_matrix   ALL PASS   28/28: dense / moe / hybrid / k2-horizon
                               x bf16 fp8_e4m3 fp8_e5m2 int8 int4 mxfp8 mxfp4
test_parallel_e2e   ALL PASS   7 cases, PP and TP token-identical,
                               including DeltaNet state across a split
test_spec_e2e       ALL PASS   dense and moe, bf16 and fp8, identical at
                               K=1,2,3,5, single process + TP + PP;
                               hybrid correctly REFUSES here
make test                      11 suites
make test-correctness          2 suites
bin/grimoire, bin/grimoire-server   compile and link
```

## NOT verified anywhere — the Tower's job

- **The batched prefill path has never executed.** No XMX in the
  container, so every gate above ran the sequential decode path.
  Prompt processing uses the batched path. Treat the first long prompt on
  the card as that path's first run.
- **The batched speculative verify on a recurrent model** — same reason.
  That is the path bug (12) says must be used for hybrids, and it is the
  one Ian's verified 49.8 TG Qwen MTP recipe exercises.
- Every XMX tile, the AOT `bmg_g31` image, the bridges, the OCuLink link,
  and every throughput number.

## Still open, in the order worth doing

1. **Pack the MoVA experts expert-major** so the routing readback
   disappears entirely. The MoE kernel cannot be reused — it is gate|up
   SwiGLU shaped, MoVA is one matrix per expert — so this needs a new
   format-templated GEMV that indexes the expert on device. Biggest
   remaining K2 prompt-processing win.
2. **Batched prefill under TP.** TP currently falls back to
   token-at-a-time for prompts and says so in the banner. PP does not
   have this problem, which is why PP is the recommended dual-GPU path.
   This also removes the TP speculation restriction from (12).
3. **DFlash under TP/PP.** Its drafter reads aux hidden states from
   target layers that PP puts on different ranks, and its batched embed
   path is not TP-aware. MTP covers dual-GPU today.
4. **F07** — single-process `GRIMOIRE_PIPELINE` runs every kernel on
   device 0's queue. Numerically correct, but GPU1's weights cross the
   link every token. Architectural: a per-layer queue threaded through
   every launcher. Use the multiprocess launchers instead.
5. **Prefix cache is disabled whenever speculation is on**
   (`save_prefix` bails on `mtp.ok || dflash2.ok`). Worth revisiting for
   the server.
