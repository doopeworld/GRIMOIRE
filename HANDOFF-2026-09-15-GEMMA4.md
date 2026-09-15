# Handoff — 2026-09-15 — Gemma-4, and three engine bugs it uncovered

Session on branch `claude/codex-b70-audit-fixes-u1adcv` (mirrored to
`codex/b70-audit-fixes-20260911`). Range `49421d0..3dbed59`, 9 commits.
All 7 gates green on an OpenCL CPU device. **No B70 was involved — rule 8
applies to every line below.**

Read `CLAUDE.md` rules 1–12 first. Rules 10, 11 and 12 were written this
session and each one cost real debugging.

---

## 1. What the session was for

Add `google/gemma-4-31B-it` to GRIMOIRE "full config as usual: mtp/dflash2
and pp/tp". The architecture write-up is `GEMMA4-2026-09-14.md` — read it
before touching gemma-4, it lists eleven items and which are done.

It turned into three things:
1. a structural refactor the rest of the engine needed anyway (per-layer
   attention geometry),
2. the gemma-4 forward path,
3. **three pre-existing engine bugs that had nothing to do with gemma-4**,
   found only because gemma-4 was the first model to exercise them.

---

## 2. The three engine bugs — the most valuable part of this session

These were NOT introduced here. They were latent and would have hit
whatever model reached them first.

### 2.1 `tie_word_embeddings` was parsed and never used

`cfg.tie_embeddings` was read from `config.json` and consumed by NOTHING.
A tied checkpoint ships no `lm_head` tensor, so the upload block was
skipped, the loader printed `ok` anyway (the `printf` sat outside the
`if`), and `lm_head` stayed a default `DevQuant` with `N == 0`.
`gemv_any` wrote nothing into `s.logits`; argmax read stale device memory.

**Symptom: the same token every step, for every prompt.** In vocabulary,
correct length, byte-identical across requests. `test_model_matrix` passed
it 7/7, because every property it checked is TRUE of a model whose logits
never change.

Fixed in `bb5f974` (alias the embedding table, don't copy it) and
`fb6f209` (under PP the LAST stage needs that table — it runs the
projection — and the VRAM-saving release must not free a buffer `lm_head`
aliases, which would be rule 1's DEVICE_LOST).

**The durable part is the test, not the fix:** `test_model_matrix` now
asserts that two different prompts give two different continuations. That
is the only check in it a dead output projection cannot satisfy. See rule 12.

### 2.2 head_dim above 256 overruns a device stack array

Every flash kernel in `attention.cpp` and `prefill.cpp` accumulates one
head into a PRIVATE array of `MAX_DPL`(16) floats per lane at `SG_SIZE`
16 — so 256 is the widest head they can hold. Four separate
`constexpr int MAX_DPL = 16` said so, none where a caller would look.

A wider head writes past the end of device stack memory: **DEVICE_LOST and
a power cycle, not a wrong number.** No supported model had exceeded it
(Qwen 128, K2/Muse 256). Gemma-4's full-attention layers are **512**.

`MAX_DPL` and `MAX_HEAD_DIM` now live in `src/kernels.hpp` with a
`static_assert` at each former site, and `Grimoire::unsupported_reason()`
refuses by name. **This is the one blocker that needs the Tower — see §5.**

### 2.3 A failed graph capture left the sequence position corrupted

`build_graph()` moves the device position, device length and host counter
to a scratch value, records, and restores them. The `catch` restored none
of it — and the saved values were declared INSIDE the `try`, so they were
out of scope exactly where they were needed. Any throw in `forward()` left
the next real decode starting from the scratch position, silently.

Gemma-4 made that throw likely: its `k_eq_v` value copy called `.wait()`,
and a host-side wait while a SYCL graph is recording throws. The queue is
in-order, so the wait was never needed. Both halves fixed in `3dbed59`.

---

## 3. What was built for gemma-4

**Per-layer attention geometry (`5282524`) — the structural one.**
`head_dim`, KV heads and RoPE moved from model-wide config into per-layer
`LayerDev`, resolved once for ALL layers right after `L.resize()` (not
only the ones a PP rank uploads, so a later whole-vector walk cannot read
a zero as a fact). ~99 call sites. `forward()`, `forward_dag()`, the
batched `prefill()` and both MTP paths read `LayerDev` now; shared
allocations size from `max_head_dim()` / `max_kv_heads()`.

For every existing architecture the per-layer and model-wide values are
identical, which is what makes `test_model_matrix` and
`test_parallel_e2e` a real regression check on the substitution.

It also closed a pre-existing overrun: the bf16 attention path packs
`q|k|v` back to back into `xperm`, needing
`M*(n_heads + 2*kv_heads)*head_dim`, but the buffer was sized from `M*W`
where `W` is the widest SINGLE projection — a bound on each, not on the sum.

**`forward_gemma4()` (`3054b4d`)** implements the Gemma sandwich, which is
NOT `forward()`'s residual graph:

```
h   = x + post_attention_layernorm(attn(input_layernorm(x)))
out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
out *= layer_scalar
```

It IS `forward_muse()`'s graph, but `forward_muse` carries Muse-isms
(query_prescale, scaleless embed norm, f16 KV caches, NoPE layers); making
those conditional would put a working, tested path at risk for nothing.

Plus: `k_eq_v` (V is the `k_proj` output taken BEFORE `k_norm` and BEFORE
RoPE), GeGLU, sliding window (narrows the scanned range rather than
masking), embedding scale, `layer_scalar`, logit softcap, proportional
RoPE with its `factor` divisor.

**Kernels added:** `launch_geglu`, `launch_geglu_batched`,
`launch_rope_proportional`, `launch_qk_norm_rope_proportional_batched`,
`launch_scale`, `launch_logit_softcap`. Every one is pinned in
`bin/test_k2_kernels` against a host reference AND against the kernel it
could be confused with (GeGLU vs swiglu: 0.77 apart; proportional vs
partial_rope: 4.05 apart; batched vs decode rotation: 0.000e+00, i.e.
byte-identical). A "matches" that cannot distinguish proves nothing.

---

## 4. Two external audits — 14 findings, all real

Both audits were commissioned by Ian and pasted back. **Every finding was
verified against `ref/gemma4.py` before anything was changed. None was a
false positive.** Details in `GEMMA4-2026-09-14.md` under the progress log.

The two most instructive:

**Gemma-4 does NOT use `(1 + w)` in its norms; Gemma-2 and Gemma-3 do.**
`Gemma4RMSNorm.forward` is `normed * self.weight` (`ref/gemma4.py:211`).
The wrong convention is exactly what the family name leads you to, and it
shifts every normalised activation while leaving output fluent. Knock-on:
a scaleless norm is a ZERO weight under `(1 + w)` and a ONES weight under
plain `w` — get the convention right and the placeholder wrong and the
model multiplies its values by zero. Rule 11.

**`gelu_new` is the TANH approximation, not exact-erf.** It maps to
`NewGELUActivation`. A confident comment here asserted the opposite and
refused a checkpoint this engine runs exactly. Plain `gelu` IS erf and
stays refused. A comment asserting what an external identifier means is
worth exactly the reference lookup behind it — that one had none.

---

## 5. What is LEFT — and what only the Tower can do

### 5.1 Gemma-4 at its real head_dim — **DONE (2026-09-15)**

`google/gemma-4-31B-it` no longer refuses.  The four flash kernels
(`launch_flash_decode`, `launch_flash_decode_batched`,
`launch_flash_prefill`, `launch_dflash2_block_attention`) are TEMPLATES on
their accumulator width and are instantiated twice: head_dim <= 256 keeps
the 16-slot kernel it has always used, and only a wider head reaches the
32-slot one.  `unsupported_reason()` refuses above 512 now, not above 256.

**The reason this section originally gave for deferring was wrong, and it
came from `b686d55`'s own commit message.** That message said
"attention.cpp's accumulator loops run to MAX_DPL unconditionally --
unlike prefill.cpp, which guards each slot with `if (j < dpl)`".  They do
not.  The guards are in BOTH files and PREDATE that commit, which changed
four lines, each a `constexpr` -> `static_assert` swap:

    git show b686d55^:src/attention.cpp | grep -c 'd < dpl'   ->  6
    grep -c 'd < dpl' src/attention.cpp                       ->  6

So "bound attention.cpp's loops by dpl first" was work that did not exist.
What actually scaled with the constant was `float acc[MAX_DPL]` -- the
private array is sized by the constant, not by the head -- so a plain bump
would have doubled it for every model including Qwen at 128.  Templating
is what confines that cost to the head that needs it, and it is why no
existing model needs re-measuring.

Verified off-card, on an OpenCL CPU device, at `28d1b30`:

- `bin/test_model_matrix` **ALL PASS (0 failed cells)**.  `gemma4-hd512`
  is a full row -- head_dim 512 loads, generates, stays in vocabulary,
  reproduces and depends on its input, across all 7 projection formats --
  and the refusal case moved up to head_dim 1024, which still refuses by
  name.
- `bin/test_spec_e2e` **ALL PASS (0 failures)**; MTP and DFlash identical
  to plain decode, single process, TP and PP.
- `bin/test_k2_kernels` **ALL PASS (0 failures)**.
- The five pre-existing rows are unchanged and green.

**STILL TOWER WORK (rule 8):** register pressure and throughput of the
32-slot instantiation on a real B70.  Nothing here is a number.  Two
things to look at when the card is up, neither of which a CPU device can
show:

1. **Occupancy at dpl 32.** The wide kernel holds 32 floats per lane
   instead of 16.  Only gemma-4 pays it, but measure it there.
2. **SLM, which is a HARD limit, not a gradient.** The three batched
   kernels take `ks` and `vs` local accessors sized `head_dim * KT` with
   KT 16, so at head_dim 512 that is 32 KB + 32 KB = **64 KB of SLM per
   work-group** -- at or over a Xe work-group's budget.  The CPU device
   used here reports 256 KB and so cannot fail this, which is exactly why
   it proves nothing about the B70.  If a batched gemma-4 launch fails
   for local memory, that is this, and the fix is tiling `KT` down for
   wide heads rather than touching the accumulator.

### 5.2 Gemma-4 batched prefill — not written

`prefill()` returns false for gemma-4 and falls back to sequential decode:
correct, and slow. The batched loop is the Qwen residual graph, so running
a gemma-4 prompt through it would contradict `forward_gemma4()` token for
token while still looking fluent. Batching it needs the sandwich graph,
the `k_eq_v` value and both rotations in batched form. A task, not a flag.

### 5.3 Gemma-4 speculation — untested, not blocked

No MTP head exists in this checkpoint (`tie_word_embeddings`, no `mtp.*`
tensors). Speculation must go the DFlash route, and the drafter exists:
**`z-lab/gemma-4-31B-it-DFlash`**. `tests/test_dflash_config.cpp` already
carries a fixture named after it. What is NOT done: nobody has run DFlash
against a gemma-4 target. HuggingFace was unreachable from the work
container all session, so the drafter's `config.json` and weight map were
never seen. One load on the Tower settles it — read the four `dflash`
banner lines, every value in them is silent when wrong.

### 5.4 Everything else that was already open

Unchanged from `DAY-ONE.md` §6: batched prefill under TP, F07
single-process `GRIMOIRE_PIPELINE`, single-process cross-device on
OCuLink, the AOT image, the cutlass bridges, and **every number**.

---

## 6. How to verify what this session claims

```bash
# toolchain: TOOLCHAIN-IN-A-CONTAINER.md, ~10 min, no GPU needed
make test && make test-correctness
./bin/test_k2_kernels ./bin/test_k2_e2e ./bin/test_model_matrix
./bin/test_parallel_e2e ./bin/test_spec_e2e
```

All 7 pass at `3dbed59` on an OpenCL CPU device, including gemma-4 under
PP and TP at 2, 3 and 4 ranks in BF16 and FP8, token-identical to a single
process.

**On the Tower, start at `DAY-ONE.md` and run
`tools/preflight_b70.sh /models/<dir>`.** It builds in the right order
(bridges first, rule 3), runs every gate, and finishes by generating text
for a human to read.

### Two traps that cost this session real time

- **The parallel gate takes ~60 minutes on a CPU device.** It is not hung.
  It spawns ~50 multi-process runs on a 4-EU device.
- **Never run two gate scripts into the same output directory.** Two runs
  overlapping produced a truncated log and a spurious `rc=1` that looked
  exactly like a code failure, and cost an hour of chasing. One run, one
  directory; check with `ps -eo pid,comm` before starting another (never
  `pgrep -f`, which matches its own command line).
