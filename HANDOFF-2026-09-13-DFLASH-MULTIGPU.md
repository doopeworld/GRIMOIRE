# Session handoff — 2026-09-13, DFlash on more than one card, and MoVA packing

Branch `claude/codex-b70-audit-fixes-u1adcv` (mirrored to
`codex/b70-audit-fixes-20260911`).

Three things closed here, all verified on an OpenCL CPU device and none
of them on a B70. Rule 9's floor, not its ceiling: no XMX tile, no
batched prefill, no bridge, no OCuLink, and not one number.

## 1. DFlash under tensor parallel

The drafter is small, so it loads REPLICATED on every rank — nothing in
the DFlash block ever called the `shard` lambda, so this was already true
by construction and only the load gate said otherwise.

Two places read target state that TP shards, and both were silent:

- **The block embed.** `launch_embed_batched` indexes the embedding table
  by absolute token id, and TP shards that table over the VOCABULARY.
  Every rank read whatever row sat at that offset in its own slice — a
  plausible embedding of the WRONG token. It now goes through
  `Grimoire::embed_rows()`, which zeroes the rows this rank does not own
  and all-reduces, exactly as `embed_one` does for the single-token case.
  New kernel: `launch_embed_batched_shard`.
- **A shared `lm_head`.** A drafter that ships no head of its own decodes
  through the target's, which TP shards the same way. Two of the wide
  paths wrote a full-width logits row from `lm_head.w.N`, which under TP
  is one rank's fraction of the vocabulary. Under TP the head now goes
  through `gemv_any`, the one path that computes the local rows and
  all-gathers them. That is `M-1 <= 15` row projections per step, so the
  batched sharded head is a measurement to make on the card, not a
  correctness question.

Neither failure would have raised anything: the ranks would simply have
drafted different tokens, their KV caches would have stopped describing
the same sequence, and the output would have stayed fluent.

## 2. DFlash under pipeline parallel

Harder, and the reason DFlash was single-GPU until now. The drafter
consumes the residual stream tapped at several TARGET layers, and a
pipeline split puts those layers on different stages, so no stage can
build the concatenated row `fc` consumes from what it computes alone.

The protocol now carries them:

- Every stage resolves the tap set from the **drafter's own config**
  (`parse_dflash_config`), so no stage can disagree about which layers to
  tap. An earlier stage allocates the tap buffer and loads no weights.
- Each stage captures the taps it owns — the guard is now
  `dflash2.target_aux`, not `dflash2.ok`, because the stage that has the
  taps the last stage needs is exactly the one with no drafter.
- At each boundary the sender writes the hidden state and then the tap
  block for the same positions; the receiver reads them in the same
  order, overwrites its copy, fills in its own taps, and forwards again.
  By the last stage the union is complete.
- The last stage hosts the drafter, exactly as it hosts the MTP head, and
  pushes the drafted block backward on the channel `argmax_token` already
  uses (`pp_sync_tokens`), so every stage speculates on identical
  candidates and rolls back to the same position.
- `pp_connect` gained two more one-hop handshakes: the draft width and
  the tap count. A stage that disagreed about either would read a
  different number of floats off the socket than the previous one wrote,
  and would desynchronise every later message — including hidden states —
  with no error where it happened.

**Cost, unmeasured:** `n_taps * hidden` floats per token per boundary on
top of the hidden state. For an 8-tap drafter that is 8x the boundary
traffic. MTP sends nothing extra. Do not prefer DFlash on two cards
until both have been run on this box, same prompt, on accepted-per-step
AND tok/s.

**Not exercised off the card:** the tap forwarding in the BATCHED prefill
path. A device without matrix hardware refuses batched prefill entirely,
so only the per-token forwarding ran here.

## 3. MoVA value projection, experts packed expert-major

`DAY-ONE.md` asked for this before quoting a K2 prefill number. The
bring-up path brought the routing table back to the HOST once per sparse
layer to decide which expert each token needed — at 4096 tokens over 45
sparse layers, the readback is what made K2 prefill unusable, not the
arithmetic.

The E experts are now concatenated along N into one `[E*N][K]` weight, so
expert `e`'s output row `n` is global row `e*N + n` and `QuantWeight`'s
own indexing resolves payload and scales with **no new decode logic** —
the packed weight is simply a taller matrix of the same format, and
`at()` on it is the same reference every kernel already has to match.
`launch_mova_value_packed` then reads `rex`/`rwt` on the device and never
tells the host what it found. One work-item owns one (token, row) pair
and walks its own top-k, so the accumulation is private: no atomics, and
a summation order fixed by `j`.

Same bytes as the E separate weights — a different layout, not a second
copy. Exactly one of `v_experts_packed` and `v_experts` is populated.

**Not used under TP**, which shards each expert's rows across ranks; a
row shard of a matrix concatenated along N would cut across expert
boundaries. The per-expert path stays for that case.

`GRIMOIRE_MOVA_PER_EXPERT=1` restores the old path, for the A/B on the
card.

## What is verified, and how

**Status at the time this file was committed:** `test_k2_kernels` and
`test_k2_e2e` had run and passed against the final build.
`test_model_matrix`, `test_parallel_e2e` and `test_spec_e2e` were still
running on a 4-core OpenCL CPU device, where each spawns many processes
and a full pass takes the better part of an hour. If the commit that
follows this one does not record their results, they did not finish --
**re-run them before trusting anything below that they cover.**

- `bin/test_k2_kernels` — the packed kernel against the host reference
  through `QuantWeight::at()`, in **all seven projection formats**,
  including a deliberate out-of-range route the kernel must skip. The
  expert-major index walks the scale array differently per format, and
  getting it wrong reads another expert's scales while still producing
  finite, plausible values.
- `bin/test_k2_e2e` — loads a K2 checkpoint and generates BOTH ways in
  one process. Packed output must be token-identical to per-expert.
- `bin/test_parallel_e2e` — its K2 case A/Bs the two MoVA paths across
  processes for free: a single process packs, TP does not.
- `bin/test_spec_e2e` — DFlash single-process, under TP and under PP,
  with a drafter forced to non-zero acceptance so both the accept and the
  rollback paths run. The mini drafter taps target layers 0 and 2 of a
  4-layer target, so a 2/2 split lands one tap on each stage: drop the
  forwarding and the test fails rather than quietly drafting on a row
  that is half zeros.

## Still open, and why none of it could be done here

- **Batched prefill under TP.** TP declines the batched path outright, so
  a TP run processes its prompt a token at a time. Making it real means
  all-gathering every projection over the whole token batch and an
  expert-parallel all-reduce for the routed half. The batched path is XMX
  end to end and a device without matrix hardware cannot run one of its
  kernels, so this is work to do ON the Tower.
- **F07**, single-process `GRIMOIRE_PIPELINE`: every kernel on device 0's
  queue. Architectural, and its whole point is device placement, which
  needs two cards.
- **Single-process cross-device.** Rule 7 — a hardware experiment.
- **Everything from `HANDOFF-2026-09-13-AUDIT.md`'s closing section**,
  unchanged: no GPU, no AOT image, no bridge execution, no OCuLink, no
  real model, no generated text read, no throughput number. Agnes has
  still never been loaded against the real checkpoint. The only DFlash
  acceptance figure on record is void — it was measured on a drafter with
  the wrong RoPE theta, a hardcoded head_dim and epsilon, invented
  sliding windows and the target's output head.

## First thing to do with this on the box

`tools/preflight_b70.sh /models/<dir>`, then read the `capabilities:`
block. It now names the DFlash mode explicitly — replicated under TP, or
"drafter on the last stage; taps forwarded from every stage" under PP —
because a silently disabled drafter looks exactly like a slow model.
