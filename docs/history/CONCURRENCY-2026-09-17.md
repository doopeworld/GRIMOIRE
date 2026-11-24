# Serving several agents at once (2026-09-17)

Read `PERF-2026-09-17.md` first if you have not: it is the measurement
that started this. Its headline was that **the server is serial**, so
every batch-side optimisation in the AMD stack it compared against
bought nothing here. This is the work that removes that.

Everything below is verified off the card. **No speed number exists**
and none can be taken from this container — the saving is XMX-shaped and
the off-card batched GEMM is a plain-SYCL fallback far slower than the
tile it replaces. Rule 8 is unchanged. What to measure on the Tower is
at the bottom.

---

## What was wrong

Two separate costs, and they were often confused with each other.

**One.** Every turn of a conversation re-read the whole history. The
prefix cache demanded a byte-identical prompt, which a growing chat
never gives — so it fired on a repeat and never on a conversation.

**Two.** `grimoire-server` held one mutex for the length of a request.
The second caller waited for the first to FINISH, not for the card. Its
own header said so. For agentic work that is the wrong unit of sharing
entirely: a decode step reads every active weight to produce ONE token,
and reads the same weights to produce eight.

## What changed

**Conversations resume.** `prefix_reuse()` finds the longest slot whose
tokens are a prefix of this request and restores to that length; the
request then processes only the new tokens. `generate_tokens` snapshots
at the END of a request, covering prompt AND reply, so the next turn is
a strict extension of it.

**There are N of them.** One snapshot meant two agents wiped each
other's every turn and both fell back to re-reading — the exact cost the
cache exists to remove, reintroduced by the second caller. Sized by
`GRIMOIRE_SEQ_SLOTS`, 1 by default.

**Resuming stopped copying the cache.** It used to memcpy every layer's
whole KV cache, sized by `max_seq` rather than by the conversation, in
both directions. The cache is now allocated `n_seq_slots` deep and
`k_cache`/`v_cache` are a VIEW into the bound slot; switching is
`bind_seq_slot()`, O(layers) pointer writes. The recurrent state is
still copied, deliberately: a KV row beyond `pos` is dead, but the conv
ring and the DeltaNet state are order-dependent and a prefill that
submitted work and then failed has advanced them.

**Batched decode.** A batched decode is a prefill whose M rows belong to
M DIFFERENT conversations rather than to consecutive positions of one.
That is why it is a parameter on `prefill()` and not a second copy of
it: every projection, the FFN, the router and the head are already
batched over M rows and are untouched. Only the attention section
differs, because only attention is per-sequence — each row rotates at
its own position, appends to its own slot, and attends to its own
history. It loops rows through the same kernels the batched arm calls,
with `tokens=1`, so a row cannot drift from what the rest of the engine
does to it.

Attention looping costs M launches and no extra bytes: batching cannot
make a row read fewer of its own keys. The saving is in the weights, and
the batch already shares those.

**Hybrids too.** A DeltaNet layer keeps its whole memory in a conv ring
and a delta state rather than a KV cache, and there was one of each per
ENGINE. Both are now per sequence slot. This is the one that matters for
Ornith.

**The scheduler.** One thread owns the engine; request threads hand it
prompts and are handed tokens back. A request arriving mid-flight joins
at the next admission rather than the back of a queue; one that finishes
leaves without disturbing the rest; one whose caller hangs up releases
its slot immediately. `/v1/models` reports the real width.

## What it refuses, by name

> **Superseded 2026-09-22.** The composable-serving commits
> (`e344cf2..bf961c3`) lifted the first three rows below: Muse, gemma-4
> and Qwen4-Exp batch; TP and PP batch (rank 0 drives the workers); a
> drafter batches on one card.  What `batch_unsupported_reason()` refuses
> today is a drafter under TP/PP, one slot, and no matrix hardware.  The
> prefix cache composes with batching too.  See
> `AUDIT-2026-09-22-TOWER-READINESS.md`; the table is kept as history.

`batch_unsupported_reason()` answers in a sentence, and the server falls
back to one request at a time with speculation intact — byte for byte
what it did before.

| refused | why |
| --- | --- |
| Muse, gemma-4, Qwen4-Exp | each has its own residual graph and its own batched path; running them through the Qwen loop would contradict their decode token for token and still read as English |
| TP, PP | collectives whose shape every rank must agree on; nothing negotiates a batch width across ranks |
| a loaded drafter | a verify batch and a multi-sequence batch want the same rows of the same path |
| one slot | nothing to batch |
| no matrix hardware | `GRIMOIRE_BATCHED_PREFILL_NOXMX=1` runs it slowly, for checking |

## Turning it on

```bash
GRIMOIRE_SEQ_SLOTS=8     # conversations the KV cache holds
GRIMOIRE_MAX_BATCH=8     # stepped together (server)
GRIMOIRE_PREFIX_CACHE=1  # turns resume instead of re-reading
```

Slots multiply the KV cache. Eight slots is eight caches, so size
`--ctx` and the slot count against the VRAM the model leaves. The banner
says what it decided:

```
    prefix cache  8 conversations resident
    batching      up to 8 sequences per step
```

If it says `one at a time` it names the reason. Read that line before
concluding the flags did nothing — that banner has located more bugs in
this project than any assertion (rule 15).

**CORRECTION (2026-09-21, external audit F8): the three flags above do
NOT yet compose.** The batchable scheduler path never calls
`prefix_reuse()`/`restore_prefix_upto()` on admission and never snapshots
on completion, so `GRIMOIRE_PREFIX_CACHE=1` has no effect once batching
is actually engaged — a growing conversation is read in full every turn
regardless. See `DAY-ONE.md` section 2c for the full account and
CLAUDE.md rule 21. This file is kept as the design rationale for the
work below it, not as the current claim of what is composed.
this project than any assertion (rule 15).

## How it is checked

`bin/test_batch_decode` — four conversations of DIFFERENT lengths
(17-41 tokens), dense, MoE and hybrid, bf16 and fp8, each required to
answer exactly what it answers alone. Then again with a stop token
chosen from what the fixture actually emits, so rows retire unevenly and
the narrowing path is covered. The batch counter is read: a path that
quietly stepped each sequence alone would pass every token comparison
there is.

`bin/test_scheduler` — six requests from six threads, released together
by a gate so they genuinely overlap, each required to answer what it
answers alone. Same counter check.

Both verified to discriminate, by building the engine wrong on purpose:

| broken on purpose | result |
| --- | --- |
| every row takes row 0's position | all four cells DIFFER |
| every row shares one DeltaNet state | both hybrid cells DIFFER |
| every row shares one conv ring | both hybrid cells DIFFER |
| the scheduler does not bind a slot | both batchable cells DIFFER |
| a conversation does not extend its own slot | 6 slots occupied, not 2 |
| a resume copies a max_seq-sized buffer | 134400 vs 527616 bytes |

**The size of the fixture was load-bearing** — see rule 19. At the
original two-to-six-token prompts, the shared-DeltaNet control PASSED
and the shared-conv-ring control failed one cell out of two. The gate
was green on its most important claim while testing nothing about it,
and raising the lengths turned the real code red as well: the batch
drivers were clearing the wrong slot.

## What to measure on the Tower

In this order. Nothing below can be answered here.

1. **Aggregate tok/s at 1, 2, 4, 8 concurrent**, same model, same
   prompt. The claim is that 8 costs about what 1 costs. The honest
   failure mode is that it does not: attention is looped per row, so at
   long context attention can become the bottleneck before the weights
   do. Find the width where aggregate tok/s stops rising — that is the
   real `GRIMOIRE_MAX_BATCH` for this box.
2. **Batching versus speculation.** They are mutually exclusive today.
   MTP at ~2 accepted/step against 8-way batching is a genuine trade and
   the repo holds NO measurement either way. Run both on the same
   prompt; read the output (rule 8).
3. **VRAM.** Eight slots is eight KV caches. Establish what `--ctx`
   actually fits alongside the model before recommending a slot count.
4. **Prefix reuse on a real conversation.** Time to first token on turn
   five of a long chat, cache on and off. This is the win that does not
   need concurrency at all and it should be the largest single one for
   agentic work.

## What is not done

- Batching and speculation together. A scheduling question — how many
  draft tokens for how many sequences — not a kernel one. Worth doing
  only if (2) says speculation wins at the concurrency actually used.
- Batching under TP or PP.
- Muse, gemma-4 and Qwen4-Exp batching. Each needs its own residual
  graph taught the same per-row attention section; the pattern is now
  established and the work is mechanical, but it is four more paths and
  four more fixture rows.
- The prefix cache still refuses when a drafter is loaded.
