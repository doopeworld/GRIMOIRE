# HANDOFF 2026-10-03 — served speed vs CLI speed, concurrency, prefix cache

Live doc: updated as each result lands. Newest section last.
Runs and raw logs: `/mnt/storage/isos/grimoire-runs/bench-1003/` on the Tower.
llama-benchy pinned to **0.4.0** (`uvx llama-benchy@0.4.0`), with `--tokenizer`
pointing at the model directory, so prompt and depth sizes are in the model's
own tokens. The fallback was gpt2 approximations.

## Context

On 2026-10-02, llama-benchy against `GRIMOIRE-ORNITH` (HTTP) measured TG
127.6 tok/s (pp2048/tg256) and 101.3 (pp4096/tg32). The CLI measured
198.6 tok/s on the same model. Ian asked why; the session paused overnight.
A power cut followed. Nothing had been started, and nothing was lost.

## 1. Why served TG was 101–128 when the CLI said 199. SOLVED.

There were three causes, with no HTTP cost to speak of. Every number below
is Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE on gpu0 (03:00.0). The first request to
each server was a short chat, which is what GRIMOIRE-ORNITH saw yesterday.

| config (llama-benchy tg, tok/s)                                 | depth ~0 | 2K    | 4K    |
|-----------------------------------------------------------------|---------:|------:|------:|
| yesterday's container: grimoire-b70, `SYCL_UR_USE_LEVEL_ZERO_V2=0` | 159–163  | 121–124 | 98–99 |
| same image, V2 adapter (env var removed)                         | 194–197  | 141–142 | 109   |
| V2 + `GRIMOIRE_DECODE_SPLITS=128` (workaround for cause b)       | 195–196  | 181–182 | 171–172 |
| CLI, engine only, differential (n160 − n32) at the same depths   | 202.5    | 180.8 | 172.0 |

a. **The legacy Level Zero adapter.** All four GRIMOIRE Unraid templates
   (ORNITH/QWEN/MUSE/DUAL) set `SYCL_UR_USE_LEVEL_ZERO_V2=0`, which forces the
   legacy UR adapter. That setting was inherited from `grimoire:b70-native`,
   which segfaulted on V2 on 09-25. The new `grimoire-b70` image runs V2
   cleanly, and V2 is the oneAPI 2026.1 default and what every CLI baseline
   used. Cost of the legacy adapter: about 1.2 ms per token (194 → 159 tok/s).

b. **The decode graph froze the attention split count at capture.**
   `build_graph()` records `forward(1)` with the host `pos` at its current
   value. `decode_splits()` therefore baked `max(8, ceil(len/32))` for the
   *capture-time* length into the graph. With one sequence slot,
   `bind_seq_slot()` returns early, so `reset()` never invalidates the graph,
   and the server captures it ONCE, on its first request. After a short first
   request, every later request ran its whole context through 8 splits: on
   Ornith that is 64 ESIMD threads on a 256-EU card. Cost: 1.6 ms/token at 2K
   and 3.4 ms/token at 4K. The CLI never showed this, because it captures
   right after the real prompt's prefill.

c. **Apples and oranges.** The CLI's 198.6 is tokens 32..160 after a
   20-token prompt. llama-benchy measures decode after 2K or 4K tokens of
   context. The engine itself is 181 at 2K and 172 at 4K. That is
   attention cost, not serving overhead.

**The HTTP path costs nothing measurable.** Once a and b are removed,
served = CLI within noise at 2K and 4K. At ~0 depth the remaining 2–3% is
the 128-split workaround itself (5.09 vs 4.94 ms/token on the CLI). The real
fix below does not pay that.

**Fix for b (branch `served-decode-fix`):** `AttnParams::capture` is set while
a graph is recorded. The launch is then sized for the whole cache (`seq_cap`),
and every decode-attention kernel (ESIMD hd256, GQA, per-head) plus
`launch_flash_merge` derives, from the *device-side* length, the split count
that direct submission would launch. The surplus splits return at once.
Direct submission is unchanged. A replayed step now does exactly the
direct-submission math at every length. Files: `src/kernels.hpp`,
`src/attention.cpp`, `src/grimoire.cpp` (`ap.capture = recording` in
`forward`, `forward_muse`, `forward_gemma4`).

**Fix for a:** remove `SYCL_UR_USE_LEVEL_ZERO_V2=0` from the templates and the
container (pending: done after validation, see below).

### Validation of the fix (bench-1003/validate_fix.txt, gpu0, 10:05)

- CLI, short context, new binary vs old (`bin/grimoire.pre1003`): n=160 text
  **IDENTICAL**; decode 5.02 ms/token = 199.4 tok/s (old binary today 4.94 ms,
  within run-to-run noise; the gpu1 sweep lane was running alongside).
- CLI, 4K context, n=96: graph replay vs `GRIMOIRE_DECODE_GRAPH=0` direct
  submission **IDENTICAL**. New direct vs old direct **IDENTICAL**, so the
  direct path is untouched.
- Served (host-built `bin/grimoire-server`, V2 adapter, NO split env, short
  first request), llama-benchy 0.4.0:

| tok/s        | depth ~0 (pp128) | 2K (pp2048) | 4K (pp4096) |
|--------------|-----------------:|------------:|------------:|
| tg32         | 196.4            | 183.6       | 171.5       |
| tg256        | 195.1            | 183.4       | 172.4       |
| CLI (engine) | 202.5 / 199.4    | 180.8       | 172.0       |
| 10-02 container | --            | 127.6 (tg256) | 101.3 (tg32) |

Served TG now equals engine TG at every depth. Prefill is unchanged at
9.5–9.8k tok/s (pp2048/pp4096).

## Superseded

- 2026-10-02 memory/notes: "the gap is a fixed per-request/per-token HTTP
  serving cost (SSE framing, JSON encode, harmony split)". **Wrong.** It was
  the adapter plus the frozen split count. Serving costs ~0.
- 2026-10-02 notes: "Ornith/Muse cannot batch: linear attention falls back
  to one request at a time". **Wrong for current code.**
  `batch_unsupported_reason()` only requires `GRIMOIRE_SEQ_SLOTS >= 2`, a
  matrix-capable device and no distributed speculation. The "one request at
  a time" line on GRIMOIRE-ORNITH came from the container being started
  without `GRIMOIRE_SEQ_SLOTS`.
- The host `bin/grimoire-server` was a 09-25 build, older than the ESIMD
  decode work. Anything launched through `tools/serve.sh` before today ran
  that stale server, and with `GRIMOIRE_W4A8=1` and the Ornith-only flags
  hard-coded. Rebuilt today.

## INCIDENT 10:05:30: gpu1 (0b:00.0) dropped off the PCIe bus. Hardware, needs a reboot.

The sweep lane on gpu1 had just started (Qwen3.8-27B-MXFP4, batching server,
first requests after the model load) when the following was logged:

```
10:05:30 pcieport 0000:00:06.2: AER: Multiple Correctable error ... Physical Layer, RxErr
10:05:30 pcieport 0000:0a:02.0: Unable to change power state from D3hot to D0, device inaccessible
10:21:14 xe 0000:0b:00.0: Force wake domain 0: wake. MMIO unreliable (forcewake register returns 0xFFFFFFFF)
```

- `lspci -s 0b:00.0` now returns nothing. The card is gone from config space.
- The neighbouring root port 00:06.0 (the Arc B580's path, 05:00.0) has logged
  correctable RxErr all morning, every 30–60 s. This is the same signature as the
  09-26 drop, when a B70 sat on 00:06.0: the x4 slot area is electrically
  marginal at Gen4 (16 GT/s). The B580 being back on 00:06.0 is the hardware
  change since TP/PP last ran cleanly on gpu0+gpu1 (09-30..10-02).
- gpu0 (03:00.0, root port 00:01.0, Gen4 x8): zero AER errors, still clean.
- The container `z6912` (grimoire-server on the dead card) is stuck spinning in
  the Level Zero driver. Per the GPU hang rule it was **not** killed. Its
  clients were stopped. **Recovery: reboot the Tower.** Before relying on gpu1
  again, force that slot to Gen3 in the BIOS and/or reseat its riser/cable.
- Everything after this runs on gpu0 only.

## 2. Concurrency 1/2/4/8 (sweep running on gpu0: bench-1003/zoo, lane-gpu0.txt)

Method: grimoire-server (host-built `bin/`, V2 adapter), `GRIMOIRE_SEQ_SLOTS=8`, llama-benchy
0.4.0 `--pp 512 --tg 64 --concurrency 1 2 4 8 --runs 2`. A batch check is also run: 4
prompts asked serially, then all 4 at once.

**Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE** (scheduler: "batching up to 8 requests per step"):

| concurrency | total tg tok/s | per request | pp512 TTFT |
|---|---:|---:|---:|
| 1 | 132.7 | 132.7 | 116 ms |
| 2 | 94.7 | 49.5 | 169 ms |
| 4 | 133.4 | 37.1 | 289 ms |
| 8 | 175.3 | 25.8 | 512 ms |
| (one-at-a-time graph path, no SEQ_SLOTS) | **195** | 195 | |

- With `GRIMOIRE_SEQ_SLOTS>=2`, **every** request goes through the batched path
  (`decode_batch` -> `prefill` with M rows). A lone request then decodes at 132 instead
  of 195 tok/s: no graph replay, and none of the M=1 ESIMD decode kernels.
- On this MoE model, 8 concurrent requests deliver less in total than one request on
  the graph path. Each extra row routes to its own experts, so weight traffic grows
  with M instead of being shared (this is a top-8-of-256 MoE).
- Batch check: 2/4 identical. The two that differ are coherent and identical for many
  tokens, then split at a near-tie ("Let me find some." / "Let me find some primes...").
  That is floating-point summation order (M=1 vs M=4 kernels), not cross-talk.

## 3. Prefix cache. It never hits for an HTTP client, on any model (diagnosed)

`prefix_reuse()` (src/grimoire.cpp) reuses a slot only when the slot's **entire**
cached token list (prompt + reply, minus the last token) is a prefix of the new
prompt. There is no partial match. For a hybrid (DeltaNet) model, partial reuse is
impossible as built, because the recurrent state is kept only at the END of the
cached sequence. Three client patterns, measured on Ornith (cache off -> on, 2nd request):

| pattern | off | on | why it cannot hit |
|---|---:|---:|---|
| shared ~2K system context, new user msg (llama-benchy `--enable-prefix-caching`) | 0.51 s | 0.51 s | the cached sequence contains the OLD user msg + reply |
| multi-turn chat, turn 2 = turn 1 + reply + new question | 0.51 s | 0.53 s | the generation prompt is `<|im_start|>assistant\n<think>\n` (tokenizer.cpp:638), but history renders the assistant turn WITHOUT `<think>\n` (:635), so it diverges right after `assistant\n` |
| raw /v1/completions continuation (prompt + completion + more) | 0.45 s | 0.46 s | the reply re-tokenizes differently: the model emitted 3 separate `\n` tokens, re-encoding the text gives one `ĊĊĊ` token (24 generated ids vs 22 on re-encode), so the full-sequence match fails |

Answers were identical with the cache on and off (no corruption). llama-benchy's
prefix test shows no change at all (`pp256 @ d4096` TTFT 492 ms off, 495 ms on).

**What would make it work (design, not done tonight):**
1. Keep a recurrent-state checkpoint at the END OF THE LAST USER MESSAGE: prefill the
   prompt minus the generation header, snapshot (DeltaNet state + conv ring, ~63 MB on
   Ornith; KV rows are already in place), then prefill the header. A turn-2 prompt
   always contains turn 1 up to that point, token-exact. Only the old reply plus the
   new message gets prefilled. That makes multi-turn chat and agents hit.
2. Match the longest common prefix that lands on a checkpoint, not the whole cached
   list. Attention-only models can truncate KV at ANY position, so add partial reuse
   for them directly.
3. Optionally checkpoint after the system message too, so llama-benchy's
   shared-context pattern (and agents with one long system prompt) hits.

## Release v1.1, published 2026-10-03 11:42 (CEST)

https://github.com/doopeworld/GRIMOIRE/releases/tag/v1.1 (pre-release, target c08ec32):
`grimoire-b70-v1.1.tar.gz`, 640,285,691 bytes, sha256 `8d1030bc028957a38142ba3e00bdfd152105a0a3c5d0c743e950917bd067fb97`.
Validated before upload with llama-benchy on gpu0, run like GRIMOIRE-ORNITH without `V2=0`.
Coherence PASSED. tg256: 195.6 / 183.5 / 172.5 tok/s at depth ~0 / 2K / 4K; pp 9.6–9.9k.
v1 is left as it was. Tool: `tools/bench/upload_release.py` (token from env `GH_TOKEN`).
The GRIMOIRE-ORNITH Unraid template was fixed: `--device /dev/dri/renderD128` (gpu0 after
the 10-03 reboot; renderD129 is now the iGPU), and the `SYCL_UR_USE_LEVEL_ZERO_V2` line was
removed. The backup is in bench-1003/templates-backup/. QWEN/MUSE/DUAL still run the old
`grimoire:b70-native` image, which needs `V2=0`, so they were left alone. Move them to
`grimoire-b70:latest` before removing it.


## 4. Full sweep, 23 checkpoints (bench-1003/zoo, zoo-table.md). gpu0, 10:58–14:30

Server: `GRIMOIRE_SEQ_SLOTS=8`, prefix cache off (phase 1) / on (phase 2). c1..c8 = total decode tok/s
(llama-benchy pp512/tg64). "batch==serial" = 4 prompts serially vs all at once (greedy); the
non-identical ones are coherent and split at near-ties. The prefix columns are the 2nd request's
wall time, cache off -> on. Muse GPTQ/MXFP4 ran c1/c2 only (each Muse run took 30–43 min).

| model | fmt | sched | c1 | c2 | c4 | c8 | c8/c1 | batch==serial | pfx chat off→on (s) | pfx raw off→on (s) | benchy pp256@d4096 TTFT off→on (ms) |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| Agnes-3.0-Flash | mxfp4 | batch | 25.7 | 19.1 | 32.0 | 48.6 | 1.89x | 4/4 | 2.61→2.63 | 2.30→2.29 | 2570→2571 |
| K2-Horizon-MoVA-36B-A4B | mxfp4 | batch | 63.6 | 42.8 | 48.2 | 52.8 | 0.83x | 0/4 | 1.33→1.33 | 1.17→0.49 | 1227→1227 |
| Muse-Glimmer-30B-GPTQ-INT4 | int4 | batch | 0.8 | 1.6 | - | - | - | 4/4 | 43.56→43.55 | 32.73→32.72 | 7050→7035 |
| Muse-Glimmer-30B-INT4-W4A16 | int4 | batch | 0.8 | 1.6 | 3.0 | 5.6 | 6.89x | 4/4 | 43.57→43.55 | 32.72→32.72 | 7055→7046 |
| Muse-Glimmer-30B-MXFP4 | mxfp4 | batch | 0.8 | 1.5 | - | - | - | 4/4 | 44.44→44.45 | 33.35→33.34 | 6947→6949 |
| Ornith-1.5-35B-A3B-FP8 | mxfp4 | batch | 132.5 | 95.4 | 134.2 | 175.9 | 1.33x | 4/4 | 0.51→0.53 | 0.45→0.46 | 490→492 |
| Ornith-1.5-35B-A3B-GPTQ-Int4 | int4 | batch | 83.7 | 40.4 | 33.4 | 30.6 | 0.37x | 4/4 | 2.21→2.22 | 2.10→2.11 | 1963→1964 |
| Ornith-1.5-35B-A3B-INT4-W4A16-AutoRound | int4 | batch | 83.9 | 41.4 | 33.8 | 31.2 | 0.37x | 4/4 | 2.20→2.21 | 2.08→2.09 | 1956→1947 |
| Ornith-1.5-35B-A3B-MTPFIX | mxfp4 | batch | 132.6 | 94.6 | 135.5 | 175.2 | 1.32x | 2/4 | 0.51→0.53 | 0.45→0.46 | 492→497 |
| Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE | mxfp4 | batch | 132.7 | 94.7 | 133.4 | 175.3 | 1.32x | 2/4 | 0.51→0.53 | 0.45→0.46 | 492→495 |
| Ornith-1.5-35B-A3B-NVFP4 | mxfp4 | batch | 131.3 | 95.8 | 131.8 | 176.6 | 1.35x | 1/4 | 0.51→0.53 | 0.46→0.46 | 496→491 |
| Ornith-1.5-35B-A3B | mxfp4 | batch | 132.8 | 96.2 | 133.6 | 177.3 | 1.34x | 2/4 | 0.51→0.53 | 0.45→0.46 | 492→492 |
| Qwen3.6-35B-A3B-GPTQ-Int4 | int4 | batch | 83.7 | 42.2 | 34.5 | 32.0 | 0.38x | 4/4 | 2.17→2.18 | 2.05→2.06 | 2014→2011 |
| Qwen3.8-27B-FP8 | mxfp4 | batch | 30.6 | 22.2 | 37.4 | 57.5 | 1.88x | 4/4 | 2.17→2.18 | 1.90→1.90 | 2131→2129 |
| Qwen3.8-27B-GPTQ-Int4-MTP-BF16 | int4 | batch | 19.1 | 16.8 | 17.0 | 16.5 | 0.87x | 4/4 | 3.10→3.12 | 2.67→2.68 | 2569→2580 |
| Qwen3.8-27B-MXFP4-AutoRound | mxfp4 | batch | 30.6 | 22.3 | 37.6 | 57.7 | 1.88x | 3/4 | 2.17→2.18 | 1.90→1.90 | 2137→2131 |
| Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16 | mxfp4 | batch | 30.6 | 22.2 | 37.7 | 57.6 | 1.88x | 3/4 | 2.17→2.19 | 1.90→1.91 | 2129→2132 |
| Qwen3.8-27B-MXFP4-GRIMOIRE | mxfp4 | batch | 30.6 | 22.3 | 37.8 | 57.5 | 1.88x | 3/4 | 2.17→2.19 | 1.90→1.91 | 2133→2131 |
| Qwen3.8-27B-NVFP4 | mxfp4 | batch | 30.6 | 22.2 | 37.4 | 57.5 | 1.88x | 4/4 | 2.16→2.18 | 1.90→1.90 | 2129→2129 |
| Qwen3.8-27B-W4A16 | int4 | batch | 19.0 | 16.7 | 16.9 | 16.5 | 0.87x | 4/4 | 3.12→3.13 | 2.69→2.69 | 2598→2587 |
| Qwen3.8-27B-int4-AutoRound | int4 | batch | 19.0 | 16.8 | 16.9 | 16.5 | 0.87x | 4/4 | 3.10→3.12 | 2.68→2.68 | 2576→2579 |
| Qwen3.8-27B | mxfp4 | batch | 30.6 | 22.3 | 37.7 | 57.5 | 1.88x | 3/4 | 2.17→2.19 | 1.90→1.90 | 2128→2131 |
| Qwen3.8-Flash-Next-NVFP4 | bf16 | batch | - | - | - | - | - | 0/4 | ERR | ERR | - |

- Every model batches without errors, except Flash-Next: every request failed with HTTP 500
  "batched decode step failed", because its file-backed PLE table supports one sequence only.
  **Fixed (served-tools):** `batch_unsupported_reason()` now names it, and the scheduler
  falls back to one at a time.
- Batching scales poorly everywhere (c8/c1 = 0.37x–1.89x; an M=4 step costs ~5.5x an M=1 graph
  step on Qwen3.8-27B). int4 checkpoints are worst: aggregate FALLS with concurrency. Cause: the
  batched decode IS prefill() (prompt-sized GEMMs, per-row conv/DeltaNet/attention, no graph,
  host syncs), and admission prefills stall every live row.
- Prefix cache: no hit anywhere except K2's raw continuation (1.17 -> 0.49 s). See section 3.

## 5. Scheduler solo fast path. Validated, branch served-tools

`decode_solo()`/`solo_ok()`: when exactly ONE sequence is live (and no PP/TP/DAG/Qwen4-Exp/
drafter), the scheduler binds its slot, puts cursor + token on the device and replays the
decode graph. decode_batch() re-records the graph after any multi-row step.
`GRIMOIRE_SCHED_SOLO=0` turns it off.

| with GRIMOIRE_SEQ_SLOTS=8 | lone request output vs one-at-a-time | c1 before -> after | c2..c8 |
|---|---|---|---|
| Ornith-1.5-35B-A3B-MXFP4 | IDENTICAL (200 + 60 tokens); overlap test IDENTICAL | 132.7 -> **193.3** | unchanged (95 / 134 / 176) |
| Qwen3.8-27B-MXFP4 | IDENTICAL | 30.6 -> **33.1** | unchanged |
| Muse-Glimmer-30B-INT4-W4A16 | IDENTICAL | 0.8 -> **18.9** | unchanged (batched Muse path still ~0.8/row) |

Ornith at depth, solo build, SEQ_SLOTS=8: tg256 196.3 / 184.5 / 173.2 at depth ~0 / 2K / 4K, the
same as the one-at-a-time server. Turning concurrency on no longer costs single-user speed.

## Release v1.2, published 2026-10-03 14:55 (CEST). State at end of day

https://github.com/doopeworld/GRIMOIRE/releases/tag/v1.2 (pre-release, target b745052):
`grimoire-b70-v1.2.tar.gz`, 640,343,202 bytes, sha256 `8a3a2c760d77672a57f89a8d27fa2b3e040b45882b13ba84f83630d03e84a7db`.
Validated before upload (bench-1003/validate_image_v12.txt), Ornith on gpu0:
- Default: coherence PASSED, tg256 195.6 / 183.5 / 172.4 at depth ~0 / 2K / 4K (same as v1.1).
- `GRIMOIRE_SEQ_SLOTS=8`: tg64 c1 193.4 (v1.1: 132.7), c2 95.2, c4 132.6, c8 174.2.

Docker images on the Tower: `grimoire-b70:latest` = v1.2 (48df3e7c6b26), `:v1.1` (322cb487f99c), `:v1` (9270d65a31c3).

**GRIMOIRE-ORNITH** was recreated on v1.2: port 6889, `--device /dev/dri/renderD128` (gpu0),
no `SYCL_UR_USE_LEVEL_ZERO_V2`, one request at a time (best single-user speed). Healthy.

**Open, for Ian:**
1. Reboot the Tower to bring gpu1 (0b:00.0) back. Container `z6912` is stuck on the dead card
   and goes away with the reboot. Before loading gpu1 again, force its slot (root port 00:06.2)
   to Gen3 in the BIOS or reseat it.
2. After any reboot, check `ls -l /dev/dri/by-path/`. Render nodes renumber, and the Unraid
   templates pin `renderD128`.
3. QWEN/MUSE/DUAL templates still use the old `grimoire:b70-native` image (it needs `V2=0`).
   Move them to `grimoire-b70:latest` and drop `V2=0` together.

## 6. Profile of one batched decode step (Ornith, gpu0, 2026-10-03 ~15:10 CEST). PAUSED here

`GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_SCHED_SOLO=0 GRIMOIRE_TIME_LAYER=all`, M short concurrent
requests. Each region is synced, so absolute times are inflated; compare the ratios.
Driver: bench-1003/prof_batch.py.

| region (ms per step) | M=1 | M=2 | M=4 | M=8 |
|---|---:|---:|---:|---:|
| routed MoE | 2.16 | 7.21 | 11.11 | **16.46** |
| post norm + route | 1.50 | 1.70 | 2.02 | 2.63 |
| DN qkv / z / gate / out projections | 2.21 | 5.28 | 5.49 | 5.86 |
| shared expert FFN + shared expert | 1.08 | 2.34 | 2.60 | 3.10 |
| attn q / kv projections | 0.42 | 1.28 | 1.29 | 1.29 |
| final norm + logits | 0.49 | 1.23 | 1.25 | 1.28 |
| DN recurrence + conv + attn flash + rope/kv (per-row loops) | 1.28 | 1.99 | 3.13 | 5.57 |
| TOTAL (timed) | 10.21 | 22.42 | 28.51 | 37.76 |

Uninstrumented wall time (solo on): M=2 23 ms, M=4 32 ms, M=8 48.5 ms per step, including
the short prefills. The solo graph step is ~5.2 ms.

Reading:
- **Routed MoE is 44% of the M=8 step** and grows ~7x from M=1, the same factor as the
  distinct-expert bytes (8 tokens x top-8 hits ~57 distinct experts/layer). It runs at about
  the same ~40% of bandwidth as at M=1. Lever: a grouped MoE kernel at ~80% of bandwidth
  (16.5 -> ~8 ms).
- **The projections jump 2.5x from M=1 to M=2 and then stay flat** (weights read once, but
  a slower small-M GEMM path than the M=1 ESIMD GEMV). Lever: an M-row ESIMD GEMV (~14 -> ~6 ms).
- **The per-row loops grow linearly** (1.3 -> 5.6 ms). Lever: one launch over all rows.
- Plus graph replay per batch size (launch overhead). Estimated M=8 step ~14–16 ms ->
  **~500–570 tok/s at c8**. 800 needs ~10 ms per step, i.e. MoE near the bandwidth roofline.

**State at pause:** GRIMOIRE-ORNITH is STOPPED (stopped for this profile). Start it again
from the Unraid template (fixed: renderD128, v1.2 image), or ask Claude to. No GPU work running.

## 7. Reference: how vLLM batches on XPU (read from its source), and its measured c1–c8

vLLM 0.30.1.dev27 + vllm-xpu-kernels 0.1.15.4, read in its own container. GRIMOIRE uses
none of it. This is the method to re-implement in GRIMOIRE's own SYCL/ESIMD code.

1. **Scheduler:** a token budget per step. Every running sequence adds 1 decode token, and
   new prompts are chunked into the rest of the budget IN THE SAME forward pass, so decode
   never stalls behind an admission.
2. **One kernel per layer over the flattened token batch [N, H]:**
   - linear layers: `int4_gemm_w4a16` / `fp4_gemm` / `fp8_gemm` on the systolic array (DPAS).
     8 rows cost what 1 row costs, and each weight is read once.
   - MoE: `remap_hidden_states` (sort token x top-k by expert, `rows_per_expert`), ONE
     grouped GEMM for gate_up over all touched experts, one act, one grouped GEMM for down,
     one `moe_gather`. ~5 launches per layer, each touched expert read once.
   - DeltaNet: ONE fused `_xpu_C.gdn_attention` per layer, covering conv update, gating and
     delta-rule recurrence for all sequences, with per-sequence state indices.
   - attention: one varlen flash-attention call over the paged KV for all sequences.
3. **Graphs per batch size** (`VLLM_XPU_ENABLE_XPU_GRAPH=1`, experimental on XPU, padded to
   capture sizes); the GDN op runs eagerly between graph pieces.

**Measured** (bench-1003/vllm, gpu0, Ornith-1.5-35B-A3B-GPTQ-Int4, XPU graphs, fp8 KV,
max-num-seqs 8, no MTP). llama-benchy pp512/tg64, total tok/s (per request):

| | c1 | c2 | c4 | c8 | tg256 @ 0 / 2K / 4K |
|---|---:|---:|---:|---:|---|
| vLLM (GPTQ int4) | 72.6 | 125.2 (65.7) | 210.6 (58.6) | **332.0** (50.8) | 73.3 / 72.0 / 71.3 |
| GRIMOIRE v1.2 (MXFP4) | **193** | 95 (50) | 133 (37) | 176 (26) | **195.6 / 183.5 / 172.4** |

vLLM's 8-row step costs 1.75x its 1-row step; GRIMOIRE's costs ~9x. GRIMOIRE leads single-stream
by 2.7x and trails at c8 by 1.9x.

Note on the competitor image: my-vllm-xpu (both the 10-03 15:37 build and the 09-24 one) has an
unresolved merge-conflict marker in vllm/model_executor/models/qwen3_dflash.py:715
(`<<<<<<< HEAD` / `>>>>>>> v0.30.0`). On XPU, get_quantization_config() imports it for every
quantized model, so vLLM dies with a SyntaxError. The reference run mounted a resolved copy of
that one file (HEAD side kept); the image was not changed.

## 8. Batched decode, GRIMOIRE's own kernels (2026-10-03 evening, main 4706273 / 18d07bc / 0dc218b)

Resumed from section 6. Ian's target is Ornith c8 around 800 tok/s.
**Physical ceiling at c8:** about 5.6 GB has to move per 8-row step:
- ~57 of 256 experts touched: 3.6 GB
- dense layers + lm_head: 1.0 GB
- 8 DeltaNet states read and written: 1.0 GB

At 600 GB/s that is 9.4 ms per step, so ~850 tok/s is the hard limit; ~600 is realistic.

All new code is in GRIMOIRE's own SYCL/ESIMD. It is default for batched decode across sequences
(`seqb`, not a verify). Speculative verify keeps its old path, because its acceptance was tuned
on that rounding.

| piece | where | switch |
|---|---|---|
| small-M MXFP4 DPAS GEMM, 16 cols/thread, exact ALU E2M1 decode, scale after DPAS | gemm_fast.cpp `launch_mxfp4_smallm` | `GRIMOIRE_SMALLM_DPAS` 1/0 |
| same for BF16 weights (router, DeltaNet gates), no decode | `launch_bf16_smallm` | same |
| grouped MoE, gate_up with weights as the DPAS A operand ("AW") | `moe_mxfp4_smallm_aw_gate_up` | `GRIMOIRE_MOE_AW=0` |
| grouped MoE down, 16 cols/thread | `launch_moe_mxfp4_smallm` | `GRIMOIRE_MOE_SMALLM_DPAS` 1/0 |
| DeltaNet step + conv, one launch for all rows | `launch_deltanet_step_rows`, `launch_causal_conv1d_split_rows` | `GRIMOIRE_ROWS_BATCH=0` |
| rope, kv append, ESIMD flash decode + merge for all rows | `launch_*_rows` (prefill.cpp, attention.cpp) | same |
| attention lengths uploaded once per step | grimoire.cpp `rows_len_dev` | same |

Debug switches:
- `GRIMOIRE_SMALLM_VERIFY=1`: new GEMMs against the reference, max|diff|/max|ref|.
- `GRIMOIRE_ROWS_VERIFY=1`: per-row reference, bit for bit.
- `GRIMOIRE_BATCH_HOST_TIMING=1`: host enqueue vs device wait per step, and scheduler step-to-step time.
- `GRIMOIRE_HOST_REGIONS=1`: host time per region.

**llama-benchy pp512/tg64, Ornith MXFP4, gpu0, total tok/s** (coherence PASSED every run):

| | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| v1.2 (morning) | 193 | 95 | 133 | 176 |
| 4706273: small-M GEMMs + DN/conv rows | 193.7 | 144.1 | 178.9 | 218.4 (peak 273) |
| 18d07bc: + attention rows | 193.7 | 147.2 | 188.6 | 234.8 (peak 297) |
| 0dc218b: + BF16 GEMM + AW gate_up | 193.9 | 146.0 | 196.7 | **254.2 (peak 330)** |
| vLLM GPTQ-int4, same card (its own container) | 72.6 | 125.2 | 210.6 | 332.0 (peak 420) |

**Device time per M=8 step** (GRIMOIRE_PROFILE_PREFILL, no syncs): 23.5 → 21.8 → **19.24 ms**.

| region | ms |
|---|---:|
| routed MoE | 10.84 |
| post norm + route | 1.36 |
| DN recurrence | 1.27 (state bandwidth) |
| DN qkv / out / z | 0.90 / 0.78 / 0.62 |
| logits | 0.67 |
| shared expert | 1.09 |
| attention (flash + rope/kv) | 0.37 |

**The step is GPU-bound and the scheduler is free.** The host enqueues a step in ~6 ms, then
blocks in the token readback until the device finishes. Steady-state step-to-step is 19.4 ms at
M≈7 and the engine call is 19.4 ms of it. Engine decode at c8 is therefore ~410 tok/s, the same
as vLLM's 420 peak.

**The remaining c8 gap is prompt admission.** GRIMOIRE prefills each new prompt on its own
(pp512 c8 4,556 tok/s total) while every decoding row waits. vLLM prefills them together
(10,294 tok/s).

Numerics:
- dense MXFP4 vs the fp32 decode GEMV: ≤ 2.6e-3. This is bf16 activations, the same as the old small-M path.
- BF16 router/gates: ≤ 1.15e-3.
- MoE vs the old grouped kernel: ≤ 1.2e-4.

Batched outputs are coherent and on-topic on Ornith MXFP4 / GPTQ-Int4, Qwen3.8-27B-MXFP4, K2
and Agnes (tools/bench/texts8.py).

**MoE kernel findings** (tools/moe_smallm_probe.cpp, Ornith shapes from DRAM, ~57 experts):
- Both old designs run at ~300 GB/s: 16-cols/thread DPAS, and the GRIMOIRE_MOE_GEMV_SMALL GEMV.
- With the E2M1 decode removed the old layout still only reaches 375–420 GB/s. So its 32-byte
  transposed tiles over short K streams are the limit.
- Deeper prefetch is slower. Contiguous expert ids don't help (not TLB).
- AW gate_up: 297 → 380 GB/s. It is now ALU-bound at ~1.3 instructions per weight byte.
- An SLM lookup-table decode is slower (219 vs 168 us).

**Next, in order:**
1. **Batch prompt admission.** Prefill all waiting prompts in one forward:
   - dense / MoE run over all tokens;
   - conv / DeltaNet / attention run per span;
   - decode rows ride along.
2. **Shared expert** (1.09 ms) as one more tile of the routed launch.
3. **Dense projections** (3.4 ms) on the AW layout.
4. **fp16 operands** for the MoE, to cut decode ALU (~5 of ~19 instructions per 16 lanes).

## PAUSED 2026-10-03 ~20:00 by Ian ("lets pause... backup everything and we resume tomorrow")

Ian's verdict on v1.3 is that c2/c4/c8 are "absolutely trash compared to vllm".

**The defect to fix first:** at c2 the server's total throughput (146) is BELOW our own
single-user speed (193). A second user makes the server slower. vLLM scales 72 → 125 → 211 → 332.

**Diagnosis:**
- A lone request runs the graph-replayed decode path, ~5.1 ms per token.
- With two or more requests in flight, every step runs through `prefill()`, the prompt path.
- That path costs **9.75 ms of device time at M=1** (GRIMOIRE_PROFILE_PREFILL, this
  evening). That is ~2x the decode graph before batching shares a single byte.
- So an M=2 step costs about as much as 2–3 solo steps, and batching only wins from M≥4.
- Decode is GPU-bound at every M, and the scheduler costs nothing (section 8).

**Tomorrow, in order:**
1. **Measure M=1/2/4/8 per region on the batched path** (interrupted tonight, nothing ran).
   First stop GRIMOIRE-ORNITH gracefully, with no connections on :6889:
   ```
   cd /mnt/storage/isos/grimoire-runs/bench-1003 && SRV_MODE=hostbin SRV_WAIT=1200 \
   SRV_ENV=$'GRIMOIRE_SEQ_SLOTS=8\nGRIMOIRE_SCHED_SOLO=0\nGRIMOIRE_PROFILE_PREFILL=1' \
   bash srv.sh up ab-dprofM gpu0 6990 Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE mxfp4 && \
   python3 prof_batch.py 6990 Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE ab-dprofM wall; \
   bash srv.sh down ab-dprofM 6990; \
   python3 tools/bench/dprof_table.py /mnt/storage/isos/grimoire-runs/bench/logs/ab-dprofM.log
   ```
   Compare it per kernel with the decode graph (GRIMOIRE_TIMELINE=1).
2. **Batched decode on the DECODE path instead of prefill(), with one graph per batch size.**
   - M-row versions of the decode kernels: row-streaming GEMVs with M accumulators for M ≤ 4,
     small-M DPAS above that.
   - The decode MoE kernels over (token, expert) pairs.
   - DN / conv / attention rows kernels reading slot and pos from device arrays (today the
     slots are baked by value as RowSlots).
   - Targets: M=2 step ≤ ~7 ms (c2 ~280+), M=8 ~14 ms (c8 ~550).
3. **Batched prompt admission:** all waiting prompts in one forward.

**State at pause:**
- main = this commit. Release v1.3 is published.
- GRIMOIRE-ORNITH runs v1.3 (aab687ac3fad) with GRIMOIRE_SEQ_SLOTS=8 on renderD128, port 6889.
  It was restarted at ~20:00 after the interrupted profile had stopped it gracefully.
- gpu1 (0b:00.0) still needs Ian's reboot.

## 9. 2026-10-04: resumed. c2–c8 now beat vLLM (main bd8b066, e449b7c)

**Tower after the reboot.**
- **Render nodes renumbered:** gpu0 (03:00.0) is now **renderD129**; renderD128 is the iGPU.
  The GRIMOIRE-ORNITH template and container were moved to renderD129 (backup:
  templates-backup/*.before-renderD129). Docker's `--device` cannot take the by-path name
  (it splits on the `:` in the PCI address), so this must be checked after every reboot.
- **AER correctable errors:** they come from the B580's switch (05:00.0 under 00:06.0) and
  once from gpu1's switch. gpu0's path (00:01.0) is clean.

**1. Measured the batched step per region, M=1/2/4/8** (tools/bench/dprof_table.py).
- Device ms per step: M=1 7.00, M=2 11.17, M=4 14.45, M=8 19.28. The decode graph does one
  token in ~5.1 ms.
- The decode timeline (GRIMOIRE_TIMELINE=1 GRIMOIRE_DECODE_GRAPH=0) puts a DeltaNet layer at
  ~109 µs, of which MoE gate_up+shared is 24.4 µs and down 11.9 µs, i.e. ~380 GB/s. The
  grouped small-M MoE kernels ran at ~220 GB/s at M=2.

**2. bd8b066: the decode step's ESIMD MoE over M tokens** (`launch_moe_shared_rows`).
- The shared expert is fused, as in decode. Per token it is bit-identical to a decode step:
  1,560 checks, 0 differences.
- Device ms per step: M=1 6.34, M=2 8.74, M=4 11.51, M=8 17.01.
- Steady state at M≈7.5 is 17.0 ms per step (~450 tok/s).
- llama-benchy c2 / c4 / c8 = 180.8 / 228.6 / 264.9.

**3. e449b7c: batched admission** (SeqBatch::spans, Grimoire::admit_batch).
- Every prompt taken in one scheduler round goes through one prefill.
- **Why it was slow:** a 512-token prompt touches all 256 experts, about 17 GB of weights,
  and took ~110 ms on its own. Eight prompts one at a time cost ~970 ms per c8 round.
- **Now:** six 512-token prompts take one 340 ms prefill.

| llama-benchy pp512/tg64, total tok/s | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| v1.3 | 193.4 | 145.8 | 198.3 | 255.2 |
| bd8b066 (MoE over M tokens) | 193.7 | 180.8 | 228.6 | 264.9 |
| **e449b7c (batched admission)** | **193.5** | **214.4** | **264.1** | **391.8** (peak 414) |
| vLLM GPTQ-int4, same card | 72.6 | 125.2 | 210.6 | 332.0 |

pp512 at c8: 7,979 tok/s (was 4,552; vLLM 10,294). Coherence PASSED.

Batched admission falls back to one prompt at a time for PP/TP, MTP/DFlash, Qwen4-exp, MoVA
(K2), PLE, tiered experts, the native GDN bridge, and when the prefix cache is on.

Checked on:
- Ornith: long mixed prompts are identical; short prompts differ only at near-ties (0.09
  top-2 margin).
- Qwen3.8-27B and Agnes: coherent.
- K2: falls back.

Fixed along the way: the one-launch DeltaNet/conv row kernels (4706273) now exclude verify
and spans rows, which are consecutive tokens of one sequence.

**Next:**
- Release v1.4 (building now).
- Decode step at M=8 (17 ms):
  - router region 1.3 ms;
  - M-row dense GEMVs;
  - the shared expert read once per step, not once per token.
- Optionally hold admission a few ms when requests arrive together: llama-benchy's 8 arrive
  as 6 + 2.

## 10. 2026-10-04 afternoon: consistent batching, top-k, release v1.5 (main 286ad6e)

**Changes:**
- **a97da18 + 286ad6e: burst admission hold.** An idle server waits up to
  GRIMOIRE_ADMIT_HOLD_MS (default 8) for the rest of a burst, so the burst is admitted as ONE
  batched prefill. A lone request is held only if the previous busy period was concurrent and
  ended < 10 s ago, so a single-user chat never waits.
- **Why:** Ian asked "so now c4 is slower". The engine was not slower; the 4-row step had gone
  11.42 → 10.71 ms. llama-benchy's c4 rounds were being admitted as 1 + 3 in 2 of 5 rounds
  (257 vs 328 tok/s).
- **8744f17: batched router top-k** on the decode step's ESIMD kernel, one thread per token,
  bit-identical. The sub-group batched kernel cost ~21 µs per layer, 0.84 ms per step, at every
  M.
- **8744f17: norm bf16 rows reused** (bn_bf_sync): small-M GEMMs read the norm's bf16 rows
  directly.
- **8744f17: wider K-split for the BF16 router GEMM.**

**Device time per batched step:** M=1 5.65, M=2 7.91, M=4 10.71, M=8 16.27 ms.

**v1.5 image** (c88c16bb8e01, sha256 1aa2237d…), llama-benchy (coherence PASSED):

| | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| total tok/s | 193.6 | 237.8 | 354.8 | 464.2 |
| vLLM | 72.6 | 125.2 | 210.6 | 332.0 |

- pp512 at c8: 9,547 tok/s.
- Single user, unchanged: 195.7 / 183.4 / 172.2 tok/s at 0 / 2K / 4K.

**Tried and dropped:**
- **Dense weights-as-A GEMM:** la_qkv at M=8 ran at 78 vs 127 GB/s.
- **MoE rows per work-group** (R = 2/4/8): no change.
- **Shared expert read once per step** (one work-group set looping all tokens + a separate
  down launch): slower, MoE M=8 9.82 → 10.90 ms, because the shared work-groups become
  stragglers. Removing the shared slot entirely saves only 0.86 ms at M=8, including its
  per-token arithmetic.

**State:** GRIMOIRE-ORNITH runs v1.5 on renderD129, port 6889, GRIMOIRE_SEQ_SLOTS=8.

**Next, if Ian wants more:**
- A decode-graph path for M rows; the batched step still pays the prefill machinery's extra
  kernels.
- Dedupe of routed experts shared by two tokens at M=8.


## 11. 2026-10-09: v1.9.0 published, Ornith "regression" closed, ONE-SLOT MTP BUG found, PAUSED ~17:10 CEST

Ian: "stop and pause everything" / "make a backup, send github push and we will resume tomorrow".
State at the pause: no containers, gpu0 idle (0.02 GiB), gpu0's AER chain 0/0/0 before and after every run.
Clocks: the Tower is CEST; Ian's Mac is IDT (+1 h).

**v1.9.0 is PUBLISHED** -- https://github.com/doopeworld/GRIMOIRE/releases/tag/v1.9.0 (tag = 78274a5,
grimoire-b70-v1.9.0.tar.gz, 641,226,938 B).

**Ornith pp512 "regression" = llama-benchy artifact, closed.**
- Same-session llama-benchy pp512, 1 / 8 users (tok/s): v1.8 7,234 / 9,761; v1.8.1 7,123 / 9,771;
  v1.8.2 6,792 / 9,236 (decode identical, 194.4 / 468 tok/s). Its time to first response rises 5 ms
  (= one Ornith decode step) from v1.8.2 on: 113.1 / 113.6 / 118.2 ms; its latency probe stays ~42 ms.
- Direct requests do not show it: the same 541-token prompt (the server counts 541 tokens on all four
  images) takes 119.0 ms on v1.8, 120.2 v1.8.1, 119.1 v1.8.2, 120.4 v1.9.0 (median of 5); llama-benchy's
  own random-slice prompts with its exact payload stream the first chunk after ~112 ms on v1.8.1 and v1.8.2.
- Ruled out: chat template (1 token more, which llama-benchy compensates), reasoning_content split,
  decoder hold-back, scheduler lone-burst hold (GRIMOIRE_ADMIT_HOLD_MS=0: v1.8.2 6,786 and v1.9.0 6,423 at
  c1). Users see no difference; why llama-benchy's run sequence does is unknown.

**BUG: MTP with ONE sequence slot corrupts the answer (committed code = v1.9.0).**
- Found while running Ian's long-context table with GRIMOIRE_SEQ_SLOTS=1 (ctx 135,168, MTP K=6,
  DRAFT_VOCAB 65536, Qwen3.8-27B-GPTQ-Int4-MTP-BF16): EVERY coding answer fails the syntax / test check at
  4K, 8K, 16K, 64K and 130K -- chunks of the answer are skipped or repeated mid-line, 4.3-4.9 tokens per
  step (5.2-5.5 when healthy). The same task passed this morning with 8 slots.
- A/B (bench-1003/ab_lc.sh, 4K coding answer, one launch each, 17:02-17:05 CEST):

  | run | attention library | ctx | slots | answer | decode | tokens/step |
  |---|---|---:|---:|---|---:|---:|
  | A | committed (78274a5) | 135,168 | 1 | FAIL, corrupted | 90.7 tok/s | 4.78 |
  | B | new (longctx-attn-wip) | 16,384 | 8 | PASS | 120.7 tok/s | 5.23 |
  | C | new | 16,384 | 1 | FAIL, corrupted | 89.4 tok/s | 4.71 |

  So the trigger is the single slot. NOT the new attention kernel, NOT the long context.
- Exposure: GRIMOIRE_SEQ_SLOTS defaults to 1 (README line 58); the README run command sets 8 and the
  Unraid templates use 8. Unknown yet: whether the default K=4 also fails at 1 slot, whether Ornith /
  other MTP models do, and which commit introduced it (candidates: 97b56da DPAS verify attention,
  1aa80fd fused DN/conv/norm, 78274a5 cheap int4 dequant; bin-prev/ holds an older server build).
  Whether v1.9.0's release notes need a warning is Ian's call (public surface).

**Long-context verify attention -- branch `longctx-attn-wip`, passes at 4K / 8 slots, not yet proven faster.**
- Change (src/attn_verify_dpas.cpp): one work-group per (KV head, split) holding the `tiles` threads
  (local id = tile), so the ~6 GQA tiles read the same K/V blocks together (DRAM once, L1 for the rest);
  at most kMaxKeysPerSplit = 1024 keys per split. The 10-08 kernel gave every tile its own work-group.
- Kernel bench (tools/bench_verify_attn.cpp, gpu0, cache sized for the longest length, 131,072; exact
  against fp64 up to 16K with error 3e-4 as before; longer lengths timing only). old = the per-row decode
  kernel + merge, new = the DPAS kernel with this change. 7 rows, us: 2K 658 -> 60; 8K 2,229 -> 252;
  16K 9,140 -> 511; 32K 23,556 -> 875; 64K 50,686 -> 839; 131K 105,546 -> 1,529 (10.9x - 69x).
  1 row: 83 -> 52, 306 -> 122, 1,192 -> 304, 3,135 -> 1,111, 6,075 -> 2,108, 11,437 -> 2,551 (1.6x - 4.5x).
- NOT measured: this change against the COMMITTED DPAS kernel at the same cache size. The 10-08 figures
  (7 rows: 2K 57, 8K 128, 16K 231 us) used a 16K cache. Suspected: with a 131K cache every kernel is
  ~2x slower even at 2K-16K (the per-row kernel: 2K 283 -> 658 us) because K is laid out
  [256 dims][seq_cap], 128 KB apart per dimension -- address translation; a blocked K layout would fix it.
- Prefill is unaffected by the change (other kernels); with ctx 135,168 and 1 slot, time to first token was
  2.21 / 4.19 / 9.29 / 59.3 / 194.0 s = 1,895 / 1,955 / 1,741 / 1,082 / 671 tok/s at 4K / 8K / 16K / 64K /
  130K (prompts 4,180 / 8,184 / 16,180 / 64,189 / 130,191 tokens). Their decode numbers (82.0 / 67.5 / 53.8 /
  21.8 / 10.8 tok/s) are INVALID: corrupted answers.

**Test kit** (Tower /mnt/storage/isos/grimoire-runs/bench-1003/, copy on Ian's Mac in
~/grimoire-work/bench-1003-scripts-20261009/): cmp_g0.sh (grim|k8v4; new variables GRIM_CTX, K8V4_MAXLEN,
K8V4_SEQS, PROBE_ARGS, NO_BENCHY, GRIM_ENV), coding_probe.py, lc_table.sh (GRIMOIRE, then the third-party
k8v4 stack, 4K-130K), ab_lc.sh, bisect_ornith.sh, bisect_hold.sh, tmplcount.sh, streamfirst*.sh. bin-oldattn/
(Tower checkout) = today's server + the committed attention library, for A/B runs.

**Other**
- Ian's Mac runs Mullvad VPN; with "Local network sharing: block" the Mac loses the whole 192.168.8.x LAN
  (it looked like a dead Tower for ~10 min). Ian turned LAN sharing on.
- Intel cloud GPU: Ian has an instance "in review"; his JupyterLab page has no SSH. Key: ~/.ssh/id_ed25519_intel_cloud
  on his Mac (comment intel-cloud-gpu). Need the SSH line from the console's "How to Connect"; read-only first.
  GRIMOIRE builds for BMG G31 only (GRIM_TARGETS adds other chips).
- A real power cut at ~15:00 CEST took down the Mac and the Tower; the Tower was back at 16:22 CEST with the
  array started and gpu0 healthy (03:00.0 = renderD128).
- A watcher I queued to start the A/B "later" started it anyway (two queue attempts, one invisible to ps);
  the 3 runs were short and each stopped its server gracefully. Next time: run it in the foreground.

**Next, in order**
1. Find the one-slot MTP corruption: reproduce with ab_lc.sh run C, then bisect 97b56da / 1aa80fd /
   78274a5 (bin-prev), compare the single-slot verify path with the batched one, test K=4 and Ornith at
   1 slot. Fix, run tools/regress_all_g0.sh, tell Ian whether v1.9.0 needs a warning.
2. Bench the work-group change against the committed kernel at the same cache size; keep it only if it wins.
3. Ian's table: GRIMOIRE vs the third-party k8v4 stack at 4K / 8K / 16K / 64K / 131K (bench-1003/lc_table.sh).
   8 slots do not fit 135K (4.1 GiB of KV per slot); use 2 slots if they fit, or fix item 1 first.
4. Intel cloud instance once it is approved.


## 12. 2026-10-09 resumed: one-slot speculative state commit

Ian explicitly resumed work after the pause in section 11. Tower is still on
`longctx-attn-wip`, starting from `2299a03`; gpu0 is PCI `03:00.0`, `[8086:e223]`,
`renderD128`. No reboot or work on gpu1 occurred.

### Findings

- Reproduced run C: one-slot Qwen3.8 GPTQ/MTP K=6 at ctx 16384, 4K coding prompt,
  syntax FAIL (unterminated string). Default K=4 and Ornith K=4 also fail.
- Fresh builds with pinned oneAPI 2026.1.1: `97b56da` PASS, `1aa80fd` FAIL;
  committed `78274a5` server/library FAIL. `bin-prev` contains the fused DeltaNet
  code too, so it cannot serve as a pre-fusion baseline. Current build with
  `GRIMOIRE_DN_VERIFY_FUSED=0` PASS.
- Root cause: `launch_deltanet_verify_rows` reads live state without advancing it;
  `generation.hpp` committed only after a rejection. Full acceptance silently lost
  the block's recurrent state. The scheduler's `decode_spec_batch` already replayed
  every accepted prefix, explaining the slot-dependent failure. Fix: commit every
  successful batched recurrent verify, including full acceptance; declined verify
  continues with a plain step and never commits unfilled buffers.
- Host test makes recurrent state influence output and covers MTP/DFlash, K=4/6,
  every rejection prefix, full acceptance and declined verify. It passes on Mac and
  Linux; restoring only the old condition fails greedy parity. The pre-existing host
  fake lacked PP stubs after the generation header changed; these are restored, and
  `<cstdio>` is included explicitly.
- Repaired served Qwen3.8 GPTQ/MTP one slot: K=6 4K fresh/repeat PASS (105.0/105.2
  tok/s); 8K fresh/repeat PASS (79.3/79.3). K=4 4K/8K PASS (86.1/70.1). 8-slot K=6
  control PASS (120.7 at 4K). These use ctx 16384; they are not the long-context table.
- Ornith's repaired 4K answer used `bool` to reject boolean endpoints. The probe's
  builtin allowlist omitted that harmless type, producing NameError on otherwise
  valid code. Added `bool`, preserving all functional assertions; all four saved
  repaired Ornith answers pass on recheck and the old corrupted one still fails.
- Scripts remain outside git in `grimoire-runs/bench-1003`; the Mac copy of
  `coding_probe.py` also includes the builtin fix. Original saved as
  `coding_probe.py.before-bool-resume`. Raw JSONL evidence: `cmp-k8v4/grim-resume-*.jsonl`.
- Server and CLI rebuilt in isolated worktrees, with no GPU mapped to build containers.
  Repaired binaries and libraries installed into the main Tower checkout's `bin/`.
  Original files preserved in `bench-1003/resume-original-bin/` (no archive).
- v1.9.0's published image remains affected; README now warns to use >=2 slots for
  serving with MTP, or disable MTP. No image or release asset was replaced.

### Next steps

1. Run `tools/regress_all_g0.sh`, and inspect its parity checks and generated text.
2. Compare committed and work-group DPAS attention with identical cache capacity.
   Committed library is preserved at `bin/attn-committed/libgrimoire_attn.so`.
3. Run both sides of Ian's table via `bench-1003/lc_table.sh`; use ctx 135168 and
   preferably 2 slots to exercise the faster scheduler verify path. Only PASS decode
   numbers are valid. Record actual prompt token counts for both checkpoints.
4. Ask Ian for Intel cloud's How to Connect SSH line; inspect with his cloud key,
   run no heavy workload.


### Resume checkpoint: full sweep and selected attention (table running)

- `tools/regress_all_g0.sh`: 44/44 completed, ALL DONE, no gpu0 guard trip. Logs:
  `/mnt/storage/isos/grimoire-runs/regress-1009-1730`, wrapper output
  `bench-1003/regress-resume.out`. Sherlock matches its reference; hot-expert text
  IDENTICAL. The printed MTP/DFlash prose comparisons differ from plain, as they
  already did in the pre-fix `regress-1009-1315`; do not describe this as exact
  speculative token parity across all real checkpoints.
- A new Ornith launch passed both 4K answers but failed both 8K answers with a real
  IndexError: interval endpoints were indexed before checking tuple length. The
  source is coherent, unlike the pre-fix corruption. These two decode rates are
  excluded; the preceding repaired launch's 4K/8K answers pass the corrected checker.
- v1.9.0 release notes now include the published warning and link to `dd286b3`.
  The image and asset are unchanged.
- Same executable, random inputs, cache capacities 16384 and 131072, pinned toolchain,
  gpu0: unconditional grouping LOSES at short context. 7 rows, 16K cache, committed
  vs grouped at 4K/8K/16K: 66/131/233 -> 118/178/330 us. At 131K cache: 4K/8K/16K
  67/230/451 -> 98/251/512; 64K/128K 1051/1850 -> 837/1525 us.
- Equal split counts at 64K/128K (committed library with KEYS_PER_SPLIT=1024):
  1552/3166 us vs grouped 837/1525. Thus grouping itself helps long context, whereas
  the original 32-split policy is better than merely raising the split count.
- Selected library: use committed tile layout until the long-context split floor
  exceeds the ordinary thread target, then group the tiles. 16K cache, 7 rows:
  4K/8K/16K = 66.8/127.5/231.0 us. 131K cache: 4K/8K/16K = 67.0/229.3/450.0,
  64K/128K = 831.9/1510.8 (26.4%/22.4% faster than committed). One-row timings
  preserved. Errors vs fp64 up to 16K unchanged; longer lengths timing only.
- Files in bench-1003: `attn-committed-16k.out`, `attn-committed-131k.out`,
  `attn-candidate-16k.out`, `attn-candidate-131k.out`,
  `attn-committed-equalsplits.out`, `attn-adaptive-16k.out`, `attn-adaptive-131k.out`.
- The selected library was rebuilt separately and installed without an active GPU
  container. Only the library changed; the verified server/CLI ABI is unchanged.
- Table is now RUNNING in foreground: `LENGTHS=4096,8192,16384,65536,131072
  GRIM_SLOTS=2 bash lc_table.sh`, output `bench-1003/lc-table-resume.out`.
  Both servers reserve 135168 context. Probe summary medians now exclude failed
  answers. Removed outer timeouts from the table wrapper; k8v4 starts with --init.
  Script updates also live in the Mac copy of bench-1003.

Next: deliver the complete GRIMOIRE/k8v4 table (PASS decode only), then ask for
Intel cloud's How to Connect SSH line and inspect only.


### Table completed; Ian changed comparison source to published GitHub results

GRIMOIRE all 10/10 answers PASS, ctx 135168 and 2 slots. Prompt/decode tok/s at
4K/8K/16K/64K/128K: 1928/120.0, 1872/121.2, 1691/115.0, 1052/77.1, 678/66.8.
Fresh TTFT: 2.22/4.48/9.80/62.47/193.68 s. Actual counts and sources are in
`RESULTS-2026-10-09-CONTEXT.md`; raw JSONL is `cmp-k8v4/grim-lc-1009-1818.jsonl`.

The attempted local k8v4 launch exited itself: its image's native library expects
TP2 local heads 12/2, not TP1 24/4. No local numbers were generated. Ian then said
"take wrapzii results from his page on github". Use his October 2 published coding
sweep for comparable 8K/64K/128K data; mark 4K/16K unreported in that sweep. His
subsequent 128K W4A8 confirmation is also documented, with its different sample
method. Do not combine historical forced-EOS fox rows with current coding rows.

All GPU containers stopped, gpu0 0.02 GiB, all monitored correctable/nonfatal/fatal
AER totals zero. Intel cloud key exists; asked Ian for How to Connect SSH line.
Next: inspect cloud hardware only when the line arrives. Run nothing heavy there.


### Table correction requested by Ian

Ian pointed out that the table used the older 128K Wrapzii row despite mentioning
his newer fix separately. Corrected `RESULTS-2026-10-09-CONTEXT.md`: Wrapzii 128K is
now 1254.9 prompt / 78.02 decode (display 1255/78.0), from the W4A8 wiring confirmation.
8K/64K remain the earlier published sweep because the wiring report does not provide
fresh prompt throughput for those lengths. The report explicitly records these
runtime/sampling differences. Do not present 1111/74.48 as his latest 128K result.


### Final pause requested by Ian — 2026-10-09

The final 16K kernel diagnostic completed. Findings and tomorrow's order are in
HANDOFF-2026-10-09-PAUSED.md. Tower idle, no GPU containers, gpu0 0.02 GiB and AER
totals zero. No jobs queued; gpu1 untouched. Resume only when Ian asks.
