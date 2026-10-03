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
