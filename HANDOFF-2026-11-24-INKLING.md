# Inkling 2 Mini on GRIMOIRE — handoff 2026-11-24 (paused by Ian)

Branch `inkling-gpu-1123`. Engine: `tools/inkling2_gpu.cpp` (+ `inkling2_mtp.inc`, `inkling2_moe_gemm.cpp`,
`inkling2_xexp.cpp`). Tower: `/mnt/storage/isos/grimoire-runs/ink-1123` (`build_ink.sh`, `run_ink.sh`, `run_tp.sh`).

## Where it stands (measured, Inkling-2-Mini-Uncensored)

| setup | value |
|---|---|
| 1 B70 prompt pp512 / pp4096 / pp16384 | 1,296 / 2,689 / 2,282 tok/s (start 11-23: 355 / 954 / 636) |
| 1 B70 generation, BF16 weights (exact) | 57.5 tok/s |
| 1 B70 generation, `INK_I8=1` | 66.5 tok/s (12 tokens = CPU reference) |
| 1 B70 generation, `INK_I4=1` (GRIMOIRE INT4 RTN + `launch_int4_smallm_wide`) | 72.1 tok/s (max/mean dlogit 2.54/0.42) |
| 2 B70 generation, `INK_I8=1` | 80.4 tok/s @512, 78.4 @4K |
| 1 B70 MTP (DSpark) | 37.6 (BF16), 44.4 (`INK_I4=1`, 2.76 tok/step) |
| 2 B70 MTP | 47.4 |

Per token (int8 dense, one B70, ~15 ms): experts ~5.5 ms, ~1000 tiny kernels ~4.3 ms, dense ~3 ms, unembed 0.8 ms.

## The wall, and the way around it (READ FIRST)

Four expert-decode designs (SIMT SLM tables; ESIMD DPAS with nibble widening; DPAS one-wave; DPAS with SLM byte
tables) ALL ran at ~300 GB/s (gate_up ~75-79 us, down ~41-44 us per visit). Do not write a fifth.

GRIMOIRE already solved this for Ornith: `src/moe_kernels.cpp` `moe_gu_esimd` / `moe_dn_esimd` ("ESIMD decode MoE,
the llm-scaler shape"), MEASURED 266/248 GB/s (SIMT) -> **477/484 GB/s**:
- a thread streams WHOLE ROWS with contiguous 64-byte block loads (`mx4_row_step`), R row pairs per thread,
  KS threads split H, partial sums meet in SLM only at the end (no tables, no barrier in the loop);
- the activation slice lives in REGISTERS, split even/odd K (= low/high nibble of each byte);
- E2M1 decode in the ALU: nibble magnitude bits -> fp16 bits 9..11 + sign bit 15 = value x 2^-14
  (`e2m1_split`), the 2^14 folded into the block scale; one FMA pair per 16 per scale block.
- down kernel: work-group = TK+1 slots, slot partials summed in slot order (routed weights applied there).

**Plan 1 (first thing):** port `moe_gu_esimd` / `moe_dn_esimd` to Inkling NVFP4: scale = E4M3 per 16 elements x
outer (amax[e]/den), apply per 16-block (8 per 128-step instead of 4), Inkling's softcapped SwiGLU in the epilogue,
gate/up rows interleaved (gate i = row 2i, up i = row 2i+1). Drop the per-weight bf16 rounding (vLLM-style dequant;
check vs CPU ref: expect mean dlogit ~0.05-0.08). Target: experts 5.5 -> ~3.2 ms/token.

**Plan 2:** verify (M=8) experts the same way as Ornith (`launch_moe_mxfp4_smallm` pattern, grouped small-M, every
expert read once per step) -- MTP is only worth it once verify reads at ~480 GB/s.

**Plan 3:** cut the ~1000 tiny kernels per token (fuse norm into mix, router+topk, attention 3->2, gate cat+blend,
post+resid+mlpconv); Xenithium/urakozz fuse exactly these. Target ~-2 ms/token.

**Plan 4 (prompt):** port GRIMOIRE's Ornith prompt MoE `moe_mxfp4_esimd` (gemm_fast.cpp, 52-68 TFLOP/s, SLM 1 KB
byte table, 128-K steps, 2 DPAS per 32 K) to NVFP4 instead of the own `InkMoeG` (~32 TFLOP/s).

**Check once (read-only sysfs, during a decode run):** GPU act_freq on gpu0 while decoding -- short kernels may run
at a lower clock than the long unembed kernel (571 GB/s).

## Do not repeat (all measured 11-24)
- `SYCL_UR_USE_LEVEL_ZERO_V2=0` -> DEVICE_LOST on gpu0 (bcs reset + coredump). Default adapter is V2.
- `INK_DEVX=1` device mailboxes in pinned host memory: no `usm_atomic_host_allocations` on B70 -> spin never sees
  the peer's writes, watchdog exit, engine resets. No P2P on the stock Tower kernel.
- 256-thread mix kernel for decode rows (tg 57 -> 54); kept only for prompt rows.
- A-prefetch in `InkMoeG` (experts 782 -> 888 ms).
- Decoding all experts up front before the oneDNN GEMMs (loses L2 locality).

## Opt-in flags (env, passed by run_ink.sh / run_tp.sh)
`INK_I8=1` int8 Wp/Wo/unembed decode; `INK_I4=1` (+`INK_I4_PARTS` bits 1 Wp, 2 Wo, 4 unembed,
`GRIMOIRE_INT4_RTN_GROUP=64`, `GRIMOIRE_INT4_RTN_SEARCH=1`); `INK_GSM=1` GRIMOIRE bf16_smallm for decode/verify dense
(MTP == plain 128/128); `INK_XEXP=1` DPAS expert decode (no gain); `INK_SECT=1` per-section prompt timing;
`INK_DENSE_ONEDNN=1`, `INK_ATT_SIMT=1`, `INK_EXP_ONEDNN=1` = older paths.
Two cards: `run_tp.sh TAG tp IDS CTX PP_LIST TG CHUNK`, `run_tp.sh TAG tpmtp CHAT CTX PP N CHUNK` (watchdog 60 s).

## Research notes 11-24
- Xenithium (github.com/Niluthon/Xenithium, Apache-2.0, SYCL/ESIMD): clone in the session scratchpad; MoE decode
  `src/kernels/moe/moe.cpp`, verify rows `moe_rows.cpp`, W4 tile `include/xenithium/kernels/tile/w4_tile.hpp`.
  Flint INT4 (MoE 256 top-6) 1 card 278 tok/s with DFlash 2, pp8K 8,166.
- urakozz (OpenCL C + Level Zero command-list replay): Ornith 1 card pp4K 11,223, MTP 312; verify M=8 costs 1.57x one
  plain step; decode dispatch 0.73 us in a replayed list.
- Reference Inkling speeds elsewhere: vLLM + DSpark RTX 5090 ~416 tok/s code / pp8K ~17.5K; llama.cpp IQ4 ~43-63 tok/s.

## 11-24 later (resumed; Ian: "at least 300 tg and 5000 pp using dual gpu and mtp")

| setup (measured) | before today | now |
|---|---|---|
| 2 B70 pipeline prompt pp16384 / pp32768 | 3,550 / 3,295 | **5,374 / 5,410** (chunk 4096) |
| 2 B70 pipeline prompt pp4096 | 3,081 | 4,288 (chunk 2048) |
| 1 B70 prompt pp4096 / pp16384 | 2,689 / 2,282 | 3,816 / 3,521 |
| 2 B70 MTP, code answer (code2.txt) | -- | **159.1 tok/s** (6.08 tok/step, 38.2 ms/step) |
| 2 B70 MTP, chat1 (384 tok) | 43.1 | 84.7 (3.14 tok/step) |
| 1 B70 MTP code2 / chat1 | -- / 34.9 | 112.5 / 58.6 |
| plain decode 1 B70 / 2 B70 | 78.3 / 86 | 79.0 / 86.3 |

Prompt: ESIMD flash attention (InkAttE, GRIMOIRE flash_esimd design), Ornith ESIMD expert GEMM ported to NVFP4 +
MXFP8 (InkMoeNvEs) with the softcapped SwiGLU in the epilogue, two-card pipeline mode `pp` (split 24).
MTP verify: InkMoeSmE (GRIMOIRE moe_mxfp4_esimd_small design, fp16 DPAS, K-permuted nibble decode 0 4 1 5 2 6 3 7,
register double buffer; in libink_xexp.so 128 GRF), INT8 Wp/Wo/unembed via b70::launch_fp8_smallm (src/gemv_decode.cpp
built as bin/libink_gd.so), router + gate down via b70::launch_bf16_smallm, InkGroupP, InkGateBlend2, INT8 drafter head.
code2.txt = code1 + "200001 200004" (answer starts as text, no thinking): DSpark accepts ~6 of 8 there.

Two-card MTP step (code2): draft 4.4 ms (both cards, redundant), verify 33.6 ms = GPU ~25 ms + waiting for the
other card 3.3 ms (experts split by id are unbalanced per visit) + ~5 ms host round trips (96 exchanges a step).
Verify GPU work per card: experts ~10 ms (~330 GB/s), small M=8 kernels ~6 ms (not split), attention 2.2, dense 1.8.
Next, in order: experts replicated on both cards + used experts split by parity; exchange straight into the
peer's slot; M=8 small-kernel fusion; faster verify expert kernel.

## 11-24 evening, continued (commits 11bb39f .. 1077102)

| two B70 (measured) | value |
|---|---|
| MTP code2 (code answer) | **251 tok/s**, 6.08 tok/step, 24.2 ms/step (draft 2.6, verify ~21.4) |
| MTP chat1 (384 tok) | **132 tok/s**, 3.17 tok/step, 24.0 ms/step |
| plain decode | 92-94 tok/s |
| one B70: MTP code2 / plain | 177 tok/s (33.3 ms/step) / 79 |

What moved it (each measured, flags = the old path):
- every expert on both cards, used experts split by parity (INK_NO_EPREP): cards wait less on each other;
- verify attention on the ESIMD flash kernel, 32/64-key slices + merge (INK_OLD_ATTB, INK_NO_ATTSPLIT, INK_ATT_CHUNK);
- verify experts K-split 6, interleaved slices on two cards (GRIMOIRE int4_smallm ilv), parity grid (INK_SM_KS);
- INT8 drafter layers + head, drafter vocabulary split across the cards (INK_DRAFT_BF16, INK_DRAFT_FULL);
- ESIMD DPAS gate blend (INK_NO_BLENDE); bias table inside InkPost (INK_SEP_BIAS); verify ring inputs stored
  by InkPost / InkResid / InkMlpConv (INK_SEP_HISTV); graph chains split at the exchanges (INK_NO_SEG);
  exchange partial copied straight into the peer's slot (INK_SEG_HOSTCOPY); decode add folded (INK_SEP_ADD).
- Tried and slower / no gain: fused top-k + grouping work-group (INK_ROUTEGRP), 128-GRF verify experts with the
  bf16 decode, L1 prefetch two steps ahead (INK_SM_PF2), spin-wait on segment events (INK_SEG_SPIN),
  folding the exchange add into 8-row verify kernels (4 PCIe reads a value).
- Bug caught by the forced-wrong-drafts check: folded ring store fell through to the prompt ring update.
Two-card verify step per card: experts ~9 ms (~380 GB/s), INT8 small-M ~2.2, attention ~1, ~5 ms of small
kernels that do not shrink with two cards, ~2.5 ms idle at 96 exchanges. 300 tok/s on code needs ~20 ms/step.
Hardware counters (unitrace -q) fail in the container (metric init assertion).
- More measured dead ends (11-24 afternoon): bf16 payload for the 8-row exchanges -- GPU 2-byte stores into
  pinned host memory: 400 ms/step and wrong tokens; device convert + copy engine: no gain (24.1 vs 24.3 ms) and
  MTP != plain from token 2 (bf16 partial sums lose precision when the two cards' partials cancel). Exchange add
  folded into per-channel verify kernels: neutral. 8-row FMA INT8 GEMV (InkEsI8M): slower than launch_fp8_smallm.
  GRIMOIRE already measured enqueue-ahead TP with host-signaled L0 events (GRIMOIRE_TP_ASYNC, 11-19): slower on the
  285H; do not redo without a free P-core.

## 11-24 late: speed tests = BetterBench (Ian: "as of now please use tool for speed test : better bench")

OpenAI server for the Inkling engine (none existed; the engine only read token-ID files):
- `tools/inkling2_gpu.cpp` mode `serve MODEL serve SOCKET CTX CHUNK CARDS`: line protocol on a Unix socket
  (GEN / TOK / END / ERR / QUIT), one or two cards (both ranks in lockstep, same RNG seed), requests one at a time.
- Sampling (temperature / top-k / top-p, vLLM order) on the engine: GPU top-32 per row (InkTopK1/2), candidates
  merged across the cards on the host; exact speculative sampling on the DSpark verify rows.
- `tools/inkling2_server.py` (Tower venv, HF tokenizers): /v1/models, /v1/chat/completions (SSE), chat_template.jinja
  rendered for text messages (thinking-effort line, default 0.9), `<|content_thinking|>` -> reasoning_content.
- `tools/serve_ink.sh start CARDS CTX [CHUNK] [PORT]` / `stop` (front end first, then QUIT to the idle engine).
- `tools/bb_ink.sh TAG [args]`: BetterBench with bench-1022/betterbench-vllm1111.json (temp 0.7 / top-p 0.95 /
  top-k 20, prefill 2K / 8K, concurrency 1/2/4/8), client on CPUs 14,15; prints the engine's tokens/step, ms/step.

