# HANDOFF 2026-10-22 -- two B70s on the redriver adapter (safetensors models)

Task (Ian, 10-22): "resume work on grimoire on qwen3.8-27b fp8 and ornith1.5 fp8 and qwen flash next
on dual gpu because i changed my m2 to oculink to a new one with redriver so it should not drop anymore."
Then: "dont work on GGUF !!!! your task is on grimoire safetensor models".
Code: this checkout (grimoire-fuse), branch longctx-attn-wip 8a7257d; bin/ = 10-10 build. GGUF work
(grimoire-gguf, feature/gguf-battlemage) is a separate effort and is NOT touched here.

## 1. Hardware after the 11:45 CEST reboot (read-only checks, 11:52-11:54)

| | PCI | render | root port | link |
|---|---|---|---|---|
| gpu0 B70 | 0000:03:00.0 | renderD128 | 00:01.0 (EX Pro dock) | Gen4 x8, 0 AER |
| gpu1 B70 | 0000:08:00.0 (NEW address) | renderD130 | 00:06.2 (M.2, new redriver adapter) | Gen4 x4 (16 GT/s) |
| iGPU | 0000:00:02.0 | renderD129 | | |

No B580 on the bus. gpu1 chain: 00:06.2 -> 06:00.0 (card switch upstream) -> 07:01.0 -> 08:00.0.
Power block from /boot/config/go applied to the whole gpu1 chain (control=on, d3cold_allowed=0).
xe probed both cards cleanly at 11:45:04 (32 GiB VRAM each).

AER on gpu1's path 8-9 min after boot, idle, nothing running:
- 06:00.0 (card's receiver, host->card direction): RxErr 131,449, BadTLP 29, BadDLLP 22; still
  rising ~5.6/s while idle (168 in 30 s). Most of the count came right after boot.
- 00:06.2 (root port's receiver, card->host direction): RxErr 8, Rollover 2, Timeout 3,149
  (replay timer) -- all at boot (11:45:04); flat afterwards.
- non-fatal / fatal: 0 everywhere. gpu0 path: 0 errors.
Reading: the card->host direction (where every old drop started: root-port RxErr then "MMIO
unreliable" within 1 s) is clean after boot; the host->card direction logs a steady stream of
correctable receive errors (link-layer replays, no data loss). The old passive adapter logged almost
nothing while idle and still dropped at first compute, so the counter alone cannot say whether this
adapter holds -- only load can. If the adapter has equalization / gain switches, a lower setting
may remove the host->card errors (hardware question for Ian).

Models moved: Qwen3.8 checkpoints are now on /mnt/models (new XFS disk, sda1); Ornith and the
Flash-Next PLE/hits files are still on /mnt/storage/Models. The Unraid DUAL templates still map
/models -> /mnt/storage/Models and renderD131/renderD130 from before.

## 2. Harness: /mnt/storage/isos/grimoire-runs/bench-1022/dual.sh

`dual.sh NAME TP|PP|G0|G1 MODEL PROJ CTX "ENV.." TESTS [benchy args]` -- start -> test -> stop.
Holds gpu0.lock then gpu1.lock (shared with the other agents), refuses if any container or render
node user exists, resolves the cards by PCI id (tools/gpunode.sh), mounts /models (storage) and
/models2 (/mnt/models). A watcher samples AER every 2 s: STOP (kill the client only, server drains
and is stopped gracefully) on any non-fatal/fatal, any new root-port receive error on gpu1, an xe
error line, or gpu1 config space unreadable; logs the switch RxErr rate every 10 s (logs/NAME.link).

## 3. Runs

### 3.1 gpu1 alone, first compute on the new adapter -- PASS (12:01-12:03)
`dual.sh g1-ornith-mx G1 /models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE mxfp4 32000`, paced upload 2 GB/s,
governor 240 W (last run with it: Ian 10-22 "you can forget the 240w limit it goes to 280w easy").
- ctest: "Paris"; 128-token story at 201.1 tok/s. llama-benchy 0.4.0 c1: coherence PASSED;
  pp512 5,212 / tg128 192.2; pp2048 7,598 / tg128 182.9 tok/s (gpu0 v1.9.0 reference: decode 193.7).
- Link during the run: switch RxErr +55..76 per 10 s (same as idle), root port rx unchanged (9),
  root Timeout +2, uncorrectable 0, no xe line. Server stopped gracefully.
- Harness bug found and fixed: the llama-benchy book server started inside dual.sh inherited the
  GPU lock fds and kept gpu0.lock held after the run (next run refused). It now closes fds 8/9.

### 3.2 Qwen3.8-27B-FP8, PP on both cards -- works, link clean (12:04-12:07)
`dual.sh pp-q38fp8 PP /models2/Qwen3.8-27B-FP8 fp8 16384`, PP_SPLIT=32, MTP off, no governor.
Ready in 54 s. ctest "Paris", story/code text coherent. llama-benchy c1: pp512 825 / tg 20.5;
pp2048 1,436 / tg 20.3 tok/s (10-06 on the old adapter: 19.5). Link: switch RxErr +16..52 per 10 s,
root rx unchanged, uncorrectable 0, no xe line; stopped gracefully.
**Ian's targets (10-22): tg 40-45 tok/s, pp ~3,500 for this model on two cards.** PP cannot reach them
(cards take turns: ceiling ~21 tok/s); TP is the path: half the weights per card per token
(13.5 GB / ~600 GB/s = 22.5 ms -> ~44 tok/s ceiling), both cards on every prefill GEMM.

### 3.3 Qwen3.8-27B-FP8, TP (Megatron-lite) + MTP on both cards (12:08-12:15)
- First attempt (tp-q38fp8-mtp): ctest "Paris" 44.2 tok/s; gen text identical to the PP run (capital,
  story); stopped by the guard at 12:09:02 in llama-benchy's first prompts: gpu1 ROOT PORT logged
  1 RxErr + 1 BadTLP (card->host, physical layer, corrected by replay). Link stayed up (DLActive+,
  BWMgmt-, config space fine), 0 uncorrectable, server drained and stopped gracefully. This is the
  heaviest card->host traffic GRIMOIRE makes: TP prefill all-reduce of fp32 activations.
  Guard relaxed (Ian wants to see dual work): stop at >=5 root-port receive errors in 10 s or >=20 per
  run, any link retrain (BWMgmt+), any uncorrectable, xe error, config space gone.
- Second run (tp-q38fp8-mtp2): coding probe (2,153-token prompt, thinking off): tests=PASS twice,
  decode 78.4 / 78.3 tok/s, 4.40 tokens per verify step, TTFT 4.30 s => prompt only ~500 tok/s.
- Why prompt is slow (src/grimoire.cpp tp_shm_allreduce): per 256-row slot (5.2 MB fp32) a blocking
  D2H + wait, single-thread CPU copy into shm, spin, CPU copy of each peer slot into own pinned buffer,
  H2D, add kernel, wait -- all serial: ~36 ms per 2,048-row all-reduce x 128 per chunk ~ 4.6 s per
  2,048 tokens. Decode: the same blocking round trip 128x per token (GPU drains each time).

### 3.4 Qwen3.8-27B-FP8, TP plain (no MTP) -- 12:12-12:14
ctest "Paris" 32.2 tok/s. llama-benchy c1: pp512 450 / tg 31.9; pp2048 510 / tg 31.5 tok/s.
Link: root-port rx +1 over the run (12 -> 13), uncorrectable 0, stopped gracefully.
Per token 31 ms vs ~22.5 ms of weight reads per card (13.5 GB at ~600 GB/s): the rest is mostly the
128 blocking host round trips of tp_shm_allreduce (GPU drains each time) + eager launches.

**Summary vs Ian's targets (tg 40-45, pp ~3,500):** decode WITH MTP already above (55-78 tok/s),
plain TP decode 32 (short), prompt 450-510 (7x short). Biggest gap = TP prefill all-reduce.

### 3.5 FAILED probe: device-side all-reduce (GPU polls its own pinned host flag) -- gpu0 engine reset
tools/tp_devar_probe.cpp, run 12:15-12:17 via g0run.sh (LIM=120): produced no output (stdout buffered,
lost at the kill) and hung; tune.sh's 120 s timeout killed the container while a polling kernel was
spinning -> xe "Engine reset: engine_class=ccs" on 0000:03:00.0 at 12:16:58. gpu0 recovered (runtime
active, AER 0/0/0), container removed. Likely: one side never saw the other's flag (GPU poll served
from cache, or GPU store not visible), and every one of 2,000 queued kernels then spun its own bound
(2^22 polls ~ seconds). Before any retry: unbuffered output, a global abort flag checked before
spinning, short spin bounds, uncached (LSC L1UC/L3UC) loads for the poll, a 20-iteration first run.
Deferred: prefill first (no device-side spinning needed there).

## 4. Code: pipelined TP prefill all-reduce (branch dual-1022 from longctx-attn-wip 8a7257d)

src/grimoire.cpp: new Grimoire::tp_pipe_allreduce for blocks > GRIMOIRE_TP_SHM_CPU_MAX (prefill), two
ranks. Pieces of 128 rows (GRIMOIRE_TP_AR_ROWS) through rings of 8 (GRIMOIRE_TP_AR_RING): pack to bf16
(GRIMOIRE_TP_AR_BF16=0 keeps fp32) -> device->host on its own in-order queue -> CPU copy into the
CPU-only shm slot -> peer's CPU copy into ITS pinned ring -> host->device on a second queue -> add.
Same transport rule as before (no P2P, no buffer known to two drivers). Both ranks add the same
bf16-rounded partials in fp32, so they keep identical bits; fp32 payload = bit-identical to the old
path. Decode/verify (<= 65536 floats) still use the old CPU-sum path. GRIMOIRE_TP_AR_PIPE=0 = old path.
Also: power governor default OFF for ranked runs too (Ian: cards do 280 W).
Old binaries saved in grimoire-runs/bench-1022/bin-before/. Built 12:25 (dev_build.sh server).

| Qwen3.8-27B-FP8 TP + MTP | old (serial fp32) | pipelined fp32 | pipelined bf16 |
|---|---:|---:|---:|
| coding probe 2,153-tok TTFT | 4.30 s | 2.37 s | 1.58 s |
| coding decode / tok per step | 78.4 / 4.40 | 78.1 / 4.40 | 77.2 / 4.34 |
| benchy pp512 | 450 | 744 | **1,030** |
| benchy pp2048 | 503 | 915 | **1,408** |
| benchy tg128 (@512 / @2048) | 72.6 / 55.3 | 54.0 / 53.0 | 59.0 / 50.0 |
tests=PASS in every run. benchy tg128 with MTP swings 54-73 across runs even with bit-identical
numerics (fp32 pipelined == old): llama-benchy takes a different book passage per run and MTP
acceptance follows the text. Deterministic coding probe: bf16 costs 4.40 -> 4.34 tok/step only.
=> bf16 payload is the default for prefill blocks.
Host time per all-reduce ~5 ms (incl. waiting for the producing GEMM), 21 MB/call at pp2048:
~3.5 GB/s per direction vs ~6.5 practical on gpu1's Gen4 x4.
Next: (1) per-phase timing, then parallel CPU copies; (2) FP8 QKV/gate_up GEMMs (prefill mm()
falls to launch_gemm_xmx for FP8); (3) overlap all-reduce with compute (two half-chunks one layer apart).

### 3.6 Ornith-1.5-35B-A3B-FP8, PP 20/20 on both cards (pipelined-AR build; PP does not use it)
Env as the GRIMOIRE-DUAL template (DEFER_MOE_GATHER, BF16_QKV, BF16_DN_QKV), ctx 32000.
| | MTP on (speculation ON) | plain, SEQ_SLOTS=8 |
|---|---:|---:|
| ctest decode | 61.1 | 113.3 |
| coding 2,153-tok TTFT / decode | 0.35 s / 80.0 (2.17 tok/step) | 0.35 s / 107.5 |
| benchy pp512 / tg128 | 3,971 / 66.8 | 3,880 / 111.0 |
| benchy pp2048 / tg128 | 6,771 / 69.0 | 6,886 / 106.9 |
tests=PASS everywhere. **Ornith on two cards should run PP WITHOUT MTP**: the MoE verify of 2.17 accepted
tokens/step costs more than it saves (verify wall). 10-06 (old adapter, v1.8.x) was 77 tok/s plain.
Link: root-port rx 17 -> 19 during the MTP run, +0 during plain; uncorrectable 0.

### 3.7 Qwen3.8-Flash-Next-NVFP4, PP 24/24 on both cards (12:33-12:39)
GRIMOIRE-DUAL-FLASH-NEXT settings: EXPERT_VRAM_PER_LAYER=340, PLE flat file + hits file on
/mnt/storage/Models/grimoire-ple; model on /mnt/models. Ready in 81 s: per rank 24 layers x 512
experts, 340/layer in VRAM (21.0 GB), 172/layer pinned RAM (10.6 GB); CPU experts on (15 threads).
ctest "Paris" 14.8 tok/s; gen text coherent (story, Fibonacci docstring).
llama-benchy c1: pp512 273 / tg 16.1; pp2048 652 / tg 16.1. Link clean (root rx +0).
Why slow: PP gives capacity, not per-token speed. 09-26 single-card synced decode profile: MoE 26
(CPU experts 17), DeltaNet 12, hyper-connection 9.5, attention 4, PLE 4, head 2.5 ms per token.
Faster Flash-Next needs its own work (TP for the HC residual, fused HC decode, decode graph).

## 5. Two-card summary (10-22, new redriver adapter, no drop, 0 uncorrectable in every run)
| model | mode | pp512 | pp2048 | tg128 c1 |
|---|---|---:|---:|---:|
| Qwen3.8-27B-FP8 | PP 32/32 | 825 | 1,436 | 20.5 |
| Qwen3.8-27B-FP8 | TP plain (old all-reduce) | 450 | 510 | 31.9 |
| Qwen3.8-27B-FP8 | TP + MTP, pipelined bf16 all-reduce | 1,030 | 1,408 | 50-59 (code 77) |
| Ornith-1.5-FP8 | PP 20/20 plain, 8 slots | 3,880 | 6,886 | 111.0 / 106.9 |
| Ornith-1.5-FP8 | PP 20/20 + MTP | 3,971 | 6,771 | 66.8 / 69.0 |
| Flash-Next-NVFP4 | PP 24/24, tiered | 273 | 652 | 16.1 |
Ian's Qwen3.8-FP8 targets: tg 40-45 (met with MTP), pp ~3,500 (not yet: 1,408).
