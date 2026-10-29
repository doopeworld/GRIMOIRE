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

## 6. Why TP prefill stops at ~1,500: gpu1's x4 link does not overlap our two directions
tools/pcie_duplex_probe.cpp (plain copies + kernels, host waits only; 22 MB, 20 reps), 13:06:
| per 22 MB | gpu0 (x8 dock) | gpu1 (x4 M.2 redriver) |
|---|---:|---:|
| D2H copy (own queue) | 1.63 ms, 14.2 GB/s | 3.24 ms, 7.1 GB/s |
| zero-copy read kernel (host->dev) | 1.96 ms, 11.8 GB/s | 3.96 ms, 5.8 GB/s |
| H2D copy | 1.83 ms, 12.6 GB/s | 3.44 ms, 6.7 GB/s |
| D2H + zero-copy read together | 2.41 ms (73% overlap) | 6.13 ms (33% overlap) |
| D2H + H2D copies together | serialized (0%) | serialized (0%) |
Reading: two copies share one copy engine; and on gpu1 the GPU's read requests (host->device pull) queue
behind its own saturating device->host posted writes on the x4 upstream. Without P2P every bf16
all-reduce therefore costs >= ~6.2 ms per 22 MB on gpu1 -> TP prefill no-overlap ceiling ~1,500 tok/s
(measured 1,483), overlap ceiling ~2,700 (compute 0.74 s per 2,153 tokens).
P2P (GPU writes into the peer's VRAM) would make both directions posted writes = full duplex, ~3.3 ms
per call, no CPU copies -- likely how vLLM gets its prompt speed. Asked Ian (his 10-07 rule: no P2P).

### 6.1 P2P: NOT available. Probe mistake: gpu1 engine reset (recovered)
- tools/p2p_query.cpp (device queries only): Level Zero reports **no peer access** 0->1 or 1->0 between
  the two B70s on this board. P2P is not an option; every byte goes through host memory.
- 13:10 extended pcie_duplex_probe on gpu1 with "kernel D2H" (a kernel with 4-64 work-groups storing
  bf16 2 bytes at a time into pinned host memory): 1.69-1.73 SECONDS per 22 MB (0.01 GB/s) -- narrow
  stores to host memory are pathological. The run hit tune.sh's 180 s limit, the launcher killed the
  container with that kernel still running -> xe "Engine reset: engine_class=ccs" on 0000:08:00.0 at
  13:12:49. gpu1 stayed on the bus (config e2238086, runtime active, DLActive+, BWMgmt-, 0 uncorrectable;
  root port correctable +12 during the probe). Same mechanism as gpu0's 12:16 reset.
  RULE: every probe variant needs an internal time budget (stop its reps when over), a tiny first
  rep, and a launcher limit that can never fire while a kernel runs. No narrow stores to host memory.

### 6.1 P2P: NOT available. Probe mistake: gpu1 engine reset (recovered)
- tools/p2p_query.cpp (device queries only): Level Zero reports **no peer access** 0->1 or 1->0 between
  the two B70s on this board. P2P is not an option; every byte goes through host memory.
- 13:10 extended pcie_duplex_probe on gpu1 with "kernel D2H" (a kernel with 4-64 work-groups storing
  bf16 2 bytes at a time into pinned host memory): 1.69-1.73 SECONDS per 22 MB (0.01 GB/s) -- narrow
  stores to host memory are pathological. The run hit the 180 s launcher limit, which killed the
  container with that kernel still running -> xe "Engine reset: engine_class=ccs" on 0000:08:00.0 at
  13:12:49. gpu1 stayed on the bus (config e2238086, runtime active, DLActive+, BWMgmt-, 0 uncorrectable;
  root port correctable +12 during the probe). Same mechanism as the gpu0 reset at 12:16.
  RULE: every probe variant needs an internal time budget (stop its reps when over), a tiny first
  rep, and a launcher limit that can never fire while a kernel runs. No narrow stores to host memory.


## 7. Afternoon 10-22: int8 payload, CPU->VRAM BAR writes, chunked FFN overlap

### 7.1 int8 all-reduce payload (GRIMOIRE_TP_AR_INT8=1, opt-in, LOSSY)
- Each 32 values -> 32 int8 + one fp32 scale (absmax/127): 1.125 B/value instead of 2 (bf16).
  Both ranks add the SAME dequantized terms in RANK order (rank 0's term first), so even a
  fused multiply-add rounds alike and the ranks keep identical bits. Zero-copy receive only.
- Qwen3.8-27B-FP8 TP, no MTP (logs tp-q8 vs tp-zc):
  | | bf16 | int8 |
  |---|---:|---:|
  | pp512 | 1,035 | 1,235 |
  | pp2048 | 1,483 | 1,799 |
  | tg32 | 31.9 | 32.0 |
  | code probe TTFT (2,153 tok) | 1.47-1.54 s | 1.20 s, PASS |
- Numerics: code-probe answer IDENTICAL to bf16 (1,282 chars, both trials). Greedy gen
  (gencmp.py vs PP baseline): bf16 TP = identical on capital/story, code differs at char 173;
  int8 TP: story differs at char 139, code at char 158. Small but real -> stays opt-in.
- int8 per-call all-reduce ~3.65 ms vs bf16 ~5.9 ms (2,153 rows): still link-bound at ~6.8 GB/s
  combined -- gpu1's link moves ~7 GB/s TOTAL whatever the direction mix (GPU-pulled reads).

### 7.2 CPU writes straight into gpu1's VRAM through the 32 GB BAR: FULL DUPLEX
- tools/bar_write_probe.cpp (one card, no device-side waits, internal time budgets): device
  allocation via SYCL memory-export (sycl::ext::oneapi::experimental::alloc_exportable_device_mem,
  opaque fd = dma-buf), mmap'd by the CPU of the SAME process. One GPU driver, no P2P, nothing
  shared with the other card.
- gpu1 results: export + mmap OK; buffer lives in VRAM (GPU reads it at VRAM speed);
  CPU -> BAR 5.46 GB/s (AVX2 non-temporal stores, 64 B write TLPs); GPU sees the data bit-exact
  (0 wrong of 5.77 M); **D2H alone 3.25 ms + CPU->BAR alone 4.30 ms, together 4.13 ms: overlap
  1.05 = full duplex** (zero-copy read path: 0.24-0.33). Root-port RxErr +0, switch +1.
- Both B70s have the full 32 GB BAR (Resizable BAR active).
- Use: the receiving rank's own CPU copies the peer piece shm -> its own VRAM ring (instead of
  -> pinned ring, then GPU reads it over PCIe). Changes one step of Ian's 10-07 transport chain
  ("peer's own pinned -> peer GPU" becomes "CPU -> peer GPU's own VRAM via BAR"); still no P2P,
  no buffer in two GPU drivers. ASK IAN before making it default.

### 7.3 Chunked FFN with streamed all-reduces (GRIMOIRE_TP_FFN_CHUNK, default 768; 0 = off)
- Dense TP layers, prompts >= 1,024 rows: the attention all-reduce goes out whole (packed into a
  device staging region, so its device->host copies never wait behind compute), the FFN runs in
  768-row chunks (post-norm, gate_up, SwiGLU, down) as soon as that chunk's rows are reduced, and
  each chunk's FFN all-reduce is pushed right after its down projection. tp_stream_push /
  tp_stream_wait keep pieces strictly in order with the same slots, flags, sequence and adds as
  tp_pipe_allreduce. FFN weights are dequantized once per layer (gemm_fast predq API) instead of
  once per GEMM call, overlapped with the attention all-reduce.
- BIT-IDENTICAL: the 2,153-token code-probe answer matches the unchunked run exactly (bf16 and
  int8, both trials) -- codecmp.py.
- Receive matters: with the zero-copy read the incoming bytes move INSIDE the add kernel, which
  waits on q behind the next chunk's FFN -> 21 ms/layer, only -2.4 ms. Copy-engine receive (pinned
  -> device ring on qh2d, ring = two layers of pieces) -> 18.5 ms/layer. GRIMOIRE_TP_STREAM_ZC=1 =
  old behaviour.

| Qwen3.8-27B-FP8 TP, no MTP | pp512 | pp2048 | tg32 | code TTFT (2,153) |
|---|---:|---:|---:|---:|
| bf16, unchunked (morning) | 1,035 | 1,483 | 31.9 | 1.47-1.54 s |
| bf16, chunked, zero-copy rx | (769 +-69, transient) | 1,626 | 31.8 | 1.38-1.43 s |
| **bf16, chunked, copy-engine rx (default now)** | **1,039** | **1,805** | 32.1 | 1.22-1.28 s |
| int8, unchunked | 1,235 | 1,799 | 32.0 | 1.20 s |
| int8, chunked, zero-copy rx | 1,242 | 1,954 | 32.2 | 1.12-1.17 s |
| **int8, chunked, copy-engine rx** | **1,240** | **2,269** | 31.9 | 0.97-1.02 s |

All code probes PASS; link: root rx +0-1 per run, 0 uncorrectable.

### 7.4 P2P: works on other 2x B70 machines, NOT possible on the Tower (kernel P2PDMA rules)
- vllm-xpu-kernels #570 (open) + vllm #54768 (open), RFC vllm #54766: a 2-rank all-reduce over
  Level Zero IPC (dma-buf fd exchange, zeMemOpenIpcHandle), handshake in-kernel with system-scope
  atomics, for messages <= 64 KiB (decode); 6.6-10.8 us per call on 2x B70 PCIe Gen5 x8. Larger
  messages stay on oneCCL. Their hazards: volatile polls see stale lines (need CrossDevice-scope
  atomics); peer PCIe writes corrupted the device's OWN writes elsewhere in the same allocation
  (separate page-aligned region, flag page the device never writes). IPC import needs
  libze_intel_gpu 1.15.39122 (1.15.37833 fails, 1.14.37435 one direction) -- ours IS 1.15.39122.
- The Tower: Intel Core Ultra 9 285H, host bridge 8086:7d06, gpu0 on root port #12, gpu1 on #11.
  Linux 6.18 drivers/pci/p2pdma.c allows P2P between root ports only for AMD Zen CPUs or a
  whitelist of Xeon host bridges; 8086:7d06 is neither. xe_dma_buf_attach then drops peer2peer
  and, for a VRAM-only allocation, returns -EOPNOTSUPP -- matches our SYCL peer query ("no" both
  ways). Not fixable by settings; would need a patched kernel AND the SoC routing P2P between its
  root ports.
- Other PRs worth reading (ideas only, nothing linked): #638 fast e4m3->half KV conversion on Xe2
  (fp8-KV prefill +75%); #622-#626 Xe2 fp8 decode GEMV with fused norm prologues.

## 8. 2026-10-24: Tower rebooted with gpu1 on another M.2 port; receive thread + BAR receive

### 8.1 Hardware after the reboot
- gpu0 0000:03:00.0 = renderD129 (root port 00:01.0, x8). gpu1 0000:07:00.0 = renderD130, now
  behind root port **00:06.0** (Gen4 x4) -> card switch 05:00.0 -> 06:01.0. iGPU = renderD128.
  dual.sh finds the chain from sysfs, so its guards follow. Idle host->card RxErr at 05:00.0
  ~1/s (old port ~5/s); root port rx +0-1 per run, 0 uncorrectable.

### 8.2 Ian's vLLM image vs GRIMOIRE, same test (Qwen3.8-27B-FP8, TP=2, no MTP, llama-benchy)
- vLLM (Ian, his latest image): pp4096 1,400, tg128 21, 56 GB VRAM over both cards.
- GRIMOIRE f03a7d4 defaults: pp4096 2,002, tg128 30.9 (int8 payload: 2,790 / 31.4).
- GRIMOIRE VRAM (xe debugfs vram0_mm, whole card): 19.3 GB per card after load, peak 21.5-21.7 GB
  per card with 4K prompts = 43.1-43.5 GB total (tools: bench-1022/vram_peak.sh).

### 8.3 Receive thread (GRIMOIRE_TP_STREAM_RX=1) and BAR receive (GRIMOIRE_TP_AR_BAR=1), both opt-in
- Receive thread: each peer piece -> my ring + its add on a second CPU thread while the main
  thread copies my pieces out; explicit backpressure through consumed flags (base+3072+64*rank,
  raised by every path that takes a peer slot); 8 shared-RAM slots per rank.
- BAR receive: the ring is the card's OWN VRAM (SYCL memory-export allocation, mmap'd by the same
  process), written with non-temporal stores + sfence + one read back; the copy engine only
  copies out. One GPU driver, no P2P -- but it changes one step of Ian's 10-07 chain.
- All code-probe answers IDENTICAL to the morning-10-22 bf16 path (1,282 chars, both trials).

| Qwen3.8-27B-FP8 TP, no MTP | pp2048 | pp4096 | tg128 | code TTFT (2,153) |
|---|---:|---:|---:|---:|
| copy-engine receive (default) | 1,805* | 2,002 | 30.9 | 1.22-1.28 s |
| + receive thread | 1,995 | 1,998 | 31.4 / 30.8 | 1.11-1.17 s |
| **+ BAR receive (thread implied)** | **2,455** | **2,568** | 31.4 / 30.9 | 0.90-0.99 s |
(*10-22 run, tg32.) At 4K the receive thread alone gains nothing: the copy engine, carrying both
directions, is the limit -- BAR takes the incoming direction off it.
| int8 payload (LOSSY, opt-in) + BAR receive | 2,566 | 2,886 | 31.4 / 30.9 | 0.86-0.92 s |
With BAR the int8 payload gains little (TTFT 0.90-0.99 -> 0.86-0.92 s): the link is no longer the
limit; compute and the exposed attention block are.

### 8.4 Next
1. Ian's call: make BAR receive (+ receive thread) the default. Without it, the receive thread alone
   could be the default (+10% at 2K, bit-identical).
2. Hide the attention block (chunk QKV / o_proj; full wavefront) -- the remaining exposed part.
3. Then Qwen3.8-Flash-Next on two cards (Ian's next priority).


## 9. 2026-10-24 evening: Qwen3.8-Flash-Next-NVFP4 on two cards (pipeline 24/24, tiered experts)

### 9.1 What others do (research first)
- vLLM recipe (recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next): TP + expert parallel (TEP); pipeline
  parallel is NOT supported with the n-gram embedding; MTP with 3 speculative tokens.
- devan-carlin/Qwen3.8-Flash-Next-W4A16 (HF) on 4x B70, vLLM fork xpu-qwen4exp: TP4 + EP,
  XPU graphs, ~19 GB weights per card, PLE table (~96 GB) mmap'd from host RAM: 53.4 tok/s decode,
  TTFT ~0.11 s. MTP not wired (GDN causal_conv1d limits on XPU). Their model file replicates the
  hyper-connection mix/combine and PLE on every rank and splits DeltaNet / attention / MoE (QSA
  layers run as dense attention there).
- On two B70s (64 GB) ~1/4-1/3 of each layer's experts must live in host RAM whatever the layout,
  so placement and how RAM experts are computed decide decode.

### 9.2 Found and fixed
1. Routing statistics (GRIMOIRE_EXPERT_HITS) held ~1,092 tokens from 10-10, so VRAM placement
   beyond the experts in that sample was arbitrary: 2.4-2.9 RAM experts per layer-token on rank 0.
   They were saved only at buffer release (never reached by a served process), as a whole-file
   rewrite from row 0 (in a pipeline: rank 1's layers over rank 0's), and a stage without the head
   returned before any save. Now tm_save_hits() adds each process's new counts to its own rows
   (pread/pwrite), after every prompt and every 256 decode tokens, on every exit.
   Two collection runs: RAM experts per call 0.57 (rank 0) / 0.37 (rank 1) at 340/layer.
2. CPU expert threads: each rank started 15 workers, 30 on the 16-core 285H; the idle rank's
   spinning workers took the busy rank's cores. 8 per rank: 0.55 -> 0.07 ms per layer call. The
   default is now cores / ranks on this host (GRIMOIRE_CPU_EXPERT_THREADS overrides).
3. Pipeline prefill chunks: a prompt shorter than 2 x GRIMOIRE_PP_CHUNK went in whole (up to
   4,095 rows): at 380 experts/layer VRAM peaked at 33.6 of 34.2 GB. Now at most PP_CHUNK rows,
   in equal chunks.
4. GRIMOIRE_Q4_TIMING decode totals print every 64 tokens for both ranks (CPU calls, ms per call,
   RAM experts per call).

### 9.3 Measured (llama-benchy c1; ctest = 128-token decode)
| config | pp2048 | pp4096 | pp8192 | tg128 | code probe decode | VRAM peak/card |
|---|---:|---:|---:|---:|---:|---:|
| 10-22 (340/layer, 15 thr, old hits, 240 W gov) | 652 | - | - | 16.1 | - | - |
| 340/layer, 15 thr, timing on (fn-collect) | 631 | 852 | - | 15-16 | 13-15 | - |
| 380/layer, 8 thr | 790 | 990 | - | 24.8 / 23.9 | 23.9-24.5 | 33.6 GB (too thin) |
| 380/layer, GPU reads RAM experts (CPU_EXPERTS=0) | 786 | 954 | - | 24.6 / 24.0 | 20.5 (worse) | 33.6 GB |
| 360/layer, 8 thr, PP_CHUNK=1024 | 643 | 777 | 821 | 23.5 / 22.1 | 22.1-23.8 | 31.3 GB |
| **360/layer, 8 thr (auto), PP chunks <= 2048 equal (default now)** | **771** | **911** | **1,105** | **23.9 / 21.8** | 21.9-23.4 | **32.3 GB** |
Greedy text depends on placement (a VRAM expert runs on the GPU, a RAM one on the CPU, rounding
differs): story prompt differs from char 97 between 340 and 380/layer. All code probes PASS.
Link (gpu1 root port 00:06.0): root rx +0 per run, 0 uncorrectable.

### 9.4 Recommended DUAL-FLASH-NEXT settings (Ian edits the template; needs an image with this code)
GRIMOIRE_EXPERT_VRAM_PER_LAYER=360, GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits (now
real statistics, ~217K / 465K routes per layer for rank 0 / rank 1 layers; backup of the 10-10 file:
bench-1022/flash-next.hits.20261010), GRIMOIRE_POWER_LIMIT_W=0. No thread or chunk variables needed.

### 9.5 Next for Flash-Next
1. Remove the per-layer host sync of the CPU-expert path (cx_ready.wait() drains the queue 24x per
   rank per token): e.g. GPU-side handling when no routed expert is RAM-resident.
2. Expert-parallel TP (both cards on every token): replicate HC mix/combine + PLE, split
   DeltaNet / attention / shared expert, experts by index (as the vLLM recipe and devan-carlin's
   XPU fork do). Today TP refuses MoE and qwen4_exp (tp_mega_refusal).
3. MTP: the NVFP4 checkpoint has an MTP layer (needs the PRE-mixer multi-stream state).

### 9.6 Device timeline and the PLE table (10-24 late)
- GRIMOIRE_TIMELINE=1 now covers forward_qwen4_exp (markers per layer: ple, hc-attn,
  deltanet/attn, hc-mlp, moe; send / head; dumped on each rank's 3rd decode token).
  One token, 39 ms: rank 0 21.6 ms (PLE 5.8, MoE 6.0, DeltaNet 5.2, HC 2.7, attn 1.8),
  rank 1 17.5 ms (MoE 5.3, DeltaNet 5.2, HC 2.6, head 2.4, attn 1.8).
- PLE = layer 1 only, 4.1-5.8 ms: its 16 rows per token are pread from the flat file, which sits
  on a USB SSD (~0.2 ms per random read) and is evicted from the page cache by the model load
  (fincore: 8.4 of 47.7 GiB resident after load).
- Fix: GRIMOIRE_PLE_RAM (default on when MemAvailable > table + 8 GB): at the first PLE use a
  detached thread copies the flat file into process memory (51.2 GB in 66 s from ZFS); rows come
  from memory once it is complete. Rank 0 only (the PLE layer).
- Result (360/layer, everything else as 9.3): pp2048 771 -> 904, pp4096 911 -> 1,081, pp8192
  1,105 -> 1,269, tg128 23.9 / 23.3 / 21.8 -> 25.6 / 24.3 / 22.7. VRAM peak unchanged 32.3 GB.
  While the copy runs (first ~66 s) decode is slower (16.9-18.6 on the code probe).

## 10. 2026-10-24 night: BAR receive is the default (Ian: "yes if it is faster and gives correct results")
- No environment: TP plain pp2048 2,494 / pp4096 2,494 / tg128 31.5 / 30.9; TP + MTP pp2048 2,430 /
  pp4096 2,486 / tg128 48.7 / 48.8 (book text), coding probe 77.4 tok/s (4.34 tokens per step).
- Answers: plain identical to the 10-22 bf16 reference; MTP identical to plain (1,282 chars, both
  trials). GRIMOIRE_TP_AR_BAR=0 restores the copy-engine receive.

## 11. End of 10-24: what the next Flash-Next step should be (measured, not guessed)
- Timeline with every expert on the GPU (GRIMOIRE_CPU_EXPERTS=0, no per-layer host sync):
  MoE 6.6 / 6.9 ms per rank-token, total 20.5 + 18.6 ms -- the SAME 39 ms as the CPU-expert path
  (MoE 6.0 / 5.3 + sync). So the per-layer CPU round trip is NOT the big cost: prefetching RAM
  experts into VRAM would save little. The MoE region costs ~0.25 ms per layer even when nothing
  waits on the host -> next: kernel-level profile of the tiered MoE decode kernels (router GEMV,
  top-k, tmoe gate_up/down, shared expert), then DeltaNet (5.2 ms per 18 layers, BF16 weights).
- Tower idle at the end: no containers, both locks free, book server stopped. dual-1022 = 17c5d1f,
  main = c0aefd0.


## 12. 2026-10-25: oneDNN, DFlash2 on two cards, Radiance (branch onednn-1025)

Ian relayed two developer suggestions: (1) add oneDNN, (2) port Radiance (AMD) to Intel.

### 12.1 oneDNN (tools/bench/onednn/RESULTS-2026-10-25-ONEDNN.md has every number)
- oneDNN v3.13.4 built from source with our compiler (tools/bench/onednn/build_onednn.sh ->
  /mnt/storage/isos/onednn, copied to third_party/onednn, gitignored). No PyTorch, no vLLM.
- One B70: gemm_fast pays a dequantize pass (FP8 ~200 GB/s, BF16 re-layout ~255 GB/s) as long as
  the GEMM at 512 rows; oneDNN decodes in the GEMM: FP8 512 x 34816 x 5120 4.09 -> 1.39 ms.
- src/onednn_gemm.{hpp,cpp} (in libgrimoire_gemm.so), ON by default (GRIMOIRE_ONEDNN=0 off):
  gemm_fast FP8-row / BF16, TP chunked FFN, the 2..16-row verify GEMMs (GRIMOIRE_SMALLM_ONEDNN=0
  off), the DFlash2 drafter's projections and head. Kernel creation per new row class is paid by a
  server warm-up at start (four one-token prompts, GRIMOIRE_SERVER_WARMUP=0 skips); the widest
  weights (N >= 32768) at 512+ rows run in row pieces (1024/768/512/256 + padded tail). Run-time M
  = ocl:ref:any, 700x slower: never.
- regress_all_g0.sh (SUITE_ENV, moved checkpoints found under /models2): 40/44 texts identical to
  10-10; Agnes / Flash-Next 5987-token prompts change wording (rounding, correct text); Muse x2
  differ with oneDNN off too.

### 12.2 Two cards, Qwen3.8-27B-FP8 (Megatron TP), code probe / llama-benchy
| | code tok/s (tokens/step) | tg128 | pp2048 | pp4096 |
|---|---:|---:|---:|---:|
| plain, oneDNN off (morning) | 31.5 | 31.5 | 2,448 | 2,542 |
| plain, oneDNN on | 31.4 | 31.4 | 2,647-2,705 | 2,639-2,667 |
| MTP (10-24) | 77.4 (4.34) | 48.8 | 2,430 | 2,486 |
| MTP now | 87.7 (4.46) | 59.2 / 64.2 | 2,728 | 2,637 |
| DFlash2 now | **120.2 (6.42)** | **69.6** | 2,572 | - |
DFlash2 under TP: tp_mega_refusal no longer refuses it; the drafter loads whole on BOTH ranks and
drafts in lockstep (the load code always intended that). Answers identical to plain TP (codecmp).
Step 52.8 ms: draft 9.0 (was 68: FP8 8-row projections through the batched GEMV, the head as 7
GEMV over the WHOLE 1.27 GB lm_head -- Megatron does not shard it, tp_enabled() is legacy TP only --,
16-row context chunks, one-pass block attention), verify 42.8 (was 48.5 before oneDNN small-M).

### 12.3 Radiance (codeberg StillDeadcode/radiance, Apache-2.0; ~/grimoire-work/thirdparty/radiance-src)
C++/HIP engine for AMD R9700 (RDNA4): core 128K lines, kernels (libr4d) 66K lines HIP gfx12xx only.
2x R9700 (P2P): Qwen3.8-27B MXFP4 222 tok/s greedy with DFlash2 x7; Flash-Next c1 177 tok/s
(TP2 + MTP 3, ~15 ms per step). Porting means its DESIGN, not its kernels:
- done today: DFlash2 x7 under TP (above);
- next: Flash-Next TP -- router replicated, each rank half of EVERY expert's width (no all-to-all),
  DeltaNet / attention heads split, hyper-connections + PLE + QSA indexer replicated, one
  all-reduce after the mixer and one after the MoE. Blockers to solve: NvExpertLayout requires
  I % 128 == 0 and 640 / 2 = 320 (Radiance: 384 / 256 by expert parity); the PLE table (51 GB) must
  not be copied twice into one host (rank 0 computes the PLE row and sends the 10240-float update);
  tiered experts + CPU experts per rank at half width.
- later: 8-bit trunk for Flash-Next (its BF16 trunk is ~6.6 GB read per token), MTP for Flash-Next.

### 12.4 Next steps
1. Verify: 42.8 ms per 8-row step vs 32 ms per plain token (all-reduce latency per block, the
   attention / DeltaNet parts of a verify).
2. Release image with oneDNN (Dockerfile builder stage builds oneDNN; libdnnl.so + Apache-2.0
   license in the runtime stage). A test image grimoire-b70:v1.10-dev (v1.9.0 runtime + this
   branch's binaries) is for Ian's Unraid containers.
3. Template advice (Ian edits): two-card Qwen3.8 DFlash2 = GRIMOIRE_MTP=0,
   GRIMOIRE_DFLASH_MODEL=<Qwen3.8-27B-DFlash2 dir>.
4. Flash-Next TP (12.3).

### 12.5 The 8-row verify, region by region (DFlash2 code probe, GRIMOIRE_TIME_LAYER=all, 25th verify)
Both ranks alike; total 42.4 ms. The weights alone (per rank, 602 GB/s) take ~22.5 ms.
| region | ms | weights floor / note |
|---|---:|---|
| FFN gate_up | 11.49 | 9.5 (FP8, 89 MB per layer per rank) |
| post norm | 6.41 | the mixer all-reduce (~100 us x 64 layers) + residual norm |
| dense FFN (down) | 6.17 | 4.7 |
| DN qkv projection | 2.99 | 2.1 |
| final norm + logits | 2.42 | 2.1 (lm_head 1.27 GB, whole on each rank) |
| DN norm + output projection | 2.24 | 1.25 |
| DN z projection | 2.01 | 1.25 |
| input norm | 1.45 | |
| DN recurrence | 1.21 | |
| attn q projection | 1.14 | 0.84 |
| attn flash 0.88, full attention 0.79, DN conv 0.72, attn kv 0.65, DN gate 0.61, SwiGLU 0.52, DN qk norm 0.49, rope 0.24 | | |
Every all-reduce of <= 65,536 floats (tp_shm_allreduce: 1-row decode and 8-row verify, 128 per
step) is a full host round trip: device->host copy + wait, CPU copy into shared RAM, flag spin, CPU
sum, host->device copy; the GPU idles throughout. gpu0's link is Gen4 x8, gpu1's Gen4 x4 (160 KB
each way ~23 us on gpu1).
- Diagnostic in the source (not yet run): GRIMOIRE_TP_SMALL_AR_TIME=1 prints, every 2,048 calls per
  row class (1 / 2-16 rows), the us per call of: compute still queued, d2h, CPU copy in, peer wait,
  CPU sum, h2d. Adds two waits per call. Run: tools/bench/onednn/tp_ar_time.sh (DFlash2 + plain).
- Then, depending on what dominates: kernel stores to pinned host memory + host polling a flag
  instead of copy + event wait; the CPU writes the sum straight into the card's own VRAM through
  the BAR (exportable memory, as the prefill receive does) instead of a host->device copy; a bf16
  payload halves the bytes but is lossy (opt-in only).

### 12.6 Unraid containers (Ian 10-25: "clean ... and fix all containers to the new settings so i just need to click play")
All ten GRIMOIRE containers recreated from their templates (create_from_template.php = Unraid's
xmlToCommand, label net.unraid.docker.managed=dockerman), stopped, image grimoire-b70:v1.10-dev.
- GPUs bound by PCI slot instead of renderD numbers (those moved on the 10-24 and 10-25 reboots):
  `--mount type=bind,src=/dev/dri/by-path/pci-0000:03:00.0-render,dst=/dev/dri/renderD128`
  (+ `pci-0000:07:00.0-render` -> renderD129 for two-card ones) `--device-cgroup-rule='c 226:* rwm'`.
  `--device` cannot take the by-path name (its colons break docker's parser). Level Zero finds the
  PCI address from the node's major:minor, not from its name: devq (bench-1003/templates-backup/
  2026-10-25/devq.cpp) with gpu0 bound as renderD130 reports 0000:03:00.0 and runs a kernel.
- Second checkpoint disk mapped in every template: /mnt/models -> /models2 (Qwen3.8 checkpoints).
- GRIMOIRE-DUAL renamed GRIMOIRE-DUAL-ORNITH (PP 20/20, 8 slots, no MTP). GRIMOIRE-DUAL-QWEN3.8 =
  multi TP + GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2, GRIMOIRE_MTP=0 (PP_SPLIT/PP_CHUNK
  removed). GRIMOIRE-DUAL-FLASH-NEXT = 360 experts per layer (9.4).
- Backups: bench-1003/templates-backup/2026-10-25 (old templates, docker inspect of every old
  container; fix_templates.py, apply_templates.sh, ctest2.sh, ctest_all.sh).
- Test of every container (bench-1003/ctest_all.sh: start, /health, ctest.py, graceful stop):
| container | result |
|---|---|
| GRIMOIRE-ORNITH | ready 43 s, Paris, 200.5 tok/s, AER unchanged |
| GRIMOIRE-QWEN3.8 | ready 73 s, Paris, 73.0 tok/s, AER unchanged |
| the other eight | NOT tested: Ian stopped the work ("pause") -- test them with bench-1003/ctest2.sh NAME PORT [dual] |

### 12.7 2026-10-26: small all-reduce through the BAR, wavefront TP prompt (lossless)
Same session, Qwen3.8-27B-FP8, both B70s, llama-benchy c1 / code probe (2,153 tokens):
| | pp2048 | pp4096 | TTFT | code decode | answers |
|---|---:|---:|---:|---:|---|
| per-layer path (GRIMOIRE_TP_WAVE=0 GRIMOIRE_TP_PUMP=0) | 2,695 | 2,671 | 0.84 s | 31.5 | reference |
| **wavefront (default)** | **3,399** | **3,114** | **0.67 s** | 31.5 | identical |
| wavefront + int8 payload (GRIMOIRE_TP_AR_INT8=1, LOSSY, opt-in) | 3,516 | 3,239 | 0.63 s | 31.6 | identical |
| int8 payload, no wavefront | 3,158 | 3,259 | 0.71 s | 31.8 | identical |
| DFlash2 + wavefront | - | - | 0.69 s | 125.8-128.5 (6.55/step) | identical |
| MTP + wavefront | 3,276 | 3,172 | 0.69 s | 88.8-88.9 (4.40/step), tg128 57.4 | identical |
- Small all-reduce (f85bd90): plain round trip at 8 rows = gpu1 d2h 28.3 + CPU copy 9.4 + sum 14.2
  + h2d 32.9 us; now pieces + CPU sum into own VRAM through the BAR: ~72 us; DFlash2 verify
  43.0 -> 40.8 ms (code 120 -> 125 tok/s). 1-row decode unchanged (~33 us, latency-bound).
- Wavefront (why): TIME_LAYER at 4,097 rows = mixer ~6.9 ms per layer with the link idle + FFN
  phase ~18.4 ms bound by gpu1's x4 link (84 MB each way per layer). Now every layer runs in the
  FFN's 768-row chunks; layer L+1 chunk c starts when layer L's FFN all-reduce of those rows is in.
  tp_stream_pump() keeps later pieces leaving while a wait spins. pp4096 is still 21 ms per layer
  (ideal ~16): next look at the attention chunks and the stream buffering.
- Containers: all ten verified (bench-1003/ctest_rest.sh, 10-26), see 12.6.
- Each B70 exposes 1 compute + 1 copy engine (zeq); CPU -> BAR ~5.5 GB/s on gpu1 = the 64-byte
  TLP efficiency cap of a Gen4 x4 link.

### 12.8 2026-10-26 afternoon: fused passes, chunk 1024, ESIMD flash on the TP rank, image v1.10-dev2
- Wavefront fused passes (84886bc): pp4096 3,074 -> 3,269 (same session), answers identical.
- Region budget, compute alone, 768-row chunks (GRIMOIRE_TP_WAVE_TIME=1 + NOAR): FFN gate_up 360
  ms, down 187, flash 105, DN qkv 90 / out 65 / z 57, SwiGLU 57, norms 50 + 49 (total 1,181).
- Chunk 1024 from 3,584 rows + ESIMD flash prefill for the 12/2-head TP rank (211f5dc): pp2048 /
  4096 / 8192 3,418 / 3,233 / 3,078 -> 3,460 / 3,356 / 3,263, answers identical.
- Image grimoire-b70:v1.10-dev2 (v1.9.0 runtime + 211f5dc binaries); all ten Unraid containers
  recreated on it and verified (bench-1003/ctest_all.sh, 18:5x): all answer, speeds as before.
- Speed table, llama-benchy c1 --runs 2, every container's own settings
  (tools/bench/onednn/speed_table_1026.sh):
| model | cards | pp2048 | pp4096 | tg128 @2048 | tg128 @4096 |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 plain (TP) | 2 | 3,464 | 3,538 | 31.9 | 31.4 |
| Qwen3.8-27B-FP8 + MTP | 2 | 3,432 | 3,540 | 48.7 | 55.0 |
| Qwen3.8-27B-FP8 + DFlash2 | 2 | 3,417 | 3,492 | 67.5 | 64.8 |
| Ornith-1.5-FP8 (PP 20/20, 8 slots) | 2 | 6,788 | 9,164 | 107.6 | 103.0 |
| Flash-Next NVFP4 (PP 24/24, 360/layer) | 2 | 905 | 1,150 | 25.2 | 24.4 |
| Ornith-1.5 MXFP4 (8 slots) | 1 | 7,983 | 8,842 | 182.6 | 172.1 |
| Qwen3.8-27B GPTQ int4 + MTP 4 | 1 | 1,999 | 2,018 | 69.8 | 65.3 |
| Flash-Next NVFP4 (112/layer) | 1 | 478 | 752 | 21.7 | 22.1 |
  The same two-card plain pp4096 measured 3,356 earlier today: ~5% between sessions.
- gpu1 link: card->host corrected errors come in bursts while IDLE (+5,297 by 17:24, +256 by
  18:37), +0 in runs; one guard stop at 17:49 (5 in 10 s). host->card ~1.5/s idle. 0 uncorrectable.
- Release Dockerfile prepared (scratchpad): oneDNN v3.13.4 built in the builder stage (SYCL GPU
  only, no graph / tests), libdnnl.so + LICENSE / THIRD-PARTY-PROGRAMS in the runtime stage.

### 12.9 2026-10-26 evening: Flash-Next on two cards with Megatron TP + expert-parallel MoE
- qwen4_exp (Qwen3.8-Flash-Next-NVFP4) now runs Megatron TP (`multi TP` / dual.sh TP): DeltaNet and
  attention heads split per rank (heads 24 -> 12, KV 2 -> 1, DN v 48 -> 24, k 16 -> 8), shared
  expert sliced, routed experts EXPERT-PARALLEL inside TP (ranking position k -> rank k % 2, each
  rank holds whole experts of its half; tm_tier_h 2 = the other rank's expert), router / HC / QSA
  indexer / embed / lm_head replicated, PLE (layer 2 only) read on rank 0 from its RAM copy and
  all-reduced (rank 1 adds zeros). Two all-reduces per layer (mixer output, MoE + shared output).
- llama-benchy c1 --runs 2, 170 experts per layer per card (tools/bench/onednn/fn_tp_test.sh):
| measure | PP 24/24 (before) | TP + EP (now) |
|---|---:|---:|
| tg128 after pp2048 | 25.2 | 31.7 |
| tg128 after pp4096 | 24.4 | 30.6 |
| code probe decode | 17.2 | 26.5 |
| code probe prefill | 706 | 682 |
| pp2048 | 905 | 713 |
| pp4096 | 1,150 | 664 |
  Answers: capital identical, story identical for 144 chars and code for 514 chars, then other
  wording, coherent. Load 79 s (PLE RAM copy 51.2 GB in 58 s on rank 0 only). AR ~1.2 ms per call.
- Decode per token (synced timers): PLE 3, HC 7.8, DN 9.5, attention 6.5, MoE 22, head 2.4 ms.
- Prompt regression, cause measured (GRIMOIRE_Q4_TIMING): only 2.3 GiB free per card -> 768-row
  prompt chunks; every chunk re-streams the pinned-RAM experts (MoE ~750 of ~1,050 ms per chunk);
  HC 224 ms per chunk is replicated on both ranks (~1.5 ms fixed per mix + ~1.1 us per row;
  GRIMOIRE_ONEDNN=0 is not better: 252 ms). NV=140 per card: pp 793 / 1,040 but tg 22.9 (the PLE
  then stays on disk: 28.8 GB pinned + 51 GB > RAM) -> keep 170.
- Fix BUILT (this commit), not run yet: layer-major qwen4_exp prefill. Per layer the mixer runs in
  sub-chunks (q4_sub_rows, default 768, GRIMOIRE_Q4_SUBCHUNK) and the routed experts run ONCE over
  all rows of the call (tiered_moe_prefill with route groups, launch_moe_unpermute_range); the call
  span comes from q4_prefill_split (GRIMOIRE_Q4_SPAN 8192, GRIMOIRE_BUDGET_DEBUG=1 prints it);
  both ranks agree on the sub-chunk at load; expert ring 64 slots under TP (GRIMOIRE_EXPERT_RING);
  GRIMOIRE_HC_TIMING=1 splits the hyper-connection mix into its passes.
- Test: tools/bench/onednn/fn_lm_test.sh (two cards TP, then one card G0 as regression check).
- Dockerfile: oneDNN v3.13.4 built in the builder stage, libdnnl.so + license files in the runtime
  stage; image grimoire-b70:v1.10 built (not validated, not released).

### 12.10 2026-10-27: Flash-Next prompt -- layer-major tested, the HC inject GEMM, sub-chunks per call
- Layer-major prefill (12.9) tested: two cards pp2048 / pp4096 713 / 664 -> 974 / 1,183 (above the old
  PP 24/24 905 / 1,150); one card 464 / 654 (regressed from 752: a fixed 768-row cut, see below).
- The hyper-connection "fixed cost" (~1.5 ms a mix) was the injection projection: hc.inject is
  [4 x 10240] BF16 and the general GEMM (f32->bf16 + launch_gemm_xmx) took ~1.5 ms a call at N = 4
  whatever M (GRIMOIRE_HC_TIMING, two cards, 4,097 rows: inject 898 of 1,293 ms HC).  New kernel
  launch_hc_inject_rows (one work-group per row, the 4 weight rows from cache): 40 ms.
  GRIMOIRE_HC_INJECT_GEMM=1 = the old GEMM.
- q4_sub_for(M): a call's sub-chunk is the largest multiple of 128 that q4_prefill_split's budget
  still holds next to the call's full-width rows and one route group (TP ranks agree on the budget
  at load); M itself when the call fits.  One card is one piece again; two cards 4,096 rows run in
  sub-chunks of 1,024 (1.94 GiB of gpu1 free at the end of a call).  GRIMOIRE_Q4_SUBGROW=0 = the
  fixed q4_sub_rows.
- Decode HC kernels: B (down + inject) one work-group per output row with K split over its
  sub-groups (was one sub-group walking a 20 KB row: ~85% of the card idle), C (up + mean) the 4
  stream dots in one walk.  GRIMOIRE_HC_CHECK=1 compares with the old ones (kept as *_sg,
  GRIMOIRE_HC_SGROW=1): max diff ~1e-6 at magnitudes 2-70.  tg unchanged within noise (+-2 tok/s).
- Final, llama-benchy c1 --runs 2 (tools/bench/onednn/fn_final.sh):
| Flash-Next NVFP4 | pp2048 | pp4096 | tg128 @2048 | tg128 @4096 | code probe prefill |
|---|---:|---:|---:|---:|---:|
| two cards TP, 10-26 evening | 713 | 664 | 31.7 | 30.6 | 682 |
| two cards TP, 10-27 morning (layer-major) | 974 | 1,183 | 31.2 | 30.1 | 883 |
| two cards TP, 10-27 final | 1,263 | 1,463 | 31.0 | 30.2 | 1,182 |
| one card, 10-27 morning | 464 | 654 | 21.95 | 21.75 | 485 |
| one card, 10-27 final | 526 | 804 | 22.2 | 21.9 | 580 |
  Answers: capital identical, story / code coherent (other wording after 97-460 chars: float order).
- Two cards, 4,097 rows, where the time goes now (GRIMOIRE_Q4_TIMING, prefetch off): MoE 1,622 ms
  (gpu1 streams its 86 RAM experts a layer, 227 MB, over the x4 link: ~32 ms a layer), HC 534,
  DeltaNet 421, attention 281, PLE 17.
- Tried, OFF by default:
  - GRIMOIRE_EXPERT_PREFETCH=<min rows>: the next layer's first 64 RAM experts copied while its
    mixer runs (ring positions run on across layers, tm_slot_ev per slot).  Slower in every form:
    issued before the MoE all-reduce pp4096 1,094, after it 1,073 (base 1,439); a 73-row call wasted
    the copies (MoE 339 -> 1,381 ms).  Likely the expert copies and the all-reduce copies share the
    copy engine, and the all-reduce is on the critical path.
  - GRIMOIRE_EP_COLD_SHARE=0.67: rank 0 (gpu0, x8) owns 2/3 of the cold experts (115 vs 57 RAM
    experts a layer): pp4096 1,648 but pp2048 1,002 and decode after pp4096 27.6 (rank 0's CPU
    experts do more per token).  --runs 1 numbers; worth a cleaner A/B before any default.
- dual.sh: a G0 run takes only gpu0.lock and ignores gpu1's own link rules; the book server no
  longer holds the caller's stdout (a `dual.sh | grep` pipeline hung); test "pleready" waits for the
  PLE table's copy into RAM.  gpu1's root port logs RxErr in bursts while idle (~440 in 4 min,
  0 uncorrectable, link up): two-card runs use RP_WIN=1000 RP_RUN=1000000 (Ian 10-25: run anyway).

### 12.11 2026-10-27 evening: decode copies by kernel (ksend), all-reduce time split, CPU pool pinning
- Two cards, one decode token (GRIMOIRE_TIMELINE): MoE 12.2, DeltaNet 6.5, attention 2.4, HC 4.1,
  PLE 4.4 (rows still from disk in that run), tail 2.5 ms.  GPU work ~13 ms; the rest is waiting.
- Small all-reduce split (GRIMOIRE_TP_SMALL_AR_TIME, now four parts): copy out 38-56 us (10 KB,
  copy-engine memcpy + event wait), peer wait 72-171 us, CPU sum 1.2, copy in 12 us.
- ksend / kwait: the copy out (and the CPU experts' route + block input) is now one work-group
  kernel on the compute queue writing pinned RAM, then a tag into a pinned flag the CPU polls.
  GRIMOIRE_KSEND_CHECK=1 compared 81,920 copies a rank with copy-engine copies: 0 differed.
  Copy out 38-56 -> 13.9 us.  Two cards tg128 31.0 / 30.2 -> 34.1 / 32.9 (+-0.2), pp unchanged;
  one card 22.2 / 21.9 -> 23.6 / 22.0.  GRIMOIRE_KSEND=0 = the memcpy.
- The waits for the peer are the CPU experts: 0.40-0.46 RAM experts a layer call per rank,
  ~0.25-0.3 ms each (GRIMOIRE_Q4_TIMING), while the GPU part of the MoE is ~30 us -- the rank with a
  RAM expert arrives late.  Layers with none no longer copy and add a zero partial.
- Two runs of the same build give different story/code text after 97-141 chars (two cards and
  one card): decode is not bit-repeatable run to run, so answer identity is no test here.
- BUILT, NOT TESTED (Ian needed the GPUs): b70::hybrid_pool_cpus pins each rank's CPU-expert pool
  threads to its share of the P-cores (0-5) and E-cores (6-13), the low-power E-cores 14-15 left
  out, the calling thread unpinned (GRIMOIRE_CPU_PIN=0 = off).  Test: bench-1022/fn_pin.sh
  (Q4 timing CPU experts ms per call, then llama-benchy).

### 12.12 2026-10-27 evening: CPU pool pinning measured, CPU-expert bench, pool start race, CPU partial folded into the MoE all-reduce
- Pinning (b70::hybrid_pool_cpus, default on), same session, two cards, llama-benchy tg128:
  pinned 37.59 / 35.63 and 37.69 / 35.91 (repeat), unpinned (GRIMOIRE_CPU_PIN=0) 31.80 / 32.68;
  prompts unchanged (pp2048 1,256-1,320, pp4096 1,436-1,461).  CPU experts per layer call
  (GRIMOIRE_Q4_TIMING): 0.069-0.088 ms pinned vs 0.115-0.139 ms unpinned; MoE region per token
  16.7-17.6 vs 21.8-23.0 ms (synced timing).
- tools/cpu_expert_bench.cpp (CPU only, no GPU): Flash-Next-sized experts (H 2560, I 640, 2.77 MB)
  in a 1 GB pool, one random expert per call at the decode cadence.
  - One thread: P-core 21.6 GMAC/s with the row in L1, 11.7 streamed from RAM (6.6 GB/s);
    E-core 9.6 / 9.5; low-power E-core 4.5 / 3.9.  Streaming, not arithmetic, is the limit.
  - Pool: one rank, 7 pinned threads: median 65 us (42 GB/s); both ranks at once: 80 us each
    (34 GB/s each, the DRAM limit); unpinned: medians 75-102 us but p99 1.2-1.7 ms and max 5-9 ms
    (descheduled workers); 3 P-core threads 125 us; two experts a call 132 us.
  - In the engine one RAM expert looked like ~170-200 us pinned, but that figure is unreliable:
    cx_ms / cx_calls also counts the short-prompt rows (tiered_moe_cpu_rows: M calls each) while
    cx_nexp counts decode experts only.  GRIMOIRE_CX_STATS=1 now times decode calls alone.
  - The pool reads the GPU's pinned memory (sycl::malloc_host, 4 KB pages, no huge pages) as fast
    as plain memory: 67.5 vs 65.9 us; plain memory on 4 KB pages 66.8 us.  Software prefetch 2/4/8
    rows ahead: 75-85 us (slower) in the bench, no change in the engine (37.54 / 36.22 tg) --
    removed.
- CPU experts' partial folded into the MoE all-reduce (GRIMOIRE_CX_FOLD, default on; =0 old way):
  the GPU partial's copy out (ksend) is queued before the CPU experts run, and the CPU adds its
  partial on the host (same fp32 add as the device's, same order) instead of host->device copy +
  device add + copy out after the CPU.  GRIMOIRE_CX_FOLD_CHECK=1: 12,288 all-reduces a rank
  compared with the old device add, 0 differed.  tg128 fold 38.01 / 36.11 and 37.99 / 35.38,
  no fold 37.63 / 35.78: about +1%.
- Busy threads during decode (tsample.sh / tsample2.sh, per-thread CPU over 1-2 s): each rank's
  decode thread (rank 0's scheduler thread, rank 1's main thread) is busy all the time and was
  NOT pinned: it ran on the low-power cores 14-15 (2.5 GHz) or on a pinned worker's CPU (rank 1's
  on its own worker's CPU 5, or on rank 0's CPU 8), and the PLE table copy thread (about 60 s at
  100% kernel time) on CPU 3.
- Decode thread pinned (b70::ScopedCpuPin around forward_qwen4_exp, cpus[0] of the rank's list:
  rank 0 CPU 0, rank 1 CPU 3; GRIMOIRE_CPU_PIN_CALLER=0 = off); the PLE copy thread and the
  ple_fetch_rows workers unpin themselves.  llama-benchy tg128 pinned 38.39 / 36.69 and 38.74 /
  36.31, unpinned 37.94 / 36.23.  GRIMOIRE_CX_STATS (decode calls only), pinned vs unpinned:
  one-expert calls 0.121-0.147 vs 0.201-0.208 ms, all calls 0.161-0.202 vs 0.236-0.271 ms, the
  caller's wait for the last worker 33-58 vs 114-139 us; calls over 1.6 ms 179-324 vs 309-786 of
  ~15-16K.  14-16 calls with RAM experts a token a rank, 1.42-1.48 experts a call.
- Remaining CPU-expert loss: 55% of calls are under 100 us (the bench's speed) but ~1-2% stall
  1.6 ms or more (a worker descheduled while it holds an item); the spinning workers are always
  runnable, so any other task that wakes on their CPU takes a slice.  The caller's wait for the
  last worker: 4.6-6.8 us a call in the bench, 33-58 us in the engine.
- Pool start race, FIXED: a worker read its starting generation inside its own new thread; one
  that started only after the first run() took that job as already seen, never ran it, and the
  caller waited forever (the bench with 14 threads hung).  start() now hands every worker the
  generation it starts from.  The engine's first call comes long after load, so it never hit it.

### 12.13 2026-10-27 night: Qwen3.8-27B-FP8 table for Ian (1 / 4 / 8 users, pp 4K / 8K / 16K); PAUSED
- Ian asked for Qwen3.8 at pp 4096 / 8192 / 16K, tg 128, 1 / 4 / 8 users: tools/bench/onednn/
  q38_conc.sh (TP, plain decode, GRIMOIRE_SEQ_SLOTS=8, ctx 18432) and q38_spec.sh (one user, MTP
  and DFlash2).  Full table: tools/bench/onednn/RESULTS-2026-10-27.md.
- Plain: pp4096 3,530 (c1) / 3,375 (c4) / 3,381 (c8); tg128 total 29.5 / 97.0 / 165.0.  pp16384
  ~3,000 at every concurrency; tg total 27.4 / 59.5 / 41.9 -- 16K x 8 users is the weak row
  (long prompts admitted between the other users' decode steps).  Batched decode under Megatron TP
  works (first time measured).
- One user: MTP tg128 56.2 / 52.9 / 51.0, DFlash2 59.3 / 67.1 / 50.5 (pp 4K / 8K / 16K); code
  probe MTP 88.8-89.2 (4.40 tok/step), DFlash2 126.0-128.4 (6.55 tok/step), tests PASS.
- Ian on VRAM ("25g ... now 30g per gpu"): 8 slots x 18,432 tokens of KV + the prompt scratch
  sized from the free VRAM at load (prefill chunk cap 18,176 tokens); no errors, no eviction.
- PAUSED by Ian ("save everything and push to git we will resume tomorrow"): no containers, no
  test scripts, GPUs idle.  Resume with: CPU-expert stalls (workers block when idle instead of
  spin-yield), timeline (bench-1022/fn_tl.sh), then Flash-Next MTP + a qwen4_exp verify path.

### 12.14 2026-10-28: Flash-Next two-card decode -- steady-state timeline, per-kernel marks, four kernel fixes
- Tower rebooted 06:41 (same GPU layout; gpu1 switch RxErr ~45/s idle, corrected, root port
  clean).  GitHub checked first: every branch pushed; 3 old probe tools committed (763e869).
- Steady state (bench-1022/fn_tl2.sh, answers only after the PLE table is in RAM): one token
  23.6 ms (42 tok/s, short context); small all-reduce 28 us a call (copy out 5.4 + peer 10 +
  sum 1 + copy in 8); CPU experts 12-13 calls a token a rank, 0.127 ms a call, no call over
  1.6 ms.  fn_tl.sh (10-27 and today) dumped a token read while the 51 GB PLE copy was still
  running (L01 ple 9.6 ms, 3 ms deltanet spikes, AR 75-240 us): not representative.
- tools/kgap_probe.cpp (gpu0, no model): a kernel slot costs the GPU 0.6-1.0 us when the host is
  ahead (empty / 64 WG / 512 WG / 1 WG x 1024 / 10 KB copy).  Fusion is NOT the lever: ~1,440
  kernels a token = ~1.4 ms; the time is inside slow kernels.
- Per-kernel marks (GRIMOIRE_TIMELINE_LAYER=4 / _LAYER2=3, "Lnn .step"; fn_tl3.sh), us, rank 0:
  HC mix: combine+norm 10.6-13, down+inject 14.2-14.8, up+mean 14.2-14.7 (x 96 a token).
  DeltaNet L04: qkv 55.2 (26.2 MB, 475 GB/s), conv 8.8, ab 2.5, gates 2.3, step 5.8, z 28.8
  (547 GB/s), norm 3.4, out 31.5 (500 GB/s), all-reduce 20.
  Attention L03: q 64.7 (487 GB/s), k 4.4, v 4.2, rope 5.6, kv 2.8, attn (indexer + flash) 47.5,
  gate 1.1, o 28.5, all-reduce 19.7.
  MoE: router 6.4, top-k 42.9 (!), ksend 2.8, vexp 1.7, VRAM experts gate_up 35-39 + down 33-38
  (~12 MB, ~180 GB/s), shared gu 7.4, swiglu 1.1, shared down 10.3, gate 2.4, scale 1.1, add 1.0,
  AR send 1.8-2.3, all-reduce 14-46 (rank 1 waits for rank 0's slower experts).  head 2.4 ms.
- Fixes, each with a switch back to the old code:
  1. Router top-k for 512 experts on the ESIMD register kernel (GRIMOIRE_TOPK_ESIMD=0; 512 fell
     to the generic sub-group SLM path).  GRIMOIRE_TOPK_CHECK=1 runs both on the same logits:
     0 of 8,192 calls a rank differed (experts and weights bit for bit).
     llama-benchy (client on CPUs 14-15 both sides): tg128 38.47 -> 40.81 (2K), 36.36 -> 39.14
     (4K); pp2048 1,235 -> 1,268, pp4096 1,453 -> 1,513.
  2. Head split over the vocabulary under Megatron TP for Qwen4-Exp (GRIMOIRE_TP_HEAD_SPLIT=0):
     each rank its rows, local argmax, (value, index) pairs exchanged over the TP sockets
     (tp_head_pick: highest value, lowest index on a tie = launch_argmax's rule).  The first
     build skipped it: tp_peer_fd is empty during build() (pp_connect runs at its end).
  3. Lean tiered-expert kernels (GRIMOIRE_TMOE_LEAN=0): gate_up work-groups of a slot routed to
     no VRAM expert of this rank leave before staging; down stages only the slots it reads, R=4.
  4. HC combine+norm with the lane's values in registers (GRIMOIRE_HC_NORM_REGS=0).
- Greedy text moved between runs of the SAME build: every served run adds its routes to
  GRIMOIRE_EXPERT_HITS, so the next load splits VRAM / RAM experts differently (CPU vs GPU
  summation order).  Exactness runs use a frozen copy (fn_ex4.sh: flash-next.hits.frozen-1028
  copied to flash-next.hits.ab before every run).
- RESULT (fn_ex4.sh, all four on vs all four off, frozen placement, client on CPUs 14-15):
  old twice: the three answers identical; new twice: identical to old, all three (capital,
  160-token story, 160-token code).  llama-benchy new: pp2048 1,348, tg128 42.71; pp4096
  1,477, tg128 40.33 -- against old (fn_ab3.sh) 1,235 / 38.47 and 1,453 / 36.36: decode +11%.
  Top-k alone gave 40.81 / 39.14, so head split + lean experts + HC norm add 3-5% on top.
- dual.sh: CLIENT_CPUS=14,15 (opt-in) keeps llama-benchy off the CPU-expert pool cores;
  old code with it: tg128 38.47 / 36.36 = 10-27's best (38.39-38.74 / 36.31-36.69), so the
  client was not what cost decode.
- 5. GX (GRIMOIRE_TMOE_GX=0 = old): the lean kernels alone left gate_up / down at ~35 us each --
  the real cost was the SLM copy of the activations, read with lanes 128 bytes apart (all 16
  lanes on one SLM bank).  Reading them from L1 as float4: gate_up 22-26 us, down 19-27 us, MoE
  region 7.27 -> 6.18 ms a token, one token 19.4 ms.  Same three answers as the old code
  (frozen placement).  llama-benchy tg128 44.60 (2K) / 42.18 (4K), pp 1,292 / 1,474.
  DAY TOTAL, Flash-Next two cards: tg128 38.47 -> 44.60 and 36.36 -> 42.18 (+16%), exact.

### 12.15 2026-10-28 (rest of the day): everything on main, B60 fixes, multi-user MTP on two cards, v1.10.0
- 08:12 gpu1 dropped off the bus while idle (fatal MalfTLP, config space ffff); Ian rebooted
  twice.  Since then: builds on CPUs 6-13 only (bench-1022/tower_build.sh), every command logged
  with a timestamp (bench-1022/logs/claude-commands-1028.log), nothing runs after a drop until
  Ian says so.
- Outside tester (B60 x2, Discord): branches all over the place, oneDNN not on main.  Fixed:
  the tested two-card work merged into main (8bae8c7), README says main is the current code
  (a048f6c).  Work goes to main the same day it is tested; no long-lived side branches.
- bf9d8db, from the B60 report: bridge "fallback" lines only with GRIMOIRE_BRIDGE_DEBUG=1 (they
  were false alarms in pure mode); builds for BMG-G31 (B70) + BMG-G21 (B60 / B580) by default;
  dev_build.sh GRIM_BUILD_CPUS; the KV cache is allocated after the weights with the slot count
  fitted to the free VRAM (GRIMOIRE_SLOTS_FIT, GRIMOIRE_VRAM_GB, GRIMOIRE_VRAM_RESERVE_GB) and
  zero-filled.
- ae25f19: two cards run MTP for several users at once (PPRequest kind 5 = one speculative batch
  step on every rank; GRIMOIRE_SPEC_MAX_SEQS default 2 under TP, 4 on one card): Qwen3.8-27B-FP8
  1 / 2 / 4 / 8 users 56.5 / 62.9 / 95.0 / 161.9 tok/s total.
- 3fb9df7 image v1.10.0: oneAPI DPC++ 2026.2.0, compute-runtime 26.35.39758.10 + IGC 2.41.5,
  oneDNN 3.14 (3.13.4 before), libdnnl.so.3 copied whatever its minor version.  Same code built
  both ways, two cards, plain decode: new 30.25 tok/s, old 29.25 (no regression).  Released as
  v1.10.0 (grimoire-b70-v1.10.0.tar.gz, 736,685,487 bytes, sha256
  6786c9c1595ea677878823346d8a3ef36fda6c4dd5489697f31b93dde7355efa); README download + `latest`.
- Unraid: four containers kept, all on grimoire-b70:v1.10.0 (ORNITH 203 tok/s, QWEN3.8 78,
  DUAL-QWEN3.8 47.6 short test, DUAL-FLASH-NEXT 53.5); the other templates stay in
  templates-user.
- Parked on their branches (rebased on main): flashnext-mtp (Flash-Next MTP, verify 285 ms a
  step -- slower than plain), mtp-batch-draft 2cddfa1 (one MTP head pass per depth for all users;
  compiled, not run).
- Tester notes kept for 10-29: prefill at 32K ~900 on B60s ("missing attention libraries");
  K4 step on B60 = draft 4.2 + check 35.4 + commit 1.1 ms; ideas: bf16 all-reduce in the check,
  fuse small kernels, drafts kept on the GPU, adaptive K / user limit; P2P on PCIe 5.0 boards.

### 12.16 2026-10-29: accuracy against the original model; INT8 at FP8 speed
- Tester's friend: "cosine only 0.95".  New tool tools/bench/accuracy (879a635): GRIMOIRE's
  decode-path logits for given token ids (grimoire --score-ids / --score-out) against the BF16
  original in transformers on the CPU (grimoire-ref:cpu, a separate measuring container), cosine /
  KL / top-1.  Results table in tools/bench/accuracy/README.md.  Engine itself: INT8 from BF16
  cosine 0.9996, top-1 98.4% (FP8 KV cache included).  Official FP8 checkpoint as --proj fp8:
  0.9968 / KL 0.030 / 94.9% -- its 128x128 block scales are unpacked and every weight rounded
  again per row; as --proj int8: 0.9981 / 0.018 / 98.0%.
- Found on the way (879a635): a quantized output head taken as bf16 (--proj bf16 or
  GRIMOIRE_LM_HEAD_BF16=1) gave garbage for GPTQ (cosine 0.03) and failed to load for native
  MXFP4; anything but plain bf16 now goes through the decoder.
- INT8 had none of FP8's fast paths (two cards: decode 23.2, 4 users 46.1, pp4096 2,448).  Now
  (commit "INT8 at FP8 speed"): oneDNN s8 weights decompression for prompts, the ESIMD
  row-streaming decode GEMV with an INT8 decode, the FP8 small-M kernel with an INT8 decode
  (fallback when oneDNN declines; the sandwich batched decode; the batched output head of a
  verify).  Tables in the commit message: two cards FP8 / INT8 pp4096 3,531 / 3,481, tg128
  28.97 / 29.10, 4 users 93.75 / 90.72; one card MTP verify 54.2 (FP8) / 53.3 (INT8 oneDNN) /
  53.6 (INT8 small-M) ms a step -- memory-bound, equal.
- Two cards never use oneDNN or the small-M kernels at 2..4 rows: tp_enabled() sets exact_verify
  in prefill(), and those rows go to launch_gemv_batch (SIMT, MB=4) for every format.
- The 10-29 morning "4 users -13%" for INT8 was mostly run-to-run spread (FP8 97.30 -> 93.75
  between two runs of the same code).
- Trap: g0run.sh / tune.sh EXTRA_ENV is ONE K=V PER LINE (b70run.sh GRIM_ENV); "A=1 B=1" made
  GRIMOIRE_MTP="1 B=1" and the debug flag never existed (a wasted gpu0 run).  dual.sh's env
  argument is space-separated and fine.
- Two cards + MTP (bench-1022/i8v4.sh), FP8 / INT8: code 88.1-88.5 / 84.9-85.4 tok/s (4.40 /
  4.21 tokens a step), pp4096 3,522 / 3,566, tg128 53.45 / 53.60, 4 users 96.61 / 96.15.
  README now says: run the FP8 checkpoint with --proj int8 (main; the v1.10.0 image is still slow
  at INT8).
- Next: image with INT8 -> DUAL-QWEN3.8 container on --proj int8; pp32768 on two cards (tester:
  ~900 at 32K on B60s); MTP multi-user items from 12.15; W8A8 prompts (INT8 DPAS = 2x bf16, lossy,
  accuracy check decides); native block-scaled FP8; native NVFP4.
