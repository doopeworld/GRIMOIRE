# HANDOFF 2026-09-25 (evening) — pure GRIMOIRE prefill vs vLLM: resume here

**Read this first tomorrow.** Supersedes the "Next steps" of
`HANDOFF-2026-09-25-PURE-B70.md`. GPUs are idle; nothing is running.

## Where we are

| Qwen3.8-27B MXFP4, ONE B70 (gpu0), pure SYCL / Level Zero | this morning | now |
|---|---|---|
| 4088-token prompt prefill | 3.235 s (1,264 tok/s) | **2.12 s (≈1,925 tok/s)** |
| 5987-token prompt + 24 generated tokens | 5.69 s | **4.12 s** |
| vLLM on one B70 (int4, llama-benchy pp4096) | 2,015.6 tok/s ≈ 2.03 s | — |

**Gap to vLLM: ~90 ms (≈4.5%).** Output is correct: the generated text is
byte-identical to the reference in every run (two runs per change).

Commits today on `main` (all pushed to github.com/doopeworld/GRIMOIRE):
`a8fdb0a` → `d02a5a6` → `d943651` → `c983eae` → `81ed403` → (this one).

## How to measure (Tower, `/mnt/storage/isos/grimoire-fuse`)

```bash
cd /mnt/storage/isos/grimoire-fuse
# build engine + bin/libgrimoire_gemm.so (~2 min, no GPU used)
docker run --rm --entrypoint bash -v $PWD:/grimoire -w /grimoire my-vllm-xpu:latest \
  -lc "bash tools/build_grimoire_only_b70.sh /grimoire"
Q=/models/Qwen3.8-27B-MXFP4-GRIMOIRE
# prefill speed (log: /tmp/grim-pp.log, look for elapsed=)
GPU=gpu0 LIM=600 bash tools/tune.sh pp /grimoire/bin/grimoire -m $Q --proj mxfp4 \
  --ctx 8192 -p "$(cat p4096.txt)" -n 1
# per-region budget: EXTRA_ENV="GRIMOIRE_TIME_LAYER=all"
# GEMM dequant vs GEMM split: EXTRA_ENV="GRIMOIRE_FAST_GEMM_TIMING=1"
# correctness (run twice, compare with the saved reference):
GPU=gpu0 LIM=600 bash tools/tune.sh c1 /grimoire/bin/grimoire -m $Q --proj mxfp4 \
  --ctx 8192 -p "$(cat real4k_ascii.txt)" -n 24
cmp <(awk "/norms /{f=1;next} f" /tmp/grim-c1.log | grep -vE "^\s|^$|prompt=") \
    ref-sherlock-5987-n24.txt && echo IDENTICAL
```
`tools/tune.sh` = pure launcher (only B70 render nodes; never the B580).

## Region budget now (4088 tokens, `GRIMOIRE_TIME_LAYER=all`, ms)

| region | ms | note |
|---|---|---|
| FFN gate_up + SwiGLU (fused) | 767 | GEMM |
| FFN down | 356 | GEMM |
| DN recurrence | 218 | `deltanet_q4_impl` |
| DN qkv projection | 156 | GEMM |
| **attn flash** | **147** | ~1/7 of its matrix-unit bound — biggest non-GEMM lever |
| DN norm + out projection | 119 | GEMM + fused norm |
| DN z projection | 92 | GEMM |
| attn q proj | 62 | GEMM |
| input norm | 42 | |
| attn tail (gate + o_proj) | 40 | |
| DN conv | 32 | |
| attn kv proj | 24 | small N, inefficient |
| DN qk norm / DN gate proj / rope | 11 / 11 / 10 | |

All 384 GEMMs: **dequant 160 ms + GEMM 1,382 ms (≈144 TFLOP/s, ~79% of bf16 peak)**.
The GEMM itself is near its practical limit; the gap must come from flash,
DeltaNet, dequant and small-N / elementwise work.

## What changed today (all in pure GRIMOIRE; A/B switches)

- `src/gemm_fast.cpp` (built into `bin/libgrimoire_gemm.so`, 256-GRF):
  - streaming MXFP4 dequant (327 → 160 ms; `GRIMOIRE_DEQUANT_TILED=1` = old)
  - GEMM: one launch over all M rows with bounds-checked 2-D block loads/stores
    (no padded tail copy); work-group order GROUP_M=4 (`GRIMOIRE_GEMM_GROUP_M`);
    L1 prefetch one K step ahead (`GRIMOIRE_GEMM_PREFETCH`, default 1)
  - padded N (N % 256 != 0, e.g. DeltaNet a|b gate N=96): 41 → 11 ms
  - fused SwiGLU epilogue, gate/up interleaved in the scratch
    (`GRIMOIRE_NO_FUSED_SWIGLU=1` = three-step path)
  - DeltaNet recurrence, 4 lanes per state row (`GRIMOIRE_DN_PREFILL_NOQ4=1`)
  - flash prefill: hardware exp2 softmax, Q packed once as bf16 in global memory
    (SLM 44 → 12 KB per work-group), longest query tiles first
  - opt-in only: fused dequant-in-K-loop GEMM (`GRIMOIRE_FAST_GEMM_FUSED=1`, slower)
- `src/prefill.cpp`: DeltaNet chunked fallback kernel; conv1d sliding window
  (78 → 32 ms, `GRIMOIRE_CONV_OLD=1`)
- `src/grimoire.cpp`: fused SwiGLU wiring; `dense_pure` (skip fp32 normed copy,
  no r1 zeroing, norms/gates write the next GEMM's bf16 input;
  `GRIMOIRE_NO_DENSE_PURE=1`)
- `src/ops.cpp`: `launch_rmsnorm_gate_silu_bf16_out`, `launch_gate_sigmoid_mul_bf16_out`

## Measured dead ends — do not retry

| idea | result |
|---|---|
| dequant fused into the GEMM K loop via SLM | 1.65× slower (4.49 s vs 3.14 s) |
| DeltaNet one row per lane (select_from_group / uniform loads) | 283 / 623 ms (vs 245) |
| DeltaNet q4 with 32 tokens per chunk | spills, 541 ms (vs 218) |
| flash O-rescale via counter-indexed plain apply | 2× slower (323 vs 158 ms) |
| flash lazy rescale (skip when max moves < 2^8) | slower (239 ms), spills |
| checked + unchecked GEMM loop copies | 13% slower than one checked loop |
| SwiGLU with gate/up columns 70 KB apart | 23% slower (fixed by interleave) |
| dequant order k-chunks-fastest / 2 or 4 chunks per work-item | 293 / 192 / 242 ms (vs 159) |
| GEMM GROUP_M 2 / 8 | 1,448 / 1,476 ms (4: 1,384) |
| GEMM prefetch distance 0 / 2 / 3 | 1,579 / 1,536 / 1,641 ms (1: 1,377) |
| `GRIMOIRE_FAST_GEMM_TIMING` before 09-25 | counted earlier kernels as dequant (fixed) |

## Next steps (in order)

1. **Flash prefill (147 ms → target ≤ 70).** It is ~1/7 of its XMX bound. The
   D=256 kernel sits at the 256-register limit (O = 128 GRF), so any extra
   state spills. Candidates: split D across two sub-groups (each owns 128
   columns of O, S shared through SLM), a bigger Q tile per sub-group, K/V
   blocks through SLM double-buffered. Iterate with a standalone bench that
   calls `launch_flash_prefill_fast` at 4088 tokens / H=24 / KVH=4 / D=256
   (tools/bench_flash_prefill.cpp has its OWN older kernel copy — do not
   trust it for this kernel).
2. **DeltaNet recurrence (218 ms).** Either the chunked WY form on the matrix
   unit (vLLM's took 112 ms) or cut the q4 kernel's SLM broadcast traffic.
3. Small-N GEMMs: fuse attention q|k|v (N=14336) and DeltaNet qkv|z|ab
   (N=16480) into one GEMM each (~20–30 ms).
4. qk-norm + rope fusions (~10 ms); dequant (160 ms) is at ~430 GB/s.
5. After the gate: llama-benchy through grimoire-server (pp4096/tg32) for the
   apples-to-apples number; decode speed; the remaining pure-mode gates
   (a full `build_b70.sh` compiled engine + 6 gates before it was stopped);
   two-card PP/TP (Ornith).

## Rules that still apply

- Pure only: no vLLM plug-ins, no `GRIMOIRE_XE2_*` / `GRIMOIRE_ONEDNN_*`,
  no `GRIMOIRE_W4A8` (frees the MXFP4 payload).
- Never the B580 (12 GB): use `tools/tune.sh` / `tools/gpunode.sh` (gpu0/gpu1).
- Never kill a container with GPU work in flight (drops the B70 off the bus).
- Every change: speed AND the correctness cmp, twice.

---

## UPDATE 2026-09-26 — resume here after Ian's reboot

**Speed:** 4088-token prefill **2.11 s (~1,935 tok/s)** on gpu0; vLLM ≈ 2.03 s.
Gap ~80 ms. Text identical to `ref-sherlock-5987-n24.txt`. Commit `10981c4`.
NOTE: `bin/` on the Tower was built from a discarded experiment (lazy rescale,
slower) — **rebuild first** (`tools/build_grimoire_only_b70.sh`).

**Hardware (Ian moved cards on 09-26):** B580 removed. gpu0 = B70 03:00.0
(renderD128, Gen4 x8). gpu1 = B70 07:00.0 (renderD130, **Gen4 x4**).
`tools/gpunode.sh` updated (checks device id 0xe223).
- **gpu1 dropped off the PCIe bus at 10:53 during its first run after the move**
  (my first sanity run on it): root port 00:06.0 logged a PCIe physical-layer
  RxErr at 16 GT/s, then the card read 0xFFFFFFFF and xe declared it wedged.
  Log: `/mnt/storage/isos/tower-logs/syslog-2026-09-26-gpu1-07.00.0-wedged.txt`.
- **Do not put load on 07:00.0 again until Ian says the slot is fixed**
  (BIOS: that slot at Gen3, or reseat riser/cable, check power). When it is:
  start with a tiny smoke test and watch `dmesg | grep -E "AER|RxErr|07:00.0"`.
- After a reboot the Unraid array is NOT auto-started (`startArray="no"`):
  `/mnt/storage` (repo + models) is missing until it is started.
- Temporary rsyslog drop rule for the wedge's reset flood lived in RAM
  (/etc/rsyslog.conf) — gone after the reboot.

**Done today:** hardware bf16 conversion (`bf16_rne`, the SYCL bfloat16
constructor links a branchy software fallback on this target), branch-free
flash softmax (147 → 137 ms), launcher guard per render node, old plug-in
launchers retired.

**Measured dead ends today (see commit messages):** flash lazy rescale (181 ms),
constant-indexed get_wi_data rescale (spills 5.7 KB), row-scale tile (same code),
fused dequant with col_major SLM B (64 SLM gathers/K step, spills), second-queue
dequant overlap (~20% only).

**Next (after rebuild + re-verify on gpu0):** DeltaNet recurrence (218 ms),
residual add fused into out/down GEMM epilogues (~20 ms), attention q|k|v and
DN qkv|z GEMM fusions (~20 ms), flash (137 ms) restructure.
