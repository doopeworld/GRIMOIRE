# GRIMOIRE — HANDOFF 2026-08-25 (post power-outage)

**Live repo:** `/mnt/storage/isos/grimoire-fuse` (== `/mnt/user/isos/grimoire-fuse`).
Supersedes `GRIMOIRE-HANDOFF-2026-08-24.md`, whose optimization map is MISLEADING —
see "Superseded" at the bottom.

**UPDATE THIS FILE EVERY TIME SOMETHING IS LEARNED. Not at end of session.**

---

## 0. Power outage 2026-08-25 ~07:27 — what survived

Everything. Verified on disk after reboot:

| Item | State |
|---|---|
| `src/grimoire.cpp` (GDN padding fix) | intact, 07:17 |
| `src/kernels.hpp` (stride arg) | intact, 07:17 |
| `src/prefill.cpp` | intact, 07:17 |
| `bin/grimoire` | built 07:21, newer than all sources |
| Model artifacts (re-converted) | intact |

Nothing needs redoing. The outage also cleared the stuck `objective_sutherland`
container that the 08-24 handoff said required a power cycle.

**GPU1 is DEAD.** `0000:35:00.0` (`renderD130`): `power/runtime_status = error`,
67 dmesg errors (DMC mmio reads `0xffffffff`, DC state write failures,
`can't suspend ... -110`, Runtime PM underflow). Level Zero enumerates only ONE
B70. **Use `renderD129` (`0000:03:00.0`) — healthy, 0 errors.**
Ian's call 2026-08-25: ignore GPU1, do not spend time on it.

---

## 1. Two root causes found 2026-08-25 — both FIXED and VERIFIED

### 1a. Converter quantized four precision-critical BF16 tensors

`tools/b70_compile_model.cpp` decided what to quantize **purely structurally** —
"any 2-D BF16 `.weight` that isn't embeddings". Norms, `A_log`, `dt_bias` survived
only by being 1-D; `conv1d` only by being 3-D. Four 2-D tensors were quantized by
rank alone:

| tensor | Qwen | Ornith | feeds |
|---|---|---|---|
| `linear_attn.in_proj_a` | 48 | 30 | `A_log` -> **exponentiated** into decay `alpha` |
| `linear_attn.in_proj_b` | 48 | 30 | `beta`; forms the matrix `chunk_inverse_opt_kernel` inverts |
| `mlp.gate` (MoE router) | — | 41 | picks 8 of 256 experts/token |
| `mlp.shared_expert_gate` | — | 41 | a 1xH single row into a sigmoid |

`src/grimoire.cpp` **already asked for `Fmt::BF16`** on all four ("small tensors
stay bf16"); `quantize_upload_t`/`concat_upload_t` computed `Fmt use = fmt` and the
native-MXFP4 fast path then ignored it. The file silently overrode the runtime.

Cost of keeping them BF16: ~47 MB (Qwen) / ~51 MB (Ornith) against 15-18 GiB.
The damage was taken for nothing.

Confirmed against reference quantizers: `Qwen3.8-27B-int4-AutoRound` uses
`{"bits":16,"data_type":"fp"}` and `Qwen3.8-27B-W4A16`'s
`modules_in_block_to_quantize` omits them. (Ian found this by asking whether the HF
pre-quantized checkpoints did it differently. They did.)

**Fix:** `keeps_bf16()` exclusion list in the converter; native-RAW branch in
`read_matrix_f32` (RAW 2-D tensors have no safetensors shard and cannot otherwise
load); loud warning in both upload paths. Both models re-converted; old artifacts
were wrong and were replaced.

**CONFIRMED — re-converting Qwen fixed crash AND slowdown at once, raw GDN ENABLED:**

| | before | after |
|---|---|---|
| PP @ M=4096 | 227.9 tok/s, or DEVICE_LOST | **1562.9 tok/s** (2620.8 ms) |
| PP @ M=1024 | DEVICE_LOST | 1453.5 tok/s |
| M>=64 stability | died every run | passes |
| decode | 23.3 tok/s | 23.1 tok/s (unchanged) |

6.9x on PP. The chunk-GDN kernel was never broken — it was fed 4-bit `beta` into a
matrix inverse.

### 1b. Chunked GDN missing vLLM's M+63 chunk padding

The chunked GDN kernel reads a whole 64-token tile from the final chunk. Max tile
base is `floor((M-1)/64)*64`, so it touches up to row **M+62**. vLLM allocates every
GDN input with `padding_size = batch_size*(chunk_size-1)` = 63, **zero-fills** them
(`torch::zeros`, not `empty`), and passes the PADDED extent as the token count:

```cpp
// vLLM csrc/xpu/gdn_attn/gdn_attn_interface.cpp
int padding_size = batch_size * (gdn::chunk_size_xe2 - 1);
torch::Tensor q = torch::zeros({non_spec_token + padding_size, num_k_heads, head_k_dim});
const int total_virtual_seqlen = q.size(0);   // PADDED, not the real length
```

Grimoire allocated only `M` rows and passed `M`. The final tile ran off the end — and
for the **head-major** `alpha`/`beta` (`gate_a[hd*stride+t]`, stride = tokens) it ran
into the *next head's* data, so corruption was deterministic but activation-dependent.

**Why it hid so long:** `grimoire_bench_prefill` feeds synthetic tokens `1000 + i%97`,
so every PP benchmark in this project's history over-read into tame repetitive data
and survived. Real text hung — reproducibly, on old AND new artifacts, so it was
never a regression.

**Fix, in `prefill()` (`src/grimoire.cpp:2352`):**
```cpp
const size_t gdn_tokens=size_t(M)+63;
```
used as the extent for `alpha`/`beta`/`fq`/`fk`/`fv` and as `gdn_pitch`; all zeroed;
padded stride passed to `launch_deltanet_native_gates` (new `stride` arg — the
producer wrote with stride `M` while the consumer read the padded stride);
`int(gdn_tokens)` passed to the raw bridge as `total_virtual_seqlen`. Sizing guards
added to `bridge_elems` and `grouped_out`.

**Verified:** real 133-token prompt produces output **character-identical** to the
fallback recurrence (correctness, not merely "stopped hanging"), 498.1 vs 337.6
tok/s. M=4096 PP unchanged at 1564.4 tok/s — correctness cost nothing.

**Lesson: benchmark with real tokenized text, never synthetic ids.**

---

## 2. GRIMOIRE_PROFILE_PREFILL is a trap — do not use at M>=64

It **does not complete at M>=64**. At M=4096 the M=32 warm-up profile prints, then
the measured pass sits at ~190% host CPU for 25+ min: two threads at ~100%, state
`R`, no `wchan` — **host spinning, GPU idle**. Not a hang, not OOM.

Mechanism: `pp_mark()` retains one live `sycl::event` per region (~700 at 64 layers)
and forces a profiling-enabled queue; the L0 adapter must emit host-visible
timestamped events, stops batching into reusable command lists, degenerates to
busy-wait.

**This is almost certainly the root of the 08-24 handoff's "incidents"** (M=256 at
"187% host CPU for minutes"; runs ending in DEVICE_LOST). Chain: profiled run goes
pathological -> someone kills it -> SIGKILL mid-submission -> device off the bus.

**Use instead:** `GRIMOIRE_TIME_LAYER=all` or `GRIMOIRE_TIME_LAYER=<n>` (added
2026-08-25). Drains the queue, reads host clock, retains no events. The binary now
warns loudly if `GRIMOIRE_PROFILE_PREFILL` is set at M>=64.

---

## 3. How to run — READ BEFORE LAUNCHING

**Always use `tools/b70run.sh`.** Never `timeout N docker run ...` — that signals the
docker *client*, not the container; the container keeps running, `--rm` never fires,
and the orphan holds the GPU until a power cycle. Never `docker run --entrypoint bash
-lc CMD` — bash becomes PID 1, does not forward SIGTERM, and `docker stop` falls
through to SIGKILL, tearing down the DRM fd mid-submission.

```bash
cd /mnt/storage/isos/grimoire-fuse
export GRIM_ENV="GRIMOIRE_XE2_GROUPED_BRIDGE=/grimoire/src/libgrimoire_xe2_grouped.so
GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_bridge.so
GRIMOIRE_XE2_GDN_RAW_BRIDGE=/grimoire/src/libgrimoire_xe2_gdn_raw.so
GRIMOIRE_ONEDNN_BRIDGE=/grimoire/src/libgrimoire_onednn.so"
bash tools/b70run.sh renderD129 600 <name> /grimoire/bin/grimoire -m /models/<model> ...
```

### OPEN ISSUE — the attention bridge needs the vLLM image

`libgrimoire_xe2_attention_bridge.so` links against `libtorch.so`, `libc10.so`,
`libc10_xpu.so`, `libtorch_xpu.so`, `libtorch_cpu.so`, `libattn_kernels_xe_2.so`,
`libgdn_attn_kernels_xe_2.so`. **None are in `grimoire:queuefix`**, so running there
silently falls back:
```
Xe2 chunk prefill unavailable; using fallback
Xe2 chunk GDN unavailable; using fallback
```
and PP collapses to ~108 tok/s. `libgrimoire_xe2_gdn_raw.so` and
`libgrimoire_xe2_grouped.so` resolve fine in that image; only the torch-linked
attention bridge does not.

`libgrimoire_xe2_bridge.so` (the w4a16 grouped path, `GRIMOIRE_XE2_BRIDGE`) is
separately broken: needs `libgrouped_gemm_xe_2.so`, which exists **nowhere** on host
or in any image. Harmless for MXFP4 artifacts, which use
`GRIMOIRE_XE2_GROUPED_BRIDGE`.

**Next action:** identify the image whose library path carries torch +
`libattn_kernels_xe_2.so` + `libgdn_attn_kernels_xe_2.so` (`b70run.sh` defaults to
`my-vllm-xpu:latest`, whose entrypoint auto-launches vLLM and dies on device
inference — bypass the entrypoint, don't run `bash -c` under it). Re-baseline there
before trusting ANY perf number.

---

## 4. Where the project stands

**Ornith: PP gate MET.** Four consecutive 4096-token E2E runs 400.7-401.8 ms =
**10,194-10,222 tok/s** vs a <=409.6 ms requirement; spread 1.1 ms. Config:
`GRIMOIRE_DEFER_MOE_GATHER=1` + `GRIMOIRE_BF16_QKV=1` + `GRIMOIRE_BF16_DN_QKV=1` +
the three Xe2 bridges. **Re-confirm once after this reboot.**

**Ornith long-prompt coherence** was the last blocker and is addressed by 1a + 1b.
Re-verify with real tokenized text.

**Qwen is the remaining work. Targets: 2,000 PP and 30 TG.**

| | now | target |
|---|---|---|
| PP @ M=4096 | 1562.9 tok/s (2620.8 ms) | 2000 tok/s (2048 ms) — **~573 ms to find** |
| TG | 23.1 tok/s | 30 tok/s |

### Qwen PP — the live lead

GEMMs were ~9% of the old 18 s baseline and are **~63% of the 2.62 s one**, so dense
tile selection finally matters. Qwen's dominant kernel is the dense FFN (~70% of
prefill FLOPs) via `grimoire_xe2_dense_mxfp4_bf16`, **hardcoded to `p128x128`** at
`src/xe2_grouped_bridge.cpp:92`. The dense autotune harness exists but its shape table
is 100% Ornith shapes (all K=2048); Qwen's FFN shapes **34816x5120** and **5120x17408**
were never tuned.

Ready to run: `tools/autotune_b70_dense_qwen.cpp` -> `/grimoire/bin/autotune_qwen`
(9 policies x Qwen's real shapes). Then make the dense entry point dispatch by shape.

### Qwen TG — the live lead

440 GB/s effective is needed and lm_head already sustains it, so it is a kernel
problem, not hardware (measured roofline 602 GB/s). Decode is **54% FFN**
(23.4 of 42.7 ms/token). Efficiency tracks output-row count because a GEMV launches
`ceil(N/32)` workgroups: lm_head (N=248320) 440 GB/s, FFN 386-393, la_qkv 322, but
`k+v` (N=2048) only **108** and `ab` (N=96) only **9**.
**Best untried lever: split-K for low-N shapes** (`k+v`, `ab`, the two RMSNorms) —
worth ~3 ms/token. The dense FFN gate|up is already fused into one N=34816 GEMV; no
fusion win left there.

**Fixed earlier:** `g_tune_unroll` defaulted to 1, overriding every format's
`GemvGeom<F>::UNROLL_DEFAULT` (MXFP4 wants 2). Cost 2.5% of decode. Now defaults to 0.
Do NOT pin `B70_UNROLL=1`.

**After Qwen PP/TG — speculative decoding.** `Qwen3.8-27B-DSpark` and
`qwen38-dflash-drafter-fp8-b70` are bf16 `Qwen3DSparkModel` drafters purpose-built for
Qwen3.8-27B (5 layers, block_size 7, tapping target layers [4,16,28,40,52]). Being bf16
they satisfy the quantize-from-BF16-ourselves rule. MTP heads exist only inside a GPTQ
checkpoint, which conflicts with that rule. DSpark is the strongest lead for the
2.5-3x TG vLLM shows.

---

## 5. Do not retry without new evidence

- **Do NOT set `GRIMOIRE_RAW_GDN_LAYER_LIMIT=0`.** Raw GDN was blamed for DEVICE_LOST
  and was innocent (cause was 4-bit gates, §1a). Disabling it kills a working fast path.
- **Do NOT chase "the dense tile was never tuned" as the cause of the crash** — it was
  a real but separate perf item, now the live PP lead (§4).
- oneDNN MXFP4 W4A16: numerically exact but 188 GB/s vs the native GEMV's 322 GB/s on
  the identical 10240x5120 shape. ~1.8x slower.
- Register-resident E2M1 decode replacing the SLM nibble LUT: 15.8 tok/s (packed
  64-bit) / 9.0 tok/s (32-bit arithmetic) vs 23.3 with the SLM table. Xe2 emulates
  64-bit variable shifts and the `e==0` case forces a per-element select. SLM table
  is correct.
- vLLM's oneDNN W4A16 route: `od_w4` is gated on `fmt == Fmt::INT4`
  (`grimoire.cpp:568`); our artifacts are MXFP4. Dead end.

**Provenance:** `b70_compile_model.cpp` rejects any non-BF16 source; both model sources
are unquantized BF16. No HF quant is in the runtime path.

---

## Superseded from GRIMOIRE-HANDOFF-2026-08-24.md

- Its **M=32 optimization map is misleading.** At M=32 the 128-row dense tile is only
  25% filled AND `native_rec` is gated on `M>=64`, so its "DN recurrence 42 ms / 29%"
  is the *fallback* recurrence, not the native chunk-GDN kernel that runs at M=4096.
  Do not plan Qwen optimization from it.
- Its "1,507.7 PP" for Qwen did not reproduce (real baseline was 227.9 before §1a).
- Its "next session" plan (install Level Zero ICD on the host, abandon Docker,
  persistent scratch arena as the OOM fix) was written under the belief that the
  failures were OOM/host-enumeration problems. They were §1a and §2. Docker is fine.

---

## 6. Post-outage re-baseline log (append as found)

**07:06 confirmed on disk — the re-converted artifacts survived, complete:**
`Qwen3.8-27B-MXFP4-GRIMOIRE/model-v2.b70` 16,421,281,344 B @ 06:49 and
`Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE/model-v2.b70` 19,649,167,040 B @ 06:53 —
both written BEFORE the ~07:2x outage. Nothing truncated.
**NOTE:** Ornith still has the OLD `model.b70` (19,616,595,648 B @ 08-24 03:12)
sitting beside `model-v2.b70`. Confirm which one the loader picks before trusting
any Ornith number, and delete the old one.

**Bridge loading SOLVED.** All bridges + the binary resolve in `my-vllm-xpu:latest`
with:
```
LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:/opt/intel/oneapi/lib:/usr/local/lib
```
`libattn_kernels_xe_2.so` + `libgdn_attn_kernels_xe_2.so` live in
`site-packages/vllm_xpu_kernels/`; torch libs in `site-packages/torch/lib/`.
Present in `:latest`, `:fast`, AND `:v27`. Canonical runner installed as
`tools/qrun.sh`; no-tuning-flags variant as `tools/bare.sh`.

**!! Ornith tuning flags CORRUPT Qwen !!** (found 2026-08-25 post-outage)
Running Qwen3.8-27B with `GRIMOIRE_DEFER_MOE_GATHER=1` + `GRIMOIRE_BF16_QKV=1` +
`GRIMOIRE_BF16_DN_QKV=1` — the Ornith gate config from §4 — produces **degenerate
repetition** ("import matplotlib.animation as np" looping). The same build with
**no tuning flags** produces clean grammatical prose. Qwen3.8-27B is **dense**;
`DEFER_MOE_GATHER` has no MoE to defer.

| Qwen run, 113-token real prompt | output | pp | tg |
|---|---|---|---|
| Ornith flags + raw GDN | degenerate repetition | 59.7 | 22.1 |
| Ornith flags + `RAW_GDN_LAYER_LIMIT=0` | degenerate repetition | 104.8 | 22.1 |
| **no tuning flags** | **coherent prose** | 116.2 | 22.1 |

Both GDN paths degenerate identically under the Ornith flags, so GDN is NOT the
of the three is responsible before re-enabling any of them on Qwen.

**Still open:** the no-flags Qwen output is coherent but ignores the prompt and
answers in Chinese, reading like an instruct model mid-reasoning. Suspect the raw
`-p` path applies no chat template. Verify against `chat_template.jinja` in the
artifact dir before treating this as a numerics bug.

**Still open:** PP is far below the 1562.9 tok/s baseline of section 1a in every run
above, but every one of them logs `Xe2 grouped GEMM unavailable
(/bridge/libgrimoire_xe2_bridge.so)` — the w4a16 bridge whose dependency
`libgrouped_gemm_xe_2.so` exists nowhere (section 3). Resolve that, or confirm MXFP4
does not need it, before reading anything into these PP numbers. Also re-run at
M=4096: 113 tokens is far too short to compare against an M=4096 figure.

**Process note:** the runs above used `tools/qrun.sh` / `tools/bare.sh`, which wrap
`tools/b70run.sh`. Do not hand-roll `docker run` — see section 3.

---

## 7. RECOVERED from the session transcript (cut 2026-08-25 05:25:13 UTC by the outage)

Recovered from `~/.claude/projects/-Users-ianernst/3641894e-….jsonl`, not re-measured.
Re-measure before relying on any of it.

### 7a. Qwen M=4096 region budget — the last thing measured before the outage

```
input norm                      44.229 ms    1.7%
DN qkv projection              212.910 ms    8.2%
DN causal conv + split          72.145 ms    2.8%
DN gate projection              59.134 ms    2.3%
DN qk norm                      13.256 ms    0.5%
DN recurrence                  112.365 ms    4.3%
DN z projection                127.571 ms    4.9%
DN norm + output projection    153.797 ms    5.9%
dense FFN                     1611.155 ms   62.2%   <-- THE LEVER
full attention                 182.851 ms    7.1%
final norm + logits              2.737 ms    0.1%
TOTAL (timed regions)         2592.151 ms
FULL E2E PP: PASS, 4096 tokens in 2632.3 ms -> 1556.0 tok/s
```

**Gate is 2000 tok/s = 2048 ms, so ~584 ms must come out.** Dense FFN alone is
1611 ms — the only region big enough to matter. A 36% cut there hits the gate by
itself. This is the measured confirmation of the autotune plan in section 4: the
dense entry point is hardcoded `p128x128` and Qwen's real FFN shapes were never
tuned. The DN group sums to ~751 ms (29%) across seven regions — real, but nothing
there is worth attacking before the FFN.

### 7b. Fallback vs raw GDN at M=4096 — why raw GDN had to be fixed, not disabled

| recurrence | M=4096 PP | works on real prompts? |
|---|---|---|
| raw GDN (before padding fix) | 1562.9 tok/s | **no — hung** |
| fallback recurrence | **544.0 tok/s** | yes, coherent |
| raw GDN (after padding fix) | **1564.4 tok/s** | **yes, coherent** |

The fallback is 2.9x slower and would need 3.7x to reach 2000 — unreachable. Fixing
raw GDN was the only path that served correctness AND the PP gate. It cost nothing:
1564.4 vs 1562.9 is statistically identical.

### 7c. State against the gates at the moment of the outage

| | GPU 0 | gate | gap |
|---|---|---|---|
| Qwen PP @4096 | 1564.4 tok/s | 2000 | 1.28x |
| Qwen TG | 23.1 tok/s | 30 | 1.30x |
| Real prompts coherent | YES | — | done |
| Ornith PP | 9702.6 tok/s (GPU 0) | 10000 | met on GPU 1 |

**Ornith's 10k gate was met on GPU 1, which is now dead (section 0).** The 9702.6
figure is GPU 0. Decide whether the gate is re-baselined to GPU 0 or blocked.

### 7d. OPEN — llama-benchy, agreed and never built

`https://github.com/eugr/llama-benchy` — Ian: "the universal tool that all dev use to
test bench speed". It drives **OpenAI-compatible HTTP endpoints only**
(`/v1/chat/completions`) and cannot invoke a local binary, so it needs a minimal HTTP
server in GRIMOIRE exposing `/v1/chat/completions` and `/v1/models`.

Beyond tooling, it fixes the methodology hole that hid the GDN bug for the project's
entire history: it sends **real prompts of arbitrary length** instead of
`1000 + i%97`. Its default is `--pp 2048` of real text — the exact case that used to
hang.

```bash
uvx llama-benchy --base-url http://192.168.8.225:8000/v1 --model grimoire-qwen \
  --pp 128 256 2048 --tg 32 64 --depth 0 1024 --runs 3 --latency-mode generation \
  --format md --save-result bench.md
```
Reports throughput, TTFR, est_ppt, e2e_ttft, peak throughput per 1 s window.

### 7e. Full task list as Ian stated it

1. Coherent output on Qwen AND Ornith at long prompts — DONE (sections 1a, 1b)
2. Re-convert both models from BF16 with the corrected quantizer — DONE (artifacts
   verified in section 6)
3. Delete the old wrongly-quantized artifacts — DONE for Qwen; **Ornith's old
   `model.b70` is still present** (section 6)
4. Apply all fixes to grimoire, keeping Ornith's speed — DONE
5. **Raise Qwen TG and PP** — the remaining work (2000 PP / 30 TG)
6. Then evaluate **MTP / DSpark / DFlash** on both models for TG (section 4)

---

## 8. Cold reboot + Qwen dense autotune — 2026-08-25 ~11:00 CEST (12:00 IDT)

**Clock note:** Tower runs CEST (UTC+2), Ian's Mac runs IDT (UTC+3). Every Tower
timestamp is one hour behind his wall clock. Not a fault; just correlate carefully.

### 8a. Cold power cycle fixed GPU1

Both B70s now `runtime_status=active` with **0 dmesg errors each**, and Level Zero
enumerates **both** plus the iGPU. The warm reboot after the outage did NOT clear
`35:00.0`; only pulling mains power did.

### 8b. !! RENDER NODE NUMBERS ARE NOT STABLE ACROSS REBOOTS !!

They flipped on this boot:

| node | before cold boot | after cold boot |
|---|---|---|
| renderD128 | iGPU (00:02.0) | **B70 #0 (03:00.0)** |
| renderD129 | **B70 #0 (03:00.0)** | iGPU (00:02.0) |
| renderD130 | B70 #1 (35:00.0) | B70 #1 (35:00.0) |

The first autotune run therefore landed on the **iGPU** and died with
`IGC: Internal Compiler Error` (the AOT binary is Xe2/BMG; the iGPU forced a
recompile-from-IR that crashed). Exit 134.

**Fix installed: `tools/gpunode.sh`** resolves `gpu0|gpu1|igpu` to the current node
via `/dev/dri/by-path/pci-<addr>-render` and refuses if `runtime_status` is unhealthy.
`tools/tune.sh` now takes `GPU=gpu0` instead of a node name.
**Never hardcode a renderD number again.**

### 8c. !! /mnt/user and /mnt/storage are DIFFERENT filesystems !!

`/mnt/user` is the shfs union; `/mnt/storage` is the pool holding the repo. New files
written through `/mnt/user` landed on the **cache** pool and were invisible at
`/mnt/storage/...`, which is what `b70run.sh` mounts as `/grimoire`. The repo's
existing files (source, binary) are identical through either path, but anything newly
written is not.
**Write to `/mnt/storage/isos/grimoire-fuse` directly.** Scripts and this handoff have
been consolidated there.

### 8d. Qwen dense autotune RESULT — 128x256 wins, but tile tuning is NOT enough

`bin/autotune_qwen`, M=4096, 9 policies x 8 real Qwen shapes = 72 candidates,
frequency-weighted, **all 72 numerically exact (max_abs = 0, zero rejects)**.

| policy | all 8 shapes | FFN only | vs 128x128 |
|---|---|---|---|
| **128x256** | **1571.8 ms** | **1105.4 ms** | **-77.3 ms** |
| 128x128 (production today) | 1649.1 ms | 1168.0 ms | 0 |
| 64x256 | 1791.0 ms | 1265.9 ms | +141.8 |
| 128x64 | 1829.5 ms | 1312.6 ms | +180.4 |
| 64x128 | 1963.5 ms | 1400.3 ms | +314.4 |
| 64x64 | 2491.3 ms | 1807.7 ms | +842.2 |
| 32x256 / 32x128 / 32x64 | 3519.6 / 4133.1 / 4935.9 ms | | +1870 / +2484 / +3287 |

**128x256 is best on every single shape**, so no per-shape dispatch is needed —
per-shape-optimal and global-128x256 both total 1571.8 ms. One-token change:
`src/xe2_grouped_bridge.cpp:107`, `launch_dense_mxfp4<p128x128>` ->
`<p128x256>`. `p128x256` already exists at line 32.

**HONEST GAP — the section 4 / section 7a plan was too optimistic.**
The gate needs **584 ms**; tile tuning yields **77.3 ms**, about 13% of it.
Projected: 2632.3 - 77.3 = ~2555 ms = **~1603 tok/s**, not 2000.
Do NOT expect the autotune to close the gate.

### 8e. NEW LEAD — ~443 ms of non-GEMM work inside the dense FFN region

At the same M=4096, on the same GPU:

| | ms |
|---|---|
| dense FFN **region** (section 7a budget) | 1611.2 |
| pure FFN **GEMM** at 128x128 (this autotune) | 1168.0 |
| **unexplained, non-GEMM** | **~443** |

That residual is **5.7x larger than the entire tile-tuning win** and is the biggest
single identified target for the 584 ms. Likely candidates: the SwiGLU epilogue,
activation materialisation between gate|up and down, or a separate
dequant/layout pass. **Profile inside the FFN region with `GRIMOIRE_TIME_LAYER`
before optimising anything else.**

### 8f. Next actions, in order

1. Apply the `p128x256` one-token change, rebuild bridges
   (`tools/build_bridges_b70.sh`), re-measure E2E. Expect ~1600 tok/s. Free and
   numerically exact.
2. Break down the ~443 ms non-GEMM FFN residual (8e). This is where the gate lives.
3. Re-confirm Ornith's 10k gate on GPU1 (`GPU=gpu1`), now that it is healthy again.
4. TG work: split-K for low-N shapes (section 4).
5. Build the OpenAI-compatible HTTP server so llama-benchy can drive it (section 7d).
