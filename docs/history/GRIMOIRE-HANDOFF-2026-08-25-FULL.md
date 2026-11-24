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

---

## 9. PP work session — 1556.0 -> 1623.2 tok/s (2026-08-25 ~12:30 IDT)

### 9a. Changes applied (all numerically exact, all reversible)

| # | change | file | gain |
|---|---|---|---|
| 1 | dense bf16 policy `p128x128` -> `p128x256` | `src/xe2_grouped_bridge.cpp` (dense_mxfp4_bf16) | ~27 tok/s |
| 2 | dense f32 policy `p128x128_f32` -> `p128x256_f32` (**new production class**, the macro one is inside `GRIMOIRE_ENABLE_AUTOTUNE`) | `src/xe2_grouped_bridge.cpp` | ~40 tok/s |
| 3 | SwiGLU 1-D range -> 2-D range (kills a per-element 64-bit div AND mod) | `src/prefill.cpp` `launch_swiglu_bf16` | ~30 ms |
| 4 | FFN sub-region timing marks | `src/grimoire.cpp` `mlp_bf16` | diagnostic |

Backups: `/tmp/xe2_grouped_bridge.cpp.bak`, `/tmp/prefill.cpp.bak`, `/tmp/b70run.sh.bak`.

**E2E: 2632.3 ms / 1556.0 tok/s -> 2523.4 ms / 1623.2 tok/s (+4.3%).**

### 9b. The f32 autotune — the down projection had NEVER been tuned

`mlp_bf16` uses TWO different dense entry points: `grimoire_xe2_dense_mxfp4` (bf16 out,
gate_up) and `grimoire_xe2_dense_mxfp4_f32` (f32 out, down). Only the bf16 one was ever
touched. `bin/autotune_qwen_f32`, 72 candidates, 0 rejects:

| policy | all shapes | ffn-down | vs 128x128 |
|---|---|---|---|
| **128x256** | **1590.1 ms** | **366.8 ms** | **-37.6 ms** |
| 128x128 (was production) | 1627.7 ms | 389.3 ms | 0 |

128x256 wins every shape on BOTH the bf16 and f32 harnesses.

### 9c. MEASURED FFN breakdown — kills the "non-GEMM overhead" theory

Sub-marks inside `mlp_bf16`, M=4096, all 64 layers:

| FFN sub-region | ms | % of E2E |
|---|---|---|
| **gate_up GEMM** | **960.0** | 38.6 |
| swiglu | 75.2 | 3.0 |
| **down GEMM** | **457.5** | 18.4 |
| (region bookkeeping) | 0.0 | 0.0 |

**The section 8e hypothesis (~443 ms of non-GEMM work in the FFN) was WRONG.** It came
from comparing the autotune's *cold* `cold_ms x freq` against *warm* production regions.
SwiGLU is only 75 ms. The FFN is 1417.6 ms of genuine GEMM. Do not chase FFN epilogue
overhead.

### 9d. THE REAL GAP — production GEMM is ~30% slower than the tuned microbenchmark

Same shape, same policy, same M=4096, same GPU:

| shape | autotune `cold_ms` x 64 | production measured | gap |
|---|---|---|---|
| ffn-gate-up (34816x5120) | 740.3 ms | **960.0 ms** | **+219.7 ms** |
| ffn-down (5120x17408) | 366.8 ms | **457.5 ms** | **+90.7 ms** |

**~310 ms unexplained, and the gate needs 475 ms.** This is now the single biggest
identified target. Production is slower than a *cold* microbenchmark of the identical
GEMM, which should not happen.

Efficiency: gate_up 9.344e13 FLOP / 0.960 s = **97.3 TFLOP/s**; down 4.673e13 / 0.4575 s
= **102.1 TFLOP/s**. Against the bench's own bf16 XMX peak (~180 TFLOP/s, implied by its
"20314.9 GFLOP/s = 11.3% of peak" line) that is ~54-57% of peak.

Hypotheses to test, cheapest first:
1. Per-call setup cost x64 layers (kernel arg marshalling, no persistent workspace).
2. Layout/stride differences between the harness's freshly-allocated buffers and the
   production activation/scale pointers.
3. Queue dependency stalls between layers that the microbenchmark never sees.
4. The harness's `cold_ms` is measured differently than assumed — verify what `run()`
   actually times before trusting the 740.3/366.8 figures.

**Ian's reference point: vLLM reaches ~2000 PP on this model.** vLLM's own grouped/dense
Xe2 kernels are in `/mnt/cache/appdata/vllm-xpu-kernels/csrc/xpu/`. Diffing the caller
side against vLLM is the method that cracked both bugs in section 1 — apply it here.

### 9e. Build recipe for the bridges (was undocumented)

```bash
docker run --rm -v /mnt/storage/isos/grimoire-fuse:/grimoire \
  -v /mnt/cache/appdata/vllm-xpu-kernels:/src \
  -v /mnt/storage/isos/sycl-cache:/cache \
  grimoire:queuefix /usr/local/bin/entry.sh bash -c \
  "ln -sf /usr/lib/x86_64-linux-gnu/libze_loader.so.1 /usr/lib/x86_64-linux-gnu/libze_loader.so && \
   cd /grimoire && bash tools/build_bridges_b70.sh /grimoire"
```
~4m25s. The `ln -sf` is REQUIRED: `grimoire:queuefix` ships only `libze_loader.so.1`
with no unversioned symlink, so `-lze_loader` fails to link without it.
`bash build_b70.sh` alone (~1m) rebuilds just the binary when no bridge changed.

`tools/b70run.sh` now passes `-w /grimoire`; without it, binaries with relative
`DT_NEEDED` entries (e.g. `bin/autotune_qwen_f32` -> `src/libdense_tune_f32.so`) exit 127.

---

## 10. THE GOLDEN APPLE: grimoire is single-GPU. vLLM's 2000 PP is almost certainly TP=2.

```cpp
sycl::queue q{sycl::gpu_selector_v, queue_props()};   // src/grimoire.cpp:728
```

**One queue, one device. There is NO tensor parallelism anywhere in the codebase** —
`grep` for tensor_parallel / tp_size / num_gpus / device_count / multi_gpu returns
nothing. The Tower has **two healthy B70s** and grimoire has always used one.

vLLM on a 2-GPU box defaults to TP=2. So the "vLLM gets 2000 PP on this model"
reference point is very likely both cards. Per GPU, grimoire at **1658 tok/s** is
already in the same class; the deficit is that half the machine is idle.

**TP=2 is worth ~1.8x on PP (1658 -> ~3000) and is the only remaining lever of that
size.** Everything in section 9 and below is single-digit percent by comparison.
Chase this before any further kernel work.

Sketch of what it needs:
- Column-shard `gate_up` (N=34816 -> 2x17408) and row-shard `down` (K=17408 -> 2x8704)
  per layer; column-shard `dn_qkv` / `fa_q` by head; the classic Megatron split.
- One all-reduce per layer after `down` and after the attention output projection.
  Between two B70s on the same host this is a P2P copy; check whether Level Zero
  exposes peer access between 03:00.0 and 35:00.0, otherwise stage through host.
- Weights halve per card (16.4 GB -> ~8.2 GB), which also frees room for longer
  contexts.

## 11. Session result and what is ruled out

**Qwen PP: 1556.0 -> 1658.1 tok/s (+6.6%), single GPU, all changes numerically exact.**

| region | before | after |
|---|---|---|
| FFN gate_up GEMM | (1611 ms whole FFN) | 934.8 ms |
| FFN down GEMM | | 446.7 ms |
| FFN swiglu | | ~75 ms |
| **FULL E2E PP** | **2632.3 ms / 1556.0 tok/s** | **2470.2 ms / 1658.1 tok/s** |

Banked changes: dense bf16 tile -> `p128x256`; dense f32 tile -> `p128x256_f32`
(the down projection had never been tuned); SwiGLU 2-D range; 11 elementwise kernels
converted off 64-bit div/mod; L2 swizzle `GRIMOIRE_GEMM_SWIZZLE_GM` default 4.

### RULED OUT with measurements -- do not re-litigate

1. **Dense tile choice is exhausted.** Production `p128x256` beats all 9 original
   policies AND four new larger tiles, measured in production with real activations:

   | policy | gate_up ms | E2E ms |
   |---|---|---|
   | **p128x256 (production)** | **934.8** | **2470.2** |
   | 8 = `MoE::w4a16_policy` | 968.1 | 2544.0 |
   | 7 = 128x128 | 1014.3 | 2605.7 |
   | 9 = 256x128 | 964.3 | 2528.8 |
   | 5 = 64x256 | 1117.5 | 2696.8 |
   | 6 = 128x64 | 1123.8 | 2726.0 |
   | 4 = 64x128 | 1213.0 | 2813.3 |
   | 10 = 256x256 (SG 8x4, vLLM's base) | 1574.5 | 3025.2 |
   | 11 = 128x256 K=64 | 1272.0 | 2779.0 |
   | 12 = 128x512 | 1799.6 | 3242.0 |

2. **We already run vLLM's exact kernel AND its exact policy.** vLLM's
   `MoE::w4a16_policy` (`csrc/xpu/grouped_gemm/xe_2/gemm_xe2_policy.hpp:81`) is
   `WGTile<128,256,32>` + `SGLayout<4,8,1>` + `XE_STORE_2D<16,8,32>` -- byte-identical
   to grimoire's `p128x256`. The GEMM core `MoE::xe_gemm_4bits` is vLLM's own header.
   **vLLM's PP advantage cannot come from a better 4-bit dense GEMM.**

3. **Elementwise kernels are not the bottleneck.** Converting 11 kernels off 64-bit
   div/mod bought +3 tok/s total. Only SwiGLU mattered (~30 ms) and it is now 75 ms.

4. **There is no large non-GEMM overhead.** Measured: gate_up 934.8 + swiglu 74.9 +
   down 446.7 accounts for the entire FFN region. ~84% of prefill is GEMM.

5. **The autotune harness's absolute numbers do not transfer.** It fills buffers with
   constants (`q.fill(x,0.125f)`, `memset(w,0x22)`); Intel's lossless memory
   compression makes those unrealistically fast, and it reports best-of-3. Its
   *rankings* are usable, its *milliseconds* are not. The "310 ms of phantom
   overhead" in section 9d was an artifact of this -- disregard it.

6. **L2 swizzle is nearly exhausted.** GM=4 is optimal (+31 tok/s); GM=2/8 are within
   noise, GM=16 is worse. The hardware L2 was already recovering most B-tile reuse.

### Efficiency, for scale

| GEMM | TFLOP/s | note |
|---|---|---|
| DN qkv (N=10240) | 134 | ~74% of the ~180 TFLOP/s bf16 XMX peak |
| DN z (N=6144) | 134 | |
| DN out | 113 | |
| FFN down (K=17408) | 102 | |
| FFN gate_up (N=34816) | 97 | the widest N, most B traffic |

Whole-model: 2.21e14 FLOP / 2.470 s = **89.5 TFLOP/s**. The 2000 tok/s gate needs
**108 TFLOP/s sustained** on one card -- i.e. every GEMM at DN-qkv quality AND zero
non-GEMM time. Not reachable single-GPU by tuning. **Hence section 10.**

### Still untouched
- TG is unchanged at ~23 tok/s (gate 30). Lever from section 4: split-K for low-N
  shapes (`k+v` at 108 GB/s, `ab` at 9 GB/s).
- Ornith's 10k gate has not been re-confirmed since the cold boot; GPU1 is healthy
  again so it can be re-run (`GPU=gpu1`).
- llama-benchy still needs the HTTP server (section 7d). **Worth doing before more
  PP work: it is Ian's chosen yardstick, and it would settle whether vLLM's 2000 is
  measured the same way as our FULL E2E PP -- and on how many GPUs.**

---

## 12. CORRECTION to section 10, and the actual explanation of the vLLM gap

**Section 10 was WRONG. Retract the TP=2 argument.** Ian confirmed vLLM reaches
~1950 PP on a **single** GPU at **pp 4096 / tg 32** — the same settings we measure.
Verified independently: the prior vLLM container on this box ran with
`ZE_AFFINITY_MASK=0` (one GPU). Single-GPU 1950 is real and is the target.

### 12a. GPU 1 is NOT faster — the old GPU0/GPU1 split was an artifact

| | GPU 0 (03:00.0) | GPU 1 (35:00.0) |
|---|---|---|
| Qwen PP @4096 | 1658.1 tok/s | 1648.0 tok/s |
| Qwen TG | 23.2 tok/s | 23.2 tok/s |

Identical within noise. The "Ornith 10k met on GPU 1, only 9702 on GPU 0" note in
[[grimoire-next-steps]] was a degraded-GPU-state artifact, not a hardware difference.
**Do not chase GPU choice.**

### 12b. THE KEY FACT — vLLM's number is on a DIFFERENT QUANTIZATION

```
docker inspect VLLM-XPU-FIRST -> Cmd:
  /models/Qwen3.8-27B-int4-AutoRound  --dtype bfloat16 ...
  ZE_AFFINITY_MASK=0
```

**vLLM's ~1950 PP is on `Qwen3.8-27B-int4-AutoRound` — int4 W4A16, not MXFP4.**
GRIMOIRE runs `Qwen3.8-27B-MXFP4-GRIMOIRE` through a cutlass MXFP4 kernel. These are
different quantizations driving different GEMM kernels. **We are not comparing the
same compute path**, and the MXFP4 kernel is where our ceiling is.

Measured efficiency of our MXFP4 path (M=4096, 64 layers, ~180 TFLOP/s bf16 XMX peak):

| GEMM | entry point | TFLOP/s |
|---|---|---|
| DN qkv (N=10240,K=5120) | `xe2_dense_mxfp4_f32` via `mm()` | 134 |
| DN z (N=6144) | same | 134 |
| DN out | same | 113 |
| FFN down (N=5120,K=17408) | `xe2_dense_mxfp4_f32` | 102 |
| FFN gate_up (N=34816,K=5120) | `xe2_dense_mxfp4` | 100 |

Whole model 2.21e14 FLOP / 2.464 s = **89.7 TFLOP/s**. 1950 tok/s needs **105
TFLOP/s** sustained — a 17% lift, plausible on one card, but not from tuning the
MXFP4 kernel (section 11 exhausted that).

### 12c. Also ruled out this session

- **M-scaling is flat.** 1024/2048/4096/8192 -> 1576.4 / 1642.7 / 1662.6 / 1635.2
  tok/s. M=4096 is already the sweet spot; prompt length does not explain the gap.
- **oneDNN W4A16 for prefill is not faster.** `GRIMOIRE_NO_F32_DENSE=1` routes `mm()`
  to the oneDNN path: DN qkv 219.8 ms (vs 204.7), DN z 132.1 (vs 123.2), DN out 143.2
  (vs 146.1), E2E 1650.1 tok/s (vs 1658.1). Slightly worse. (The earlier note that
  oneDNN loses on GEMV/decode also holds for prefill GEMM.)
- **The autotune harness shows NO N-scaling penalty** — gate_up 126.2 TFLOP/s and
  dn-qkv 127.2 TFLOP/s at 128x256. So the production 134-vs-100 spread is not the
  shape. It remains unexplained and is the one open thread on the MXFP4 path.

### 12d. NEXT ACTION — measure vLLM on this box, then diff

Do not guess further. Reproduce vLLM's 1950 here and profile it:

```bash
docker run -d --name vllm-bench --device /dev/dri \
  -e ZE_AFFINITY_MASK=0 -v /mnt/storage/Models:/models -p 3551:3551 \
  my-vllm-xpu:latest /models/Qwen3.8-27B-int4-AutoRound --dtype bfloat16 \
  --port 3551 --host 0.0.0.0 --max-model-len 32768 --max-num-batched-tokens 16384 \
  --block-size 64 --trust-remote-code
# then, per section 7d:
uvx llama-benchy --base-url http://192.168.8.225:3551/v1 \
  --model /models/Qwen3.8-27B-int4-AutoRound --pp 4096 --tg 32 --runs 3
```

Two outcomes, both decisive:
1. **vLLM hits ~1950 on int4-AutoRound.** Then the lever is the **quantization
   format**: build an int4/W4A16 GRIMOIRE artifact (the converter emits MXFP4 only
   today) and use the int4 GEMM. `d.od_w4` / `od_plans` machinery already exists in
   `mm()`; it just never has int4 weights to work with.
2. **vLLM does NOT hit 1950 on this hardware.** Then the target number came from a
   different machine or build and GRIMOIRE at 1663 is already competitive.

Either way this settles it with one measurement instead of more kernel guessing.
It also finally exercises llama-benchy (section 7d), which is Ian's chosen yardstick.

---

## 13. GROUND TRUTH — vLLM measured on this box. Ian's number is confirmed.

Measured 2026-08-25 ~12:45 IDT. `my-vllm-xpu:latest` (vLLM 0.26.1rc1.dev799+g535412a5d),
`ZE_AFFINITY_MASK=0` (SINGLE GPU), `/models/Qwen3.8-27B-int4-AutoRound`, driven by
llama-benchy 0.4.0 at pp 4096 / tg 32, 3 runs:

```
| model                              |   test |            t/s |       ttfr (ms) |   e2e_ttft (ms) |
| /models/Qwen3.8-27B-int4-AutoRound | pp4096 | 2015.64 ± 2.51 | 1737.08 ± 46.20 | 1737.08 ± 46.20 |
| /models/Qwen3.8-27B-int4-AutoRound |   tg32 |   27.95 ± 0.01 |                 |                 |
```

| | vLLM int4-AutoRound | GRIMOIRE MXFP4 | vLLM ahead |
|---|---|---|---|
| **PP 4096** | **2015.6 tok/s** | 1662.6 | **+21%** |
| **TG 32** | **27.95 tok/s** | 23.2 | **+20%** |

**Both on ONE B70. vLLM clears the 2000 PP gate; GRIMOIRE does not.** Every
"it may not be comparable" hypothesis is now dead — same GPU, same box, same
pp/tg, same prompt length.

### 13a. THE CAUSE — a different int4 kernel, not a better MXFP4 kernel

From vLLM's own startup log:
```
[inc_ark_ops.py:25] Successfully imported auto_round_kernel.
[inc_ark_ops.py:37] Successfully loaded auto_round_kernel backend library.
... quantization=inc ...
[qwen_gdn_linear_attn.py:158] Using Triton/FLA GDN prefill kernel
```

vLLM runs Intel Neural Compressor's **`auto_round_kernel`** — a dedicated **int4**
GEMM (`auto_round_kernel_xpu.cpython-312-x86_64-linux-gnu.so`, 30 MB, in
`/opt/venv/lib/python3.12/site-packages/auto_round_kernel/`). The checkpoint's
`quantization_config` is `{"autoround_version":"0.14.2","bits":4,"data_type":"int"}`.

GRIMOIRE runs a **cutlass MXFP4** kernel on an artifact produced by our own
`tools/b70_compile_model.cpp`, which does plain round-to-nearest MXFP4 from BF16 —
**no AutoRound, no compressed-tensors calibration**.

So the 21% is a *kernel + format* difference, not tuning. Section 11 already proved
the MXFP4 tile space is exhausted, and section 12c killed oneDNN-W4A16 and M-scaling.

**Independent confirmation of this morning's work:** that AutoRound config pins
`model.language_model.layers.*.linear_attn.in_proj_a` to `bits: 16` — exactly the
BF16-critical-tensor rule derived in section 1a. Intel reached the same conclusion.

### 13b. Options, in order of expected value

1. **Produce an int4 (AutoRound) GRIMOIRE artifact and add an int4 GEMM path.**
   `b70_compile_model.cpp` emits MXFP4 only. `mm()` already has an int4 branch
   (`w.od_w4` / `od_plans`, `grimoire.cpp:2515`) that never fires because no artifact
   carries int4 weights. Note oneDNN-int4 measured *slower* for prefill (section 12c),
   so this only pays if paired with a better int4 kernel than oneDNN.
2. **Call `auto_round_kernel` directly.** It is the kernel that actually wins. BUT it
   is a pybind11 extension with hidden visibility — `nm -D` exposes no callable
   `gemm/woq/int4` C symbols, so there is no clean C ABI to dlopen the way GRIMOIRE
   already dlopens vLLM's `libattn_kernels_xe_2.so`. Would need the upstream
   AutoRound/INC C++ source, or a Python bridge (unacceptable in the hot path).
3. **Quantize with AutoRound but export MXFP4.** AutoRound supports MXFP4 export;
   this would improve *accuracy* but runs through our same cutlass kernel, so expect
   **no PP/TG gain**. Do not confuse this with option 1.
4. **Accept MXFP4 and attack the remaining unexplained spread** — production gate_up
   100 TFLOP/s vs DN qkv 134 TFLOP/s with the *same* entry point, while the autotune
   harness shows both at ~127. Still unexplained (section 12c) and worth ~336 ms if
   closed, which alone would reach ~1940 tok/s.

### 13c. TG is the same story

vLLM 27.95 vs GRIMOIRE 23.2 on the same card. The gate is 30, so vLLM is close and we
are not. Section 4's split-K-for-low-N lever (`k+v` at 108 GB/s, `ab` at 9 GB/s)
remains untouched and is the cheapest TG work; but the int4-kernel difference likely
accounts for much of this gap too.

**Lifecycle note:** the vLLM container was stopped with `docker stop -t 300` (never
killed) and both GPUs verified `active` afterwards, per the GPU hang rule.

---

## 14. AutoRound MXFP4 quantization — DONE, but not yet loadable

**Produced:** `/mnt/storage/Models/Qwen3.8-27B-MXFP4-AutoRound/Qwen3.8-27B-mxfp-w4g32`
in **232 s** (RTN, `iters=0`), peak 15.9 GB RAM / 8.83 GB VRAM.
Script: `tools/ar_mxfp4.py` (auto-round 0.14.2, scheme `MXFP4`, `act_bits=16`
weight-only, group_size 32, sym, `format=llm_compressor`).

`quantization_config` confirms real compressed-tensors MXFP4:
`num_bits 4, group_size 32, type "float", scale_dtype torch.uint8 (E8M0), symmetric`.

BF16-pinned via real module names (no regex): `linear_attn.in_proj_a`,
`linear_attn.in_proj_b`, `mlp.gate`, `mlp.shared_expert_gate`, `lm_head`.

### 14a. Three bugs hit, two fixed

1. **auto-round 0.14.2 crash** — `revert_checkpoint_conversion_mapping` reverses
   transformers' `_checkpoint_conversion_mapping` and the reversed target keeps a
   dangling `\1`, so `re.subn` raises *"invalid group reference 1"* on the first
   packed tensor. Nothing to do with our config. **Fixed** by patching the function
   at source in site-packages to `return name, 0` (monkeypatching the module does
   NOT work — `shard_writer` imports the symbol by name).
2. **Wrong architecture in the export** — AutoRound wrote no `architectures` key;
   the source is `Qwen3_5ForConditionalGeneration` (multimodal, text tower under
   `language_model.*` plus a `visual.*` tower). **Worked around** by setting
   `architectures` explicitly.
3. **STILL OPEN — vLLM cannot load it.** With the text-only arch
   (`Qwen3_5ForCausalLM`) vLLM fails at weight load:
   `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'`.
   With the multimodal arch it fails with
   `no module or parameter named 'model'` because AutoRound emitted
   `model.layers.*` while that arch expects `language_model.model.layers.*` — a
   direct consequence of neutralising the conversion mapping in fix 1.

### 14b. Artifact is 66 GB — NOT clean

Expected ~14 GB for MXFP4. AutoRound wrote `model_extra_tensors.safetensors` with
**1198 "missing" tensors** — the unquantized visual tower and everything outside the
text blocks, at full BF16. Before this artifact is useful it needs either
`--language-model-only`-style pruning of the visual tower, or a re-run that targets
only `model.layers.*`.

### 14c. What this does and does not buy

AutoRound changes weight *values*, not the format or the kernel. Feeding an
AutoRound-MXFP4 artifact to GRIMOIRE would run the **same cutlass MXFP4 GEMM** at the
same speed — it buys **accuracy**, not PP/TG. The PP gap measured in section 13
(2015.6 int4/auto_round_kernel vs 1662.6 MXFP4/cutlass) is a *kernel* difference.

The experiment that would separate "MXFP4 format is slower on B70" from "our MXFP4
kernel is slower" is to run this artifact in vLLM and compare against its int4
2015.6 — blocked by 14a.3. That remains the single highest-value next measurement.

### 14d. Ornith regression check — CLEAN

My changes did not break or slow Ornith. Verified after all edits:
`FULL E2E PP: PASS, 4096 tokens in 449.2 ms -> 9118.2 tok/s`, TG 111.6 tok/s,
and the L2 swizzle is neutral for it (GM=1 9118.2 vs GM=4 9106.2).
No model artifact and no converter file was modified at any point today.

**However Ornith now measures ~9,100 tok/s against the ~9,702 (GPU 0) / 10,194
(GPU 1) in the 2026-08-24 notes.** Not caused by my changes (GM=1 reproduces the old
code path and gives the same number). Most likely the price of the section 1a
BF16-critical-tensor fix: Ornith is the MoE model, so `mlp.gate` (router) and
`mlp.shared_expert_gate` now run BF16 instead of 4-bit. The old `model.b70`
(2026-08-24 03:12) is still on disk beside `model-v2.b70`, so this is directly
testable. **Open question — verify before claiming the 10k gate still holds.**

Source backups moved off `/tmp` (RAM on Unraid, lost on reboot) to
`/mnt/storage/isos/grimoire-fuse/.backups-20260825/`.

---

## 15. Ornith regression — investigated. NOT caused by this session's code.

Ian's report: "this morning Ornith was running fine with coherent output at 10,000".
Today it measures ~9,110-9,180. Investigated directly.

### 15a. PROOF the code changes are not responsible

The runtime overrides added in section 9 make the pre-session kernel configuration
reproducible without rebuilding. Same binary, same artifact, back-to-back on GPU 0:

| Ornith config | PP |
|---|---|
| **ORIGINAL** (`DENSE_POLICY=7`, `DENSE_POLICY_F32=7`, `SWIZZLE_GM=1`) | 9100.7 tok/s |
| **THIS SESSION** (p128x256 / p128x256_f32 / GM=4) | **9116.2 tok/s** |

**Identical, and this session's config is marginally FASTER.** The dense tile changes,
the f32 policy change, the SwiGLU 2-D range and the L2 swizzle are all neutral for
Ornith — they were tuned on Qwen's shapes and Ornith's MoE path barely touches them.

Also checked and excluded:
- **GPU choice** — GPU 1 gives 9181.9, GPU 0 gives 9116.2. Both cards, same result.
- **Missing grouped-GEMM library** — `libgrouped_gemm_xe_2.so` was indeed absent from
  the container's path and has now been copied into `src/` (it lives in
  `/mnt/cache/appdata/vllm-xpu-kernels/build/lib.linux-x86_64-cpython-312/vllm_xpu_kernels/`).
  But the message it caused (`Xe2 grouped GEMM unavailable`) is the **w4a16** grouped
  path; Ornith is MXFP4 and uses `grimoire_xe2_grouped_mxfp4_bf16` from
  `libgrimoire_xe2_grouped.so`, which loads fine. Restoring it changed nothing
  (9110.3). Harmless for Ornith, still worth having for INT4 work.
- **Model artifact** — unchanged all day (`model-v2.b70`, 2026-08-25 06:53). No
  converter file and no artifact was modified this session.

### 15b. What the numbers actually are, in order

| when | Ornith PP | artifact |
|---|---|---|
| 2026-08-24 (notes) | 10,194-10,222 (GPU 1) | OLD `model.b70`, pre-BF16 fix |
| 2026-08-24 (notes) | 9,702.6 (GPU 0) | OLD `model.b70`, pre-BF16 fix |
| 2026-08-25 05:24 UTC (this morning's transcript) | 9,702.6 (GPU 0) | reported post-fix |
| 2026-08-25 13:1x (today, measured) | 9,116.2 (GPU 0) / 9,181.9 (GPU 1) | `model-v2.b70` |

**The 10,194 figure predates the BF16-critical-tensor fix.** That fix (section 1a) is
what made long-prompt output coherent, and on Ornith it moves `mlp.gate` (the MoE
router, 41 tensors) and `mlp.shared_expert_gate` from 4-bit to BF16. Ornith is the MoE
model, so unlike Qwen this sits on its hot path. **A speed cost there is the expected
price of the correctness fix** — but it has NOT been measured directly, because the old
`model.b70` is a v1 format the current loader rejects (`invalid or non-B70 native
model`), so a clean A/B on the artifact is not possible without re-converting.

**A ~6% gap remains unexplained** between this morning's reported 9,702.6 and today's
9,116-9,182 on the same artifact and same code path. Candidates not yet excluded:
GPU clock/thermal state (the notes already attribute an earlier 400.7 vs 409-419 ms
shift to a reboot, and the box has been under continuous GPU load for 2.5 h), or a
flag/bridge difference in how this morning's figure was taken. The xe driver on this
host exposes no `gt_cur_freq_mhz` or `temp1_input`, so throttling could not be
confirmed or ruled out.

### 15c. Honest status

Ornith is **not broken**: it loads, runs, and produces coherent output at 9,116-9,182
PP and ~111 tok/s TG. It is **below** the 10k gate. The dominant, documented cause is
the correctness fix, not this session's tuning — which is proven neutral. **Whether to
keep BF16 routers (correct, ~10% slower) or revert (fast, incoherent on long prompts)
is Ian's call, not mine to make silently.**

To settle it properly: re-convert Ornith with the current converter but with
`mlp.gate` / `mlp.shared_expert_gate` left in MXFP4 (keeping only `in_proj_a` /
`in_proj_b` in BF16), and measure. That isolates the router cost from the delta-rule
gate cost, and may recover most of the speed while keeping coherence — the delta-rule
gates are the ones with the exponentiation/matrix-inverse amplification argument, the
router is a much weaker case.

---

## 16. RETRACTION of section 15 — Ornith was NEVER regressed. I measured it wrong.

**Section 15's premise was false and its conclusion is withdrawn.** Ian was right on
both counts: the BF16 fix did not cost PP, and the drop was caused by a change — mine,
in how I invoked the benchmark, not in the code.

### 16a. The two mistakes in my Ornith runs

This morning's run (transcript line 1138) that produced 422.2 ms was:
```
GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_raw.so
tools/b70run.sh ... /grimoire/bin/grimoire -m ...Ornith... --proj mxfp4 --ctx 8192
```
Every Ornith run I did today used instead:
- `--proj int4` **not** `--proj mxfp4` — quantizes the projections to int4, a different
  and slower path for this model
- `libgrimoire_xe2_attention_bridge.so` **not** `libgrimoire_xe2_attention_raw.so` —
  the wrong attention library (my `tools/qrun.sh` / `tune.sh` hardcoded the wrong one)

### 16b. Corrected measurements — Ornith is FASTER than this morning

| Ornith @4096, GPU | config | ms | tok/s |
|---|---|---|---|
| GPU 0 | **correct** (`--proj mxfp4`, attention_raw) | **418.3** | **9,792.3** |
| GPU 1 | **correct** | **417.0** | **9,822.5** |
| GPU 0 | 2026-08-25 this morning, post-BF16-fix (notes) | 422.2 | 9,701 |
| GPU 0 | pre-BF16-fix (notes) | 419.2 | 9,771 |
| GPU 0/1 | my wrong config (`--proj int4`, attention_bridge) | ~449 | ~9,116-9,182 |

**The BF16-critical-tensor fix cost ~0.7% on Ornith — nothing.** Section 15's
"correctness-vs-speed trade" was invented to explain an artifact of my own bad command
line. There is no trade to make and nothing for Ian to decide. Withdrawn.

The ~3.7% still separating 9,822 from the 2026-08-24 record of 10,194-10,222 is
consistent with that record's own note: it was taken immediately after a reboot, and
this box has been under continuous GPU load for ~2.5 h. Re-measure on a fresh boot
before treating the gate as unmet.

### 16c. Also still true from section 15

The code-change A/B results stand and remain useful: reverting `prefill.cpp` entirely
(9123.8), forcing the original tiles and no swizzle (9100.7), and restoring
`libgrouped_gemm_xe_2.so` (9110.3) all reproduce the same number **as each other**,
because they all shared my wrong `--proj int4` / wrong-attention-bridge invocation.
They correctly show this session's kernel changes are neutral for Ornith; they simply
could not reveal the flag error, since every arm had it.

### 16d. THE RUN RECIPE — use this, do not improvise

```bash
cd /mnt/storage/isos/grimoire-fuse
# ORNITH (MoE):  --proj mxfp4, --ctx 8192, attention_RAW bridge
GPU=gpu1 EXTRA_ENV="GRIMOIRE_BENCH_PREFILL_TOKENS=4096
GRIMOIRE_DEFER_MOE_GATHER=1
GRIMOIRE_BF16_QKV=1
GRIMOIRE_BF16_DN_QKV=1
GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_raw.so" \
  bash tools/tune.sh orn /grimoire/bin/grimoire \
  -m /models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE --proj mxfp4 --ctx 8192
```
**Ornith and Qwen do NOT share a flag set.** Ornith's MoE flags corrupt Qwen
(section 6) and Qwen's `--proj int4` costs Ornith ~7%. `tools/qrun.sh` and
`tools/tune.sh` default to the QWEN attention bridge — override it for Ornith.

**Lesson for me, recorded plainly:** I spent an hour building an elaborate wrong
explanation (a correctness-vs-speed trade, thermal throttling, a missing library)
for a number produced by my own mis-typed command line. The transcript of the run
that set the baseline was on disk the whole time. **Read the command that produced
the number you are trying to reproduce, before theorising about why it changed.**

---

## 17. ANSWERED: int4 beats MXFP4 on the FFN shapes. Path to 2000 PP is now concrete.

Measured kernel-vs-kernel, same GPU, same shapes, M=4096, random (non-compressible)
data. vLLM's `torch.ops._xpu_C.int4_gemm_w4a16` vs grimoire's production MXFP4:

| shape | int4 g128 | grimoire MXFP4 | |
|---|---|---|---|
| ffn-gate-up 34816x5120 | **11.44 ms / 127.6 TF** | 14.61 ms / 100.0 TF | int4 **1.28x** |
| ffn-down 5120x17408 | **4.32 ms / 169.1 TF** | 6.98 ms / 104.6 TF | int4 **1.62x** |
| dn-qkv 10240x5120 | 3.47 ms / 123.9 TF | **3.20 ms / 134.3 TF** | *grimoire 1.08x* |

group_size 128 beats 32 for int4 (gate-up 11.44 vs 12.16, down 4.32 vs 5.14).

**Projected:** converting ONLY the two FFN weights to int4 saves
`64 x (3.16 + 2.66) = 373 ms` -> 2470 -> **~2097 ms = ~1953 tok/s**, while KEEPING
MXFP4 for dn-qkv where grimoire is faster. That is the 2000 gate, and it explains
vLLM's 2015.6 exactly.

### 17a. No external kernel needed — grimoire's template already has INT4

`csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp:59` — `enum class B_DTYPE { BITS16, INT4,
MXFP4, PER_TENSOR_FP8, MXFP8, BLOCK_FP8 }`. Grimoire's `xe2_grouped_bridge.cpp` only
ever instantiates `MoE::B_DTYPE::MXFP4` (lines 105, 115). **The same cutlass template,
same policy, same build, with `B_DTYPE::INT4` gives the int4 path.** No pybind, no
`auto_round_kernel`, no runtime dependency on vLLM.

(Section 13b's pessimism about the pybind `.so` having no C ABI is therefore moot —
we never needed to call it.)

### 17b. The work, concretely

1. **Bridge:** add `grimoire_xe2_dense_int4_bf16` / `_f32` — copy the existing dense
   entry points, swap `B_DTYPE::MXFP4` -> `B_DTYPE::INT4`. Reuse `p128x256`.
   Re-run the production policy sweep for int4 (`GRIMOIRE_DENSE_POLICY`) since the
   optimum may differ.
2. **Converter** (`tools/b70_compile_model.cpp`): emit int4 group-128 for
   `mlp.gate_up_proj` / `mlp.down_proj` only; keep MXFP4 elsewhere; keep the
   `keeps_bf16()` list. Add `NativeEncoding::INT4_*`; the runtime already has
   `Fmt::INT4`.
3. **Loader** (`qwen35_loader.cpp`): map the new encoding to `Fmt::INT4` with correct
   `row_bytes` / `row_scales`.
4. Measure. Expect ~1950 PP. Then re-check TG.

**A mixed-format artifact is the right answer** — int4 where int4 wins (FFN), MXFP4
where MXFP4 wins (dn-qkv). Neither vLLM nor a generic quantizer does this; it is
exactly the "quantized specifically for B70" artifact Ian has been asking for.

### 17c. TG, unchanged and independent

Still 23.2 vs 28 target. Lever untouched: split-K for low-N GEMV shapes — `k+v`
(N=2048) at 108 GB/s and `ab` (N=96) at **9 GB/s** against a 602 GB/s roofline,
worth ~3 ms of the 43 ms/token. Do in parallel; no dependency on the int4 work.

---

## 18. !! RETRACTION OF SECTION 17 !! int4 is NOT the answer. Do not build it.

Section 17 concluded int4 beats MXFP4 by 1.28-1.62x and recommended a mixed-format
artifact. **That comparison was invalid and the conclusion is withdrawn.**

The error: it compared **vLLM's int4 measured in isolation** (a standalone microbench,
one weight matrix reused across repetitions, hot in cache) against **grimoire's MXFP4
measured in production** (64 distinct weight matrices streaming, 5.7 GB per forward
pass, nothing cached). Apples to oranges.

### 18a. Correct comparison — GRIMOIRE's own INT4 vs its own MXFP4

`tools/bench_int4_dense.cpp`, same binary, same policy (`p128x256`), same random
non-compressible buffers, M=4096, best-of-5:

| shape | MXFP4 | INT4 (best group) | winner |
|---|---|---|---|
| ffn-gate-up 34816x5120 | **11.840 ms / 123.3 TF** | 12.435 ms (g32) / 117.4 TF | MXFP4 |
| ffn-down 5120x17408 | **5.921 ms / 123.3 TF** | 6.042 ms (g32) / 120.8 TF | MXFP4 |
| dn-qkv 10240x5120 | **3.462 ms / 124.1 TF** | 3.473 ms (g128) / 123.7 TF | tie |

**INT4 and MXFP4 are equivalent on this kernel — MXFP4 marginally ahead.** The
`B_DTYPE::INT4` instantiation works (added as `launch_dense_int4` in
`xe2_grouped_raw_launcher.hpp`, kept for reference) but buys nothing.

**Do not write the int4 converter. Do not build the mixed-format artifact.**
Section 17's plan would have cost days for zero gain.

### 18b. What the numbers ACTUALLY say

| | ms/layer, ffn-gate-up |
|---|---|
| vLLM int4, isolated | 11.44 |
| **grimoire MXFP4, isolated** | **11.84** |
| grimoire MXFP4, in production | **14.61** |

**Our kernel is within 3.5% of vLLM's.** The kernel was never the problem — section 13
and 17 both mis-attributed the gap.

**The gap is our pipeline, not our GEMM:** the same kernel is 2.77 ms/layer slower in
production than in isolation. Over 64 layers that is **~177 ms** on gate-up, plus
~68 ms on ffn-down (6.98 production vs 5.92 isolated) = **~245 ms**.
2470 - 245 = **~2225 ms = ~1841 tok/s**, and that is only the FFN.

Caveat, stated honestly: part of that delta is inherent — the isolated bench reuses
ONE weight matrix across repetitions so it sits in cache, while production streams 64
distinct 89 MB matrices. The isolated number is therefore optimistic **for vLLM too**;
both microbenches share the bias, which is exactly why comparing microbench-to-
microbench (11.44 vs 11.84) is the valid comparison and microbench-to-production is not.

### 18c. Where to look next — the pipeline, not the kernel

vLLM reaches 2015.6 end-to-end with a kernel 3.5% faster than ours. So its advantage
is almost entirely **outside the GEMM**. Candidates, none yet tested:
1. **Chunked prefill** — vLLM ran with `enable_chunked_prefill=True` and
   `max_num_batched_tokens=16384`. It may never do a single M=4096 pass at all.
2. **Kernel overlap / fewer syncs.** GRIMOIRE's prefill is a serial chain of ~11
   regions per layer on an in-order queue. vLLM uses inductor with
   `combo_kernels: True` and `benchmark_combo_kernel: True` — it fuses adjacent
   elementwise work into the GEMM epilogues.
3. **Weight streaming/prefetch.** 5.7 GB of weights per forward pass is the dominant
   traffic; whoever prefetches the next layer's weights during the current GEMM wins.

Measure vLLM's per-layer breakdown (torch profiler on the same prompt) and diff it
against the section 9c region budget. That is the same diff-against-vLLM method that
solved both bugs in section 1, applied to the pipeline instead of the kernel.

### 18d. TG untouched

Still 23.2 vs 28. The split-K lever for low-N GEMV shapes (`k+v` 108 GB/s, `ab`
**9 GB/s** vs a 602 GB/s roofline) remains the cheapest independent win and is
unaffected by any of the above.

---

## 19. ROOT CAUSE LOCATED: the FFN GEMM is DRAM-bound on B re-reads

`tools/bench_stream.cpp` isolates the one variable that separates the microbench from
production: **how many distinct weight matrices stream through**.

| shape | 1 matrix reused (cache-hot) | 64 distinct matrices | production (s9c) |
|---|---|---|---|
| ffn-gate-up | 11.865 ms | **14.650 ms** | 14.607 ms |
| ffn-down | 5.921 ms | **7.446 ms** | 6.980 ms |
| delta/layer | | **+2.785 / +1.526 ms** | |

**The 64-matrix number reproduces production to within 0.3%.** So:
- The overhead is NOT pipeline, NOT sync, NOT launch cost, NOT instrumentation.
- It IS memory traffic: with distinct weights nothing stays cached, and the kernel
  pays full DRAM cost for every re-read of B.

### 19a. The arithmetic

gate_up B payload is 89 MB/layer. Reading it **once** at the measured 602 GB/s roofline
costs 0.148 ms. The observed penalty is **2.785 ms = ~19x that** — so B is being
re-read roughly 19 times per layer.

That matches the tile schedule: `mt = 4096/128 = 32` m-tiles, `nt = 34816/256 = 136`
n-tiles. The default walk sweeps all n-tiles for one m-tile, so **B is re-streamed once
per m-tile = 32x**. The `GRIMOIRE_GEMM_SWIZZLE_GM=4` grouping (s9) cut that to ~8x,
which is exactly the ~21 ms/layer -> 14.6 ms/layer improvement already banked.

**Remaining headroom: 64 x 2.785 = 178 ms (gate-up) + 64 x 1.526 = 98 ms (down)
= ~276 ms.** 2470 - 276 = **~2194 ms = ~1867 tok/s** if B re-reads were eliminated
entirely; a large fraction of that is realistically reachable.

### 19b. Why the swizzle stalled at GM=4

Measured GM=2/4/8/16 -> 942/936/937/953 ms. Grouping M gives B-tile reuse across GM
workgroups, but the A working set grows as `GM x 1.31 MB`, so past GM=4 A starts
evicting B and the two effects cancel. **The fix is 2-D blocking over BOTH m and n**,
sized to L2: with BM=8 / BN=33 the traffic model gives
`(32/8)x89 + (136/33)x41.9 = 529 MB/layer` versus `2890 MB` for the naive walk — a
5.5x reduction, where the current GM=4 achieves only ~4x on the B term alone.

### 19c. Next action

1. Replace the GM-only swizzle in `launch_dense_mxfp4` with a **2-D block swizzle**
   (`BM`, `BN` both env-tunable), sweep on the real shapes.
2. Cross-check against vLLM: it hits 2015.6 with a kernel only 3.5% faster than ours
   in isolation (s18b), so it must already be doing this. Its tile scheduler is in
   `csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2.hpp` — read how it maps
   `group_id -> (m_tile, n_tile)` and copy the scheme.
   **This is the same "check the vLLM pipeline" move that took Ornith from 700 to
   10,000.**
3. `tools/bench_stream.cpp` is the right harness for iterating: it reproduces
   production timing without a 20 s model load.

### 19d. State of the tree at handover

- `launch_dense_int4` added to `xe2_grouped_raw_launcher.hpp` — works, unused, keep
  for reference only (s18).
- `tools/bench_int4_dense.cpp`, `tools/bench_stream.cpp` — new, built, in `bin/`.
- Production config unchanged and verified: Qwen 1662.6 tok/s, Ornith 9792/9822.
- Backups in `.backups-20260825/`.

---

## 20. Streaming-condition sweep — corrects s18, and sizes the real headroom

`tools/bench_sweep.cpp`: every policy AND int4 group size, measured with **64 distinct
weight matrices** (production memory conditions), M=4096.

### 20a. ffn-gate-up (N=34816, K=5120)

| policy | ms/layer | TFLOP/s |
|---|---|---|
| **INT4 g128** | **13.799** | **105.8** |
| INT4 g32 | 14.689 | 99.4 |
| MXFP4 128x256 | 14.648 | 99.7 |
| MXFP4 256x128 | 15.091 | 96.8 |
| MXFP4 128x128 | 15.878 | 92.0 |
| MXFP4 64x64 | 23.578 | 61.9 |

### 20b. ffn-down (N=5120, K=17408)

| policy | ms/layer | TFLOP/s |
|---|---|---|
| **INT4 g128** | **7.026** | **103.9** |
| INT4 g32 | 7.422 | 98.4 |
| MXFP4 128x256 | 7.446 | 98.1 |
| MXFP4 128x128 | 7.961 | 91.7 |
| MXFP4 64x64 | 11.515 | 63.4 |

### 20c. This partially un-retracts s18 — with a correction

Section 18 concluded "int4 buys nothing". That was measured **cache-hot with one
matrix**, and the harness picked g32 as int4's best. Under **streaming** the ranking
changes: **int4 g128 beats MXFP4 on both FFN shapes.**

- gate-up: 14.648 -> 13.799 = **0.849 ms/layer**
- down:    7.446 -> 7.026 = **0.420 ms/layer**
- **64 x 1.269 = ~81 ms** -> 2468 - 81 = ~2387 ms = **~1714 tok/s**

So int4 g128 is worth ~3%, not the 1.28-1.62x of s17 nor the "nothing" of s18.
**Group size is the active variable, not int vs float**: MXFP4 is locked to group 32,
so ffn-down (K=17408) pays 544 dequant-rescales per output tile against 136 for g128.
That is why the gap appears on the large-K shape and not on gate-up.

### 20d. Honest headroom accounting for the 2000 gate

Current 2468.3 ms (1659.5 tok/s). Gate needs 2048 ms. **Must find 420 ms.**

| source | ms | status |
|---|---|---|
| int4 g128 on both FFN GEMMs | ~81 | measured; needs converter + loader work |
| vLLM's ffn-down advantage beyond our int4 (5.355 vs 7.026) | ~107 | cause unknown |
| **GEMM total** | **~190** | |
| **remaining, outside the GEMM** | **~230** | not yet located |

The FFN GEMMs are now within ~6% of vLLM's on gate-up and the whole-GEMM headroom is
~190 ms of the 420 needed. **The rest is not in the dense GEMM.** Next place to look is
the non-FFN 991 ms: `full attention` 229.5 ms (vLLM uses its FLASH_ATTN backend) and
the seven DN regions totalling ~715 ms (vLLM uses a Triton/FLA fused GDN prefill
kernel). Those two subsystems are where the remaining ~230 ms must come from.

### 20e. Incident

One `DEVICE_LOST` during this sweep, caused by my own harness bug: the bf16 scale
buffer was sized for g128 while the g32 variant reads 4x more scales -> out-of-bounds.
Fixed by sizing for g32. **Both GPUs recovered `active`, no D-state, no stuck
containers**, and Qwen re-measured clean at 1659.5 tok/s afterwards. Production code
was never involved -- the fault was in `tools/bench_sweep.cpp` only.
