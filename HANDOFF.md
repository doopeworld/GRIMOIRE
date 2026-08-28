# GRIMOIRE — Handoff

Bare-metal SYCL/DPC++ inference engine for the Intel Arc Pro B70 (Battlemage
G31). Reads Hugging Face safetensors directly, AOT-compiled kernels, USM device
pointers. No PyTorch / OpenVINO / llama.cpp. Target model family is Qwen3.5/3.6
MoE (hybrid Gated-DeltaNet + full attention, 256 experts top-8 + shared expert).

Work from the **pristine v30 source** (the tarball the owner uploaded first).
Do NOT start from any of the v31–v42 experiments — they introduced a batched
prefill path with an unresolved GPU hazard and should be treated as scratch.

---

## 1. Current known-good state

- Decode (TG) works and is fast: ~190–195 tok/s on the B70 for the 35B MoE at
  MXFP4, measured, `--proj bf16`.
- Output is correct with the chat template (produces coherent text).
- `make test` (host CPU tests) passes.
- The engine selects the GPU via `sycl::queue` with `in_order` + `gpu_selector_v`
  in `src/main.cpp`. NOTE: there are TWO GPUs visible on the owner's box — the
  discrete **Arc Pro B70** and an integrated **Arc iGPU**. Pin the discrete one
  (`ONEAPI_DEVICE_SELECTOR=level_zero:0` or verify by name) so you never
  benchmark the iGPU by accident.

### IMPORTANT environment caveat (unresolved, verify first)
The Dockerfile installs `intel-oneapi-compiler-dpcpp-cpp` **unpinned**, so a
rebuild pulls whatever is newest. As of this writing the newest is
**2026.1.1-325**. During the last session, rebuilt images produced garbage
output while the owner's vLLM (which uses a frozen/older oneAPI) worked fine on
the same GPU. This strongly suggests a compiler-version regression, but it was
NOT definitively confirmed. **First thing to do:** pin the compiler to a known
value and rebuild `--no-cache`, then run the plain "Hello" chat test. Available
versions include 2025.3.3-30, 2025.2.x, 2026.0.0, 2026.1.0, 2026.1.1. Pin in the
Dockerfile line:
```
intel-oneapi-compiler-dpcpp-cpp=2025.3.3-30
```
If a pinned older compiler produces correct output and 2026.1.1 does not, that
is the regression — pin permanently and note it. If pristine v30 is garbage on
every compiler, the problem is elsewhere and must be root-caused before any
feature work (do not build features on a broken base).

Sanity test (no template = tests the model math directly; must continue " Paris"):
```
grimoire -m /models/Ornith-1.0-35B-MXFP4 --proj bf16 --raw -p "The capital of France is" -n 8
```
Chat test (must be coherent English):
```
grimoire -m /models/Ornith-1.0-35B-MXFP4 --proj bf16 -p "Hello" -n 20
```

---

## 2. Goal A — Prompt-processing (PP / prefill) speed

Target: match or beat vLLM's `--pp 4096` (~10k tok/s on this box); the owner
believes ~14k is reachable.

### The real problem
`Grimoire::forward()` processes ONE token per call. Prefill currently loops it
per prompt token, so the fast batched XMX GEMM is never used on real prompts.
The "~4550 tok/s" figure that has been quoted is a MICRO-BENCHMARK of a single
qkv projection at M=4096 (`grimoire_bench_prefill`), NOT end-to-end prefill.

### What exists to build on
- `src/gemm_xmx.cpp` — batched XMX (`joint_matrix`) GEMM, proven. A sweep found
  `M_PER_SG=2` is the sweet spot (~8778 GFLOP/s) — use that, not the default.
- `src/prefill.cpp` — already contains `launch_gemm_batched` and
  `launch_deltanet_prefill` (resident-state DeltaNet for prefill). These are the
  batched building blocks.
- Do NOT pre-dequantize weights to bf16 before the XMX GEMM: the sweep showed
  `xmx+deq` is SLOWER than feeding MXFP4 straight in (bf16 is 2× the bytes, and
  this path is bandwidth-bound). Feed the quantized weights to the kernel.

### What must be written
A real batched `forward_prefill(ids, n)` that runs all n prompt tokens in one
pass. Batch the expensive, token-independent work (all projections via XMX; the
routed-expert MoE across tokens; DeltaNet via the prefill kernel). Cheap
elementwise ops (norms, RoPE, gates) can stay per-token initially.

### Hard lessons from the failed attempt (avoid these)
- The last attempt hit `UR_RESULT_ERROR_DEVICE_LOST` at pp≥128. Root cause was
  NOT found. Symptoms: per-kernel `q.wait()` made it run; per-layer wait still
  crashed. That points at an inter-kernel buffer hazard (a kernel reading a
  batched buffer before the writer finished) OR an out-of-bounds index in one of
  the batched kernels, not a simple queue-depth issue. Whoever does this should
  validate each batched kernel's indexing against a CPU reference (as was done
  for a plain batched GEMM and for causal attention — those CPU tests passed),
  then bring up the batched path one layer-type at a time (full-attention layers
  first, then DeltaNet), checking the FIRST generated token matches the
  per-token path before trusting anything.
- VRAM: batch scratch was sized as `N × …` and one buffer (`moe_h` =
  N·top_k·moe_inter·4) hit 64 GiB at N=4096 — impossible on a 32 GB card. Real
  prefill must be **chunked** (process e.g. 256–512 tokens per pass, loop the
  chunks), exactly like vLLM's `--max-num-batched-tokens`. Size scratch for the
  chunk, not the whole prompt.
- A batched shared-expert buffer was sized by `cfg.shared_inter` but written at
  stride `sh_gu.w.N/2` (the real projection width). Size batch buffers by the
  ACTUAL tensor dims, not the config field.

---

## 3. Goal B — MTP (multi-token prediction) for faster TG (~400 tok/s)

The owner runs MTP successfully in vLLM and wants it here. It is genuinely
model-supported, not something to invent.

### The blocker: MTP weights only exist in GPTQ checkpoints
- `Ornith-1.0-35B-MXFP4` (the working model) has ZERO `mtp.*` tensors — that
  export stripped them.
- The models WITH `mtp.*` weights are all **GPTQ** (`quant_method: gptq`,
  bits 4, group_size 128, `sym: true`, `desc_act: false`):
  - `Qwen3.6-35B-MTP` (MoE)
  - `Qwen3.6-35B-A3B-GPTQ-Int4` (MoE) ← owner's stated fast-MTP target
  - `Qwen3.8-27B-GPTQ-Int4-MTP-BF16` (DENSE)
- So MTP requires a **GPTQ loader first**. There is no MXFP4/INT4 shortcut.

### GPTQ format facts (verified against the real files)
- Tensor prefix is nested: `model.language_model.layers.N.…` (vision-style),
  not `model.layers.N.…`.
- Per projection, a triple replaces `.weight`: `.qweight` (I32), `.qzeros`
  (I32), `.scales` (F16), plus `.g_idx` (I32, ignorable since desc_act=false —
  it was verified to be the identity group map `k//128`).
- Shapes (gate_proj example, logical out=512 in=2048): qweight `[in/8, out]` =
  `[256,512]`; qzeros `[in/group, out/8]` = `[16,64]`; scales `[in/group, out]`
  = `[16,512]`. down_proj is reversed dims (out=2048 in=512 → qweight `[64,2048]`).
- Dequant formula (AutoGPTQ convention, VERIFIED bit-exact against a Python
  reference across many output channels): `w = (nibble - (zero_nibble + 1)) *
  scale`. The `+1` on the packed zero is the single most common porting error.
- A correct host dequantizer was written and CPU-verified (dequant matched the
  reference to ~1e-7; a full pipeline dequant→requant-to-MXFP4→GEMV matched
  ground truth to ~0.16 rel-L2, which is expected for stacked 4-bit). This code
  is NOT in the pristine v30 tree — it lived in the scratch branch. It can be
  re-derived from the formula above, or lifted from the scratch tarball if kept.
- Multi-shard gotcha: a single layer's 256 experts can SPAN two shard files
  (e.g. layer 16 had e0 in shard 3, e255 in shard 4). Resolve each of the
  qweight/qzeros/scales tensors to ITS OWN shard index, not one shared index.

### Recommended MTP approach
1. GPTQ loader: detect `quant_method: gptq`; for each projection, dequant the
   triple to f32 and feed the existing `quantize()` to land it as MXFP4/INT4 in
   VRAM (do NOT keep bf16 — a 35B would not fit). Reuse the same path for the
   routed experts (per-expert triples) assembled into the fused MoE block; and
   for the shared-expert gate|up concat.
   - This alone unlocks running ALL the owner's GPTQ models, which is
     independently valuable.
2. Load the MTP head: `mtp.fc` + one `mtp.layers.0.*` block (attention + MoE +
   shared expert). Config: `mtp_num_hidden_layers=1`,
   `mtp_use_dedicated_embeddings=false` (reuse the main embedding),
   `tie_word_embeddings=false` (reuse the main lm_head).
3. Speculative decode loop: main model emits token t0 and hidden h; run
   `mtp.fc([h ⊕ embed(t0)])` → the mtp layer → lm_head → draft token t1; verify
   by running the main model on [t0, t1] as a 2-token batch (this reuses the
   batched forward from Goal A); accept t1 if it matches, else fall back.
   `mtp_num_hidden_layers=1` → 1 draft token → up to ~2× TG. That is the path to
   the ~400 tok/s target.

Note: the DSpark/DFlash drafter models in the owner's collection
(`qwen38-dflash-drafter-fp8-b70`, `Qwen3.8-27B-DSpark`, hidden_size 5120,
`num_target_layers 64`) are a SEPARATE, heavier speculative-decode mechanism
aimed at the 27B DENSE line — bespoke confidence head, Markov head
(`markov_rank 256`), mask token 248077. That is a later, larger port; do the
built-in `mtp.*` head first for the MoE models.

---

## 4. Goal C — Run on every model (dense + all MoE quantizations)

The loader (`src/qwen35_loader.cpp`) is Qwen3.5-MoE-specific and only reads the
MXFP4 `weight_packed`/`weight_scale` expert layout. To generalize:

- **Formats:** add the GPTQ path from Goal B (covers `-GPTQ-`, and by extension
  the MTP models). The owner also has `auto-round`, `awq`, and `fp8`
  quantizations — each is a distinct on-disk packing; add them as separate
  decode paths feeding the same `quantize()`/upload flow. Owner model list
  includes: Ornith-35B (MXFP4/FP8/int4-AutoRound/AWQ), Qwen3.6-27B and
  Qwen3.8-27B (dense, various GPTQ/AWQ/AutoRound/FP8/int4-ov), Mistral-Small-24B
  (dense, OpenVINO-IR — may be OV-only, check for safetensors), Muse-Glimmer-30B
  (dense, 52 layers, INT4/MXFP4).
- **Architectures:** add an arch dispatch keyed on `config.json`
  `architectures`/`model_type`. Dense models are the EASIEST new arch — they are
  the existing layer minus the MoE router/experts (single MLP per layer). Do a
  dense Qwen (L=64) first; it validates the non-MoE path with minimal new code.
  Then Mistral/Llama-style attention if needed.

Suggested order: GPTQ loader (unblocks MTP + a third of the model zoo) →
dense-Qwen arch → remaining quant formats → other archs.

---

## 5. Repo map (pristine v30)

```
src/main.cpp            queue/device selection (in_order + gpu_selector_v)
src/grimoire.cpp        engine: build, forward (single-token), graph capture, generate
src/qwen35_loader.cpp   config.json + safetensors -> tensor refs (Qwen3.5-MoE specific)
src/safetensors.cpp     mmap + header parse + read_raw (by byte offset)
src/quantize.cpp        host quantize() to the 7 formats; PackedWeight/QuantWeight
src/gemm_xmx.cpp        prefill GEMM, joint_matrix/XMX (M_PER_SG=2 sweet spot)
src/gemv_decode.cpp     decode GEMV, bandwidth-bound
src/attention.cpp       single-token FlashDecoding (online softmax)
src/deltanet.cpp        single-token Gated DeltaNet step
src/prefill.cpp         batched GEMM + DeltaNet-prefill building blocks (for Goal A)
src/moe_kernels.cpp     fused MoE gate_up / down kernels (single-token)
src/ops.cpp             norms, RoPE (partial rotary), gates, KV append, probe
src/tokenizer.cpp       byte-level BPE + chat template (248320 vocab)
include/b70/*.hpp       Config, TensorRef, QuantWeight, MoeLayer, kernel decls
tests/                  host CPU validation — `make test`, no GPU needed
docker/Dockerfile       build image; PIN the oneAPI compiler here (see §1)
build_b70.sh            AOT compile for intel_gpu_bmg_g31
```

### Debug flags in the CLI (useful for bring-up)
- `--raw` — feed prompt with NO chat template (tests model math directly).
- `--top N` — print the N highest logits per step (id, value, %, decoded text):
  tells you if the right token was a close second (small bug) or nowhere (math
  broken).
- `GRIMOIRE_DEBUG=1` — per-stage numeric probes; NOTE this calls `q.wait()`
  everywhere and roughly halves throughput. It is the instrument, not the engine
  — never quote its tok/s as the real speed.

### Working style the owner expects
Find the root cause before rebuilding — read the code and inspect the real files
(the owner can run any command and read any model file on request), rather than
building repeatedly to see what happens. Keep changes minimal and reversible.
Verify kernel math against a CPU reference before running on the GPU. One
focused change at a time.
