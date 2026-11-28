# LoRA adapters (dense trunks, single card)

GRIMOIRE loads a PEFT [LoRA](https://huggingface.co/docs/peft/lora) adapter on top of a
dense (non-MoE) checkpoint: `--lora <dir>` on the CLI, or `GRIMOIRE_LORA=<dir>` in the
environment. The adapter is applied to every layer's attention (q/k/v/o) and MLP
(gate/up/down) modules, on the trunk only -- the speculative/MTP/DFlash heads are never
LoRA'd. It is not fused into the weights; each delta is a small SYCL kernel pair added on
top of the projection, so the adapter is visible to any token that the engine emits.

```
make native          # AOT build, both B70 dies by default
bin/native-<targets>/grimoire --model /models/<checkpoint> --proj mxfp4 --lora /path/to/adapter \
    -p "The answer is" -n 128
```

The server takes the same flag:

```
bin/grimoire-server --model /model --proj mxfp4 --lora /path/to/adapter --port 8000
```

## What is supported

- PEFT adapters with `peft_type: "LORA"` in `adapter_config.json`, `r >= 1`,
  `lora_alpha > 0`, and F32 `lora_A` / `lora_B` tensors in `adapter_model.safetensors`.
- Modules: `q_proj` / `k_proj` / `v_proj` / `o_proj` (attention), `gate_proj` / `up_proj` /
  `down_proj` (MLP). A module absent from the adapter is simply not applied (the engine
  skips its delta); a module present in the adapter but unsupported (or past the model's
  layer count) fails the load with a message naming the tensor.
- Full-attention (GQA) layers only carry q/k/v/o deltas. Linear-attention (DeltaNet) layers
  carry the MLP deltas only: a q/k/v/o delta on a non-full-attention layer is rejected.
- Qwen3-style `attn_output_gate` checkpoints (q_proj emits [q|gate]) are supported; the
  delta covers the full packed output. The same is not true for K2/MoVA models: their
  V is router-routed and is rejected outright. A q/k/v delta on any layer is refused.

## What is refused at startup

- MoE trunks (`is_moe()`) and K2/MoVA models (`is_k2`).
- Tensor parallel (`--tp N`) and pipeline parallel (`--pp N`): the delta kernels are
  single-card layouts. The check is in `Grimoire::build()`; the load fails with the reason.
- Muse (sandwich norms), gemma-4 (dual-geometry layers) and Qwen4-Exp (hyper-connections /
  indexer): their forward paths are not the dense trunk path, so the adapter could be
  silently half-applied.
- Adapters carrying a `v` delta on a MoVA (k2_sparse) layer.
- `use_dora`, `use_rslora`, non-integer or non-positive `r`, missing/non-positive
  `lora_alpha`, non-LORA `peft_type`, F16/BF16/INT weights, or a `lora_A`/`lora_B` pair
  that is incomplete or malformed (missing B, extra A rows, …). A failure anywhere names
  the offending tensor, layer, file or module.

## Design

Loader (`src/lora.cpp`, host-safe):

1. Parses `adapter_config.json` (name = `base_model_name_or_path`), pads `r` to `r16`
   (a multiple of 16) and computes `scale = lora_alpha / r`.
2. Reads every `base_model.model.[model.language_model.]layers.{L}.{module_path}.lora_{A,B}.weight`
   tensor, validates the module name, layer range, full-attention placement and shapes
   (`A` is `[r, k]`, `B` is `[n, r]`, both F32), and converts F32 -> bf16.
3. Packs every (layer, module) into one contiguous host blob (`A` [r16][k], `B` [n][r16],
   row-major, row-padded with zeros). Per-layer views point into the blob; the engine
   hands one view to each delta launch.

Device (`src/lora_dev.cpp`, SYCL): the blob is copied once at load; `ensure_scratch(rows)`
grows a shared fp32-packed-bf16 mid buffer (`[rows][r16]`) as needed. Each module pair
launches two kernels on the engine's queue:

- mid: `mid[r][j] = bf16(sum_c bf16(A[j][c]) * x[r][c])` -- fp32 mid, rounded to bf16 once;
- delta: `out[r*out_ld + out_off + i] += scale * sum_j bf16(B[i][j]) * bf16(mid[r][j])`
  -- accumulated in fp32 into the caller's fp32 buffer.

Single-card layouts, matching the engine's dense path:
- column-parallel (q/k/v, gate/up): the delta lands in the module's own output columns;
  the up half of the fused gate|up row is written at offset `inter` (`n = inter`);
- row-parallel (o, down): the delta lands in the full output row.

## Delta placement in the forward paths

- Decode: q/k/v deltas land on `s.qkv` / `s.zbuf` / `s.bbuf` before the split-qgate /
  norm / RoPE; the o delta lands on `s.moe_y` after the o_proj gemv and before the all-reduce; gate/up land on `s.sh_g` before SwiGLU; the down delta lands on `s.moe_y`
  after the down GEMV.
- Prefill: the same deltas land on `t0` / `t3` / `t4` (q/k/v), `t3` + `r0` (o), and
  `t0` / `r0` (gate/up/down). With a LoRA adapter set, the dense prefill takes the
  reference path (the fp32 gate|up buffer and plain `mm` projections): the fused/bf16
  fast paths and the xe2/xe2_attention fused paths are off. This is the correctness
  baseline, not the fastest one -- the delta needs the fp32 buffers the reference path
  owns. The mid scratch is grown to the prompt's row count before the layer loop; a
  growth invalidates the decode command graph (`graph_ok = false`) so the next decode
  re-records with the new mid pointer.

## Verification

Host gates (no GPU, C++17): `make test_lora` (and `make test`, which includes it) covers
the config rejects, the mapping/shape rejects, the r16 pad/zero-rows, the F32 -> bf16
packing, the reference delta math against an independent hand-rolled reference (m=1 and
m=32, every module and layer), the zero-adapter invariance, and the hermes-shaped Qwen3.8-27B
mapping (hidden 5120, 48 query heads, head_dim 128, attn_output_gate, 16 GQA layers) as a
synthetic fixture. The real adapter (Qwen/Qwen3.8-27B, r=32/alpha=64, 24 query heads, head_dim
256, 4 KV heads, 16 GQA layers at i%4==3) was additionally verified by loading it against
the checkpoint's exact geometry; it passes (hidden 5120, 17408 intermediate, 16 attn layers,
all 64 layers carry the MLP deltas).

GPU e2e on the Arc Pro B70 pair is verified: the AOT image (`docker build -f docker/Dockerfile`,
B70_TARGET=intel_gpu_bmg_g31, oneAPI 2026.2.1 + ocloc 26.40) loaded the real hermes adapter on
a live card and generated with it -- `r=32, r16=32, scale=2.000 [Qwen/Qwen3.8-27B]`, 64 layers
resident, decode and 4096-token prefill both through the delta kernels, and a correct short
generation (`The capital of France is` -> `Paris.`, finish=stop). This ran the same build the
host gates ran; the adapter is exercised on device end to end.

Note on the environment: in the stock `intel/oneapi:2026.2.0-devel` container on the build box,
`sycl::device::get_devices()` segfaults before any engine print runs (a plain SYCL probe with no
GRIMOIRE code crashes the same way, with and without a LoRA adapter set). GRIMOIRE's own
`docker/Dockerfile` -- the kobuk-PPA user-mode driver stack -- enumerates the same cards fine,
so use that image, not the stock devel one, for GPU runs on that box.
