# GRIMOIRE weight-loading architecture and GGUF source

The numbered inventory below records the original pre-GGUF source audit; its
line numbers describe that baseline. The GGUF adapter and native Q4_K/Q6_K
execution are now implemented. See [NATIVE_QK.md](NATIVE_QK.md) for the
current upload, dispatch, guard and verification status. Q8_0/IQ have no
native kernel dispatch.

## 1. Build and run

- `Makefile`'s default `all` runs the host `test` target; host compilation uses C++17 and `-Iinclude`.
  Its `native` target separately builds native-only SYCL binaries with external bridges disabled
  (`Makefile:1-11`, `Makefile:105-120`, `Makefile:125-160`).
- `build_b70.sh` sources oneAPI, requires `icpx`, probes `ocloc` for each selected Battlemage die, and
  defaults `B70_TARGET` to `intel_gpu_bmg_g31,intel_gpu_bmg_g21`; `B70_OUT` changes the standalone probe
  output name (`build_b70.sh:28-46`, `build_b70.sh:66-118`).
- It first builds the standalone `b70_native_inference` probe from `src/main.cpp`, then
  `bin/libgrimoire_gemm.so`, `bin/grimoire`, and `bin/grimoire-server`; the two inference binaries link
  the GEMM library with `$ORIGIN` rpath (`build_b70.sh:136-158`, `build_b70.sh:173-223`). `src/main.cpp`
  is a synthetic GEMV/device benchmark, not the model CLI (`src/main.cpp:19-21`, `src/main.cpp:67-115`).
- The AOT command uses C++20, `-fsycl`, `-O3`, `-fno-fast-math`, and a 256-GRF backend option per
  selected die; the large-M GEMM library is a separate device image so that option does not affect all
  engine kernels (`build_b70.sh:92-99`, `build_b70.sh:120-152`, `build_b70.sh:173-186`).
- `build_b70.sh` does **not** rebuild optional `src/*_bridge.cpp` shared objects.
  `tools/build_bridges_b70.sh` builds grouped, attention, GDN, and oneDNN bridges and then calls
  `build_b70.sh` when all bridges are selected (`build_b70.sh:370-378`,
  `tools/build_bridges_b70.sh:45-109`).
- Those bridges compile against headers/libraries in the `my-vllm-xpu:latest` builder image; the
  Dockerfile's runtime stage starts from Ubuntu 24.04 with Intel GPU/oneAPI runtime packages. The
  default `Dockerfile` instead builds a standalone image with pinned oneAPI 2026.1 and driver stack
  versions (`docker/Dockerfile.b70-native:3-28`, `Dockerfile:18-34`).
- `tools/preflight_b70.sh` uses `grimoire-dev:latest` for compilation and `grimoire-b70:latest` for
  execution; it checks `ldd` for missing dependencies and torch/vLLM linkage, then runs host and device
  gates (`tools/preflight_b70.sh:75-124`, `tools/preflight_b70.sh:128-152`). The pure launcher mounts
  `bin/` and `tools/`, hides `/opt/grimoire/lib`, and does not expose optional plugin `.so` files
  (`tools/b70run.sh:65-89`).
- CLI accepts `-m`, `--proj`, `--ctx`, `-p/--prompt`, `-n/--n-predict`, `--prefill-only`, and
  `--prefix-cache-test`. With no prompt it runs a load report (`tools/grimoire_main.cpp:38-93`). The
  server accepts `--model`, `--proj`, `--ctx`, `--port`, `--host`, `--dflash-model`, and the listed
  Ornith flags (`tools/grimoire_server.cpp:77-125`).
- Both CLI and server parse `--proj` into `Fmt::{BF16,FP8_E4M3,FP8_E5M2,INT8,INT4,MXFP8,MXFP4}` and
  default to INT4. `--ctx` defaults to 8192 in those entry points (`tools/grimoire_main.cpp:39-63`,
  `tools/grimoire_server.cpp:81-100`).
- `tools/serve.sh` defaults to `--proj mxfp4 --ctx 8192`, resolves a GPU name through `gpunode.sh`, and
  passes through selected `GRIMOIRE_*` environment variables (`tools/serve.sh:9-17`,
  `tools/serve.sh:24-55`). `tools/tune.sh` resolves `GPU`, passes `EXTRA_ENV`, and calls `b70run.sh`
  with `LIM` (`tools/tune.sh:1-9`).
- `b70run.sh` uses a detached Docker container with `--init`, an in-container `/usr/bin/timeout`,
  `docker wait`, and a long stop timeout; it propagates nonzero container status
  (`tools/b70run.sh:18-29`, `tools/b70run.sh:65-113`).
- An in-tree benchmark launcher invokes `uvx llama-benchy@0.4.0` against the HTTP `/v1` endpoint with
  `--tokenizer` pointing to the model directory, `--pp`, `--tg`, `--concurrency`, `--runs`, and
  `--format md`; its `SRV_ENV` includes sequence slots and spec stats
  (`tools/bench/ours_benchy.sh:2-12`). The served-speed handoff likewise pins llama-benchy 0.4.0 and the
  model tokenizer ( `HANDOFF-2026-10-03-SERVED.md:1-7`).
- `Makefile` defines host gates for formats, attention, safetensors, MoE, DeltaNet, ops, tokenizer,
  GPTQ, K2, DFlash, Agnes, Gemma-4, Qwen4-Exp, plus separate generation/HTTP and native-model targets
  (`Makefile:13-120`).
- `build_b70.sh` builds the engine gates from synthetic checkpoints for architecture/format matrix,
  PP/TP, speculation, Gemma-4 prefill, Qwen4-Exp, NVFP4, prefix reuse, batching, scheduler, and PP
  server; most are `spir64` correctness builds, and `GRIMOIRE_SKIP_GATES` suppresses their build
  (`build_b70.sh:264-345`).
- `preflight_b70.sh` runs these gates, using a two-device container for tests that fork ranks, then runs
  a real-model generation whose text is printed for human inspection (`tools/preflight_b70.sh:153-215`,
  `tools/preflight_b70.sh:234-276`). None of these is a GGUF gate in the inspected lists
  (`Makefile:105-120`, `build_b70.sh:321-345`).

## 2. Load pipeline, end to end

1. CLI `-m` or server `--model` supplies a **directory**, not a checkpoint file
   (`tools/grimoire_main.cpp:46-49`, `tools/grimoire_server.cpp:86-89`). Both load the tokenizer from
   that directory before the engine (`src/grimoire.cpp:16062-16069`,
   `tools/grimoire_server.cpp:143-155`).
2. `grimoire_load()` packages the selected projection `Fmt` and context length into `UploadOptions` and
   calls `Grimoire::build(dir,opt,err)` (`src/grimoire.cpp:16132-16139`).
3. `Grimoire::build()` calls `ck.load(dir,err)`, copies `ck.cfg`, asks `unsupported_reason()` whether
   the engine can execute that parsed architecture, and only then uploads weights
   (`src/grimoire.cpp:4086-4094`, `src/grimoire.cpp:4147-4156`).
4. `Qwen35Model::load()` reads `config.json` itself. `find_scalar()` chooses the shallowest key match;
   for multimodal wrappers it extracts `text_config` and parses text keys there first
   (`src/qwen35_loader.cpp:55-95`, `src/qwen35_loader.cpp:156-222`). `HFModel::discover()` also parses
   flat scalar config values, but `Qwen35Model::load()` uses its own richer parser for this path
   (`src/safetensors.cpp:400-438`, `src/qwen35_loader.cpp:156-222`).
5. The model loader checks for `model-v3.b70`, then `model-v2.b70`; otherwise `HFModel::discover()`
   enumerates **all** `.safetensors` files in lexical order. Despite the header comment mentioning an
   index, this implementation does **not** parse `model.safetensors.index.json`
   (`src/qwen35_loader.cpp:747-782`, `src/safetensors.cpp:440-455`,
   `include/b70/safetensors.hpp:109-119`).
6. Each safetensors shard is opened read-only and memory mapped. The header parser records dtype, shape
   and data offsets, checks byte ranges and shape/byte count, and rejects duplicate names **within a
   shard** (`src/safetensors.cpp:252-360`). `Qwen35Model::load()` inserts each shard's tensor records
   into `index` by name; a duplicate name across shards would overwrite the earlier map entry in this
   code (`src/qwen35_loader.cpp:767-781`).
7. `prefix` becomes `model.language_model.` if that embedding exists, else `model.`. The loader resolves
   embeddings, final norm, head, each layer and experts by HF names, and rejects a set of missing
   required tensors (`src/qwen35_loader.cpp:785-883`, `src/qwen35_loader.cpp:1205-1285`).
8. `linear_ref(base)` handles `.weight` plus scale metadata, NVFP4 or compressed INT4
   `.weight_packed` pairs, and GPTQ `.qweight/.qzeros/.scales`. Its generic packed branch does **not**
   resolve ordinary U8 MXFP4 `.weight_packed` pairs; the separate `packed()` resolver is used for Muse
   FFN weights. These paths rewrite physical packed shapes to logical `[N,K]` and keep companion
   descriptors in `TensorRef` (`src/qwen35_loader.cpp:831-855`, `src/qwen35_loader.cpp:1178-1197`,
   `src/qwen35_loader.cpp:1288-1404`, `include/b70/qwen35.hpp:288-327`).
9. After resolving and validating the model, `build()` calls `ck.unmap_all()` before device allocations.
   Later safetensors tensor reads use `pread()` through the retained shard descriptors
   (`src/grimoire.cpp:4219-4225`, `include/b70/qwen35.hpp:442-467`, `src/safetensors.cpp:368-396`).
10. Embeddings and norms are uploaded separately. Dense projections use `quantize_upload_t()`, merged
   gate/up and `la_ab` use `concat_upload_t()`, and experts are staged into contiguous expert-major
   device blocks before constructing `MoeLayer` `QuantWeight` views (`src/grimoire.cpp:4228-4296`,
   `src/grimoire.cpp:4656-4668`, `src/grimoire.cpp:4920-5101`).
11. For generic matrices `read_matrix_f32()` converts the source to f32; `quantize()` packs the
   requested destination; `upload_packed()` copies payload, scale and zero buffers to device memory
   (`src/grimoire.cpp:979-1096`, `src/grimoire.cpp:1389-1421`, `src/grimoire.cpp:1424-1489`,
   `src/quantize.cpp:80-104`).
12. There are direct paths for compatible BF16, row-scaled FP8, compressed-tensors MXFP4, GPTQ/INT4,
   compressed INT4, and native `.b70` packed layouts. Thus `--proj` is a requested destination for
   conversions, **not** a guarantee that every saved matrix is re-quantized to that format
   (`src/grimoire.cpp:1320-1376`, `src/grimoire.cpp:1391-1421`, `src/grimoire.cpp:1704-1754`).
13. `build()` stores uploaded projections in `Grimoire::L` (`std::vector<LayerDev>`),
   embeddings/norms/head in engine fields, and runtime state/scratch beside them. It ends with `return
   true`; generation then calls prefill/forward on that resident engine (`src/grimoire.cpp:2215-2252`,
   `src/grimoire.cpp:2380`, `src/grimoire.cpp:6751`, `src/grimoire.cpp:16062-16097`).

**Boundary:** `Qwen35Model`/`TensorRef` are host checkpoint descriptors. The loader-to-execution
boundary is the successful return of `Grimoire::build()` with populated `Grimoire::L`; `LayerDev`
contains `DevQuant`, whose `QuantWeight` is passed to dense kernels, and `MoeLayer`, whose `QuantWeight`
views are passed to MoE kernels (`include/b70/qwen35.hpp:288-327`, `src/grimoire.cpp:888-919`,
`src/grimoire.cpp:2215-2252`, `src/grimoire.cpp:6751`, `src/grimoire.cpp:3001-3024`,
`src/grimoire.cpp:15503-15527`).

## 3. Tensor and weight descriptors

- `STTensor` records source name, safetensors dtype, physical shape and byte offsets. `SafeTensors` owns
  an fd/mapping and exposes `read_raw()`/`read_f32()` (`include/b70/safetensors.hpp:42-56`,
  `include/b70/safetensors.hpp:64-106`).
- `TensorRef` locates a shard or native record, carries the source `STTensor`, and marks GPTQ,
  compressed INT4, row-scaled/block-scaled FP8 and two NVFP4 scale conventions; it also retains
  auxiliary scale/zero/global-scale descriptors and a GPTQ group size
  (`include/b70/qwen35.hpp:288-327`).
- `QuantWeight` is the kernel-facing view: `fmt`, output rows `N`, input columns `K`, `payload`,
  `scales`, `zeros`, `row_bytes`, `row_scales`. It does not own the allocations. `DevQuant` owns the
  device pointers and additionally tracks TP row slices, oneDNN scale copies, FP16 copy, and a symmetric
  INT4 companion (`include/b70/weights.hpp:46-55`, `src/grimoire.cpp:888-919`).
- `Fmt` has exactly seven values: BF16, FP8 E4M3, FP8 E5M2, INT8, INT4, MXFP8 and MXFP4. No GGUF block
  format is represented (`include/b70/formats.hpp:228-284`).
- All matrices are logical row-major `[N_out][K_in]`. BF16 has 16-bit payload and no scale; FP8 and INT8
  have an f32 scale per output row; INT4 has BF16 group scales and U8 zeros; MXFP8/MXFP4 have E8M0 byte
  scales per 32 K elements (`include/b70/weights.hpp:25-36`, `include/b70/weights.hpp:58-73`).
- INT4 group width is inferred from `K` and `row_scales` by `int4_gshift()`; the decoder uses a zero
  value of `0xff` to mean signed two's-complement s4 instead of affine U4
  (`include/b70/weights.hpp:77-106`, `include/b70/formats.hpp:319-326`).
- `PackedWeight` owns host payload/scales/zeros and creates a `QuantWeight` view; `upload_packed()`
  supplies device ownership in `DevQuant` (`include/b70/weights.hpp:120-142`,
  `src/grimoire.cpp:1424-1489`).

| Source representation | Current load treatment | Evidence |
|---|---|---|
| Plain BF16 | Raw device copy when BF16 is requested and no source scaling applies; otherwise f32 conversion and destination packing. | `src/grimoire.cpp:1334-1344`, `src/grimoire.cpp:1411-1421` |
| Other ordinary safetensors scalar dtypes | `SafeTensors::read_f32()` converts recognized F64/F32/F16, FP8, integral and bool elements to f32 for the generic matrix path; recognition of a scalar dtype does not itself establish a valid model-weight layout. | `include/b70/safetensors.hpp:42-43`, `src/safetensors.cpp:66-84`, `src/safetensors.cpp:91-141`, `src/grimoire.cpp:1059-1066`, `src/grimoire.cpp:1411-1421` |
| Plain FP8 with `.weight_scale` | `row_scaled` metadata; direct payload and f32 per-row scales when destination FP8 matches; otherwise multiply scales in f32 before repack. | `src/qwen35_loader.cpp:1342-1351`, `src/grimoire.cpp:1320-1333`, `src/grimoire.cpp:1057-1066` |
| Block FP8 with `.weight_scale_inv` | Validate 128×128 scale grid; multiply f32 values by the tile scale before destination packing. | `src/qwen35_loader.cpp:1352-1363`, `src/grimoire.cpp:1067-1078` |
| compressed-tensors MXFP4 `.weight_packed/.weight_scale` | Muse FFN `packed()` resolves this pair; `linear_ref()` does not recognize an ordinary U8 MXFP4 pair. Once resolved, any quantized requested destination keeps the saved E2M1 nibbles and E8M0 group-32 scales as MXFP4; BF16 request uses f32 decode. | `src/qwen35_loader.cpp:831-855`, `src/qwen35_loader.cpp:1178-1197`, `src/qwen35_loader.cpp:1365-1388`, `src/grimoire.cpp:982-1000`, `src/grimoire.cpp:1346-1376` |
| compressed-tensors INT4 `.weight_packed/.weight_scale` | Logical shape expansion; an INT4 request with accepted group size uses the direct reader; other non-MXFP4 requests use f32 decode. An MXFP4 request reaches an earlier name-based packed copy, without checking `compressed_int4`, which is a source-format collision. | `src/qwen35_loader.cpp:1368-1387`, `src/grimoire.cpp:973-979`, `src/grimoire.cpp:1002-1022`, `src/grimoire.cpp:1358-1376`, `src/grimoire.cpp:1400-1421` |
| GPTQ / AutoRound `.qweight/.qzeros/.scales` | Resolve logical `[out,in]`; repack to runtime INT4 when requested/accepted, otherwise dequantize to f32 then pack. | `src/qwen35_loader.cpp:1389-1404`, `src/grimoire.cpp:1082-1096`, `src/grimoire.cpp:1406-1421` |
| compressed-tensors NVFP4 | Detect `.weight_global_scale`; decode E2M1 with E4M3 per-16 scales and **divide** by its global scale, then destination-pack. | `src/qwen35_loader.cpp:1299-1317`, `src/grimoire.cpp:1023-1055`, `include/b70/nvfp4.hpp:110-127` |
| modelopt NVFP4 | Detect U8 `.weight`, E4M3 `.weight_scale`, scalar `.weight_scale_2`; same local decode but **multiply** by its global scale. | `src/qwen35_loader.cpp:1319-1341`, `include/b70/nvfp4.hpp:106-117` |
| Native `.b70` | `native_view()` produces a `QuantWeight`; saved packed precision is copied as authoritative, subject to critical-BF16 policy. | `src/qwen35_loader.cpp:1406-1417`, `src/grimoire.cpp:1391-1400` |

`keep_qwen_bf16()` protects embeddings, default head, MTP tensors, DeltaNet a/b projections,
router/shared gate and K2 value router from generic packing; `GRIMOIRE_QUANT_LM_HEAD` changes the head
exception (`src/qwen35_loader.cpp:124-154`). `--proj` may also fall back to BF16 when K is not a whole
destination group (`src/grimoire.cpp:1379-1387`).

**W4A8 pointer hazard:** conversion frees an MXFP4 payload and its scales, sets `dq.w.payload/scales` to
null, and keeps `dq.w.fmt == MXFP4`; the symmetric INT4 bytes live in `dq.i4/i4s`
(`src/grimoire.cpp:5144-5175`). `gemv_any()` routes through `has_i4()` first
(`src/grimoire.cpp:3001-3024`). Any new path must test the pointer/companion, never use `fmt == MXFP4`
alone as permission to dereference payload; this is CLAUDE.md rule 1 and is visible in the code
(`CLAUDE.md:284-292`, `src/grimoire.cpp:5149-5175`). W4A8 conversion also skips `N % 256 != 0` to avoid
an unclamped tile read (`src/grimoire.cpp:5150-5159`).

## 4. Architectures and Ornith's hybrid MoE path

- One `Qwen35Config`/`Qwen35Model` represents dense and MoE Qwen3.5-style hybrids, with per-layer
  `LINEAR_ATTN` or `FULL_ATTN`; `cfg.is_moe()` is `n_experts > 0` (`include/b70/qwen35.hpp:20-38`,
  `include/b70/qwen35.hpp:63-69`, `include/b70/qwen35.hpp:274-285`).
- The parser recognizes special model families for Muse Glimmer, Agnes, Gemma-4, Qwen4-Exp and
  K2-Horizon; it uses the configured `layer_types` for linear/full selection and rejects an unrecognized
  layer-type spelling. Generic Qwen/Ornith loading is driven by that table and resolved tensor names,
  not by an Ornith-specific class (`src/qwen35_loader.cpp:162-182`, `src/qwen35_loader.cpp:548-553`,
  `src/qwen35_loader.cpp:657-727`, `src/qwen35_loader.cpp:878-883`).
- Core config keys consumed for this path: `hidden_size`, `num_hidden_layers`, `vocab_size`,
  `num_attention_heads`, `num_key_value_heads`, `head_dim`, `rms_norm_eps`, `post_norm_eps`,
  `rope_theta`, `partial_rotary_factor`, `attn_output_gate`, `tie_word_embeddings`
  (`src/qwen35_loader.cpp:223-235`).
- DeltaNet geometry keys: `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`,
  `linear_value_head_dim`, `linear_conv_kernel_dim` (`src/qwen35_loader.cpp:373-377`). MoE keys:
  `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size`; a
  dense path uses `intermediate_size` (`src/qwen35_loader.cpp:379-383`).
- `layer_types` entries containing known linear/DeltaNet spellings become `LINEAR_ATTN`; known full
  spellings become `FULL_ATTN`. Unknown values are refused rather than treated as full attention
  (`src/qwen35_loader.cpp:657-727`). The `qwen35.hpp` comments identify the 35B-A3B example as 40
  layers, 30 linear and 10 full (`include/b70/qwen35.hpp:20-35`); runtime dimensions still come from
  `config.json` (`src/qwen35_loader.cpp:223-235`, `src/qwen35_loader.cpp:657-732`).
- The text prefix is `model.language_model.` when that embedding exists, otherwise `model.`. The head is
  looked up as `lm_head` or prefixed `lm_head`; embeddings and final norm are
  `${prefix}embed_tokens.weight` and `${prefix}norm.weight` (`src/qwen35_loader.cpp:785-789`,
  `src/qwen35_loader.cpp:858-876`).
- Every layer starts at `${prefix}layers.<L>.`, with `input_layernorm.weight` and
  `post_attention_layernorm.weight` (`src/qwen35_loader.cpp:878-886`).
- Linear-attention tensors are `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`,
  `conv1d.weight`, `A_log`, `dt_bias`, `norm.weight`, and `out_proj`; the projection resolver adds
  `.weight` or recognized packed suffixes. Their destination slots are `la_qkv`, `la_z`, `la_ab`,
  `la_out`, plus BF16 conv/A-log/dt/norm (`src/qwen35_loader.cpp:949-964`, `src/grimoire.cpp:2218-2223`,
  `src/grimoire.cpp:4656-4668`).
- Full-attention tensors are `self_attn.{q,k,v,o}_proj`, `q_norm.weight`, and `k_norm.weight`; the
  device layer holds corresponding projections, norms and KV-cache views
  (`src/qwen35_loader.cpp:965-971`, `src/grimoire.cpp:2224-2226`, `src/grimoire.cpp:2315-2323`).
- A MoE layer resolves `mlp.gate` router, `mlp.shared_expert.{gate,up,down}_proj`, optional
  `mlp.shared_expert_gate`, and per-expert `mlp.experts.<e>.{gate,up,down}_proj` tensors
  (`src/qwen35_loader.cpp:1115-1161`).
- For BF16 Ornith-style fused experts, the loader can slice `experts.gate_up_proj` and
  `experts.down_proj` logically into per-expert gate/up/down references; the code gives their physical
  shapes as `[E,2I,H]` and `[E,H,I]` (`src/qwen35_loader.cpp:1135-1177`).
- The uploader lays out each expert as gate rows then up rows, followed by down rows in a separate
  block. `MoeLayer::gate_up` is `[E*2I,H]`, `MoeLayer::down` is `[E*H,I]`, with device buffers for
  payload, scale and INT4 zero points (`src/grimoire.cpp:4941-4965`, `src/grimoire.cpp:5032-5101`,
  `include/b70/moe.hpp:47-55`).
- Shared gate/up is concatenated into `d.sh_gu`; shared down and router are separate `DevQuant` fields.
  The router is requested as BF16 in the MoE upload path (`src/grimoire.cpp:4920-4927`,
  `src/grimoire.cpp:5104-5115`).
- `tie_word_embeddings` is parsed. If no explicit `lm_head` exists, `build()` aliases the embedding
  device buffer as a BF16 head and tracks ownership to avoid freeing it twice; missing head without
  tying is an error (`src/qwen35_loader.cpp:233-235`, `src/grimoire.cpp:4328-4358`).
- MTP is optional via `GRIMOIRE_MTP`. When enabled, the loader asks for `mtp.fc`, pre-FC norms, a
  one-layer attention/MLP head and possibly MoE head experts; projection references go through
  `linear_ref()` so FP8 scales are retained. Target `--proj` does not dictate MTP head packing
  (`src/grimoire.cpp:5255-5304`, `src/grimoire.cpp:5334-5426`).
- `Grimoire::unsupported_reason()` is the post-parse architectural gate before upload; it contains
  explicit refusals for unsupported Qwen4-Exp configurations, head geometry, and GeGLU outside Gemma-4
  (`src/grimoire.cpp:3943-4050`, `src/grimoire.cpp:4074-4084`, `src/grimoire.cpp:4147-4156`).

## 5. Kernel inventory relevant to GGUF

Phase 3 step 1 implements the native Q4_K/Q6_K path described in
[NATIVE_QK.md](NATIVE_QK.md). The inventory below records the original
Strategy A baseline and its kernel insertion points.

The existing kernels take runtime `QuantWeight`/`MoeLayer` views, not safetensors records
(`include/b70/weights.hpp:46-55`, `include/b70/moe.hpp:85-102`). Thus a GGUF reader that produces a
supported runtime view can reuse the paths below **after its tensor has been decoded/repacked or
otherwise made layout-compatible**. At the baseline of this audit, no dispatch switch listed
Q4_K, Q5_K, Q6_K, Q8_0 or IQ. The current tree adds native Q4_K/Q6_K dispatch.

| Path | Accepted runtime view / condition | Selection and reuse point |
|---|---|---|
| General prefill GEMM | `launch_gemm_xmx()` switches all seven `Fmt` values. INT8/INT4 dequantize into BF16 staging for DPAS when ordinary BF16 activations are used. | `src/gemm_xmx.cpp:470-486`; reusable with any correctly populated supported `QuantWeight`. |
| Fast large-M prefill | `gemm_fast_supported()` requires M≥32, a payload, N divisible by 16, K divisible by its tile width, and bounded BF16 scratch. Dequant-to-VNNI supports all seven formats; MXFP4 has a streaming optimization for K%128=0, N%256=0. | `src/gemm_fast.cpp:399-433`, `src/gemm_fast.cpp:2954-3003`; `GRIMOIRE_NO_FAST_GEMM` disables it. |
| Fused dense SwiGLU | Requires fast GEMM plus MXFP4 or accepted INT4 stream, K%128=0, tile-aligned N and gate/up width. | `src/gemm_fast.cpp:3021-3030`; `GRIMOIRE_NO_FUSED_SWIGLU` disables it. |
| Grouped MoE prefill | Native grouped path is MXFP4-only and requires payload+scales, expert width divisible by its column tile, K tile alignment and whole expert grouping. ESIMD adds row-stride and SwiGLU/down width constraints. | `src/gemm_fast.cpp:1863-1869`, `src/gemm_fast.cpp:1891-1915`; `GRIMOIRE_MOE_ESIMD` selects ESIMD, `GRIMOIRE_MOE_GROUPED_OFF` bypasses it at call sites (`src/grimoire.cpp:15503-15527`). |
| Grouped fallback / bridge | If MXFP4 grouped conditions fail, prefill can slice each expert and call `launch_gemm_xmx()`; optional signed-INT4 grouped bridge is selected when loaded and the weights were converted to signed s4. | `src/grimoire.cpp:15538-15595`, `src/grimoire.cpp:5066-5082`. |
| Decode GEMV | Generic `launch_gemv()` dispatches all seven formats; ESIMD fast branches have separate guards for MXFP4, INT4 and FP8 E4M3. | `src/gemv_decode.cpp:1133-1181`, `src/gemv_decode.cpp:1697-1707`; `GRIMOIRE_ESGEMV*` flags control fast branches. |
| Decode MoE | `launch_moe_gate_up/down` and batch variants switch all seven formats. A separate ESIMD MXFP4 path requires valid payload/scales and aligned row geometry. | `src/moe_kernels.cpp:937-969`, `src/moe_kernels.cpp:1048-1089`; `GRIMOIRE_MOE_ESIMD_DECODE` controls that branch. |
| Small-M dense DPAS | MXFP4 and INT4 accept M=1..16 with K%512=0, N%16=0 and aligned payload/strides; BF16 accepts M=1..16 with K%128=0; FP8 E4M3 accepts M=1..8 with K%512=0. | `src/gemm_fast.cpp:3600-3635`, `src/gemv_decode.cpp:1385-1408`; `src/grimoire.cpp:13973-14016` selects them for batches. |
| Small-M grouped MoE | MXFP4 only, M=1..16, K%512=0, expert dimensions and buffers aligned; `GRIMOIRE_MOE_AW` selects its gate/up variant. | `src/gemm_fast.cpp:3650-3674`. |
| W4A8 | Converts eligible MXFP4 weights into symmetric INT4+f32 group scales; `N%256==0` is mandatory for the tile. Converted `DevQuant` has null MXFP4 payload. | `src/grimoire.cpp:5139-5188`; `GRIMOIRE_W4A8` gate at `src/grimoire.cpp:176-183`. |
| Optional Xe2/CUTLASS bridge | The grouped bridge exports MXFP4/BF16 dense and routed GEMM plus signed-INT4 W4A8 tiles. `mm()` selects the dense MXFP4 symbols for matching views; grouped MoE selects the MXFP4 symbol at M≥32 when matrix support and other guards pass. The library path can be set with `GRIMOIRE_XE2_GROUPED_BRIDGE`; grouped MXFP4 tile choice depends on routed rows per expert. | `src/xe2_grouped_bridge.cpp:117-196`, `src/xe2_grouped_bridge.cpp:235-322`, `src/grimoire.cpp:133-166`, `src/grimoire.cpp:13591-13611`, `src/grimoire.cpp:14222-14239`, `src/grimoire.cpp:15409-15410`. |
| Optional oneDNN / Xe2 bridges | oneDNN MXFP4 or symmetric INT4 GEMV branches are opt-in; the grouped signed-INT4 bridge reads the expert blocks. Pure launchers hide these optional plugins. | `src/grimoire.cpp:3017-3024`, `src/grimoire.cpp:3060-3115`, `src/grimoire.cpp:15538-15548`, `tools/b70run.sh:65-79`. |
| Attention / DeltaNet | Operate on projected activations and KV/recurrent state, not on original checkpoint encoding. Fast flash prefill accepts head dimension 128 or 256; fallback handles the other supported geometry. | `src/gemm_fast.cpp:1074-1085`, `src/prefill.cpp:1800-1814`; reusable after correct projection weights are loaded. |

The native-GGUF-kernel insertion points would be `launch_gemv()`/`gemv_any()` for decode,
`launch_gemm_xmx()` or the `mm()` selection in prefill for dense projections, and the `MoeLayer`
gate/up/down dispatch for routed experts (`src/grimoire.cpp:3001-3024`, `src/grimoire.cpp:13973-14016`,
`src/grimoire.cpp:15503-15527`). That is an insertion analysis, not an API already present. A native
Q4_K/Q8_0 path would need a new descriptor/dispatch representation because `Fmt` and
`QuantWeight::scale_for()` only encode the seven existing layouts (`include/b70/formats.hpp:231-284`,
`include/b70/weights.hpp:46-73`).

## 6. Tokenizer

The inventory below describes the original sidecar-only implementation. The
GGUF adapter now uses `tokenizer.ggml.*` when the sidecar is absent, and the
same chat-template family detection reads `tokenizer.chat_template`. It accepts
`gpt2` with `pre=qwen35` and `llama` SentencePiece BPE; other model/pre names
fail explicitly. A directory with one text GGUF or a complete split set can
load without sidecars, as can an explicit `.gguf` path. When both sources exist,
the sidecar wins and the loader checks vocabulary and ordered merge ranks.

- `Tokenizer::load(dir)` requires `dir/tokenizer.json`; missing file or missing `vocab` is an error
  (`src/tokenizer.cpp:143-156`). It parses vocabulary IDs, ordered BPE merges, and `added_tokens`,
  extending its ID table for specials outside the base vocabulary (`src/tokenizer.cpp:159-219`,
  `src/tokenizer.cpp:238-291`).
- It identifies EOS/BOS from known added-token spellings, including ChatML, IFM, and Harmony
  alternatives (`src/tokenizer.cpp:294-310`). It performs byte-level BPE with pre-tokenization and
  bypasses BPE for special-token text (`src/tokenizer.cpp:451-555`, `src/tokenizer.cpp:558-605`).
- It reads `chat_template.jinja`, falling back to `tokenizer_config.json`, to **detect** a supported
  XML-tool template family. Rendering is implemented in C++; it does not execute arbitrary Jinja
  (`src/tokenizer.cpp:221-236`, `src/tokenizer.cpp:637-719`, `src/tokenizer.cpp:749-775`).
- The server/CLI call this tokenizer before loading the model, so a GGUF-only file with embedded
  `tokenizer.ggml.*` metadata would currently fail before reaching any weight reader
  (`tools/grimoire_server.cpp:143-155`, `src/grimoire.cpp:16062-16069`).
- **Proposal implication:** either require a sidecar `tokenizer.json` (and compatible template files
  where needed), or add a separate GGUF-tokenizer adapter that fills the same
  vocabulary/merge/special/template behavior. The current implementation cannot be pointed at GGUF
  metadata (`src/tokenizer.cpp:143-156`, `src/tokenizer.cpp:221-310`). Exact GGUF metadata compatibility
  is unverified in this tree.

## 7. PROPOSAL — minimal-disruption GGUF integration

This section specifies candidate changes, not existing APIs or verified GGUF file contents. The target
interfaces and HF names on the right are verified in this tree; any GGUF name on the left must be
checked against an actual Ornith GGUF tensor directory before code assumes it
(`src/qwen35_loader.cpp:785-883`, `src/qwen35_loader.cpp:949-1177`).

1. Add a GGUF file/header/block reader beside `SafeTensors`, at the source-discovery layer, with strict
   bounds, type and shape validation analogous to the current shard reader
   (`src/safetensors.cpp:252-360`, `src/safetensors.cpp:400-455`). Treat model path as a GGUF file or a
   directory containing one explicitly; current entry points accept a directory only
   (`tools/grimoire_main.cpp:46-49`, `tools/grimoire_server.cpp:86-89`).
2. Provide an internal source abstraction or extend `TensorRef`/`Qwen35Model` so tensor lookup, raw
   reads, and matrix-to-f32 reads work for both safetensors and GGUF. `TensorSource` is a **suggested
   name**, not a current class. Avoid pretending GGUF has an `STTensor` dtype: its blocks need their own
   descriptor and logical shape (`include/b70/qwen35.hpp:288-327`, `include/b70/qwen35.hpp:408-467`,
   `src/grimoire.cpp:979-1096`).
3. Translate GGUF metadata into the exact `Qwen35Config` values used by `build()`; validate
   `layer_types`, DeltaNet geometry, MoE count/top-k and tie semantics before upload. The current parser
   requires `config.json` and does not inspect GGUF metadata (`src/qwen35_loader.cpp:156-235`,
   `src/qwen35_loader.cpp:373-383`, `src/qwen35_loader.cpp:657-727`). A sidecar `config.json` is the
   smaller first step.
4. Translate external tensor names to the existing HF names **before** `Qwen35Model` resolves layers.
   Keep the mapper explicit and versioned by observed GGUF naming; reject unrecognized, duplicate or
   ambiguous names. Current resolution relies on exact strings and checks only a subset of required
   fields (`src/qwen35_loader.cpp:949-971`, `src/qwen35_loader.cpp:1115-1177`,
   `src/qwen35_loader.cpp:1205-1285`).

Candidate name checklist, **unverified until an actual Ornith GGUF is inspected**:

| GGUF-side name/pattern to investigate | Required HF-side target or transformation | Verified target code |
|---|---|---|
| `token_embd`, output/output norm | `${prefix}embed_tokens.weight`, `lm_head.weight` or tied alias, `${prefix}norm.weight` | `src/qwen35_loader.cpp:858-876`, `src/grimoire.cpp:4328-4358` |
| `blk.N.attn_qkv` | May need splitting or reordering into `self_attn.q_proj/k_proj/v_proj`; actual shape/order must be read from the GGUF. | `src/qwen35_loader.cpp:965-971` |
| `blk.N.attn_output`, `attn_q_norm`, `attn_k_norm` | `self_attn.o_proj`, `q_norm.weight`, `k_norm.weight` after shape verification. | `src/qwen35_loader.cpp:965-971` |
| `blk.N.ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` | Expert-major `mlp.experts.<e>.{gate,up,down}_proj` refs or a validated fused slice; confirm expert order, transpose and gate/up order. | `src/qwen35_loader.cpp:1132-1177`, `src/grimoire.cpp:5032-5101` |
| `blk.N.ffn_gate_inp`, shared-FFN names | `mlp.gate`, `mlp.shared_expert.*`, optional shared gate; check router weight semantics. | `src/qwen35_loader.cpp:1115-1130` |
| `blk.N.ssm_*` | Map each observed tensor individually to `linear_attn.in_proj_qkv/z/a/b`, `conv1d.weight`, `A_log`, `dt_bias`, `norm.weight`, `out_proj`; do not infer the QKV concatenation order from the name. | `src/qwen35_loader.cpp:949-964`, `src/grimoire.cpp:4656-4668` |
| MTP names, if present | `mtp.fc`, norms, attention, MLP/router/experts as actually required by the optional head. | `src/grimoire.cpp:5255-5296` |

5. **Strategy A — load-time f32 decode and destination pack.** Implement a checked GGUF block decoder
   returning logical row-major `[N,K]` f32 for every supported source type. Then feed the existing
   `quantize()`/upload path, including per-input decoding before gate/up concatenation. This follows the
   NVFP4 precedent (`src/grimoire.cpp:1023-1055`, `src/grimoire.cpp:1411-1421`,
   `src/grimoire.cpp:1746-1754`). It permits reuse of all seven-format GEMV/GEMM/MoE paths above, but a
   quantized GGUF block decoded and repacked to MXFP4/INT4 loses precision in general; tests must
   compare against an independent source decode, not only against the same repack
   (`include/b70/nvfp4.hpp:35-50`, `src/quantize.cpp:80-104`).
6. **Strategy B — native block execution or proven lossless repack.** Add a GGUF-specific runtime
   format/view plus decode GEMV, prefill GEMM and expert path, or prove a particular GGUF block can be
   rearranged exactly into an existing `QuantWeight` encoding. Existing `Fmt` has no
   Q4_K/Q5_K/Q6_K/Q8_0/IQ entry, and the MXFP4 direct-copy shortcut requires **its** E2M1+E8M0/32
   layout, not merely four bits per value (`include/b70/formats.hpp:231-284`,
   `src/grimoire.cpp:1346-1376`, `include/b70/weights.hpp:25-36`). Exact lossless compatibility for the
   requested GGUF types is **not established in this repository**.
7. Preserve `keep_qwen_bf16()` policy, small-N W4A8 guards, MTP and tied-head behavior regardless of
   source format. Map a GGUF source into `DevQuant`/`MoeLayer` so the existing forward paths remain
   source-agnostic; add a new runtime dispatch only if Strategy B is chosen
   (`src/qwen35_loader.cpp:124-154`, `src/grimoire.cpp:5139-5188`, `src/grimoire.cpp:5255-5304`,
   `src/grimoire.cpp:4328-4358`).
8. Add a host format gate for GGUF bounds/type/block decoding and a synthetic hybrid-MoE fixture with
   known tensor values, then a B70 end-to-end text/parity gate. Existing host, device, architecture, and
   NVFP4 gates show where to integrate such checks (`Makefile:49-60`, `build_b70.sh:290-345`,
   `tools/preflight_b70.sh:234-276`). This is a proposed test plan, not a claim that such gates exist.

### Risks to resolve before choosing the strategy

- GGUF expert, QKV and DeltaNet tensor shapes/order are not described by the current source reader; a
  name-only mapping can produce plausible but wrong output. The current loader derives shapes from
  source records and refuses some mismatches (`src/grimoire.cpp:927-934`,
  `src/qwen35_loader.cpp:949-971`, `src/qwen35_loader.cpp:1205-1285`).
- Fused gate/up tensors may have different row order or per-partition scales; `concat_upload_t()` builds
  gate then up rows and NVFP4 decoding preserves separate partition scales before merging
  (`src/grimoire.cpp:1619-1635`, `src/grimoire.cpp:1667-1702`, `src/grimoire.cpp:1746-1754`).
- The current `quantize_upload_t()` MXFP4 shortcut checks for `weight_packed` in the name before its
  compressed-INT4 branch and does not exclude `r.compressed_int4` when `fmt==MXFP4`. A source adapter
  must carry explicit block type through this decision; reusing the shortcut for GGUF by suffix would
  copy the wrong payload/scale layout (`src/grimoire.cpp:973-979`, `src/grimoire.cpp:1358-1376`,
  `src/grimoire.cpp:1400-1405`).
- Generic `quantize()` needs finite f32 values and whole K groups for INT4/MX formats. Destination
  fallback and `keep_qwen_bf16()` can make actual VRAM footprint differ from a requested `--proj` label
  (`src/quantize.cpp:80-103`, `src/grimoire.cpp:1379-1387`, `src/qwen35_loader.cpp:124-154`).
- A native format must cover **both** dense and routed experts in decode and prefill, including small-M
  and head paths; adding one `Fmt` switch arm alone cannot reach all kernel families
  (`src/gemm_xmx.cpp:470-486`, `src/gemv_decode.cpp:1697-1707`, `src/moe_kernels.cpp:963-1007`,
  `src/grimoire.cpp:13973-14016`).
- The tokenizer is an independent load prerequisite. GGUF weights with no sidecar `tokenizer.json` fail
  before `Qwen35Model::load()` is reached (`tools/grimoire_server.cpp:143-155`,
  `src/tokenizer.cpp:143-156`).

## 8. Open questions and B70-only verification

- Inspect the **actual** first target GGUF: metadata keys, architecture identifier, tensor names,
  shapes, quantized block types, expert ordering, DeltaNet/conv/A-log/dt representation, MTP presence,
  embedding/head tie and tokenizer metadata. No GGUF parser or fixture in the inspected
  format/loader/test paths establishes these values (`include/b70/formats.hpp:231-233`,
  `src/qwen35_loader.cpp:747-782`, `Makefile:105-120`).
- Determine whether a sidecar `config.json` and `tokenizer.json` exist for the intended GGUF
  distribution. The current loader and tokenizer both require those filenames in a directory
  (`src/qwen35_loader.cpp:156-160`, `src/tokenizer.cpp:143-147`).
- Verify an independent GGUF block-decoder reference and per-format numeric parity for Q4_K_M, Q8_0,
  Q5_K, Q6_K and IQ variants before measuring a requantized engine. The only checked-in quantizer
  enumerates seven **destination** formats (`src/quantize.cpp:80-86`,
  `include/b70/formats.hpp:231-233`).
- Measure Strategy A load time, peak host RAM, VRAM, and end-task quality on the B70. The existing
  loader materializes an f32 matrix for generic conversion and makes per-layer expert staging buffers,
  so those are concrete memory-pressure points (`src/grimoire.cpp:1411-1421`,
  `src/grimoire.cpp:4973-4981`).
- If Strategy B is attempted, measure native GGUF block GEMV, M=2..16, M≥32, and grouped expert paths
  **on B70** against the existing decoded/repacked baseline; the source only proves dispatch and tile
  guards, not throughput of a new layout (`src/gemv_decode.cpp:1697-1707`,
  `src/gemm_fast.cpp:2954-3003`, `src/gemm_fast.cpp:3650-3674`).
- Validate output tokens and readable text after long prompt, batched serving, recurrent-state reuse and
  MTP (if available). `preflight_b70.sh` prints real generated text after synthetic gates; the hybrid
  path maintains DeltaNet state and KV cache per slot (`tools/preflight_b70.sh:254-276`,
  `src/grimoire.cpp:2306-2323`).
- Verify behavior under `GRIMOIRE_W4A8=1` with a GGUF-sourced converted weight: the surviving `fmt` tag
  and null payload make this a runtime pointer-routing check, not a source-format label check
  (`src/grimoire.cpp:5149-5175`, `src/grimoire.cpp:3001-3024`).
- Verify any optional bridge build separately and test the pure path independently. `build_b70.sh`
  leaves bridge `.so` files untouched; the pure launcher hides plugin paths (`build_b70.sh:370-378`,
  `tools/build_bridges_b70.sh:100-109`, `tools/b70run.sh:65-79`).
