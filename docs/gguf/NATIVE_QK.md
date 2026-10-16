# Phase 3 step 1: native Q4_K and Q6_K

`--proj native` (CLI or server) or `GRIMOIRE_GGUF_NATIVE=1` enables raw GGUF
Q4_K/Q6_K weights. Raw retention applies to GGUF inputs only. Without it, the
Strategy A decode and requantize path is unchanged. The existing
`keep_qwen_bf16` policy still applies to embeddings, DeltaNet a/b, routers,
and other critical tensors; `output.weight` is explicitly allowed to remain
native Q6_K for logits. The GGUF MTP layer remains excluded by the adapter.
Native mode uses **MXFP4 as the base projection** for any other projection
that the BF16 policy does not preserve. The CLI and server reject an explicit
`--proj` other than `native` or `mxfp4` when the native environment flag is
set; the engine also applies MXFP4 to API callers whose upload options still
default to INT4. The load banner reports this base and the actual native
Q4_K/Q6_K tensor counts and bytes.

## Storage and transforms

Each runtime row is one GGML row with K/256 consecutive super-blocks.
Q4_K uses 144 bytes per block (f16 `d`, f16 `dmin`, packed scale/min bytes,
then 128 nibble bytes); Q6_K uses 210 (128 low-nibble bytes, 64 high-bit
bytes, 16 signed scales, f16 `d`). Scales and minima are inside `payload`.
`QuantWeight.scales` and `.zeros` are null, and `row_scales` is zero. Native
upload refuses a named tensor if K is not divisible by 256.

QKV V rows, z rows, and other T4 row permutations move complete compressed
rows. Expert gate and up rows are concatenated as bytes per expert; down has
its own runtime format so a Q4_K gate/up and a Q6_K down work together.
An expert tensor in another GGUF format uses the MXFP4 base. If gate and up
would require different runtime formats, loading refuses them by tensor name
because their single packed matrix has one `Fmt`.
The stored `ssm_out` column permutation cannot be reversed on compressed
blocks. Immediately before that projection, a gather puts the activation
from grouped V-head order into GGUF tiled V-head order, only when the actual
weight is native. The float and BF16 prefill inputs and float decode input
use the same host-tested index map. Decode writes the gather to `s.bbuf`:
its allocation starts at `Hv*Dv` floats and grows for every layer's `la_z`,
`k_proj`, and `v_proj` output. In the DAG path, the projection waits on the
gather, and the next layer's input waits on that projection through `e_h`
and `e_moe`; a later full-attention `v_proj` cannot overwrite it early.

## Dispatch and fast-path guard inventory

The inventory below covers the projection-bearing hits from
`grep -n "Fmt::MXFP4\|Fmt::INT4\|gemm_fast_supported\|smallm\|esimd\|ESIMD\|W4A8\|has_i4\|bridge\|\.scales\\b" src/*.cpp`.
Attention/router ESIMD and benchmarks in that grep do not read a `QuantWeight`.

| Site | Disposition |
|---|---|
| `src/gemv_decode.cpp:1133-1201,1682-1713` | FP8/INT4/MXFP4 ESIMD attempts require exact formats, intact payload and side scales. Generic and batched GEMV dispatch Q4_K/Q6_K through `GemvStepGGUF`. |
| `src/gemv_decode.cpp:1391-1408` | FP8 small-M requires its exact format; the INT4 small-M caller uses `int4_smallm_ok`. |
| `src/gemm_fast.cpp:127-228,399-440` | VNNI simple and tiled dequant stage Q4_K/Q6_K with `scale=1` and `decode_elem`; no native side-scale access. MXFP4 and INT4 streaming branches require exact formats. |
| `src/gemm_fast.cpp:348-353,1883-1904` | INT4 stream, fused MXFP4, and grouped MXFP4/ESIMD predicates require their own formats, payload and side scales. |
| `src/gemm_fast.cpp:2960-3034` | `gemm_fast_supported` intentionally accepts native weights for generic VNNI staging. `EPI=0/2` uses `dequant_any`; `EPI=1` is entered only after `gemm_fast_swiglu_supported` accepts MXFP4 or INT4. |
| `src/gemm_fast.cpp:3609-3674` | MXFP4, BF16, INT4, and grouped MXFP4 small-M predicates check exact formats and pointers. |
| `src/gemm_xmx.cpp:145-165,473-487` | Float XMX staging starts at `scale=1`, branches on scale-bearing formats, and dispatches Q4_K/Q6_K. The integer pipeline at `:525` dispatches INT8/INT4 only. |
| `src/prefill.cpp:70-140` | Plain SYCL batched GEMM calls `GemvStepGGUF` for Q4_K/Q6_K and uses no side scale. |
| `src/moe_kernels.cpp:90-290,962-1103` | Routed and shared decode use `GemvStepGGUF`; all gate/up, down, batched and MoVA switches include Q4_K/Q6_K. |
| `src/moe_kernels.cpp:630-954` | ESIMD kernels read side scales, but `mx4_rows_ok` at `:941` requires MXFP4, payload and scales. |
| `src/ops.cpp:2110-2170` | MoVA packed fallback gives Q4_K/Q6_K scale 1 and dispatches both formats. Router ESIMD reads logits, not weights. |
| `src/grimoire.cpp:1388-1423,1747-1763` | Raw upload and gate/up concatenation copy complete rows; mismatched gate/up types or K not divisible by 256 are refused by tensor name. |
| `src/grimoire.cpp:4038-4072` | TP row sharding slices by `row_bytes`; null native side scales and zeros remain null. |
| `src/grimoire.cpp:4861,5369-5419,5457` | Fused DeltaNet upload excludes GGUF. W4A8 conversion requires MXFP4, payload and N multiple of 256; native weights are skipped. |
| `src/grimoire.cpp:11220-11264,12263-12293,13093-13113,14241-14293` | Draft, sandwich, Qwen4-Exp and main prefill small-M/bridge selections either check exact formats or call format-aware generic GEMV/XMX. |
| `src/grimoire.cpp:13870-13904,14490-14586,14657-14670` | Native mode disables MXFP4-only parallel modes and norm shortcut. Dense/aux/BF16 bridges check format and payload; grouped FFN bridge checks both gate/up and down. |
| `src/grimoire.cpp:14980-15000,15339,15471-15479` | Native `ssm_out` gathers before float or BF16 GEMM. Attention/router bridge uses MXFP4 predicate; native input takes generic GEMM. |
| `src/grimoire.cpp:15628-15958` | Small-M and grouped MoE predicates require MXFP4 for both weights. Native weights take per-expert `slice_quant_rows` and XMX fallback; gate/up and down use their own `row_bytes`. Shared expert uses generic GEMM. |
| `src/grimoire.cpp:16033-16055` | Native lm_head misses the MXFP4/INT4/FP8-only batch shortcuts and reaches native GEMV for each row. |
| `src/native_model.cpp:124-132` | `slice_quant_rows` advances payload by native `row_bytes` and touches side scales only when non-null. |
| `src/quantize.cpp:83-110` | Generic quantizer explicitly refuses Q4_K/Q6_K; raw GGUF blocks enter through the native upload path. |

The oneDNN W4A8 bridge in `src/grimoire.cpp:3190-3265` requires the
converted INT4 companion (`has_i4`) or exact MXFP4 plus payload and scales.
The remaining oneDNN FP16 paths are Muse-only. `GRIMOIRE_FUSE_DN_PROJECTIONS`
is excluded for GGUF at `src/grimoire.cpp:4861`. The `ssm_out` gather sites
are `src/grimoire.cpp:9167,9891,10341,13260,14981-14998`.

## Validation

`make test` includes `test_gguf_native`: the exact device decoder functions
run on the host and match both the independent ggml-style decoder and the
checked-in GGUF goldens bit for bit, including a two-super-block row;
native-format host GEMV/GEMM match dequantized f32 products; raw QKV V-row
permutation retains whole blocks; row slicing retains null side scales; and
the ssm_out gather equals inverse column permutation.

Only the B70 can verify SYCL compilation and execution here. On the B70:

1. Compile `build_b70.sh` with `icpx`, recording any compiler errors and the
   exact command. Use `tools/build_bridges_b70.sh` if a bridge changes.
2. Load a real GGUF with `--proj native` and with the environment flag. Check
   the banner for MXFP4 base, raw Q4_K/Q6_K counts and named BF16-policy
   tensors. Check named refusals with an unsupported or malformed tensor.
3. Generate from two different prompts and inspect both continuations; also
   compare decode and prefill logits/token sequences against Strategy A and
   llama.cpp. Exercise short and long prefill, full attention, DeltaNet,
   routed and shared MoE, and a Q6_K lm_head.
4. Exercise `GRIMOIRE_W4A8=1`, the small-M verify path, and native mode with
   bridge flags set; check no freed MXFP4 payload is read. Exercise TP/PP if
   the production configuration uses them.
5. Measure KLD and throughput only after generation is coherent. The GGUF
   Q4_K_M KLD target is near llama.cpp's 0.061 against BF16, compared with
   Strategy A's measured 0.216 for Q4_K_M and 0.196 for BF16 GGUF
   re-quantized to MXFP4.
