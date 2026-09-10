# Native BF16 conversion and saved formats

This branch adds a C++17 offline quantizer and extends the C++/SYCL loader to
preserve the resulting formats. The first supported integration is the existing
Qwen3.5/Ornith model loader. Supporting a weight encoding does **not** imply
support for every model architecture or every GPTQ/AWQ/FP8 checkpoint convention.

The quantizer uses round-to-nearest weights. It does **not** run AutoRound,
activation calibration, GPTQ optimization, or a language-quality evaluation.
Existing GPTQ import/repacking is a separate path. Do not describe these new
INT4 exports as AutoRound or GPTQ models.

## Build and conversion

Host tools require a Linux, little-endian, 64-bit environment with a C++17
compiler. They have no GPU, Python, PyTorch, vLLM, or SYCL runtime dependency.

```sh
make tools
bin/grimoire-quantize /models/ornith-bf16 /models/ornith-int4 --format int4 --bundle
bin/grimoire-quantize /models/ornith-bf16 /models/ornith-fp8 --format fp8 --bundle
bin/inspect_native_model /models/ornith-int4/model-v3.b70 --list
```

These are reproducibility commands; the tools have also been compiled during
development. The converter keeps the BF16 source untouched. Every requested
output must be new. Temporary files are written beside the destination, flushed,
reopened for structural validation, and published without overwriting anything.
Failed conversions remove their temporary output.

`--bundle` creates a directory containing `model-v3.b70`, configuration,
tokenizer files, and a quantization manifest. Reloading that directory does not
need the original BF16 shards. Without `--bundle`, OUTPUT is a single `.b70`
file; inference still requires the matching model configuration and tokenizer.
The loader discovers `model-v3.b70` before the legacy `model-v2.b70` filename.

The default format is `int4`, with `--policy experts`: only routed experts are
quantized. `--policy linear` also quantizes eligible projections. Embeddings,
the output head, norms, delta-rule a/b gates, routers, shared-expert gates,
convolution weights, and MTP weights retain source precision. This conservative
policy is a starting point for quality validation, not a calibrated optimum.
Other choices are `bf16`, `fp8`/`fp8-e4m3`, `fp8-e5m2`, `int8`, `mxfp4`, `mxfp8`.
Unaligned non-expert matrices remain BF16. An incompatible expert row width is
rejected; it cannot silently produce mixed expert formats that the runtime
cannot assemble.

Conversion streams rows through a roughly 32 MiB FP32 window, including fused
`[E,2I,H]` and `[E,H,I]` expert tensors. It reports reconstruction errors over
quantized **weights**, not perplexity, output coherence, or benchmark speed.

## File contract

The writer emits version 3; the reader also accepts version 2 RAW/MXFP4 files.
The header remains 48 bytes and each tensor record 288 bytes. Tensor names,
source dtype, original shape, encoding, and byte ranges are retained.
Payloads, scale regions, and the TOC begin at 4096-byte boundaries. There is no
row padding. `padded_n/k` must match the flattened logical dimensions; the
legacy `tile_n/k = 64/256` fields identify this layout, not a universal kernel
tile choice. The target field remains `0x031`.

For a matrix `[N,K]` (higher-rank expert tensors flatten the leading dimensions):

| Encoding | ID | Payload | Scale region |
|---|---:|---|---|
| RAW | 0 | Original dtype bytes | None |
| MXFP4 | 1 | E2M1, two nibbles per byte, low nibble first | E8M0 `[N,K/32]` |
| INT4 affine | 2 | Unsigned 0–15 nibbles, low nibble first | BF16 scales `[N,K/128]`, then uint8 zeros of the same shape |
| FP8 E4M3 | 3 | E4M3 `[N,K]` | Float32 `[N]`, per output channel |
| FP8 E5M2 | 4 | E5M2 `[N,K]` | Float32 `[N]`, per output channel |
| INT8 | 5 | Signed bytes `[N,K]` | Float32 `[N]`, per output channel |
| MXFP8 | 6 | E4M3 `[N,K]` | E8M0 `[N,K/32]` |

INT4 reconstructs `(code - zero) * scale`. Zero points in files are 0–15;
the runtime-only signed-INT4 sentinel `0xff` is not a disk encoding. The
quantizer includes zero when choosing a group's range and rounds its scale
to BF16 before choosing codes. This fixes the one-sided-group bug where
weights around 10–11 previously reconstructed near 1.

The reader rejects invalid versions/targets, overflowing dimensions or offsets,
overlapping regions, duplicate or unterminated names, unknown dtypes/encodings,
incorrect byte counts, invalid scales, and invalid INT4 zero points. It does
not provide a cryptographic checksum or detect every possible finite-valued
payload corruption.

## Inference integration and native-only build

Native v3 precision is authoritative: runtime `--proj` does not requantize its
matrices. Dense concatenation preserves matching payload/scale/zero planes.
Fused-expert slicing advances all offsets, including RAW BF16 offsets. Prefill
and tensor-parallel slicing use the correct scale element size (4 bytes for
FP8/INT8, 2 for INT4, 1 for MX formats). Expert uploads wait on the queue that
actually owns those allocations before releasing host staging buffers.

```sh
# AOT build requires Intel DPC++ and an ocloc version supporting this target.
make native SYCL_CXX=icpx

# Portable SPIR-V compilation check on a host without the B70 AOT compiler.
make native SYCL_CXX=/path/to/sycl/clang++ SYCL_TARGET=spir64
```

Both commands build the CLI and HTTP server with `GRIMOIRE_NATIVE_ONLY`.
This compile-time profile disables **all** optional external bridges, including
the repository's Torch/vLLM bridges. Execution uses the native SYCL/XMX paths.
The SYCL runtime and its Level Zero backend are still required on the GPU host.
Portable SPIR-V compilation is not evidence of B70 AOT performance.

Native v3 MoE artifacts currently reject `GRIMOIRE_MTP=1` before weight upload:
the separate legacy MTP expert loader assumes MXFP4. MTP integration, additional
architectures, calibrated quantization, and optimized per-format kernels remain
separate work. The original build script and other agents' branches are retained.

## Validation

```sh
make test
make test-native
```

The host regression tests cover one-sided/constant/zero INT4 groups, codec and
GEMV errors, GPTQ repacking, malformed safetensors, all seven export/reload
formats, unchanged raw tensors and metadata, every fused expert slice, scale
strides, INT4/FP8 streaming chunk boundaries, version-2 compatibility, overwrite
protection, non-finite input rejection, and failed-export cleanup. The existing
tokenizer test skips when no real model directory is supplied; the synthetic
fixture tokenizer is only used to test copying files.

Before a performance claim, validate real Ornith logits and text against a
trusted reference using the same tokenizer and prompt. Compare BF16 and each
quantized mode on held-out prompts, check prefill/decode agreement and KV-cache
behavior, then run **llama-benchy** at `--pp 4096 --tg 32` on the B70. Record
the model revision, format/policy, hardware, driver, context, and cache state.
No real-model coherence or 10,300 prefill / 125 generation tokens/s result has
been established by the host tests.
