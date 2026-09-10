# Local validation, 2026-09-10

Branch: `codex/b70-quantizer-20260910`.
Base: `qwen-tg-restored` at `390c9d5f94f34f3995a87ed1554a499c78b157a1`.
The work is isolated from the original branch.

## Completed

- Built the host quantizer and native-model inspector with g++.
- Built and linked both inference CLI and HTTP server with Intel LLVM SYCL,
  targeting portable `spir64`, with `GRIMOIRE_NATIVE_ONLY` enabled.
- Both inference executables start and print help.
- Inspected executable dependencies: the inference binaries need the SYCL
  runtime and ordinary C/C++ system libraries, with no Torch/vLLM dependency.
  The native profile removes external bridge loading at compile time.
- `make test`: seven host suites pass (formats, attention, safetensors, MoE,
  DeltaNet, ops, GPTQ). The tokenizer suite reports its existing skip because
  no real model/tokenizer was supplied.
- `make test-native`: all seven formats convert and reload correctly, preserving
  packed payloads, scales, zeros, raw tensors, and model/tokenizer metadata.
  Every synthetic gate/up/down expert slice matches independently quantized
  source rows. Prefill/TP row slicing uses the same tested helper.
- INT4 and FP8 conversion crosses the 32 MiB input window correctly; rows
  before and after the chunk boundary preserve their bytes and scales.
- Malformed native files and safetensors are rejected. Existing outputs and
  original BF16 sources remain unchanged. Failed exports leave no published
  output or temporary directory.
- AddressSanitizer, UndefinedBehaviorSanitizer, and float-cast-overflow checks
  pass for the converter/native integration suite and safetensors suite.
  LeakSanitizer cannot run under this environment's process tracing, so these
  runs use `ASAN_OPTIONS=detect_leaks=0`; leak checking is **not** claimed.
- `git diff --check` passes.

The sanitizer runs exposed unaligned typed loads in the existing safetensors
decoder. Those reads now use alignment-safe copies, with explicit regression
coverage for floating and integer dtypes. Additional fixes cover overflowing
header lengths, dimensions and offsets, and duplicate tensor names.

## Not established

- No Intel Arc Pro B70 is available in this environment.
- The compiler recognizes `intel_gpu_bmg_g31`, but a B70 AOT image has not been
  built. The attempt to download Intel's missing `ocloc` package was blocked by
  a cancelled network approval. No GPU driver or system package was installed.
- No real Ornith checkpoint has completed inference here. Synthetic weight
  tests do not prove coherent text, tokenizer correctness, model-level logit
  parity, perplexity, or quantized language quality.
- No llama-benchy performance measurement has been run. The requested
  10,300 prefill and 125 generation tokens/s at `--pp 4096 --tg 32` remain
  targets, not results.
- Quantization is round-to-nearest, without AutoRound or activation calibration.
- The native-only profile disables optional external acceleration bridges;
  it has not been optimized or benchmarked against their performance.
- Native v3 MoE MTP loading is explicitly rejected until its separate expert
  loader supports the saved formats.

These results establish a compiled and tested conversion/loading foundation.
They do not establish a production-ready or fastest-market inference engine.
