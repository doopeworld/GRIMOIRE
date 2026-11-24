# Native Inkling 2 Mini port

This branch is the start of a native Grimoire port for `OpenUncensored/Inkling-2-Mini-Uncensored`. It reads the original `model.llm.*` safetensors instead of converting them to GGUF or treating the model as Qwen. The current executable is a **CPU reference for correctness tests**, not the B60 executor or a server fallback. It supports one text sequence, fixed loop passes and greedy token IDs. It does not implement vision, MTP, variable passes, tokenization or production serving.

## Device-free checks

```sh
make inkling2-reference test-inkling2
bin/grimoire-inkling2-reference --inspect-header config.json target-header.prefix
bin/grimoire-inkling2-reference --inspect /path/to/checkpoint
bin/grimoire-inkling2-reference --tokens /path/to/checkpoint ids.txt 32 4 last-logits.f32
```

`--inspect-header` needs only the safetensors length and complete JSON header, not weights. `--inspect` and `--tokens` need `config.json` and the original `model-00001-of-00001.safetensors` data. `ids.txt` is whitespace-separated decimal token IDs. For comparison, the optional binary output contains the last prompt position's logits, not the last generated token's. Padded output IDs are excluded. The Qwen loader rejects the model explicitly until the native device executor exists.

## What was verified

- The original checkpoint header at `OpenUncensored/Inkling-2-Mini-Uncensored@4e5f06d72b5a70c9dcd5e8046aea576eb3ee8487` passed shape, dtype and byte-range checks. Layers 6–17 (inclusive) run three times, producing 48 logical visits: 40 sliding and eight full attention. Logical visits have distinct KV and convolution histories and share the 24 physical layers' weights.
- Twenty physical layers use NVFP4 with E4M3 scales per 16 channels and a multiplier per expert. Four use MXFP8 with E8M0 scales per 32 channels. Gate and up rows alternate in the source. Routing bias selects experts but does not weight their outputs. SwiGLU uses sigmoid of the uncapped gate.
- A six-layer synthetic native-format checkpoint exercises both expert formats, both attention types, three passes, nonzero message gates, convolution history, sliding-window wrap and relative-distance cutoffs. The pinned Transformers eager text graph (`046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3`) recomputed full prefixes; Grimoire appended one token at a time. All 61 output logits matched at each of five prefixes (1, 3, 4, 7 and 11 tokens). Four greedy IDs matched: `13,24,35,17`. Reset, invalid padded input and a truncated header were tested separately. Scripts, raw outputs and hashes are in `K8V4-benchmark-fixes/inkling2-grimoire-port/` alongside this worktree.
- The real trained weights and B60 execution have **not** been tested. Neither result above establishes throughput or trained-model parity.

## First B60 building block: one selected expert

`Checkpoint::expert_matrix` exposes the original bytes and scale for one selected expert. `inkling2::launch_expert_ffn` in `src/gemm_fast.cpp` consumes the same views with device-accessible pointers. It expands only that expert into a reusable BF16 matrix in the order Grimoire's existing XMX kernel reads. It applies the NVFP4 outer scale before rounding weights, preserves alternating gate/up rows, and rounds at the same boundaries as the text reference. It does not route tokens, mix expert outputs, or change stock dispatch. It is a correctness-first implementation, not an optimized small-batch kernel.

The host test matched 15,925,248 BF16 values bit-for-bit against the scalar reader across both Mini matrix shapes and both packed formats. Intel oneAPI 2026.2 compiled the B60-targeted (`intel_gpu_bmg_g21`) shared library and linked `tools/test_inkling2_expert_sycl.cpp` without GPU device nodes. The executable passed its no-device `--contract` check. Its 1- and 17-row device comparisons for NVFP4 and MXFP8 have **not** run. The packed scratch for the larger Mini projection is 10,616,832 bytes per in-flight expert invocation, plus row-dependent activation and projection scratch.

## Remaining B60 integration

1. Stage packed weights once per physical layer. The original text payload is 18,889,358,496 bytes (17.59 GiB); the ideal raw BF16 KV payload at context 2,048 is 224 MiB for one sequence (`40 × 1024` sliding positions plus `8 × 2048` full positions, each holding a 1024-wide key and value). Neither figure includes scratch, allocation padding or other live memory. Do not widen all expert weights at rest.
2. Qualify the selected-expert device fixture on a separately reserved B60, then wire routing, expert accumulation and the fixed-pass text executor. Reuse Grimoire projections and attention code where their math and layouts match Inkling's.
3. Compare native device logits and routes with the CPU reference, then compare fixed prompts against the pinned Transformers path on the trained weights. Measure speed and live memory after correctness and placement pass; validate long context, vision and MTP separately.
