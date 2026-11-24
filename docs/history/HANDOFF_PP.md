# Grimoire PP optimization handoff

## Objective

Reach coherent, real end-to-end vLLM-class prompt processing on Intel B70 (target
9,000–11,000 PP), while preserving the existing ~106–110 TG decode result. Do
not accept projection-only numbers or isolated kernel throughput as completion.

## Host and paths

- Local repo: `/Users/ianernst/grimoire-work/grimoire`
- Remote: `root@192.168.8.225`
- Remote repo: `/mnt/storage/isos/grimoire-fuse`
- Models: `/mnt/storage/Models`
- vLLM XPU kernel source: `/mnt/cache/appdata/vllm-xpu-kernels`
- Working grouped GEMM library:
  `/mnt/cache/appdata/vllm-xpu-kernels/build/temp/libgrouped_gemm_xe_2.so`
- Container: `my-vllm-xpu:latest`

## Established results

- Coherent decode: about 106–110 TG.
- AutoRound/GPTQ/FP8 model loading has been validated.
- Current coherent full batched prefill:
  - 21 tokens: 153.4 PP
  - 161 tokens: 390.5 PP
- An expert-permuted implementation launching XMX once per active expert was
  coherent but only 106.4 PP at 161 tokens because it makes about 256 small
  launches per layer. This approach is rejected.
- User's measured vLLM baseline is 9,000–11,000 PP.

## Root cause and target design

The current MoE implementation rereads weights or dispatches many small expert
kernels. vLLM instead performs:

1. remap tokens by expert;
2. compact active experts and compute rows per expert;
3. one persistent grouped gate/up GEMM;
4. fused activation;
5. one persistent grouped down GEMM;
6. gather.

vLLM also uses FA2 chunk-prefill attention; Grimoire's current prefill attention
still scans history per query and must later be replaced by tiled chunk-prefill.

## Current implementation step

New files:

- `src/xe2_grouped_bridge.hpp`
- `src/xe2_grouped_bridge.cpp`

The bridge calls already-exported `MoE::MoEGEMMLauncher` BF16/W4A16 template
instantiations from `libgrouped_gemm_xe_2.so`, avoiding another device template
instantiation. Exported policies include `w4a16_policy`, `_m_8`, `_m_16`, and
`_m_32`.

The first bridge compile failed on `cuda_runtime_api.h` because it omitted the
working CUTLASS build defines and forced include. The exact working compile flags
were recovered from `build/temp/compile_commands.json`, notably:

```text
-include /src/csrc/sycl_first.h
-fsycl-targets=spir64_gen
-DCUTLASS_ENABLE_HEADERS_ONLY
-DCUTLASS_ENABLE_SYCL
-DSYCL_INTEL_TARGET
-DCUTLASS_VERSIONS_GENERATED
-fno-sycl-instrument-device-code
```

Also include `/src/csrc`, CUTLASS tools/util, applications, and Torch include
paths exactly as in that compile command. Retry the bridge compile with those
flags, link to `-L/src/build/temp -lgrouped_gemm_xe_2`, then inspect it with
`nm -D -C` and `ldd`.

## Integration after the bridge links

Before replacing full MoE, build one exact equivalence probe using real layer
weights. The persistent W4A16 kernel expects its own signed INT4 packing and does
not consume Grimoire's affine zero points directly. Determine the real zero-point
distribution, convert affine nibbles into the vLLM signed/sign-magnitude format,
and preserve the original payload for decode. Add native gate/up and down buffers
per layer rather than converting per prompt.

Use BF16 grouped outputs:

- BF16 permuted input `[R,H]`;
- grouped gate/up BF16 output `[R,2I]`;
- BF16 fused SwiGLU `[R,I]`;
- grouped down BF16 output `[R,H]`;
- exact weighted unpermute/gather into the existing float residual path.

Only after persistent MoE equivalence and coherence pass, benchmark real full
prefill. Then replace the current attention scan with tiled FA2-style
chunk-prefill and benchmark again.

## Constraints

- Preserve unrelated dirty worktree changes.
- Use `apply_patch` for edits.
- Run `git diff --check` and local tests.
- Validate production AOT build and coherent generated text on the B70.
- Do not expose container environment values; a prior broad Docker inspection
  leaked a Redis credential, which must be rotated separately.
- Do not report completion until real coherent end-to-end PP is normal/vLLM class.

## 2026-08-24 resumed GDN profiling

The vendored raw-GDN profiling header is now Torch-free.  Its upstream
`torch::Tensor` adapter is compile-guarded, and the `vllm::xpu::is_bmg()` call
was replaced by the compile-time BMG assumption because this bridge is built
specifically for `intel_gpu_bmg_g31`.  The architecture probe had left an
unresolved `c10::xpu::get_raw_device` symbol, causing `dlopen` to silently skip
the raw bridge.  The rebuilt library now passes a direct `dlopen` check and
`ldd` has no Torch, c10, Python, or vLLM dependency.

The original five-pass profile across the 30 GDN layers was approximately:

- prepare: 0.13 ms total;
- compute A: 8.1 ms total;
- inverse: 3.1 ms total;
- compute W/U: 119 ms total;
- forward/state: 103 ms total.

`chunk_compute_wu_kernel` assigned one workgroup per chunk and serially looped
over all 32 value heads.  It now maps workgroups across `(value head, chunk)`.
The coherent route/output probe still passes exactly, and W/U fell from roughly
3.9--4.1 ms per layer to 0.98--1.04 ms per layer (about 4x).  Forward/state is
now the dominant internal GDN pass at roughly 3.4--3.6 ms per layer, so a
parallel affine prefix design is now justified.

The clean 4,096-token single-queue run after this change was 831.0 ms, or
4,928.9 PP; measured decode remained 82.5 tok/s.  This run used the genuinely
Torch-free raw bridge.  The 10,000 PP acceptance gate is not yet met.
