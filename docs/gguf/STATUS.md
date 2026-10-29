# GRIMOIRE GGUF — status (2026-10-21)

Goal: GRIMOIRE reads any Hugging Face GGUF natively (own parser + kernels, no llama.cpp inside).
Branch `feature/gguf-battlemage` on GitHub. Details: `MODEL_MATRIX.md`, `PERFORMANCE_REPORT.md`.

## Done

| Item | Result |
|---|---|
| GGUF reader | all 35 ggml quant types, bit-exact vs reference |
| No extra files needed | tokenizer + chat template read from the GGUF; split files; `.gguf` path |
| Ornith-1.5-35B (qwen35moe) | BF16 = **identical tokens** to original; Q4_K_M works |
| Qwen3.8-27B (qwen35) | BF16 = **identical tokens**; 5 Unsloth quants (incl. IQ2/IQ1) work |
| gemma-4-12B | 4 files work (BF16, Q4_K_M, UD-Q4_K_XL) |
| Tokenizer bug (all models) | fixed: now identical to Hugging Face (297,193 tokens) |
| Qwen3-8B norm bug (engine) | fixed and GPU-verified: prompt-path max logit difference **9.96 → 0.25**; native Q4_K_M produces coherent, distinct continuations on science and Fibonacci prompts (responses truncated in thinking; final answers not evaluated) |
| Qwen3-8B accuracy | same Q8_0 file versus llama.cpp: BF16 execution KLD **0.006557**, native execution **0.005382** (16 × 512-token chunks; measured before the prefill fix, using sequential scoring) |
| Host regression suite | `make -s test` passed at ea11022 (exit 0) |
| Fast Q5_K decode (10-21, ea11022) | cause: B70 compiler miscompiles a per-lane byte shift; Q5_K now on the fast 16-lane path. Qwen3.8 UD-Q4_K_XL: identical ids fast vs reference, decode ~2 -> ~9-10 tok/s |
| Native decode speed (10-21, ea11022) | Ornith Q4_K_M native: ESIMD decode GEMV + fused MoE decode, **~40 -> ~132 tok/s** (MXFP4 162); KLD 0.0649 (unchanged) |

## Test results (Ornith Q4_K_M, gpu0)

| | Prompt speed (tok/s) | Generation | Accuracy error (KLD, lower = better) |
|---|---:|---:|---:|
| llama.cpp SYCL | 1,959 | 104 | 0.061 |
| llama.cpp Vulkan | 1,833 | 50 | — |
| vLLM (other format) | 10,590 | 73 | — |
| GRIMOIRE, MXFP4 mode | **9,682** | **172** | 0.216 |
| GRIMOIRE, native mode — before | 1,983 | 11 | 0.064 |
| GRIMOIRE, native mode — 10-18 | 5,581 | ~38 | 0.064 |
| GRIMOIRE, native mode — 10-21 (CLI, agent-measured) | 5,886 | ~132 | **0.0649** |

## Bug hunt today: native mode on Qwen3.8 UD-Q4_K_XL

| Native types enabled | Output |
|---|---|
| Q4_K + Q6_K | correct |
| + Q8_0 | correct |
| + IQ4_XS | correct |
| + Q5_K | was garbage -> **fixed** (B70 rerun: correct) |
| + IQ4_NL | correct |
| all 6 native types together | **correct** (B70) |

## Remaining (scope: Ornith-1.5, Qwen3.8-27B, Qwen3.8-Flash-Next only)

| # | Task | State |
|---|---|---|
| 1 | Fast Q5_K decode kernel | **done** (ea11022, B70-verified) |
| 2a | Ornith native speed -> MXFP4 (9,682 / 172) | decode ~132 tok/s; prefill 5,886 (gap: grouped native MoE prefill tile) |
| 2b | Qwen3.8-27B native decode speed | ~10 tok/s on UD-Q4_K_XL: slow, not yet worked on |
| 3 | IQ4_NL bug on the fast path (Qwen3.8 UD-Q4_K_XL, 6 tensors) | decode vs prefill logits drift 3.5-4.0 (others 0.2-0.8); workaround `GRIMOIRE_GGUF_NATIVE_TYPES` without iq4_nl |
| 4 | Accuracy (KLD) Qwen3.8 | UD-Q4_K_XL native 0.106 / top-1 86.9% (16x512); llama.cpp on same file not yet run; 51 tensors fall back to MXFP4 |
| 5 | Qwen3.8-Flash-Next GGUF (too big for 1 card, needs RAM offload) | not started; waiting for Ian's choice of file |

Out of scope (branches kept, not merged): `gguf/arch` (llama/mistral/qwen2), `gguf/moe` (Qwen3-MoE), gemma-4 KLD.
`bin/grimoire-server` on the Tower is still the 10-17 build (CLI `bin/grimoire` is ea11022).
