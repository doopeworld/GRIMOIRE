# GRIMOIRE GGUF — status (2026-10-18)

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
| Host regression suite | `make -s test` passed after the prefill fix (exit 0) |

## Test results (Ornith Q4_K_M, gpu0)

| | Prompt speed (tok/s) | Generation | Accuracy error (KLD, lower = better) |
|---|---:|---:|---:|
| llama.cpp SYCL | 1,959 | 104 | 0.061 |
| llama.cpp Vulkan | 1,833 | 50 | — |
| vLLM (other format) | 10,590 | 73 | — |
| GRIMOIRE, MXFP4 mode | **9,682** | **172** | 0.216 |
| GRIMOIRE, native mode — before | 1,983 | 11 | 0.064 |
| GRIMOIRE, native mode — now | 5,581 | ~38 | **0.064** |

## Bug hunt today: native mode on Qwen3.8 UD-Q4_K_XL

| Native types enabled | Output |
|---|---|
| Q4_K + Q6_K | correct |
| + Q8_0 | correct |
| + IQ4_XS | correct |
| + Q5_K | was garbage -> **fixed** (B70 rerun: correct) |
| + IQ4_NL | correct |
| all 6 native types together | **correct** (B70) |

## Remaining

| # | Task | State |
|---|---|---|
| 1 | Fast Q5_K decode kernel (now uses the slower safe kernel) | to do |
| 2 | Native mode speed -> match MXFP4 mode (9,682 / 172) | 5,581 / ~38 now |
| 3 | Qwen3-MoE GPU validation; broader Qwen3 generation evaluation | Qwen3-8B GPU smoke test passed; MoE not yet GPU-validated |
| 4 | Accuracy (KLD) for Qwen3.8 + gemma-4 | reference file ready |
| 5 | New engine support: llama, mistral, qwen2, gpt-oss | not started |
| 6 | Qwen3.8-Flash-Next (too big for 1 card, needs RAM offload) | not started |
