# GRIMOIRE on real Hugging Face GGUF files — validation matrix

Every row is a real file downloaded from Hugging Face (pinned revision, sha256 verified),
loaded by GRIMOIRE **as a bare `.gguf` path with no tokenizer.json/config.json beside it**, on
gpu0 (Arc Pro B70, 03:00.0). "ok" = loads, two different prompts generate coherent text that
depends on the input (CLAUDE.md rules 8, 12). Logs: `/mnt/user/isos/grimoire-runs/gguf-validate/`.
Harness: `benchmarks/gguf/validate_models.sh`.

## Architecture `qwen35moe` — Ornith-1.5-35B-A3B (`ornith-ai/...-GGUF@12393612`)

| File | Types | Mode | Result |
|---|---|---|---|
| BF16 | BF16, F32 | mxfp4 | **token-identical to HF safetensors** (4/4 prompts) |
| Q4_K_M | Q4_K, Q6_K | mxfp4 | ok — KLD vs BF16 0.216 |
| Q4_K_M | Q4_K, Q6_K | **native** | ok — **KLD 0.064 / top-1 88.8%** (llama.cpp 0.061 / 89.3%) |

## Architecture `qwen35` (dense) — Qwen3.8-27B (`unsloth/Qwen3.8-27B-GGUF@4ca72078`)

| File | Types in file | Mode | Result |
|---|---|---|---|
| BF16 (split, 2 shards) | BF16, F32 | mxfp4 | **token-identical to HF safetensors** (2/2); 1.27 G bf16 values exact |
| UD-Q4_K_XL | IQ3_S IQ4_NL IQ4_XS Q3_K Q4_K Q5_K Q6_K Q8_0 | mxfp4 | ok |
| UD-Q4_K_XL | same | native (Q4_K/Q6_K) | ok |
| UD-IQ4_XS | IQ2_S IQ2_XS IQ3_S IQ3_XXS IQ4_NL IQ4_XS Q2_K Q3_K Q4_K Q5_K Q6_K Q8_0 | mxfp4 | ok |
| UD-IQ2_XXS | IQ1_M IQ1_S IQ2_S IQ2_XS IQ2_XXS IQ3_S IQ3_XXS IQ4_XS Q2_K Q3_K Q4_K Q8_0 (no MTP head) | mxfp4 | ok |
| Q4_0 | Q4_0 Q4_1 Q5_K Q6_K Q8_0 | mxfp4 | ok |
| Q8_0 | Q8_0 | mxfp4 | ok |

## Architecture `gemma4` — gemma-4-12b-it (`unsloth/gemma-4-12b-it-GGUF@fc034cff`)

| File | Types | Mode | Result |
|---|---|---|---|
| BF16 | BF16, F32 | bf16 | ok (tokenizer "gemma4" built from GGUF) |
| Q4_K_M | Q4_K, Q6_K | mxfp4 | ok |
| Q4_K_M | Q4_K, Q6_K | native | ok |
| UD-Q4_K_XL | Q4_K Q5_K Q6_K | mxfp4 | ok |

Not yet measured for gemma4 / Qwen3.8: KLD vs llama.cpp BF16 (in progress for Qwen3.8).

## Coverage summary

- ggml tensor types decoded at load: all 35 (bit-exact vs upstream gguf-py / ported ggml C).
  Exercised on real files so far: 17 types (above).
- Native in-kernel types: Q4_K, Q6_K (validated); Q5_K, Q8_0, IQ4_XS, IQ4_NL (host-tested, B70 pending).
- File features: single file, split shards, files without MTP head, mmproj/imatrix siblings ignored.
- Tokenizers from GGUF metadata: gpt2/qwen35 (== HF on 297,193 WikiText-2 ids), gemma4, llama SPM.

## Refused by name (not yet implemented)

- `qwen4exp` (Qwen3.8-Flash-Next): GGUFs are 72–200 GB; needs GRIMOIRE's RAM/SSD expert tiering
  (FLASH-NEXT-TIERED.md) wired to GGUF. Planned.
- `qwen3`, `qwen3moe`: engine audit + adapter in progress (norm convention differs from Qwen3.5).
- `llama`, `mistral`, `qwen2`, `gpt-oss`, others: need engine support.
- tokenizer pre-tokenizers other than `qwen35` for gpt2-BPE (e.g. `qwen2`, `llama3`) until each is
  verified bit-exact against HF.
