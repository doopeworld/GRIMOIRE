# GRIMOIRE GGUF — performance and accuracy report (Ornith-1.5-35B-A3B, Arc Pro B70)

All numbers measured on the Tower, **gpu0 only** (Arc Pro B70, PCI 0000:03:00.0, Gen4 x8),
2026-10-15/16. GRIMOIRE reads the GGUF itself (C++ parser + decoders, decode-at-load "Strategy A"
into its MXFP4 kernels). llama.cpp appears only as a competitor/reference. Logs under
`/mnt/user/isos/grimoire-runs/` on the Tower; `benchmarks/gguf/baseline.csv` for Phase 1.

Model files: `ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612` (Q4_K_M sha256 `42739874…`, BF16 sha256
`044d7f8b…`); HF safetensors `ornith-ai/Ornith-1.5-35B-A3B@10fbf86f`.

## 1. Correctness

- **Gate A (BF16 GGUF == safetensors): PASSED** — 4/4 prompts token-identical over 64 greedy
  tokens (`benchmarks/gguf/gate_bf16.sh`, commit 40da565). All 509,719,168 bf16 values and 171,008
  norm values reconstructed exactly; A_log 384/384 equal to the HF original.
- Q4_K_M GGUF loads (Q4_K 379 / Q6_K 64 / F32 310) and generates coherent text.
- GRIMOIRE is deterministic: repeated perplexity runs produce byte-identical per-token dumps.
- **Tokenizer bug found and fixed (all models, not GGUF-specific):** GRIMOIRE split whitespace runs
  such as `" \n \n "` that HF keeps as one pre-token (298,520 vs 297,193 tokens on WikiText-2).
  After fix 462c165, all 297,193 ids equal HF `tokenizers` 0.22.2 (`tools/tokenizer_vs_hf.py`).
  Known remaining gap: GRIMOIRE does not apply NFC normalization (no effect on NFC input).

## 2. Speed (served, llama-benchy 0.4.0 `--no-cache --exact-tg`, concurrency 1, 5 runs)

| Engine | Weights | pp512 | pp2048 | pp4096 | pp8192 | tg128 |
|---|---|---:|---:|---:|---:|---:|
| **GRIMOIRE (this branch) ← Q4_K_M GGUF** | decode→MXFP4 | — | — | **9,682** | 8,779 | 172 |
| GRIMOIRE (this branch) ← BF16 GGUF | decode→MXFP4 | — | — | 9,641 | 8,905 | 172 |
| GRIMOIRE (this branch) ← BF16 safetensors | →MXFP4 | — | — | 9,565 | — | — |
| GRIMOIRE v1.9.0 image ← BF16 safetensors | →MXFP4 | 6,666 | 9,071 | 9,430 | 8,775 | 172 |
| vLLM 0.31 (cross-format) | GPTQ-Int4 | 5,939 | 9,940 | **10,590** | 10,144 | 73 |
| llama.cpp SYCL (llama-server) | Q4_K_M | 1,414 | 1,851 | 1,959 | 2,050 | 104 |
| llama.cpp Vulkan (llama-server) | Q4_K_M | 1,260 | 1,966 | 1,833 | 1,544 | 50 |

GRIMOIRE-on-GGUF prefill = **4.9× llama.cpp SYCL, 5.3× Vulkan** on the same file (roadmap Gate C
part 1 met), **91% of vLLM** (Gate C part 2 not met), decode 1.7× SYCL and 2.4× vLLM.

## 3. Accuracy (WikiText-2 test, llama-perplexity protocol: ctx 512, first 64 chunks, 16,320 scored tokens, identical token ids in every row)

KLD/top-1 are against the **BF16 model's full distribution** (llama.cpp BF16 run saved with
`--kl-divergence-base`; GRIMOIRE reads the same file via `--ppl-kld-base`, llama.cpp's formulas).

| Configuration | PPL | PPL / BF16 | Mean KLD | Same top-1 | RMS Δp |
|---|---:|---:|---:|---:|---:|
| llama.cpp ← BF16 (reference) | 7.656 | 1.000 | — | — | — |
| llama.cpp ← Q4_K_M (native Q4_K/Q6_K) | 7.736 | 1.010 | **0.061** | **89.3%** | 6.8% |
| GRIMOIRE ← BF16 → MXFP4 (GRIMOIRE today) | 8.119 | 1.060 | 0.196 | 80.2% | 12.0% |
| GRIMOIRE ← Q4_K_M → MXFP4 | 8.203 | 1.071 | 0.216 | 79.5% | 12.8% |

**Reading:** the Q4_K→MXFP4 requant itself adds little (KLD 0.196 → 0.216). The dominant loss is
GRIMOIRE's MXFP4 weight format (E2M1, power-of-two scale per 32): ~3.2× the KLD and ~2× the top-1
disagreement of Q4_K_M at a similar size. Not yet separated: how much of GRIMOIRE's 0.196 is
weight format vs. other engine numerics (activation precision, KV) — native Q4_K execution will
answer that directly.

## 4. Consequence for Phase 3

Native Q4_K/Q6_K execution is not only "format support": it is the way to give GGUF models
llama.cpp-level accuracy (KLD ~0.06) at GRIMOIRE speed. Next: design + kernels, gated first on
KLD (target ≈ llama.cpp's 0.061) and token sanity, then on speed (≥ 9.6k pp4096 now, vLLM 10.6k).
