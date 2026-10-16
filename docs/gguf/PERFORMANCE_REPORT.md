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

## 5. Native Q4_K/Q6_K execution (`--proj native`, 2026-10-16)

GGUF Q4_K/Q6_K blocks are kept on the device and decoded inside the kernels (commits c46f0cf,
fa721d3). Ornith Q4_K_M: 371 tensors native (Q4_K 310 = 14.66 GiB, Q6_K 61 = 4.69 GiB incl. the
output head); remaining small projections MXFP4; critical tensors BF16.

**Accuracy** (WikiText-2, first 16 chunks of 512, vs the BF16 distribution, llama.cpp formulas):

| Configuration | Mean KLD | Same top-1 | PPL / BF16 |
|---|---:|---:|---:|
| **GRIMOIRE native Q4_K/Q6_K** | **0.064** | **88.8%** | **1.008** |
| llama.cpp native Q4_K_M (64 chunks) | 0.061 | 89.3% | 1.010 |
| GRIMOIRE Q4_K_M -> MXFP4 (Strategy A) | 0.239 | 77.9% | 1.067 |

Native execution brings GRIMOIRE to llama.cpp's accuracy on the same file (3.7x lower KLD than
the MXFP4 requant). Generation coherent on test prompts.

**Prefill speed** (CLI `--prefill-only 4096`, synthetic ids, `GRIMOIRE_TIME_LAYER=all` overhead included):

| Path | routed MoE | total | tok/s |
|---|---:|---:|---:|
| MXFP4 (Strategy A) | 164 ms | 470 ms | 8,716 |
| native, per-expert fallback (`GRIMOIRE_MOE_NATIVE_GROUPED=0`) | 1,702 ms | 2,066 ms | 1,983 |
| **native, grouped Q4_K/Q6_K MoE kernel** | **369 ms** | 734 ms | **5,581** |

Prefill-path correctness: decode-vs-prefill max |Δlogit| at position 510 = 0.42 (grouped) vs
0.44 (per-expert reference) vs 0.99 (MXFP4 path). Dense projections in native mode are already
close to MXFP4 (DN qkv 43 vs 37 ms, DN out 52 vs 33 ms). Decode with the generic native GEMV is
~11 tok/s (vs ~170 MXFP4) — the fast native GEMV is the next item.

Open speed items: grouped native MoE 369 -> ~164 ms (MXFP4 kernel parity), native DN out/qkv
projections, fast native decode GEMV + MoE decode.
