# GRIMOIRE v1.10.4 — faster MXFP4 speculative decoding, two-GPU prompts for 4-bit models, Flash-Next on a busy machine

Experimental release built from `315167c9f8e8512ef2dbacc123ca04adfa9d7918` (branch `mxfp4-smallm-1105`).

## What is faster

| Change | Measured on this machine (Arc Pro B70) |
|---|---|
| MXFP4 MTP verify / batched decode: a cheaper exact dequant (nibble pairs), a 128-register kernel for up to 8 rows, new split plans (also past 8 rows) | Qwen3.8-27B MXFP4 + MTP, one card, same binary: coding decode 101.6 -> **122.6** tok/s (4K prompt), 90.4 -> 104.8 (16K); tg128 one user 60.1 -> **75.8** (+26%). Kernels 13-30% faster per call on Qwen3.8-27B verify shapes |
| Two GPUs: 4-bit FFN weights (GPTQ / INT4, MXFP4, GGUF) on the overlapped prompt path | Qwen3.8-27B GPTQ + MTP on two B70s: pp4096 1,733 -> **3,093** tok/s (one card: 2,062); MXFP4 3,106 |
| Two GPUs: bf16 partials in the decode / verify exchange, and each card adds its peer's partial itself | Qwen3.8-27B-FP8 + MTP on two B70s: coding decode 83.1 -> **93.9** tok/s, 4 users 98.0 -> 102.7; answers byte-identical with the peer step on or off |
| CPU-expert pool: no waiting for workers that are off their core; one card's decode thread stays free | Flash-Next served with 4 busy processes: **2.55 -> 16.81** tok/s, answers identical; quiet machine unchanged |
| Native GGUF experts grouped for small batches by default (v1.10.3's opt-in) | Ornith Q4_K_M, 8 users: 133.8 -> **362.7** tok/s total |
| Tiered experts: default VRAM budget at most 60% of the card | a 12 GB B580 / 24 GB B60 no longer asks for 16 GB of experts; a B70 keeps 16 GB |

Switches back to v1.10.3's behaviour: `GRIMOIRE_MX_SMALLM_256=1`, `GRIMOIRE_TP_WAVE_4BIT=0`, `GRIMOIRE_TP_SMALL_BF16=0`,
`GRIMOIRE_TP_SMALL_PEER=0`, `GRIMOIRE_MOE_NATIVE_SMALLM=0`, `GRIMOIRE_CPU_PIN_CALLER=1`.

## Validation

The image's own binaries on one Arc Pro B70, llama-benchy 0.4.0, pp4096 / tg128, v1.10.3 -> v1.10.4, same machine and test.
Concurrent decode is total throughput.

| Checkpoint and mode | Prompt, 1 user | Decode, 1 user | Concurrent decode | Checks |
|---|---:|---:|---:|---|
| Qwen3.8-27B GPTQ-Int4 + MTP | 2,062 -> 2,061 | 61.6 -> 74.3 | 171.9 -> 177.2 (4 users) | coding 2/2 (127.9 tok/s, was 127.8), 8/8 concurrent |
| Qwen3.8-27B UD-Q4_K_XL native + MTP | 1,969 -> 1,967 | 56.2 -> 58.1 | 143.8 -> 141.1 (4 users) | coding 2/2 (98.8 tok/s, was 95.5), 8/8 concurrent |
| Ornith Q4_K_M native | 7,130 -> 7,254 | 155.2 -> 155.0 | 133.8 -> **362.7** (8 users) | Paris, coherence |
| Ornith Q4_K_M converted to MXFP4 | 8,525 -> 8,624 | 174.8 -> 175.1 | 377.7 -> 389.2 (8 users) | Paris, coherence |
| Ornith safetensors MXFP4 | 8,633 -> 8,547 | 174.8 -> 175.1 | 377.1 -> 388.4 (8 users) | Paris, coherence |
| Qwen3.8-Flash-Next NVFP4, one card | pp512 181.1 | 24.35 | -- | answers equal the host run 3/3 |

The one-user tg128 cells are two-run samples (MTP acceptance varies with the text); the coding probes are the steadier
measure. Every row passed the benchmark's coherence check; worst speed change against v1.10.3: -1.9%.

Two GPUs (gpu1 on a PCIe Gen4 x4 link), the candidate's binaries: Qwen3.8-27B GPTQ + MTP pp4096 3,093, tg128 88.8 one user /
138.6 four users, coding 8/8; Qwen3.8-27B-FP8 + MTP coding 93.9 tok/s; Flash-Next tg128 45.7. From the same learned expert
placement, two-GPU Flash-Next with fp32 partials gives v1.10.3's answers exactly; with the default bf16 partials one of three
answers is reworded.

44-case regression: 38 answers identical to v1.10.3's run, 6 reworded where two tokens were nearly tied (DFlash verify
summation order, Flash-Next's learned placement, and Muse-Glimmer, whose wording also changed with v1.10.3's own binary
today). Every answer stays correct.

The image contains G31 (B70) and G21 (B60 / B580) targets.

## Load the image

```bash
curl -LO https://github.com/doopeworld/GRIMOIRE/releases/download/v1.10.4/grimoire-b70-v1.10.4.tar.gz
docker load < grimoire-b70-v1.10.4.tar.gz
docker tag grimoire-b70:v1.10.4 grimoire-b70:latest
```

Image ID: `sha256:e0101d360a04f397cd5b1731321b6cd28e9e5c48cae6d3f3e95caf5cf9fbe3ba`.
Asset size: 1,034,948,396 bytes. gzip integrity check passed.
Asset SHA-256: `9ebefe9c21e13d685974249ccb256f9a1ab2fb93912573525572f1be84a2ae30`.
