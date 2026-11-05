# GRIMOIRE v1.10.4 — faster MXFP4 speculative decoding, two-card prompts for 4-bit models, Flash-Next on a busy machine

Experimental release built from `315167c9f8e8512ef2dbacc123ca04adfa9d7918` (branch `mxfp4-smallm-1105`).

## What is faster

| Change | Measured on this machine (one or two Arc Pro B70) |
|---|---|
| MXFP4 MTP verify / batched decode, up to 8 rows: a 128-register kernel with a cheaper exact dequant (nibble pairs) and its own split plan | Qwen3.8-27B MXFP4 + MTP, one card: coding decode 101.6 -> **122.6** tok/s (4K prompt), 90.4 -> **104.8** (16K); tg128 one user 60.1 -> **75.8** (+26%). It now decodes faster than Wrapzii's Swift GPTQ-Int4 at 4K (116.5) |
| MXFP4 verify past 8 rows (2-4 users with MTP): the same dequant and a new split plan in the GEMM library | Qwen3.8-27B shapes 17-30% faster per call at 14 rows (gate_up 263 -> 185 us, lm_head 1,591 -> 1,210, down 128 -> 106) |
| Two cards: 4-bit FFN weights (MXFP4, GPTQ / INT4, GGUF) on the overlapped prompt path | Qwen3.8-27B GPTQ + MTP on two B70s: pp4096 1,733 -> **3,054** tok/s (one card: 2,062); MXFP4 3,106 |
| Two cards: bf16 partials in the decode / verify all-reduce, and each card adds its peer's partial itself | half the bytes over the second card's link, answers byte-identical (FILL: gate) |
| CPU-expert pool: no waiting for workers that are off their core; one card's decode thread stays free | Flash-Next served with 4 busy processes: 3.91 -> 24.06 tok/s (11-02), same answers |
| Native GGUF experts grouped for small batches by default | Ornith Q4_K_M, 8 users: FILL |
| Tiered-expert VRAM budget at most 60% of the card | a 12 GB B580 / 24 GB B60 no longer asks for 16 GB of experts; a B70 keeps 16 GB |

Every change has a switch back: `GRIMOIRE_MX_SMALLM_256=1`, `GRIMOIRE_TP_WAVE_4BIT=0`, `GRIMOIRE_TP_SMALL_BF16=0`,
`GRIMOIRE_TP_SMALL_PEER=0`, `GRIMOIRE_MOE_NATIVE_SMALLM=0`, `GRIMOIRE_CPU_PIN_CALLER=1`.

## Correctness

MXFP4 kernels: every output within 8e-7 (relative to the row maximum) of the fp32 decode GEMV on all Qwen3.8-27B and
Ornith shapes; served answers pass (coding 4/4 at 4K and 16K, 4 users 8/8, coherence). Speculative decoding sums in a
different order, so some generated texts change wording where two tokens were nearly tied: FILL regression summary.

## Validation

FILL: one-card release table (image's own binaries), two-card table, Flash-Next, regression, B580 first numbers.

## Load the image

```bash
curl -LO https://github.com/doopeworld/GRIMOIRE/releases/download/v1.10.4/grimoire-b70-v1.10.4.tar.gz
docker load < grimoire-b70-v1.10.4.tar.gz
docker tag grimoire-b70:v1.10.4 grimoire-b70:latest
```

FILL: image ID, asset size, SHA-256.
