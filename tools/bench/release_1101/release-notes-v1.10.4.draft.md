# GRIMOIRE v1.10.4 — Flash-Next on a busy machine, faster two-card exchange, B60 / B580 expert budget

Experimental release built from `35201c20f04edbe07a8cca1fcd04d960bda8242c`.

| Change | What it does | Measured here |
|---|---|---|
| CPU-expert pool: no waiting for workers that are off their core | Flash-Next (experts partly on the CPU) keeps its speed when other processes load the CPU | Flash-Next served, 4 busy processes: 3.91 -> 24.06 tok/s (11-02); same answers, no cost on a quiet machine |
| One card: the decode thread stays free | The CPU-expert pool no longer pins work onto the core the GPU-feeding thread runs on | Flash-Next CLI matrix, 5 loads x pinned / free: identical text in all 10 rows |
| Native GGUF experts grouped for small batches by default | v1.10.3's `GRIMOIRE_MOE_NATIVE_SMALLM=1` is now on (`=0` = off) | Ornith Q4_K_M, 8 users: FILL |
| Two cards: bf16 partials in the decode / verify all-reduce | Half the bytes over the second card's link, summed in fp32 in rank order | Qwen3.8-27B FP8, two B70s + MTP: code 81.4 -> 83.1 tok/s, 4 users 95.7 -> 98.0, every coding answer byte-identical |
| Tiered-expert VRAM budget at most 60% of the card | v1.10.3 asked a 12 GB B580 for 16 GB of experts | B580, Ornith NVFP4 with experts in RAM: FILL |

`GRIMOIRE_TP_SMALL_BF16=0` restores fp32 partials. A B70 keeps its 16 GB expert budget.

## Validation

FILL: one-card table (image's own binaries), two-card Flash-Next, Flash-Next regression cases, B580 first numbers.

## Load the image

```bash
curl -LO https://github.com/doopeworld/GRIMOIRE/releases/download/v1.10.4/grimoire-b70-v1.10.4.tar.gz
docker load < grimoire-b70-v1.10.4.tar.gz
docker tag grimoire-b70:v1.10.4 grimoire-b70:latest
```

FILL: image ID, asset size, SHA-256.
