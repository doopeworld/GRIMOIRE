# GRIMOIRE v1.10.5 — small cards decode with the CPU, steadier multi-user bursts, faster INT4 / GGUF speculation

DRAFT (numbers marked XX come from the gate). Experimental release built from `9a2e424229a6f3034af818b9ab29444dd7f17c77`
(branch `mxfp4-smallm-1105`).

## What is faster

| Change | Measured on this machine (Arc Pro B70) |
|---|---|
| Tiered experts, one card: the experts kept in system RAM are computed by the CPU while the GPU runs the VRAM ones (as Flash-Next already did) | Ornith-1.5-35B-A3B NVFP4 with a 6 GB expert budget (2/3 of the experts in RAM, the situation of a 12 GB B580 or a 24 GB B60): tg128 XX -> XX tok/s |
| Server: a burst with a long prompt (>= 2,048 tokens) is admitted as one batch (waits up to 100 ms for the rest) | Qwen3.8-27B MXFP4 + MTP, 4 users x 16K prompts: tg128 107 +- 62 -> 148.5 +- 8 tok/s total |
| MTP verify: the split-K plans of the INT4 (<= 16 rows), GGUF (<= 8 rows) and MXFP4 o / z kernels re-derived from sweeps; rope + KV append in one launch for all rows | Qwen3.8-27B GPTQ + MTP: XX; UD-Q4_K_XL + MTP: XX |

Switches back to v1.10.4's behaviour: `GRIMOIRE_CPU_SPLIT_DECODE=0`, `GRIMOIRE_ADMIT_HOLD_LONG_MS=8`.

## Validation

XX (the image's own binaries, v1.10.4 -> v1.10.5 table from verify_image_logs_v1105.py)

44-case regression: XX identical to v1.10.4's run; XX reworded (listed).

The image contains G31 (B70) and G21 (B60 / B580) targets. The small-card change was measured on a B70 with a reduced
expert budget, not on a B60 or B580.

## Load the image

```bash
curl -LO https://github.com/doopeworld/GRIMOIRE/releases/download/v1.10.5/grimoire-b70-v1.10.5.tar.gz
docker load < grimoire-b70-v1.10.5.tar.gz
docker tag grimoire-b70:v1.10.5 grimoire-b70:latest
```

Image ID: XX. Asset size: XX bytes. Asset SHA-256: XX.
