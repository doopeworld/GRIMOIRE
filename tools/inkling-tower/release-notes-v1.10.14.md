## GRIMOIRE v1.10.14 — Inkling 2 Mini with the DFlash2 drafter

v1.10.13 plus the Inkling 2 Mini server drafting with
[incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) — a block-diffusion drafter with a
candidate selector — instead of the model's own DSpark head. Output stays exact (speculative sampling; greedy output
equals plain decoding). Built for Arc Pro B70 (G31) and Arc Pro B60 / Arc B580 (G21). Every other model runs as in
v1.10.13.

```
# /path/to/models/Inkling-2-Mini            (or Inkling-2-Mini-Uncensored)
# /path/to/models/Inkling-2-Mini-DFlash2    (hf download incoai/Inkling-2-Mini-DFlash2 --local-dir ...)
docker run -d --name grimoire-inkling --stop-timeout 60 \
  --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
  -v /path/to/models:/models:ro -e INK_MODEL=/models/Inkling-2-Mini -p 6911:6911 \
  -e INK_CTX=262144 --entrypoint grimoire-inkling grimoire-b70:v1.10.14
```

The drafter is used automatically when its folder sits next to the model (`<models>/Inkling-2-Mini-DFlash2`) or in
`<model>/dflash2`. `INK_DFLASH2=<dir>` names it; `INK_DFLASH2=0` keeps DSpark.

### Measured (two Arc Pro B70, Inkling-2-Mini-Uncensored)

| test | DSpark (v1.10.12) | DFlash2 (v1.10.14) |
|---|---|---|
| BetterBench quick (temperature 0.7, top-p 0.95, top-k 20), weighted decode, on the images | 141.8 tok/s | **173.5 tok/s** |
| tokens per decode step on BetterBench (engine count) | 2.90 | 3.62 |
| greedy code answer / chat answer, 384 tokens | 251 / 137 tok/s | **304 / 236 tok/s** |
| one B70, greedy code answer / chat answer | 177 / 88 tok/s | **213 / 149 tok/s** |
| BetterBench 1 / 2 / 4 / 8 users, total (served one at a time) | 125 / 124 / 126 / 122 | 157 / 155 / 157 / 154 |
| BetterBench prefill 2K / 8K | 2,869 / 3,613 tok/s | 2,870 / 3,609 tok/s |
| memory per card at a 262,144-token context | 30.3 GiB | 27.0 GiB |

The drafter's layers run in INT8 (the server default `INK_I8=1`); BF16 gave the same acceptance and was slower, INT4
accepted less. BetterBench quick varies about ±6% from run to run.

Source: branch `inkling-df2-1126` (also `main`), engine `tools/inkling2_df2.inc`.

Image archive: `grimoire-b70-v1.10.14.tar.gz`, 1,908,118,213 bytes, sha256 `3429fbcfcbe95998a5c7f423248fbe7d6a2f870b2b7a369cf340a348339b8ca9`.
Load with `docker load -i grimoire-b70-v1.10.14.tar.gz`.
