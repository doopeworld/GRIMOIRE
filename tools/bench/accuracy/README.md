# Accuracy check: GRIMOIRE against the original model

Compares GRIMOIRE's logits with a reference implementation (the original checkpoint in Hugging Face
transformers, on the CPU) on the same token ids, position by position:

- **cosine** of the two logit vectors,
- **KL divergence** KL(reference || GRIMOIRE) of the softmax distributions,
- **top-1 agreement**: how often both pick the same most likely token.

The reference runs in its own container (`ref.Dockerfile`); nothing of it is part of GRIMOIRE.

## Run it

```
# 1. reference image (CPU-only PyTorch + transformers), once
docker build -t grimoire-ref:cpu -f tools/bench/accuracy/ref.Dockerfile tools/bench/accuracy

# 2. reference logits of the first 256 tokens of a text, from the BF16 original
mkdir -p /data/acc && cp tools/bench/accuracy/*.py some-text.txt /data/acc/
docker run --rm -v /path/to/models:/models:ro -v /data/acc:/acc grimoire-ref:cpu \
    python /acc/ref_logits.py /models/Qwen3.8-27B /acc/some-text.txt 256 /acc/text.ids /acc/ref.bin

# 3. GRIMOIRE's logits for the same ids (one card), any --proj
docker run --rm --device /dev/dri/renderD128 -v /path/to/models:/models -v /data/acc:/acc \
    --entrypoint /grimoire/bin/grimoire grimoire-b70:latest \
    -m /models/Qwen3.8-27B-FP8 --proj fp8 --ctx 1024 --score-ids /acc/text.ids --score-out /acc/g-fp8.bin

# 4. compare (vocabulary size from the model's config.json)
docker run --rm -v /data/acc:/acc grimoire-ref:cpu \
    python /acc/cmp_logits.py /acc/ref.bin /acc/g-fp8.bin 248320 fp8
```

`grimoire --score-ids` feeds the ids one at a time through the decode path, the path generation uses.
`GRIMOIRE_LM_HEAD_BF16=1` keeps the checkpoint's output head instead of converting it to `--proj`.

## Results 2026-10-29: Qwen3.8-27B, one Arc Pro B70

256 tokens of English prose (the opening of "A Scandal in Bohemia"); reference = the BF16 original in
transformers 5.21 / PyTorch 2.15 on the CPU.

| GRIMOIRE run | cosine mean / min | KL mean | top-1 |
|---|---|---:|---:|
| `--proj int8` from the BF16 original | 0.9996 / 0.9975 | 0.002 | 98.4% |
| `--proj fp8` from the BF16 original | 0.9983 / 0.9908 | 0.014 | 97.7% |
| official FP8 checkpoint, `--proj int8` | 0.9981 / 0.9741 | 0.018 | 98.0% |
| official FP8 checkpoint, `--proj fp8` | 0.9968 / 0.9328 | 0.030 | 94.9% |
| official NVFP4 checkpoint, `--proj int8` | 0.9878 / 0.9074 | 0.181 | 91.4% |
| `--proj mxfp4` from the BF16 original | 0.9718 / 0.8754 | 0.234 | 90.6% |
| `--proj int4` from the BF16 original | 0.9759 / 0.8813 | 0.255 | 88.3% |
| a third-party GPTQ-Int4 checkpoint, `--proj int4` | 0.9697 / 0.9058 | 0.374 | 85.9% |
| the same GPTQ checkpoint, `--proj int8` | 0.9713 / 0.9079 | 0.380 | 85.2% |
| official NVFP4 checkpoint, `--proj mxfp4` | 0.9613 / 0.8067 | 0.454 | 83.2% |

What it says:

- The engine itself follows the original closely (INT8: cosine 0.9996, KL 0.002). That run also uses
  GRIMOIRE's FP8 KV cache, so the KV cache is not where accuracy goes.
- The official FP8 checkpoint loses accuracy in GRIMOIRE's conversion, not in the format: its 128x128
  block scales are unpacked and every weight is rounded again with one scale per row (94.9% top-1,
  against 97.7-98.0% when the weights are rounded once).  **Run these checkpoints with
  `--proj int8`**: 98.0% top-1, and since 2026-10-29 the same speed as `--proj fp8` on one card
  and on two (prompt, decode, several users, MTP).
- An NVFP4 checkpoint converted to MXFP4 at load is the least accurate run (83.2%); the same file
  held as INT8 gives 91.4%.  Running NVFP4 natively is planned.
- The GPTQ checkpoint's loss is in the file itself: held as INT8 it scores the same.
