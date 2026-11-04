# GRIMOIRE v1.10.3 — three optional speed modes

Experimental release built from `546ce890beaba05e025dac6434decb8e03c42450`.

Default behavior matches v1.10.2: the 44-run regression on this source gave answers identical to the
previous run. This image adds three options, all off by default:

| Option | What it does | Measured here | Limit |
|---|---|---|---|
| `GRIMOIRE_MOE_NATIVE_SMALLM=1` | Native GGUF experts share reads across small batches | Ornith Q4_K_M, 8 users: 133.8 -> 359.3 tok/s total; one user unchanged | Native GGUF on one card, 2 to 16 rows |
| `GRIMOIRE_MTP_GPU_DRAFT=1` | MTP drafts stay on the device between depths, no host readback | Qwen GPTQ, 4 users: 171.9 -> 183.2 tok/s total | Variance overlaps across runs; not a strong speed claim |
| `GRIMOIRE_ONEDNN_W8A8=3` | oneDNN 8-bit activations on the widest prompt projections | Qwen FP8 as INT8, pp4096: 2072 -> 2457 tok/s | Top-1 against the BF16 original 98.4% -> 97.7% on a limited corpus |

## Validation

Measured on one Arc Pro B70 on 2026-11-02, using this image's own packaged executables and
libraries. llama-benchy 0.4.0, pp4096/tg128; every configuration passed the coherence check.
Concurrent decode is total throughput. Qwen uses eight slots, MTP K=6 and draft vocab 65536;
Ornith uses eight slots without MTP.

| Checkpoint and mode | Prompt, one user | Decode, one user | Concurrent decode, total |
|---|---:|---:|---:|
| Qwen3.8-27B GPTQ-Int4 + MTP | 2,062 | 61.6 | 171.9 (4 users) |
| Qwen3.8-27B UD-Q4_K_XL native + MTP | 1,969 | 56.2 | 143.8 (4 users) |
| Ornith-1.5 Q4_K_M native | 7,130 | 155.2 | 133.8 (8 users) |
| Ornith-1.5 Q4_K_M converted to MXFP4 | 8,525 | 174.8 | 377.7 (8 users) |
| Ornith-1.5 safetensors MXFP4 | 8,633 | 174.8 | 377.1 (8 users) |

Both Qwen modes passed two single-request and eight concurrent coding answers. Qwen's one-user tg128 figures are lower than
v1.10.2's, but each is a two-run sample with overlapping ranges: GPTQ 61.62 +/- 3.92 against 67.08 +/- 2.79, GGUF 56.25 +/- 4.87
against 59.41 +/- 8.62. The fixed coding probe (2,153-token prompt) is unchanged: GPTQ 127.8 tok/s against 127.9, GGUF 95.5 against
96.2. MTP acceptance varies with the text.

The image contains G31 (B70) and G21 (B60/B580) targets. Only a B70 was tested here. Two-GPU
runtime validation was not repeated: the second card's link on the test machine still logs errors.

## Known limitations

- Ornith's concurrent coding answers: 4/8 to 8/8 depending on the weights, not gated. The same prompts fail when run alone, with
  clean Python that skips a step, so this is a model mistake near a tie and not a batching defect.
- Flash-Next decode slows down badly when other processes are busy. Fixed in source on `cpu-pool-late-workers-1102`, not in this image.
- Ornith GGUF MTP is slower than plain decode; leave it off. GGUF files without an MTP head reject MTP.
  DFlash on GGUF and Flash-Next GGUF remain unvalidated or unsupported as described in the README.

## Load the image

```bash
curl -LO https://github.com/doopeworld/GRIMOIRE/releases/download/v1.10.3/grimoire-b70-v1.10.3.tar.gz
docker load < grimoire-b70-v1.10.3.tar.gz
docker tag grimoire-b70:v1.10.3 grimoire-b70:latest
```

Image ID: `sha256:3663f7a7c5cc745b7a16ee5de8b57ba7da37f8bace87cc4ad238af3f9b761e45`.
Asset size: 923,617,984 bytes. gzip integrity check passed.
Asset SHA-256: `972338bee4334215c5e0c5ba4a1d8cc5834601efbafbcaa652640ca71259d5f6`.
