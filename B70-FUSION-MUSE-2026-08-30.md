# Grimoire B70 FUSION — Muse Glimmer checkpoint

This packages the known-working Intel XPU FUSION execution path as a Grimoire
runtime. It preserves the W4A16 kernel, FlashAttention 2, XPU graphs and DFlash
block scheduler while using the flattened B70 runtime image.

## Accepted result

Single Intel Arc Pro B70, Muse-Glimmer-30B-INT4-W4A16, DFlash k=15,
llama-benchy 0.4.0:

| measurement | result |
|---|---:|
| coherence | PASS |
| PP4096 | 2127.58 +/- 26.21 tok/s |
| TG256 sustained | 44.87 +/- 4.14 tok/s |
| TG256 peak | 58.67 +/- 2.87 tok/s |

The unmodified FUSION reference was 2121.32 PP, 46.85 sustained TG and 51.88
peak TG. Sustained generation overlaps the reference within run variance; the
accepted peak is 13.1% higher.

## Modes and gates

- `prefill`: target-only server with DFlash disabled. Its acceptance gate is
  PP4096 >= 2000 tok/s, independent of speculative decoding.
- `dflash`: target plus Muse assistant, 15 speculative mask tokens, and guarded
  single-request B70 context compaction. The accepted result above is this mode.

The compaction is enabled only for the measured single-request latency path and
is inert for batches with more than one request.

## Build and launch

```bash
IMAGE=grimoire:b70-fusion tools/build_b70_fusion_image.sh
tools/serve_b70_fusion.sh prefill
tools/serve_b70_fusion.sh dflash
```

Use different `NAME` or `PORT` values when retaining both containers. Do not
run both on the same B70 simultaneously.
