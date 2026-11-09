# Qwen 27B and Flash-Next calibration decision

The previously prepared ff77f2b experiment completed on 2026-11-08/09. Keep the
existing defaults. Qwen INT8 norm folding did not improve the measured logit
agreement. Group 64 imatrix is an optional accuracy choice for Flash-Next INT4,
with a measured generation-speed cost.

| Flash-Next dense INT4 option | Mean cosine vs BF16 | Mean KL | Top-token match | pp512 tok/s | tg128 tok/s |
|---|---:|---:|---:|---:|---:|
| Default group 128 | .9672 | .0849 | 96.9% | 211.95±2.47 | 39.47±.74 |
| Imatrix group 128 | .9679 | .0824 | 96.9% | 224.26±3.61 | 43.20±1.55 |
| Imatrix group 64 | .9747 | .0610 | 97.9% | 217.74±2.10 | 36.55±.03 |

Accuracy uses the saved 512-position Flash reference. Speed uses one B70, 120
experts per layer, heat/fill off, frozen placement reset before each run,
INT4 projections and unchanged NVFP4 experts. The timing pairs are two runs
per configuration, not a general speed guarantee. The experiment build lacks
the newer heat engine; these rates are not the production heat/fill profile's
rates. Compare the options within this table.

Qwen 27B INT8 matched BF16's top token on 98.4% of the saved 256-position sample.
Mean KL was .0023 without folding and .0032 with alpha .2, with the same top-token
agreement. Keep INT8 unmodified. The precision-correct fold remains opt-in.

## Options

- `GRIMOIRE_IMATRIX=/path/to/model-specific-imatrix.gguf` uses importance-weighted
  INT4 rounding when converting dense source weights at load time.
- `GRIMOIRE_INT4_RTN_GROUP=64` selects the smaller group; default 128.
- `GRIMOIRE_INT4_RTN_SEARCH=1` searches ranges without an imatrix; default off.
- `GRIMOIRE_IMATRIX_FOLD=<alpha>` folds eligible input norms and projections;
  leave unset for the measured Qwen INT8 profile.

The imatrix must match the model's layers and input-channel widths. Missing or
wrong-width entries keep the existing rounding. MTP/vision weights do not use
backbone statistics. These options do not re-quantize stored NVFP4 experts or
make a pre-quantized checkpoint equivalent to a dense source.

On Tower, the verified Flash matrix is mounted at
`/models/grimoire-acc/imatrix-flash-next.gguf`. The Qwen matrix is at
`/models/grimoire-acc/imatrix-qwen38-27b.gguf`. Both were restored and verified
before the sweep; no model or calibration downloads were repeated.

Evidence: bench-1022/logs/accuracy-1108.out, acc1108-*.VERIFIED,
imx-perf1109.out, imxp1109-fn-*.{run,server.log,benchy.md}.
