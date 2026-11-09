# Qwen 27B and Flash-Next candidate validation — 2026-11-09

The requested scope passed on the packaged `grimoire-b70:v1.10.5-rc3` image,
using the image's own executables. The running uncensored Qwen profile and both
saved Flash profiles now use this image. This is a scoped candidate, not a new
full-model regression certification.

## Running service and saved profiles

| Profile | Port | Configuration | Final state |
|---|---:|---|---|
| GRIMOIRE-DUAL-QWEN3.8 | 6897 | Uncensored FP8 checkpoint loaded INT8; TP2; MTP K6; context 131072; one slot | Running, health and answer passed |
| GRIMOIRE-FLASH-NEXT | 6893 | INT4; one B70; heat/fill; context 65536; one slot | Smoke passed, stopped |
| GRIMOIRE-DUAL-FLASH-NEXT | 6898 | INT8; two B70s; TP2; context 65536; one slot | Smoke passed, stopped |

The current Qwen model directory is `/models2/Qwen3.8-27B-Uncensored-FP8`.
OpenWebUI remains stopped as requested. Its current connection was preserved.
The B580 was never mapped or used. Upstream links are Gen4 x8 and Gen4 x4.
DockerMan templates were backed up before changes and recreated with the
existing DockerMan helper. Model settings and GPU bindings were preserved.

## Packaged-image measurements

All speeds are tokens/second. Benchy rows have two repetitions; TG follows the
specified prefill. Coding probes are separate workloads and must not be quoted
as `tg128` rates.

| Configuration | Context setting | Bench prefill | Prefill rate | tg128 |
|---|---:|---|---:|---:|
| Current uncensored Qwen INT8 TP2, MTP | 131072 | pp4096 | 3581.83 ± 20.02 | 66.45 ± 2.95 |
| Official Qwen INT8 TP2, DFlash2 host history | 131072 | pp4096 | 3208.17 ± 2.73 | 68.46 ± 4.22 |
| Flash-Next INT8 TP2 | 65536 | pp512 | 688.89 ± 8.52 | 65.08 ± 0.24 |
| Flash-Next INT8 TP2 | 65536 | pp2048 | 1318.42 ± 29.35 | 52.88 ± 0.10 |
| Flash-Next INT4 single, heat/fill | 65536 | pp512 | 223.30 ± 2.41 | 50.44 ± 1.76 |
| Flash-Next INT4 imatrix group128, heat/fill | 65536 | pp512 | 232.07 ± 2.53 | 51.23 ± 2.61 |

Current uncensored MTP passed both coding probes at 116.13 and 115.99 tok/s.
Official Qwen DFlash2 with pinned host history processed an actual 128162-token
prompt: 98.24 seconds to first token, 312 completion tokens at 81.47 tok/s,
syntax and functional checks passed. This validates long-prompt execution on
that checkpoint; it is separate from the current uncensored MTP profile.
The user's 256K experiment remains deferred, and the saved context stays 131K.

## Implementation and decisions

- Level Zero IPC peer copies are implemented for two-rank Megatron TP and
  adjacent pipeline stages. Default remains off; `GRIMOIRE_P2P=auto` negotiates
  with safe startup fallback, and `required` refuses unsupported topology.
  Same-device IPC protocol, multi-chunk sums, host partials, BF16, pipeline
  round trips, malformed descriptor handling and mode selection passed.
  **Tower reports no cross-B70 peer access; native imports fail.** Actual TP
  and PP fallback and required-mode refusal were tested. Physical peer-copy
  performance on a P2P-capable tower remains unmeasured.
- `GRIMOIRE_DFLASH_TAPS_HOST=1` preserves full lossless FP32 target history in
  pinned RAM, with bounded GPU staging and RAM/slot accounting. At 131K the
  history is 12.5 GiB per rank; staging defaults to 50 MiB per rank. FP8 KV is
  separate. Host/device answers matched; host budget tests passed.
- The previously prepared 11-case calibration sweep is complete. Keep Qwen
  INT8 unchanged and Flash's group128 default. Group64 imatrix improves the
  saved logit sample at a generation cost and stays opt-in. Packaged CLI
  reproduces group64 cosine .9747298131, KL .0610457622, top1 .978515625.
- Flash MTP remains off; completed tests did not beat plain generation.
  Keep dual-card Flash TP: the already completed PP experiment did not win.
  The expert-wait diagnostic showed similar rank times, roughly 27 us per
  reduction and 68 us queued GPU work; it is not a speed benchmark.

Native streaming, tool calls and tool-result follow-up passed against the API.
OpenWebUI's browser was not retested while the application is stopped.
All three updated user profiles passed their own health/answer smoke checks.
Independent post-deployment verification found no new B70 errors and all 12
available upstream nonfatal/fatal AER totals were zero.

## Provenance and evidence

- Compiled source: `b77a56e038126562bc214ff4d778032d0b183bf1`.
- Image: `sha256:b2c80ad766f14355cd15ca8008551dea9294c2e36e0f5baf73796a2fa25420ff`.
- CLI SHA256: `a29aeaf20eb01e3bf70c627293a2a9d7006a08a8eb0eb9ce9e3ea8b4d2a391ce`.
- Server SHA256: `2cd04763990fb8a9b41a98c5a4e27a014640b4186752307c0b8418d0d389abb1`.
- Manifest inside image: `/usr/share/doc/grimoire/build-manifest.json`.
- Source branch: `p2p-1108`; commits after b77a56e change documentation and
  automation only. `src` and `include` still match the compiled revision.
- Tower evidence base: `/mnt/storage/isos/grimoire-runs/bench-1022`.
- Gate: `logs/candidate-validation1109.out`, `CANDIDATE_SCOPE_PASS`.
- Profiles: `logs/profile-activation1109.json`, `logs/live-verification1109.json`.
- Backup: `profiles-backup1109-20261109-130956`.
- Expert diagnostic: `logs/ep-wait1109.out`, `logs/v1109-fn-epwait.*`.
- Evidence archive: `engine-final-1109.tar.gz`; earlier phase in
  `engine-phase1-1109.tar.gz` on the local results directory.

The rc3 image overlays the changed server and CLI onto the exact validated rc2
runtime. Unchanged libraries were checked rather than rebuilt. G31 and G21 AOT
targets were compiled in a CPU-only builder; this does not claim B580 testing.
The older generic SPIR64 build failed during model loading and was discarded;
the release-flags AOT build and packaged candidate passed all scoped checks.
