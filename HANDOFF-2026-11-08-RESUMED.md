# Grimoire resume, 2026-11-08

Scope: Qwen3.8-27B and Qwen3.8-Flash-Next only. The B580 belongs to ComfyUI.

## Recovered checkpoint

Tower: `root@192.168.8.225`, logs at `/mnt/storage/isos/grimoire-runs/bench-1022/logs`.
Release source: `f2a98b7e6263f3c70bb0d90c0fb50d0d4e52a786`, clean worktree
`/mnt/storage/isos/grimoire-release-1106`.
Candidate image: `grimoire-b70:v1.10.5-rc2`, ID
`sha256:717960860fad7cd194ef9a7bf6d35787059f7c7dc084e8b04348c9290b112210`.
The CLI, server and GEMM library hashes match the successful build log and candidate image.

Completed 11-06; do not repeat:

- G31/G21 release build: `RELEASE_BUILD_PASS` at 20:40 CET.
- Fill off/on A/B: `FILL_AB_END` at 20:51. Decode ~52 vs ~55 tok/s at pp512.
- One-card Flash-Next MTP with heat/fill: completed 20:59. Plain 53.24, K3 37.13,
  K2 47.16 tok/s. Leave MTP disabled for Flash-Next.
- Host-binary two-card checks: Qwen GPTQ+MTP, Qwen INT8+DFlash2 (coding 130.1/130.9 tok/s),
  Flash-Next INT8. All stopped gracefully.
- Full regression: 44/44 artifacts, 41 identical answers, three documented wording changes,
  no empty answers, exit 0. Complete 21:48.
- Candidate image assembly: `IMAGE_ASSEMBLY_PASS` at 21:48.
- Dev PP experiment: completed 20:16, actually 28/20 layers at 340 experts/layer.
  Plain ~47.9, heat ~47.5 tok/s at pp512, while the compiler was running.

Actual stopping point: `V1106_ALL_STOP image validation` at 22:05:51 CET.
`run_v1106_all.sh` exported `IMG=grimoire-b70:latest`; `validate_v1106.sh` preserved that
value using `${IMG:-...}`. Every `r1106-img-*` run therefore used v1.10.4's own binaries,
not rc2. The heat-default assertion caught this; those runs cannot validate rc2.
No build or full regression rerun is needed.

## 11-08 execution

GPU mapping verified after reboot: B70 gpu0 03:00.0/renderD128, B70 gpu1
07:00.0/renderD130; B580 0b:00.0/renderD131, excluded. GPU0 Gen4 x8, GPU1 **Gen4 x4**,
confirmed by Ian and PCI link status. Older results used Gen3 x4 and are historical,
not direct performance controls for today's two-card measurements.

`run_resume_1108.sh` runs seven missing checks with the pinned candidate image's own
binaries, preserves historical logs, and validates artifacts before advancing. Fresh
logs use `r1108-*`, controller log `logs/resume-1108.out`. Successful cases get a
`.VERIFIED` marker; restarting skips those cases and refuses to overwrite partial runs.
Balanced PP sets `GRIMOIRE_PP_LAYERS=24,24` explicitly. The seventh run enables heat
on each stage; the plain PP and TP controls keep the multi-rank default (heat off).

The generic validator's image selection is fixed locally: `VALIDATE_IMG` is its explicit
override, so an inherited benchmark `IMG` cannot select the old runtime by accident.

Further unfinished work identified in the saved handoff: imatrix INT4 rounding/folding
accuracy experiments, source branch `mxfp4-smallm-1105` at `ff77f2b`; no build, accuracy
outputs, or calibration files existed on the Tower at the recovery audit. Temporary Mac downloads are gone.
Existing reference logits and token files remain available. Profile changes and release
publication had been reserved for explicit approval in the prior session; neither is
part of this recovery controller.

The calibration files have now been restored to `grimoire-acc`, with GGUF magic checked:

- Qwen: SHA-256 `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1`.
- Flash-Next: SHA-256 `a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`.

`run_remaining_1108.sh` (initial Tower PID 434834) waits for all seven candidate checks
to pass, then builds the existing `ff77f2b` experiment on CPUs 6-13, without GPU access.
Only after the build passes does `run_accuracy_1108.sh` score Qwen INT4, weighted INT4,
group-64 variants, input-norm folds, INT8 and AutoRound, followed by Flash-Next INT4
variants. It uses existing reference logits, requires exact output dimensions, freezes
expert hits, and leaves Flash-Next heat disabled for accuracy comparisons. Comparison
uses NumPy via uv; rebuilding the missing PyTorch reference image is unnecessary.
Logs: `remaining-1108.out`, `imx-build-1106.out`, `accuracy-1108.out`, `acc1108-*.log`.
New score outputs: `grimoire-acc/acc1108-*.bin`; successful cases get `.VERIFIED` markers.
The release candidate source and image remain frozen throughout these separate experiments.

## Scoped candidate checks completed, 16:17 CET

Seven of seven new cases passed provenance, readiness, coherence, answer/coding,
clean shutdown and link checks; all link counters stayed zero. Source script fixes
also pass shell syntax / Python compile checks; the strengthened original image
validator correctly rejects the old v1.10.4 runs at its provenance check.

| Configuration | Decode at pp512 | Decode at pp2048 |
|---|---:|---:|
| Flash-Next, one B70, BF16, heat off | 24.08 | — |
| Flash-Next, one B70, INT4, heat + fill | 52.27 | — |
| Flash-Next, TP2, INT8 | 64.50 | 52.99 |
| Flash-Next, PP2 24/24, INT8 | 51.40 | 43.63 |
| Flash-Next, PP2 24/24, INT8 + heat | 48.56 | 41.81 |

PP + heat cannot also fit the PLE table in RAM; plain PP and TP both loaded it.
These are end-to-end default configurations, not an isolated measurement of the heat policy.
Keep TP for two-card Flash-Next on today's link; the queued PP experiment is now complete.

Qwen GPTQ+MTP (one B70): 2/2 coding answers at 123.1/123.3 tok/s, concurrent 8/8;
benchy pp4096 2013.6, tg128 72.75 alone / 194.20 total at c4.
Qwen FP8 checkpoint as INT8+DFlash2 (two B70): 2/2 coding answers at 138.9/139.4 tok/s;
benchy pp4096 2856.6, tg128 at pp4096 79.40 (at pp512 95.69).

Full rate table: `logs/summary-1108.md`. Complete scoped evidence is saved locally in
`scoped-checks-1108.tar.gz` (67 files) and extracted under `logs/`.
This is a scoped Qwen/Flash-Next pass. It does not relabel the failed historical
full-model image validator as a full release pass.

At 16:17:25 CET the previously unbuilt imatrix branch started building in
`grimoire-runs/dev-1106`. Its existing host test passed (synthetic weighted error 0.51x
plain rounding). No runtime accuracy result is claimed until scoring completes.
