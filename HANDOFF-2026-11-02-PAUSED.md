# Paused by Ian for the day: full Grimoire engine, 2026-11-02 21:05 CET

Ian: "pause all work and save everything for resume tomorrow, send to git what needs to be and we stop for today".
**Nothing restarts until Ian says so.** Numbers are in
[RESULTS-2026-11-02.md](tools/bench/onednn/RESULTS-2026-11-02.md); the morning's checkpoints are in
[HANDOFF-2026-11-02-GPUS-HANDED.md](HANDOFF-2026-11-02-GPUS-HANDED.md).

## State at the pause

- No containers, no GPU holders, both locks free, no book server, no watcher, nothing queued.
- GPU0 `03:00.0` = renderD128: AER 0/0/0 after all of today's runs, no xe errors. Check `lspci` again after any reboot.
- GPU1 `07:00.0` = renderD130 stays off limits: since the 16:40 boot, switch `05:00.0` 10.15 million corrected errors,
  root port `00:06.0` 46,870, 0 uncorrectable.
- Published: v1.10.2 is still latest. GitHub `main` is still `b0674cd`. `grimoire-b70:latest` is still v1.10.2.
- **v1.10.3 candidate is ready and validated, not published.** Local image `grimoire-b70:v1.10.3-rc1`,
  `sha256:3663f7a7c5cc745b7a16ee5de8b57ba7da37f8bace87cc4ad238af3f9b761e45`, source `546ce89`, G31+G21, 3.48 GB.
  `logs/validate-v1103.out` ends `IMAGE_LOG_CHECK_PASS`. The numbers are in the results file and the release notes draft.
- **Next candidate is code only.** Branch `cpu-pool-late-workers-1102` at `1881830`, on top of `3163c12`, not merged:
  CPU-expert pool fix, one card's decode thread left free, native GGUF experts grouped by default.
  Its G31+G21 build was stopped by me at 21:05 on Ian's pause (21 minutes in, `RELEASE_BUILD_FAIL rc=137` in
  `logs/release-build-1102b.out`, CPU only, not a compile failure). It produced no binaries; restart it whole.
- `grimoire-cxpool-1102/tree/bin` holds binaries of the **first** pool version (`2ea7f54`), not of that tree's HEAD.
  They are what the Flash-Next tables measured. Do not treat them as the final code.

The one number to read before publishing: Qwen GPTQ + MTP one-user decode 67.1 -> 61.6 tok/s (-8.1%, the gate's worst).
This is llama-benchy's two-run sample with overlapping ranges (61.62 +/- 3.92 against 67.08 +/- 2.79). The fixed coding probe on the same
image is unchanged: 127.8 tok/s against 127.9, same 341 tokens and 5.40 tokens per update. Decision: not a regression, does not hold the release.

## Exact resume order, after Ian says resume

Check `docker ps` and `lspci` first and leave anything of Ian's alone. All scripts are in `grimoire-runs/bench-1022`, copies in
[tools/bench/release_1101](tools/bench/release_1101). GPU work is one job at a time, GPU0 only.

1. Publish v1.10.3:
   - `bash package_v1103.sh sha256:3663f7a7c5cc745b7a16ee5de8b57ba7da37f8bace87cc4ad238af3f9b761e45` (no GPU).
   - Fetch, then fast-forward `main` from `b0674cd` to `engine-integration-1101`. Ian edits the README on GitHub, so check first.
   - README: download link and validation table.
   - Fill the asset lines in [the release notes draft](tools/bench/release_1101/release-notes-v1.10.3.draft.md), then
     `tools/bench/upload_release.py v1.10.3 ASSET 546ce890beaba05e025dac6434decb8e03c42450 NOTES` as a NEW release, keeping v1.10.2. Standing
     authorization and token rules are in the release workflow notes.
   - `bash switch_v1103.sh sha256:3663f7a7...`, then `bash smoke_v1103.sh`. Check the download URL.
2. `bash run_release_build_v1104.sh`: CPU only, cores 6-13, CLI and server at the same time, about an hour. It reuses the two libraries from
   the v1.10.3 build and refuses if anything but `cpu_experts` and `grimoire.cpp` changed.
3. On those binaries (`grimoire-release-1102b`), GPU0, in order: Flash-Next short case and served, quiet and with load (`run_pin_matrix_1102.sh` with
   `POOL_TREE` and `MATRIX_TAG`, `run_fn_serve_1102.sh`); text must match. Ornith GGUF `conc4,seq4` with no option set must match today's
   `r1102-opt-moe`. Then `GRIM_TREE=/mnt/storage/isos/grimoire-release-1102b tools/regress_all_g0.sh` against `regress-1102-1701`.
4. Merge the branch to main the same day, assemble and validate as v1.10.4 with a Flash-Next configuration added, publish.
5. Continue [the roadmap](ROADMAP-2026-10-31.md). My read of the accuracy table, unmeasured: native NVFP4 dense kernels look worth more than
   native block-scaled FP8. NVFP4 converted to MXFP4 scores 83.2% top-1 against 91.4% as INT8, while FP8 already reaches 98.0% with `--proj int8`.
