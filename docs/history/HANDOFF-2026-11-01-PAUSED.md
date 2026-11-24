# Paused for tomorrow: full Grimoire engine, 2026-11-01

Ian: “pause everything, save everything for resume tomorrow”. No automatic resume.
Pause verified around 20:36 CET / 21:36 Israel on November 1.

## Resources at pause

- No running Docker containers and no process holding any `/dev/dri` node.
- GPU0 is 03:00.0 / renderD128; root/switch correctable, nonfatal and fatal totals are all zero.
- GPU1 07:00.0 remains OFF LIMITS: its link errors were not repaired. A generic resume does not
  authorize using it. No peer-memory imports or manual GPU resets on this Tower.
- Regression controller stopped before another case could start, then terminated after every GPU
  container exited. Its wrapper exit 137 records that intentional CPU-controller termination.
- `grim-rx-w-s` is stopped, exit 143 during weight loading, with logs preserved. No manual GPU
  kill/reset command was issued. Do not count this case as a completed test or assume normal exit.
- CPU-only release build stopped on request; no compiler remains. The verification probe command
  was aborted before execution: no Tower probe worktree, compiled probe or probe GPU run exists.
- Our localhost:8999 book server is stopped. No queued benchmark/build/restart exists.
- `grimoire-b70:latest` is still v1.10.2, image `b14de3b48ba0`; no v1.10.3 image was assembled,
  promoted or published. User profiles and GPU1 configuration were not changed.

## Source checkpoints

| Work | Saved commit / GitHub branch | Local source |
|---|---|---|
| Combined engine | `546ce89` / `engine-integration-1101` (plus this pause documentation) | Mac `grimoire-work/grimoire-engine-20261101`; Tower `/mnt/storage/isos/grimoire-engine-1101` |
| Native grouped MoE prototype | `99756cb` / `native-moe-smallm-1031` | Mac `grimoire-resume-20261031`; Tower `grimoire-native-smallm-1031` |
| Revised device MTP + diagnostics | `01fa420` / `mtp-device-drafts-1031` | Mac `grimoire-mtp-device-20261031`; Tower `grimoire-mtp-device-1031` |
| W8A8 modes 0/1/2/3 | `e215dc4` / `onednn-w8a8-1031` | Mac `grimoire-onednn-w8a8-20261031`; Tower `grimoire-onednn-w8a8-1031` |
| Isolated wide-INT4 register probe | `b010153` / `verify-grf-probe-1101` | Mac `grimoire-verify-grf-20261101`; **not compiled or run** |

GitHub main remains `b0674cd17d5cb2679fba4e4fe2227700fc95ebbb`. Do not force-push or reset
Tower's divergent local main. The Tower's prototype worktrees still contain SCP-applied changes
on older local branch heads; their binary diffs and statuses are saved in the pause snapshots.
Use the Mac/GitHub commits above as the source record, not those old Tower HEADs alone.

The actual combined G31 CLI/server/shared libraries were built and tested on November 1.
Their hashes are in `pause-state.json`. The native/MTP prototype trees' old CLI binaries must
not be substituted for that combined CLI. All new performance options remain OFF by default.

## Completed engine evidence

Full measurements and limitations: [RESULTS-2026-11-01.md](tools/bench/onednn/RESULTS-2026-11-01.md).
The whole unfinished project remains [ROADMAP-2026-10-31.md](ROADMAP-2026-10-31.md).

- Native grouped MoE: 32 independent FP64/guard cases pass, six native GGUF formats. Paired
  Ornith OFF -> ON tg128 total 133.74 -> 364.82 at eight users; single-user unchanged.
  Both paired modes passed only 6/8 concurrent coding samples. Combined build: 357.95 at
  eight users, coherence/Paris pass, **5/8** concurrent coding. Keep experimental; no universal
  coding or token-parity claim.
- oneDNN W8A8: mode 1 pp4096 2077 -> 2909 tok/s, but BF16-reference top-1 98.4% -> 93.8%.
  Mode 3 keeps only wide local output projections: 2527 +/- 46.60 tok/s, top-1 97.7%, code 2/2,
  prose 2/2 and coherence pass, decode unchanged at 20.6 tok/s. Mode 2 top-1 95.7%, not timed.
  Accuracy is 256 independent Sherlock prefixes, not a general task benchmark. All four
  score runs completed with diagnostic persistent cache disabled; first 31 prefixes match
  mode 0 bit-for-bit. The earlier JIT cache-writer segfault and recovery are documented.
- Revised device MTP: all 38 paired responses match text/token count/finish, including 28
  uneven-budget requests. Both modes pass 10/10 code and coherence. Three-run c2 96.14 ->
  105.99 and c4 178.62 -> 184.10 tok/s, with overlapping variance. Verify still costs roughly
  43-46 ms per step. Seven synthetic dense/hybrid cases pass; MoE equality also fails on
  unchanged b0674cd and remains a separate unresolved gate. Do not weaken the oracle.
- Combined Qwen native GGUF + MTP: code 2/2, concurrent code 8/8, coherence and Paris pass.
  One-run pp4096/tg128 2069.15/51.45 single, 1942.25/136.24 total at four users.
- DFlash2/native GGUF: actually drafts, code 2/2 and concurrent code 8/8/coherence pass on
  v1.10.2. Slower than native MTP in that configuration. Fast verify differs at token 36 in
  the 64-token fixture; exact mode matches but is slower. Ornith DFlash remains unhelpful.

## Incomplete gates and exact resume point

1. **Full regression interrupted, not passed.** Started at 20:21 from frozen combined source
   `546ce89`, using `GRIM_TREE=/mnt/storage/isos/grimoire-engine-1101` through the primary
   `tools/regress_all_g0.sh`. Output `/mnt/storage/isos/grimoire-runs/regress-1101-2021`;
   main log `bench-1022/logs/regress-1101.out`. **22/44 cases completed** through Agnes long;
   W4A16 `w-s` was interrupted during loading. Sherlock n24 matches its reference. Restart
   the full suite after explicit resume and compare every answer with `regress-1030-1300`.
   Hold `gpu0.lock` nonblocking. Never run the unchanged primary binaries by accident.
2. **G31+G21 release compilation incomplete.** Isolated source checkout
   `/mnt/storage/isos/grimoire-release-1101`, branch `release-build-1101`, source `546ce89`.
   CPU-only `grimoire-dev:latest`, cores 6-13, no devices, not privileged. Stopped during the
   CLI build; no `RELEASE_BUILD_PASS`. Treat partial binaries as unusable. Re-run both
   production build scripts with `GRIM_TARGETS=intel_gpu_bmg_g31,intel_gpu_bmg_g21`, then
   the official strip step and hashes. Existing oneDNN 3.14 source install is copied there.
3. Packaging recipe is saved in [tools/bench/release_1101](tools/bench/release_1101).
   It was **not executed**. It verifies the exact v1.10.2 base image, identical oneDNN and
   entrypoint, copies newly compiled G31/G21 binaries into an immutable build context and
   writes a provenance manifest. Runtime/compiler/driver versions do not change. Tower
   copies: `bench-1022/assemble-image-1101.sh` and `Dockerfile.runtime-1101`.
4. Validate the candidate image's OWN binaries (`HOSTBIN=0`) on the standard five served
   configurations and each advertised opt-in, with llama-benchy, answer checks and link
   guards. Only then advance main, tag/promote/upload a NEW v1.10.3 release, preserving
   v1.10.2 fallback. Standing user authorization to upload a validated image is recorded in
   `grimoire-release-image-workflow.md`; no new permission question is needed.
5. Wide INT4 verification probe is the next isolated experiment. It compares identical
   production split-K arithmetic at default 128 GRF against the 256-GRF library, and tests
   the narrow-row split-K plan separately. FP64 references, output guards, GS64/128 and
   M=9/14/16 are included, with rotating weight allocations. No engine dispatch change.
   Create Tower worktree from `verify-grf-probe-1101`, copy the combined G31 runtime libraries,
   compile CPU-only on spare cores, then run with the GPU0 guard only after other work ends.
   Its build command must preserve the literal linker rpath `'-Wl,-rpath,$ORIGIN'`.
6. Continue the full roadmap: native block-scaled FP8, mixed-format native GGUF gate/up,
   verification/fusion and portable behavior. Adaptive MTP was already slower on one B70;
   Flash-Next MTP is slow/divergent and remains separate. B60 runtime tests, real TP and P2P
   need suitable hardware. Do not relabel those as complete after publishing an image.

## Evidence saved for tomorrow

Mac `/Users/ianernst/grimoire-work/results-20261101-paused/` contains all 70 Nov-1 logs,
partial regression logs, scripts and source-status/binary-hash snapshots. Earlier paired
results remain in `results-20261031-engine/`; release assembly drafts also exist in
`results-20261101-release/`. Tower originals remain under `grimoire-runs/bench-1022` and
`regress-1101-2021`; snapshots are `bench-1022/paused-1101-snapshots/`.

No model data, generated binaries or log archives were added to git. Source and notes are
backed up as commits; raw evidence exists on both Mac and Tower. Resume only when Ian asks.
