# 2026-10-31: resume the interrupted v1.10.2 release

## Recovered checkpoint

Ian asked Codex to resume Claude's exact interrupted work. The matching Claude session is
`88f1d983-7bd4-487b-8e63-525d50b62a07`, last tool result 2026-10-30 12:01:35 UTC. Its last two
tools read the batched expert dispatch in `src/grimoire.cpp`; no kernel change was in progress.
The Tower checkout was clean for tracked files, branch `bd-main`, HEAD
`6e170a3e287ad452e4ecbdaaa2318d683dd6d162`, also GitHub main when checked on 10-31.
Preserve its untracked files.

The two background jobs finished after Claude's limit:

- `regress_1030.sh`: **44 runs completed**, `ALL DONE`, `REGRESS_EXIT 0`.
  Results `/mnt/storage/isos/grimoire-runs/regress-1030-1300`; comparison against
  `regress-1029-1410`: **42 identical texts, 2 Muse texts differ** (`mu-s`, `z-mumx`).
  Do not describe this as 44 behavioral passes or universal token parity.
- `grimoire-b70:v1.10.2-rc1`: built successfully on 10-30 at 14:13 Tower time.
  Image ID `sha256:b14de3b48ba009c1ad6eb2cdf9af6684438825f172a34e1fb76553e037769bbd`,
  2,852,324,307 bytes uncompressed. `/tmp/img-1102rc1.log` disappeared at the 10-31 reboot;
  the image and persistent regression logs survived. The build used the default G31 + G21
  targets. B60/B580 hardware was not tested here.

## Hardware and run discipline

10-31 boot about 07:41 CET: gpu0 B70 `03:00.0` is `renderD128`, iGPU `renderD129`, gpu1 B70
`07:00.0` is `renderD130`. All resumed workloads use only gpu0 through the established lock
and guarded launchers. No GPU workload was running at resume.

GPU0's PCIe path has zero correctable/nonfatal/fatal AER. GPU1 is still paused: its switch
logs hundreds of corrected receive errors per second even while idle. Do not resume
two-card benchmarks from a generic request to continue; see the earlier gpu1 guard stop.
Each server is stopped gracefully immediately after its clients finish.

## Packaged image validation

All tests use `IMG=grimoire-b70:v1.10.2-rc1 HOSTBIN=0`, so no host executable, library or
entrypoint is substituted. Harness `/mnt/storage/isos/grimoire-runs/bench-1022/dual.sh`, G0;
llama-benchy 0.4.0, pp4096/tg128, eight sequence slots. Qwen: MTP K=6, draft vocab 65536,
ctx16384, two measured runs. Ornith: no MTP, ctx32000, one measured run.

| Packaged image workload | pp4096, one user | tg128, one user | Concurrent tg128 total | Answer checks |
|---|---:|---:|---:|---|
| Qwen GPTQ-Int4 + MTP | 2020.07 | 67.08 | 180.47 (4 users) | coherence; coding 2/2; concurrent coding 8/8 |
| Qwen UD-Q4_K_XL, native + MTP | 1972.94 | 59.41 | 141.55 (4 users) | coherence; coding 2/2; concurrent coding 8/8 |
| Ornith Q4_K_M, native | 7258.69 | 155.19 | 133.78 (8 users) | Paris; coherence |
| Ornith Q4_K_M, converted to MXFP4 | 8752.84 | 175.14 | 378.44 (8 users) | Paris; coherence |
| Ornith safetensors, MXFP4 | 8803.61 | 174.72 | 378.93 (8 users) | Paris; coherence |

The fixed coding prompt has 2153 tokens, produces a complete tested answer, and gives
127.9 tok/s on GPTQ versus 96.2 tok/s on native GGUF. These are not tg128 measurements.
llama-benchy's book text varies across requests; MTP acceptance and timing vary with it.

Full logs: `bench-1022/logs/r1031-img-{q38,orn,qg-mtp,og-native,og-mxfp4}.*`.
Mac evidence and scripts: `/Users/ianernst/grimoire-work/results-20261031/`.
The evidence includes all 44 old/new regression logs, not just their comparison summary.

The two changed Muse cases were each repeated twice with the packaged CLI through
`g0run.sh`, `HOSTBIN=0`, `GRIM_IMAGE=grimoire-b70:v1.10.2-rc1`, `GRIMOIRE_PRINT_IDS=1`.
INT4: all 64 generated IDs repeat exactly; MXFP4: all 40 repeat exactly. The image's INT4
wording matches 10-30; its MXFP4 wording matches 10-29. Thus cross-build text identity is not
claimed. Both texts discuss the requested Rayleigh scattering; these length-capped checks
are not a full evaluation of answer quality. Script `bench-1022/check_muse_image_1031.sh`,
summary `logs/r1031-muse-image.out`, logs `logs/r1031-muse-{int4,mxfp4}-{1,2}.log`.

## Native Ornith small-batch expert path: next implementation

The slowdown is reproduced in both the 10-30 host build and the packaged image: eight users
are slower in total than one in `--proj native`. This is an existing performance limit.
The fast MXFP4 path is unaffected (about 379 tok/s at eight users).

The interrupted source investigation is now complete:

- `prefill()` in `src/grimoire.cpp` selects MXFP4 grouped small-M kernels for `1 < M < 32`.
  Their matrix-unit kernel is in `src/gemm_fast.cpp`, `moe_mxfp4_smallm_impl`.
- The native grouped dispatch is inside the later **`M >= 32`** branch.
  Native batches of 2/4/8 tokens reach `launch_moe_gate_up_batched` and
  `launch_moe_down_batched`, which reread weights for each token.
- This Ornith file's gate/up experts are **Q4_K** (`2048 x 512 x 256`), and down experts
  are **Q6_K** (`512 x 2048 x 256`). A Q4-only implementation cannot fix the pair.

Use the existing device routing/count/offset tables for native small-M too. Add expert-table
indexing to the native DPAS dequantization already validated in `src/native_smallm.hpp`,
cap the grid to the touched-expert bound `min(moe_tab_tiles, M * top_k)`, then use the existing
SwiGLU and unpermute operations. Preserve a fallback and exact-verify behavior. Merely enabling
the large 256-row grouped GEMM below 32 wastes its tile on padding.

Before enabling it: a host oracle for Q4_K and Q6_K with different per-expert row counts,
shared experts and padded table entries; actual answers at 1/2/4/8 users; then a controlled
speed comparison with 133.78 tok/s at eight users. No fix or improvement is claimed yet.

## Release and container status

Published as an experimental prerelease:
https://github.com/doopeworld/GRIMOIRE/releases/tag/v1.10.2

- `grimoire-b70-v1.10.2.tar.gz`: 812,335,998 bytes, gzip integrity PASS.
- SHA-256: `cb96f0e810f0661f3f4fb1d2efd1de32e58dd43be995db372bb78538e14e8750`.
- GitHub's asset API reports the same byte count and digest, state `uploaded`; the download
  URL returns HTTP 200. The tag targets source `6e170a3`, not this later documentation commit.
- Local final tag and `latest` point at image `b14de3b48ba0`; fallback `v1.10.1` remains.
- Two validated single-card profiles (`GRIMOIRE-QWEN3.8`, `GRIMOIRE-ORNITH`) were recreated from
  their Unraid templates on v1.10.2. Original XML and stopped-container inspections were backed
  up under `bench-1003/templates-backup/2026-10-31-v1102/` first.
- Added `GRIMOIRE-QWEN3.8-GGUF` on port **6894** from the Qwen template: native UD-Q4_K_XL,
  MTP K6, draft vocab 65536, eight slots, ctx16384, GPU0 bound by PCI slot. It uses the
  existing file `/mnt/models/unsloth-Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf`.
- All three profiles passed startup, the capital answer (`Paris`), a 128-token generation,
  graceful stop and unchanged GPU0 AER. Summary `logs/r1031-profile-smoke.out` ends
  `PROFILE_SMOKE_PASS`. All are left stopped. Two-card profiles were not retested or changed.

The visible interrupted work is complete: regression reviewed, packaged image validated and
published, README served table updated, GGUF profile added and checked, native-MoE investigation
recorded. No engine source was changed in this resumption. The next engine implementation is
the native small-M expert path above; it is not part of the v1.10.2 release.
