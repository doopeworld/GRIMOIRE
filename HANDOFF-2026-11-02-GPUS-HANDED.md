# GPUs handed to Ian: full Grimoire engine, 2026-11-02 17:52 CET

Ian resumed the work at 17:01 ("can you resume from here or not?", then "so?"), then at 17:52 wrote
"i need to use both gpus now". **Nothing restarts until Ian says so.** Read
[HANDOFF-2026-11-01-PAUSED.md](HANDOFF-2026-11-01-PAUSED.md) for the older checkpoints, this file for what changed.

## State at the hand-over

- `COMFYUI-BETA` started 17:52:45 and is Ian's. Never stop it. No Grimoire container or GPU holder,
  both locks free, watcher stopped, no book server, no queue.
- GPU0 `03:00.0` = renderD128: AER 0/0/0 after all of today's runs, no xe errors. The Tower rebooted at 16:40.
- GPU1 `07:00.0` = renderD130 stays off limits: switch `05:00.0` logged 536,375 corrected errors in the first
  ~10 idle minutes after boot (~900/s), root port 152, 0 uncorrectable.
- The release build was **stopped from outside**: docker events show SIGTERM 17:52:36, SIGKILL 17:52:46
  (rc=137), ComfyUI start 17:52:45. Our own `docker pause` arrived at 17:52:53 and found no container.
  So it was not a compile failure. The most likely cause is Ian clearing the way for ComfyUI; that part is an inference.
- `grimoire-b70:latest` is still v1.10.2 `b14de3b48ba0`. No v1.10.3 image exists. GitHub main is `b0674cd`.

## Results today

Full 44-case regression, combined build, `regress-1102-1701`, `logs/regress-1102.out`:
**44 of 44 answer texts are identical to `regress-1030-1300`** using the suite's own text filter, 0 empty,
Sherlock n=24 identical to its reference, exit 0. `docker inspect` confirmed the runs mounted the
`grimoire-engine-1101` bin and tools. This is identity with the previous build. It is not 44 behavioral
passes: the new options are off in this suite, and the existing `text vs plain: differs` speculation lines remain.

Flash-Next short case, one slow number, explained except for a final quiet run:

| Run | Binaries | CPU build running | `fn-s` seconds | CPU experts, ms per call |
|---|---|---|---:|---:|
| 10-29 14:10 | older | no | 3.53 | 0.384 |
| 10-29 12:48 | older | image build | 14.69 | 2.994 |
| 11-02 suite | combined | release build | 16.67 | 3.472 |
| 11-02 warm, twice | combined | release build | 17.23, 16.61 | 3.670, 3.581 |
| 11-02 | v1.10.2 | release build | 16.63 | 3.579 |

Same text in all 11-02 runs. Old binaries are equally slow under the same load, so **this is not a regression in the
combined build**. A cold page cache was tested and rejected. The pool is pinned to CPUs 1-13, and the compile
was on 6-13. Still unmeasured: the combined build with no compile running (`run_fn_ab_1102.sh`).
Portability finding for the roadmap: one busy core costs this path about 5x, which an ordinary desktop will hit.

Wide INT4 verify probe (`tools/bench/onednn/i4wide_grf_probe.cpp`, first run on the card):
54 cases, 0 failures, FP64 reference within 5e-7, output guards intact, 6 s, no driver errors.
`logs/vgrf-all-1102.log`. Median whole-call microseconds at M=16:

| Weight | Group | Current plan | Narrow plan (the M <= 8 plan) | Change |
|---|---:|---:|---:|---:|
| gate_up 34816 x 5120 | 128 | 228 | 197 | -14% |
| down 5120 x 17408 | 128 | 128 | 109 | -15% |
| 10240 x 5120 | 128 | 84 | 74 | -12% |
| 6144 x 5120 | 128 | 56 | 51 | -8% |
| head 248320 x 5120 | 128 | 1400 | 1251 | -11% |
| gate_up 34816 x 5120 | 64 | 341 | 252 | -26% |
| down 5120 x 17408 | 64 | 156 | 165 | +6% |
| head 248320 x 5120 | 64 | 2576 | 1371 | -47% |

Below ~300 outputs the narrow plan costs about 1.5 us per call at group 64. These are isolated kernel numbers. The
engine is unchanged and no real-model gain is measured yet. M <= 8 already uses this plan, so single-user MTP
at K=6 (7 rows) cannot gain. Multi-user verify and K >= 8 reach this path.

**Do not cite that probe's "256 vs 128 registers" columns.** The probe and `libgrimoire_gemm.so` compile
`int4_smallm_fdq_impl<2,GS>` under one SYCL kernel name, so all three columns may have run one image.
The plan comparison stands, because the plan is a run-time argument, but which register file produced it is unknown.
`bin/i4wide_grf_probe2` is built with `-Dint4_smallm_fdq_impl=..._probe128` so its kernels have their own names.
It has NOT been run.

The same collision exists in production: `kernel_names_1102.sh` finds `native_qk_esgemv<F>` for six GGUF
formats, and the diagnostic `int4_smallm_impl<1,GS>`, in both the server and the 256-GRF library
(`native_smallm.hpp` includes `native_qk_esimd.hpp`). Which image serves GGUF native decode is unknown, and its
speed under each is unmeasured. Hypothesis to check, not a finding: the 10-31 `INVALID_KERNEL_ARGUMENT_INDEX`
on `native_smallm_impl` may have been this same collision, since its cause was never proven.

Native grouped experts, the 5/8 concurrent coding result: the failing answers are **clean, valid Python that skips
sorting** (240-251 tokens against 288-307 for passing ones), not corrupted text. Option-off runs fail the
same way, and with the option off the same user's answer changes between rounds. This points at a near-tie model
mistake rather than kernel corruption, but it is not settled. `seq4` (same prompts, one at a time) is written, not run.

## Exact resume order, after Ian says resume

Check `docker ps` and `lspci` first, and leave ComfyUI to Ian. All scripts are in `grimoire-runs/bench-1022`, copies
in [tools/bench/release_1101](tools/bench/release_1101). GPU work is one job at a time, GPU0 only.

1. `run_release_server_1102.sh`: CPU-only, finishes the stopped build. `bin/grimoire` from 17:43 is kept; the
   server took more than the 51 minutes it had. Partial binaries are not a release. Must end `RELEASE_BUILD_PASS`.
2. Meanwhile, GPU0: `GRIM_TREE=/mnt/storage/isos/grimoire-verify-grf-1102 LIM=900 TAILN=70 bash tools/g0run.sh vgrf2 /grimoire/bin/i4wide_grf_probe2`
   from `grimoire-fuse`, holding `gpu0.lock`. It answers the register question and says where the narrow plan can live.
3. After the build: `run_fn_ab_1102.sh /mnt/storage/isos/grimoire-engine-1101 new-quiet`, closing the Flash-Next row above.
4. `assemble-image-1101.sh` (never run; BuildKit 0.37 and image layout already checked), then
   `validate_v1103_1102.sh`: five v1.10.2 configurations plus the three options, plus `conc4,seq4` on three Ornith
   formats. `verify_image_logs_1102.py` prints the before/after table and gates the release.
5. Package as `package_v1102.sh` did, update main (fast-forward from `b0674cd`), publish a NEW v1.10.3 keeping
   v1.10.2 (standing authorization in the release workflow notes), then the profile switch and smoke as on 10-31.
6. Narrow plan on the real model: single opt-in switch, Qwen GPTQ MTP at 2/4/8 users and K=8, answers then speed.
   Keep it out of v1.10.3, whose binaries are already validated by then.
7. Continue [the roadmap](ROADMAP-2026-10-31.md): GGUF kernel-name collision, CPU-expert pool behavior on a busy
   host, native block-scaled FP8, mixed GGUF gate/up.
