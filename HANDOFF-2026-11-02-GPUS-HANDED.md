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

## Resumed 2026-11-02 19:11: Ian said "resume work"

ComfyUI had exited just before; nothing else was holding a GPU. GPU1's switch was at ~10 million corrected errors,
still off limits. GPU0 AER 0.

**Wide INT4 verify probe: the 47% gain reported at 17:56 was wrong, and there is nothing to build.** The first probe's
kernels shared their SYCL names with `libgrimoire_gemm.so`, and the runtime ran the probe's own 128-register image for
every column, including "256". Its "current plan" numbers match the 128-register column below within 3%. Shipping
kernels are on 256 registers and were never in that comparison. Runs with own kernel names, median us at M=16, 54 cases each, 0 failures:

| Weight | Group | Shipping: 256 registers, current plan | 128, current plan | 128, narrow plan | 256, narrow plan |
|---|---:|---:|---:|---:|---:|
| gate_up 34816 x 5120 | 128 | 197 | 224 | 199 | 188 |
| down 5120 x 17408 | 128 | 112 | 131 | 113 | 132 |
| 10240 x 5120 | 128 | 71 | 85 | 74 | 87 |
| 6144 x 5120 | 128 | 49 | 56 | 51 | 58 |
| head 248320 x 5120 | 128 | 1283 | 1412 | 1257 | 1231 |
| gate_up 34816 x 5120 | 64 | 220 | 337 | 256 | 199 |
| down 5120 x 17408 | 64 | 122 | 155 | 168 | 146 |
| head 248320 x 5120 | 64 | 1418 | 2635 | 1396 | 1357 |

Shipping numbers are from the 256-register run (`logs/probe3-1102.out`). Its second column, the probe's own copy of the
same kernel, matches the library within 1%, which checks the method. The narrow plan is 4-10% faster on the two widest weights and
15-22% slower on the others. Decision: dropped; no engine change.

**Kernel-name collision in production: harmless as far as the evidence goes.** The probe shows the main binary's image wins over
the library's. `native_qk_esgemv<F>` has its only caller in `gemv_decode.cpp`, in the main binary, so GGUF decode gets the
image it was built for and the library's copy is dead code (compiled because `native_qk_esgemv_try` is an inline
function in a header). One diagnostic is void: `GRIMOIRE_I4_SMALLM_256=1` at M <= 8 calls `int4_smallm_impl<1,GS>` from the library
and gets the main binary's 128-register copy. Inferred from the probe and the call sites; not traced at run time. The 10-31
`INVALID_KERNEL_ARGUMENT_INDEX` cause remains unproven.

**Ornith's coding failures are a model mistake, settled.** Same four prompts, two rounds, GGUF native, integration binaries,
`logs/orn-seq-1102.out`, `orn_compare_1102.py`:

| | Pass together | Pass alone | Round 0 = round 1, together | Round 0 = round 1, alone |
|---|---:|---:|---:|---:|
| Option off | 6/8 | 6/8 | 0/4 | 4/4 |
| `GRIMOIRE_MOE_NATIVE_SMALLM=1` | 6/8 | 6/8 | 4/4 | 4/4 |

Alone, prompt 0 fails in every round in both modes (clean Python, skips the sort, 257 tokens), and all 8 alone answers are
byte-identical between option off and on. Together, batching changes which prompt trips, not how many.

**CPU-expert pool: root cause found and fixed on branch `cpu-pool-late-workers-1102`, not in v1.10.3.**
`CpuExperts::run` waited for every worker, including one whose core had gone to another process: 3,225 of 3,347 us
per call. Now late workers miss the job and only workers inside are waited for. `cpu_expert_bench`, mean or median as noted, same output
hash on every row and with one thread:

| Layout | Other load | Old | New |
|---|---|---:|---:|
| Pinned, as Flash-Next on this CPU | one build | median 3,392 us | 137 us |
| Pinned | build + 1 busy core | median 10,439 us | 145 us |
| Pinned | build + 4 busy cores | median 6,388 us | 137 us |
| Pinned | build + all 13 worker cores | mean 6,623 us | 1,381 us |
| Unpinned, as non-hybrid desktops | builds | mean 1,196 us | 188 us |
| Unpinned | builds + 4 busy cores | mean 1,611 us | 226 us |

Thread sanitizer: no reports; the control with relaxed atomics gives 43, so the clean run means something.
Quiet-machine cost and the real engine are still to measure.

## Order from here

All scripts are in `grimoire-runs/bench-1022`, copies in [tools/bench/release_1101](tools/bench/release_1101).
GPU work is one job at a time, GPU0 only.

1. `run_release_server_1102.sh`: running since 19:12. `bin/grimoire` from 17:43 is kept. Must end `RELEASE_BUILD_PASS`.
2. After the build: `run_fn_ab_1102.sh /mnt/storage/isos/grimoire-engine-1101 new-quiet`, closing the Flash-Next row above.
3. Pool branch: engine build running (`build_cxpool_cli_1102.sh`); Flash-Next text identity and seconds under load and quiet, then merge.
4. `assemble-image-1101.sh` (never run; BuildKit 0.37 and image layout already checked), then
   `validate_v1103_1102.sh`: five v1.10.2 configurations plus the three options, plus `conc4,seq4` on three Ornith
   formats. `verify_image_logs_1102.py` prints the before/after table and gates the release.
5. Package as `package_v1102.sh` did, update main (fast-forward from `b0674cd`), publish a NEW v1.10.3 keeping
   v1.10.2 (standing authorization in the release workflow notes), then the profile switch and smoke as on 10-31.
6. Continue [the roadmap](ROADMAP-2026-10-31.md): native block-scaled FP8, mixed GGUF gate/up.
