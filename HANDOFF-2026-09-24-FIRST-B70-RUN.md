# HANDOFF 2026-09-24 — first real run of `main` on the B70s

**Read this first when resuming.** Stopped mid-investigation because Ian needed
the GPUs. Nothing is running; both B70s were healthy (`runtime_status=active`)
when this was written.

## TL;DR

- `main` (`a529127`) **builds and runs on the B70**. Qwen3.8-27B on a SHORT
  prompt is coherent (~25 tok/s plain decode, no MTP/W4A8).
- **Blocker: long-prompt prefill is NONDETERMINISTIC on the card.** Same
  binary, same 5987-token prompt, same card -> different garbage every run
  ("abortionists…", "ollama…", "葭…", fluent-but-unrelated Chinese/Italian).
  It is **not a regression**: the 09-07 build of `390c9d5` does it too
  (one earlier "good" run was luck — see §3). Nothing that uses long prompts
  (PP numbers, llama-benchy pp4096, Ornith, PP2/TP2 speed) can be trusted
  until this is fixed (rule 8).
- Localized so far: `GRIMOIRE_PREFILL_LAYER_LIMIT=1` is **deterministic**;
  the full 64-layer prefill is not. **Next step: find the first
  non-deterministic layer** (§5, step 1).
- Harness bugs that made today's first preflight run lie are **fixed and
  committed** (§2).

## 1. Hardware facts (changed since the docs were written)

| slot | card | Level Zero idx | render node (today) | real host link |
|---|---|---|---|---|
| 03:00.0 | Arc Pro **B70** (`gpu0`) | 0 | renderD128 | Gen4 **x8** |
| 07:00.0 | Arc **B580, 12 GB — never a GRIMOIRE card** | 1 | renderD130 | Gen4 x2 |
| 0b:00.0 | Arc Pro **B70** (`gpu1`, was 35:00.0) | 2 | renderD131 | Gen4 **x2** |
| 00:02.0 | iGPU "Arc(TM) Graphics" | 3 | renderD129 | — |

- The engine's `rank_device()` takes every GPU whose name contains "Arc" —
  that includes the B580 AND the iGPU. With all of `/dev/dri` mounted, rank 1
  of every two-card run landed on the **B580**. `ZE_AFFINITY_MASK` does not
  help: it filters only Level Zero, and SYCL also enumerates OpenCL GPUs.
- B70 #1's root port 00:06.2 is **x4-capable but trained x2**; the B580 sits on
  the neighbouring port 00:06.0, also x2. They may share lanes. `DAY-ONE.md`
  says "OCuLink Gen4 x4" — measured x2. Worth checking with the B580 unplugged:
  `lspci -s 00:06.2 -vv | grep LnkSta`.
- The card-internal switch always reports 2.5GT/s x1; ignore it.

## 2. Harness fixes committed with this handoff

All in `tools/`, same commit as this file:

- `gpunode.sh`: `gpu1` -> **0000:0b:00.0** (was 35:00.0, no longer exists).
  The B580 deliberately has no name.
- `pp2run.sh`, `tp2run.sh`, `serve_pp2.sh`, `serve_tp2.sh`, `nrun.sh`,
  `preflight_b70.sh` (two-card gates): mount **only the two B70 render
  nodes** (resolved by PCI via `gpunode.sh`), no `--privileged`, no full
  `/dev/dri`. Verified: ranks print `GPU 0/2 … 03:00.0` / `GPU 1/2 … 0b:00.0`.
  Ranks talk over AF_UNIX sockets, so `--privileged` was never needed.
  `nrun.sh`'s old mask assumed Level Zero enumerates in render-node order —
  false here (it would have picked the iGPU).
- `preflight_b70.sh`: refuses to run unless it is run FROM
  `/mnt/storage/isos/grimoire-fuse` (tune.sh/b70run.sh always mount that
  checkout, so running it elsewhere tested a different tree), and the
  two-card gates now get tune.sh's environment (bridges + LD_LIBRARY_PATH)
  like the single-card ones.

Why the FIRST preflight today was fake: I had pulled into
`/mnt/user/appdata/GRIMOIRE` while tune.sh ran the stale `grimoire-fuse`
checkout (`qwen-tg-restored`), whose old `b70run.sh` still exited 0 on
failure. Every single-card gate "PASSED" with `No such file or directory`.

Tower checkout state: `grimoire-fuse` is now on **`main`**. The previous
`qwen-tg-restored` build (bin/ + src/*.so, built 09-07) is backed up at
`/mnt/storage/isos/grimoire-fuse-prev-build-qwen-tg-restored-20260924/`; the
stale extra .so files (bestla, fa2, …) were moved to
`/mnt/storage/isos/grimoire-fuse-stale-20260924/`. Nothing was deleted.

## 3. Findings (real B70, `main` a529127 unless noted)

### 3a. Long-prompt prefill is nondeterministic  — BLOCKER

Qwen3.8-27B-MXFP4-GRIMOIRE, `--proj mxfp4 --ctx 8192`, prompt =
`real4k_ascii.txt` (5987 tokens after the chat template):

| run | first tokens |
|---|---|
| main, plain | `abortion abortionist abortionists…` |
| main, W4A8 | fluent Italian unrelated to the prompt |
| main, 117-token prefix | `antry, and the next day, and the next day…` |
| main, 3336-token prefix | `爵 爵 爵…` |
| 09-07 build of 390c9d5 (x3) | Chinese about std::shared_ptr / `葭 葭…` / `AIMD AIMD…` |
| 390c9d5 rebuilt today (x2) | `ollama…` / `ABI ABI…` |

A 22-token prompt was coherent ("The sky appears blue because of … Rayleigh
scattering…"). Short prompts take different kernels: DeltaNet runs the
per-token decode kernel for M<=16, own chunked kernel for 16<M<64, and the
Xe2 chunk-GDN bridge for M>=64 at pos 0.

Ruled out (each tested on the card):
- **not a code regression** — the bisect `390c9d5..a529127` "found"
  `5c46b84`, but the control rebuild of `390c9d5` is also garbage and the
  09-07 binary is garbage 3 of 4 runs. The bisect is INVALID; ignore it.
  (CLAUDE.md: a bisect controls for the commit, not for the machine — and a
  single coherent run is not a "good" verdict.)
- not the compiler: all three binaries say icpx 2026.1.1 (20260724).
- not the bridge .so files: new engine + all-old bridges is still garbage.
- not the GDN bridge: `GRIMOIRE_RAW_GDN_LAYER_LIMIT=0` (own chunked kernel)
  still garbage.
- not the Level Zero V2 adapter: `SYCL_UR_USE_LEVEL_ZERO_V2=0` (legacy)
  still garbage and still varies.
- not host/queue sync: `GRIMOIRE_DEBUG=1` (waits everywhere) still varies
  (first argmax id 1573 vs id 16).
- not only XMX: `GRIMOIRE_BATCHED_PREFILL_NOXMX=1` still varies
  (id 935 vs 1719).
- `GRIMOIRE_PREFILL_LAYER_LIMIT=1`: **identical twice** (id 220, logit
  6.4559). So the variation enters after layer 0.

Model layout: 64 layers, `full_attention_interval=4` -> layers 0,1,2 are
DeltaNet, **3** is the first full-attention layer (then 7, 11, 15…).
Prime suspect: the full-attention prefill path at large M (reads past the
written K/V extent, uninitialized padding, or a race inside the kernel).
Also remember `grimoire-gdn-chunk-padding` (M+63 zero-filled padding).

The same nondeterminism may explain 3c–3f below — re-check them after the fix.

### 3b. Ornith long prompt -> GPU hang / DEVICE_LOST

`Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE`, 5987-token prompt, Ornith flags:
`generation failed: … UR_RESULT_ERROR_DEVICE_LOST`, then the process sat
until the 900 s limit. Kernel: ccs engine reset + devcoredump at 15:26:08
("LR job cleanup", preempt timeout 640 ms). Coredump saved:
`/root/xe-coredump-03-152608.txt` on the Tower.

### 3c. Muse architecture segfaults (mini fixture)

`muse` bf16/fp8 cells: SIGSEGV (signal 11, host side) right after load, on
the first forward. After such a crash the NEXT GPU job often hung until its
timeout (copy-engine reset at the kill) — this cascade is what made the gates
slow. The fixture has head_dim 16; `grimoire_xe2_dflash_paged_f16` refuses
anything but head_dim 128 cleanly (returns 1), so the fault is later.
Repro model staged at `grimoire-fuse/.muse-repro/` (vocab 128). Backtrace
command, not yet run:

    GPU=gpu0 LIM=300 bash tools/tune.sh musegdb \
      /opt/intel/oneapi/debugger/2026.1/opt/debugger/bin/gdb-oneapi -batch \
      -ex run -ex bt --args /grimoire/bin/test_model_matrix /grimoire/.muse-repro bf16 128

The real Muse checkpoint (head_dim 128) was NOT run yet.

### 3d. Hybrid MTP speculation changes the output

`test_spec_e2e`: `hybrid+mtp` bf16 AND fp8, K=1,2,3,5 -> `CHANGED THE
OUTPUT` (fp8 falls into `18 6 18 6…`). DAY-ONE.md calls this a stop
condition for the Qwen MTP recipe. dense+mtp / moe+mtp say "identical" but
accepted **0 of 65** drafts, so their rollback path never ran. Checked and
NOT the cause: the verify batch (pos>0, M<=6) never takes the GDN bridge
path; it uses the per-token kernel that does write `spec_dn_steps`.

### 3e. gemma4: single process != PP2 == TP2

gemma4 bf16/fp8 and gemma4-hd512: PP2 and TP2 agree token-for-token with
each other and both differ from single-process (from token 3). Suspect the
single-process batched prefill (see 3a).

### 3f. parallel-ffn hang

`test_parallel_e2e` hung in the `parallel-ffn bf16` row until 900 s.

### What passed on the card (real, not the fake first run)

`test_k2_kernels` (every kernel vs host reference, 0 failures),
`test_k2_e2e`; `test_model_matrix` rows dense/moe/hybrid/k2-horizon x all 7
formats; `test_parallel_e2e` PP2/TP2 == single for dense, moe, k2, hybrid.
Not run yet on the card: gemma4_prefill, qwen4_exp_e2e, nvfp4_e2e,
prefix_reuse, batch_prefix, batch_decode, batch_spec, batch_parallel,
scheduler, pp_server.

### Numbers (timing only — see rule 8)

- Qwen short prompt, main, plain decode: 96 tokens in 3.82 s total.
- 09-07 build, 5987-token prompt: pp 1375.6 tok/s, tg 24.2 tok/s — but its
  output is nondeterministic, so this is kernel timing, not a result.
- No MTP / W4A8 / llama-benchy / PP2 / TP2 speed numbers yet.

## 4. Traps found today

- **tune.sh applies NO recipe.** `docker/recipes/*.env` is applied by
  `grimoire_recipe_entry.sh`, which only exists in the `grimoire:b70-native`
  image. Through tune.sh you must pass W4A8/MTP/etc. in `EXTRA_ENV`.
  `serve_qwen.sh` sets its own (includes `MTP_EXACT_VERIFY=1`).
- QWEN-RECIPE.md says do NOT set `MTP_EXACT_VERIFY`; serve_qwen.sh sets it.
  Measure both once 3a is fixed.
- A crash with GPU work in flight -> engine reset; the next job may hang.
  The card survived ~10 engine resets today without falling off the bus.
- `docker cp` from a kept (exited) gate container recovers its `/tmp`
  (per-cell logs of test_model_matrix live there).

## 5. Next steps, in order

1. **Find the first nondeterministic layer** (each run ~25 s):

       cd /mnt/storage/isos/grimoire-fuse
       for L in 3 4; do for i in 1 2; do
         GPU=gpu0 LIM=600 EXTRA_ENV="GRIMOIRE_DEBUG=1
       GRIMOIRE_PREFILL_LAYER_LIMIT=$L" bash tools/tune.sh lim$L-$i /grimoire/bin/grimoire \
           -m /models/Qwen3.8-27B-MXFP4-GRIMOIRE --proj mxfp4 --ctx 8192 \
           -p "$(cat real4k_ascii.txt)" -n 4
         grep -m1 "argmax ->" /tmp/grim-lim$L-$i.log
       done; done

   Same argmax id+logit on both runs = deterministic up to that layer.
   If 3 is stable and 4 is not -> full-attention prefill (layer 3). Then
   bisect inside the layer (attention bridge vs own flash kernel, K/V write,
   padding). Always run at least TWICE — one run proves nothing.
2. Fix 3a, then re-run: `SKIP_BUILD=1 tools/preflight_b70.sh
   /models/Qwen3.8-27B-MXFP4-GRIMOIRE` (from grimoire-fuse, `PROJ=mxfp4`).
3. Then speed: `/root/b70_bench.sh` and `/root/b70_bench2.sh` on the Tower
   are ready (Qwen plain vs W4A8 vs W4A8+MTP vs +EXACT_VERIFY on the same
   prompt, Ornith 1-card / PP2 / TP2, llama-benchy, remaining gates).
   b70_bench2.sh takes the PID of b70_bench.sh as $1 (use any dead PID to
   run it alone).
4. Muse backtrace (3c), parallel-ffn hang (3f), then re-check 3d/3e.

## 6. Where things are (Tower)

- Logs: `/root/preflight-b70-20260924-1438.log`, `/root/b70-bench-20260924-1511/`,
  `/root/b70-diag-*`, `/root/b70-diag2-*`, `/root/b70-repeat-1603/`,
  `/root/b70-race-1605/`, `/root/b70-race2-1607/`, `/root/bisect-*.log`,
  `/root/matrix-tmp/` (model-matrix per-cell logs), `/tmp/grim-*.log`.
- Scripts: `/root/b70_bench.sh`, `/root/b70_bench2.sh`, `/root/b70_diag.sh`,
  `/root/b70_diag2.sh`, `/root/b70_repeat.sh`, `/root/b70_race.sh`,
  `/root/b70_race2.sh`, `/root/bisect_step.sh`.
- Untracked helpers inside grimoire-fuse (not committed): `.old-build/`
  (09-07 binary + its bridges), `.bisect-bins/`, `.muse-repro/`.
- Bisect worktree: `/mnt/storage/isos/grimoire-bisect` (bisect reset; safe to
  `git worktree remove`).
- `/mnt/user/appdata/GRIMOIRE` also has a clean `main` build (same commit);
  it is NOT what tune.sh runs.
