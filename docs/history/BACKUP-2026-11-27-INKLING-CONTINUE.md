# FULL BACKUP — Inkling 2 Mini on GRIMOIRE: everything needed to continue (2026-11-27)

Written for a fresh session (Claude or any assistant) with no memory of 11-23 .. 11-26. Read this file, then
[TODO-2026-11-27.md](TODO-2026-11-27.md) (the work list), then [HANDOFF-2026-11-26-DFLASH2.md](HANDOFF-2026-11-26-DFLASH2.md)
(today's detail) and [HANDOFF-2026-11-24-INKLING.md](HANDOFF-2026-11-24-INKLING.md) (the engine's history and its
dead ends).

## 0. Rules from Ian (do not break these)

- **Only measured / read facts.** Unknown = say so. Do the task assigned, nothing else.
- **Short messages**: 2-4 lines, tables for results, report when finished.
- **The B580 (0b:00.0) is ONLY for ComfyUI.** GRIMOIRE work = the two B70s (03:00.0 = gpu0, 07:00.0 = gpu1).
- **No new containers on the Tower without his OK** (the test scripts below start `--rm` / named test containers
  and remove them; Ian started that kind of work himself on 11-26, but ask before anything long-running).
- **Never kill a container with GPU work in flight; never re-run a workload that dropped a GPU without his go.**
  Never set `SYCL_UR_USE_LEVEL_ZERO_V2=0`: it gives DEVICE_LOST on the Inkling engine.
- **Publish an image / release only when Ian says yes** (he said yes to v1.10.14 on 11-26).
- **Check the GPUs are free before testing**: `docker ps`, and `/proc/*/fd` holders of renderD129 / renderD130.
  Ian's GRIMOIRE-DUAL-INKLING2 container holds BOTH B70s while it runs (his Open WebUI chats use it).
- Tower unreachable from the Mac -> first `ping 192.168.8.1`, `mullvad status`, `mullvad lan get` (Mullvad blocks the LAN).
- Mac shell is bash 3.2: no `$(cat <<EOF)`; write multi-line scripts to files or use `ssh root@192.168.8.225 'bash -s' <<'EOF'`.
- Commit messages: list **Findings** and **Next steps**; end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## 1. State on 2026-11-27 (all measured / read, nothing assumed)

| item | state |
|---|---|
| Release | **v1.10.14 PUBLISHED** (prerelease tag at 58e22ad): https://github.com/doopeworld/GRIMOIRE/releases/tag/v1.10.14 ; asset `grimoire-b70-v1.10.14.tar.gz` 1,908,118,213 B, sha256 `3429fbcfcbe95998a5c7f423248fbe7d6a2f870b2b7a369cf340a348339b8ca9` = GitHub's digest |
| Code | branch `inkling-df2-1126`; GitHub `main` = `inkling-df2-1126` (this commit's parent chain: bc4254b is the last commit before this backup). Local worktree `~/grimoire-work/grimoire-inkling-1123` |
| Tower repo | `/mnt/storage/isos/grimoire-fuse`, branch `bd-main`, `origin` = `git@github.com:doopeworld/GRIMOIRE.git`; pushes of `inkling-df2-1126` and `inkling-df2-1126:main` go from there (the Mac pushes the branch too) |
| Image | `grimoire-b70:v1.10.14` (e158126f9559) = `v1.10.13` + `inkling2_gpu` + `libink_xexp.so` of 2fd3605, G31 + G21; also `v1.10.14-rc1` (f92f54d5, identical bins) |
| Ian's container | `GRIMOIRE-DUAL-INKLING2` (template `my-GRIMOIRE-DUAL-INKLING2.xml`), image v1.10.14, port 6899, `INK_CTX=262144`, `INK_DRAFT_TAU=0.25`, `/mnt/storage/Models` at `/models`, host network. Open WebUI connection: `http://localhost:6899/v1` (11-26 it had three slashes) |
| At the time of writing | GRIMOIRE-DUAL-INKLING2 and open-webui are **Exited (0)** (stopped ~14 h ago); nothing else runs; GPUs idle |
| Tower GPUs | 03:00.0 = renderD129, 07:00.0 = renderD130, B580 0b:00.0 = renderD131, iGPU renderD128 (after the 11-26 16:54 boot; resolve by `/dev/dri/by-path`, addresses move on reboot) |
| Models | `/mnt/storage/Models/Inkling-2-Mini` (thinkingmachines @7a67ae2), `Inkling-2-Mini-Uncensored` (OpenUncensored @4e5f06d, what Ian runs and what every number below used), `Inkling-2-Mini-DFlash2` (incoai @4f9088ea, model.safetensors sha256 `217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee`) |

## 2. What the engine is (files)

All in `tools/` (include/ and src/ are the shared GRIMOIRE tree; the engine builds standalone):

| file | what |
|---|---|
| `inkling2_gpu.cpp` (~4,100 lines) | the engine: loader, visits loop `Engine::run(M)`, decode graph, two-card TP with host-mediated exchanges, pipeline mode `pp`, prompt path, `serve` mode (Unix socket line protocol), test modes (`check`, `bench`, `mtp`, `tp`, `tpmtp`, `pp`) |
| `inkling2_mtp.inc` | DSpark drafter, verify (`experts_small`, `commit_verify`), `mtp_step` / `mtp_step_s` (exact speculative sampling), top-k / candidates |
| `inkling2_df2.inc` | **DFlash2 drafter** (new 11-26): `load_drafter` (auto-detect), `load_df2`, `f2_block` (draft), `f2_commit` (context K/V), `f2_quant`, `f2_mm`, `f2_embed` |
| `inkling2_moe_gemm.cpp` | prompt expert GEMM (InkMoeG), DPAS attention (InkAttD), `libink_moe.so` (256 GRF) |
| `inkling2_xexp.cpp` | ESIMD kernels: verify experts `InkMoeSmE` (`ink_moe_small_esimd`), prompt experts, gate blend; `libink_xexp.so` |
| `inkling2_server.py` | OpenAI HTTP front end (chat template, SSE, `reasoning_content`) |
| `grimoire_inkling.sh` | image entrypoint `grimoire-inkling` (engine + front end) |
| `bench_ink_sm.cpp` | microbenchmark of the verify expert kernels |
| `inkling-tower/` | **copies of the Tower-only scripts** (section 6): run / A/B / validate / bench / release build / Dockerfile / test prompts |

`src/gemv_decode.cpp` (built as `libink_gd.so`) = GRIMOIRE's INT8 small-M GEMV used for verify dense layers; `libgrimoire_gemm.so`
= GRIMOIRE's GEMM / flash libs. `libink_moe.so`, `libink_gd.so`, `libgrimoire_gemm.so` in the v1.10.14 image are the
v1.10.12 builds (their sources did not change; `ink-rel-1124b/src/bin/`).

### 2.1 How a decode step runs (two B70, 8-row verify)

1. DFlash2 `f2_block` drafts 7 tokens (5 INT8 layers over [anchor, mask x7], top-16 candidates from the INT8 target head,
   edge scores, walk) ~2.1 ms. DSpark = `draft_block` (kept, `INK_DFLASH2=0`).
2. `run_seg(M)` verifies M <= 8 rows through 48 visits (graph chains split at the 96 host exchanges a step; each card
   holds ALL experts, used experts split by parity).
3. `commit_verify` + `draft_commit` (`f2_commit` for DFlash2) store the accepted rows' ring state and drafter context.
4. Sampling: exact speculative sampling on the verify rows (`mtp_step_s`); `INK_DRAFT_TAU` (0.25) stops drafting when the
   product of q over the block falls below it (DFlash2 q = softmax(score / T) over the 16 candidates).

## 3. DFlash2 drafter: facts to keep

- Config (`Inkling-2-Mini-DFlash2/config.json`): 5 layers, hidden 2304, 32 q / 8 KV heads of 128, inter 6912, all layers
  sliding window 2048, rope 1e4, block 8, mask token 200056, selector top-16 rank 256, `selector_softcap_mode
  after_correction`, taps 5 / 23 / 41 / 48 (= TAP0..TAP3, the same taps DSpark uses), conv 2 taps group 16 (`kernel_projection` 576 = 2 sides x 2 taps x 144 groups).
- The anchor row is `embed_norm(embed[token])`; mask rows are `embed_norm(embed[mask])`. Context K/V =
  `k_norm(rope(k_proj(hidden_norm(fc(taps)))))`, V unnormalised. Candidates come from the TARGET head over the 200058 real
  tokens; the selector softcap (30) is applied after adding the edge correction.
- Not bit-exact to PyTorch (INT8 layers, own kernels); correctness was checked by: greedy output = plain decode
  (one card chat1 first 200 of 384 tokens, code2 all 256-384), forced-correct / forced-wrong draft checks in `mtp` mode,
  acceptance numbers close to the model card's (code 7.1-7.7 tok/step vs 4.8-5.0 on SGLang's mixed prompts).
- Env switches (all in `inkling2_df2.inc`): `INK_DFLASH2` (dir / `0`), `INK_F2_I4=1` (INT4 layers, accepts less),
  `INK_F2_BF16=1` (BF16 layers, +1 ms/step), `INK_F2_QTRUNC=1` (top-k/top-p on q, no gain), `INK_F2_GTAU` (greedy verify width).

## 4. Measurements (two B70 unless noted; Inkling-2-Mini-Uncensored; BetterBench = Ian's speed tool, quick = 5 passes, ±6%)

### 4.1 Decode
| test | DSpark | DFlash2 |
|---|---|---|
| BetterBench quick weighted, host binary, same session 11-26 | 139.7 (2.90 tok/step, 22.5 ms/step) | **179.2** (3.62, 22.0) |
| BetterBench quick weighted, on the images (v1.10.12 vs v1.10.14, 262K, Ian's template layout) | 141.8 | **173.5** |
| BetterBench concurrency 1/2/4/8 total tok/s (requests served one at a time) | 124.9 / 123.5 / 125.9 / 121.9 | 157.0 / 154.7 / 156.6 / 153.8 |
| greedy `tpmtp` code2 / chat1 | 251 / 136.5 | **303.8 / 236.0** (7.37 / 5.63 tok/step, ~24 ms/step) |
| one B70 greedy `mtp` code2 / chat1 | 177 / 88.3 | **212.8 / 148.5** (7.08 / 4.79) |
| plain decode (no speculation) two B70 / one B70 | 88-94 / 73 | same |
| memory per card at 262,144 ctx | 30.26 GiB | 26.95 GiB |

Per-category BetterBench (DFlash2, tok/s | tokens per step): chat 173.7 | 3.06, code 159.1 | 4.10, file_edit 180.6 | 4.49,
json 160.2 | 4.41, math 154.5 | 4.75, prose 155.0 | 2.83, reasoning 159.6 | 2.94, summarization 176.6 | 4.36.

### 4.2 Prompt (prefill) — measured 11-24/25 on the v1.10.11/12 engine; v1.10.14 does not touch the prompt path
| setup | tok/s |
|---|---|
| server (tensor parallel), BetterBench 2K / 8K | 2,870 / 3,609 |
| server, prompt of 27.8K / 110.7K / 216.4K | 3,573 / 2,927 / 2,334 |
| engine pipeline mode `pp`, 4K (chunk 2048) / 16K / 32K (chunk 4096) | 4,288 / 5,475 / 5,557 |
| one B70, 4K / 16K | 3,816 / 3,521 |

Decode after a long prompt (v1.10.12): 129.2 / 131.5 / 87.8 tok/s after 27.8K / 110.7K / 216.4K.

### 4.3 Where a decode step goes (two B70, ~22-24 ms)
| part | ms | measured by |
|---|---|---|
| draft | ~2.1 | `INK_MTP_TIME=1` |
| verify experts (`InkMoeSmE`, ~13 of 25 distinct experts a card, ~390 GB/s) | ~9.6 | ablation `INK_ABL=1` (wrong output, timing only; 12.0 ms verify without them vs 21.6) |
| INT8 small-M dense | ~2.2 | `INK_ABL=2` |
| rest (~1000 small kernels, 96 exchanges, attention) | ~9 | remainder |

### 4.4 Tried on 11-26 and NOT kept
| change | result |
|---|---|
| `INK_DRAFT_TAU` 0.1 / 0.25 / 0.5 | 169.3 / **179.2** / 177.7 |
| q truncation to top-k/top-p (`INK_F2_QTRUNC=1`) | 178.3 (no gain) |
| argmax drafts under sampling (`INK_GREEDY_DRAFTS=1`) | 153.2 |
| INT4 drafter | chat1 5.63 -> 4.61 tok/step; code same |
| BF16 drafter | same acceptance, 24.9 vs 23.9 ms/step |
| verify expert K split `INK_SM_KSG` / `KSD` (2/1 two cards) | bench 148.2 -> 142.6 us/visit but engine 24.1-24.4 -> 25.2-25.3 ms/step: opt-in only |
| `INK_SM_TNT` 4 / 12, `INK_SM_PF2` | no change (23.8-24.5) |
| bench_ink_sm one card K split 2/1 | much worse (281.8 -> 356.8 us) |

Earlier dead ends (11-24, do not repeat): see HANDOFF-2026-11-24-INKLING.md ("Do not repeat").

## 5. The work list (full version: TODO-2026-11-27.md)

1. **Verify expert kernels 390 -> ~550 GB/s** (~-3 ms/step, ~+15%): tiled expert layout read by decode + prompt + verify kernels (no room for a 2nd copy).
2. **Several conversations at once** (c2-c8): per-conversation slots; today requests run one at a time.
3. **INT8 small-M dense** (~2.2 ms).
4. **Small kernels + 96 exchanges** (~9 ms).
5. **More tokens a step** on chat / prose / reasoning (2.8-3.1): draft trees from the lattice, adaptive window.
6. **Prompt speed**: urakozz (friend, public engine github.com/urakozz/intel-arc-inference-server, main 6c0fd33 on 11-26)
   reports **6,200 tok/s on a 64K prompt, pipeline parallel, INT8, two B70** (Ian 11-26/27, second-hand number).
   Ours: pipeline `pp` 5,557 at 32K (64K not measured), server TP 3,573. First measure `pp` at 64K, then put pipeline
   prefill in the server — CHECK the weight/KV layout difference first (pp keeps full dense weights per card; the TP
   server shards them), and the memory at 262K.
7. **1M context** (urakozz reaches 1,048,576; Inkling model_max_length 1048576): ours 262,144 = 26.95 GiB a card;
   computed KV at 1M = 16 GiB a card as built -> INT8 KV (`kv8_scheme.h` in his repo) + `INK_NO_EPREP=1` is the route (not built).
8. Older: Qwen3.8-Flash-Next MTP verify, QFN prompt, Inkling copy speed (11-25 handoff).

## 6. Commands (Tower; scripts are in `tools/inkling-tower/`, live in `/mnt/storage/isos/grimoire-runs/ink-1123/`)

Layout: `X=/mnt/storage/isos/grimoire-runs/ink-1123`; sources of the dev build in `$X/src/{include,src,tools}`; binaries
`$X/src/bin/`; logs `$X/logs/`. **Edit/create Tower files only through `/mnt/storage/...` paths, not `/mnt/user`.**

```
# copy changed sources, build (CPU only, 20-60 s; NEVER while a dev server from $X/src runs)
scp tools/inkling2_gpu.cpp tools/inkling2_mtp.inc tools/inkling2_df2.inc tools/inkling2_xexp.cpp root@192.168.8.225:$X/src/tools/
ssh root@192.168.8.225 'cd $X && nice -n 19 docker run --rm --name ink-1123-builder --network none --cpuset-cpus 2-13 \
   -v $PWD:/work -w /work grimoire-dev:latest bash /work/build_ink.sh 2>&1 | grep -E "error|BUILD_OK"'

# one B70 (gpu0), greedy, drafter vs plain:   INK_DFLASH2=/df2 mounts /mnt/storage/Models/Inkling-2-Mini-DFlash2 at /df2
INK_I8=1 INK_DFLASH2=/df2 ./run_ink.sh TAG mtp /work/code2.txt 4096 64 384 2048
# two B70, greedy MTP (short plain phase):
INK_I8=1 INK_DFLASH2=/df2 INK_TP_PLAIN_N=8 INK_MTP_TIME=1 ./run_tp.sh TAG tpmtp /work/chat1.txt 4096 64 384 2048
# unitrace kernel profile:  WRAP="/prof/unitrace -d" ... ; then ./profsum.sh logs/TAG.log 40
# BetterBench A/B through the server (starts/stops serve_ink.sh itself; ~10 min an arm):
CHUNK=4096 ./ab_ink.sh "ds:INK_DFLASH2=0" "f2:INK_DFLASH2=/df2"
# verify-expert microbench:  ./run_bench.sh 25 0 300   (two-card case)   ./run_bench.sh 25 -1 300 (one card)
```
(`run_ink.sh` = gpu0 only; `run_tp.sh` = both B70s; every `INK_*` variable you set is passed into the container;
`run_ink.sh` / `run_tp.sh` refuse to start while another container runs.)

Release flow (as done for v1.10.14):
1. `grimoire-runs/ink-rel-1126/build_ink_rel_1126.sh` (G31 + G21 AOT; rebuilds `inkling2_gpu` + `libink_xexp.so`; the other libs are copied from `ink-rel-1124b`) from a `git archive` tar of the commit.
2. `grimoire-runs/image-ink-1126/Dockerfile` (copy in `tools/inkling-tower/Dockerfile.v1.10.14`): `FROM grimoire-b70:v1.10.13` + the bins; `docker build -t grimoire-b70:vX .`
3. `ink-1123/validate_ink_image2.sh IMAGE TAG` — Ian's template layout (models at /models, 262144 ctx), greedy coherence, BetterBench quick, `docker stop` -> exit 0 + "serve: quit", GPUs free.
4. `docker save IMAGE | gzip -6 > grimoire-runs/grimoire-b70-vX.tar.gz` (gzip -1 is > 2 GiB = over GitHub's asset limit), `gzip -t`, sha256.
5. README section + download block; release notes `release-notes-vX.md` (copy in `tools/inkling-tower/`).
6. Upload from the Tower (30 MB/s): the classic GitHub token is the newest `ghp_...` in `~/.claude/history.jsonl` on the Mac, piped over ssh stdin, never argv / disk:
   `grep -o 'ghp_[A-Za-z0-9]*' ~/.claude/history.jsonl | tail -1 | ssh root@192.168.8.225 'read -r GH_TOKEN; export GH_TOKEN; python3 /mnt/storage/isos/grimoire-fuse/tools/bench/upload_release.py TAG ARCHIVE COMMIT_SHA NOTES.md "NAME"'`
   (creates a prerelease at COMMIT_SHA if missing; verify the asset digest after). Then fast-forward main: `cd /mnt/storage/isos/grimoire-fuse && git push origin BRANCH BRANCH:main`.
7. Ian updates his own template's image tag (do NOT edit his templates/containers).

## 7. If something looks wrong

- Container exits right after start / hangs: `docker logs`, look for `WATCHDOG` (120 s without progress), `dmesg | grep 0000:0[37]` for GPU resets (gpu1 07:00.0 is on a Gen4 x4 OCuLink link; corrected AER errors are not failures).
- After a Tower reboot: `/mnt` empty = array not started (Ian starts it in the Unraid UI); `lspci -nn | grep -E "e223|e20b"`, re-check renderD numbers; the `uv`/pip caches are wiped.
- Open WebUI does not list the model: connection URL must be `http://localhost:6899/v1` (host network), check `curl http://127.0.0.1:6899/v1/models`.
- Two-card greedy chat diverges from plain decode near token 40 (DSpark and DFlash2 alike): known verify-vs-decode rounding, not the drafter.
- `INK_ABL`, `INK_SM_*`, `INK_F2_*` are debug / experiment switches; `INK_ABL` produces WRONG output on purpose.

## 8. Open questions (unknown, do not guess)

- urakozz's 6,200 tok/s: only Ian's relayed numbers (64K, pipeline parallel, INT8; earlier said 32K tensor parallel, corrected). Prompt content, chunk size and card set unknown.
- Whether pipeline prefill in our server can keep the TP decode layout without extra memory at 262K.
- Whether 1M context fits with INT8 KV + split experts (computed only).
- The 64K pipeline-mode number of OUR engine (never measured).
