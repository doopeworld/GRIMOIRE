# Working on GRIMOIRE — read this first

This file is for any AI assistant (Claude, or otherwise) picking up this
project. Ian runs multiple assistants on GRIMOIRE at different times — this
doc exists so none of them re-derive or re-break what's already settled.

## Where things run

All real work happens on **the Tower** (`root@192.168.8.225`), a Unraid box
with 2x Intel Arc Pro B70 (Battlemage), reachable over SSH. The repo lives at
`/mnt/storage/isos/grimoire-fuse`. This GitHub repo is the SOURCE OF TRUTH for
code; the Tower is where it's built and run. If the Tower rebooted and
`/mnt/storage/...` is missing, the Unraid ARRAY may not have auto-started —
check before assuming data loss.

Models live under `/mnt/storage/Models/` on the Tower, never in git.

## Backup workflow — GIT, NOT TARBALLS

As of 2026-08-28 this project backs up via `git add && git commit && git push`
to `git@github.com:doopeworld/GRIMOIRE.git`, run from the Tower. **Do not
create `.tar.gz` backups** — any tarballs you find on the Desktop or Tower
predate this and are stale, safe to ignore.

Commit at real checkpoints: a verified working state, right before a risky
kernel/architecture change, and end of session — not only when explicitly
asked. Only committed work survives a lost session; there is no tarball
safety net anymore, so commit discipline matters more, not less.

`.gitignore` excludes all generated content (`bin/`, `*.so`, `runtime-b70/`,
logs, `.bak*`, model files). Keep it that way — this repo is source only.

Auth: an SSH deploy key lives on the Tower (`~/.ssh/id_ed25519`), registered
to Ian's GitHub account. The `known_hosts` warning on push
(`hostfile_replace_entries: ... Operation not permitted`) is a harmless
Unraid mount quirk — push succeeds anyway, ignore it.

## Hard-won rules — do not re-learn these the expensive way

**1. `GRIMOIRE_W4A8=1` frees the MXFP4 payload of every converted weight.**
Any code path that still dereferences `w.payload` after conversion reads a
freed pointer ON DEVICE — this is a `DEVICE_LOST`, not a clean error, and it
drops a B70 off the PCI bus (`forcewake register returns 0xFFFFFFFF`,
needs a POWER CYCLE to recover, no software fix). Also: a converted weight
still reports `fmt == MXFP4` — checking format is NOT a safety check, check
the pointer. After touching weight conversion, `grep -n "\.w\.payload"` and
confirm every hit is guarded or routed through `mm`/`mmb`/`mmbb`/`gemv_any`.

**2. `GRIMOIRE_DAG=1` (out-of-order queue) produces GARBAGE, not a speedup.**
It looks like a free +10% PP because the out-of-order queue overlaps kernels
— but it doesn't express all real dependencies, so it's a race and the
output is wrong. `FULL E2E PP: PASS` does NOT catch this (the self-check
isn't sensitive to ordering bugs) — only generating text and reading it did.
Never trust a queue-ordering change without generating and reading text.

**3. `build_b70.sh` does NOT rebuild the cutlass bridges** (the `.so` files
in `src/`). After touching a bridge (`xe2_grouped_bridge.cpp` etc.), you MUST
run `tools/build_bridges_b70.sh`, which rebuilds bridges then calls
`build_b70.sh`. Running `build_b70.sh` alone silently leaves a stale `.so` —
this caused a DEVICE_LOST that looked like a kernel bug but was a 5-minute-
old bridge.

**4. Any W4A8 GEMM tile is 256 wide in N; the B 2-D block loads do NOT clamp
to the tensor.** A weight with `N % 256 != 0` (e.g. Ornith's deltanet `a/b`
projection, N=96) reads hundreds of KB past the end of the tensor ->
DEVICE_LOST. Guard `N % 256` before routing a small-N weight through a W4A8
tile; use the GEMV instead.

**5. Small-M batches need a SMALL TILE, not the production 128-row tile.**
The 128-row W4A8/MXFP4 tile fetches 128 rows of A per tile regardless of how
few tokens are given — at M=4 that's ~89 MB/layer of pure padding. A 16-row
tile (`m16x256`) is weight-bound and flat from M=1 to M=16. This is what
makes a speculative verify batch or short prefill affordable; forcing the
wide/128 tile on a small batch is a measured DISASTER (batch cost went UP,
not down).

**6. Never `docker run` the B70 directly with a bare `timeout` wrapper.**
`timeout N docker run` orphans the container and wedges the GPU; a bash-as-
PID-1 entrypoint also swallows SIGTERM and escalates to SIGKILL, which tears
down mid-submission and drops the card off the bus. Always launch through
`tools/tune.sh` / `tools/vtune.sh` / `tools/pp2run.sh` / `tools/tp2run.sh`,
which wrap `b70run.sh`'s safe `--init` + in-container `timeout` + detached
`docker wait` pattern.

**7. Multi-GPU on this box is currently on a USB4/Thunderbolt link for one
card.** Ian is swapping to OCuLink Gen4 x4 PCIe soon. Until then: raw
single-process cross-device allocation fails once the process holds a large
host mmap (measured, not theoretical — see `ORNITH-2026-08-27.md`).
Multiprocess PP/TP (matching vLLM's own architecture: one process per GPU,
socket/IPC handoff) IS the working approach on this hardware — see
`tools/pp2run.sh`, `tools/tp2run.sh`, `HANDOFF-2026-08-27-PP-TP-COMPLETE.md`.
Do not re-attempt single-process multi-device until OCuLink is confirmed
installed.

**8. Never quote a benchmark number from a config that never generated text.**
Multiple false leads in this project (BesTLA, DAG, several kernel "wins")
were caught only because someone bothered to run `-p "prompt" -n N` and read
the output. A PASS on a numeric self-check is not sufficient.

## Where to look for current status

Read the newest-dated `.md` at the repo root first (sort by date in the
filename or `git log --oneline -- '*.md'`) — these are running handoff logs
per session, most recent state and next-steps at the bottom of each file:

- `GRIMOIRE-2026-08-26-TG.md` — Qwen TG/PP work, W4A8 kernel details
- `ORNITH-2026-08-27.md` — Ornith TG work, speculative-decoding link reviews,
  ranked next-steps plan
- `HANDOFF-2026-08-27-PP-TP-COMPLETE.md` — multiprocess PP/TP, prefix cache,
  Muse Glimmer integration, and the CURRENT next-steps (Muse batched prefill)

`ref/` holds extracted vLLM reference implementations (DFlash2, Muse Glimmer
modeling code) — pulled from the vLLM nightly image specifically so nobody
has to guess an architecture's exact forward pass again. Read the actual
reference before implementing a new model or speculative decoding scheme;
several sessions were burned guessing norm order / concat order instead of
extracting the real code first.
