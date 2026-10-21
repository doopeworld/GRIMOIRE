# GGUF work — brief for coding agents (orchestrated session, 2026-10-19)

Orchestrator: Claude Opus 5.5. Coding agents: Codex (gpt-6.1-sol) and Claude Code (Sonnet 5.5).
Each agent owns ONE task from `docs/gguf/STATUS.md` "Remaining". Read, in this order:
`docs/gguf/STATUS.md`, this file, `grimoire-gguf/CLAUDE.md` (rules 1-22 are binding; the
"CURRENT PRIORITY" block at its top is older than the GGUF branch and is NOT your task),
then the docs your task names (`NATIVE_QK.md`, `MODEL_MATRIX.md`, `PERFORMANCE_REPORT.md`,
`QWEN3_AUDIT.md`, `HANDOFF-2026-10-15.md`, `ARCHITECTURE.md`, `FORMAT_SUPPORT.md`).

## Where code lives

- Mac base clone: `~/Desktop/grimoire-gguf` on `feature/gguf-battlemage` (base 57ec09e).
  **Do not commit there and do not switch its branch.** Only the orchestrator merges into it.
- Your Mac worktree: `~/Desktop/grimoire-gguf-wt-<task>` on branch `gguf/<task>` (already created).
- No SYCL compiler on the Mac. Host tests: `make -s test` (and the specific `make test_*`) must pass
  before you push anything.
- Ship code to the Tower: `git push tower gguf/<task>` from your worktree (remote `tower` =
  `root@192.168.8.225:/mnt/user/isos/grimoire-gguf`). The Mac cannot push to GitHub.
- Tower: `ssh root@192.168.8.225`. Main Tower clone `/mnt/user/isos/grimoire-gguf` holds the
  CURRENT GOOD `bin/grimoire` (built 10-18 from 57ec09e). **Never rebuild, edit or check out
  anything in that main clone.** Measurement-only tasks use its `bin/` read-only
  (`GRIM_TREE=/mnt/user/isos/grimoire-gguf`).
- Code tasks build in their own Tower worktree:
  `cd /mnt/user/isos/grimoire-gguf && git worktree add /mnt/user/isos/gguf-wt/<task> gguf/<task>`
  and point `GRIM_TREE` / launchers at it. Find out how 10-18's `bin/grimoire` was built
  (`build_b70.sh`, `tools/build_grimoire_only_b70.sh`, `tools/dev_build.sh` — the latter hardcodes
  the grimoire-fuse path, so pass/adapt the repo path; check `docker images`) and use the same
  compiler/image. Record the exact build command in your report.
- Models: `/mnt/models/...` (GGUF + eval files, `/mnt/models/eval/wiki.test.raw`, KLD bases
  `kld-q38-bf16.bin`, `kld-q3-8b-q8.bin`, `kld-bf16.bin`), HF safetensors under
  `/mnt/storage/Models/`. `/mnt/models` has ~167 GB free: do not fill it; ask before > 40 GB.
  Downloads: `benchmarks/gguf/pfetch.sh` (HF throttles single connections).

## GPU rules (non-negotiable)

- **gpu0 ONLY** (PCI 0000:03:00.0; render node from `tools/gpunode.sh gpu0`, it changes per boot).
  The second B70 (now at 0000:07:00.0) is OFF LIMITS — never run anything on it.
- Only ONE GPU job at a time across all agents: wrap EVERY GPU launch in
  `flock /mnt/user/isos/grimoire-runs/gpu0.lock <command>` (blocking is fine; never bypass it,
  never hold it while idle/editing; keep each locked run as short as possible).
- Launch only through the repo wrappers (`tools/b70run.sh`, `tools/g0run.sh`, `tools/tune.sh`,
  `benchmarks/gguf/*.sh`) — never a bare `docker run`/`timeout docker run` on the GPU (rule 6).
  Remove every container when its test ends. Never set `SYCL_UR_USE_LEVEL_ZERO_V2=0`.
- After any device fault (DEVICE_LOST, AER errors, `g0run.sh` exit 9) STOP all GPU work and report;
  do not retry blindly.
- Rule 8: never quote a number from a config whose generated text you did not read.

## Wave-2 notes (orchestrator, 2026-10-19 ~17:00)

- 15:02 gpu0 engine reset + DEVICE_LOST was Qwen3-30B-A3B Q4_K_M `--proj native` (first token
  never produced); further exec-queue resets at 15:07. The card recovered (engine reset, not a bus
  drop) and a 53-minute KLD run afterwards was clean. The orchestrator judged gpu0 healthy and
  removed the DEVICE_FAULT markers. If YOU cause a new fault: stop GPU work for that line of
  investigation, capture dmesg + devcoredump, and report; do not loop on a reproducer.
- Give every GPU run a hard limit (b70run.sh timeout argument / LIM) sized to the job, ≤ 15 min
  unless the measurement truly needs more; a hang must not hold the lock for an hour.
- Codex agents hit their usage limit mid-task; wave-2 Sonnet agents continue their branches.
  Read the existing commits/diff on your branch first and build on them (or revert with reason).
- Tower sandbox note: worktree builds must create `bin/` first (a probe link failed on a missing
  `bin/`).

## Git / reporting

- Commit on `gguf/<task>` at real checkpoints with clear messages; push to `tower`.
- Put run logs under `/mnt/user/isos/grimoire-runs/gguf-<task>/`.
- Update docs for your task in your branch (e.g. `NATIVE_QK.md`, `MODEL_MATRIX.md`,
  `PERFORMANCE_REPORT.md`) but do NOT edit `docs/gguf/STATUS.md` — the orchestrator owns it.
- Final report: what changed (commits), what was verified where (host / B70, exact commands and
  log paths), measured numbers, the text you read, and what is NOT verified (rule 14).
