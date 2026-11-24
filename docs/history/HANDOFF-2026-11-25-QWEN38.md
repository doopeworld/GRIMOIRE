# Handoff 2026-11-25 — publish v1.10.12, then Qwen3.8 vs Xenithium (two B70s)

Ian 11-24 night: "yes publish" (v1.10.12), "i go sleep you will publish tomorrow"; earlier: "when you are finished go back
working on qwen3.8, look results from xenithium" and "as of now please use tool for speed test : better bench".
The Tower stopped answering (ping, ssh) right after the v1.10.12 archive was written, before the upload.

## 1. First, when the Tower is up

1. Publish v1.10.12 (approved): archive `/mnt/storage/isos/grimoire-runs/grimoire-b70-v1.10.12.tar.gz`
   (1,789,002,763 B, sha256 c4d8715da57b4b5a288434f1c7cfd7735e09201b72e49a81c63c719afdf9bcb3), notes
   `grimoire-runs/release-notes-v1.10.12.md`, tag target f49e347 (branch inkling-gpu-1123, already on GitHub):
   `grep -o 'ghp_[A-Za-z0-9]*' ~/.claude/history.jsonl | tail -1 | ssh root@192.168.8.225 'read -r GH_TOKEN; export
   GH_TOKEN; cd /mnt/storage/isos/grimoire-runs && python3 upload_release_1124.py v1.10.12 grimoire-b70-v1.10.12.tar.gz
   f49e347c6c91787cbe0f5c4f8937820788f07396 release-notes-v1.10.12.md "GRIMOIRE v1.10.12 — Inkling 2 Mini at 262K context"'`
   then check GitHub's asset digest = the sha256 above.  If the image is gone after a reboot: `docker images | grep v1.10.12`
   (image ca4d43a6; rebuild from grimoire-runs/image-ink-1124b if needed).
2. README on main: v1.10.12 in the summary paragraph, the "Get the image" block and the Inkling section (262K, decode
   at 216K 87.8 tok/s).  Worktree `grimoire-runs/gh-1124` (branch main-update-1124 = main 1da7280).
3. Check after a reboot: `lspci` (B70s at 03:00.0 and 07:00.0, B580 0b:00.0 untouched), render nodes by PCI path.
   Container GRIMOIRE-DUAL-INKLING2 (port 6899, v1.10.12, ctx 262144) is created and stopped.

## 2. Qwen3.8 vs Xenithium: what they publish (xenithium-bench easy mode, their 4x B70 box at 220 W)

| two B70 | decode | 1/4/8/16 streams | prefill 8K / 32K / 128K | decode at 32K / 109K |
|---|---|---|---|---|
| Qwen3.8-27B Swift-1.5, DSpine b16 + trees + prompt lookup, int8 KV, 16 slots | 272 | 227 / 461 / 687 / 769 | 4,542 / 4,081 / 2,775 | 174 / 151 |
| Qwen3.8-27B AutoRound, same setup | 273 | 224 / 448 / 652 / 783 | 4,557 / 4,119 / 2,793 | 173 / 135 |
| Qwen3.8-Flash-Next W4A16, DFlash 2 | 145 | 120 / 183 / 205 / 214 | 2,676 / 2,717 / 2,577 | 123 / 99 |

Ian's screenshot (qwen38-w4a16-intel = Flash-Next) is their 4-card run: 351 / 945 at 16 streams / 7,943 at 250K.
Their 27B chat request: update gap 19.3 ms, 3.18 tokens per update (164 tok/s); code 243, JSON 433, edits 290: prompt
lookup (XEN_LOOKUP=4) on copy-like text.  Preset models/presets/qwen38-27b-intel*.env: XEN_DRAFTER_TREE=7,
XEN_DRAFTER_WIDE=7, XEN_DRAFTER_CUT=0.02, XEN_KV=int8 (lossy: not wanted as our default), XEN_PREFILL_GEMM=onednn-all
from 512 rows.  Source clone: ~/grimoire-work/ref-xenithium/src_repo at 9e542a1.

## 3. Plan

1. Same-method baselines on our two B70s, nothing changed: BetterBench (Ian's tool; betterbench-vllm1111.json, quick)
   AND xenithium-bench easy mode (host-only program: `scripts/build-host.sh xenithium-bench` in a --rm grimoire-dev
   container) against GRIMOIRE v1.10.12 serving Swift-1.5 GPTQ-Int4 on two cards (TP2 + MTP) and Flash-Next two cards.
   Old reference (11-13, BetterBench): GRIMOIRE TP2 134.6 weighted vs vLLM 131.0.
2. Measure our step: draft / verify / exchange / commit (GRIMOIRE_SPEC_TIME=1) against their 19.3 ms update gap.
3. Biggest lever first.  Candidates:
   - Step time: the two-card exchange, the small-kernel count (the Inkling work cut verify by fusing and splitting).
   - Drafter: adaptive draft length (Inkling: prod q < 0.25 gave +11 % on BetterBench), DFlash 2 / DSpine b16 with trees,
     prompt lookup inside the spec step.
4. Every GPU run under a hang watch; B580 never; no lossy defaults.
