#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s (TP, no MTP), second A/B (2026-10-25 afternoon, row pieces):
#   1-2  the 10-24 tp-def command, oneDNN (row rule) on / off
#   3-5  prompts of eight NEW lengths, no warm-up, one run each (the first, 1024, warms the
#        server's one-time kernels): what a user with varying prompts sees, for the row rule,
#        one primitive per exact M, and oneDNN off.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
BASE="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0"
NEWLEN="--pp 1024 1100 1300 1500 1700 1900 2100 2300 2500 --tg 8 --runs 1 --no-warmup"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-rule   TP $M fp8 16384 "$BASE GRIMOIRE_ONEDNN=1 GRIMOIRE_ONEDNN_VERBOSE=1" code,benchy --pp 2048 4096 --tg 128 --runs 2
run tp-off2   TP $M fp8 16384 "$BASE" code,benchy --pp 2048 4096 --tg 128 --runs 2
run nl-rule   TP $M fp8 16384 "$BASE GRIMOIRE_ONEDNN=1 GRIMOIRE_ONEDNN_VERBOSE=1" benchy $NEWLEN
run nl-exact  TP $M fp8 16384 "$BASE GRIMOIRE_ONEDNN=1 GRIMOIRE_ONEDNN_EXACT_M=1" benchy $NEWLEN
run nl-off    TP $M fp8 16384 "$BASE" benchy $NEWLEN
echo ALL_DONE
