#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s (TP, no MTP) with GRIMOIRE_ONEDNN=1 and the server's start-up
# warm-up (tools/grimoire_server.cpp): the first request must no longer pay the one-time
# kernel compiles (cold code-probe TTFT was 1.42 s vs 0.97 off), and nothing else may move.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
BASE="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0"
NEWLEN="--pp 1024 1100 1300 1500 1700 1900 2100 2300 2500 --tg 8 --runs 1 --no-warmup"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-warm  TP $M fp8 16384 "$BASE GRIMOIRE_ONEDNN=1 GRIMOIRE_ONEDNN_VERBOSE=1" code,benchy --pp 2048 4096 --tg 128 --runs 2
run nl-warm  TP $M fp8 16384 "$BASE GRIMOIRE_ONEDNN=1" benchy $NEWLEN
pkill -f "http.server 8999" 2>/dev/null
echo ALL_DONE
