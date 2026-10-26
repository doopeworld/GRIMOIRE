#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s: where a 4096-token TP prompt spends its 1.55 s (pp4096 ~2,650 tok/s).
#   1  GRIMOIRE_TIME_LAYER=all: per-region device time (drains the queue at every region, so the
#      all-reduce overlap is gone -- compute per region is what it shows)
#   2  GRIMOIRE_TP_AR_TIMING=1: the chunked FFN's wait on transfers, pipelined all-reduce phases
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
BASE="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-pf-al TP $M fp8 16384 "$BASE GRIMOIRE_TIME_LAYER=all" benchy --pp 4096 --tg 1 --runs 1
run tp-pf-ar TP $M fp8 16384 "$BASE GRIMOIRE_TP_AR_TIMING=1" benchy --pp 4096 --tg 1 --runs 1
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo ALL_DONE
