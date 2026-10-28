#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s, wavefront TP prompt at 4,097 rows: time to the last kernel and host
# waits per phase (GRIMOIRE_TP_AR_TIMING=1), with the all-reduce (bf16, int8) and WITHOUT any
# (GRIMOIRE_TP_WAVE_NOAR=1: debug, wrong output -- the cards' own compute, the floor).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0 GRIMOIRE_TP_AR_TIMING=1"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-wp-bf TP $M fp8 16384 "$B" benchy --pp 4096 --tg 1 --runs 1
run tp-wp-i8 TP $M fp8 16384 "$B GRIMOIRE_TP_AR_INT8=1" benchy --pp 4096 --tg 1 --runs 1
run tp-wp-no TP $M fp8 16384 "$B GRIMOIRE_TP_WAVE_NOAR=1" benchy --pp 4096 --tg 1 --runs 1
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
for n in bf i8 no; do echo "== $n"; grep -h "TP wavefront prompt rank 0: 409" logs/tp-wp-$n.server.log | tail -2; done
echo ALL_DONE
