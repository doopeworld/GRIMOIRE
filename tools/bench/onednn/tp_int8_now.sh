#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s: today's build with the int8 all-reduce payload (GRIMOIRE_TP_AR_INT8=1,
# opt-in, LOSSY) vs bf16 (default), same session: benchy pp2048 / pp4096 / tg32, code-probe answers.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
BASE="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-i8-on  TP $M fp8 16384 "$BASE GRIMOIRE_TP_AR_INT8=1" code,benchy --pp 2048 4096 --tg 32 --runs 2
run tp-i8-off TP $M fp8 16384 "$BASE" code,benchy --pp 2048 4096 --tg 32 --runs 2
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo "== answers int8 vs bf16 payload"; python3 /mnt/storage/isos/grimoire-fuse/tools/bench/codecmp.py logs/tp-i8-on.code.json logs/tp-i8-off.code.json
echo ALL_DONE
