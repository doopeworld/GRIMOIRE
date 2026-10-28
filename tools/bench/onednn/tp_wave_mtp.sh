#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s: MTP with the wavefront TP prompt (default) -- code probe + benchy,
# answers vs the plain wave run (tp-wv1).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-wvm TP $M fp8 16384 "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=1 GRIMOIRE_SPEC_TIME=1" code,benchy --pp 2048 4096 --tg 128 --runs 2
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo "== answers MTP+wave vs plain wave"; python3 /mnt/storage/isos/grimoire-fuse/tools/bench/codecmp.py logs/tp-wvm.code.json logs/tp-wv1.code.json
echo ALL_DONE
