#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s: wavefront prompt chunk size (GRIMOIRE_TP_FFN_CHUNK, multiple of 256;
# default 768) -- benchy pp2048 / pp4096, same session.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-wc768  TP $M fp8 16384 "$B" code,benchy --pp 2048 4096 --tg 1 --runs 2
run tp-wc1024 TP $M fp8 16384 "$B GRIMOIRE_TP_FFN_CHUNK=1024" code,benchy --pp 2048 4096 --tg 1 --runs 2
run tp-wc1536 TP $M fp8 16384 "$B GRIMOIRE_TP_FFN_CHUNK=1536" code,benchy --pp 2048 4096 --tg 1 --runs 2
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
C=/mnt/storage/isos/grimoire-fuse/tools/bench/codecmp.py
echo "== answers 1024 vs 768"; python3 $C logs/tp-wc1024.code.json logs/tp-wc768.code.json
echo "== answers 1536 vs 768"; python3 $C logs/tp-wc1536.code.json logs/tp-wc768.code.json
echo ALL_DONE
