#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s, wavefront prompt region budget at 4,097 rows (GRIMOIRE_TP_WAVE_TIME=1,
# diagnostic: drains the queue after every region): compute alone (NOAR, debug) and with the exchange.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0 GRIMOIRE_TP_WAVE_TIME=1"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-wr-no TP $M fp8 16384 "$B GRIMOIRE_TP_WAVE_NOAR=1" benchy --pp 4096 --tg 1 --runs 1
run tp-wr-ar TP $M fp8 16384 "$B" benchy --pp 4096 --tg 1 --runs 1
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
for n in no ar; do echo "== $n"; awk '/TP wavefront region budget \(409[0-9] rows/{f=1;c++} f&&c==2{print} /TOTAL \(timed regions\)/{if(f&&c==2)exit; f=0}' logs/tp-wr-$n.server.log; done
echo ALL_DONE
