#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s, after verify-through-oneDNN became the default and the drafter's
# block attention split its key range:
#   1  DFlash2, drafter stage timer (attention must drop from 8.3 ms)  -- code probe
#   2  DFlash2 clean: code probe + benchy (vs 99.7-101.6 / 45.2)
#   3  MTP clean: code probe + benchy (vs 79.2 / 50.4-51.6) -- its verify now goes through oneDNN
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
BASE="GRIMOIRE_POWER_LIMIT_W=0"
DF="GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-df5t TP $M fp8 16384 "$BASE $DF GRIMOIRE_DFLASH_TIME=1 GRIMOIRE_SPEC_TIME=1" code
run tp-df5  TP $M fp8 16384 "$BASE $DF GRIMOIRE_SPEC_TIME=1" code,benchy --pp 2048 --tg 128 --runs 2
run tp-mtp5 TP $M fp8 16384 "$BASE GRIMOIRE_MTP=1 GRIMOIRE_SPEC_TIME=1" code,benchy --pp 2048 4096 --tg 128 --runs 2
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo ALL_DONE
