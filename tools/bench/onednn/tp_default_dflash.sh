#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s (TP), after oneDNN became the default and Megatron TP stopped
# refusing a DFlash drafter:
#   1  defaults (oneDNN on with no variable), plain decode -- must match tp-warm2
#   2  + DFlash2 drafter (block 8), loaded whole on each rank, drafting in lockstep -- greedy,
#      so the code probe's answers must be IDENTICAL to round 1's
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
BASE="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-def3 TP $M fp8 16384 "$BASE" code,benchy --pp 2048 4096 --tg 128 --runs 2
run tp-df2  TP $M fp8 16384 "$BASE GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_SPEC_STATS=1 GRIMOIRE_SPEC_TIME=1" code,benchy --pp 2048 --tg 128 --runs 2
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo ALL_DONE
