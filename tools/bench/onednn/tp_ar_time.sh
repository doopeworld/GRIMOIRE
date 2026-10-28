#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s: where a small all-reduce's time goes (GRIMOIRE_TP_SMALL_AR_TIME=1,
# diagnostic -- it adds two waits per call, so these runs are slower than normal).
#   1  DFlash2 code probe: the 8-row verify's all-reduces
#   2  plain TP code probe: the 1-row decode all-reduces
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
BASE="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_TP_SMALL_AR_TIME=1"
DF="GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-art-df TP $M fp8 16384 "$BASE $DF GRIMOIRE_SPEC_TIME=1" code
run tp-art-pl TP $M fp8 16384 "$BASE GRIMOIRE_MTP=0" code
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo ALL_DONE
