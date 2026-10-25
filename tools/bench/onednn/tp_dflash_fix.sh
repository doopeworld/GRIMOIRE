#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s (TP) after the drafter fixes (onednn-1025 8550947):
#   1  DFlash2, clean numbers (code probe + benchy)
#   2  DFlash2 with GRIMOIRE_DFLASH_TIME=1 (fenced stage split; code probe only)
#   3  MTP (its TP verify's lm_head now goes through gemm_tp -> oneDNN), vs 10-24 tp-def-mtp
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
BASE="GRIMOIRE_POWER_LIMIT_W=0"
DF="GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-df3   TP $M fp8 16384 "$BASE $DF" code,benchy --pp 2048 --tg 128 --runs 2
run tp-df3t  TP $M fp8 16384 "$BASE $DF GRIMOIRE_DFLASH_TIME=1" code
run tp-mtp3  TP $M fp8 16384 "$BASE GRIMOIRE_MTP=1" code,benchy --pp 2048 4096 --tg 128 --runs 2
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo ALL_DONE
