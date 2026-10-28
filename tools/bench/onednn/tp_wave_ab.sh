#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s: the wavefront TP prompt (default) vs the per-layer path
# (GRIMOIRE_TP_WAVE=0 GRIMOIRE_TP_PUMP=0 = exactly the old behaviour), same session.
#   1-2  plain TP: code probe + benchy pp2048 / pp4096 / tg32 (10-26: 2,661 / 2,626 / 32.0)
#   3    int8 payload + wave: pp (10-26 without wave: 3,158 / 3,259)
#   4    DFlash2 + wave: code probe (taps for the drafter are stored chunk by chunk)
# Answers: codecmp 1 vs 2, 4 vs 1.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
BASE="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0"
DF="GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-wv1 TP $M fp8 16384 "$BASE GRIMOIRE_TP_AR_TIMING=1" code,benchy --pp 2048 4096 --tg 32 --runs 2
run tp-wv0 TP $M fp8 16384 "$BASE GRIMOIRE_TP_WAVE=0 GRIMOIRE_TP_PUMP=0" code,benchy --pp 2048 4096 --tg 32 --runs 2
run tp-wv8 TP $M fp8 16384 "$BASE GRIMOIRE_TP_AR_INT8=1" code,benchy --pp 2048 4096 --tg 32 --runs 2
run tp-wvd TP $M fp8 16384 "$BASE $DF GRIMOIRE_SPEC_TIME=1" code
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
C=/mnt/storage/isos/grimoire-fuse/tools/bench/codecmp.py
echo "== answers wave vs per-layer (plain TP)"; python3 $C logs/tp-wv1.code.json logs/tp-wv0.code.json
echo "== answers DFlash2+wave vs plain wave"; python3 $C logs/tp-wvd.code.json logs/tp-wv1.code.json
echo "== answers int8+wave vs per-layer bf16"; python3 $C logs/tp-wv8.code.json logs/tp-wv0.code.json
grep -h "TP wavefront prompt rank 0" logs/tp-wv1.server.log | tail -3
echo ALL_DONE
