#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s: the small all-reduce through the BAR in pieces (default) vs the
# plain host round trip (GRIMOIRE_TP_SMALL_BAR=0), same session.
#   1-2  DFlash2 code probe, BAR / plain (vs 119-121 tok/s, verify 42.8 ms on 10-25)
#   3-4  plain TP code probe + benchy, BAR / plain (tg 31.4 on 10-25)
#   5    DFlash2 with GRIMOIRE_TP_SMALL_AR_TIME=1: the new path's us per call (plain path: 86 us at 8 rows)
# Answers: codecmp of 1 vs 2 and 3 vs 4 at the end.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
BASE="GRIMOIRE_POWER_LIMIT_W=0"
DF="GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-sb-df1 TP $M fp8 16384 "$BASE $DF GRIMOIRE_SPEC_TIME=1" code
run tp-sb-df0 TP $M fp8 16384 "$BASE $DF GRIMOIRE_SPEC_TIME=1 GRIMOIRE_TP_SMALL_BAR=0" code
run tp-sb-pl1 TP $M fp8 16384 "$BASE GRIMOIRE_MTP=0" code,benchy --pp 2048 --tg 128 --runs 2
run tp-sb-pl0 TP $M fp8 16384 "$BASE GRIMOIRE_MTP=0 GRIMOIRE_TP_SMALL_BAR=0" code,benchy --pp 2048 --tg 128 --runs 2
run tp-sb-dft TP $M fp8 16384 "$BASE $DF GRIMOIRE_TP_SMALL_AR_TIME=1" code
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
C=/mnt/storage/isos/grimoire-fuse/tools/bench/codecmp.py
echo "== answers DFlash2 BAR vs plain path"; python3 $C logs/tp-sb-df1.code.json logs/tp-sb-df0.code.json
echo "== answers plain TP BAR vs plain path"; python3 $C logs/tp-sb-pl1.code.json logs/tp-sb-pl0.code.json
echo "== answers DFlash2 BAR vs plain TP BAR"; python3 $C logs/tp-sb-df1.code.json logs/tp-sb-pl1.code.json
echo "== answers plain TP BAR vs 10-26 morning plain"; python3 $C logs/tp-sb-pl1.code.json logs/tp-art-pl.code.json
echo "== spec time, BAR then plain path"
grep -h "rank0.*spec time per step" logs/tp-sb-df1.server.log | tail -1
grep -h "rank0.*spec time per step" logs/tp-sb-df0.server.log | tail -1
grep -h "TP small all-reduce" logs/tp-sb-dft.server.log | tail -4
echo ALL_DONE
