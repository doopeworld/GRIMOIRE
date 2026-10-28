#!/bin/bash
# Ian 10-26: llama-benchy pp4096 / tg128 table, Qwen3.8-27B-FP8 on both B70s, every mode, one session.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
B="GRIMOIRE_POWER_LIMIT_W=0"
DF="GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-b-old  TP $M fp8 16384 "$B GRIMOIRE_MTP=0 GRIMOIRE_TP_WAVE=0 GRIMOIRE_TP_PUMP=0" benchy --pp 4096 --tg 128 --runs 2
run tp-b-pl   TP $M fp8 16384 "$B GRIMOIRE_MTP=0" benchy --pp 4096 --tg 128 --runs 2
run tp-b-pl8  TP $M fp8 16384 "$B GRIMOIRE_MTP=0 GRIMOIRE_TP_AR_INT8=1" benchy --pp 4096 --tg 128 --runs 2
run tp-b-mtp  TP $M fp8 16384 "$B GRIMOIRE_MTP=1" benchy --pp 4096 --tg 128 --runs 2
run tp-b-mtp8 TP $M fp8 16384 "$B GRIMOIRE_MTP=1 GRIMOIRE_TP_AR_INT8=1" benchy --pp 4096 --tg 128 --runs 2
run tp-b-df   TP $M fp8 16384 "$B GRIMOIRE_MTP=0 $DF" benchy --pp 4096 --tg 128 --runs 2
run tp-b-df8  TP $M fp8 16384 "$B GRIMOIRE_MTP=0 $DF GRIMOIRE_TP_AR_INT8=1" benchy --pp 4096 --tg 128 --runs 2
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo ALL_DONE
