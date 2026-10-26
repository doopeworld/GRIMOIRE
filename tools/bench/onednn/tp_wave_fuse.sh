#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s: wavefront prompt with fused passes (default) vs separate
# (GRIMOIRE_TP_WAVE_FUSE=0), same session; then the compute floor with the fused passes
# (GRIMOIRE_TP_WAVE_NOAR=1, debug, wrong output; was 1,211-1,215 ms at 4,097 rows).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0 GRIMOIRE_TP_AR_TIMING=1"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-wf1 TP $M fp8 16384 "$B" code,benchy --pp 2048 4096 --tg 32 --runs 2
run tp-wf0 TP $M fp8 16384 "$B GRIMOIRE_TP_WAVE_FUSE=0" code,benchy --pp 2048 4096 --tg 32 --runs 2
run tp-wfn TP $M fp8 16384 "$B GRIMOIRE_TP_WAVE_NOAR=1" benchy --pp 4096 --tg 1 --runs 1
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
C=/mnt/storage/isos/grimoire-fuse/tools/bench/codecmp.py
echo "== answers fused vs separate"; python3 $C logs/tp-wf1.code.json logs/tp-wf0.code.json
echo "== answers fused vs this morning's wavefront"; python3 $C logs/tp-wf1.code.json logs/tp-wv1.code.json
for n in wf1 wf0 wfn; do echo "== $n"; grep -h "TP wavefront prompt rank 0: 409" logs/tp-$n.server.log | tail -2; done
echo ALL_DONE
