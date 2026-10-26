#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s: wavefront prompt with the automatic chunk (768, 1024 from 3,584 rows)
# and the ESIMD flash prefill for the 12/2-head TP rank (both default now) vs 768 + joint_matrix
# (GRIMOIRE_TP_WAVE_CHUNK=768 GRIMOIRE_FLASH_ESIMD=0), same session; pp2048 / 4096 / 8192.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-ad1 TP $M fp8 16384 "$B" code,benchy --pp 2048 4096 8192 --tg 32 --runs 2
run tp-ad0 TP $M fp8 16384 "$B GRIMOIRE_TP_WAVE_CHUNK=768 GRIMOIRE_FLASH_ESIMD=0" code,benchy --pp 2048 4096 8192 --tg 32 --runs 2
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo "== answers new defaults vs 768 + joint_matrix"; python3 /mnt/storage/isos/grimoire-fuse/tools/bench/codecmp.py logs/tp-ad1.code.json logs/tp-ad0.code.json
echo ALL_DONE
