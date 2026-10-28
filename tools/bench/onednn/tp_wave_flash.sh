#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s, wavefront prompt: (1) the ESIMD flash prefill forced on for the TP
# rank's 12 query / 2 KV heads (GRIMOIRE_FLASH_ESIMD=1; by default only 24/4 takes it), with
# GRIMOIRE_FLASH_VERIFY=6 printing ESIMD vs joint_matrix for the first six calls; (2)-(3) chunk
# sizes 1024 / 1536 (GRIMOIRE_TP_FFN_CHUNK; default 768: pp2048 3,383, pp4096 3,287 at 17:49).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-fe1    TP $M fp8 16384 "$B GRIMOIRE_FLASH_ESIMD=1 GRIMOIRE_FLASH_VERIFY=6" code,benchy --pp 2048 4096 --tg 1 --runs 2
run tp-wc1024 TP $M fp8 16384 "$B GRIMOIRE_TP_FFN_CHUNK=1024" code,benchy --pp 2048 4096 --tg 1 --runs 2
run tp-wc1536 TP $M fp8 16384 "$B GRIMOIRE_TP_FFN_CHUNK=1536" code,benchy --pp 2048 4096 --tg 1 --runs 2
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
C=/mnt/storage/isos/grimoire-fuse/tools/bench/codecmp.py
echo "== answers ESIMD flash vs default"; python3 $C logs/tp-fe1.code.json logs/tp-wc768.code.json
echo "== answers 1024 vs 768"; python3 $C logs/tp-wc1024.code.json logs/tp-wc768.code.json
echo "== answers 1536 vs 768"; python3 $C logs/tp-wc1536.code.json logs/tp-wc768.code.json
grep -h "flash verify" logs/tp-fe1.server.log | head -8
echo ALL_DONE
