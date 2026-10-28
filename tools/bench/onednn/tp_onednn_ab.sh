#!/bin/bash
# Qwen3.8-27B-FP8 on both B70s (TP, no MTP), the 10-24 tp-def command, with oneDNN on and then
# off (same new binary), so the two rows differ only in GRIMOIRE_ONEDNN.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
./dual.sh tp-dnnl TP /models2/Qwen3.8-27B-FP8 fp8 16384 "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0 GRIMOIRE_ONEDNN=1 GRIMOIRE_ONEDNN_VERBOSE=1" code,benchy --pp 2048 4096 --tg 128 --runs 2 2>&1
[ -e logs/tp-dnnl.STOP ] && exit 3
sleep 3
./dual.sh tp-def-1025 TP /models2/Qwen3.8-27B-FP8 fp8 16384 "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0" code,benchy --pp 2048 4096 --tg 128 --runs 2 2>&1
echo ALL_DONE
