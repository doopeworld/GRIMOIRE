#!/bin/bash
# 10-27: GRIMOIRE_KSEND_CHECK=1 on two cards: every kernel copy (all-reduce copy out, CPU experts'
# route) compared byte for byte with a copy-engine copy of the same device bytes; must be 0 differed.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
RP="RP_WIN=1000 RP_RUN=1000000"
env $RP LOAD_MAX=1500 ./dual.sh fn-kschk TP $F bf16 65536 "$B $PLE GRIMOIRE_KSEND_CHECK=1" gen > logs/fn-kschk.run 2>&1
grep -E "\[gen\]|STOP|LEFT|ready=0" logs/fn-kschk.run | cut -c1-160
grep -E "ksend check" logs/fn-kschk.server.log | tail -4
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_KSCHK_DONE
