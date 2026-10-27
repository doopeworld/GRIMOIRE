#!/bin/bash
# 10-27: two cards, Flash-Next decode: where one token's ~31 ms goes (GRIMOIRE_TIMELINE=1: device time
# between region markers, host waits included), plus the small all-reduce timer.  The three greedy
# answers only (the dump is the 4th decode step of the first request).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
RP="RP_WIN=1000 RP_RUN=1000000"
env $RP LOAD_MAX=1500 ./dual.sh fn-tl TP $F bf16 65536 "$B $PLE GRIMOIRE_TIMELINE=1 GRIMOIRE_TP_SMALL_AR_TIME=1" gen > logs/fn-tl.run 2>&1
grep -E "\[gen\]|STOP|LEFT" logs/fn-tl.run | cut -c1-160
grep -E "rank0" logs/fn-tl.server.log | grep -E "TOTAL|device span|sum of|layer marks only|small all-reduce" | head -14 | cut -c1-160
grep -E "rank0" logs/fn-tl.server.log | grep -E "^\[rank0\]   (L0[0-3] |start|head|tail)" | head -14 | cut -c1-120
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_TL_DONE
