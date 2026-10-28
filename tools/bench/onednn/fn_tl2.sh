#!/bin/bash
# 10-28: two cards, Flash-Next decode timeline in steady state: the same as fn_tl.sh, but the
# answers start only after the PLE table is in RAM (fn-tl on 10-27/10-28 dumped a token read
# while the copy was still running: L01 ple 9.6 ms).  GRIMOIRE_TIMELINE=1 + the small
# all-reduce timer; the dump is the 4th decode step of the first request.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
RP="RP_WIN=1000 RP_RUN=1000000"
env $RP LOAD_MAX=1500 ./dual.sh fn-tl2 TP $F bf16 65536 "$B $PLE GRIMOIRE_TIMELINE=1 GRIMOIRE_TP_SMALL_AR_TIME=1 GRIMOIRE_CX_STATS=1" pleready,gen > logs/fn-tl2.run 2>&1
grep -E "\[gen\]|STOP|LEFT|PLE table in memory" logs/fn-tl2.run | cut -c1-160
grep -E "TOTAL|device span|sum of|layer marks only" logs/fn-tl2.server.log | head -24 | cut -c1-160
grep -E "TP small all-reduce \(BAR" logs/fn-tl2.server.log | tail -2 | cut -c1-330
grep -E "CPU experts in decode" logs/fn-tl2.server.log | tail -2 | cut -c1-330
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_TL2_DONE
