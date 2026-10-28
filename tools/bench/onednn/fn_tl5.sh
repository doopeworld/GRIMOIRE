#!/bin/bash
# 10-28: fn_tl2.sh with the per-kernel marks of layers 3 (full attention) and 4 (DeltaNet)
# (GRIMOIRE_TIMELINE_LAYER / _LAYER2) -- which kernels of a layer are slow.  No all-reduce
# timer this time (it adds waits); PLE table in RAM first.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
RP="RP_WIN=1000 RP_RUN=1000000"
env $RP LOAD_MAX=1500 ./dual.sh fn-tl5 TP $F bf16 65536 "$B $PLE GRIMOIRE_TIMELINE=1 GRIMOIRE_CX_STATS=1" pleready,gen > logs/fn-tl5.run 2>&1
grep -E "\[gen\]|STOP|LEFT|PLE table in memory" logs/fn-tl5.run | cut -c1-160
grep -E "TOTAL|device span|layer marks only" logs/fn-tl5.server.log | head -6 | cut -c1-160
for r in 0 1; do echo "== rank $r, layers 3 and 4, step by step (us)"
  grep -E "^\[rank$r\]   L0[34] " logs/fn-tl5.server.log | awk '{printf "%s %s %s | ", $2, $3, $4} NR%6==0 {print ""}'; echo; done
grep -E "CPU experts in decode" logs/fn-tl5.server.log | tail -2 | cut -c1-330
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_TL5_DONE
