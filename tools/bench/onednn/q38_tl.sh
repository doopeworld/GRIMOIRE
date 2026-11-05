#!/bin/bash
# 10-28: Qwen3.8-27B-FP8, two cards, plain decode (no MTP): where one token goes (GRIMOIRE_TIMELINE=1:
# every layer, and step by step in the first DeltaNet and the first full-attention layer) plus the
# small all-reduce timer.  Same settings as q38_conc.sh (fp8, ctx 18432, 8 slots), one user.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
RP="RP_WIN=1000 RP_RUN=1000000"
env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh q38-tl TP /models2/Qwen3.8-27B-FP8 fp8 18432 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_TIMELINE=1 GRIMOIRE_TP_SMALL_AR_TIME=1" gen > logs/q38-tl.run 2>&1
grep -E "\[gen\]|STOP|LEFT" logs/q38-tl.run | cut -c1-160
L=logs/q38-tl.server.log
grep -E "^\[rank0\]   (TOTAL|device span|sum of|\(layer)" $L | cut -c1-120
echo "== rank 0: the detailed layers, then the first 8 layer totals"
grep -E "^\[rank0\]   " $L | sed -n '/device timeline/,/TOTAL/p' | grep -v -E "^\[rank0\]   L[0-9][0-9] " | head -60 | cut -c1-90
grep -E "^\[rank0\]   L[0-9][0-9] " $L | head -8 | cut -c1-90
grep -E "^\[rank0\]   (lm_head|final_norm|incr_pos|embed|start)" $L | head -6 | cut -c1-90
grep -E "TP small all-reduce" $L | tail -2 | cut -c1-330
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo Q38_TL_DONE
