#!/bin/bash
# 10-27: two cards, Flash-Next, 170 experts/layer in VRAM per card.  A/B of the prompt levers:
#   base  = default (ring prefetch off, sub-chunks grow when memory allows)
#   pref  = GRIMOIRE_EXPERT_PREFETCH=1024 (next layer's first RAM experts copied after the MoE all-reduce)
#   cold  = GRIMOIRE_EP_COLD_SHARE=0.67 (rank 0 = gpu0, x8 link, takes 2/3 of the cold experts)
#   both  = pref + cold
# Each: the code probe first (the 51 GB PLE copy into RAM finishes meanwhile), then llama-benchy
# pp2048/pp4096 tg128 --runs 1.  No pipes on dual.sh (its output goes to a file).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 GRIMOIRE_BUDGET_DEBUG=1"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
RP="RP_WIN=1000 RP_RUN=1000000"
run() {  # run TAG EXTRA
  env $RP LOAD_MAX=1500 ./dual.sh fn-ab2-$1 TP $F bf16 65536 "$B $PLE $2" code,benchy --pp 2048 4096 --tg 128 --runs 1 > logs/fn-ab2-$1.run 2>&1
  echo "== $1 [$2]"
  grep -E "\[code\] \| ~|STOP|LEFT|ready=0|refus" logs/fn-ab2-$1.run | cut -c1-200
  grep -E "^\| " logs/fn-ab2-$1.benchy.md 2>/dev/null | awk -F'|' 'NR>2{printf "     %-8s %-18s\n", $3, $4}'
  grep -E "tiered experts:|card free at the end" logs/fn-ab2-$1.server.log | grep -E "rank" | sort -u | tail -6 | cut -c1-200
  sleep 3
}
run base ""
run pref "GRIMOIRE_EXPERT_PREFETCH=1024"
run cold "GRIMOIRE_EP_COLD_SHARE=0.67"
run both "GRIMOIRE_EXPERT_PREFETCH=1024 GRIMOIRE_EP_COLD_SHARE=0.67"
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_AB2_DONE
