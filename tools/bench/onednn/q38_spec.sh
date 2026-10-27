#!/bin/bash
# 10-27 (Ian): Qwen3.8-27B-FP8 on both cards, ONE user, with drafting: MTP (the model's own draft
# head) and DFlash2 (the separate drafter) -- llama-benchy pp 4096 / 8192 / 16384, tg 128.  Waits
# for the several-users run (q38_conc.sh) first.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
for i in $(seq 1 180); do grep -q Q38_DONE logs/q38_conc.out 2>/dev/null && break; sleep 10; done
sleep 5
RP="RP_WIN=1000 RP_RUN=1000000"
for v in mtp:"GRIMOIRE_MTP=1" df2:"GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2"; do
  t=${v%%:*}; e=${v#*:}
  env $RP LOAD_MAX=1500 ./dual.sh q38-$t TP /models2/Qwen3.8-27B-FP8 fp8 18432 "GRIMOIRE_POWER_LIMIT_W=0 $e" code,benchy --pp 4096 8192 16384 --tg 128 --runs 1 > logs/q38-$t.run 2>&1
  echo "== Qwen3.8-27B-FP8, 2 cards, 1 user, $t"
  grep -E "\[code\]|STOP|LEFT|ready=0" logs/q38-$t.run | cut -c1-200
  grep -E "^\|" logs/q38-$t.benchy.md 2>/dev/null
  sleep 3
done
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo Q38_SPEC_DONE
