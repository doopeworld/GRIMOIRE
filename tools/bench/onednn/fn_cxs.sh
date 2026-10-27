#!/bin/bash
# 10-27: two cards, decode: the decode thread pinned to its rank's free P-core during decode steps
# (GRIMOIRE_CPU_PIN_CALLER, default on) vs left to the scheduler, with the CPU experts' real time
# in decode (GRIMOIRE_CX_STATS: decode calls only, no syncs added; the caller's wait for the last
# worker; a histogram of call times) and the engine's busy threads sampled (tsample2.sh).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
RP="RP_WIN=1000 RP_RUN=1000000"
for v in cpin:1 nocpin:0 cpin2:1; do
  t=${v%%:*}; p=${v##*:}
  ( for i in $(seq 1 360); do grep -q "ready=1" logs/fn-$t-tp.run 2>/dev/null && break; sleep 5; done
    ./tsample2.sh 40 1 > logs/tsample-$t.out 2>&1 ) &
  env $RP LOAD_MAX=1500 ./dual.sh fn-$t-tp TP $F bf16 65536 "$B $PLE GRIMOIRE_CX_STATS=1 GRIMOIRE_CPU_PIN_CALLER=$p" gen,pleready,benchy --pp 2048 4096 --tg 128 --runs 2 > logs/fn-$t-tp.run 2>&1
  wait
  echo "== $t (decode thread pinned: $p)"
  grep -E "\[gen\]|STOP|LEFT|ready=0" logs/fn-$t-tp.run | cut -c1-160
  grep -E "CPU experts: pool" logs/fn-$t-tp.server.log | head -2 | cut -c1-200
  grep -E "^\| /" logs/fn-$t-tp.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
  grep -E "CPU experts in decode" logs/fn-$t-tp.server.log | tail -2 | cut -c1-420
  sleep 3
done
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_CXS_DONE
