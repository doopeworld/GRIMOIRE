#!/bin/bash
# 10-27: two cards, decode, CPU pool pinning A/B.  First the CPU experts' ms per layer call
# (GRIMOIRE_Q4_TIMING, host-timed, printed every 64 tokens) pinned vs GRIMOIRE_CPU_PIN=0, then
# llama-benchy pinned / unpinned / pinned again (run-to-run noise is ~1 tok/s).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
RP="RP_WIN=1000 RP_RUN=1000000"
for v in pin:1 nopin:0; do
  t=${v%%:*}; p=${v##*:}
  env $RP LOAD_MAX=1500 ./dual.sh fn-$t-q4 TP $F bf16 65536 "$B $PLE GRIMOIRE_CPU_PIN=$p GRIMOIRE_Q4_TIMING=1" gen,pleready,gen > logs/fn-$t-q4.run 2>&1
  echo "== $t: CPU experts per layer call"
  grep -E "STOP|LEFT|ready=0" logs/fn-$t-q4.run
  grep -E "CPU experts:" logs/fn-$t-q4.server.log | head -4 | cut -c1-200
  grep -E "qwen4_exp decode per token" logs/fn-$t-q4.server.log | tail -4 | cut -c1-300
  sleep 3
done
for v in pin:1 nopin:0 pin2:1; do
  t=${v%%:*}; p=${v##*:}
  env $RP LOAD_MAX=1500 ./dual.sh fn-$t-tp TP $F bf16 65536 "$B $PLE GRIMOIRE_CPU_PIN=$p" gen,pleready,benchy --pp 2048 4096 --tg 128 --runs 2 > logs/fn-$t-tp.run 2>&1
  echo "== $t: llama-benchy"
  grep -E "\[gen\]|STOP|LEFT|ready=0" logs/fn-$t-tp.run | cut -c1-160
  grep -E "^\| /" logs/fn-$t-tp.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
  sleep 3
done
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_PIN_DONE
