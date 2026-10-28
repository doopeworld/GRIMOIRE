#!/bin/bash
# 10-27: two cards, decode: CPU experts per layer call (GRIMOIRE_Q4_TIMING prints them every 64
# tokens, host-timed), then llama-benchy with today's build (ksend + no empty CPU-expert add).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
RP="RP_WIN=1000 RP_RUN=1000000"
env $RP LOAD_MAX=1500 ./dual.sh fn-pin-q4 TP $F bf16 65536 "$B $PLE GRIMOIRE_Q4_TIMING=1" gen,pleready,gen > logs/fn-pin-q4.run 2>&1
grep -E "STOP|LEFT|ready=0" logs/fn-pin-q4.run
grep -E "qwen4_exp decode per token" logs/fn-pin-q4.server.log | tail -4 | cut -c1-300
sleep 3
env $RP LOAD_MAX=1500 ./dual.sh fn-pin-tp TP $F bf16 65536 "$B $PLE" gen,pleready,benchy --pp 2048 4096 --tg 128 --runs 2 > logs/fn-pin-tp.run 2>&1
grep -E "\[gen\]|STOP|LEFT|ready=0" logs/fn-pin-tp.run | cut -c1-160
grep -E "^\| /" logs/fn-pin-tp.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_PIN_DONE
