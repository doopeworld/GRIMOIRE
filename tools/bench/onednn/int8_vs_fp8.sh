#!/bin/bash
# 10-29: INT8 final (oneDNN first for INT8 like FP8; the small-M INT8 decode as its fallback).
# Build, then
#   1. gpu0 smoke: --proj int8 + MTP, GRIMOIRE_SMALLM_VERIFY=1 (which kernels, how close)
#   2. two cards, MTP on, 8 slots: INT8 then FP8 -- coding probe + benchy pp4096 / tg128 c1 / c4
B=/mnt/user/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-fuse
NL='
'
echo "$(date '+%F %T') i8v4.sh (build int8-fast final; gpu0 smoke; two-card MTP int8 vs fp8)" >> $B/logs/claude-commands-1028.log
{
date '+%T build'
bash $B/tower_build.sh all > /tmp/i8v4b.txt 2>&1 || { echo "STOP: build failed"; grep -B2 -A6 "error:" /tmp/i8v4b.txt | head -30; exit 1; }
tail -1 /tmp/i8v4b.txt
date '+%T 1. gpu0 smoke'
(cd $R && EXTRA_ENV="GRIMOIRE_MTP=1${NL}GRIMOIRE_SPEC_STATS=1${NL}GRIMOIRE_SMALLM_VERIFY=1" LIM=900 bash tools/g0run.sh i8v4-sm -m /models2/Qwen3.8-27B-FP8 --proj int8 --ctx 2048 \
   -p "Write a Python function that returns the n-th Fibonacci number, with a short docstring and three unit tests." -n 64) || { echo "STOP: gpu0 run failed"; tail -8 /tmp/grim-i8v4-sm.log; exit 1; }
grep -h "smallm verify" /tmp/grim-i8v4-sm.log | head -12
grep -h "spec: " /tmp/grim-i8v4-sm.log | cut -c1-160
awk '/smallm verify/ && $NF+0 > 0.02 {bad=1} END {exit bad}' /tmp/grim-i8v4-sm.log || { echo "STOP: a small-M kernel is off by more than 2% of max|ref|"; exit 1; }
grep -q "smallm verify" /tmp/grim-i8v4-sm.log || { echo "STOP: no smallm verify lines"; exit 1; }
cd $B
RP="RP_WIN=1000 RP_RUN=1000000"
for p in int8 fp8; do
  echo "== $(date +%T) 2. two cards MTP --proj $p"
  env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh i8v4-$p TP /models2/Qwen3.8-27B-FP8 $p 18432 \
    "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=1 GRIMOIRE_SEQ_SLOTS=8" code,benchy --pp 4096 --tg 128 --concurrency 1 4 --runs 2 > logs/i8v4-$p.run 2>&1
  grep -E "\[code\].*trial|STOP|LEFT|ready=0" logs/i8v4-$p.run | cut -c1-170
  grep -E "^\| /" logs/i8v4-$p.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-14s %-18s\n", $3, $4}'
  grep -q STOP logs/i8v4-$p.run && exit 9
  sleep 3
done
for x in $(pgrep -f "http[.]server 8999"); do kill $x; done
echo "== $(date +%T) I8V4_DONE"
} > $B/logs/i8v4.outer 2>&1
