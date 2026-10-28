#!/bin/bash
# 10-27 (Ian): Qwen3.8-27B-FP8 on both cards (Megatron TP, plain decode -- several users at once
# do not work with MTP / DFlash2 -- 8 sequence slots): llama-benchy pp 4096 / 8192 / 16384,
# tg 128, 1 / 4 / 8 users at once.  Waits for the Flash-Next chain (cxs_chain.sh) to finish first.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
for i in $(seq 1 120); do grep -qE "CHAIN_DONE|BUILD_FAILED|not started" logs/cxs_chain.out 2>/dev/null && break; sleep 10; done
sleep 5
RP="RP_WIN=1000 RP_RUN=1000000"
env $RP LOAD_MAX=1500 ./dual.sh q38-conc TP /models2/Qwen3.8-27B-FP8 fp8 18432 "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0 GRIMOIRE_SEQ_SLOTS=8" code,benchy --pp 4096 8192 16384 --tg 128 --concurrency 1 4 8 --runs 1 > logs/q38-conc.run 2>&1
echo "== Qwen3.8-27B-FP8, 2 cards, TP, 8 slots"
grep -E "\[code\]|STOP|LEFT|ready=0" logs/q38-conc.run | cut -c1-200
grep -E "scheduler:" logs/q38-conc.server.log | head -2 | cut -c1-200
grep -E "^\|" logs/q38-conc.benchy.md 2>/dev/null
grep -E "error|fail|500" logs/q38-conc.run | head -5 | cut -c1-200
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo Q38_DONE
