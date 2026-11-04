#!/bin/bash
# After the v1.10.4 gate ends (done or stopped), on a quiet machine, one GPU job at a time:
# 1. Two B70s, Qwen3.8-27B GPTQ-Int4 + MTP (the one-card flagship, never measured on two cards), Megatron TP,
#    the tp-small-bf16-1104 build (bf16 small all-reduce default), the one-card release table's tests.
# 2. The B580 comparison: GRIMOIRE v1.10.4's G21 build (grimoire-release-1102b) vs llama.cpp SYCL and Vulkan,
#    Ornith-1.5-35B-A3B, experts partly in RAM.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
until grep -qE "^V1104_ALL_(DONE|STOP)" "$B/logs/v1104-all.out"; do sleep 30; done
until [ -z "$(docker ps -q)" ]; do sleep 10; done
name=d1104-gptq-tp
echo "== $(date +%T) $name"
GRIM_TREE=/mnt/storage/isos/grimoire-next-1104 HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 \
  bash ./dual_engine_1102.sh $name TP /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_TP_SMALL_AR_TIME=1" \
  code,conc4,benchy --pp 4096 --tg 128 --concurrency 1 4 --runs 2 > logs/$name.run 2>&1
echo "   rc=$?"
grep -E "ready=|answers pass|Coherence|\| ~2000|pp4096|tg128|server stopped|LEFT RUNNING|STOP|error|failed" logs/$name.run | cut -c1-150
grep -E "TP small all-reduce \(BAR" logs/$name.server.log 2>/dev/null | tail -2 | cut -c1-220
until [ -z "$(docker ps -q)" ]; do sleep 10; done
GTREE=/mnt/storage/isos/grimoire-release-1102b NCPU=26 bash "$B/run_b580_1104.sh" v1104 grimoire sycl vulkan
