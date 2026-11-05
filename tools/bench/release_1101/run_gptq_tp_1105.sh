#!/bin/bash
# 11-05: two B70s, Qwen3.8-27B GPTQ-Int4 + MTP (the one-card release row r1104-img-q38 settings), Megatron TP,
# binary = grimoire-next-1104/bin (044442a, server, G31) with GRIMOIRE_TP_SMALL_BF16=1 = the 35201c2 default.
# Runs beside the CPU-only v1.10.4 build (cores 6-13); the server is not pinned.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
name=d1105-gptq-tp
echo "== $(date +%T) $name"
GRIM_TREE=/mnt/storage/isos/grimoire-next-1104 HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 \
  bash ./dual_engine_1102.sh $name TP /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_TP_SMALL_BF16=1" \
  code,conc4,benchy --pp 4096 --tg 128 --concurrency 1 4 --runs 2 > logs/$name.run 2>&1
echo "   rc=$?"
grep -E "ready=|answers pass|Coherence|pp4096|tg128|server stopped|LEFT RUNNING|STOP|error|failed" logs/$name.run | cut -c1-150
grep -E "TP small all-reduce" logs/$name.server.log 2>/dev/null | tail -2 | cut -c1-220
echo "GPTQ_TP_DONE $(date +%T)"
