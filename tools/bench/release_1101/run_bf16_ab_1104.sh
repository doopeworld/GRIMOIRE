#!/bin/bash
# Two B70s, Qwen3.8-27B-FP8 tensor parallel + MTP with the GRIMOIRE-DUAL-QWEN3.8 profile's settings, the
# tp-small-bf16-1104 build: the small all-reduce with fp32 partials (today's path) and with bf16 partials
# (GRIMOIRE_TP_SMALL_BF16=1), back to back, same binaries. GRIMOIRE_TP_SMALL_AR_TIME prints where each
# all-reduce's time goes. Waits for the test build, then removes logs/HOLD-v1104 so the v1.10.4 gate runs.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
trap 'rm -f "$B/logs/HOLD-v1104"' EXIT
until grep -qE "^BUILD_(PASS|FAIL)" logs/build-next-1104.out; do sleep 15; done
grep -qx BUILD_PASS logs/build-next-1104.out || { echo "AB_STOP build failed"; exit 9; }
until [ -z "$(docker ps -q)" ]; do sleep 5; done
export GRIM_TREE=/mnt/storage/isos/grimoire-next-1104 HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=1 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_TP_SMALL_AR_TIME=1"
for v in 0 1; do
  name=ab1104-qwen-tp-bf16-$v
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" TP /models2/Qwen3.8-27B-FP8 fp8 16384 "$E GRIMOIRE_TP_SMALL_BF16=$v" \
    code,conc4,benchy --pp 512 --tg 128 --concurrency 1 4 --runs 2 > "logs/$name.run" 2>&1
  rc=$?
  grep -E "ready=|small all-reduce: bf16|answers pass|Coherence|\| ~2000|pp512|tg128|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-150
  grep -E "TP small all-reduce \(BAR" "logs/$name.server.log" 2>/dev/null | tail -4 | cut -c1-220
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "AB_STOP $name rc=$rc"; exit 9
  fi
done
echo "AB_DONE $(date +%T)"
