#!/bin/bash
# Two B70s with the settings of Ian's two dual profiles, one server at a time; stops at the first failure.
#   run_dual_1104.sh TAG CONFIG...   CONFIG = qwen (Qwen3.8-27B-FP8 tensor parallel + MTP) | fn (Flash-Next two cards)
# Binaries: IMG=<image> HOSTBIN=0 = that image's own; HOSTBIN=1 GRIM_TREE=<tree> = a build tree's bin/.
# DUAL_EXTRA="A=1 B=2" (11-05) is appended to the server environment of every config.
set -u
TAG=$1; shift
X=${DUAL_EXTRA:+ $DUAL_EXTRA}
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
export LOAD_MAX=${LOAD_MAX:-1500} CLIENT_CPUS=14,15
run() {
  local name=$1; shift
  echo "== $(date +%T) $name image=${IMG:-} hostbin=${HOSTBIN:-} tree=${GRIM_TREE:-}"
  bash ./dual_engine_1102.sh "$name" "$@" > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|answer:|answers pass|Coherence|pp4096|pp2048|tg128|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-140
  [ -e "logs/$name.server.log" ] && grep -E "CPU experts in decode" "logs/$name.server.log" | tail -1 | cut -c1-190
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "DUAL_STOP $name rc=$rc"; exit 9
  fi
}
for c in "$@"; do
  case $c in
    qwen) run "d-$TAG-qwen-tp" TP /models2/Qwen3.8-27B-FP8 fp8 16384 \
            "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=1 GRIMOIRE_SEQ_SLOTS=8$X" \
            code,conc4,benchy --pp 4096 --tg 128 --concurrency 1 4 --runs 2 ;;
    fn)   run "d-$TAG-fn-tp" TP /models2/Qwen3.8-Flash-Next-NVFP4 bf16 65536 \
            "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits GRIMOIRE_CX_STATS=1$X" \
            pleready,gen,benchy --pp 2048 --tg 128 --concurrency 1 --runs 2 ;;
    *) echo "unknown config $c"; exit 2 ;;
  esac
done
echo "DUAL_DONE $(date +%T)"
