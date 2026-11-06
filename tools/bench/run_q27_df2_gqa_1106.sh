#!/bin/bash
# 11-06 evening: the DFlash2 drafter's block attention with GQA-shared K/V tiles (73f1b47) on two B70s (Qwen3.8-27B,
# FP8 checkpoint as --proj int8, Megatron TP, M = 8).  The kernel is bit-identical row for row, so the two runs must
# draft the same tokens: the spec lines (accepted / drafted / steps) must MATCH; 'attention convert' should drop.
#   q27g-old (GRIMOIRE_DFLASH_ATTN_GQA=0) / q27g-new   code probe + GRIMOIRE_DFLASH_TIME, then
#   q27g-b   new, untimed: code probe + llama-benchy pp512 / tg128 x2
# Waits for run_heat_build2_1106.sh <WANT> and for idle GPUs. Holds logs/HOLD-q27g. Log: logs/q27-df2-gqa-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
M=/models2/Qwen3.8-27B-FP8
WANT=${1:?commit the build must be at}
cd "$B" || exit 9
exec >>"$B/logs/q27-df2-gqa-1106.out" 2>&1
echo "Q27G_START $(date +%T) want=$WANT"
until grep -q "source=$WANT" logs/heat-build2-1106.out 2>/dev/null && grep -qE "^HEAT2_BUILD_(PASS|FAIL)" logs/heat-build2-1106.out; do sleep 20; done
grep -qx HEAT2_BUILD_PASS logs/heat-build2-1106.out || { echo "Q27G_STOP build failed"; grep -m8 error logs/heat-build2-1106.out; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-q27g
trap 'rm -f "$B/logs/HOLD-q27g"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 CODE_LENGTHS=4000 CODE_REPEATS=1
BASE="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_SPEC_STATS=1"
run() { # name env tests...
  local name=$1 env=$2; shift 2
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" TP "$M" int8 16384 "$BASE $env" "$@" > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "answer:|tests=|tg128|pp512|server stopped|STOP|RxErr during" "logs/$name.run" | cut -c1-200
  grep -hE "spec: DFlash2" "logs/$name.server.log" | grep rank0 | tail -2 | cut -c1-200
  grep -h -A15 "\[rank0\]   DFlash native timing call 30," "logs/$name.server.log" | grep -E "attention convert|TOTAL" | cut -c1-120
  [ -e "logs/$name.STOP" ] && { echo "Q27G_STOP link guard"; exit 1; }
  return 0
}
run q27g-old "GRIMOIRE_DFLASH_TIME=1 GRIMOIRE_DFLASH_ATTN_GQA=0" code
run q27g-new "GRIMOIRE_DFLASH_TIME=1" code
a=$(grep -hE "spec: DFlash2" logs/q27g-old.server.log | grep rank0 | sed 's/.*--//' | md5sum)
b=$(grep -hE "spec: DFlash2" logs/q27g-new.server.log | grep rank0 | sed 's/.*--//' | md5sum)
[ "$a" = "$b" ] && echo "  spec counts IDENTICAL (old vs GQA)" || echo "  spec counts DIFFER (old vs GQA)"
run q27g-b "" code,benchy --pp 512 --tg 128 --concurrency 1 --runs 2
echo "Q27G_END $(date +%T)"
