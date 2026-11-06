#!/bin/bash
# 11-06 evening: the verify's hyper-connection mix over all rows at once (47f7a5c) -- Flash-Next MTP K=3, row by row
# (GRIMOIRE_HC_ROWS=0) vs all rows at once (default).  The multi-row kernels compute every row exactly as the
# one-token ones, so the answers must be IDENTICAL between each pair; the verify should lose the HC re-reads.
# Waits for run_heat_build2_1106.sh <47f7a5c or later> (grimoire-runs/radiance-heat-1106b). Frozen hits every run.
#   one card (gpu0, INT4 dense part, 120 a layer):     hc1-rows0 / hc1-rows   gen + spec / stage timing, then benchy
#   two cards (TP, INT8, 170 a layer; gpu1 Gen3 x4):   hc2-rows0 / hc2-rows   the same
# Holds logs/HOLD-hcrows. Log: logs/hc-rows-ab-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
PD=/mnt/storage/Models/grimoire-ple
WANT=${1:?commit the build must be at}
cd "$B" || exit 9
exec >>"$B/logs/hc-rows-ab-1106.out" 2>&1
echo "HCR_START $(date +%T) want=$WANT"
until grep -q "source=$WANT" logs/heat-build2-1106.out 2>/dev/null && grep -qE "^HEAT2_BUILD_(PASS|FAIL)" logs/heat-build2-1106.out; do sleep 20; done
grep -qx HEAT2_BUILD_PASS logs/heat-build2-1106.out || { echo "HCR_STOP build failed"; grep -m5 -E "error" logs/heat-build2-1106.out; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-hcrows
trap 'rm -f "$B/logs/HOLD-hcrows"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 GRIMOIRE_SPEC_STATS=1"
run() { # name mode proj ctx env tests...
  local name=$1 mode=$2 proj=$3 ctx=$4 env=$5; shift 5
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$mode $proj $env]"
  bash ./dual_engine_1102.sh "$name" "$mode" "$F" "$proj" "$ctx" "$E $env" "$@" > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp512|RxErr during" "logs/$name.run" | cut -c1-200
  grep -hE "spec time per step|qwen4_exp prefill M=4" "logs/$name.server.log" 2>/dev/null | tail -3 | cut -c1-200
  [ -e "logs/$name.STOP" ] && { echo "HCR_STOP link guard: $(cat "logs/$name.STOP")"; exit 1; }
  return 0
}
cmp() { # base new
  python3 - "$B/logs" "$1" "$2" <<'PY'
import json, sys
d, b0, n0 = sys.argv[1:4]
try:
    b, a = (json.load(open(f"{d}/{x}.gen.json")) for x in (b0, n0))
except Exception as e:
    print("  no gen json", e); sys.exit()
for k in b:
    x, y = b[k]["text"], a.get(k, {}).get("text", "")
    i = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
    print("  %s vs %s %-8s %s" % (n0, b0, k, "SAME" if x == y else "first difference at character %d of %d" % (i, len(x))))
PY
}
T1="GRIMOIRE_SPEC_TIME=1 GRIMOIRE_Q4_TIMING=1"
BQ="benchy --pp 512 --tg 128 --concurrency 1 --runs 2"
run hc1-rows0 G0 int4 8192 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_HC_ROWS=0 $T1" pleready,gen
run hc1-rows  G0 int4 8192 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 $T1" pleready,gen
cmp hc1-rows0 hc1-rows
run hc1-rows-b G0 int4 8192 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120" pleready,$BQ
run hc2-rows0 TP int8 65536 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 GRIMOIRE_HC_ROWS=0 $T1" pleready,gen
run hc2-rows  TP int8 65536 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 $T1" pleready,gen
cmp hc2-rows0 hc2-rows
run hc2-rows-b TP int8 65536 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=170" pleready,benchy --pp 2048 --tg 128 --concurrency 1 --runs 2
echo "HCR_END $(date +%T)"
