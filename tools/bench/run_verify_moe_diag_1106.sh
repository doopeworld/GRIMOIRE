#!/bin/bash
# 11-06 evening: the verify's MoE stage in detail (9d465de: GRIMOIRE_Q4_TIMING's MoE detail line), and two cheap cuts:
# the routes to the host by one ksend kernel (GRIMOIRE_VERIFY_KSEND) and narrow BF16 weights in one launch
# (GRIMOIRE_VERIFY_NARROW).  Flash-Next MTP K=3, fenced (GRIMOIRE_SPEC_TIME + GRIMOIRE_Q4_TIMING), gen only:
#   vm1-old / vm1-new   one card (gpu0, INT4 dense part, 120 a layer): both cuts off / on
#   vm2-old / vm2-new   two cards (TP, INT8, 170 a layer; gpu1 Gen3 x4): both cuts off / on
#   vm2-b               two cards, cuts on, unfenced: llama-benchy pp2048 / tg128 x2 (7130e23 HC build: 42.41)
# Waits for run_heat_build2_1106.sh <WANT> (grimoire-runs/radiance-heat-1106b). Frozen hits before every run.
# Holds logs/HOLD-vmoe. Log: logs/verify-moe-diag-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
PD=/mnt/storage/Models/grimoire-ple
WANT=${1:?commit the build must be at}
cd "$B" || exit 9
exec >>"$B/logs/verify-moe-diag-1106.out" 2>&1
echo "VM_START $(date +%T) want=$WANT"
until grep -q "source=$WANT" logs/heat-build2-1106.out 2>/dev/null && grep -qE "^HEAT2_BUILD_(PASS|FAIL)" logs/heat-build2-1106.out; do sleep 20; done
grep -qx HEAT2_BUILD_PASS logs/heat-build2-1106.out || { echo "VM_STOP build failed"; grep -m8 -E "error" logs/heat-build2-1106.out; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-vmoe
trap 'rm -f "$B/logs/HOLD-vmoe"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 GRIMOIRE_SPEC_STATS=1"
TT="GRIMOIRE_SPEC_TIME=1 GRIMOIRE_Q4_TIMING=1"
OLD="GRIMOIRE_VERIFY_KSEND=0 GRIMOIRE_VERIFY_NARROW=0"
run() { # name mode proj ctx env tests...
  local name=$1 mode=$2 proj=$3 ctx=$4 env=$5; shift 5
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$mode $proj $env]"
  bash ./dual_engine_1102.sh "$name" "$mode" "$F" "$proj" "$ctx" "$E $env" "$@" > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp2048|RxErr during" "logs/$name.run" | cut -c1-200
  grep -hE "spec time per step|qwen4_exp prefill M=4" "logs/$name.server.log" 2>/dev/null | tail -4 | cut -c1-260
  [ -e "logs/$name.STOP" ] && { echo "VM_STOP link guard: $(cat "logs/$name.STOP")"; exit 1; }
  return 0
}
run vm1-old G0 int4 8192 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 $TT $OLD" pleready,gen
run vm1-new G0 int4 8192 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 $TT" pleready,gen
run vm2-old TP int8 65536 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 $TT $OLD" pleready,gen
run vm2-new TP int8 65536 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 $TT" pleready,gen
run vm2-b TP int8 65536 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=170" pleready,benchy --pp 2048 --tg 128 --concurrency 1 --runs 2
echo "VM_END $(date +%T)"
