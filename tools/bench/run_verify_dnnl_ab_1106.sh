#!/bin/bash
# 11-06 evening: do the verify's small-M GEMMs lose to oneDNN's per-call cost at 4 rows?  The small-M verify path
# (0948a3d) sends FP8 / INT8 / BF16 weights with N >= 256 to oneDNN (GRIMOIRE_SMALLM_ONEDNN, on) -- ~8 GEMMs a layer,
# ~400 a step -- else to the DPAS small-M kernels.  Flash-Next MTP K=3, fenced stage split (GRIMOIRE_SPEC_TIME,
# GRIMOIRE_Q4_TIMING), gen only; runs after run_hc_rows_ab_1106.sh (same build, multi-row HC on in all runs):
#   one card (INT4 dense part: oneDNN takes only the BF16 router there)  vd1-dnnl0
#   two cards (INT8 dense part: oneDNN takes most projections)            vd2-dnnl0
# against hc1-rows / hc2-rows of the HC A/B (oneDNN on).  Holds logs/HOLD-vdnnl. Log: logs/verify-dnnl-ab-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/verify-dnnl-ab-1106.out" 2>&1
echo "VD_START $(date +%T)"
until grep -qE "^HCR_(STOP|END)" logs/hc-rows-ab-1106.out 2>/dev/null; do sleep 20; done
grep -q "^HCR_STOP" logs/hc-rows-ab-1106.out && { echo "VD_STOP the HC A/B stopped"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-vdnnl
trap 'rm -f "$B/logs/HOLD-vdnnl"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 GRIMOIRE_SPEC_STATS=1 GRIMOIRE_SPEC_TIME=1 GRIMOIRE_Q4_TIMING=1 GRIMOIRE_SMALLM_ONEDNN=0"
run() { # name mode proj ctx env
  local name=$1 mode=$2 proj=$3 ctx=$4 env=$5
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$mode $proj $env]"
  bash ./dual_engine_1102.sh "$name" "$mode" "$F" "$proj" "$ctx" "$E $env" pleready,gen > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|RxErr during" "logs/$name.run" | cut -c1-200
  grep -hE "spec time per step|qwen4_exp prefill M=4" "logs/$name.server.log" 2>/dev/null | tail -3 | cut -c1-200
  [ -e "logs/$name.STOP" ] && { echo "VD_STOP link guard: $(cat "logs/$name.STOP")"; exit 1; }
  return 0
}
run vd1-dnnl0 G0 int4 8192 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120"
run vd2-dnnl0 TP int8 65536 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=170"
echo "VD_END $(date +%T)"
