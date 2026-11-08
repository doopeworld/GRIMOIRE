#!/bin/bash
# 11-06 evening: Flash-Next on TWO cards as a PIPELINE (each card 24 of the 48 layers, no all-reduce per layer) instead
# of Megatron TP + expert parallel. Why now: one card with the heat engine + more experts on the card (58.0 tok/s at 180
# a layer) beats two cards TP (52.9 at 170 a layer each: 96 all-reduces a token at ~26 us + waiting for the slower
# rank). A stage holds half the layers, so ~twice the experts a layer fit (here 340 of 512), and its CPU experts never
# run at the same time as the other stage's. Release tree grimoire-release-1106 (HOSTBIN), INT8 dense part (the
# two-card scope), frozen hits:
#   fnpp-340      PP, 340 a layer, the load's placement (heat off with several ranks by default)
#   fnpp-340-h8   + GRIMOIRE_EXPERT_HEAT=8 GRIMOIRE_EXPERT_HEAT_TP=1 (each stage moves its own layers' experts)
# gen + llama-benchy pp512 / pp2048 / tg128 x2. Link guard as every two-card run (gpu1 at Gen3 x4 today).
# Waits for idle GPUs and no HOLD files. Holds logs/HOLD-fnpp. Log: logs/fn-pp-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1106
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fn-pp-1106.out" 2>&1
echo "FN_PP_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
grep -qx RELEASE_BUILD_PASS logs/release-build-1106.out || { echo "FN_PP_STOP no passing build"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fnpp
trap 'rm -f "$B/logs/HOLD-fnpp"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PP_LAYERS=24,24 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1"
run() { # name env
  local name=$1 env=$2
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$env]"
  bash ./dual_engine_1102.sh "$name" PP "$F" int8 8192 "$E $env" pleready,gen,benchy --pp 512 2048 --tg 128 \
    --concurrency 1 --runs 2 > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp512|pp2048|RxErr during" "logs/$name.run" | cut -c1-170
  grep -hE "tiered experts:|expert heat engine: (on|off)|expert heat engine: [0-9]+ steps|CPU experts in decode \(1[0-9]{3}|prefill chunk|error|failed" \
    "logs/$name.server.log" 2>/dev/null | tail -8 | cut -c1-200
  [ -e "logs/$name.STOP" ] && { echo "FN_PP_STOP link guard"; exit 1; }
  return 0
}
run fnpp-340    "GRIMOIRE_EXPERT_VRAM_PER_LAYER=340"
run fnpp-340-h8 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=340 GRIMOIRE_EXPERT_HEAT=8 GRIMOIRE_EXPERT_HEAT_TP=1"
echo "FN_PP_END $(date +%T)"
