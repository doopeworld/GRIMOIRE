#!/bin/bash
# 11-06 evening: how many Flash-Next experts a layer ONE B70 should hold with the INT4 dense part (4.28 GiB of trunk +
# embedding after the 48 layers, 31.9 GiB card): the one-card profile still says 112 (set for a BF16 trunk) and the
# engine's default is min(16 GiB, 60% of the card) = ~135.  Every RAM expert of a token runs on the CPU (~0.15 ms a call,
# ~40 calls a token at 120), so more on the card should mean faster decode and less re-streaming in prompts.
#   vram-150-s8 / vram-180-s8 / vram-180-off   (heat-ab4 has 120 off / S=8 x3 for the base)
# Startup lines kept: the tiered split, the prefill chunk the VRAM budget allows, any slot-fit warning.
# Waits for idle GPUs and no HOLD files. Holds logs/HOLD-heatvram. Log: logs/heat-vram-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/heat-vram-1106.out" 2>&1
echo "HEAT_VRAM_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
grep -qx HEAT2_BUILD_PASS logs/heat-build2-1106.out || { echo "HEAT_VRAM_STOP no passing build"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-heatvram
trap 'rm -f "$B/logs/HOLD-heatvram"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1"
run() { # name env
  local name=$1 env=$2
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$env]"
  bash ./dual_engine_1102.sh "$name" G0 "$F" int4 8192 "$E $env" pleready,gen,benchy --pp 512 --tg 128 \
    --concurrency 1 --runs 2 > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp512" "logs/$name.run" | cut -c1-170
  grep -hE "tiered experts:|prefill chunk up to|memory:|WARNING|expert heat engine: [0-9]+ steps|CPU experts in decode \(768|error|failed" \
    "logs/$name.server.log" 2>/dev/null | grep -v "^\s*$" | tail -7 | cut -c1-200
}
run vram-150-s8  "GRIMOIRE_EXPERT_VRAM_PER_LAYER=150 GRIMOIRE_EXPERT_HEAT=8"
run vram-180-s8  "GRIMOIRE_EXPERT_VRAM_PER_LAYER=180 GRIMOIRE_EXPERT_HEAT=8"
run vram-180-off "GRIMOIRE_EXPERT_VRAM_PER_LAYER=180"
echo "HEAT_VRAM_END $(date +%T)"
