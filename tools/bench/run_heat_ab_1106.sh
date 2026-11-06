#!/bin/bash
# 11-06: the heat engine (Radiance's runtime expert placement, GRIMOIRE_EXPERT_HEAT) on ONE card (gpu0), Flash-Next
# NVFP4 with the INT4 dense part, served: the load's placement (the frozen 10-28 hits) against the heat engine,
#   at 120 experts a layer in VRAM (the one-card profile shape, the head ~1.5 GB) and at 60 a layer (about what a
#   24 GB B60 holds beside the INT4 dense part -- the card where placement matters most):
#   h0-120 / h32-120 / h0-60 / h32-60     gen + llama-benchy pp512 / tg128 x2 (the heat engine learns from the
#   gen answers and the benchy warm-up before the measured runs)
# CPU-expert stats (GRIMOIRE_CX_STATS) say how many experts a decode token computed on the CPU; the heat engine prints
# its moves every 1024 steps. Frozen hits restored before every run. Holds logs/HOLD-heatab. Log: logs/heat-ab-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/heat-ab-1106.out" 2>&1
echo "HEAT_AB_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
until grep -qE "^HEAT_BUILD_(PASS|FAIL)" logs/heat-build-1106.out 2>/dev/null; do sleep 20; done
grep -qx HEAT_BUILD_PASS logs/heat-build-1106.out || { echo "HEAT_AB_STOP build failed"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-heatab
trap 'rm -f "$B/logs/HOLD-heatab"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
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
  grep -hE "expert heat engine|CPU experts in decode|tiered experts|error|failed" "logs/$name.server.log" 2>/dev/null |
    tail -6 | cut -c1-230
}
run heat1106-h0-120 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120"
run heat1106-h32-120 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_EXPERT_HEAT=32"
run heat1106-h0-60 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=60"
run heat1106-h32-60 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=60 GRIMOIRE_EXPERT_HEAT=32"
python3 - "$B/logs" <<'PY'
import json, sys
d = sys.argv[1]
for base, new in (("heat1106-h0-120", "heat1106-h32-120"), ("heat1106-h0-60", "heat1106-h32-60")):
    try:
        b, a = (json.load(open(f"{d}/{x}.gen.json")) for x in (base, new))
    except Exception as e:
        print("  no gen json", e); continue
    for k in b:
        x, y = b[k]["text"], a.get(k, {}).get("text", "")
        i = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
        print("  %s vs %s %-8s %s" % (new, base, k, "SAME" if x == y else "first difference at character %d of %d" % (i, len(x))))
PY
echo "HEAT_AB_END $(date +%T)"
