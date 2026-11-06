#!/bin/bash
# 11-06 (3): the heat engine with pinned homes (a demotion is a pointer flip, only host -> device copies) and its
# copies spread over the step's routed layer calls (radiance-heat-1106b, run_heat_build2_1106.sh), ONE card (gpu0),
# Flash-Next NVFP4, INT4 dense part, frozen hits before every run, gen + llama-benchy pp512 / tg128 x2:
#   off / 8 / 16 / 32 swaps a step at 120 experts a layer (the one-card profile shape), off / 16 at 60 (a B60's room).
# The first version (copies back to spare pinned slots, all issued at a step's start; heat-ab-1106 / heat-ab2):
# 120 a layer off 41.96, S=8 49.13, S=32 39.17; 60 a layer off 35.50, S=32 31.14 tok/s.
# Holds logs/HOLD-heatab3. Log: logs/heat-ab3-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/heat-ab3-1106.out" 2>&1
echo "HEAT_AB3_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
until grep -qE "^HEAT2_BUILD_(PASS|FAIL)" logs/heat-build2-1106.out 2>/dev/null; do sleep 20; done
grep -qx HEAT2_BUILD_PASS logs/heat-build2-1106.out || { echo "HEAT_AB3_STOP build failed"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-heatab3
trap 'rm -f "$B/logs/HOLD-heatab3"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
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
  grep -hE "expert heat engine|CPU experts in decode \((640|768)|tiered experts|error|failed" "logs/$name.server.log" 2>/dev/null |
    tail -6 | cut -c1-230
}
run heat3-h0-120  "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120"
run heat3-h8-120  "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_EXPERT_HEAT=8"
run heat3-h16-120 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_EXPERT_HEAT=16"
run heat3-h32-120 "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_EXPERT_HEAT=32"
run heat3-h0-60   "GRIMOIRE_EXPERT_VRAM_PER_LAYER=60"
run heat3-h16-60  "GRIMOIRE_EXPERT_VRAM_PER_LAYER=60 GRIMOIRE_EXPERT_HEAT=16"
python3 - "$B/logs" <<'PY'
import json, sys
d = sys.argv[1]
for base, new in (("heat3-h0-120", "heat3-h8-120"), ("heat3-h0-120", "heat3-h16-120"), ("heat3-h0-120", "heat3-h32-120"),
                  ("heat3-h0-60", "heat3-h16-60")):
    try:
        b, a = (json.load(open(f"{d}/{x}.gen.json")) for x in (base, new))
    except Exception as e:
        print("  no gen json", e); continue
    for k in b:
        x, y = b[k]["text"], a.get(k, {}).get("text", "")
        i = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
        print("  %s vs %s %-8s %s" % (new, base, k, "SAME" if x == y else "first difference at character %d of %d" % (i, len(x))))
PY
echo "HEAT_AB3_END $(date +%T)"
