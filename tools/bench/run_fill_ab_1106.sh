#!/bin/bash
# 11-06 evening: GRIMOIRE_EXPERT_FILL (the VRAM the load left holds more experts, the RAM tier's most routed by the hits
# file; on by default with the heat engine's pinned homes) on the release tree grimoire-release-1106 (its own build,
# HOSTBIN), ONE card (gpu0), Flash-Next NVFP4, INT4 dense part, 120 a layer at load, heat engine at its default (on, 8),
# frozen hits before every run: interleaved fill off / on pairs, gen + llama-benchy pp512 / pp4096 / tg128 x2 (a long
# prompt: the fill leaves room for prompt chunks of 8192 rows -- check pp4096 did not lose).
# Waits for idle GPUs and no HOLD files. Holds logs/HOLD-fillab. Log: logs/fill-ab-1106.out.
#   run_fill_ab_1106.sh WANT   (the release tree's commit)
set -u
WANT=${1:?commit}
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1106
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fill-ab-1106.out" 2>&1
echo "FILL_AB_START $(date +%T) want=$WANT"
until grep -q "source=$WANT" logs/release-build-1106.out 2>/dev/null &&
      grep -qE "^RELEASE_BUILD_(PASS|FAIL)" logs/release-build-1106.out; do sleep 20; done
grep -qx RELEASE_BUILD_PASS logs/release-build-1106.out || { echo "FILL_AB_STOP build failed"; grep -m8 error logs/release-build-1106.out; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fillab
trap 'rm -f "$B/logs/HOLD-fillab"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1"
run() { # name env
  local name=$1 env=$2
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$env]"
  bash ./dual_engine_1102.sh "$name" G0 "$F" int4 8192 "$E $env" pleready,gen,benchy --pp 512 4096 --tg 128 \
    --concurrency 1 --runs 2 > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp512|pp4096" "logs/$name.run" | cut -c1-170
  grep -hE "expert heat engine: on|expert fill|prefill chunk up to|memory:|WARNING|expert heat engine: [0-9]+ steps|CPU experts in decode \(1[0-9]{3}|error|failed" \
    "logs/$name.server.log" 2>/dev/null | tail -7 | cut -c1-220
}
for p in a b; do
  run fill-off-$p "GRIMOIRE_EXPERT_FILL=0"
  run fill-on-$p  ""
done
python3 - "$B/logs" <<'PY'
import json, re, sys
d = sys.argv[1]
def rate(n, key):
    try:
        for l in open(f"{d}/{n}.run"):
            m = re.match(r"\s*%s\s+([\d.]+)" % key, l)
            if m: return float(m.group(1))
    except Exception: pass
    return None
for key in ("pp512", "pp4096", "tg128"):
    print("  %-6s off %s  on %s" % (key, [rate(f"fill-off-{p}", key) for p in "ab"], [rate(f"fill-on-{p}", key) for p in "ab"]))
try:
    a, b = (json.load(open(f"{d}/fill-{x}-a.gen.json")) for x in ("off", "on"))
    for k in a: print("  gen %-8s %s" % (k, "SAME" if a[k]["text"] == b.get(k, {}).get("text") else "reworded"))
except Exception as e: print("  no gen json", e)
PY
echo "FILL_AB_END $(date +%T)"
