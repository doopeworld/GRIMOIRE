#!/bin/bash
# 11-06 evening: the heat engine (radiance-heat-1106b, pinned homes + spread copies) in INTERLEAVED off / on pairs, so
# drift and the noisy first runs cancel: ONE card (gpu0), Flash-Next NVFP4, INT4 dense part, frozen hits before every
# run, gen + llama-benchy pp512 / tg128 x2.
#   120 experts a layer (the one-card profile shape): off / S=8, three pairs
#   60 a layer (a B60's room): off / S=16, two pairs
# heat-ab3 (single runs): 120 off 49.05 +-6.39, S=8 53.40, S=16 51.38, S=32 52.61; 60 off 40.67 +-3.09, S=16 46.76.
# Waits for idle GPUs and no HOLD files. Holds logs/HOLD-heatab4. Log: logs/heat-ab4-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/heat-ab4-1106.out" 2>&1
echo "HEAT_AB4_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
grep -qx HEAT2_BUILD_PASS logs/heat-build2-1106.out || { echo "HEAT_AB4_STOP no passing build"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-heatab4
trap 'rm -f "$B/logs/HOLD-heatab4"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
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
  grep -hE "expert heat engine: [0-9]+ steps|CPU experts in decode \(768|error|failed" "logs/$name.server.log" 2>/dev/null |
    tail -3 | cut -c1-200
}
for p in a b c; do
  run heat4-off-120-$p "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120"
  run heat4-s8-120-$p  "GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_EXPERT_HEAT=8"
done
for p in a b; do
  run heat4-off-60-$p  "GRIMOIRE_EXPERT_VRAM_PER_LAYER=60"
  run heat4-s16-60-$p  "GRIMOIRE_EXPERT_VRAM_PER_LAYER=60 GRIMOIRE_EXPERT_HEAT=16"
done
python3 - "$B/logs" <<'PY'
import re, sys, statistics as st
d = sys.argv[1]
def tg(n):
    try:
        for l in open(f"{d}/{n}.run"):
            m = re.match(r"\s*tg128\s+([\d.]+)", l)
            if m: return float(m.group(1))
    except Exception: pass
    return None
for tag, off, on, ps in (("120", "off-120", "s8-120", "abc"), ("60", "off-60", "s16-60", "ab")):
    a = [tg(f"heat4-{off}-{p}") for p in ps]; b = [tg(f"heat4-{on}-{p}") for p in ps]
    print("  %s a layer: off %s  on %s" % (tag, a, b))
    a = [x for x in a if x]; b = [x for x in b if x]
    if a and b: print("    mean off %.2f  on %.2f  (%+.1f%%)" % (st.mean(a), st.mean(b), 100 * (st.mean(b) / st.mean(a) - 1)))
PY
echo "HEAT_AB4_END $(date +%T)"
