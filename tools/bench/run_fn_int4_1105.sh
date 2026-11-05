#!/bin/bash
# 11-05 19:05 (Ian: focus Qwen3.8-27B + Flash-Next; one card INT4, two cards INT8/FP8): Flash-Next on ONE card with its
# dense part (attention, DeltaNet, shared expert -- everything but the NVFP4 routed experts and the BF16
# hyper-connections) as INT4 instead of BF16 (--proj int4), candidate build (grimoire-release-1105b). The BF16
# baseline with the same placement is fnmtp1105-base (tg128 25.36, 90 experts a layer in VRAM).
#   int4-90  : the same placement (90 a layer): the dense part's bytes per token only
#   int4-140 : the VRAM the smaller dense part frees, given to experts (140 a layer)
# The three fixed answers compared with the BF16 run's; frozen hits copied first. Holds logs/HOLD-fnint4.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105b
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fn-int4-1105.out" 2>&1
echo "FN_INT4_START $(date +%T)"
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fnint4
trap 'rm -f "$B/logs/HOLD-fnint4"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1"
for spec in "int4-90 90" "int4-140 140"; do
  set -- $spec
  name=fnq1105-$1
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" G0 "$F" int4 8192 "$E GRIMOIRE_EXPERT_VRAM_PER_LAYER=$2" \
    pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2 > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp512" "logs/$name.run" | cut -c1-170
  grep -iE "tiered experts|GiB free|dense|int4|error|failed|CPU experts in decode \(128" "logs/$name.server.log" | head -8 | cut -c1-220
  python3 - "$B/logs" "$name" <<'PY'
import json, sys
d, n = sys.argv[1], sys.argv[2]
try:
    b, a = (json.load(open(f"{d}/{x}.gen.json")) for x in ("fnmtp1105-base", n))
except Exception as e:
    print("no gen json", e); sys.exit()
for k in b:
    x, y = b[k]["text"], a.get(k, {}).get("text", "")
    i = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
    print("  %s vs bf16 %-8s %s" % (n, k, "SAME" if x == y else "first difference at character %d of %d" % (i, len(x))))
PY
done
cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
echo "FN_INT4_END $(date +%T)"
