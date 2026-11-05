#!/bin/bash
# 11-05 19:20: Flash-Next on TWO cards (the DUAL-FLASH-NEXT profile's settings: Megatron TP + expert-parallel tiered
# experts, ctx 65536), candidate build, the dense part BF16 (as served) vs INT8 (Ian: two cards INT8 / FP8):
#   bf16-170 : as the profile (170 experts a layer of the rank's 256 in VRAM)
#   int8-170 : the same placement: the dense part's bytes only
#   int8-185 : the VRAM the 8-bit dense part frees, given to experts
# Frozen hits copied to flash-next.hits.ab before every run (the profile's file is not touched). Three fixed answers
# compared with the bf16 run's; llama-benchy pp2048 / tg128. Holds logs/HOLD-fndual.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105b
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fn-dual-1105.out" 2>&1
echo "FN_DUAL_START $(date +%T)"
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fndual
trap 'rm -f "$B/logs/HOLD-fndual"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1"
for spec in "bf16-170 bf16 170" "int8-170 int8 170" "int8-185 int8 185"; do
  set -- $spec
  name=fnd1105-$1
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" TP "$F" "$2" 65536 "$E GRIMOIRE_EXPERT_VRAM_PER_LAYER=$3" \
    pleready,gen,benchy --pp 2048 --tg 128 --concurrency 1 --runs 2 > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp2048" "logs/$name.run" | cut -c1-170
  grep -iE "tiered experts|error|failed|out of memory|CPU experts in decode \(128" "logs/$name.server.log" | head -6 | cut -c1-220
  [ "$1" = bf16-170 ] || python3 - "$B/logs" "$name" <<'PY'
import json, sys
d, n = sys.argv[1], sys.argv[2]
try:
    b, a = (json.load(open(f"{d}/{x}.gen.json")) for x in ("fnd1105-bf16-170", n))
except Exception as e:
    print("no gen json", e); sys.exit()
for k in b:
    x, y = b[k]["text"], a.get(k, {}).get("text", "")
    i = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
    print("  %s vs bf16 %-8s %s" % (n, k, "SAME" if x == y else "first difference at character %d of %d" % (i, len(x))))
PY
done
cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
echo "FN_DUAL_END $(date +%T)"
