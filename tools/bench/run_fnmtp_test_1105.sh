#!/bin/bash
# 11-05: first run of the Flash-Next MTP head (branch flashnext-mtp, written 10-28, never run; merged as fnmtp-1105
# 1058c74), ONE card (gpu0), while the GPUs are otherwise idle (release build on the CPU). Holds logs/HOLD-fnmtp.
#   1. shadow: GRIMOIRE_MTP=1 GRIMOIRE_MTP4_SHADOW=1 -- the head drafts the next token at every step, the log counts
#      how often the target produced it; no speculation, the answers are the target's (fn_mtp1.sh, one card)
#   2. speculation, the same placement: base (no MTP) vs GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 -- the three fixed answers
#      compared (a batched verify sums in another order: count differences, read them), tg128 one user
# Placement: the frozen 10-28 hits file copied to flash-next.hits.ab before every run (never the profile's file);
# 100 experts a layer in VRAM (one card holds 112 without the head, which takes ~1.5 GB).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/fnmtp-1105
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fnmtp-test-1105.out" 2>&1
echo "FNMTP_TEST_START $(date +%T)"
until grep -qE "^FNMTP_BUILD_(PASS|FAIL)" logs/fnmtp-build-1105.out 2>/dev/null; do sleep 20; done
grep -qx FNMTP_BUILD_PASS logs/fnmtp-build-1105.out || { echo "FNMTP_TEST_STOP build failed"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fnmtp
trap 'rm -f "$B/logs/HOLD-fnmtp"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=100 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1"
run() { # name env tests [benchy]
  local name=$1 env=$2; shift 2
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$env]"
  bash ./dual_engine_1102.sh "$name" G0 "$F" bf16 8192 "$E $env" "$@" > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128" "logs/$name.run" | cut -c1-170
  grep -iE "mtp4|MTP4 shadow|speculation|accepted|error|failed|refus" "logs/$name.server.log" | head -14 | cut -c1-220
}
run fnmtp1105-shadow "GRIMOIRE_MTP=1 GRIMOIRE_MTP4_SHADOW=1" pleready,gen
if grep -q "MTP4 shadow:" logs/fnmtp1105-shadow.server.log 2>/dev/null; then
  run fnmtp1105-base "" pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2
  run fnmtp1105-k3 "GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 GRIMOIRE_SPEC_STATS=1" pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2
  python3 - "$B/logs" <<'PY'
import json, sys
d = sys.argv[1]
def load(n):
    try: return json.load(open(f"{d}/{n}.gen.json"))
    except Exception as e: return {}
b, a = load("fnmtp1105-base"), load("fnmtp1105-k3")
for k in b:
    x, y = b[k]["text"], a.get(k, {}).get("text", "")
    n = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
    print("  k3 vs base %-8s %s" % (k, "SAME" if x == y else "first difference at character %d of %d" % (n, len(x))))
PY
else
  echo "no shadow statistics: the head did not draft (see the server log) -- no speculation run"
fi
cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
echo "FNMTP_TEST_END $(date +%T)"
