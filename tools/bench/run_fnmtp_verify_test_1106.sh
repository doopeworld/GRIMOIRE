#!/bin/bash
# 11-06: the Flash-Next MTP verify on the small-M path (fnmtp-verify-1106), ONE card (gpu0), the one-card profile's
# dense part (--proj int4), 120 experts a layer in VRAM for every run (the head takes ~1.5 GB; 11-05: int4 + 140 a layer
# without the head = 45.75 tok/s). 11-05 baseline: MTP K=3 through the prompt path 5.27 tok/s against 25.36 plain
# (BF16 dense part), verify 394 ms a step. Holds logs/HOLD-fnv; waits for the build and for idle GPUs.
#   0. CPU pool alone, GPUs idle: tools/test_cpu_ffn_rows, all threads (a call per row vs one call, Flash-Next experts)
#   1. base       no MTP                                         gen + llama-benchy pp512 / tg128 c1 x2
#   2. k3         GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 (new verify)    gen + benchy; answers vs base
#   3. k3-time    + GRIMOIRE_SPEC_TIME=1 (fenced phases)          gen
#   4. k3-old     + GRIMOIRE_Q4_VERIFY_SMALLM=0 (11-05 path)      gen   -> verify ms old vs new, same config
#   5. k3-q4t     + GRIMOIRE_Q4_TIMING=1 (stage split, fenced)    gen
#   6. k3-dv      + GRIMOIRE_MTP_DRAFT_VOCAB=65536                 gen + benchy
#   7. k3-cx0     + GRIMOIRE_CX_ROWS=0 (a pool call per row)      benchy
# Placement: the frozen 10-28 hits file copied to flash-next.hits.ab before every run (never the profile's file).
# Runs 1 and 2 also write GRIMOIRE_ROUTE_TRACE (every MoE call's expert ids, ~1 KB a token) to
# $PD/route-fnv1106-{base,k3}.bin; bin/route_sim replays them at the end: RAM experts per decode token and layer
# for today's placement (the frozen hits, 120 a layer), the hindsight placement and caches that follow the
# routing (NInfer-Offload v0.4.0: +41-57% decode on Flash-Next from such a cache) -- before writing one.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/fnmtp-verify-1106
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
exec >>"$B/logs/fnmtp-verify-test-1106.out" 2>&1
echo "FNV_TEST_START $(date +%T)"
until grep -qE "^FNV_BUILD_(PASS|FAIL)" logs/fnmtp-verify-build-1106.out 2>/dev/null; do sleep 20; done
grep -qx FNV_BUILD_PASS logs/fnmtp-verify-build-1106.out || { echo "FNV_TEST_STOP build failed"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-fnv
trap 'rm -f "$B/logs/HOLD-fnv"' EXIT
echo "== $(date +%T) CPU pool alone (no GPU)"
docker run --rm --name grim-cxrows-1106 -v "$T":/grimoire -w /grimoire grimoire-dev:latest bash -lc '
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    ./bin/test_cpu_ffn_rows 0 40' 2>&1 | tail -8
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1"
K3="GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=3 GRIMOIRE_SPEC_STATS=1"
run() { # name env tests [benchy args]
  local name=$1 env=$2; shift 2
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "== $(date +%T) $name [$env]"
  bash ./dual_engine_1102.sh "$name" G0 "$F" int4 8192 "$E $env" "$@" > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "\[gen\]|STOP|LEFT RUNNING|ready=0|server stopped|tg128|pp512" "logs/$name.run" | cut -c1-170
  grep -E "spec:|draft|verify|commit|qwen4_exp prefill M=|QSA prefill|tiered experts|error|failed|refus" \
    "logs/$name.server.log" | grep -v "^ *$" | tail -12 | cut -c1-230
}
cmp_gen() { # name: its three answers against base
  python3 - "$B/logs" "$1" <<'PY'
import json, sys
d, n = sys.argv[1], sys.argv[2]
try:
    b, a = (json.load(open(f"{d}/{x}.gen.json")) for x in ("fnv1106-base", n))
except Exception as e:
    print("  no gen json", e); sys.exit()
for k in b:
    x, y = b[k]["text"], a.get(k, {}).get("text", "")
    i = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
    print("  %s vs base %-8s %s" % (n, k, "SAME" if x == y else "first difference at character %d of %d" % (i, len(x))))
PY
}
BENCHY="benchy --pp 512 --tg 128 --concurrency 1 --runs 2"
rm -f "$PD"/route-fnv1106-base.bin "$PD"/route-fnv1106-k3.bin
run fnv1106-base "GRIMOIRE_ROUTE_TRACE=/models/grimoire-ple/route-fnv1106-base.bin" pleready,gen,$BENCHY
run fnv1106-k3 "$K3 GRIMOIRE_ROUTE_TRACE=/models/grimoire-ple/route-fnv1106-k3.bin" pleready,gen,$BENCHY
cmp_gen fnv1106-k3
run fnv1106-k3-time "$K3 GRIMOIRE_SPEC_TIME=1" pleready,gen; cmp_gen fnv1106-k3-time
run fnv1106-k3-old "$K3 GRIMOIRE_SPEC_TIME=1 GRIMOIRE_Q4_VERIFY_SMALLM=0" pleready,gen; cmp_gen fnv1106-k3-old
run fnv1106-k3-q4t "$K3 GRIMOIRE_SPEC_TIME=1 GRIMOIRE_Q4_TIMING=1" pleready,gen
run fnv1106-k3-dv "$K3 GRIMOIRE_MTP_DRAFT_VOCAB=65536" pleready,gen,$BENCHY; cmp_gen fnv1106-k3-dv
run fnv1106-k3-cx0 "$K3 GRIMOIRE_CX_ROWS=0" pleready,$BENCHY
cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
echo "== $(date +%T) expert placement replay (CPU only)"
for n in base k3; do
  [ -s "$PD/route-fnv1106-$n.bin" ] || { echo "  no trace for $n"; continue; }
  docker run --rm --name grim-routesim-1106 --network none --cpuset-cpus 6-13 -e N="$n" -v "$T":/grimoire \
    -v "$PD":/ple:ro -w /grimoire grimoire-dev:latest bash -lc '
      set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
      ./bin/route_sim "/ple/route-fnv1106-$N.bin" 120 /ple/flash-next.hits.frozen-1028 22' 2>&1 | sed "s/^/  $n: /"
done
echo "FNV_TEST_END $(date +%T)"
