#!/bin/bash
# GPU0, one server at a time: Flash-Next served by two trees under the same conditions. Same requests, same text expected.
#   run_fn_serve_1102.sh SUFFIX   runs old (integration binaries) then new (pool branch). Stops at any failure.
set -u
SUF=$1
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
export IMG=grimoire-b70:latest HOSTBIN=1 LOAD_MAX=900 CLIENT_CPUS=14,15
exec >"$B/logs/fn-serve-$SUF.out" 2>&1
F=/models2/Qwen3.8-Flash-Next-NVFP4
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=112 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_CX_STATS=1"
run() {
  local name=$1 tree=$2
  echo "== $(date +%T) $name tree=$tree builders=$(docker ps --format "{{.Names}}" | grep -c build) load=$(cut -d" " -f1 /proc/loadavg)"
  GRIM_TREE=$tree bash ./dual_engine_1102.sh "$name" G0 "$F" bf16 8192 "$E" pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2 > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|tokens .*s  |pp512|tg128|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-130
  grep -E "CPU experts in decode" "logs/$name.server.log" | tail -1 | cut -c1-190
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "FN_SERVE_STOP $name rc=$rc"; exit 9
  fi
}
run fn-serve-old-$SUF /mnt/storage/isos/grimoire-engine-1101
run fn-serve-new-$SUF /mnt/storage/isos/grimoire-cxpool-1102/tree
python3 - "$B/logs/fn-serve-old-$SUF.gen.json" "$B/logs/fn-serve-new-$SUF.gen.json" <<PY
import json,sys
a,b=json.load(open(sys.argv[1])),json.load(open(sys.argv[2]))
for k in a: print(k,"text old vs new:", "IDENTICAL" if a[k]["text"]==b[k]["text"] else "DIFFERS", "seconds %.2f -> %.2f"%(a[k]["seconds"],b[k]["seconds"]))
PY
echo "FN_SERVE_DONE $(date +%T)"
