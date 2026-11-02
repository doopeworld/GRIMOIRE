#!/bin/bash
# GPU0, one server at a time, integration binaries: Ornith native GGUF, the same four coding prompts together and
# one at a time, native grouped experts off and on. Tells a model mistake from a batching effect. Stops at any failure.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
export GRIM_TREE=/mnt/storage/isos/grimoire-engine-1101 HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=900 CLIENT_CPUS=14,15
exec >"$B/logs/orn-seq-1102.out" 2>&1
S="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8"
OG=/models2/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf
OTOK=/mnt/storage/Models/Ornith-1.5-35B-A3B
run() {
  local name=$1 env=$2
  echo "== $(date +%T) $name [$env]"
  TOKDIR=$OTOK bash ./dual_engine_1102.sh "$name" G0 "$OG" native 32000 "$env" conc4,seq4 > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|answer|pass|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-150
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "ORN_SEQ_STOP $name rc=$rc"; exit 9
  fi
}
run orn-seq-off-1102 "$S"
run orn-seq-on-1102  "$S GRIMOIRE_MOE_NATIVE_SMALLM=1"
echo "ORN_SEQ_DONE $(date +%T)"
