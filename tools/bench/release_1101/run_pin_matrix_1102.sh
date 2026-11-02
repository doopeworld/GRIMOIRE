#!/bin/bash
# GPU0, one job at a time; stops at the first failure. Flash-Next short case on the pool branch binaries:
# the decode thread pinned (engine default) or free, against loads a desktop could really have.
#   pin0  = one process stuck on the decode thread's own core (worst case)
#   freeN = N ordinary processes the scheduler may place anywhere
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=${POOL_TREE:-/mnt/storage/isos/grimoire-cxpool-1102/tree}
TAG=${MATRIX_TAG:-pm}
cd "$B" || exit 9
exec >"$B/logs/pin-matrix-$TAG-1102.out" 2>&1
R=/mnt/storage/isos/grimoire-runs
exec 9>"$R/gpu0.lock"
end() { echo "PIN_MATRIX_END rc=$1 $(date +%T)"; exit "$1"; }
flock -n 9 || { echo "gpu0.lock held"; end 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then echo "renderD128 busy"; end 75; fi
n=$(docker ps --format "{{.Names}}" | wc -l); [ "$n" = 0 ] || { echo "containers running: $(docker ps --format "{{.Names}}")"; end 9; }
cd /mnt/storage/isos/grimoire-fuse || end 9
SHORT="Explain in two sentences why the sky is blue."
txt() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
one() { # NAME PINCALLER LOAD
  local name=$1 pc=$2 load=$3 pids=() c i
  case $load in
    pin0) taskset -c 0 sh -c "while :; do :; done" & pids+=($!) ;;
    free*) for i in $(seq 1 "${load#free}"); do sh -c "while :; do :; done" & pids+=($!); done ;;
  esac
  local env="GRIMOIRE_EXPERT_VRAM_PER_LAYER=112
GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin
GRIMOIRE_CPU_PIN_CALLER=$pc"
  GRIM_TREE=$T EXTRA_ENV="$env" LIM=900 bash tools/g0run.sh rx-$name -m /models2/Qwen3.8-Flash-Next-NVFP4 \
    --proj bf16 --ctx 8192 -p "$SHORT" -n 64 9>&- >/dev/null 2>&1
  local rc=$?
  for c in "${pids[@]:-}"; do [ -n "$c" ] && kill "$c" 2>/dev/null; done
  cp "/tmp/grim-rx-$name.log" "$B/logs/fn-$name-1102.log"
  local e=$(grep -m1 -oE "elapsed=[0-9.]+s" "/tmp/grim-rx-$name.log")
  local c2=$(grep -oE "\([0-9.]+ ms each\)" "/tmp/grim-rx-$name.log" | tail -1)
  local same=differs
  cmp -s <(txt "/tmp/grim-rx-$name.log") <(txt "$R/regress-1102-1701/fn-s.log") && same=IDENTICAL
  echo "| $load | pin_caller=$pc | $e | $c2 | text $same | rc=$rc"
  [ "$rc" = 0 ] || end 9
}
for load in none pin0 free1 free4 free14; do
  one $TAG-$load-p1 1 $load
  one $TAG-$load-p0 0 $load
done
end 0
