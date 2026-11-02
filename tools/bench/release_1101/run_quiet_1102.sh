#!/bin/bash
# After every build has ended. GPU0, one job at a time; stops at the first failure.
#  1 pool bench, old and new, nothing else running: what the fix costs on a quiet machine
#  2 Flash-Next short case, CLI: old and new, quiet, then with one busy core (worker 9, or the decode thread core 0)
#  3 Flash-Next served: old then new, quiet
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
OLD=/mnt/storage/isos/grimoire-engine-1101
NEW=/mnt/storage/isos/grimoire-cxpool-1102/tree
cd "$B" || exit 9
exec >"$B/logs/quiet-1102.out" 2>&1
end() { echo "QUIET_END rc=$1 $(date +%T)"; exit "$1"; }
n=$(docker ps --format "{{.Names}}" | wc -l); [ "$n" = 0 ] || { echo "containers still running: $(docker ps --format "{{.Names}}")"; end 9; }
echo "== $(date +%T) pool bench quiet"
for t in old new old new; do bash ./cxb_ab_1102.sh $t "" || end 9; done
ab() { # TREE TAG "busy cores"
  local pids=() c rc
  for c in $3; do taskset -c "$c" sh -c "while :; do :; done" & pids+=($!); done
  bash ./run_fn_ab_1102.sh "$1" "$2"; rc=$?
  for c in "${pids[@]:-}"; do [ -n "$c" ] && kill "$c" 2>/dev/null; done
  echo "--- $2 busy=[${3:-none}] rc=$rc"
  grep -E "elapsed=|CPU experts: [0-9]+ layer calls|text vs" "logs/fn-ab-$2-1102.out" | cut -c1-120
  [ "$rc" = 0 ] || end 9
}
echo "== $(date +%T) Flash-Next CLI"
ab "$OLD" old-quiet ""
ab "$NEW" new-quiet ""
ab "$OLD" old-busy9 "9"
ab "$NEW" new-busy9 "9"
ab "$OLD" old-busy0 "0"
ab "$NEW" new-busy0 "0"
echo "== $(date +%T) Flash-Next served, quiet"
bash ./run_fn_serve_1102.sh quiet; rc=$?
cut -c1-170 logs/fn-serve-quiet.out
[ "$rc" = 0 ] || end 9
end 0
