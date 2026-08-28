#!/bin/bash
# Runs inside a container which exposes exactly the two requested B70 render
# nodes. Each child is a separate process/Level Zero context; this is the
# important distinction for USB-C/USB4 external docks.
set -u

SOCKET="${GRIMOIRE_PP_SOCKET:-/tmp/grimoire-pp.sock}"
R0_LOG=/tmp/grimoire-pp-rank0.log
R1_LOG=/tmp/grimoire-pp-rank1.log
rm -f "$SOCKET" "$R0_LOG" "$R1_LOG"

GRIMOIRE_PP_RANK=1 "$@" >"$R1_LOG" 2>&1 &
P1=$!
GRIMOIRE_PP_RANK=0 "$@" >"$R0_LOG" 2>&1 &
P0=$!

R0=0; R1=0
wait "$P0" || R0=$?
wait "$P1" || R1=$?

echo "===== PP rank 0 / first GPU ====="
sed -n '1,240p' "$R0_LOG"
echo "===== PP rank 1 / second GPU ====="
sed -n '1,240p' "$R1_LOG"
echo "PP exits: rank0=$R0 rank1=$R1"

if [ "$R0" -ne 0 ]; then exit "$R0"; fi
exit "$R1"
