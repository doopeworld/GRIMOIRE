#!/bin/bash
# Two independent Grimoire TP ranks in the known-good dual-XPU container.
set -u
SOCKET="${GRIMOIRE_TP_SOCKET:-/tmp/grimoire-tp.sock}"
R0_LOG=/tmp/grimoire-tp-rank0.log
R1_LOG=/tmp/grimoire-tp-rank1.log
rm -f "$SOCKET" "$R0_LOG" "$R1_LOG"

GRIMOIRE_TP_RANK=1 "$@" >"$R1_LOG" 2>&1 & P1=$!
GRIMOIRE_TP_RANK=0 "$@" >"$R0_LOG" 2>&1 & P0=$!
R0=0; R1=0
wait "$P0" || R0=$?
wait "$P1" || R1=$?
echo "===== TP rank 0 / first GPU ====="
sed -n '1,240p' "$R0_LOG"
echo "===== TP rank 1 / second GPU ====="
sed -n '1,240p' "$R1_LOG"
echo "TP exits: rank0=$R0 rank1=$R1"
if [ "$R0" -ne 0 ]; then exit "$R0"; fi
exit "$R1"
