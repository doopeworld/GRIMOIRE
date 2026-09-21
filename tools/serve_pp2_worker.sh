#!/bin/bash
# Runs inside the two-card container: starts both pipeline stages of
# grimoire-server.  Rank 1 loads its layers and then follows rank 0
# (grimoire_pp_worker_loop); only rank 0 binds the HTTP port.
#
# Ranks start from the BACK, because later stages listen and earlier ones
# connect.  Rank 0 retries its connect for ten minutes anyway, so the
# order is about readable logs, not correctness.
set -u

MODEL="${1:?model dir}"
PORT="${2:?port}"
PROJ="${3:-fp8}"
CTX="${4:-8192}"

SOCKET="${GRIMOIRE_PP_SOCKET:-/tmp/grimoire-pp-serve.sock}"
rm -f "${SOCKET}"* 2>/dev/null || true

SRV=/grimoire/bin/grimoire-server
ARGS=(--model "$MODEL" --proj "$PROJ" --ctx "$CTX" --port "$PORT")

GRIMOIRE_PP_RANK=1 "$SRV" "${ARGS[@]}" 2>&1 | sed 's/^/[rank1] /' &
P1=$!
GRIMOIRE_PP_RANK=0 "$SRV" "${ARGS[@]}" 2>&1 | sed 's/^/[rank0] /' &
P0=$!

# The front end is the one that matters: if it exits, the worker is
# useless and holding a card, so take it down rather than leave it.
wait "$P0"; R0=$?
kill "$P1" 2>/dev/null || true
wait "$P1" 2>/dev/null || true
echo "front end exited $R0"
exit "$R0"
