#!/bin/bash
# Runs inside the two-card container: starts both ranks of grimoire-server.
# Rank 1 loads its share and then follows rank 0 (grimoire_pp_worker_loop);
# only rank 0 binds the HTTP port.
#
# GRIMOIRE_SERVE_MODE picks the split: PP (default, serve_pp2.sh -- rank 1
# holds the later layers) or TP (serve_tp2.sh -- both ranks hold a slice
# of every weight).  The worker loop and this script are the same for both;
# only the rank variable and the socket differ.
#
# Ranks start from the BACK, because later stages listen and earlier ones
# connect.  Rank 0 retries its connect for ten minutes anyway, so the
# order is about readable logs, not correctness.
set -u

MODE="${GRIMOIRE_SERVE_MODE:-PP}"
case "$MODE" in PP|TP) ;; *) echo "GRIMOIRE_SERVE_MODE must be PP or TP" >&2; exit 2 ;; esac

MODEL="${1:?model dir}"
PORT="${2:?port}"
PROJ="${3:-fp8}"
CTX="${4:-8192}"

if [ "$MODE" = TP ]; then
  SOCKET="${GRIMOIRE_TP_SOCKET:-/tmp/grimoire-tp-serve.sock}"
  export GRIMOIRE_TP_SOCKET="$SOCKET" GRIMOIRE_TP_WORLD_SIZE="${GRIMOIRE_TP_WORLD_SIZE:-2}"
else
  SOCKET="${GRIMOIRE_PP_SOCKET:-/tmp/grimoire-pp-serve.sock}"
fi
rm -f "${SOCKET}"* 2>/dev/null || true
RANKVAR="GRIMOIRE_${MODE}_RANK"

SRV=/grimoire/bin/grimoire-server
ARGS=(--model "$MODEL" --proj "$PROJ" --ctx "$CTX" --port "$PORT")

# `cmd | sed &` makes $! the PID of SED, not of the server: the exit status
# below was sed's (always 0, so a crashed front end reported success -- the
# same shape as the b70run.sh bug in CLAUDE.md rule 21), and `kill "$P1"`
# stopped rank 1's log filter rather than rank 1.  Process substitution
# keeps the prefixed logs and makes $! the server itself.
env "$RANKVAR=1" "$SRV" "${ARGS[@]}" > >(sed 's/^/[rank1] /') 2>&1 &
P1=$!
env "$RANKVAR=0" "$SRV" "${ARGS[@]}" > >(sed 's/^/[rank0] /') 2>&1 &
P0=$!

# `docker stop` reaches this script (tini forwards to its child, which is
# this bash), not the servers.  Without a trap bash dies on the TERM, tini
# exits, and the kernel SIGKILLs both ranks when the container's PID 1 goes
# away.  Hand the signal to the servers and keep waiting for them.
trap 'kill -TERM "$P0" "$P1" 2>/dev/null' TERM INT

# Wait for a PID to EXIT and return its status.  A trapped signal makes
# `wait` return early (status > 128) while the process is still running.
reap() {
  local st=0
  wait "$1"; st=$?
  while kill -0 "$1" 2>/dev/null; do wait "$1"; st=$?; done
  return "$st"
}

# The front end is the one that matters: if it exits, the worker is
# useless and holding a card, so take it down rather than leave it.
reap "$P0"; R0=$?
kill -TERM "$P1" 2>/dev/null || true
reap "$P1" || true
echo "front end exited $R0"
exit "$R0"
