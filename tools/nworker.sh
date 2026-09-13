#!/bin/bash
# N independent GRIMOIRE ranks inside the container nrun.sh set up.  Each
# is a separate process with its own Level Zero context, which is the
# distinction that matters for cards on separate links.
set -u

MODE="${1:?pp or tp}"; shift
N="${1:?world size}"; shift

if [ "$MODE" = pp ]; then
    RANK_VAR=GRIMOIRE_PP_RANK; WORLD_VAR=GRIMOIRE_PP_WORLD_SIZE
    SOCKET="${GRIMOIRE_PP_SOCKET:-/tmp/grimoire-pp.sock}"
else
    RANK_VAR=GRIMOIRE_TP_RANK; WORLD_VAR=GRIMOIRE_TP_WORLD_SIZE
    SOCKET="${GRIMOIRE_TP_SOCKET:-/tmp/grimoire-tp.sock}"
fi

# A pipeline chain opens one socket per adjacent pair, named base-1,
# base-2, ...; a stale one from a previous run makes a rank connect to
# nothing and hang until the container timeout.
rm -f "$SOCKET" "$SOCKET"-* /tmp/grimoire-"$MODE"-rank*.log

PIDS=()
# Later ranks listen, so start from the back.  Rank 0 retries anyway, but
# starting the listeners first keeps the logs readable when it goes wrong.
for (( r=N-1; r>=0; r-- )); do
    LOG=/tmp/grimoire-$MODE-rank$r.log
    env "$RANK_VAR=$r" "$WORLD_VAR=$N" "$@" >"$LOG" 2>&1 &
    PIDS[$r]=$!
done

FAIL=0
for (( r=0; r<N; r++ )); do
    RC=0
    wait "${PIDS[$r]}" || RC=$?
    EXITS[$r]=$RC
    [ "$RC" -ne 0 ] && FAIL=$RC
done

for (( r=0; r<N; r++ )); do
    echo "===== ${MODE^^} rank $r / GPU $r ====="
    sed -n '1,240p' "/tmp/grimoire-$MODE-rank$r.log"
done
printf '%s exits:' "${MODE^^}"
for (( r=0; r<N; r++ )); do printf ' rank%d=%d' "$r" "${EXITS[$r]}"; done
echo

exit "$FAIL"
