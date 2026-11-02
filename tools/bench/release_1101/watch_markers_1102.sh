#!/bin/bash
# Read only. One line per new marker in the logs named on the command line; ends when every one of the
# END markers has appeared or after 29 minutes. Also says when a job vanished without its end marker.
#   watch_markers_1102.sh "END_REGEX_1 END_REGEX_2 ..." "SHOW_REGEX" log...
cd /mnt/storage/isos/grimoire-runs/bench-1022/logs || exit 9
ENDS=$1 SHOW=$2; shift 2
declare -A seen
stop=$((SECONDS+1740))
while [ "$SECONDS" -lt "$stop" ]; do
  while IFS= read -r l; do
    [ -n "${seen[$l]:-}" ] && continue
    seen[$l]=1; echo "$(date +%T) $l" | cut -c1-170
  done < <(grep -hE "$SHOW" "$@" 2>/dev/null)
  all=1
  for e in $ENDS; do grep -qhE "$e" "$@" 2>/dev/null || all=0; done
  if [ "$all" = 1 ]; then echo "$(date +%T) WATCH_ALL_ENDED"; exit 0; fi
  sleep 20
done
echo "$(date +%T) WATCH_TIMEOUT containers: $(docker ps --format "{{.Names}}" | tr "\n" " ")"
