#!/bin/bash
# 10-27: where the engine's busy threads run during decode.  Every few seconds, for each
# grimoire-server process: threads that used more than 20% of a CPU over one second, with the CPU
# they last ran on (/proc/<pid>/task/<tid>/stat field 39) and their share of that second.
#   tsample.sh <samples> [gap s]
n=${1:-20}; gap=${2:-3}
hz=$(getconf CLK_TCK)
for s in $(seq 1 $n); do
  declare -A t0=()
  pids=$(pgrep -f "grimoire/bin/grimoire-server")
  [ -z "$pids" ] && { sleep $gap; continue; }
  for p in $pids; do
    for f in /proc/$p/task/*/stat; do
      read -r -a a < $f 2>/dev/null || continue
      t0["$p/${a[0]}"]=$(( a[13] + a[14] ))
    done
  done
  sleep 1
  out=""
  for p in $pids; do
    for f in /proc/$p/task/*/stat; do
      read -r -a a < $f 2>/dev/null || continue
      k="$p/${a[0]}"; [ -n "${t0[$k]}" ] || continue
      d=$(( a[13] + a[14] - t0[$k] ))
      pc=$(( d * 100 / hz ))
      [ $pc -gt 20 ] && out="$out ${a[0]}@cpu${a[38]}:${pc}%"
    done
  done
  echo "$(date +%T) pids $(echo $pids | tr '\n' ' ') busy:$out"
  unset t0
  sleep $gap
done
