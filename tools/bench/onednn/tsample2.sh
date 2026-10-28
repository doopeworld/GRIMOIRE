#!/bin/bash
# 10-27: per-thread user / kernel CPU of the engine processes over 2 s windows, with the CPU each
# thread last ran on; only threads above 10%.   tsample2.sh <samples> [gap s]
n=${1:-20}; gap=${2:-2}
snap() {
  for p in $(pgrep -f "^/grimoire/bin/grimoire-server"); do
    for t in $(ls /proc/$p/task 2>/dev/null); do
      awk -v p=$p -v t=$t '{print t, p, $14, $15, $39}' /proc/$p/task/$t/stat 2>/dev/null
    done
  done | sort
}
for s in $(seq 1 $n); do
  a=$(snap); sleep 2; b=$(snap)
  echo "== $(date +%T)"
  join <(echo "$a") <(echo "$b") | awk '{u=$7-$3; s=$8-$4; if (u+s>20) printf "  pid %s tid %s user %3d%% sys %3d%% cpu %s\n", $2, $1, u/2, s/2, $9}'
  sleep $gap
done
