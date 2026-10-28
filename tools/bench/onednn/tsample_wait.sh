#!/bin/bash
# 10-27: once the first fold benchy run's server is up (fn-fold-tp), sample the engine's busy
# threads (tsample.sh) through its decode phases.
S=/mnt/user/isos/grimoire-runs/bench-1022
for i in $(seq 1 360); do
  grep -q "ready=1" $S/logs/fn-fold-tp.run 2>/dev/null && break
  sleep 5
done
grep -q "ready=1" $S/logs/fn-fold-tp.run 2>/dev/null || exit 1
$S/tsample.sh 70 2 > $S/logs/tsample.out 2>&1
