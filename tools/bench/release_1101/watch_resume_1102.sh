#!/bin/bash
# Read-only watcher: one line per event, exits when the regression and the release build have both ended.
# Events: an answer differs from 10-30, every 11 cases, any stop, a card vanishing, uncorrectable link errors,
# build milestones. A new watcher supersedes the old one, and state survives re-arming.
R=/mnt/storage/isos/grimoire-runs
L=$R/bench-1022/logs/regress-1102.out
RB=$R/bench-1022/logs/release-build-1101.out
BASE=$R/regress-1030-1300
S=$R/bench-1022/logs/watch-1102.state
PF=$R/bench-1022/logs/watch-1102.pid
touch "$S"
echo $$ > "$PF"
txt() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
emit() { grep -qxF "$1" "$S" && return; echo "$1" >> "$S"; echo "$1"; }
while :; do
  [ "$(cat "$PF")" = "$$" ] || exit 0

  # Regression: judge each finished case against the 10-30 answer, with the suite's own filter.
  OUT=$(grep -m1 '^results: ' "$L" 2>/dev/null | cut -d' ' -f2)
  if [ -n "$OUT" ] && [ -d "$OUT" ]; then
    now=$(date +%s)
    for f in "$OUT"/*.log; do
      [ -f "$f" ] || continue
      n=${f##*/}; n=${n%.log}
      grep -qxF "CASE $n" "$S" && continue
      [ $((now - $(stat -c %Y "$f"))) -ge 5 ] || continue
      echo "CASE $n" >> "$S"
      if [ -f "$BASE/$n.log" ]; then
        cmp -s <(txt "$f") <(txt "$BASE/$n.log") || echo "ANSWER_DIFFERS $n vs 10-30"
      else
        echo "NO_BASELINE $n"
      fi
      c=$(grep -c '^CASE ' "$S")
      [ $((c % 11)) -eq 0 ] && echo "PROGRESS $c/44 cases done at $(date +%H:%M)"
    done
  fi
  grep -q '^REGRESSION_REFUSED' "$L" 2>/dev/null && emit "$(grep -m1 '^REGRESSION_REFUSED' "$L" | cut -c1-200)"
  grep -q '^STOP:' "$L" 2>/dev/null && emit "$(grep -m1 '^STOP:' "$L" | cut -c1-220)"
  grep -q '^ALL DONE' "$L" 2>/dev/null && emit "REGRESSION_ALL_DONE"
  X=$(grep -m1 '^REGRESSION_EXIT=' "$L" 2>/dev/null)
  [ -n "$X" ] && emit "$X"

  # Build milestones and end.
  grep -qx 'RELEASE_CLI_BUILT' "$RB" 2>/dev/null && emit "RELEASE_CLI_BUILT"
  grep -qx 'RELEASE_SERVER_BUILT' "$RB" 2>/dev/null && emit "RELEASE_SERVER_BUILT"
  grep -q '^RELEASE_BUILD_FAIL' "$RB" 2>/dev/null && emit "$(grep -m1 '^RELEASE_BUILD_FAIL' "$RB" | cut -c1-200)"
  grep -qx 'RELEASE_BUILD_PASS' "$RB" 2>/dev/null && emit "RELEASE_BUILD_PASS"

  # Hardware: a card vanishing, or anything uncorrectable, on either card's path.
  for d in 0000:03:00.0 0000:07:00.0; do
    v=$(od -An -tx2 -N2 "/sys/bus/pci/devices/$d/config" 2>/dev/null | tr -d ' ')
    [ "$v" = 8086 ] || emit "GPU_GONE $d vendor=$v"
  done
  for p in 0000:00:01.0 0000:01:00.0 0000:00:06.0 0000:05:00.0; do
    for k in nonfatal fatal; do
      t=$(grep -h TOTAL "/sys/bus/pci/devices/$p/aer_dev_$k" 2>/dev/null | awk '{print $2}')
      [ "${t:-0}" = 0 ] || emit "LINK_UNCORRECTABLE $p $k=$t"
    done
  done

  if grep -q '^REGRESSION_EXIT=' "$L" 2>/dev/null && grep -qE '^RELEASE_BUILD_(PASS|FAIL)' "$RB" 2>/dev/null; then
    echo "WATCH_END $(date +%H:%M)"
    exit 0
  fi
  sleep 15
done
