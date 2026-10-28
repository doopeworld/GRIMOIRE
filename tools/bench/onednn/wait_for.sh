#!/bin/bash
# wait_for.sh FILE PATTERN MAXSEC [TAIL]: wait until PATTERN appears in FILE (or MAXSEC pass), then print its tail.
f=$1; pat=$2; max=${3:-1800}; tl=${4:-40}
t=0
while [ $t -lt $max ]; do
  grep -qE "$pat" "$f" 2>/dev/null && break
  sleep 15; t=$((t + 15))
done
grep -qE "$pat" "$f" 2>/dev/null && echo "FOUND after ${t}s" || echo "TIMEOUT after ${t}s"
tail -n $tl "$f" | cut -c1-240
