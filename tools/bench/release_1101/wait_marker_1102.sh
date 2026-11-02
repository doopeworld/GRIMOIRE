#!/bin/bash
# wait_marker_1102.sh FILE MARKER [TAIL_LINES]: wait until a line starts with MARKER, then print the end of FILE.
f=$1 m=$2 n=${3:-60}
until grep -q "^$m" "$f" 2>/dev/null; do sleep 5; done
tail -n "$n" "$f"
