#!/bin/bash
# Compare two regress_all_g0.sh result directories run by run: generated text (the suite's own
# txt() extraction) identical or not, and the elapsed time of each.
A=$1 B=$2
txt() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
el() { grep -oE "elapsed=[0-9.]+" "$1" | tail -1 | cut -d= -f2; }
same=0 diff=0
printf "%-12s %-10s %9s %9s\n" run text "$(basename $A)" "$(basename $B)"
for f in "$A"/*.log; do
  n=$(basename "$f" .log); g="$B/$n.log"
  [ -f "$g" ] || { printf "%-12s %-10s\n" "$n" "MISSING"; continue; }
  if cmp -s <(txt "$f") <(txt "$g"); then t=identical; same=$((same+1)); else t=DIFFERS; diff=$((diff+1)); fi
  printf "%-12s %-10s %9s %9s\n" "$n" "$t" "$(el "$f")" "$(el "$g")"
done
echo "identical: $same   differ: $diff"
