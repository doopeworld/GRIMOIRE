#!/bin/bash
# Compare the finished v1.10.4 regression with the 11-02 run of the v1.10.3 source (regress-1102-1701). Read only.
R=/mnt/storage/isos/grimoire-runs
L=$R/bench-1022/logs/regress-v1104.out
A=$(grep '^results: ' "$L" | tail -1 | cut -d' ' -f2)
B=$R/regress-1102-1701
txt() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
el() { grep -oE 'elapsed=[0-9.]+' "$1" | tail -1 | cut -d= -f2; }
echo "new: $A  files: $(ls "$A" | wc -l)   baseline files: $(ls "$B" | wc -l)"
same=0; diff=0; nobase=0; empty=0
out=$(mktemp)
for f in "$A"/*.log; do
  n=${f##*/}
  [ -f "$B/$n" ] || { nobase=$((nobase+1)); echo "NOBASE $n"; continue; }
  # An empty answer on both sides would compare as the same, so count it apart.
  [ -n "$(txt "$f")" ] || { empty=$((empty+1)); echo "EMPTY_ANSWER $n"; }
  if cmp -s <(txt "$f") <(txt "$B/$n"); then same=$((same+1)); else diff=$((diff+1)); echo "DIFF $n"; fi
  printf '%s %s %s\n' "$n" "$(el "$f")" "$(el "$B/$n")"
done > "$out"
grep -E '^(DIFF|NOBASE|EMPTY_ANSWER)' "$out"
echo "same=$same diff=$diff nobase=$nobase empty=$empty"
echo "--- largest time changes (new vs 11-02), then totals"
grep -E '\.log [0-9.]+ [0-9.]+$' "$out" | awk '{printf "%+.1f%% %s new=%s base=%s\n", 100*($2-$3)/$3, $1, $2, $3}' | sort -t% -k1 -g | sed -n '1,4p;$p'
grep -E '\.log [0-9.]+ [0-9.]+$' "$out" | awk '{a+=$2;b+=$3} END{printf "total elapsed new=%.1fs base=%.1fs\n", a, b}'
rm -f "$out"
echo "--- suite lines"
grep -E '^(ALL DONE|STOP:|REGRESSION_EXIT)|Sherlock n=24|text vs plain|hot-expert' "$L" | cut -c1-150
echo "--- gpu0 path AER"
for d in 00:01.0 01:00.0 02:01.0; do grep -h TOTAL /sys/bus/pci/devices/0000:$d/aer_dev_correctable /sys/bus/pci/devices/0000:$d/aer_dev_nonfatal /sys/bus/pci/devices/0000:$d/aer_dev_fatal | tr '\n' ' '; echo " <- $d"; done
echo "--- xe errors on gpu0 since boot"
dmesg | grep -E "xe 0000:03:00.0" | grep -iE "error|wedge|reset|fail|timedout" | tail -5 | cut -c1-160
echo "--- gpu now"
docker ps --format '{{.Names}} {{.Status}}'
fuser -v /dev/dri/renderD128 2>&1 | cut -c1-100
flock -n "$R/gpu0.lock" true && echo "gpu0 lock free" || echo "gpu0 lock HELD"
