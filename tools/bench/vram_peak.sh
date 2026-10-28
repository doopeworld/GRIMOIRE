#!/bin/bash
# vram_peak.sh SECONDS OUT -- VRAM in use on both B70s (xe debugfs vram0_mm "usage", the whole
# card, every process), sampled every second; the peak goes on the last line of OUT.
R=/mnt/storage/isos/grimoire-fuse
p0=$($R/tools/gpunode.sh gpu0 2>&1 >/dev/null | grep -oE '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | head -1)
p1=$($R/tools/gpunode.sh gpu1 2>&1 >/dev/null | grep -oE '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | head -1)
use() { awk '/usage:/{print $2; exit}' /sys/kernel/debug/dri/$1/vram0_mm; }
end=$((SECONDS + $1)); m0=0; m1=0
echo "# gpu0 $p0  gpu1 $p1  (bytes in use)" > "$2"
while [ $SECONDS -lt $end ]; do
  u0=$(use $p0); u1=$(use $p1)
  [ "$u0" -gt "$m0" ] && m0=$u0
  [ "$u1" -gt "$m1" ] && m1=$u1
  echo "$(date +%T) $u0 $u1" >> "$2"
  sleep 1
done
echo "peak gpu0 $((m0 >> 20)) MiB  gpu1 $((m1 >> 20)) MiB  total $(((m0 + m1) >> 20)) MiB" >> "$2"
