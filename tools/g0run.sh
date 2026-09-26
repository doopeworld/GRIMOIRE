#!/bin/bash
# g0run.sh NAME ARGS... -- run bin/grimoire on gpu0 ONLY, then check gpu0's PCIe
# link for new AER errors and the kernel log for xe errors on it.  Stops (exit 9)
# at the first sign of trouble so nothing else gets queued behind it.
set -u
cd /mnt/storage/isos/grimoire-fuse
NAME="$1"; shift
BIN=/grimoire/bin/grimoire                      # or any /grimoire/bin/... probe
case "${1:-}" in /grimoire/*) BIN="$1"; shift ;; esac
aer() { for p in 0000:00:01.0 0000:01:00.0; do
          for f in aer_dev_correctable aer_dev_nonfatal aer_dev_fatal; do
            tr '\n' ' ' < /sys/bus/pci/devices/$p/$f | grep -oE "TOTAL_ERR_[A-Z]+ [0-9]+"; done; done | tr '\n' ' '; }
PCI=$(bash tools/gpunode.sh gpu0 2>&1 >/dev/null | grep -oE "0000:[0-9a-f:.]+")
[ "$PCI" = 0000:03:00.0 ] || { echo "gpu0 is $PCI, expected 0000:03:00.0 -- refusing" >&2; exit 9; }
before=$(aer); t0=$(date +%s)
GPU=gpu0 LIM=${LIM:-600} EXTRA_ENV="${EXTRA_ENV:-}" bash tools/tune.sh "$NAME" "$BIN" "$@" >/dev/null 2>&1
after=$(aer)
xerr=$(dmesg -T --since "@$t0" 2>/dev/null | grep -E "03:00.0|00:01.0|01:00.0" | grep -iE "error|wedge|reset|fail" | head -3)
if [ "$BIN" = /grimoire/bin/grimoire ]; then
    echo "$(grep -oE "prompt=[0-9]+ generated=[0-9]+ finish=[a-z]+ elapsed=[0-9.]+s|generation failed.*" /tmp/grim-$NAME.log | tail -1)"
else
    tail -n ${TAILN:-12} /tmp/grim-$NAME.log
fi
if [ "$before" != "$after" ] || [ -n "$xerr" ]; then
    echo "STOP: gpu0 link/driver errors: before[$before] after[$after] $xerr" >&2; exit 9
fi
