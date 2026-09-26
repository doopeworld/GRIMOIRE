#!/bin/bash
# Resolve a stable GPU name to the CURRENT render node.
# Render-node minors and PCI slots both change (reboots, cards moved -- the
# Tower's layout changed three times between 2026-09-24 and 09-26), so the
# B70s are found by PCI DEVICE ID, not by slot:
#   gpunode.sh gpu0  -> the first  Arc Pro B70 (8086:e223) in PCI-address order
#   gpunode.sh gpu1  -> the second Arc Pro B70
#   gpunode.sh igpu  -> the Arc iGPU at 0000:00:02.0
#   gpunode.sh 0000:bb:dd.f -> that exact PCI function
# An Arc B580 (8086:e20b, 12 GB) or any other card is never handed out as
# gpu0/gpu1.  The chosen PCI address is printed on stderr.
set -u
b70s() {
    for d in /sys/bus/pci/devices/*; do
        [ "$(cat "$d/vendor" 2>/dev/null)" = 0x8086 ] || continue
        [ "$(cat "$d/device" 2>/dev/null)" = 0xe223 ] || continue
        case "$(cat "$d/class" 2>/dev/null)" in 0x030000|0x038000) basename "$d" ;; esac
    done | sort
}
case "${1:?gpu0|gpu1|igpu|PCI address}" in
  gpu0) PCI=$(b70s | sed -n 1p) ;;
  gpu1) PCI=$(b70s | sed -n 2p) ;;
  igpu) PCI=0000:00:02.0 ;;
  *)    PCI="$1" ;;
esac
[ -n "$PCI" ] || { echo "no such B70 ($1): found $(b70s | tr '\n' ' ')" >&2; exit 2; }
L="/dev/dri/by-path/pci-${PCI}-render"
[ -e "$L" ] || { echo "no render node for $PCI" >&2; exit 2; }
N=$(basename "$(readlink -f "$L")")
ST=$(cat "/sys/bus/pci/devices/$PCI/power/runtime_status" 2>/dev/null || echo unknown)
if [ "$ST" != "active" ] && [ "$ST" != "suspended" ]; then
    echo "REFUSING: $PCI runtime_status=$ST (not healthy)" >&2; exit 3
fi
echo "$1 -> $PCI ($N)" >&2
echo "$N"
