#!/bin/bash
# Resolve a stable GPU name to the CURRENT render node.
# Render-node minor numbers are NOT stable across reboots -- they flipped on
# 2026-08-25 (renderD129 was B70#0, became the iGPU). Always resolve by PCI addr.
#   gpunode.sh gpu0  -> B70 at 0000:03:00.0  (host link PCIe Gen4 x8)
#   gpunode.sh gpu1  -> B70 at 0000:07:00.0  (host link PCIe Gen4 x4)
#   gpunode.sh igpu  -> Arc iGPU at 0000:00:02.0
# Slots change when cards are moved: until 2026-09-25 gpu1 was 0000:0b:00.0
# (Gen4 x2) and 0000:07:00.0 held an Arc B580 (12 GB); on 2026-09-26 the
# B580 is out of the machine and the second B70 sits at 07:00.0.  gpu0/gpu1
# are therefore checked to BE a B70 (PCI device 0xe223) -- a slot that now
# holds anything else is refused instead of silently used.
set -u
case "${1:?gpu0|gpu1|igpu}" in
  gpu0) PCI=0000:03:00.0; WANT=0xe223 ;;
  gpu1) PCI=0000:07:00.0; WANT=0xe223 ;;
  igpu) PCI=0000:00:02.0; WANT= ;;
  *)    PCI="$1";         WANT= ;;
esac
L="/dev/dri/by-path/pci-${PCI}-render"
[ -e "$L" ] || { echo "no render node for $PCI" >&2; exit 2; }
if [ -n "$WANT" ]; then
    DEV=$(cat "/sys/bus/pci/devices/$PCI/device" 2>/dev/null || echo none)
    [ "$DEV" = "$WANT" ] || { echo "REFUSING: $PCI is device $DEV, not a B70 ($WANT)" >&2; exit 4; }
fi
N=$(basename "$(readlink -f "$L")")
ST=$(cat "/sys/bus/pci/devices/$PCI/power/runtime_status" 2>/dev/null || echo unknown)
if [ "$ST" != "active" ] && [ "$ST" != "suspended" ]; then
    echo "REFUSING: $PCI runtime_status=$ST (not healthy)" >&2; exit 3
fi
echo "$N"
