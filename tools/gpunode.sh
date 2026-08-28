#!/bin/bash
# Resolve a stable GPU name to the CURRENT render node.
# Render-node minor numbers are NOT stable across reboots -- they flipped on
# 2026-08-25 (renderD129 was B70#0, became the iGPU). Always resolve by PCI addr.
#   gpunode.sh gpu0  -> B70 at 0000:03:00.0
#   gpunode.sh gpu1  -> B70 at 0000:35:00.0
#   gpunode.sh igpu  -> Arc iGPU at 0000:00:02.0
set -u
case "${1:?gpu0|gpu1|igpu}" in
  gpu0) PCI=0000:03:00.0 ;;
  gpu1) PCI=0000:35:00.0 ;;
  igpu) PCI=0000:00:02.0 ;;
  *)    PCI="$1" ;;
esac
L="/dev/dri/by-path/pci-${PCI}-render"
[ -e "$L" ] || { echo "no render node for $PCI" >&2; exit 2; }
N=$(basename "$(readlink -f "$L")")
ST=$(cat "/sys/bus/pci/devices/$PCI/power/runtime_status" 2>/dev/null || echo unknown)
if [ "$ST" != "active" ] && [ "$ST" != "suspended" ]; then
    echo "REFUSING: $PCI runtime_status=$ST (not healthy)" >&2; exit 3
fi
echo "$N"
