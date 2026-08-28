#!/bin/bash
# =====================================================================
#  preflight.sh  --  run this on the UNRAID HOST (not in a container)
#  before building anything. Every check below has bitten someone on
#  the Unraid forums with a Battlemage card.
#
#    ./preflight.sh
# =====================================================================

RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; RST=$'\e[0m'
ok()   { echo "${GRN}  OK${RST}   $*"; }
warn() { echo "${YEL} WARN${RST}   $*"; }
bad()  { echo "${RED} FAIL${RST}   $*"; FAILED=1; }
FAILED=0

echo "=== b70 preflight ==="
echo

# ---------------------------------------------------------------------
echo "[1] Kernel"
KVER=$(uname -r)
KMAJ=$(echo "$KVER" | cut -d. -f1)
KMIN=$(echo "$KVER" | cut -d. -f2)
echo "       running $KVER"
# Battlemage is xe-only; i915 never supported it. Discrete BMG landed
# properly around 6.13, and Unraid 7.2.4 shipped 6.12.54 which is NOT
# enough. Unraid 7.3.1+ (kernel 6.18) is what people report working.
if [ "$KMAJ" -gt 6 ] || { [ "$KMAJ" -eq 6 ] && [ "$KMIN" -ge 13 ]; }; then
    ok "kernel is new enough for discrete Battlemage"
else
    bad "kernel $KVER predates usable discrete Battlemage support.
         Unraid 7.2.x ships 6.12.x and will NOT drive a B70.
         Upgrade to Unraid 7.3.1 or later before going further."
fi

# ---------------------------------------------------------------------
echo
echo "[2] GPU driver"
if lsmod 2>/dev/null | grep -q '^xe '; then
    ok "xe module loaded"
elif lsmod 2>/dev/null | grep -q '^i915 '; then
    bad "i915 is loaded, xe is not. i915 has NEVER supported Battlemage.
         Check that nothing is forcing i915 in your syslinux config."
else
    bad "neither xe nor i915 loaded"
fi

# ---------------------------------------------------------------------
echo
echo "[3] Battlemage devices"
GPUS=$(lspci -nn 2>/dev/null | grep -iE '8086:e2[0-9a-f]{2}' || true)
if [ -z "$GPUS" ]; then
    warn "no Intel 8086:e2xx device found by lspci; listing all VGA/Display:"
    lspci -nn 2>/dev/null | grep -iE 'vga|display|3d' | sed 's/^/       /'
else
    echo "$GPUS" | sed 's/^/       /'
    N=$(echo "$GPUS" | wc -l)
    ok "$N Battlemage-class device(s) found"
fi

# ---------------------------------------------------------------------
echo
echo "[4] PCIe link width and speed"
echo "       This is the single most important number for dual-GPU."
for d in /sys/bus/pci/devices/*/; do
    VEN=$(cat "$d/vendor" 2>/dev/null)
    DEV=$(cat "$d/device" 2>/dev/null)
    [ "$VEN" = "0x8086" ] || continue
    case "$DEV" in 0xe2*) ;; *) continue ;; esac
    SLOT=$(basename "$d")
    CW=$(cat "$d/current_link_width" 2>/dev/null || echo "?")
    CS=$(cat "$d/current_link_speed" 2>/dev/null || echo "?")
    MW=$(cat "$d/max_link_width" 2>/dev/null || echo "?")
    MS=$(cat "$d/max_link_speed" 2>/dev/null || echo "?")
    echo "       $SLOT  negotiated: x$CW @ $CS   (card capable of x$MW @ $MS)"
    if [ "$CW" != "?" ] && [ "$CW" -lt 8 ] 2>/dev/null; then
        warn "  x$CW is narrow. Tensor parallelism all-reduces every layer;
              on a narrow link that overhead can eat the second GPU's gain.
              Measure before assuming TP is the right split."
    fi
done

# ---------------------------------------------------------------------
echo
echo "[5] Resizable BAR"
echo "       Without ReBAR the host can only see a 256 MB window into"
echo "       32 GB of VRAM. Arc performance collapses without it."
for d in /sys/bus/pci/devices/*/; do
    VEN=$(cat "$d/vendor" 2>/dev/null)
    DEV=$(cat "$d/device" 2>/dev/null)
    [ "$VEN" = "0x8086" ] || continue
    case "$DEV" in 0xe2*) ;; *) continue ;; esac
    SLOT=$(basename "$d")
    # BAR2 is the VRAM aperture on Arc. resource line 2, fields are
    # start end flags in hex.
    LINE=$(sed -n '3p' "$d/resource" 2>/dev/null)
    if [ -n "$LINE" ]; then
        S=$(echo "$LINE" | awk '{print $1}')
        E=$(echo "$LINE" | awk '{print $2}')
        SZ=$(( (E - S + 1) / 1048576 ))
        if [ "$SZ" -ge 8192 ]; then
            ok "$SLOT BAR2 = ${SZ} MiB, ReBAR is on"
        else
            bad "$SLOT BAR2 = ${SZ} MiB. Enable 'Resizable BAR' AND
             'Above 4G Decoding' in BIOS, then reboot."
        fi
    fi
done

# ---------------------------------------------------------------------
echo
echo "[6] Render nodes (what the container needs)"
if ls /dev/dri/renderD* >/dev/null 2>&1; then
    ls -l /dev/dri/ | sed 's/^/       /'
    RGID=$(stat -c '%g' /dev/dri/renderD128 2>/dev/null)
    NRENDER=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
    ok "$NRENDER render node(s); group id $RGID"
    echo
    echo "       Pass this to docker:  --group-add $RGID  --device /dev/dri"
else
    bad "no /dev/dri/renderD* nodes. The driver is not binding the cards."
fi

# ---------------------------------------------------------------------
echo
echo "[7] Power and thermals"
echo "       Two B70s draw up to 460 W (230 W each at reference)."
echo "       Your Core Ultra 9 285H is a 45 W / 115 W BGA mobile part."
echo "       Confirm the PSU and the board's slot power delivery before"
echo "       running both cards at full tilt. Check the 12V-2x6 cabling."

# ---------------------------------------------------------------------
echo
echo "[8] IOMMU"
if dmesg 2>/dev/null | grep -qi 'DMAR: IOMMU enabled\|Intel-IOMMU'; then
    warn "IOMMU is on. Fine for containers, but if you ever passed these
         cards to a VM, make sure they are NOT bound to vfio-pci now:
           lspci -nnk | grep -A3 -i e2"
else
    ok "IOMMU not forcing anything"
fi

echo
if [ "$FAILED" = "1" ]; then
    echo "${RED}Preflight FAILED.${RST} Fix the items above; the container will not"
    echo "work until they pass."
    exit 1
fi
echo "${GRN}Preflight passed.${RST}"
echo "Next:  docker compose -f docker/docker-compose.yml build"
