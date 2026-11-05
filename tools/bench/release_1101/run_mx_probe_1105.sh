#!/bin/bash
# GPU0, one job: tools/mx4rows_probe (branch mxfp4-smallm-1105): MXFP4 small-M, library (256 GRF) vs the
# 128-GRF port with the INT4 plan, plus dequant variants. One small shape first, then all. The container exits
# by itself. EXTRA_ENV (one K=V per line) is passed through, e.g. SWEEP=1.
set -u
R=/mnt/storage/isos/grimoire-runs
L=$R/bench-1022/logs/mx-probe-1105${TAG:+-$TAG}.out
W=/mnt/storage/isos/grimoire-mx-smallm-1105
exec >"$L" 2>&1
end() { echo "MXPROBE_END rc=$1 $(date -Iseconds)"; exit "$1"; }
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "gpu0.lock held"; end 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then echo "renderD128 busy"; end 75; fi
[ "$(basename "$(readlink -f /sys/class/drm/renderD128/device)")" = 0000:03:00.0 ] || { echo "renderD128 is not gpu0"; end 9; }
cd /mnt/storage/isos/grimoire-fuse || end 9
echo "== $(date +%T) smoke: shape 4"
GRIM_TREE=$W LIM=300 TAILN=40 bash tools/g0run.sh mxp-smoke /grimoire/bin/mx4rows_probe 4 9>&-
rc=$?
cp /tmp/grim-mxp-smoke.log "$R/bench-1022/logs/mxp-smoke-1105${TAG:+-$TAG}.log"
[ "$rc" -eq 0 ] || end 9
echo "== $(date +%T) all shapes"
GRIM_TREE=$W LIM=1200 TAILN=400 bash tools/g0run.sh mxp-all /grimoire/bin/mx4rows_probe ${SHAPE:-} 9>&-
rc=$?
cp /tmp/grim-mxp-all.log "$R/bench-1022/logs/mxp-all-1105${TAG:+-$TAG}.log"
end "$rc"
