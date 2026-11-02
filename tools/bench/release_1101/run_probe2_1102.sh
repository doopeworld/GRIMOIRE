#!/bin/bash
# GPU0, one job: wide INT4 verify probe with its own kernel names, so "256 vs 128 registers" is a real comparison.
# One small shape first, then all shapes. Stops at the first failure. The container exits by itself.
set -u
R=/mnt/storage/isos/grimoire-runs
L=$R/bench-1022/logs/probe2-1102.out
W=/mnt/storage/isos/grimoire-verify-grf-1102
exec >"$L" 2>&1
end() { echo "PROBE2_END rc=$1 $(date -Iseconds)"; exit "$1"; }
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "gpu0.lock held"; end 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then echo "renderD128 busy"; end 75; fi
cd /mnt/storage/isos/grimoire-fuse || end 9
echo "== $(date +%T) smoke: shape 0"
GRIM_TREE=$W LIM=300 TAILN=20 bash tools/g0run.sh vgrf2-smoke /grimoire/bin/i4wide_grf_probe2 0 9>&-
rc=$?
cp /tmp/grim-vgrf2-smoke.log "$R/bench-1022/logs/vgrf2-smoke-1102.log"
[ "$rc" -eq 0 ] || end 9
echo "== $(date +%T) all shapes"
GRIM_TREE=$W LIM=900 TAILN=70 bash tools/g0run.sh vgrf2 /grimoire/bin/i4wide_grf_probe2 9>&-
rc=$?
cp /tmp/grim-vgrf2.log "$R/bench-1022/logs/vgrf2-all-1102.log"
end "$rc"
