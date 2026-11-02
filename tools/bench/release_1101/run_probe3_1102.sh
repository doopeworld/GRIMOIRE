#!/bin/bash
# GPU0, one job: local kernels on 256 registers with their own names. "local_same" should equal "lib" (method check);
# "local_narrow" is the only combination still untested: the narrow plan where production actually runs.
set -u
R=/mnt/storage/isos/grimoire-runs
L=$R/bench-1022/logs/probe3-1102.out
W=/mnt/storage/isos/grimoire-verify-grf-1102
exec >"$L" 2>&1
end() { echo "PROBE3_END rc=$1 $(date -Iseconds)"; exit "$1"; }
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "gpu0.lock held"; end 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then echo "renderD128 busy"; end 75; fi
cd /mnt/storage/isos/grimoire-fuse || end 9
GRIM_TREE=$W LIM=300 TAILN=20 bash tools/g0run.sh vgrf3-smoke /grimoire/bin/i4wide_grf_probe3 0 9>&-
rc=$?
cp /tmp/grim-vgrf3-smoke.log "$R/bench-1022/logs/vgrf3-smoke-1102.log"
[ "$rc" -eq 0 ] || end 9
GRIM_TREE=$W LIM=900 TAILN=70 bash tools/g0run.sh vgrf3 /grimoire/bin/i4wide_grf_probe3 9>&-
rc=$?
cp /tmp/grim-vgrf3.log "$R/bench-1022/logs/vgrf3-all-1102.log"
end "$rc"
