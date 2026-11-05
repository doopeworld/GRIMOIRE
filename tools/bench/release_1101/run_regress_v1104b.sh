#!/bin/bash
# Full 44-case regression of the combined v1.10.4 candidate (11-05, mxfp4-smallm-1105), gpu0 only, launched the same way as 11-02.
# Holds gpu0.lock without waiting; refuses a busy card; nothing is queued behind it.
set -u
T=/mnt/storage/isos/grimoire-release-1105
R=/mnt/storage/isos/grimoire-runs
L=$R/bench-1022/logs/regress-v1104b.out
exec >>"$L" 2>&1
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "REGRESSION_REFUSED gpu0.lock held $(date -Iseconds)"; echo "REGRESSION_EXIT=75"; exit 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then
  echo "REGRESSION_REFUSED renderD128 busy $(date -Iseconds)"; echo "REGRESSION_EXIT=75"; exit 75
fi
echo "source=$(git -C "$T" rev-parse --short HEAD) tree=$T start=$(date -Iseconds)"
export GRIM_TREE="$T"
bash /mnt/storage/isos/grimoire-fuse/tools/regress_all_g0.sh 9>&-
rc=$?
echo "REGRESSION_EXIT=$rc end=$(date -Iseconds)"
exit "$rc"
