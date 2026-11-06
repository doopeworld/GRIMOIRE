#!/bin/bash
# Full 44-case regression of the v1.10.5-rc2 candidate (grimoire-release-1106), gpu0 only, launched as v1.10.4's
# (run_regress_v1104b.sh, results regress-1105-1154). Flash-Next's hot-expert case reads GRIMOIRE_EXPERT_HITS
# (flash-next.hits): v1.10.4's regression ran on the snapshot flash-next.hits.det-1105, restored here and the current
# file put back at the end. The Flash-Next cases now run the heat engine (on by default on one card) -- their texts may
# reword near ties against v1.10.4's. Holds gpu0.lock without waiting; refuses a busy card; nothing is queued behind it.
set -u
T=/mnt/storage/isos/grimoire-release-1106
R=/mnt/storage/isos/grimoire-runs
H=/mnt/storage/Models/grimoire-ple/flash-next.hits
L=$R/bench-1022/logs/regress-v1106.out
exec >>"$L" 2>&1
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "REGRESSION_REFUSED gpu0.lock held $(date -Iseconds)"; echo "REGRESSION_EXIT=75"; exit 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then
  echo "REGRESSION_REFUSED renderD128 busy $(date -Iseconds)"; echo "REGRESSION_EXIT=75"; exit 75
fi
[ -e "$H.pre-1106" ] || cp -p "$H" "$H.pre-1106"
cp "$H.det-1105" "$H"
trap 'cp -p "$H.pre-1106" "$H"; echo "hits file put back $(sha256sum "$H" | cut -c1-16)"' EXIT
echo "hits snapshot $(sha256sum "$H" | cut -c1-16) (det-1105)"
echo "source=$(git -C "$T" rev-parse --short HEAD) tree=$T start=$(date -Iseconds)"
export GRIM_TREE="$T"
bash /mnt/storage/isos/grimoire-fuse/tools/regress_all_g0.sh 9>&-
rc=$?
echo "REGRESSION_EXIT=$rc end=$(date -Iseconds)"
exit "$rc"
