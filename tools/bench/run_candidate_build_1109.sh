#!/bin/bash
set -euo pipefail
REV=${1:?source revision}
B=/mnt/storage/isos/grimoire-runs/bench-1022
X=/mnt/storage/isos/grimoire-release-1109
exec >"$B/logs/candidate-build1109.out" 2>&1
exec 6>"$B/logs/candidate-build1109.lock"
flock -n 6 || exit 9
grep -qx IMX_PERFORMANCE_PASS "$B/logs/imx-perf1109.out"
test ! -e "$X/bin/grimoire-server"
echo "BUILDER_SOURCE $REV"
docker image inspect grimoire-dev:latest --format '{{.Id}}'
# No devices and no networking; both architecture images are compiled on CPUs.
docker run --rm --name grim-candidate1109-builder --network none --cpuset-cpus 0-15 \
 -e "GRIM_SOURCE_COMMIT=$REV" -v "$X:/work" \
 -v /mnt/storage/isos/grimoire-release-1106:/release:ro -w /work \
 grimoire-dev:latest bash /work/tools/bench/build_candidate_1109.sh
