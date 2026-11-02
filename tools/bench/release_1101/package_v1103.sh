#!/usr/bin/env bash
# Package the validated candidate. Refuses unless the validation gate passed on this exact image.
set -euo pipefail
RUNS=/mnt/storage/isos/grimoire-runs
L=$RUNS/bench-1022/logs
EXPECTED=$1
test "$(docker image inspect -f "{{.Id}}" grimoire-b70:v1.10.3-rc1)" = "$EXPECTED"
grep -qx IMAGE_LOG_CHECK_PASS "$L/validate-v1103.out"
test ! -e "$RUNS/grimoire-b70-v1.10.3.tar.gz"
if docker image inspect grimoire-b70:v1.10.3 >/dev/null 2>&1; then
  test "$(docker image inspect -f "{{.Id}}" grimoire-b70:v1.10.3)" = "$EXPECTED"
fi
docker tag grimoire-b70:v1.10.3-rc1 grimoire-b70:v1.10.3
tmp=$(mktemp "$RUNS/.grimoire-v1.10.3.XXXXXX.gz")
docker save grimoire-b70:v1.10.3 | taskset -c 6-13 gzip -6 > "$tmp"
gzip -t "$tmp"
mv "$tmp" "$RUNS/grimoire-b70-v1.10.3.tar.gz"
cd "$RUNS"
sha256sum grimoire-b70-v1.10.3.tar.gz | tee grimoire-b70-v1.10.3.tar.gz.sha256
stat -c "asset bytes: %s" grimoire-b70-v1.10.3.tar.gz
echo PACKAGE_PASS
