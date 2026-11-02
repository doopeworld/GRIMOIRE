#!/usr/bin/env bash
# Update only the validated single-card profiles; preserve stopped-container settings.
set -euo pipefail
T=/boot/config/plugins/dockerMan/templates-user
B=/mnt/storage/isos/grimoire-runs/bench-1003
BK=$B/templates-backup/2026-11-02-v1103
EXPECTED=$1
exec 8>/mnt/user/isos/grimoire-runs/gpu0.lock
flock -n 8 || exit 9
test -z "$(docker ps -q)" || exit 9
test "$(docker image inspect -f "{{.Id}}" grimoire-b70:v1.10.3)" = "$EXPECTED"
test ! -e "$BK"
NAMES="GRIMOIRE-QWEN3.8 GRIMOIRE-ORNITH GRIMOIRE-QWEN3.8-GGUF"
for name in $NAMES; do
  test "$(docker inspect -f "{{.State.Running}}" "$name")" = false
  grep -q "<Repository>grimoire-b70:v1.10.2</Repository>" "$T/my-$name.xml"
done
mkdir -p "$BK"
for name in $NAMES; do
  cp -p "$T/my-$name.xml" "$BK/"
  docker inspect "$name" > "$BK/$name.inspect.json"
done
for name in $NAMES; do
  sed -i "s#<Repository>grimoire-b70:v1.10.2</Repository>#<Repository>grimoire-b70:v1.10.3</Repository>#" "$T/my-$name.xml"
  php "$B/create_from_template.php" "$name"
done
docker tag grimoire-b70:v1.10.3 grimoire-b70:latest
docker inspect -f "{{.Name}} {{.Config.Image}} {{.State.Status}}" $NAMES
echo SWITCH_PASS
