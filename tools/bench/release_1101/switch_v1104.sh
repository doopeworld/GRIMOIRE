#!/usr/bin/env bash
# v1.10.4 (11-05): the three single-card profiles (on v1.10.3) and the two dual profiles that have containers (still on
# v1.10.0) move to grimoire-b70:v1.10.4. Templates and container settings are backed up first; containers are
# recreated from their Unraid templates (created, not started). Nothing may run.
set -euo pipefail
T=/boot/config/plugins/dockerMan/templates-user
B=/mnt/storage/isos/grimoire-runs/bench-1003
BK=$B/templates-backup/2026-11-05-v1104
EXPECTED=$1
exec 8>/mnt/user/isos/grimoire-runs/gpu0.lock 9>/mnt/user/isos/grimoire-runs/gpu1.lock
flock -n 8 || exit 9
flock -n 9 || exit 9
test -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" || exit 9   # CPU-only builds (no devices) may run
test "$(docker image inspect -f "{{.Id}}" grimoire-b70:v1.10.4)" = "$EXPECTED"
test ! -e "$BK"
SINGLE="GRIMOIRE-QWEN3.8 GRIMOIRE-ORNITH GRIMOIRE-QWEN3.8-GGUF"
DUAL="GRIMOIRE-DUAL-QWEN3.8 GRIMOIRE-DUAL-FLASH-NEXT"
for name in $SINGLE; do
  test "$(docker inspect -f "{{.State.Running}}" "$name")" = false
  grep -q "<Repository>grimoire-b70:v1.10.3</Repository>" "$T/my-$name.xml"
done
for name in $DUAL; do
  test "$(docker inspect -f "{{.State.Running}}" "$name")" = false
  grep -q "<Repository>grimoire-b70:v1.10.0</Repository>" "$T/my-$name.xml"
done
mkdir -p "$BK"
for name in $SINGLE $DUAL; do
  cp -p "$T/my-$name.xml" "$BK/"
  docker inspect "$name" > "$BK/$name.inspect.json"
done
for name in $SINGLE; do
  sed -i "s#<Repository>grimoire-b70:v1.10.3</Repository>#<Repository>grimoire-b70:v1.10.4</Repository>#" "$T/my-$name.xml"
  php "$B/create_from_template.php" "$name"
done
for name in $DUAL; do
  sed -i "s#<Repository>grimoire-b70:v1.10.0</Repository>#<Repository>grimoire-b70:v1.10.4</Repository>#" "$T/my-$name.xml"
  php "$B/create_from_template.php" "$name"
done
docker tag grimoire-b70:v1.10.4 grimoire-b70:latest
docker inspect -f "{{.Name}} {{.Config.Image}} {{.State.Status}}" $SINGLE $DUAL
echo SWITCH_PASS
