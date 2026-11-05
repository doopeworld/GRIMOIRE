#!/usr/bin/env bash
# v1.10.5 (11-05): the five GRIMOIRE profiles (three single-card, two dual; all on v1.10.4 since 13:13) move to
# grimoire-b70:v1.10.5. Templates and container settings are backed up first; containers are recreated from their
# Unraid templates (created, not started). Nothing of GRIMOIRE may run (CPU-only builds may).
set -euo pipefail
T=/boot/config/plugins/dockerMan/templates-user
B=/mnt/storage/isos/grimoire-runs/bench-1003
BK=$B/templates-backup/2026-11-05-v1105
EXPECTED=$1
exec 8>/mnt/user/isos/grimoire-runs/gpu0.lock 9>/mnt/user/isos/grimoire-runs/gpu1.lock
flock -n 8 || exit 9
flock -n 9 || exit 9
test -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-" | grep -iv "^comfyui")" || exit 9
test "$(docker image inspect -f "{{.Id}}" grimoire-b70:v1.10.5)" = "$EXPECTED"
test ! -e "$BK"
ALL="GRIMOIRE-QWEN3.8 GRIMOIRE-ORNITH GRIMOIRE-QWEN3.8-GGUF GRIMOIRE-DUAL-QWEN3.8 GRIMOIRE-DUAL-FLASH-NEXT"
for name in $ALL; do
  test "$(docker inspect -f "{{.State.Running}}" "$name")" = false
  grep -q "<Repository>grimoire-b70:v1.10.4</Repository>" "$T/my-$name.xml"
done
mkdir -p "$BK"
for name in $ALL; do
  cp -p "$T/my-$name.xml" "$BK/"
  docker inspect "$name" > "$BK/$name.inspect.json"
done
for name in $ALL; do
  sed -i "s#<Repository>grimoire-b70:v1.10.4</Repository>#<Repository>grimoire-b70:v1.10.5</Repository>#" "$T/my-$name.xml"
  php "$B/create_from_template.php" "$name"
done
docker tag grimoire-b70:v1.10.5 grimoire-b70:latest
docker inspect -f "{{.Name}} {{.Config.Image}} {{.State.Status}}" $ALL
echo SWITCH_PASS
