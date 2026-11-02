#!/usr/bin/env bash
set -euo pipefail
B=/mnt/storage/isos/grimoire-runs/bench-1003
L=/mnt/storage/isos/grimoire-runs/bench-1022/logs
BK=$B/templates-backup/2026-11-02-v1103
for entry in GRIMOIRE-QWEN3.8:6890 GRIMOIRE-ORNITH:6889 GRIMOIRE-QWEN3.8-GGUF:6894; do
  name=${entry%:*}
  port=${entry##*:}
  if [ -f "$B/logs/ctest2-$name.log" ]; then
    cp -p "$B/logs/ctest2-$name.log" "$BK/ctest2-$name.before.log"
  fi
  bash "$B/ctest2.sh" "$name" "$port" > "$L/r1102-profile-$name.out" 2>&1
  cat "$L/r1102-profile-$name.out"
  grep -q "answer: .Paris." "$L/r1102-profile-$name.out"
  grep -q "AER unchanged" "$L/r1102-profile-$name.out"
  test "$(docker inspect -f "{{.State.Running}}" "$name")" = false
  cp "$B/logs/ctest2-$name.log" "$L/r1102-profile-$name.server.log"
done
echo PROFILE_SMOKE_PASS
