#!/usr/bin/env bash
# v1.10.5 profile smoke (11-05): each switched profile starts from its template, answers ctest.py's questions
# (Paris) and stops. Single-card profiles: AER unchanged on gpu0's path. Dual profiles: gpu1's link logs
# corrected errors by itself, so only uncorrectable errors fail them (ctest2.sh exits 8 on those).
set -euo pipefail
B=/mnt/storage/isos/grimoire-runs/bench-1003
L=/mnt/storage/isos/grimoire-runs/bench-1022/logs
for entry in GRIMOIRE-QWEN3.8:6890 GRIMOIRE-ORNITH:6889 GRIMOIRE-QWEN3.8-GGUF:6894 \
             GRIMOIRE-DUAL-QWEN3.8:6897:dual GRIMOIRE-DUAL-FLASH-NEXT:6898:dual; do
  IFS=: read -r name port dual <<< "$entry"
  bash "$B/ctest2.sh" "$name" "$port" $dual > "$L/r1105-profile-$name.out" 2>&1 || true
  cat "$L/r1105-profile-$name.out"
  grep -q "answer: .Paris." "$L/r1105-profile-$name.out"
  if grep -q "UNCORRECTABLE" "$L/r1105-profile-$name.out"; then echo "SMOKE_STOP $name uncorrectable"; exit 1; fi
  [ -n "$dual" ] || grep -q "AER unchanged" "$L/r1105-profile-$name.out"
  test "$(docker inspect -f "{{.State.Running}}" "$name")" = false
  cp "$B/logs/ctest2-$name.log" "$L/r1105-profile-$name.server.log"
done
echo PROFILE_SMOKE_PASS
