#!/bin/bash
# 11-05: after the B580 comparison ends: switch the five profiles to the published v1.10.4 (switch_v1104.sh), smoke them
# (smoke_v1104.sh), then the after-publish experiments (run_after_publish_exp_1105.sh). One GPU job at a time.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
exec >"$B/logs/switch-exp-1105.out" 2>&1
until grep -qE "^B580_(DONE|STOP)" "$B/logs/after-gate-1104b.out" 2>/dev/null; do sleep 30; done
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
ID=$(docker image inspect -f '{{.Id}}' grimoire-b70:v1.10.4)
echo "== $(date +%T) switch to $ID"
bash "$B/switch_v1104.sh" "$ID"; rc=$?
echo "SWITCH_RC=$rc"
if [ "$rc" = 0 ]; then
  echo "== $(date +%T) smoke"
  bash "$B/smoke_v1104.sh"; echo "SMOKE_RC=$?"
fi
echo "== $(date +%T) experiments"
bash "$B/run_after_publish_exp_1105.sh"
echo "SWITCH_EXP_DONE $(date +%T)"
