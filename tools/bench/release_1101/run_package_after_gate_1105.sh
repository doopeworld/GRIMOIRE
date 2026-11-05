#!/bin/bash
# 11-05: package v1.10.4 as soon as the combined gate passes (V1104B_ALL_DONE after its continuation), holding
# logs/HOLD-pkg so the B580 runs (CPU-heavy llama.cpp) wait for gzip. The Muse check (GPU0, text only) may run beside it.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
exec >"$B/logs/package-v1104.out" 2>&1
touch "$B/logs/HOLD-pkg"
trap 'rm -f "$B/logs/HOLD-pkg"' EXIT
gate_over() { awk '/^V1104B_ALL_CONTINUE/{c=1; e=0} c && /^V1104B_ALL_(DONE|STOP)/{e=1} END{exit !(c && e)}' "$B/logs/v1104b-all.out"; }
until gate_over; do sleep 30; done
awk '/^V1104B_ALL_CONTINUE/{c=1} c && /^V1104B_ALL_DONE/{d=1} END{exit !d}' "$B/logs/v1104b-all.out" ||
  { echo "PACKAGE_SKIPPED gate did not pass"; exit 1; }
ID=$(docker image inspect -f '{{.Id}}' grimoire-b70:v1.10.4-rc1)
echo "image $ID $(date +%T)"
bash "$B/package_v1104.sh" "$ID"
echo "PACKAGE_END rc=$? $(date +%T)"
