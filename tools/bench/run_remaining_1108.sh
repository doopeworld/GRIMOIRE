#!/bin/bash
# Serialize the outstanding experiments after candidate image / PP verification.
set -euo pipefail
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B"
exec >>logs/remaining-1108.out 2>&1
exec 5>logs/remaining-1108.lock
flock -n 5 || exit 9
echo "REMAINING_WAIT $(date -Iseconds)"
until grep -q '^RESUME_CHECKS_PASS ' logs/resume-1108.out; do
  if flock -n logs/resume-1108.lock true; then
    echo "REMAINING_STOP candidate checks stopped without a pass $(date -Iseconds)"
    exit 9
  fi
  sleep 10
done
grep -qx CALIBRATION_FETCH_PASS logs/fetch-calibration-1108.out
if ! grep -qx IMX_BUILD_PASS logs/imx-build-1106.out 2>/dev/null; then
  test ! -e logs/imx-build-1106.out
  echo "IMX_BUILD_BEGIN $(date -Iseconds)"
  bash ./run_imx_build_1106.sh ff77f2b 6-13
fi
echo "IMX_BUILD_COMPLETE $(date -Iseconds)"
bash ./run_accuracy_1108.sh
echo "REMAINING_DONE $(date -Iseconds)"
