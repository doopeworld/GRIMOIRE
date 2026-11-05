#!/bin/bash
# 11-05: plan sweep of the library's MXFP4 small-M kernel past 8 rows (GRIMOIRE_MX_SMALLM16_KS), gpu0, the probe with
# ONLY_LIB16=1 (library M = 12 / 14 / 16 and the M = 16 check only). Holds logs/HOLD-mx16 (the gate waits) and gpu0.lock.
set -u
R=/mnt/storage/isos/grimoire-runs
B=$R/bench-1022
W=/mnt/storage/isos/grimoire-mx-lib-1105
exec >"$B/logs/mx16-sweep-1105.out" 2>&1
touch "$B/logs/HOLD-mx16"
trap 'rm -f "$B/logs/HOLD-mx16"' EXIT
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "gpu0.lock held"; exit 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then echo "renderD128 busy"; exit 75; fi
cd /mnt/storage/isos/grimoire-fuse || exit 9
for ks in 0 1 2 4 8 12 16 20 24 32; do
  ex="ONLY_LIB16=1"
  [ "$ks" != 0 ] && ex="$ex
GRIMOIRE_MX_SMALLM16_KS=$ks"
  echo "== KS=$ks $(date +%T)"
  EXTRA_ENV="$ex" GRIM_TREE=$W LIM=300 TAILN=200 bash tools/g0run.sh "mx16-ks$ks" /grimoire/bin/mx4rows_probe 9>&- > /dev/null
  grep -E "^[a-z]|library|check" "/tmp/grim-mx16-ks$ks.log"
done
echo "MX16_SWEEP_END $(date +%T)"
