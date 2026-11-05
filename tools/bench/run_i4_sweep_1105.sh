#!/bin/bash
# 11-05: INT4 small-M split-plan sweep on gpu0 (tools/i4sweep_probe, engine launchers), one process per KS value:
# GRIMOIRE_I4_SMALLM_KS for <= 8 rows and GRIMOIRE_I4_SMALLM16_KS for 9-16 rows set together. Tree: grimoire-mx-lib-1105
# (its bin/ holds the probe and the G31 library with the 9-16-row override). Holds logs/HOLD-i4 and gpu0.lock.
set -u
R=/mnt/storage/isos/grimoire-runs
B=$R/bench-1022
W=/mnt/storage/isos/grimoire-mx-lib-1105
exec >"$B/logs/i4-sweep-1105.out" 2>&1
touch "$B/logs/HOLD-i4"
trap 'rm -f "$B/logs/HOLD-i4"' EXIT
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "gpu0.lock held"; exit 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then echo "renderD128 busy"; exit 75; fi
cd /mnt/storage/isos/grimoire-fuse || exit 9
for ks in 0 1 2 3 4 6 8 12 16 20 24 32; do
  ex=""
  [ "$ks" != 0 ] && ex="GRIMOIRE_I4_SMALLM_KS=$ks
GRIMOIRE_I4_SMALLM16_KS=$ks"
  echo "== KS=$ks $(date +%T)"
  EXTRA_ENV="$ex" GRIM_TREE=$W LIM=300 TAILN=200 bash tools/g0run.sh "i4s-ks$ks" /grimoire/bin/i4sweep_probe 9>&- > /dev/null
  grep -E "^[a-z]|engine|check" "/tmp/grim-i4s-ks$ks.log"
done
echo "I4_SWEEP_END $(date +%T)"
