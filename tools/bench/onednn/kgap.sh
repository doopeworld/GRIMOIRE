#!/bin/bash
# 10-28: build tools/kgap_probe.cpp (grimoire-dev, oneAPI only) and run it on gpu0 alone
# (tools/g0run.sh: link + xe checks, LIM 120 s; the probe itself runs well under a second).
R=/mnt/storage/isos/grimoire-fuse; S=/mnt/user/isos/grimoire-runs/bench-1022
cd $R || exit 2
docker run --rm -v $R:/grimoire -w /grimoire grimoire-dev:latest bash -lc \
  "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; icpx -fsycl -O3 -std=c++20 tools/kgap_probe.cpp -o bin/kgap_probe" > $S/logs/build-kgap.log 2>&1
[ -x bin/kgap_probe ] || { echo KGAP_BUILD_FAILED; tail -20 $S/logs/build-kgap.log; exit 1; }
exec 8>/mnt/user/isos/grimoire-runs/gpu0.lock
flock -n 8 || { echo "gpu0.lock is held -- not starting"; exit 9; }
LIM=120 TAILN=14 bash tools/g0run.sh kgap /grimoire/bin/kgap_probe
echo KGAP_SH_DONE
