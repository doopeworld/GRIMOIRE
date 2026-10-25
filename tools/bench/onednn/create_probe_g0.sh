#!/bin/bash
# Build and run dnnl_create_probe on gpu0 only (gpu0.lock held).
set -u
ROOT=/mnt/storage/isos/onednn
REPO=/mnt/storage/isos/grimoire-fuse
OUT=$ROOT/results
docker run --rm -v $ROOT:/onednn -w /onednn grimoire-dev:latest bash -lc \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; icpx -fsycl -O2 -std=c++20 -I/onednn/install-3.13.4/include dnnl_create_probe.cpp -L/onednn/install-3.13.4/lib -ldnnl -Wl,-rpath,/onednn/install-3.13.4/lib -o dnnl_create_probe' \
  > "$OUT/create_build.log" 2>&1 || { echo "build failed"; tail -20 "$OUT/create_build.log"; exit 1; }
exec 8>/mnt/user/isos/grimoire-runs/gpu0.lock
flock -n 8 || { echo "gpu0.lock is held by another job -- not starting"; exit 9; }
NODE=$(bash $REPO/tools/gpunode.sh gpu0) || exit 2
docker rm -f grim-dnnlcreate >/dev/null 2>&1
docker run --name grim-dnnlcreate --init --stop-timeout 300 --device /dev/dri/$NODE \
  -v $ROOT:/onednn --entrypoint /usr/bin/timeout grimoire-b70:latest --signal=TERM --kill-after=60 600 \
  /onednn/dnnl_create_probe 2048 2153 2304 768 1023 512 300 64 > "$OUT/create.log" 2>&1
echo "exit=$?"
docker rm grim-dnnlcreate >/dev/null 2>&1
cat "$OUT/create.log"
