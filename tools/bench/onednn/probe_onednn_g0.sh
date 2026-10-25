#!/bin/bash
# gemm_fast_probe through GRIMOIRE's own launch_gemm_fast with GRIMOIRE_ONEDNN=1 (gpu0 only):
# the host-reference check must PASS and the time must match benchdnn's.
set -u
ROOT=/mnt/storage/isos/onednn
REPO=/mnt/storage/isos/grimoire-fuse
OUT=$ROOT/results
exec 8>/mnt/user/isos/grimoire-runs/gpu0.lock
flock -n 8 || { echo "gpu0.lock is held by another job -- not starting"; exit 9; }
NODE=$(bash $REPO/tools/gpunode.sh gpu0) || exit 2
SH="512 34816 5120  512 5120 17408  2048 34816 5120  4096 17408 5120  4096 5120 8704  64 34816 5120"
for F in fp8 bf16; do
  docker rm -f grim-gemmprobe >/dev/null 2>&1
  docker run --name grim-gemmprobe --init --stop-timeout 300 --device /dev/dri/$NODE \
    -v $REPO/bin:/grimoire/bin:ro -v $ROOT:/onednn --tmpfs /opt/grimoire/lib \
    -e GRIMOIRE_ONEDNN=1 -e GRIMOIRE_ONEDNN_VERBOSE=1 \
    --entrypoint /usr/bin/timeout grimoire-b70:latest --signal=TERM --kill-after=60 900 \
    /onednn/gemm_fast_probe $F $SH > "$OUT/grim_dnnl_$F.log" 2>&1
  echo "probe $F exit=$?"
  docker rm grim-gemmprobe >/dev/null 2>&1
done
