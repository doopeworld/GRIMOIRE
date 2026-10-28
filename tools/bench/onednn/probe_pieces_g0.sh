#!/bin/bash
# gemm_fast_probe through launch_gemm_fast on gpu0 only, three ways at the same row counts:
#   pieces = GRIMOIRE_ONEDNN=1 (row pieces, the new default way), exact = one primitive per M,
#   off = GRIMOIRE's dequantize + gemm_bf16_vnni.  Host check must PASS everywhere; the verbose
#   log must show primitives for 1024 / 512 / 256 / 64 rows only.
set -u
ROOT=/mnt/storage/isos/onednn
REPO=/mnt/storage/isos/grimoire-fuse
OUT=$ROOT/results
docker run --rm -v $REPO:/grimoire:ro -v $ROOT:/onednn -w /onednn grimoire-dev:latest bash -lc \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; icpx -fsycl -O2 -std=c++20 -I/grimoire/src -I/grimoire/include gemm_fast_probe.cpp -L/grimoire/bin -lgrimoire_gemm -Wl,-rpath,/grimoire/bin -o gemm_fast_probe' \
  > "$OUT/probe_build3.log" 2>&1 || { echo "probe build failed"; tail -20 "$OUT/probe_build3.log"; exit 1; }
exec 8>/mnt/user/isos/grimoire-runs/gpu0.lock
flock -n 8 || { echo "gpu0.lock is held by another job -- not starting"; exit 9; }
NODE=$(bash $REPO/tools/gpunode.sh gpu0) || exit 2
SH=""
for m in 64 300 512 1000 2153 4096; do SH="$SH $m 34816 5120"; done
SH="$SH 1000 5120 17408 2153 5120 17408"
for V in pieces exact off; do
  case $V in
    pieces) ENVS=(-e GRIMOIRE_ONEDNN=1 -e GRIMOIRE_ONEDNN_VERBOSE=1) ;;
    exact)  ENVS=(-e GRIMOIRE_ONEDNN=1 -e GRIMOIRE_ONEDNN_EXACT_M=1) ;;
    off)    ENVS=() ;;
  esac
  for F in fp8 bf16; do
    docker rm -f grim-gemmprobe >/dev/null 2>&1
    docker run --name grim-gemmprobe --init --stop-timeout 300 --device /dev/dri/$NODE \
      -v $REPO/bin:/grimoire/bin:ro -v $ROOT:/onednn --tmpfs /opt/grimoire/lib "${ENVS[@]}" \
      --entrypoint /usr/bin/timeout grimoire-b70:latest --signal=TERM --kill-after=60 900 \
      /onednn/gemm_fast_probe $F $SH > "$OUT/grim_${V}_$F.log" 2>&1
    echo "probe $V $F exit=$?"
    docker rm grim-gemmprobe >/dev/null 2>&1
  done
done
