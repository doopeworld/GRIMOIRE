#!/bin/bash
# Runs oneDNN's benchdnn (performance mode, weights read cold from memory) on gpu0 only,
# inside the GRIMOIRE runtime image, holding gpu0.lock for the whole run.
# Usage: run_benchdnn_g0.sh <cases file> <tag>      (cases: KNOBS ;; PROBLEMS per line)
# Output: /mnt/storage/isos/onednn/results/<tag>.csv (+ .log)
set -u
CASES="${1:?cases file}"; TAG="${2:?tag}"
ROOT=/mnt/storage/isos/onednn
OUT=$ROOT/results; mkdir -p "$OUT"
LOCKDIR=/mnt/user/isos/grimoire-runs
exec 8>"$LOCKDIR/gpu0.lock"
flock -n 8 || { echo "gpu0.lock is held by another job -- not starting"; exit 9; }
NODE=$(bash /mnt/storage/isos/grimoire-fuse/tools/gpunode.sh gpu0) || exit 2
for c in $(docker ps --format '{{.Names}}'); do
    if docker inspect --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$c" \
         2>/dev/null | tr ' ' '\n' | grep -qx "/dev/dri/$NODE"; then
        echo "REFUSING: /dev/dri/$NODE is held by $c"; exit 3
    fi
done
CNAME=grim-onednn-$TAG
docker rm -f "$CNAME" >/dev/null 2>&1 || true
cp "$CASES" "$OUT/$TAG.cases"
# The time limit lives inside the container (TERM, then 60 s to drain) and is set far above
# the expected runtime (~70 problems x ~2 s), so it is never what ends the run.
docker run -d --name "$CNAME" --init --stop-timeout 300 \
    --device "/dev/dri/$NODE" \
    -v $ROOT:/onednn \
    -e LD_LIBRARY_PATH=/onednn/install-3.13.4/lib \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
    --entrypoint /usr/bin/timeout grimoire-b70:latest \
    --signal=TERM --kill-after=60 1800 \
    bash /onednn/bench_loop.sh /onednn/results/$TAG.cases >/dev/null
echo "launched $CNAME on $NODE"
RC=$(docker wait "$CNAME")
docker logs "$CNAME" > "$OUT/$TAG.log" 2>&1
grep '^perf,' "$OUT/$TAG.log" > "$OUT/$TAG.csv"
docker rm "$CNAME" >/dev/null 2>&1
echo "exit=$RC rows=$(wc -l < "$OUT/$TAG.csv") log=$OUT/$TAG.log"
