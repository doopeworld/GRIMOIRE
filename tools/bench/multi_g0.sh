#!/bin/bash
# multi_g0.sh -- the "multi" entrypoint with one card per rank, tested on gpu0 ALONE: both pipeline
# ranks get device 0 (GRIMOIRE_MULTI_DEVICES=0,0), Qwen3.8-27B GPTQ split 32/32.  Only renderD128 is
# mapped.  Refuses if anything holds a GPU; stops the container right after the checks.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003; R=/mnt/storage/isos/grimoire-fuse; PORT=6898; NAME=zc-multi-g0
aer() { for p in 0000:00:01.0 0000:01:00.0; do for f in aer_dev_correctable aer_dev_nonfatal aer_dev_fatal; do
  tr '\n' ' ' < /sys/bus/pci/devices/$p/$f | grep -oE "TOTAL_ERR_[A-Z]+ [0-9]+" | awk '{printf "%s ", $2}'; done; done; }
[ "$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render)" = /dev/dri/renderD128 ] || { echo "gpu0 is not renderD128 -- refusing"; exit 9; }
[ -z "$(docker ps -q)" ] || { echo "a container is running -- refusing"; docker ps --format '{{.Names}}'; exit 9; }
for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q "/dev/dri/renderD12[89]\|/dev/dri/renderD13[01]" && { echo "a GPU is held by $(cat $p/comm) -- refusing"; exit 9; }; done
before=$(aer); echo "gpu0 AER before: $before"
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --init --stop-timeout 300 --network host --ipc=host \
  --device /dev/dri/renderD128 -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v $R/docker-entrypoint.sh:/usr/local/bin/docker-entrypoint.sh:ro \
  -v /mnt/storage/Models:/models -v /mnt/storage/isos/grimoire-cache:/cache \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu -e ZE_AFFINITY_MASK=0 -e GRIMOIRE_MULTI_DEVICES=0,0 \
  -e GRIMOIRE_MULTI_GPUS=2 -e GRIMOIRE_PP_SPLIT=32 -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
  -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/cache/sycl \
  grimoire-b70:latest multi --model /models/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 --proj int4 --ctx 8192 \
  --host 0.0.0.0 --port $PORT >/dev/null
ok=0; for i in $(seq 1 200); do sleep 3
  curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -qx $NAME || break; done
echo "ready=$ok"
docker logs $NAME 2>&1 | grep -E "rank [01] -> GPU|multiprocess PP|connected|scheduler|error|fail|no device" | head -8 | cut -c1-160
if [ $ok = 1 ]; then python3 $B/ask.py $PORT 300 | tail -1; python3 $B/ctest.py $PORT | tail -1; fi
docker logs $NAME > $B/logs/$NAME.log 2>&1
docker stop --time 120 $NAME >/dev/null; docker rm $NAME >/dev/null 2>&1
echo "stopped; running containers now: $(docker ps -q | wc -l)"
after=$(aer); echo "gpu0 AER after:  $after"
