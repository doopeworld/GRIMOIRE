#!/bin/bash
# ab_ornith_pp.sh -- gpu0 only.  Ornith pp512 read 6,793 / 9,137 tok/s (1 / 8 users) on the v1.8.3
# image vs 7,118 / 9,635 in the v1.8 validation.  The MXFP4 code did not change in v1.8.3, so:
# same session, same benchmark -- (a) the v1.8 image, (b) v1.8.3 with interleaved admission off,
# (c) v1.8.3 default again (clock drift check).  One server at a time, stopped right after its run.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003; PORT=6901
BOOK="--book-url http://127.0.0.1:8999/1661-0.txt"
O=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
BASE="ZE_AFFINITY_MASK=0
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
SYCL_CACHE_PERSISTENT=1
SYCL_CACHE_DIR=/cache/sycl
GRIMOIRE_SEQ_SLOTS=8"
busy() {
  local c h
  c=$(for id in $(docker ps -q); do
        docker inspect --format '{{.Name}} {{.HostConfig.Privileged}} {{range .HostConfig.Devices}}{{.PathOnHost}},{{end}}' $id
      done | awk '$2=="true" || $3 ~ /dri/ {print $1}' | tr '\n' ' ')
  h=$(for f in /proc/[0-9]*/fd/*; do l=$(readlink "$f" 2>/dev/null)
        case "$l" in /dev/dri/renderD*) p=${f#/proc/}; p=${p%%/*}; echo "$p:$(cat /proc/$p/comm 2>/dev/null):$l";; esac
      done | sort -u | tr '\n' ' ')
  if [ -n "$c$h" ]; then echo "GPU BUSY: containers=[$c] render holders=[$h]"; return 0; fi
  return 1
}
cur=""
cleanup() {
  [ -z "$cur" ] && return
  docker ps --format '{{.Names}}' | grep -qx "$cur" || return
  if curl -s -m 5 http://localhost:$PORT/health 2>/dev/null | grep -q ok; then
    BENCH_OUT=$B bash $B/srv2.sh down $cur $PORT
  else
    echo "LEFT RUNNING, NOT HEALTHY (maybe mid-load, not stopped): $cur -- check by hand"
  fi
}
trap cleanup EXIT
run() {  # run NAME IMAGE EXTRA-ENV
  local nm=$1 img=$2 env="$BASE${3:+
$3}"
  busy && exit 1
  cur=$nm
  SRV_MODE=image SRV_IMAGE=$img SRV_ENV="$env" SRV_WAIT=900 BENCH_OUT=$B \
    bash $B/srv2.sh up $nm gpu0 $PORT $O mxfp4 > /dev/null || { echo "UP FAILED: $nm"; exit 1; }
  timeout 1200 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$O \
    --tokenizer /mnt/storage/Models/$O $BOOK --pp 512 --tg 64 --concurrency 1 8 --runs 2 \
    --format md --save-result $B/$nm.md > $B/logs/$nm.out 2>&1
  echo "== $nm ($img ${3:-default}): rc=$? $(grep -h Coherence $B/logs/$nm.out | head -1)"
  grep -E "^\| /models" $B/$nm.md | awk -F'|' '{printf "   %-14s %s\n", $3, $4}'
  BENCH_OUT=$B bash $B/srv2.sh down $nm $PORT > /dev/null; cur=""
}
run ab-o-v18 grimoire-b70:v1.8 ""
run ab-o-v183-il0 grimoire-b70:v1.8.3 "GRIMOIRE_INTERLEAVE_CHUNK=0"
run ab-o-v183 grimoire-b70:v1.8.3 ""
echo AB DONE
