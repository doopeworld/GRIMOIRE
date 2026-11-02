#!/bin/bash
# CPU only, no GPU device. The CPU-expert pool bench with chosen cores kept busy, as a build or a desktop would.
#   cxb_ab_1102.sh TAG "busy cores" [calls] [experts a call] [gap us]
# Engine thread list: 14 threads, workers pinned to 1-13, as Flash-Next runs on this CPU.
set -u
TAG=$1 BUSY=${2:-} CALLS=${3:-3000} N=${4:-2} GAP=${5:-600}
W=/mnt/storage/isos/grimoire-cxpool-1102/tree
pids=()
for c in $BUSY; do taskset -c "$c" sh -c "while :; do :; done" & pids+=($!); done
cleanup() { for p in "${pids[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
echo "== $TAG busy=[${BUSY:-none}] build=$(docker ps --format "{{.Names}}" | grep -c grim-release-build) load=$(cut -d" " -f1 /proc/loadavg)"
docker run --rm --name grim-cxb-$$ -v "$W":/grimoire -w /grimoire grimoire-dev:latest bash -lc \
  "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; bin/cpu_expert_bench.$TAG ffn 0 1 0 1 $CALLS $N $GAP" 2>&1 | grep -vE "^pool:|AnonHuge" | cut -c1-170
