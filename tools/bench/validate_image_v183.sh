#!/bin/bash
# validate_v183_g0.sh -- release image v1.8.3 (ESIMD FP8 decode GEMV, FP8 small-M DPAS GEMM,
# one card per rank in "multi"), validated on gpu0 ONLY.  gpu1 is off limits: two-GPU runs are
# Ian's to start (the two-GPU script was retired as validate_v183_TWO_GPU_DO_NOT_RUN.sh).
# Every server lives only for its own test and is stopped right after it.
#  1. Qwen3.8-27B GPTQ + MTP defaults, 8 slots, --ctx 16384: tool calling end to end, then
#     llama-benchy pp512/pp4096 x tg128 x 1/8 users -- the same run as validate_v182.sh.
#  2. Same model, MTP draft head in FP8 (GRIMOIRE_MTP_HEAD_FMT=fp8): the new FP8 kernels inside
#     the image (GEMV at 1 row, DPAS small-M when 2-4 requests draft together); 1 and 4 users.
#     A broken FP8 head would accept ~nothing and drop 1-user decode to plain speed.
#  3. Ornith MXFP4, 8 slots: pp512/tg64 x 1/8 users (MXFP4 path unchanged; sanity).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003; PORT=6901; IMG=grimoire-b70:v1.8.3
BOOK="--book-url http://127.0.0.1:8999/1661-0.txt"
BASE="ZE_AFFINITY_MASK=0
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
SYCL_CACHE_PERSISTENT=1
SYCL_CACHE_DIR=/cache/sycl"
mkdir -p $B/logs

aer() {  # gpu0's path (root port, switch, card) and gpu1's switch port -- read only
  local out="" p f v
  for p in 00:01.0 01:00.0 03:00.0 09:00.0; do
    out+="$p"
    for f in correctable nonfatal fatal; do
      v=$(tr '\n' ' ' < /sys/bus/pci/devices/0000:$p/aer_dev_$f 2>/dev/null | grep -oE "TOTAL_ERR_[A-Z]+ [0-9]+" | awk '{print $2}')
      out+=" ${f:0:3}=${v:--}"
    done
    out+="; "
  done
  echo "$out"
}
busy() {  # anything on any GPU?  running containers with GPU devices, processes holding a render node
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
cleanup() {  # stop our server only if it is up and idle; a server that is not healthy may be mid-load
  [ -z "$cur" ] && return
  docker ps --format '{{.Names}}' | grep -qx "$cur" || return
  if curl -s -m 5 http://localhost:$PORT/health 2>/dev/null | grep -q ok; then
    BENCH_OUT=$B bash $B/srv2.sh down $cur $PORT
  else
    echo "LEFT RUNNING, NOT HEALTHY (maybe mid-load, not stopped): $cur -- check by hand"
  fi
}
trap cleanup EXIT
up() {  # up NAME MODEL PROJ ENV [CTX]
  busy && exit 1
  echo "AER before $1: $(aer)"
  cur=$1
  SRV_MODE=image SRV_IMAGE=$IMG SRV_ENV="$4" SRV_CTX="${5:-8192}" SRV_WAIT=900 \
    BENCH_OUT=$B bash $B/srv2.sh up $1 gpu0 $PORT $2 $3 || { echo "UP FAILED: $1"; exit 1; }
}
bench() {  # bench NAME MODEL OUT ARGS...
  local nm=$1 m=$2 out=$3; shift 3
  timeout 2400 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$m \
    --tokenizer /mnt/storage/Models/$m $BOOK "$@" --format md --save-result $B/$out.md > $B/logs/$out.out 2>&1
  echo "  benchy rc=$? $(grep -h Coherence $B/logs/$out.out | head -1)"
  grep -E "^\| /models" $B/$out.md | awk -F'|' '{printf "   %-14s %s\n", $3, $4}'
}
down() {  # down NAME
  BENCH_OUT=$B bash $B/srv2.sh down $1 $PORT; cur=""
  echo "AER after $1: $(aer)"
}

# llama-benchy's book, served locally (gutenberg.org can time out); gone after a reboot
if ! curl -s -m 3 -o /dev/null http://127.0.0.1:8999/1661-0.txt; then
  (cd $B/book && nohup setsid python3 -m http.server 8999 --bind 127.0.0.1 > $B/logs/book-server.log 2>&1 &)
  sleep 2
fi
curl -s -m 3 -o /dev/null -w "book server: HTTP %{http_code}\n" http://127.0.0.1:8999/1661-0.txt
docker image inspect $IMG --format "image $IMG {{.Id}} created {{.Created}}" | cut -c1-90

Q=Qwen3.8-27B-GPTQ-Int4-MTP-BF16
echo "== 1. $Q + MTP defaults, 8 slots, --ctx 16384"
up v183g0-q $Q int4 "$BASE
GRIMOIRE_MTP=1
GRIMOIRE_SEQ_SLOTS=8" 16384
python3 $B/tool_e2e.py $PORT 2>&1 | grep -E "^[0-9]\." | cut -c1-200
bench v183g0-q $Q v183g0-q --pp 512 4096 --tg 128 --concurrency 1 8 --runs 2
down v183g0-q

echo "== 2. $Q + MTP, FP8 draft head (new FP8 kernels), 8 slots"
up v183g0-f8 $Q int4 "$BASE
GRIMOIRE_MTP=1
GRIMOIRE_MTP_HEAD_FMT=fp8
GRIMOIRE_SEQ_SLOTS=8" 16384
docker logs v183g0-f8 2>&1 | grep -iE "mtp|fp8|head" | head -6 | sed 's/^/    /'
bench v183g0-f8 $Q v183g0-f8 --pp 512 --tg 128 --concurrency 1 4 --runs 2
down v183g0-f8

O=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
echo "== 3. $O, 8 slots"
up v183g0-o $O mxfp4 "$BASE
GRIMOIRE_SEQ_SLOTS=8"
bench v183g0-o $O v183g0-o --pp 512 --tg 64 --concurrency 1 8 --runs 2
down v183g0-o
echo ALL DONE
