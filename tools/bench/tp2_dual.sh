#!/bin/bash
# tp2_dual.sh -- Qwen3.8-27B-FP8 on BOTH B70s with Megatron tensor parallel (host bin/ build).
# RUN ONLY WITH IAN'S EXPLICIT GO FOR gpu1.  One container per configuration, stopped right after
# its own benchmark; link errors on both cards' paths are read before and after each run, and the
# script stops at the first new fatal / non-fatal error or a jump in gpu1's correctable count.
#   tp2_dual.sh [mtp] [plain]     (default: mtp then plain)
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003; R=/mnt/storage/isos/grimoire-fuse; PORT=6897
M=Qwen3.8-27B-FP8
BOOK="--book-url http://127.0.0.1:8999/1661-0.txt"
PATHS="00:01.0 01:00.0 00:06.2 09:00.0 0a:01.0"
aer() { local o="" p f v; for p in $PATHS; do o+="$p"; for f in correctable nonfatal fatal; do
  v=$(tr '\n' ' ' < /sys/bus/pci/devices/0000:$p/aer_dev_$f 2>/dev/null | grep -oE "TOTAL_ERR_[A-Z]+ [0-9]+" | awk '{print $2}')
  o+=" ${f:0:3}=${v:--}"; done; o+="; "; done; echo "$o"; }
cor09() { tr '\n' ' ' < /sys/bus/pci/devices/0000:09:00.0/aer_dev_correctable | grep -oE "TOTAL_ERR_COR [0-9]+" | awk '{print $2}'; }
bad() { local p f v; for p in $PATHS; do for f in nonfatal fatal; do
  v=$(tr '\n' ' ' < /sys/bus/pci/devices/0000:$p/aer_dev_$f 2>/dev/null | grep -oE "TOTAL_ERR_[A-Z]+ [0-9]+" | awk '{print $2}')
  [ -n "$v" ] && [ "$v" != 0 ] && { echo "$p $f=$v"; return 0; }; done; done; return 1; }
guard() {
  [ "$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render)" = /dev/dri/renderD128 ] || { echo "gpu0 is not renderD128 -- refusing"; return 1; }
  [ "$(readlink -f /dev/dri/by-path/pci-0000:0b:00.0-render)" = /dev/dri/renderD131 ] || { echo "gpu1 is not renderD131 -- refusing"; return 1; }
  lspci -s 0b:00.0 | grep -q B70 || { echo "gpu1 is not on the bus -- refusing"; return 1; }
  [ -z "$(docker ps -q)" ] || { echo "a container is running -- refusing: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; return 1; }
  local p; for p in /proc/[0-9]*; do
    ls -l $p/fd 2>/dev/null | grep -q "/dev/dri/renderD1[23][0-9]" && { echo "a GPU is held by $(cat $p/comm) -- refusing"; return 1; }
  done
  bad && { echo "uncorrectable link errors already logged -- refusing"; return 1; }
  return 0; }
cur=""
cleanup() { [ -n "$cur" ] || return; docker ps --format '{{.Names}}' | grep -qx "$cur" || return
  if curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q ok; then
    docker logs $cur > $B/logs/$cur.log 2>&1; docker stop --time 120 $cur >/dev/null; docker rm $cur >/dev/null 2>&1
  else echo "LEFT RUNNING, NOT HEALTHY (maybe mid-load): $cur -- check by hand, do NOT kill it"; fi; }
trap cleanup EXIT
if ! curl -s -m 3 -o /dev/null http://127.0.0.1:8999/1661-0.txt; then
  (cd $B/book && nohup setsid python3 -m http.server 8999 --bind 127.0.0.1 > $B/logs/book-server.log 2>&1 &); sleep 2
fi
run() {  # run NAME "ENV..." BENCHY-ARGS...
  local nm=$1 extra=$2; shift 2
  guard || exit 9
  local c0=$(cor09)
  echo "== $nm (${extra:-plain})   AER before: $(aer)"
  local envs=(-e ONEAPI_DEVICE_SELECTOR=level_zero:gpu -e ZE_AFFINITY_MASK=0,1 -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
              -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/cache/sycl -e GRIMOIRE_MULTI_GPUS=2)
  local kv; for kv in $extra; do envs+=(-e "$kv"); done
  docker rm -f $nm >/dev/null 2>&1
  docker run -d --name $nm --init --stop-timeout 300 --network host --ipc=host \
    --device /dev/dri/renderD128 --device /dev/dri/renderD131 -v /dev/dri/by-path:/dev/dri/by-path:ro \
    -v $R/bin:/grimoire/bin:ro -v $R/tools:/grimoire/tools:ro --tmpfs /opt/grimoire/lib \
    -v $R/docker-entrypoint.sh:/usr/local/bin/docker-entrypoint.sh:ro \
    -v /mnt/storage/Models:/models -v /mnt/storage/isos/grimoire-cache:/cache "${envs[@]}" \
    grimoire-b70:latest multi TP --model /models/$M --proj fp8 --ctx 16384 --host 0.0.0.0 --port $PORT >/dev/null
  cur=$nm
  local ok=0 i; for i in $(seq 1 200); do sleep 3
    curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && { ok=1; break; }
    docker ps --format '{{.Names}}' | grep -qx $nm || break
    bad >/dev/null && { echo "   LINK ERROR during load: $(bad)"; break; }; done
  echo "   ready=$ok after $((i*3))s"
  docker logs $nm 2>&1 | grep -E "Megatron|all-reduce|parallel|speculation|refus|error|fail" | head -8 | cut -c1-200 | sed 's/^/   /'
  if [ $ok = 1 ]; then
    timeout 600 python3 $B/ctest.py $PORT 2>&1 | sed 's/^/   /'
    timeout 900 python3 $B/gen_save.py $PORT $B/logs/$nm.json 2>&1 | sed 's/^/   /'
    if [ $# -gt 0 ]; then
      timeout 1500 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$M \
        --tokenizer /mnt/storage/Models/$M $BOOK "$@" --format md --save-result $B/$nm.md > $B/logs/$nm.out 2>&1
      echo "   benchy rc=$? $(grep -h Coherence $B/logs/$nm.out | head -1)"
      grep -E "^\| /models" $B/$nm.md | awk -F'|' '{printf "     %-14s %s\n", $3, $4}'
    fi
  fi
  cleanup; cur=""
  local c1=$(cor09)
  echo "   stopped; containers: $(docker ps -q | wc -l); AER after: $(aer)  (09:00.0 correctable +$((c1-c0)))"
  bad && { echo "UNCORRECTABLE LINK ERROR -- stopping here"; exit 8; }
  [ $((c1-c0)) -gt 20 ] && { echo "gpu1 link errors jumped -- stopping here"; exit 8; }
}
want="${*:-mtp plain}"
for t in $want; do
  case $t in
    mtp)   run zc-tp2-mtp "GRIMOIRE_MTP=1" --pp 512 2048 --tg 128 --runs 2 ;;
    plain) run zc-tp2-plain "" ;;
  esac
done
echo "TP2 DUAL DONE"
