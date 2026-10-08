#!/bin/bash
# tp_ladder_old_g0.sh -- tp_ladder_g0.sh with YESTERDAY'S build (image grimoire-b70:v1.9-dev: the
# shared-memory exchange registered with the GPU driver, the accounted prompt budget), gpu0 ONLY.  Where does the two-rank tensor-parallel workload start to strain a
# card?  Both ranks on gpu0 (GRIMOIRE_MULTI_DEVICES=0,0, only gpu0's render node mapped), Qwen3.8 GPTQ
# int4, --ctx 16384, the host bin/ build.  A sampler logs every 200 ms: VRAM in use (the driver's
# vram0_mm), system-memory GTT in use (eviction shows here), card power (energy counter); new xe /
# pcieport / AER kernel lines are collected per step.  llama-benchy c1, tg64, one run per prompt size
# 512 -> 2048 -> 4096 -> 8192, for each configuration:
#   nocap      GRIMOIRE_PREFILL_CHUNK=32768  (a whole prompt in one call: the old behaviour)
#   chunk2048  GRIMOIRE_PREFILL_CHUNK=2048
# Two ranks on ONE card share its VRAM, so this is harsher on memory than two cards.  Refuses if
# anything holds a GPU; each container is stopped right after its benchmark.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003; R=/mnt/storage/isos/grimoire-fuse; PORT=6898
M=Qwen3.8-27B-GPTQ-Int4-MTP-BF16
BOOK="--book-url http://127.0.0.1:8999/1661-0.txt"
G0=$(basename "$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render)")
HW=$(ls -d /sys/bus/pci/devices/0000:03:00.0/hwmon/* | head -1); DBG=/sys/kernel/debug/dri/0000:03:00.0
OUT=$B/logs/tp_ladder_old_g0; mkdir -p $OUT
guard() {
  case "$G0" in renderD*) ;; *) echo "gpu0 render node not found -- refusing"; return 1;; esac
  [ -z "$(docker ps -q)" ] || { echo "a container is running -- refusing: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; return 1; }
  local p; for p in /proc/[0-9]*; do
    ls -l $p/fd 2>/dev/null | grep -q "/dev/dri/renderD1[23][0-9]" && { echo "a GPU is held by $(cat $p/comm) -- refusing"; return 1; }
  done; return 0; }
sampler() {  # sampler CSV  (runs until CSV.run disappears)
  local f=$1 e0 e1 t0 t1 v g
  e0=$(cat $HW/energy1_input); t0=$(date +%s%N)
  echo "t vram_MiB gtt_MiB power_W" > $f
  while [ -f $f.run ]; do
    sleep 0.2
    e1=$(cat $HW/energy1_input); t1=$(date +%s%N)
    v=$(awk '/usage:/{print int($2/1048576); exit}' $DBG/vram0_mm)
    g=$(awk '/usage:/{print int($2*4/1024); exit}' $DBG/gtt_mm)
    echo "$(( t1/1000000 )) $v $g $(( (e1-e0)*1000/(t1-t0) ))" >> $f
    e0=$e1; t0=$t1
  done; }
window() {  # window CSV FROM_MS TO_MS -> "vram max / gtt max / power max, mean"
  awk -v a=$2 -v b=$3 'NR>1 && $1>=a && $1<=b {if($2>v)v=$2; if($3>g)g=$3; if($4>p)p=$4; s+=$4; n++}
    END {printf "VRAM max %d MiB, GTT max %d MiB, power max %d W mean %d W", v, g, p, n?s/n:0}' $1; }
cur=""
cleanup() { [ -n "$cur" ] || return; docker ps --format '{{.Names}}' | grep -qx "$cur" || return
  if curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q ok; then
    docker logs $cur > $OUT/$cur.log 2>&1; docker stop --time 120 $cur >/dev/null; docker rm $cur >/dev/null 2>&1
  else echo "LEFT RUNNING, NOT HEALTHY (maybe mid-load): $cur -- check by hand"; fi; }
trap 'cleanup; rm -f $OUT/*.csv.run' EXIT
if ! curl -s -m 3 -o /dev/null http://127.0.0.1:8999/1661-0.txt; then
  (cd $B/book && nohup setsid python3 -m http.server 8999 --bind 127.0.0.1 > $B/logs/book-server.log 2>&1 &); sleep 2
fi
run() {  # run NAME CHUNK
  local nm=$1 chunk=$2 csv=$OUT/$1.csv
  guard || exit 9
  echo "== $nm (GRIMOIRE_PREFILL_CHUNK=$chunk), gpu0 = $G0"
  touch $csv.run; sampler $csv & local sp=$!
  docker rm -f $nm >/dev/null 2>&1
  docker run -d --name $nm --init --stop-timeout 300 --network host --ipc=host \
    --device /dev/dri/$G0 -v /dev/dri/by-path:/dev/dri/by-path:ro \
    -v /mnt/storage/Models:/models -v /mnt/storage/isos/grimoire-cache:/cache \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu -e ZE_AFFINITY_MASK=0 -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
    -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/cache/sycl -e GRIMOIRE_PREFILL_CHUNK=$chunk \
    -e GRIMOIRE_MULTI_GPUS=2 -e GRIMOIRE_MULTI_DEVICES=0,0 \
    grimoire-b70:v1.9-dev multi TP --model /models/$M --proj int4 --ctx 16384 --host 0.0.0.0 --port $PORT >/dev/null
  cur=$nm
  local ok=0 i; for i in $(seq 1 200); do sleep 3
    curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && { ok=1; break; }
    docker ps --format '{{.Names}}' | grep -qx $nm || break; done
  echo "   ready=$ok after $((i*3))s; idle after load: $(window $csv $(( $(date +%s%3N) - 1500 )) $(date +%s%3N))"
  docker logs $nm 2>&1 | grep -E "all-reduce|chunk cap|error|fail" | head -4 | cut -c1-160 | sed 's/^/   /'
  if [ $ok = 1 ]; then
    local P a b d0 st
    for P in 512 2048 4096 8192; do
      d0=$(dmesg | wc -l); a=$(date +%s%3N)
      timeout 900 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$M \
        --tokenizer /mnt/storage/Models/$M $BOOK --pp $P --tg 64 --runs 1 \
        --format md --save-result $OUT/$nm-pp$P.md > $OUT/$nm-pp$P.out 2>&1; st=$?
      b=$(date +%s%3N)
      echo "   pp$P: rc=$st $(grep -E "^\| /models" $OUT/$nm-pp$P.md 2>/dev/null | awk -F'|' '{printf "%s=%s ", $3, $4}' | tr -s ' ')"
      echo "      $(window $csv $a $b)"
      dmesg | tail -n +$((d0+1)) | grep -E "xe |pcieport|AER" | head -4 | cut -c1-160 | sed 's/^/      KERNEL: /'
      curl -s -m 3 http://127.0.0.1:$PORT/health | grep -q ok || { echo "   server no longer healthy -- stopping this ladder"; break; }
    done
  fi
  cleanup; cur=""
  rm -f $csv.run; wait $sp 2>/dev/null
  echo "   stopped; containers: $(docker ps -q | wc -l); gpu0 AER: $(for p in 00:01.0 01:00.0; do tr '\n' ' ' < /sys/bus/pci/devices/0000:$p/aer_dev_correctable | grep -oE 'TOTAL_ERR_COR [0-9]+'; done | tr '\n' ' ')"
}
for t in ${*:-nocap chunk2048}; do
  case $t in
    nocap)     run zc-lado-nocap 32768 ;;
    chunk2048) run zc-lado-c2048 2048 ;;
  esac
done
echo "OLD LADDER DONE"
