#!/bin/bash
# pm_probe_g0.sh -- gpu0 ONLY.  How many runtime-PM sleep/wake cycles does a normal GRIMOIRE
# llama-benchy run put the card (and every bridge on its PCIe path) through?  A sampler reads
# power/runtime_status of gpu0 03:00.0 and its bridges 01:00.0 / 00:01.0 every 100 ms and logs
# each transition; the xe lines of dmesg are kept too.  Nothing is changed on the host.
# One single-rank server (Qwen3.8 GPTQ, image v1.8.3), llama-benchy c1 pp512/2048/4096 tg128,
# stopped right after.  Refuses if anything holds a GPU.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003; PORT=6901; IMG=grimoire-b70:v1.8.3
M=Qwen3.8-27B-GPTQ-Int4-MTP-BF16
BOOK="--book-url http://127.0.0.1:8999/1661-0.txt"
G0=$(basename "$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render)")
[ -n "$G0" ] || { echo "gpu0 render node not found"; exit 9; }
[ -z "$(docker ps -q)" ] || { echo "a container is running -- refusing"; exit 9; }
for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q "/dev/dri/renderD1[23][0-9]" && { echo "a GPU is held by $(cat $p/comm) -- refusing"; exit 9; }; done
DEVS="0000:03:00.0 0000:01:00.0 0000:00:01.0"
OUT=$B/logs/pm_probe_g0
mkdir -p $OUT; : > $OUT/transitions.log
d0=$(dmesg | wc -l)
for d in $DEVS; do echo "$d start active=$(cat /sys/bus/pci/devices/$d/power/runtime_active_time) suspended=$(cat /sys/bus/pci/devices/$d/power/runtime_suspended_time)"; done
touch $OUT/run
( declare -A last
  while [ -f $OUT/run ]; do
    t=$(date +%s.%N | cut -c1-14)
    for d in $DEVS; do s=$(cat /sys/bus/pci/devices/$d/power/runtime_status)
      [ "${last[$d]:-}" != "$s" ] && echo "$t $d $s" >> $OUT/transitions.log && last[$d]=$s; done
    sleep 0.1
  done ) &
SAMPLER=$!
if ! curl -s -m 3 -o /dev/null http://127.0.0.1:8999/1661-0.txt; then
  (cd $B/book && nohup setsid python3 -m http.server 8999 --bind 127.0.0.1 > $B/logs/book-server.log 2>&1 &); sleep 2
fi
SRV_MODE=image SRV_IMAGE=$IMG SRV_ENV="ZE_AFFINITY_MASK=0
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
SYCL_CACHE_PERSISTENT=1
SYCL_CACHE_DIR=/cache/sycl" SRV_CTX=8192 SRV_WAIT=600 BENCH_OUT=$B bash $B/srv2.sh up pm-g0 gpu0 $PORT $M int4 | head -2
if curl -s -m 3 http://127.0.0.1:$PORT/health | grep -q ok; then
  timeout 1200 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model /models/$M \
    --tokenizer /mnt/storage/Models/$M $BOOK --pp 512 2048 4096 --tg 128 --runs 2 \
    --format md --save-result $OUT/benchy.md > $OUT/benchy.out 2>&1
  echo "benchy rc=$? $(grep -h Coherence $OUT/benchy.out | head -1)"
  BENCH_OUT=$B bash $B/srv2.sh down pm-g0 $PORT
fi
sleep 3; rm -f $OUT/run; wait $SAMPLER 2>/dev/null
for d in $DEVS; do echo "$d end   active=$(cat /sys/bus/pci/devices/$d/power/runtime_active_time) suspended=$(cat /sys/bus/pci/devices/$d/power/runtime_suspended_time)"; done
echo "transitions per device (each 'suspended' line = one sleep, each 'active' after it = one wake):"
awk '{n[$2" "$3]++} END {for (k in n) print "  " k, n[k]}' $OUT/transitions.log | sort
echo "new xe / pcieport lines in dmesg:"; dmesg | tail -n +$((d0+1)) | grep -E "xe |pcieport|AER" | tail -8
echo "PM PROBE DONE"
