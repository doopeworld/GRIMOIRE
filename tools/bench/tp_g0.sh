#!/bin/bash
# tp_g0.sh -- Megatron tensor parallel, tested on gpu0 ALONE (only renderD128 is mapped):
#   A  one rank, the host bin/ build, Qwen3.8-27B GPTQ-Int4, no MTP           (reference)
#   B  "multi TP", both ranks on device 0 (GRIMOIRE_MULTI_DEVICES=0,0)        (greedy text must match A)
#   C  B + MTP                                                                 (MTP path under TP)
# Two ranks on ONE card share its memory bandwidth, so B/C time the communication overhead,
# not the two-card speed-up.  Refuses if anything holds a GPU; each container is stopped right
# after its own requests.  Usage: tp_g0.sh [A] [B] [C]   (default: all three)
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1003; R=/mnt/storage/isos/grimoire-fuse; PORT=6898
M=Qwen3.8-27B-GPTQ-Int4-MTP-BF16
aer() { local o="" p f v; for p in 00:01.0 01:00.0; do o+="$p"; for f in correctable nonfatal fatal; do
  v=$(tr '\n' ' ' < /sys/bus/pci/devices/0000:$p/aer_dev_$f | grep -oE "TOTAL_ERR_[A-Z]+ [0-9]+" | awk '{print $2}')
  o+=" ${f:0:3}=$v"; done; o+="; "; done; echo "$o"; }
guard() {
  [ "$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render)" = /dev/dri/renderD128 ] || { echo "gpu0 is not renderD128 -- refusing"; return 1; }
  [ -z "$(docker ps -q)" ] || { echo "a container is running -- refusing: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; return 1; }
  local p; for p in /proc/[0-9]*; do
    ls -l $p/fd 2>/dev/null | grep -q "/dev/dri/renderD1[23][0-9]" && { echo "a GPU is held by $(cat $p/comm) -- refusing"; return 1; }
  done; return 0; }
cur=""
cleanup() { [ -n "$cur" ] || return; docker ps --format '{{.Names}}' | grep -qx "$cur" || return
  if curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q ok; then
    docker logs $cur > $B/logs/$cur.log 2>&1; docker stop --time 120 $cur >/dev/null; docker rm $cur >/dev/null 2>&1
  else echo "LEFT RUNNING, NOT HEALTHY (maybe mid-load): $cur -- check by hand"; fi; }
trap cleanup EXIT
run() {  # run NAME single|tp "ENV1 ENV2 ..."
  local nm=$1 mode=$2 extra=$3
  guard || exit 9
  echo "== $nm ($mode${extra:+, $extra})   gpu0 AER before: $(aer)"
  docker rm -f $nm >/dev/null 2>&1
  local envs=(-e ONEAPI_DEVICE_SELECTOR=level_zero:gpu -e ZE_AFFINITY_MASK=0 -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
              -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/cache/sycl -e GRIMOIRE_PREFILL_CHUNK=1024)
  local kv; for kv in $extra; do envs+=(-e "$kv"); done
  local common=(-d --name $nm --init --stop-timeout 300 --network host --ipc=host
                --device /dev/dri/renderD128 -v /dev/dri/by-path:/dev/dri/by-path:ro
                -v $R/bin:/grimoire/bin:ro -v $R/tools:/grimoire/tools:ro --tmpfs /opt/grimoire/lib
                -v $R/docker-entrypoint.sh:/usr/local/bin/docker-entrypoint.sh:ro
                -v /mnt/storage/Models:/models -v /mnt/storage/isos/grimoire-cache:/cache)
  local srv=(--model /models/$M --proj int4 --ctx 4096 --host 0.0.0.0 --port $PORT)
  if [ $mode = single ]; then
    docker run "${common[@]}" "${envs[@]}" --entrypoint /grimoire/bin/grimoire-server grimoire-b70:latest "${srv[@]}" >/dev/null
  else
    docker run "${common[@]}" "${envs[@]}" -e GRIMOIRE_MULTI_GPUS=2 -e GRIMOIRE_MULTI_DEVICES=0,0 \
      grimoire-b70:latest multi TP "${srv[@]}" >/dev/null
  fi
  cur=$nm
  local ok=0 i; for i in $(seq 1 200); do sleep 3
    curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && { ok=1; break; }
    docker ps --format '{{.Names}}' | grep -qx $nm || break; done
  echo "   ready=$ok after $((i*3))s"
  docker logs $nm 2>&1 | grep -E "Megatron|legacy|parallel|speculation|capabilities|rank [01]: (worker|connected)|error|fail|refus" | head -12 | cut -c1-200 | sed 's/^/   /'
  if [ $ok = 1 ]; then
    timeout 1200 python3 $B/gen_save.py $PORT $B/logs/$nm.json 2>&1 | sed 's/^/   /'
    timeout 600 python3 $B/ctest.py $PORT 2>&1 | tail -1 | sed 's/^/   /'
  else
    docker logs $nm 2>&1 | tail -25 | sed 's/^/   | /'
  fi
  cleanup; cur=""
  echo "   stopped; containers running: $(docker ps -q | wc -l); gpu0 AER after: $(aer)"
}
want="${*:-A B C}"
for t in $want; do
  case $t in
    A) run zc-tp-a single "" ;;
    B) run zc-tp-b tp "" ;;
    C) run zc-tp-c tp "GRIMOIRE_MTP=1" ;;
  esac
done
python3 - "$B" <<'EOF'
import json, sys, os
B = sys.argv[1]
def load(n):
    p = "%s/logs/%s.json" % (B, n)
    return json.load(open(p)) if os.path.exists(p) else None
a = load("zc-tp-a")
for n in ("zc-tp-b", "zc-tp-c"):
    x = load(n)
    if not (a and x): continue
    for tag in a:
        ta, tx = a[tag]["text"], x[tag]["text"]
        same = 0
        while same < min(len(ta), len(tx)) and ta[same] == tx[same]: same += 1
        print("%s vs A  %-8s identical=%s  common prefix %d of %d chars  (%.2fs vs %.2fs)" % (
            n, tag, ta == tx, same, len(ta), x[tag]["seconds"], a[tag]["seconds"]))
EOF
echo "TP G0 DONE"
