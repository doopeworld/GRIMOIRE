#!/bin/bash
# First measurements on the Arc B580 (Battlemage G21, the B60's chip, 12 GB), one server at a time; stops at the first
# failure. Ornith-1.5-35B-A3B does not fit 12 GB, so experts go partly to system RAM in every engine:
#   grimoire   v1.10.3's own G21 build, the NVFP4 checkpoint, tiered experts (GRIMOIRE_EXPERT_VRAM_GB=6: v1.10.3's
#              default budget of 16 GB is larger than the card)
#   sycl|vulkan llama.cpp's official server-intel / server-vulkan image, the Q4_K_M GGUF, --n-cpu-moe NCPU
# Same llama-benchy test for all: pp512 / pp2048, tg128, one user, two runs. Client on cores 14-15.
#   run_b580_1104.sh TAG ENGINE... [NCPU=24]
set -u
TAG=$1; shift
B=/mnt/storage/isos/grimoire-runs/bench-1022
B3=/mnt/storage/isos/grimoire-runs/bench-1003
cd "$B" || exit 9
NCPU=${NCPU:-24}
OTOK=/mnt/storage/Models/Ornith-1.5-35B-A3B
BEN="--pp 512 2048 --tg 128 --concurrency 1 --runs 2"
PB=$(for d in /sys/bus/pci/devices/*; do [ "$(cat $d/vendor)" = 0x8086 ] && [ "$(cat $d/device)" = 0xe20b ] && basename $d; done | head -1)
NB=$(basename "$(readlink -f /dev/dri/by-path/pci-$PB-render)")
chain() { readlink -f /sys/bus/pci/devices/$1 | tr / '\n' | grep -E '^[0-9a-f]{4}:'; }
unc() { local s=0 p v; for p in $(chain $PB); do for v in $(awk '/TOTAL/{print $2}' /sys/bus/pci/devices/$p/aer_dev_nonfatal /sys/bus/pci/devices/$p/aer_dev_fatal 2>/dev/null); do s=$((s+v)); done; done; echo $s; }
echo "B580 = $PB = $NB, link $(cat /sys/bus/pci/devices/$(chain $PB | head -1)/current_link_speed) x$(cat /sys/bus/pci/devices/$(chain $PB | head -1)/current_link_width)"
exec 7>/mnt/user/isos/grimoire-runs/b580.lock
llama() { # name image
  local name=$1 img=$2 port=6895 ok=0 i
  flock -n 7 || { echo "b580.lock held"; exit 9; }
  fuser /dev/dri/$NB >/dev/null 2>&1 && { echo "$NB is busy"; exit 9; }
  local u0=$(unc)
  if ! curl -s -m 3 -o /dev/null http://127.0.0.1:8999/1661-0.txt; then   # llama-benchy's book, as dual_engine
    (cd $B3/book || exit; exec 7>&- > /dev/null 2>&1 < /dev/null
     nohup setsid python3 -m http.server 8999 --bind 127.0.0.1 > $B/logs/book-server.log 2>&1 &); sleep 2
  fi
  echo "== $(date +%T) $name $img --n-cpu-moe $NCPU"
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --name "$name" --network host --device /dev/dri/$NB -v /mnt/models:/models2:ro "$img" \
    -m /models2/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf -ngl 999 --n-cpu-moe "$NCPU" -c 8192 -fa on \
    -t 14 --host 127.0.0.1 --port $port > /dev/null || exit 9
  for i in $(seq 1 180); do
    curl -s -m 2 http://127.0.0.1:$port/health 2>/dev/null | grep -q ok && { ok=1; break; }
    docker ps --format '{{.Names}}' | grep -qx "$name" || break
    sleep 5
  done
  docker logs "$name" 2>&1 | grep -E "offload|buffer size|using device|SYCL|Vulkan|ggml_sycl|found .* devices" | head -12 | cut -c1-160 > "logs/$name.server.head"
  cat "logs/$name.server.head"
  if [ "$ok" = 1 ]; then
    taskset -c 14,15 timeout 600 python3 $B3/ctest.py $port 2>&1 | grep -E "answer|ttft" | sed 's/^/   [ctest] /'
    taskset -c 14,15 timeout 1800 uvx llama-benchy@0.4.0 --base-url http://localhost:$port/v1 --model m \
      --tokenizer $OTOK --book-url http://127.0.0.1:8999/1661-0.txt $BEN --format md --save-result "logs/$name.benchy.md" > "logs/$name.benchy.out" 2>&1
    grep -E "^\| " "logs/$name.benchy.md" | awk -F'|' '{printf "     %-16s %-22s %s\n", $3, $4, $5}'
  else echo "   server did not come up"; docker logs "$name" 2>&1 | tail -5; fi
  docker stop -t 60 "$name" > /dev/null; docker logs "$name" > "logs/$name.server.log" 2>&1; docker rm "$name" > /dev/null
  [ "$(unc)" = "$u0" ] || { echo "UNCORRECTABLE errors on the B580 path"; exit 9; }
  flock -u 7
  [ "$ok" = 1 ] || exit 9
}
for e in "$@"; do
  case $e in
    grimoire|grimoire-default)
              if [ -n "${GTREE:-}" ]; then export GRIM_TREE=$GTREE HOSTBIN=1 IMG=grimoire-b70:latest   # a build tree's bin/
              else export HOSTBIN=0 IMG=grimoire-b70:v1.10.3; fi
              # grimoire-default (11-05): no GRIMOIRE_EXPERT_VRAM_GB, the budget v1.10.4 picks (<= 60% of the card)
              vr="GRIMOIRE_EXPERT_VRAM_GB=6"; [ "$e" = grimoire-default ] && vr=""
              LOAD_MAX=1500 CLIENT_CPUS=14,15 TOKDIR=$OTOK \
                bash ./dual_engine_1102.sh "b5-$TAG-$e" B5 /models/Ornith-1.5-35B-A3B-NVFP4 mxfp4 8192 \
                "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_TIERED_MOE=1 $vr GRIMOIRE_CX_STATS=1" \
                ctest,benchy $BEN > "logs/b5-$TAG-$e.run" 2>&1
              rc=$?
              grep -E "ready=|answer:|Coherence|pp512|pp2048|tg128|tiered experts|server stopped|LEFT RUNNING|STOP" "logs/b5-$TAG-$e.run" | cut -c1-150
              grep -E "tiered experts|prompt attention|Device|device:" "logs/b5-$TAG-$e.server.log" 2>/dev/null | head -6 | cut -c1-160
              { [ "$rc" = 0 ] && grep -q "server stopped gracefully" "logs/b5-$TAG-$e.run"; } || { echo "B580_STOP $e rc=$rc"; exit 9; } ;;
    sycl)     llama "b5-$TAG-llama-sycl" ghcr.io/ggml-org/llama.cpp:server-intel ;;
    vulkan)   llama "b5-$TAG-llama-vulkan" ghcr.io/ggml-org/llama.cpp:server-vulkan ;;
    *) echo "unknown engine $e"; exit 2 ;;
  esac
done
echo "B580_DONE $(date +%T)"
