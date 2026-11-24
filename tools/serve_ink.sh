#!/bin/bash
# Inkling 2 Mini OpenAI server on the Tower (GRIMOIRE engine `inkling2_gpu serve` + tools/inkling2_server.py).
#   serve_ink.sh start CARDS CTX [CHUNK] [PORT]   -- 1 card = 03:00.0 (INK_PCI overrides), 2 cards = 03:00.0 + 07:00.0
#   serve_ink.sh stop                              -- front end first, then QUIT to the idle engine, container removed
# Never touches the B580.  Env INK_* switches are passed to the engine container.
X=/mnt/storage/isos/grimoire-runs/ink-1123
M=${MODEL:-/mnt/storage/Models/Inkling-2-Mini-Uncensored}
NAME=ink-1123-serve
SOCK=$X/serve/ink.sock
cmd=$1; shift
if [ "$cmd" = start ]; then
  CARDS=${1:-2}; CTX=${2:-32768}; CHUNK=${3:-4096}; PORT=${4:-6911}   # chunk 4096: 8K prefill 3,276 -> 3,612 (BetterBench)
  if docker ps -a --format "{{.Names}}" | grep -q "^ink-1123-"; then echo "an ink-1123 container exists: stop it first"; docker ps -a | grep ink-1123; exit 1; fi
  mkdir -p $X/serve $X/logs; rm -f $SOCK
  P0=${INK_PCI:-0000:03:00.0}
  DEV="--device $(readlink -f /dev/dri/by-path/pci-$P0-render):/dev/dri/renderD128"; SEL=level_zero:0
  if [ "$CARDS" = 2 ]; then
    DEV="--device $(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render):/dev/dri/renderD128 --device $(readlink -f /dev/dri/by-path/pci-0000:07:00.0-render):/dev/dri/renderD129"; SEL=level_zero:0,1
  fi
  ENVS=""
  for v in $(env | grep -oE "^INK_[A-Z0-9_]+" | grep -v "^INK_PCI$"); do ENVS="$ENVS -e $v=${!v}"; done
  docker run -d --name $NAME --network none $DEV -v "$X:/work" -v "$M:/inkling:ro" \
    -e ONEAPI_DEVICE_SELECTOR=$SEL -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 -e INK_I8=${INK_I8:-1} -e INK_ESMOE=${INK_ESMOE:-1} $ENVS \
    --entrypoint /bin/bash grimoire-b70:v1.10.10 -c "cd /work/src && ./bin/inkling2_gpu /inkling serve /work/serve/ink.sock $CTX $CHUNK $CARDS" > /dev/null || exit 1
  t=0
  until docker logs $NAME 2>&1 | grep -q "^READY"; do
    sleep 3; t=$((t+3))
    if [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" != true ]; then echo "engine exited:"; docker logs $NAME 2>&1 | tail -20; docker rm $NAME > /dev/null; exit 1; fi
    if [ $t -ge 900 ]; then echo "engine not ready after 15 min (left running)"; docker logs $NAME 2>&1 | tail -5; exit 2; fi
  done
  echo "engine ready after ${t}s"
  nohup $X/venv/bin/python -u $X/src/tools/inkling2_server.py --model-dir $M --sock $SOCK --ctx $CTX --port $PORT --log $X/logs/serve-requests.jsonl > $X/logs/serve-front.log 2>&1 &
  echo $! > $X/serve/front.pid
  for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:$PORT/v1/models > /dev/null && break; sleep 1; done
  curl -s -m 2 http://127.0.0.1:$PORT/v1/models; echo
elif [ "$cmd" = stop ]; then
  [ -f $X/serve/front.pid ] && kill $(cat $X/serve/front.pid) 2>/dev/null; rm -f $X/serve/front.pid
  sleep 1
  if [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ]; then
    $X/venv/bin/python -c "import socket,sys; s=socket.socket(socket.AF_UNIX); s.settimeout(30); s.connect(sys.argv[1]); s.sendall(b'QUIT\n'); s.close()" $SOCK
    t=0
    while [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ]; do sleep 2; t=$((t+2)); [ $t -ge 300 ] && { echo "engine still running after QUIT (left running)"; exit 2; }; done
  fi
  docker logs $NAME > $X/logs/serve-engine-$(date +%m%d-%H%M).log 2>&1; tail -3 $X/logs/serve-engine-$(date +%m%d-%H%M).log
  echo "engine exit $(docker inspect -f '{{.State.ExitCode}}' $NAME 2>/dev/null)"; docker rm $NAME > /dev/null 2>&1
  for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q "renderD1" && echo "GPU HELD BY $(basename $p) $(cat $p/comm 2>/dev/null)"; done
  echo stopped
fi
