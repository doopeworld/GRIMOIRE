#!/bin/bash
# Validate image $1 the way the GRIMOIRE-DUAL-INKLING2 template runs it: /mnt/storage/Models at /models,
# INK_MODEL=/models/Inkling-2-Mini-Uncensored (the DFlash2 drafter next to it is found automatically), INK_CTX 262144,
# two B70s (03:00.0 + 07:00.0, never the B580).  Named test container, removed after.  Greedy coherence, BetterBench
# quick (decode, prefill, concurrency), then docker stop must end the engine cleanly (exit 0, "serve: quit").
set -u
IMG=$1; TAG=${2:-imgval2}
X=/mnt/storage/isos/grimoire-runs/ink-1123
M=/mnt/storage/Models
NAME=ink-1123-imgval
cd $X
if docker ps -a --format "{{.Names}}" | grep -q "^ink-1123-"; then echo "an ink-1123 container exists"; exit 1; fi
N0=$(readlink -f /dev/dri/by-path/pci-0000:03:00.0-render); N1=$(readlink -f /dev/dri/by-path/pci-0000:07:00.0-render)
docker run -d --name $NAME --stop-timeout 60 --device $N0:/dev/dri/renderD128 --device $N1:/dev/dri/renderD129 \
  -v $M:/models:ro -e INK_MODEL=/models/Inkling-2-Mini-Uncensored -e INK_CTX=${INK_CTX:-262144} $(env | grep -oE "^INK_[A-Z0-9_]+=[^ ]*" | grep -v "^INK_CTX=" | sed "s/^/-e /" | tr "\n" " ") \
  -p 127.0.0.1:6911:6911 --entrypoint grimoire-inkling $IMG > /dev/null || exit 1
t=0
until curl -s -m 2 http://127.0.0.1:6911/v1/models > /dev/null; do
  sleep 3; t=$((t+3))
  [ "$(docker inspect -f '{{.State.Running}}' $NAME)" = true ] || { echo "container exited"; docker logs $NAME 2>&1 | tail -20; docker rm $NAME > /dev/null; exit 1; }
  [ $t -ge 900 ] && { echo "not ready after 15 min (left running)"; exit 2; }
done
echo "ready after ${t}s"
docker logs $NAME 2>&1 | grep -E "dflash2|mtp drafter|device total" | head -4
venv/bin/python greedy_check.py logs/greedy-$TAG.json
python3 - "$TAG" <<'PY'
import json, sys
a = json.load(open("logs/greedy-%s.json" % sys.argv[1])); b = json.load(open("logs/greedy-ag.json"))
for x, y in zip(a, b):
    sa = x["reasoning"] + "|" + x["content"]; sb = y["reasoning"] + "|" + y["content"]
    n = 0
    while n < min(len(sa), len(sb)) and sa[n] == sb[n]: n += 1
    print("greedy vs the 11-24 host-binary run (DSpark): same leading chars %d of %d / %d" % (n, len(sa), len(sb)))
    print("  answer head: %r" % (x["content"][:160]))
PY
cd /mnt/storage/isos/grimoire-runs/bench-1022/betterbench1109 && PYTHONPATH=. taskset -c 14,15 $X/venv/bin/python -u -m betterbench.cli run \
  --endpoint http://127.0.0.1:6911/v1 --model inkling-2-mini --config ../betterbench-vllm1111.json --max-model-len 32768 \
  --quick --name ink-$TAG --note image=$IMG --note cards=2 --out $X/logs/bb-$TAG.json --no-update-check > $X/logs/bb-$TAG.out 2>&1 < /dev/null
echo "betterbench exit $?"
cd $X
grep -E "^\*\*Combined|^\| (1|2|4|8) \||^\| (2000|8000) " logs/bb-$TAG.out
docker stop $NAME > /dev/null
echo "container exit $(docker inspect -f '{{.State.ExitCode}}' $NAME)"
docker logs $NAME 2>&1 | grep -E "serve: quit|WATCHDOG|rror" | tail -3
docker logs $NAME > logs/$TAG-container.log 2>&1
docker rm $NAME > /dev/null
for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -qE "renderD129|renderD130" && echo "GPU node held by $(basename $p) $(cat $p/comm)"; done
dmesg | tail -30 | grep -iE "xe 0000:0[37]:00.0.*(reset|timeout|wedged|fault)"
echo VALIDATE_DONE
