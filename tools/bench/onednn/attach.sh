#!/bin/bash
# attach.sh NAME MODE TESTS [llama-benchy args...] -- run dual.sh's tests on a server dual.sh left
# loading (its guard stopped the run during load), then stop it gracefully.  MODE G0 | G1 | TP | PP
# decides the locks.  10-27.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B3=/mnt/storage/isos/grimoire-runs/bench-1003
NAME=$1 MODE=$2 TESTS=$3; shift 3
L=logs/$NAME; PORT=6899
F=$(docker inspect $NAME --format '{{range .Args}}{{.}} {{end}}' | grep -oE "\-\-model [^ ]+" | cut -d" " -f2)
case $F in /models2/*) HOSTM=/mnt/models/${F#/models2/} ;; *) HOSTM=/mnt/storage/Models/${F#/models/} ;; esac
exec 8>/mnt/user/isos/grimoire-runs/gpu0.lock 9>/mnt/user/isos/grimoire-runs/gpu1.lock
[ "$MODE" = G1 ] || flock -n 8 || { echo "gpu0.lock is held by another job"; exit 9; }
[ "$MODE" = G0 ] || flock -n 9 || { echo "gpu1.lock is held by another job"; exit 9; }
t0=$(date +%s); ok=0
while [ $(( $(date +%s) - t0 )) -lt 1500 ]; do
  curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "container gone"; break; }
  sleep 5
done
echo "== $NAME ($F) ready=$ok after $(( $(date +%s) - t0 )) s"
if [ $ok = 1 ]; then
  IFS=, read -r -a tl <<< "$TESTS"
  for t in "${tl[@]}"; do
    case $t in
      gen)    timeout 900 python3 $B3/gen_save.py $PORT $L.gen.json 2>&1 | sed 's/^/   [gen] /' ;;
      code)   timeout 1200 python3 $B3/coding_probe.py --base http://127.0.0.1:$PORT --model m --lengths 2000 \
                --repeats 1 --out $L.code.json 2>&1 | sed 's/^/   [code] /' ;;
      benchy)
        if ! curl -s -m 3 -o /dev/null http://127.0.0.1:8999/1661-0.txt; then
          (cd $B3/book && exec 8>&- 9>&- && nohup setsid python3 -m http.server 8999 --bind 127.0.0.1 > /mnt/user/isos/grimoire-runs/bench-1022/logs/book-server.log 2>&1 < /dev/null &); sleep 2
        fi
        timeout 1800 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model $F --tokenizer $HOSTM \
          --book-url http://127.0.0.1:8999/1661-0.txt "$@" --format md --save-result $L.benchy.md > $L.benchy.out 2>&1
        [ -e $L.benchy.md ] && grep -E "^\| " $L.benchy.md | awk -F'|' '{printf "     %-16s %-22s %s\n", $3, $4, $5}' ;;
    esac
  done
fi
# stop only a healthy server (a loading one must finish first)
if curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q ok; then
  docker logs $NAME > $L.server.log 2>&1; docker stop --time 300 $NAME > /dev/null; docker rm $NAME > /dev/null 2>&1
  echo "server stopped gracefully"
else
  echo "!! $NAME not healthy -- left running, check by hand"
fi
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo ATTACH_DONE
