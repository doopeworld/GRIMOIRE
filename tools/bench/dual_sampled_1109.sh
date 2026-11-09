#!/bin/bash
# 11-09: close controller/benchmark lock descriptors in the detached book server.
# dual.sh -- one GRIMOIRE configuration on the B70s: start -> test -> stop, with link guards.
# Written 2026-10-22, after gpu1 moved to the redriver M.2-OCuLink adapter (root port 00:06.2).
#
#   dual.sh NAME MODE MODEL PROJ CTX "ENV=.. ENV=.." TESTS [llama-benchy args...]
#     MODE   TP | PP   both B70s through the image's "multi" entrypoint, rank 0 = gpu0
#            T0        (10-29) "multi TP" with BOTH ranks on gpu0 (GRIMOIRE_MULTI_DEVICES=0,0):
#                      the two-card code paths without touching gpu1 -- correctness, not speed
#            G1 | G0   one card only (G1 = the second B70)
#            B5        (11-04) the Arc B580 alone (Battlemage G21, the B60's chip); holds b580.lock only
#     MODEL  path inside the container: /models/<dir> = /mnt/storage/Models,
#            /models2/<dir> = /mnt/models (Qwen3.8 checkpoints since 10-22)
#     TESTS  comma list of ctest, gen, code, prose, benchy (benchy gets the extra args),
#            pleready (wait for the PLE table's copy into RAM),
#            concN (N coding answers at once, default 4, unit tests, two rounds; conc_code.py)
#
# Holds /mnt/user/isos/grimoire-runs/gpu0.lock then gpu1.lock for the whole run (the other
# agents use the same locks).  Server = host bin/ of grimoire-fuse in grimoire-b70:latest;
# IMG=<image> HOSTBIN=0 runs an image's OWN binaries instead (a release image as shipped).
# Link guard, sampled every 2 s from container start to stop:
#   STOP only when a card fails: any non-fatal/fatal AER on either card's path, any xe error line in
#        syslog, or gpu1's config space reading ffffffff. Corrected errors and link retrains are logged,
#        never a reason to stop (11-04, Ian: an error count does not mean the card is not working).  (10-22: the first TP
#        prefill under load logged 1 RxErr + 1 BadTLP there, corrected by replay, link stayed up.)  On STOP the test client is killed; the server
#        finishes its in-flight work and is stopped gracefully -- never killed.
#   LOG  the host->card RxErr count at gpu1's switch port (~5/s idle on this adapter) and the
#        root port's replay timeouts, every 10 s, in logs/NAME.link.
set -u
NAME=$1 MODE=$2 MODEL=$3 PROJ=$4 CTX=$5 EXTRA=$6 TESTS=$7; shift 7
[ "$MODE" = B5 ] && { echo "B5 refused: the Arc B580 is Ian's ComfyUI card (11-05 17:17), never used for GRIMOIRE"; exit 2; }
B=/mnt/storage/isos/grimoire-runs/bench-1022; B3=/mnt/storage/isos/grimoire-runs/bench-1003
R=${GRIM_TREE:-/mnt/storage/isos/grimoire-fuse}; PORT=${PORT:-6899}; LOAD_MAX=${LOAD_MAX:-900}
L=$B/logs/$NAME; mkdir -p $B/logs; rm -f $L.STOP $L.DONE $L.client $L.link
export BB_PROGRESS="$L.progress.jsonl"
case $MODEL in /models2/*) HOSTM=/mnt/models/${MODEL#/models2/} ;; /models/*) HOSTM=/mnt/storage/Models/${MODEL#/models/} ;;
  *) echo "MODEL must be /models/... or /models2/..."; exit 2 ;; esac
[ -d "$HOSTM" ] || [ -f "$HOSTM" ] || { echo "no checkpoint at $HOSTM"; exit 2; }   # a .gguf file too (10-30)

exec 7>/mnt/user/isos/grimoire-runs/b580.lock 8>/mnt/user/isos/grimoire-runs/gpu0.lock 9>/mnt/user/isos/grimoire-runs/gpu1.lock
# a one-card run holds only its own card's lock (10-27); B5 = the Arc B580 alone (11-04)
if [ "$MODE" = B5 ]; then flock -n 7 || { echo "b580.lock is held by another job -- not starting"; exit 9; }
else
[ "$MODE" = G1 ] || flock -n 8 || { echo "gpu0.lock is held by another job -- not starting"; exit 9; }
[ "$MODE" = G0 ] || [ "$MODE" = T0 ] || flock -n 9 || { echo "gpu1.lock is held by another job -- not starting"; exit 9; }
fi

N0=$($R/tools/gpunode.sh gpu0 2>/dev/null) && N1=$($R/tools/gpunode.sh gpu1 2>/dev/null) ||
  { echo "cannot resolve both B70s"; exit 9; }
if [ "$MODE" = B5 ]; then   # the "gpu1" slot (its link checks, its render node) becomes the Arc B580 (8086:e20b)
  PB=$(for d in /sys/bus/pci/devices/*; do [ "$(cat $d/vendor)" = 0x8086 ] && [ "$(cat $d/device)" = 0xe20b ] && basename $d; done | head -1)
  [ -n "$PB" ] && N1=$($R/tools/gpunode.sh "$PB" 2>/dev/null) || { echo "cannot resolve the B580"; exit 9; }
fi
P0=$(basename "$(readlink -f /sys/class/drm/$N0/device)"); P1=$(basename "$(readlink -f /sys/class/drm/$N1/device)")
chain() { readlink -f /sys/bus/pci/devices/$1 | tr / '\n' | grep -E '^[0-9a-f]{4}:'; }
PORTS="$(chain $P0) $(chain $P1)"; RP1=$(chain $P1 | head -1); SW1=$(chain $P1 | sed -n 2p)
cnt() { awk -v k=$2 '$1==k{print $2}' /sys/bus/pci/devices/$1/aer_dev_$3 2>/dev/null; }
uncorr() { local s=0 p v; for p in $PORTS; do for v in $(cnt $p TOTAL_ERR_NONFATAL nonfatal) $(cnt $p TOTAL_ERR_FATAL fatal); do s=$((s+v)); done; done; echo $s; }
rp_rx() { echo $(( $(cnt $RP1 RxErr correctable) + $(cnt $RP1 BadTLP correctable) + $(cnt $RP1 BadDLLP correctable) )); }
linkstat() { echo "gpu1 $P1: switch $SW1 RxErr=$(cnt $SW1 RxErr correctable) BadTLP=$(cnt $SW1 BadTLP correctable) BadDLLP=$(cnt $SW1 BadDLLP correctable);" \
  "root $RP1 rx=$(rp_rx) Timeout=$(cnt $RP1 Timeout correctable) Rollover=$(cnt $RP1 Rollover correctable);" \
  "uncorrectable(all ports)=$(uncorr); gpu0 path correctable=$(for p in $(chain $P0); do cnt $p TOTAL_ERR_COR correctable; done | awk '{s+=$1} END{print s+0}')"; }

watch_link() {
  local u0=$(uncorr) r0=$(rp_rx) s0=$(cnt $SW1 RxErr correctable) t0=$(cnt $RP1 Timeout correctable)
  local sl=$(wc -l < /var/log/syslog) n=0 s_last=$s0 r_win=$(rp_rx) why u r cfg xe bw=""
  while [ ! -e $L.DONE ]; do
    sleep 2; n=$((n+1)); why=""
    u=$(uncorr); r=$(rp_rx); cfg=$(od -An -tx4 -N4 /sys/bus/pci/devices/$P1/config 2>/dev/null | tr -d ' ')
    xe=$(tail -n +$((sl+1)) /var/log/syslog 2>/dev/null | grep -E "xe 0000:(${P0#0000:}|${P1#0000:})" |
         grep -viE "severity=correct|corrected" |
         grep -iE "unreliable|timedout|reset|wedged|lost|fail|fault|error|guc" | head -1 | cut -c1-200)
    [ "$u" != "$u0" ] && why="uncorrectable AER on a GPU path ($u0 -> $u)"
    # 11-04, Ian: corrected link errors are repaired by the link and the card keeps working, so they are
    # only logged (every 10 s in $L.link). A run stops only when a card actually fails.
    if [ "$MODE" != G0 ] && [ "$MODE" != T0 ]; then
    if [ -z "$bw" ] && lspci -vv -s ${RP1#0000:} 2>/dev/null | grep -q "BWMgmt+"; then
      bw=seen; echo "$(date +%T) NOTE gpu1 root port reports a hardware link retrain (BWMgmt+)" >> $L.link; fi
    [ "$cfg" = ffffffff ] || [ -z "$cfg" ] && why="gpu1 config space unreadable ($cfg): the card is gone"
    fi
    [ -n "$xe" ] && why="xe: $xe"
    if [ -n "$why" ] && [ ! -e $L.STOP ]; then
      echo "$(date +%T) STOP: $why" > $L.STOP; cat $L.STOP
      [ -e $L.client ] && kill $(cat $L.client) 2>/dev/null
    fi
    if [ $((n % 5)) = 0 ]; then local s=$(cnt $SW1 RxErr correctable)
      echo "$(date +%T) switch RxErr +$((s - s_last))/10s (run +$((s - s0)))  root Timeout +$(( $(cnt $RP1 Timeout correctable) - t0 ))  root rx $r (+$((r - r_win)))  unc $u" >> $L.link
      s_last=$s; r_win=$r; fi
  done; }

guard() {
  local c image devices privileged
  for c in $(docker ps -q); do
    image=$(docker inspect -f '{{.Config.Image}}' "$c")
    devices=$(docker inspect -f '{{json .HostConfig.Devices}}' "$c")
    privileged=$(docker inspect -f '{{.HostConfig.Privileged}}' "$c")
    if [ "$image" != grimoire-dev:latest ] || { [ "$devices" != null ] && [ "$devices" != '[]' ]; } || [ "$privileged" != false ]; then
      echo "a non-builder container is running -- refusing: $c ($image)"; return 1
    fi
  done
  local p; for p in /proc/[0-9]*; do
    ls -l $p/fd 2>/dev/null | grep -q "/dev/dri/renderD" && { echo "a GPU is held by $(cat $p/comm) ($p) -- refusing"; return 1; }
  done
  [ "$(uncorr)" = 0 ] || { echo "uncorrectable link errors already logged -- refusing"; return 1; }
  return 0; }

client() {  # client TAG cmd... : run one test client; the link watcher may kill it
  local tag=$1; shift; [ -e $L.STOP ] && return 1
  # CLIENT_CPUS=14,15 (10-28, opt-in): the client (llama-benchy is Python on this host) off the
  # cores the engine pins its CPU-expert pools to
  [ -n "${CLIENT_CPUS:-}" ] && set -- taskset -c "$CLIENT_CPUS" "$@"
  "$@" > >(sed -u "s/^/   [$tag] /" | tee -a $L.out) 2>&1 & local p=$!
  echo $p > $L.client; wait $p; local rc=$?; rm -f $L.client; return $rc; }

ple_ready() {  # Flash-Next: the PLE table's copy into RAM is done (else rows come from disk)
  local i; for i in $(seq 1 150); do
    docker logs $NAME 2>&1 | grep -q "PLE table in memory" && { echo "   PLE table in memory ($((i * 2)) s after the tests began)"; return 0; }
    docker logs $NAME 2>&1 | grep -q "PLE table: copying" || { echo "   no PLE copy in this run"; return 0; }
    sleep 2
  done; echo "   PLE copy still running after 300 s"; }

started=0
finish() {
  # A failed GPU must not enter Docker's finite stop timer: after that
  # timer Docker sends SIGKILL even if the driver cannot drain the work.
  if [ $started = 1 ] && [ -e "$L.STOP" ]; then
    docker logs "$NAME" > "$L.server.log" 2>&1
    touch "$L.DONE"
    echo "GPU GUARD: $NAME LEFT IN PLACE; no stop timeout or automatic GPU restart"
    return
  fi
  if [ $started = 1 ] && docker ps --format '{{.Names}}' | grep -qx $NAME; then
    if curl -s -m 5 http://127.0.0.1:$PORT/health | grep -q ok; then
      docker logs $NAME > $L.server.log 2>&1; docker stop --time 300 $NAME > /dev/null; docker rm $NAME > /dev/null 2>&1
      echo "   server stopped gracefully"
    else
      docker logs $NAME > $L.server.log 2>&1
      echo "   !! $NAME LEFT RUNNING, NOT HEALTHY (mid-load or a card is gone) -- check by hand, do NOT kill it"
    fi
  fi
  touch $L.DONE; sleep 2
  echo "   link after: $(linkstat)"; [ -e $L.STOP ] && echo "   !! $(cat $L.STOP)"
  [ -e $L.link ] && echo "   switch RxErr during run: $(tail -1 $L.link | grep -oE 'run \+[0-9]+')  over $(( $(wc -l < $L.link) * 10 )) s"; }
trap finish EXIT

guard || exit 9
case $MODE in
  TP|PP) DEVS=(--device /dev/dri/$N0 --device /dev/dri/$N1); ENTRY=(multi $MODE); AFF=0,1 ;;
  G0)    DEVS=(--device /dev/dri/$N0); ENTRY=(server); AFF=0 ;;
  G1|B5) DEVS=(--device /dev/dri/$N1); ENTRY=(server); AFF=0 ;;
  T0)    DEVS=(--device /dev/dri/$N0); ENTRY=(multi TP); AFF=0 ;;
  *) echo "MODE must be TP, PP, T0, G0, G1 or B5"; exit 2 ;;
esac
envs=(-e ONEAPI_DEVICE_SELECTOR=level_zero:gpu -e ZE_AFFINITY_MASK=$AFF -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
      -e SYCL_CACHE_PERSISTENT=1 -e SYCL_CACHE_DIR=/cache/sycl)
case $MODE in TP|PP) envs+=(-e GRIMOIRE_MULTI_GPUS=2) ;; T0) envs+=(-e GRIMOIRE_MULTI_GPUS=2 -e GRIMOIRE_MULTI_DEVICES=0,0) ;; esac
for kv in $EXTRA; do envs+=(-e "$kv"); done
echo "== $NAME  $MODE  $MODEL --proj $PROJ --ctx $CTX  [$EXTRA]  gpu0=$P0/$N0 gpu1=$P1/$N1"
echo "   link before: $(linkstat)"
watch_link &
docker rm -f $NAME > /dev/null 2>&1
IMG=${IMG:-grimoire-b70:latest}
if [ "${HOSTBIN:-1}" = 1 ]; then
  BINM=(-v $R/bin:/grimoire/bin:ro -v $R/tools:/grimoire/tools:ro --tmpfs /opt/grimoire/lib
        -v $R/docker-entrypoint.sh:/usr/local/bin/docker-entrypoint.sh:ro)
else BINM=(); fi
echo "   image $IMG, $([ "${HOSTBIN:-1}" = 1 ] && echo "host bin/" || echo "the image's own binaries")"
docker run -d --name $NAME --init --stop-timeout 300 --network host --ipc=host "${DEVS[@]}" \
  -v /dev/dri/by-path:/dev/dri/by-path:ro ${BINM[@]+"${BINM[@]}"} \
  -v /mnt/storage/Models:/models -v /mnt/models:/models2 -v /mnt/storage/isos/grimoire-cache:/cache "${envs[@]}" \
  $IMG "${ENTRY[@]}" --model $MODEL --proj $PROJ --ctx $CTX --host 0.0.0.0 --port $PORT > /dev/null || exit 9
started=1; t0=$(date +%s); ok=0
while [ $(( $(date +%s) - t0 )) -lt $LOAD_MAX ]; do
  sleep 3
  curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -qx $NAME || break
  [ -e $L.STOP ] && break
done
echo "   ready=$ok after $(( $(date +%s) - t0 )) s"
docker logs $NAME 2>&1 | grep -E "Megatron|all-reduce|parallel|speculat|refus|governor|upload|split|layers|VRAM|tiered|experts|error|fail|warn" |
  grep -v "^\s*$" | head -14 | cut -c1-190 | sed 's/^/   | /'
[ $ok = 1 ] || exit 8
BOOK="--book-url http://127.0.0.1:8999/1661-0.txt"
if [[ ",$TESTS," == *,benchy,* ]] && ! curl -s -m 3 -o /dev/null http://127.0.0.1:8999/1661-0.txt; then
  # (its own fds first: a backgrounded `a && b &` list forks a bash that keeps our stdout open)
  (cd $B3/book || exit; exec 6>&- 7>&- 8>&- 9>&- > /dev/null 2>&1 < /dev/null
   nohup setsid python3 -m http.server 8999 --bind 127.0.0.1 > $B/logs/book-server.log 2>&1 &); sleep 2
fi
IFS=, read -r -a tl <<< "$TESTS"
for t in "${tl[@]}"; do
  [ -e $L.STOP ] && break
  case $t in
    ctest)  client ctest timeout 600 python3 $B3/ctest.py $PORT ;;
    conc*)  cu=${t#conc}; client $t timeout 1500 python3 $B/conc_code.py $PORT ${cu:-4} $L.$t.json ;;
    seq*)   cu=${t#seq}; client $t timeout 1500 python3 $B/seq_code_1102.py $PORT ${cu:-4} $L.$t.json ;;
    ragged) client ragged timeout 900 python3 $B/mtp_ragged_1031.py $PORT $L.ragged.json ;;
    sampledapi) client sampledapi timeout 900 python3 $B/sampled-api1109.py $PORT $L.sampledapi.json || exit 1 ;;
    sampledcompare) client sampledcompare python3 $B/compare_sampled1109.py $B/logs/s1109-matched-swift-plain.sampledapi.json $L.sampledapi.json || exit 1 ;;
    betterbench) client betterbench timeout 7200 uv run --offline --no-project --with numpy python -u $B/observe_betterbench1109.py run --endpoint http://127.0.0.1:$PORT/v1 --model $MODEL --config $B/betterbench-match1109.json --max-model-len 262144 --out $L.betterbench.json --no-update-check || exit 1 ;;
    api1109) client api1109 timeout 600 python3 $B/api_checks_1109.py $PORT $L.api.json $MODEL ;;
    pleready) ple_ready ;;
    gen)    client gen timeout 900 python3 $B3/gen_save.py $PORT $L.gen.json ;;
    code)   client code timeout 1200 python3 $B3/coding_probe.py --base http://127.0.0.1:$PORT --model m \
              --lengths ${CODE_LENGTHS:-2000} --repeats ${CODE_REPEATS:-1} --out $L.code.json ;;
    prose)  client prose timeout 1200 python3 $B3/coding_probe.py --base http://127.0.0.1:$PORT --model m --task prose \
              --lengths ${CODE_LENGTHS:-2000} --repeats ${CODE_REPEATS:-1} --out $L.prose.json ;;
    benchy) client benchy timeout 1800 uvx llama-benchy@0.4.0 --base-url http://localhost:$PORT/v1 --model $MODEL \
              --tokenizer ${TOKDIR:-$HOSTM} $BOOK "$@" --format md --save-result $L.benchy.md
            [ -e $L.benchy.md ] && grep -E "^\| " $L.benchy.md | awk -F'|' '{printf "     %-16s %-22s %s\n", $3, $4, $5}' ;;
  esac
done
echo "   tests done"
