#!/bin/bash
# run_served.sh ENGINE [TAG] -- Phase 1 served baseline: one engine's OpenAI endpoint on gpu0,
# measured with the SAME llama-benchy invocation for every engine (the comparison of record).
#
#   ENGINE = llamacpp-sycl | llamacpp-vulkan            Ornith Q4_K_M GGUF (same file)
#          | grimoire:<image>:<model dir>:<proj>        e.g. grimoire:grimoire-b70:v1.9.0:Ornith-1.5-35B-A3B:mxfp4
#          | vllm                                       Ornith GPTQ-Int4 (cross-format), tools/bench/vllm_conc4k.sh settings
#
# Fairness: llama-benchy --no-cache (unique prompts + cache_prompt=false) defeats llama-server's
# prompt cache and vLLM's automatic prefix caching (also disabled server-side); GRIMOIRE's prefix
# cache is off by default.  --exact-tg makes every engine decode exactly tg tokens.  One engine at a
# time; the container is stopped as soon as its run ends (CLAUDE.md).  Coherence: llama-benchy's own
# coherence check plus the server log are kept beside the numbers (rule 8).
set -u
ENGINE=${1:?engine}; TAG=${2:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUTROOT=/mnt/user/isos/grimoire-runs/gguf-baseline/served
B=/mnt/storage/isos/grimoire-runs/bench-1003
TOK=/mnt/storage/Models/Ornith-1.5-35B-A3B      # one tokenizer for every engine (same model family)
PORT=6950
PP="512 2048 4096 8192"; TG="32 128"; RUNS=${RUNS:-5}; CTX=16384

NODE=$(bash "$REPO/tools/gpunode.sh" gpu0) || exit 2
busy() {
  local c p
  c=$(for id in $(docker ps -q); do docker inspect --format '{{.Name}} {{.HostConfig.Privileged}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$id"; done \
      | awk -v d="/dev/dri/$NODE" '{for(i=3;i<=NF;i++) if($i==d) print $1} $2=="true"{print $1}')
  p=$(for f in /proc/[0-9]*/fd/*; do [ "$(readlink "$f" 2>/dev/null)" = "/dev/dri/$NODE" ] && echo "${f#/proc/}"; done | cut -d/ -f1 | sort -u | tr '\n' ' ')
  [ -n "$c$p" ] && { echo "GPU BUSY: containers=[$c] pids=[$p]"; return 0; }; return 1
}
curl -s -m 3 -o /dev/null http://127.0.0.1:8999/1661-0.txt || { echo "book server on :8999 is down (bench-1003/book)"; exit 2; }
busy && exit 3

NAME=""; MODEL_ID=""; KIND=""
wait_ready() {  # $1 = url that answers when ready, $2 = seconds
  for i in $(seq 1 $(( $2 / 3 ))); do
    sleep 3
    curl -s -m 3 "$1" 2>/dev/null | grep -qE '"ok"|"id"|ok' && { echo "ready after $((i*3))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "EXITED during load:"; docker logs "$NAME" 2>&1 | tail -20; return 1; }
  done
  echo "not ready in $2 s"; return 1
}
stop_server() {
  [ -z "$NAME" ] && return
  for i in $(seq 1 60); do [ "$(ss -Htn state established "( sport = :$PORT )" | wc -l)" -eq 0 ] && break; sleep 2; done
  docker logs "$NAME" > "$OUT/server.log" 2>&1
  docker stop --time 120 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
  echo "stopped $NAME"; NAME=""
}
trap stop_server EXIT

case $ENGINE in
  llamacpp-sycl|llamacpp-vulkan)
    BK=${ENGINE#llamacpp-}
    [ $BK = sycl ] && IMAGE=ghcr.io/ggml-org/llama.cpp:full-intel@sha256:c68de96b6d45634398b648b1ad9efb2f5cb0d1d7e95fa45e9657cd78610d427f \
                   || IMAGE=ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:267614252d76fa9e08599483f86f45d2472739ed557e90298ef1ac5ffdabd223
    OUT=$OUTROOT/$ENGINE${TAG:+-$TAG}; mkdir -p "$OUT"; NAME=gguf-srv-$BK; MODEL_ID=ornith-q4km
    docker rm -f $NAME >/dev/null 2>&1
    docker run -d --name $NAME --init --stop-timeout 300 --network host \
      --device /dev/dri/$NODE --group-add "$(stat -c %g /dev/dri/$NODE)" \
      -v /mnt/models/Ornith-1.5-35B-A3B-GGUF:/models:ro \
      $( [ $BK = sycl ] && echo "-e ONEAPI_DEVICE_SELECTOR=level_zero:0" ) \
      --entrypoint /app/llama-server "$IMAGE" \
      -m /models/Ornith-1.5-35B-Q4_K_M.gguf --alias $MODEL_ID -ngl 99 -fa on -c $CTX -b 2048 -ub 2048 \
      --parallel 1 --host 127.0.0.1 --port $PORT >/dev/null || exit 4
    echo "image: $IMAGE" > "$OUT/engine.txt"
    wait_ready http://127.0.0.1:$PORT/health 900 || exit 5 ;;
  grimoire:*)
    IFS=: read -r _ GIMG GTAG GMODEL GPROJ <<< "$ENGINE"; GIMG=$GIMG:$GTAG
    OUT=$OUTROOT/grimoire-${GTAG}-${GMODEL}-${GPROJ}${TAG:+-$TAG}; mkdir -p "$OUT"; NAME=gguf-srv-grim; MODEL_ID=/models/$GMODEL
    echo "image: $GIMG $(docker image inspect "$GIMG" --format '{{.Id}}')" > "$OUT/engine.txt"
    SRV_MODE=image SRV_IMAGE=$GIMG SRV_CTX=$CTX SRV_WAIT=1800 BENCH_OUT=$OUT \
    SRV_ENV="ZE_AFFINITY_MASK=0
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
SYCL_CACHE_PERSISTENT=1
SYCL_CACHE_DIR=/cache/sycl" \
      bash $B/srv2.sh up $NAME gpu0 $PORT "$GMODEL" "$GPROJ" | tee -a "$OUT/engine.txt" || { NAME=""; exit 5; } ;;
  vllm)
    OUT=$OUTROOT/vllm-gptq-int4${TAG:+-$TAG}; mkdir -p "$OUT"; NAME=gguf-srv-vllm; M=Ornith-1.5-35B-A3B-GPTQ-Int4
    docker rm -f $NAME >/dev/null 2>&1
    docker run -d --name $NAME --ipc=host --shm-size=10g --network host --init --stop-timeout 300 \
      --device /dev/dri/$NODE -v /mnt/storage/Models:/models \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e ZE_AFFINITY_MASK=0 \
      -v $B/vllm/qwen3_dflash.resolved.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py:ro \
      -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/intel/oneapi/ccl/latest/lib:/opt/intel/oneapi/mpi/latest/lib:/opt/intel/oneapi/tcm/1.5/lib:/opt/intel/oneapi/umf/1.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib:/usr/local/lib \
      my-vllm-xpu:latest /models/$M --dtype float16 --port $PORT --host 127.0.0.1 --max-model-len $CTX \
      --max-num-seqs 8 --kv-cache-dtype fp8 --max-num-batched-tokens 16384 --block-size 64 \
      --no-enable-prefix-caching --language-model-only --trust-remote-code --gpu-memory-utilization 0.90 >/dev/null || exit 4
    echo "image: my-vllm-xpu:latest $(docker image inspect my-vllm-xpu:latest --format '{{.Id}}') model: $M (cross-format: GPTQ-Int4 safetensors)" > "$OUT/engine.txt"
    wait_ready http://127.0.0.1:$PORT/v1/models 1500 || exit 5
    MODEL_ID=$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])') ;;
  *) echo "unknown engine $ENGINE"; exit 2 ;;
esac

{ echo "date: $(date -Is)"; echo "engine: $ENGINE"; echo "gpu0: $NODE";
  echo "link 00:01.0: $(cat /sys/bus/pci/devices/0000:00:01.0/current_link_speed) x$(cat /sys/bus/pci/devices/0000:00:01.0/current_link_width)";
  echo "llama-benchy 0.4.0: --pp $PP --tg $TG --runs $RUNS --concurrency 1 --no-cache --exact-tg"; } >> "$OUT/engine.txt"
timeout 5400 uvx llama-benchy@0.4.0 --base-url http://127.0.0.1:$PORT/v1 --model "$MODEL_ID" --tokenizer $TOK \
  --book-url http://127.0.0.1:8999/1661-0.txt --pp $PP --tg $TG --concurrency 1 --runs $RUNS \
  --no-cache --exact-tg --format md --save-result "$OUT/benchy.md" > "$OUT/benchy.log" 2>&1
echo "llama-benchy rc=$?  $(grep -h -i coherence "$OUT/benchy.log" | head -1)"
cat "$OUT/benchy.md" 2>/dev/null
stop_server
echo "SERVED $ENGINE DONE -> $OUT"
