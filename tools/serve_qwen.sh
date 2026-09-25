#!/bin/bash
# grimoire-server for Qwen3.8-27B with the VERIFIED Qwen MTP recipe.
# tools/serve.sh hardcodes GRIMOIRE_DEFER_MOE_GATHER / BF16_QKV / BF16_DN_QKV,
# which are ORNITH flags -- on Qwen they cut MTP decode from 49.8 to 18.3.
# See QWEN-RECIPE.md.  EXTRA holds newline-separated K=V for A/B toggles.
set -u
MODEL="${MODEL:-/models/Qwen3.8-27B-MXFP4-GRIMOIRE}"
PORT="${PORT:-8099}"
GPU="${GPU:-gpu0}"
IMAGE="${GRIM_IMAGE:-my-vllm-xpu:latest}"
CNAME="${CNAME:-grim-server}"
cd /mnt/storage/isos/grimoire-fuse
NODE=$(bash tools/gpunode.sh "$GPU") || exit 2
echo "GPU=$GPU -> $NODE, model=$MODEL, port=$PORT"
# NEVER `docker rm -f` a container that may hold the GPU: that is an immediate
# SIGKILL, it tears the DRM fd down mid-submission, the batch never signals its
# fence and the card takes a coredump (see the header of tools/b70run.sh).
# Stop it politely first and only force-remove a container that is already dead.
if docker ps --format "{{.Names}}" | grep -q "^${CNAME}$"; then
  echo "stopping existing $CNAME (draining GPU work) ..."
  docker stop --time 120 "$CNAME" >/dev/null 2>&1 || true
fi
docker rm "$CNAME" >/dev/null 2>&1 || true
ENVARGS=()
while IFS= read -r kv; do [ -n "$kv" ] && ENVARGS+=(-e "$kv"); done <<< "GRIMOIRE_W4A8=1
GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_EXACT_VERIFY=1
GRIMOIRE_MTP_DRAFT_VOCAB=131072
${EXTRA:-}"
docker run -d --name "$CNAME" -w /grimoire --init --stop-timeout 300 \
  -p "${PORT}:${PORT}" \
  --device "/dev/dri/${NODE}:/dev/dri/${NODE}" \
  -v /mnt/storage/isos/grimoire-fuse/bin:/grimoire/bin:ro -v /mnt/storage/isos/grimoire-fuse/tools:/grimoire/tools:ro --tmpfs /opt/grimoire/lib \
  -v /mnt/storage/Models:/models \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  "${ENVARGS[@]}" \
  --entrypoint /grimoire/bin/grimoire-server \
  "$IMAGE" \
  --model "$MODEL" --proj mxfp4 --ctx 8192 --host 0.0.0.0 --port "$PORT" >/dev/null
echo "waiting for ready ..."
for i in $(seq 1 80); do
  sleep 3
  curl -s "http://localhost:${PORT}/health" 2>/dev/null | grep -q ok && { echo "ready on :${PORT}"; exit 0; }
  docker ps --format "{{.Names}}" | grep -q "^${CNAME}$" || { echo "container exited:"; docker logs "$CNAME" 2>&1 | tail -12; exit 1; }
done
echo "not ready in 240s"; docker logs "$CNAME" 2>&1 | tail -12; exit 1
