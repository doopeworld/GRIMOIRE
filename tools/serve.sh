#!/bin/bash
# grimoire-server launcher: OpenAI-compatible HTTP server, one image for every
# model -- only MODEL changes. Resolves the B70 by PCI address, never a
# hardcoded render node (they swap across reboots).
#
# Usage: serve.sh <MODEL_DIR> [PORT] [GPU]
#   serve.sh /models/Qwen3.8-27B-MXFP4-GRIMOIRE 8099 gpu0
set -u
MODEL="${1:?model dir, e.g. /models/Qwen3.8-27B-MXFP4-GRIMOIRE}"
PORT="${2:-8099}"
GPU="${3:-gpu0}"
IMAGE="${GRIM_IMAGE:-my-vllm-xpu:latest}"
CNAME="${CNAME:-grim-server}"

cd /mnt/storage/isos/grimoire-fuse
NODE=$(bash tools/gpunode.sh "$GPU") || { echo "cannot resolve $GPU" >&2; exit 2; }
echo "GPU=$GPU -> $NODE, model=$MODEL, port=$PORT"

if docker ps --format "{{.Names}}" | grep -q "^${CNAME}$"; then
  echo "stopping existing $CNAME (draining GPU work) ..."
  docker stop --time 120 "$CNAME" >/dev/null 2>&1 || true
fi
docker rm "$CNAME" >/dev/null 2>&1 || true
# Serving knobs pass through from the host ONLY IF SET there: `-e NAME` with
# no value copies the host's value and adds nothing when it is unset, so
# the default launch is unchanged.  Without these the batching, prefix-
# cache and speculation settings could not reach the container at all.
docker run -d --name "$CNAME" -w /grimoire --init --stop-timeout 300 \
  -p "${PORT}:${PORT}" \
  --device "/dev/dri/${NODE}:/dev/dri/${NODE}" \
  -v /mnt/storage/isos/grimoire-fuse/bin:/grimoire/bin:ro -v /mnt/storage/isos/grimoire-fuse/tools:/grimoire/tools:ro --tmpfs /opt/grimoire/lib \
  -v /mnt/storage/Models:/models \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e GRIMOIRE_W4A8=1 \
  -e GRIMOIRE_DEFER_MOE_GATHER=1 \
  -e GRIMOIRE_BF16_QKV=1 \
  -e GRIMOIRE_BF16_DN_QKV=1 \
  -e GRIMOIRE_SEQ_SLOTS \
  -e GRIMOIRE_MAX_BATCH \
  -e GRIMOIRE_PREFIX_CACHE \
  -e GRIMOIRE_PREFIX_SLOTS \
  -e GRIMOIRE_MTP \
  -e GRIMOIRE_MTP_K \
  -e GRIMOIRE_DFLASH_MODEL \
  -e GRIMOIRE_DFLASH_M \
  -e GRIMOIRE_SPEC_STATS \
  -e GRIMOIRE_DECODE_GRAPH \
  --entrypoint /grimoire/bin/grimoire-server \
  "$IMAGE" \
  --model "$MODEL" --proj mxfp4 --ctx 8192 --host 0.0.0.0 --port "$PORT"

echo "waiting for ready ..."
for i in $(seq 1 60); do
  sleep 3
  if curl -s "http://localhost:${PORT}/health" 2>/dev/null | grep -q ok; then
    echo "ready on :${PORT}"; exit 0
  fi
  docker ps --format '{{.Names}}' | grep -q "^${CNAME}$" || {
    echo "container exited during load:" >&2
    docker logs "$CNAME" 2>&1 | tail -8 >&2; exit 1; }
done
echo "did not become ready in 180s" >&2; exit 1
