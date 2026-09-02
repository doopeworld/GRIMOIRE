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

docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run -d --name "$CNAME" -w /grimoire --init --stop-timeout 300 \
  -p "${PORT}:${PORT}" \
  --device "/dev/dri/${NODE}:/dev/dri/${NODE}" \
  -v /mnt/storage/isos/grimoire-fuse:/grimoire \
  -v /mnt/storage/Models:/models \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e GRIMOIRE_W4A8=1 \
  -e GRIMOIRE_DEFER_MOE_GATHER=1 \
  -e GRIMOIRE_BF16_QKV=1 \
  -e GRIMOIRE_BF16_DN_QKV=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:/opt/intel/oneapi/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/usr/local/lib:/grimoire/src \
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
