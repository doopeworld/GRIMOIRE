#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-dflash}
IMAGE=${IMAGE:-grimoire:b70-fusion}
MODEL_ROOT=${MODEL_ROOT:-/mnt/storage/Models}
TARGET=${TARGET:-/models/Muse-Glimmer-30B-INT4-W4A16}
ASSISTANT=${ASSISTANT:-/models/Muse-Glimmer-assistant}
PORT=${PORT:-3552}
NAME=${NAME:-grimoire-b70-fusion-${MODE}}

case "$MODE" in
  prefill)
    SPEC_ARGS=()
    COMPACT=0
    ;;
  dflash)
    SPEC_ARGS=(--speculative-config
      "{\"method\":\"dflash\",\"model\":\"$ASSISTANT\",\"num_speculative_tokens\":15}")
    COMPACT=1
    ;;
  *)
    echo "usage: $0 {prefill|dflash}" >&2
    exit 2
    ;;
esac

if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  echo "container already exists: $NAME" >&2
  exit 3
fi

exec docker run -d --name "$NAME" \
  --ipc=host --privileged --security-opt label=disable --shm-size=10g \
  --device /dev/dri:/dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v "$MODEL_ROOT:/models" \
  --network host \
  -e ZE_AFFINITY_MASK=0 \
  -e "VLLM_XPU_DFLASH_COMPACT_SINGLE=$COMPACT" \
  "$IMAGE" \
  "$TARGET" \
  --dtype float16 --port "$PORT" --host 0.0.0.0 \
  --max-model-len 12288 --gpu-memory-utilization 0.95 \
  --max-num-seqs 16 --kv-cache-dtype auto \
  --max-num-batched-tokens 16384 --block-size 64 \
  --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  "${SPEC_ARGS[@]}"
