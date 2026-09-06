#!/bin/bash
# llama-benchy pp4096/tg32 against grimoire-server, one config per call.
# EXTRA holds newline-separated K=V passed to the server.
set -u
PORT="${PORT:-8099}"
cd /mnt/storage/isos/grimoire-fuse
EXTRA="${EXTRA:-}" PORT="$PORT" bash tools/serve_qwen.sh >/tmp/serve_$$.log 2>&1 || {
  echo "server failed"; tail -12 /tmp/serve_$$.log; exit 1; }
cd /tmp
uvx llama-benchy --base-url "http://localhost:${PORT}/v1" \
  --model /models/Qwen3.8-27B-MXFP4-GRIMOIRE \
  --pp 4096 --tg 32 --runs "${RUNS:-3}" --latency-mode generation --format md 2>&1 \
  | grep -E "pp4096|tg32|Coherence|FAILED|model +\|"
