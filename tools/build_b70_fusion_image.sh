#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_IMAGE=${BASE_IMAGE:-my-vllm-xpu:fusion-runtime}
IMAGE=${IMAGE:-grimoire:b70-fusion}

docker build \
  --file "$ROOT/docker/Dockerfile.b70-fusion" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --tag "$IMAGE" \
  "$ROOT"

docker run --rm --entrypoint python3 "$IMAGE" -c '
import py_compile
import torch
import vllm
import vllm_xpu_kernels
py_compile.compile(
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
    doraise=True,
)
print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("xpu kernels", vllm_xpu_kernels.__file__)
'

docker image inspect "$IMAGE" --format '{{.RepoTags}} {{.Size}} bytes'
