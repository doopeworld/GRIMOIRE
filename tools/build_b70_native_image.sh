#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
XPU_KERNELS=${XPU_KERNELS:-/mnt/cache/appdata/vllm-xpu-kernels}
IMAGE=${IMAGE:-grimoire:b70-native}

test -f "$XPU_KERNELS/csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2.hpp" || {
  echo "missing custom XPU kernel tree: $XPU_KERNELS" >&2
  exit 2
}

docker buildx build --load \
  --build-context "xpu-kernels=$XPU_KERNELS" \
  --file "$ROOT/docker/Dockerfile.b70-native" \
  --tag "$IMAGE" \
  "$ROOT"

docker run --rm --entrypoint /bin/sh "$IMAGE" -c '
  set -eu
  command -v python >/dev/null 2>&1 && exit 10 || true
  command -v python3 >/dev/null 2>&1 && exit 10 || true
  find / -xdev -type f \( -name "libtorch*" -o -name "libc10*" \) -print -quit | grep -q . && exit 11 || true
  find /opt /usr/local -type f \( -name "libur_adapter_cuda*" -o -name "libur_adapter_hip*" \) -print -quit | grep -q . && exit 11 || true
  ldd /usr/local/bin/grimoire | grep "not found" && exit 12 || true
  ldd /opt/grimoire/lib/libgrimoire_xe2_grouped.so | grep "not found" && exit 13 || true
  test -n "$(find /opt/intel/oneapi/compiler -name "libur_adapter_level_zero.so*" -print -quit)"
'

docker image inspect "$IMAGE" \
  --format 'built {{.RepoTags}}: {{.Size}} bytes'
