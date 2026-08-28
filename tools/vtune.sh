#!/bin/bash
# vLLM runner. Exposes ALL of /dev/dri because oneCCL enumerates the whole
# directory (it needs the card* nodes, not just renderD*), which is why the
# single-node b70run.sh fails with "could not open device directory".
# GPU selection therefore happens via ZE_AFFINITY_MASK instead.
set -u
cd /mnt/storage/isos/grimoire-fuse
NODE=$(bash tools/gpunode.sh "${GPU:-gpu0}") || exit $?
echo "GPU=${GPU:-gpu0} -> $NODE (ZE_AFFINITY_MASK=${MASK:-0})"
export GRIM_ENV="${GRIM_EXTRA_LD:-LD_LIBRARY_PATH}=/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:/opt/intel/oneapi/lib:/usr/local/lib
ZE_AFFINITY_MASK=${MASK:-0}
VLLM_TARGET_DEVICE=xpu
VLLM_ENABLE_V1_MULTIPROCESSING=0
${EXTRA_ENV:-}"
exec bash tools/vllmrun.sh "$NODE" "${LIM:-2400}" "$@"
