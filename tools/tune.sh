#!/bin/bash
# GRIMOIRE runner. GPU={gpu0|gpu1} resolved by PCI addr, never a hardcoded node.
set -u
cd /mnt/storage/isos/grimoire-fuse
NODE=$(bash tools/gpunode.sh "${GPU:-gpu0}") || exit $?
echo "GPU=${GPU:-gpu0} -> $NODE"
export GRIM_ENV="ONEAPI_DEVICE_SELECTOR=level_zero:gpu
${EXTRA_ENV:-}"
exec bash tools/b70run.sh "$NODE" "${LIM:-2400}" "$@"
