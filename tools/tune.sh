#!/bin/bash
# GRIMOIRE runner. GPU={gpu0|gpu1} resolved by PCI addr, never a hardcoded node.
set -u
cd /mnt/storage/isos/grimoire-fuse
NODE=$(bash tools/gpunode.sh "${GPU:-gpu0}") || exit $?
echo "GPU=${GPU:-gpu0} -> $NODE"
export GRIM_ENV="LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:/opt/intel/oneapi/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/usr/local/lib:/grimoire/src
ONEAPI_DEVICE_SELECTOR=level_zero:gpu
GRIMOIRE_XE2_GROUPED_BRIDGE=/grimoire/src/libgrimoire_xe2_grouped.so
GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_bridge.so
GRIMOIRE_XE2_GDN_RAW_BRIDGE=/grimoire/src/libgrimoire_xe2_gdn_raw.so
GRIMOIRE_ONEDNN_BRIDGE=/grimoire/src/libgrimoire_onednn.so
${EXTRA_ENV:-}"
exec bash tools/b70run.sh "$NODE" "${LIM:-2400}" "$@"
