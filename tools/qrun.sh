#!/bin/bash
# Canonical GRIMOIRE run: vLLM image (has torch + xe2 attn kernels) + all bridges.
set -u
NODE="${NODE:-renderD129}"
LIM="${LIM:-900}"
NAME="${1:?run name}"; shift

export GRIM_ENV="LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:/opt/intel/oneapi/lib:/usr/local/lib
GRIMOIRE_XE2_GROUPED_BRIDGE=/grimoire/src/libgrimoire_xe2_grouped.so
GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_bridge.so
GRIMOIRE_XE2_GDN_RAW_BRIDGE=/grimoire/src/libgrimoire_xe2_gdn_raw.so
GRIMOIRE_ONEDNN_BRIDGE=/grimoire/src/libgrimoire_onednn.so
GRIMOIRE_DEFER_MOE_GATHER=1
GRIMOIRE_BF16_QKV=1
GRIMOIRE_BF16_DN_QKV=1
${EXTRA_ENV:-}"

cd /mnt/storage/isos/grimoire-fuse
bash tools/b70run.sh "$NODE" "$LIM" "$NAME" "$@"
