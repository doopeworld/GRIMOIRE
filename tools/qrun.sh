#!/bin/bash
# RETIRED 2026-09-26: this is the old plug-in launcher (vLLM bridges + torch
# library path) and its default node renderD129 is the iGPU now.  GRIMOIRE
# runs pure since 0283b7d -- use tools/tune.sh (resolves the B70 by PCI).
echo "$(basename "$0") is retired: use tools/tune.sh (pure GRIMOIRE, B70 by PCI)" >&2; exit 2
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
