#!/bin/bash
# Minimal run: bridges + library path only. NO model-specific tuning flags.
set -u
export GRIM_ENV="LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:/opt/intel/oneapi/lib:/usr/local/lib
GRIMOIRE_XE2_GROUPED_BRIDGE=/grimoire/src/libgrimoire_xe2_grouped.so
GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_bridge.so
GRIMOIRE_XE2_GDN_RAW_BRIDGE=/grimoire/src/libgrimoire_xe2_gdn_raw.so
GRIMOIRE_ONEDNN_BRIDGE=/grimoire/src/libgrimoire_onednn.so
${EXTRA_ENV:-}"
P="Machine learning systems that run large language models on consumer hardware face a fundamental tension between memory bandwidth and arithmetic throughput. When a model is quantized to four bits, the weights shrink dramatically, but the arithmetic units must now spend cycles unpacking those weights before they can be multiplied. On integrated and discrete GPUs alike, this unpacking often becomes the bottleneck rather than the matrix multiplication itself. Explain in detail why this happens, what the roofline model predicts for such workloads, and which specific optimizations recover the lost throughput."
cd /mnt/storage/isos/grimoire-fuse
exec bash tools/b70run.sh renderD129 900 "$1" /grimoire/bin/grimoire -m "$2" -p "$P" -n 80
