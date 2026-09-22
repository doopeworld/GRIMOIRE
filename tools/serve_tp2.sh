#!/bin/bash
# grimoire-server across TWO B70s with TENSOR parallel: both cards hold a
# slice of every weight, rank 0 serves HTTP, rank 1 follows it.
#
#   serve_tp2.sh <MODEL_DIR> [PORT]
#   PROJ=fp8 CTX=8192 serve_tp2.sh /models/Ornith-1.5-35B-A3B 8099
#
# WHY THIS EXISTS (2026-09-22).  The engine gained TP serving in the
# composable-serving work -- TP workers run grimoire_pp_worker_loop, and
# rank 0's scheduler drives them with the same admit / batch-step control
# messages PP uses -- but there was no launcher, so the only thing that
# had ever run it was bin/test_batch_parallel.  This is serve_pp2.sh with
# the split changed, and the same worker script in TP mode.
#
# WHAT TP DOES AND DOES NOT DO HERE -- read before choosing it over PP:
#   * every sharded projection is all-gathered through PINNED HOST memory
#     over a Unix socket, once per projection per step.  That is many more
#     host round trips per token than PP's one per stage boundary.  Nobody
#     has measured either on OCuLink; measure both before preferring one.
#   * prompt processing under TP is TOKEN AT A TIME: prefill() declines TP
#     outside a batched decode step.  Long prompts will be slow to start.
#   * batching needs GRIMOIRE_SEQ_SLOTS > 1 (export it before launching).
#   * the prefix cache WORKS under TP (unlike PP) with GRIMOIRE_PREFIX_CACHE
#     =1; every rank resumes from its own snapshot and they agree on it.
#   * a drafter under TP serves one request at a time (TP + speculation
#     does not batch).
# The banner on rank 0 says which of these it decided (rule 15).
set -u

MODEL="${1:?model dir, e.g. /models/Ornith-1.5-35B-A3B}"
PORT="${2:-8099}"
IMAGE="${GRIM_IMAGE:-my-vllm-xpu:latest}"
CNAME="${CNAME:-grim-server-tp2}"
PROJ="${PROJ:-fp8}"
CTX="${CTX:-8192}"

cd /mnt/storage/isos/grimoire-fuse

if docker ps --format '{{.Names}}' | grep -q '^grim-'; then
    echo "REFUSING: a grim-* container is already running" >&2; exit 3
fi
docker rm -f "$CNAME" >/dev/null 2>&1 || true

# Serving knobs pass through from the host ONLY IF SET there: `-e NAME` with
# no value copies the host's value and adds nothing when it is unset, so
# the default launch is unchanged.
#
# Both cards in one container, exactly as tp2run.sh / serve_pp2.sh do it:
# separate processes, separate Level Zero contexts, full DRM access.
CID=$(docker run -d --name "$CNAME" -w /grimoire --init --stop-timeout 300 \
    -p "${PORT}:${PORT}" \
    --ipc=host --privileged --shm-size=10g \
    --device /dev/dri:/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v /mnt/storage/isos/grimoire-fuse:/grimoire \
    -v /mnt/storage/Models:/models \
    -e ZE_AFFINITY_MASK=0,1 \
    -e GRIMOIRE_SERVE_MODE=TP \
    -e GRIMOIRE_TP_WORLD_SIZE=2 \
    -e GRIMOIRE_TP_SOCKET=/tmp/grimoire-tp-serve.sock \
    -e GRIMOIRE_DEFER_MOE_GATHER=1 \
    -e GRIMOIRE_BF16_QKV=1 \
    -e GRIMOIRE_BF16_DN_QKV=1 \
    -e GRIMOIRE_XE2_GROUPED_BRIDGE=/grimoire/src/libgrimoire_xe2_grouped.so \
    -e GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_bridge.so \
    -e GRIMOIRE_XE2_GDN_RAW_BRIDGE=/grimoire/src/libgrimoire_xe2_gdn_raw.so \
    -e GRIMOIRE_ONEDNN_BRIDGE=/grimoire/src/libgrimoire_onednn.so \
    -e GRIMOIRE_SEQ_SLOTS \
    -e GRIMOIRE_MAX_BATCH \
    -e GRIMOIRE_PREFIX_CACHE \
    -e GRIMOIRE_PREFIX_SLOTS \
    -e GRIMOIRE_MTP \
    -e GRIMOIRE_MTP_K \
    -e GRIMOIRE_DFLASH_MODEL \
    -e GRIMOIRE_DFLASH_M \
    -e GRIMOIRE_SPEC_STATS \
    -e LD_LIBRARY_PATH=/grimoire/src:/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:/opt/intel/oneapi/lib:/usr/local/lib \
    --entrypoint /grimoire/tools/serve_pp2_worker.sh "$IMAGE" \
    "$MODEL" "$PORT" "$PROJ" "$CTX")

if [ -z "$CID" ]; then echo "launch failed" >&2; exit 4; fi
echo "launched $CNAME ${CID:0:12} on port $PORT, tensor parallel over 2 cards"
echo "  front end : rank 0, serving http://0.0.0.0:${PORT}"
echo "  worker    : rank 1, following rank 0"
echo
echo "  docker logs -f $CNAME"
echo "  curl -s localhost:${PORT}/v1/models"
echo
echo "READ THE BANNER before believing anything (rule 15).  Rank 0 prints"
echo "'parallel TENSOR, rank 0 of 2' and what the scheduler decided.  Without"
echo "GRIMOIRE_SEQ_SLOTS>1 it is one request at a time, which is expected."
