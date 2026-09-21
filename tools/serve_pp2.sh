#!/bin/bash
# grimoire-server across TWO B70s: an OpenAI-compatible HTTP front end on
# one card and a pipeline worker on the other.
#
#   serve_pp2.sh <MODEL_DIR> [PORT] [SPLIT]
#   serve_pp2.sh /models/Ornith-1.5-35B-A3B 8099 24
#
# SPLIT is how many layers rank 0 keeps; the rest go to rank 1.  Give the
# SLOWER card fewer layers -- pipeline throughput is set by the slowest
# stage, so an even split makes the slower card the ceiling.
#
# WHY THIS EXISTS.  Until 2026-09-18 pipeline parallel was CLI-only:
# tools/pp2run.sh runs bin/grimoire with one prompt and exits.  serve.sh
# serves HTTP but opens exactly one render node.  So a model that needs
# two cards -- Ornith at fp8, anything at bf16 -- could be RUN but not
# SERVED, which is the configuration agentic work actually uses.
#
# The missing piece was small: every stage runs the same generation loop
# and they stay in step by exchanging a message per token, but the prompt
# only ever reached rank 0.  The CLI never noticed because every rank is
# launched with the same -p.  The front end now forwards each request
# down the pipe and the workers follow it (grimoire_pp_worker_loop).
#
# CONCURRENCY: the scheduler falls back to ONE REQUEST AT A TIME under
# pipeline parallel -- batching across sequences is not implemented for
# PP (batch_unsupported_reason says so, and the banner prints it).  The
# prefix cache is off under PP too.  On one card, int4/mxfp4, you get
# both; this is the two-card path and it is serial.  That is still the
# difference between serving the model and not.
set -u

MODEL="${1:?model dir, e.g. /models/Ornith-1.5-35B-A3B}"
PORT="${2:-8099}"
SPLIT="${3:-${GRIMOIRE_PP_SPLIT:-24}}"
IMAGE="${GRIM_IMAGE:-my-vllm-xpu:latest}"
CNAME="${CNAME:-grim-server-pp2}"
PROJ="${PROJ:-fp8}"
CTX="${CTX:-8192}"

cd /mnt/storage/isos/grimoire-fuse

if docker ps --format '{{.Names}}' | grep -q '^grim-'; then
    echo "REFUSING: a grim-* container is already running" >&2; exit 3
fi
docker rm -f "$CNAME" >/dev/null 2>&1 || true

# Both cards in one container, exactly as pp2run.sh does it: separate
# processes, separate Level Zero contexts, full DRM access.
CID=$(docker run -d --name "$CNAME" -w /grimoire --init --stop-timeout 300 \
    -p "${PORT}:${PORT}" \
    --ipc=host --privileged --shm-size=10g \
    --device /dev/dri:/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v /mnt/storage/isos/grimoire-fuse:/grimoire \
    -v /mnt/storage/Models:/models \
    -e ZE_AFFINITY_MASK=0,1 \
    -e GRIMOIRE_PP_WORLD_SIZE=2 \
    -e GRIMOIRE_PP_SPLIT="$SPLIT" \
    -e GRIMOIRE_PP_SOCKET=/tmp/grimoire-pp-serve.sock \
    -e GRIMOIRE_DEFER_MOE_GATHER=1 \
    -e GRIMOIRE_BF16_QKV=1 \
    -e GRIMOIRE_BF16_DN_QKV=1 \
    -e GRIMOIRE_XE2_GROUPED_BRIDGE=/grimoire/src/libgrimoire_xe2_grouped.so \
    -e GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_bridge.so \
    -e GRIMOIRE_XE2_GDN_RAW_BRIDGE=/grimoire/src/libgrimoire_xe2_gdn_raw.so \
    -e GRIMOIRE_ONEDNN_BRIDGE=/grimoire/src/libgrimoire_onednn.so \
    -e LD_LIBRARY_PATH=/grimoire/src:/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:/opt/intel/oneapi/lib:/usr/local/lib \
    --entrypoint /grimoire/tools/serve_pp2_worker.sh "$IMAGE" \
    "$MODEL" "$PORT" "$PROJ" "$CTX")

if [ -z "$CID" ]; then echo "launch failed" >&2; exit 4; fi
echo "launched $CNAME ${CID:0:12} on port $PORT, split $SPLIT"
echo "  front end : rank 0, serving http://0.0.0.0:${PORT}"
echo "  worker    : rank 1, following rank 0"
echo
echo "  docker logs -f $CNAME"
echo "  curl -s localhost:${PORT}/v1/models"
echo
echo "READ THE BANNER before believing anything (rule 15).  Rank 0 prints"
echo "what it decided; 'batching one at a time -- pipeline parallel' is"
echo "expected here and is not an error."
