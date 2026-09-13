#!/bin/bash
# N-GPU launcher for pipeline or tensor parallel.  pp2run.sh/tp2run.sh do
# exactly two cards; this does any number, because GRIMOIRE has to run on
# whatever Battlemage cards the box has -- one B70, two B70s, two B70s
# plus a B580, four of anything.
#
# The container topology is the one that is known to work and is copied
# from pp2run.sh verbatim: host IPC, privileged DRM, 10 GiB SHM, full
# /dev/dri plus by-path.  ZE_AFFINITY_MASK is built from the node count
# rather than hardcoded to 0,1.
#
#   tools/nrun.sh pp renderD128,renderD129,renderD130 1800 pp3 \
#       /grimoire/bin/grimoire -m /models/<dir> --proj fp8 -p '...' -n 64
#
# For PP with more than two cards GRIMOIRE_PP_LAYERS is REQUIRED -- the
# engine refuses to guess a split, and it is what lets a small card carry
# fewer layers than a big one:
#
#   GRIMOIRE_PP_LAYERS=30,30,12 tools/nrun.sh pp <nodes> 1800 pp3 ...
set -u

MODE="${1:?mode: pp or tp}"; shift
NODES="${1:?comma-separated render nodes, e.g. renderD128,renderD129}"; shift
LIMIT="${1:?time limit in seconds}"; shift
NAME="${1:?short run name}"; shift

case "$MODE" in pp|tp) ;; *) echo "mode must be pp or tp" >&2; exit 2 ;; esac

IFS=',' read -r -a NODE_LIST <<< "$NODES"
N=${#NODE_LIST[@]}
[ "$N" -ge 1 ] || { echo "need at least one render node" >&2; exit 2; }
for NODE in "${NODE_LIST[@]}"; do
    [ -e "/dev/dri/$NODE" ] || { echo "no such render node: $NODE" >&2; exit 2; }
done

# Refuse early rather than let the container start and die inside: the
# engine requires an explicit split for more than two pipeline stages,
# and finding that out from a container log is needless friction.
if [ "$MODE" = pp ] && [ "$N" -gt 2 ] && [ -z "${GRIMOIRE_PP_LAYERS:-}" ]; then
    echo "GRIMOIRE_PP_LAYERS is required for pipeline parallel over $N GPUs" >&2
    echo "  e.g. GRIMOIRE_PP_LAYERS=30,30,12 (one positive count per GPU," >&2
    echo "       summing to the model's layer count; give the small card fewer)" >&2
    exit 2
fi

if docker ps --format '{{.Names}}' | grep -q '^grim-'; then
    echo "REFUSING: a grim-* container is already running" >&2; exit 3
fi

# 0,1,...,N-1 -- the visibility list every child shares, exactly as the
# two-card script does.  Each child then selects its own device by rank.
MASK=$(seq -s, 0 $((N-1)))

CNAME="grim-$NAME"
docker rm -f "$CNAME" >/dev/null 2>&1 || true
ENVARGS=(
    -e ZE_AFFINITY_MASK="$MASK"
    -e GRIMOIRE_WORLD_SIZE="$N"
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn
    -e VLLM_XPU_ENABLE_XPU_GRAPH=1
    -e VLLM_USE_V2_MODEL_RUNNER=1
    -e CCL_ATL_TRANSPORT=ofi
    -e CCL_ZE_IPC_EXCHANGE=sockets
    -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
    -e VLLM_XPU_FUSED_MOE_USE_MXFP4_FP8=1
    -e TORCH_COMPILE_BACKEND=inductor
    -e ZE_SHARED_FORCE_DEVICE_ALLOC=1
    -e VLLM_TARGET_DEVICE=xpu
    -e ONECCL_BINDINGS_FOR_PYTORCH_ENV_MODE=p2p
    -e LD_LIBRARY_PATH=/grimoire/src:/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels:/opt/intel/oneapi/lib:/usr/local/lib
    -e GRIMOIRE_XE2_GROUPED_BRIDGE=/grimoire/src/libgrimoire_xe2_grouped.so
    -e GRIMOIRE_XE2_ATTN_BRIDGE=/grimoire/src/libgrimoire_xe2_attention_bridge.so
    -e GRIMOIRE_XE2_GDN_RAW_BRIDGE=/grimoire/src/libgrimoire_xe2_gdn_raw.so
    -e GRIMOIRE_ONEDNN_BRIDGE=/grimoire/src/libgrimoire_onednn.so
    -e GRIMOIRE_DEFER_MOE_GATHER=1
    -e GRIMOIRE_BF16_QKV=1
    -e GRIMOIRE_BF16_DN_QKV=1
)
# Pass through the split and any device remap the caller set.
for v in GRIMOIRE_PP_LAYERS GRIMOIRE_PP_SPLIT GRIMOIRE_DEVICES; do
    [ -n "${!v:-}" ] && ENVARGS+=(-e "$v=${!v}")
done
if [ -n "${GRIM_ENV:-}" ]; then
    while IFS= read -r kv; do
        [ -n "$kv" ] && ENVARGS+=(-e "$kv")
    done <<< "$GRIM_ENV"
fi

CID=$(docker run -d --name "$CNAME" -w /grimoire --init --stop-timeout 300 \
    --ipc=host --privileged --shm-size=10g \
    --device /dev/dri:/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v /mnt/storage/isos/grimoire-fuse:/grimoire \
    -v /mnt/storage/Models:/models \
    "${ENVARGS[@]}" --entrypoint /usr/bin/timeout my-vllm-xpu:latest \
    --signal=TERM --kill-after=60 "$LIMIT" \
    /grimoire/tools/nworker.sh "$MODE" "$N" "$@")

if [ -z "$CID" ]; then echo "launch failed" >&2; exit 4; fi
echo "launched $CNAME ${CID:0:12} for $N GPU(s) [$NODES] mode=$MODE (limit ${LIMIT}s)"
RC=$(docker wait "$CNAME" 2>/dev/null || echo wait-failed)
docker logs "$CNAME" >"/tmp/$CNAME.log" 2>&1
echo "exit=$RC log=/tmp/$CNAME.log"
if [ "$RC" = 0 ]; then
    docker rm "$CNAME" >/dev/null 2>&1
else
    echo "NON-ZERO EXIT -- container $CNAME kept for inspection" >&2
fi
