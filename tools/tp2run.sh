#!/bin/bash
# Functional two-rank tensor-parallel launcher using the proven vLLM XPU
# container/device topology. Initial TP keeps weights replicated and shards
# decode projection rows; storage/expert sharding follows correctness.
set -u
NODE0="${1:?first render node}"; shift
NODE1="${1:?second render node}"; shift
LIMIT="${1:?time limit in seconds}"; shift
NAME="${1:?short run name}"; shift
for NODE in "$NODE0" "$NODE1"; do
    [ -e "/dev/dri/$NODE" ] || { echo "no such render node: $NODE" >&2; exit 2; }
done
if docker ps --format '{{.Names}}' | grep -q '^grim-'; then
    echo "REFUSING: a grim-* container is already running" >&2; exit 3
fi
CNAME="grim-$NAME"
docker rm -f "$CNAME" >/dev/null 2>&1 || true
ENVARGS=(
    -e ZE_AFFINITY_MASK=0,1
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
)
if [ -n "${GRIM_ENV:-}" ]; then
    while IFS= read -r kv; do [ -n "$kv" ] && ENVARGS+=(-e "$kv"); done <<< "$GRIM_ENV"
fi
CID=$(docker run -d --name "$CNAME" -w /grimoire --init --stop-timeout 300 \
    --ipc=host --privileged --shm-size=10g --device /dev/dri:/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v /mnt/storage/isos/grimoire-fuse:/grimoire \
    -v /mnt/storage/Models:/models "${ENVARGS[@]}" \
    --entrypoint /usr/bin/timeout my-vllm-xpu:latest \
    --signal=TERM --kill-after=60 "$LIMIT" /grimoire/tools/tp2worker.sh "$@")
if [ -z "$CID" ]; then echo "launch failed" >&2; exit 4; fi
echo "launched $CNAME ${CID:0:12} for $NODE0+$NODE1 (limit ${LIMIT}s)"
RC=$(docker wait "$CNAME" 2>/dev/null || echo wait-failed)
docker logs "$CNAME" >"/tmp/$CNAME.log" 2>&1
echo "exit=$RC log=/tmp/$CNAME.log"
if [ "$RC" = 0 ]; then docker rm "$CNAME" >/dev/null 2>&1;
else echo "NON-ZERO EXIT -- container $CNAME kept for inspection" >&2; fi
