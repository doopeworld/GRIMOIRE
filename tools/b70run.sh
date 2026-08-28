#!/bin/bash
# ---------------------------------------------------------------------------
# b70run.sh -- safe single-GPU Grimoire launcher.
#
# Every wedged B70 in this project traces to container lifecycle, not kernels:
#
#   * `docker run --entrypoint bash -lc CMD` makes bash PID 1.  PID 1 gets no
#     default signal dispositions, and bash does not forward SIGTERM to the
#     child it is waiting on, so `docker stop` always fell through to SIGKILL
#     after its 10s grace.  SIGKILL tears the DRM fd down mid-submission: the
#     batch never signals its fence, the process parks in D state at
#     dma_fence_default_wait, and the device falls off the PCI bus.
#
#   * `timeout N docker run ...` signals the docker *client*, not the
#     container.  The container keeps running, --rm never fires, and the
#     orphan holds the GPU until a power cycle.
#
# This script closes both:
#   --init          tini is PID 1: forwards signals, reaps children.
#   exec grimoire   the workload is a direct child, not a grandchild of bash.
#   inside timeout  the limit lives IN the container, so the process always
#                   exits by its own choice and the container exits normally.
#   -d + wait       detached, so no client-side signal can orphan anything.
#   --stop-timeout  GPU work gets time to drain before any SIGKILL.
#
# Usage: b70run.sh <renderNode> <limitSeconds> <name> <command...>
#   b70run.sh renderD130 900 base32 /grimoire/bin/grimoire -m /models/... 
# Env for the workload is passed through GRIM_ENV (newline-separated K=V).
# ---------------------------------------------------------------------------
set -u

NODE="${1:?render node, e.g. renderD130}"; shift
LIMIT="${1:?time limit in seconds}";       shift
NAME="${1:?short run name}";               shift

if [ ! -e "/dev/dri/$NODE" ]; then echo "no such render node: $NODE" >&2; exit 2; fi

# Refuse to start if anything is already holding a GPU.
if docker ps --format '{{.Names}}' | grep -q '^grim-'; then
    echo "REFUSING: a grim-* container is already running:" >&2
    docker ps --format '  {{.ID}} {{.Names}} {{.Status}}' | grep grim- >&2
    exit 3
fi

CNAME="grim-$NAME"
docker rm -f "$CNAME" >/dev/null 2>&1 || true

ENVARGS=()
if [ -n "${GRIM_ENV:-}" ]; then
    while IFS= read -r kv; do
        [ -n "$kv" ] && ENVARGS+=(-e "$kv")
    done <<< "$GRIM_ENV"
fi

# `timeout --signal=TERM --kill-after=60` inside the container: the workload
# gets a clean TERM first and a full minute to drain GPU work before KILL.
CID=$(docker run -d --name "$CNAME" -w /grimoire \
    --init \
    --stop-timeout 300 \
    --device "/dev/dri/$NODE" \
    -v /mnt/storage/isos/grimoire-fuse:/grimoire \
    -v /mnt/storage/Models:/models \
    "${ENVARGS[@]}" \
    --entrypoint /usr/bin/timeout \
    my-vllm-xpu:latest \
    --signal=TERM --kill-after=60 "$LIMIT" "$@")

if [ -z "$CID" ]; then echo "launch failed" >&2; exit 4; fi
echo "launched $CNAME ${CID:0:12} on $NODE (limit ${LIMIT}s)"

# Server-side wait.  Nothing here can orphan the container: if this script is
# interrupted the container still owns its own lifetime and its own timeout.
RC=$(docker wait "$CNAME" 2>/dev/null || echo "wait-failed")
docker logs "$CNAME" > "/tmp/$CNAME.log" 2>&1
echo "exit=$RC  log=/tmp/$CNAME.log"

if [ "$RC" = "0" ]; then
    docker rm "$CNAME" >/dev/null 2>&1
else
    echo "NON-ZERO EXIT -- container $CNAME kept for inspection (docker rm when done)" >&2
fi
