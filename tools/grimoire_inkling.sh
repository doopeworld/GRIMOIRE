#!/bin/bash
# GRIMOIRE Inkling 2 Mini server (OpenAI API) inside the grimoire-b70 image: the engine (one or two cards) + the HTTP
# front end.  Run with --entrypoint grimoire-inkling.
#   INK_MODEL   model directory (default /model)                     PORT / HOST   (default 6911 / 0.0.0.0)
#   INK_CARDS   1 or 2 (default: the number of visible GPUs)         INK_CTX       context tokens (default 32768)
#   INK_CHUNK   prompt rows per pass (default 4096)                  INK_NAME      model name in the API (inkling-2-mini)
#   INK_DFLASH2 the DFlash2 drafter (incoai/Inkling-2-Mini-DFlash2, ~1.3x faster decode than the built-in DSpark
#               head): used automatically when it sits next to the model (<models>/Inkling-2-Mini-DFlash2) or in
#               <model>/dflash2; INK_DFLASH2=<dir> names it, INK_DFLASH2=0 keeps DSpark
#   INK_*       engine switches (see the engine source), e.g. INK_DRAFT_TAU (0.25), INK_NO_EPREP=1 (two cards with
#               half the experts each: needed when one card cannot hold all experts, e.g. two Arc Pro B60)
# docker stop: the front end exits, the engine ends the running request at a decode-step boundary and quits (give the
# container --stop-timeout 60 so a long prompt can finish its pass).
set -u
MODEL=${INK_MODEL:-/model}
CTX=${INK_CTX:-32768}
CHUNK=${INK_CHUNK:-4096}
PORT=${PORT:-6911}
HOST=${HOST:-0.0.0.0}
NAME=${INK_NAME:-inkling-2-mini}
if [ -z "${INK_CARDS:-}" ]; then
  INK_CARDS=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
  [ "$INK_CARDS" -gt 2 ] && INK_CARDS=2
fi
[ "$INK_CARDS" -ge 1 ] || { echo "grimoire-inkling: no GPU visible (map /dev/dri/renderD*)"; exit 1; }
export INK_I8=${INK_I8:-1} INK_ESMOE=${INK_ESMOE:-1} UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
if [ -z "${ONEAPI_DEVICE_SELECTOR:-}" ]; then
  if [ "$INK_CARDS" = 2 ]; then export ONEAPI_DEVICE_SELECTOR=level_zero:0,1; else export ONEAPI_DEVICE_SELECTOR=level_zero:0; fi
fi
SOCK=/tmp/inkling-engine.sock
rm -f "$SOCK"
D=/grimoire/inkling
"$D/inkling2_gpu" "$MODEL" serve "$SOCK" "$CTX" "$CHUNK" "$INK_CARDS" &
EPID=$!
FPID=""
stop() {
  [ -n "$FPID" ] && kill -TERM "$FPID" 2>/dev/null
  kill -TERM "$EPID" 2>/dev/null
  wait "$EPID"
  exit 0
}
trap stop TERM INT
# the engine creates its socket after loading and warming up
while [ ! -S "$SOCK" ]; do
  kill -0 "$EPID" 2>/dev/null || { wait "$EPID"; echo "grimoire-inkling: engine exited during start"; exit 1; }
  sleep 1
done
python3 -u "$D/inkling2_server.py" --model-dir "$MODEL" --sock "$SOCK" --ctx "$CTX" --port "$PORT" --host "$HOST" --name "$NAME" &
FPID=$!
wait -n "$EPID" "$FPID"
st=$?
kill -TERM "$FPID" "$EPID" 2>/dev/null
wait "$EPID"
exit $st
