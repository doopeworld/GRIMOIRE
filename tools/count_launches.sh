#!/usr/bin/env bash
# =====================================================================
#  count_launches.sh -- kernel launches per DECODE TOKEN.
#
#  The one performance number that is honest to take off the card: it is
#  a COUNT, not a timing, so it does not depend on the machine.  Rule 8
#  forbids quoting a benchmark from a config that never generated text;
#  it does not forbid counting what the engine submits, and launch count
#  is what the AMD RDNA4 stack's single biggest decode win was measured
#  in (1477 -> ~1080 per step, worth +14% single-stream, +24% at 8
#  concurrent).
#
#  METHOD.  Generate N tokens, then N+10, and subtract.  Model load and
#  prompt prefill are identical in both runs and cancel exactly; what
#  remains is ten decode steps.  Counting a single run instead would
#  fold the load in and report a number several times too large.
#
#  Usage:
#     tools/count_launches.sh <probe-binary> <model-dir>
#  where the probe takes <dir> <n-tokens>.  bin/grimoire works too if
#  you give it a prompt; the point is only that N is controllable.
#
#  ON THE B70, read the result next to the capability banner's graph
#  line.  GRIMOIRE replays a captured command graph when it can, which
#  removes per-launch submission cost wholesale -- so a high count
#  matters far less when capture is live, and matters a lot when it is
#  not.  Check which before spending anything on fusion.
# =====================================================================
set -u
BIN="${1:?usage: count_launches.sh <probe> <model-dir>}"
DIR="${2:?usage: count_launches.sh <probe> <model-dir>}"
LO="${LO:-1}"
HI="${HI:-11}"

run() {
  GRIMOIRE_DEVICE_ANY="${GRIMOIRE_DEVICE_ANY:-}" SYCL_UR_TRACE=2 \
    "$BIN" "$DIR" "$1" 2>&1 | grep -c "urEnqueueKernelLaunch"
}

a=$(run "$LO")
b=$(run "$HI")
span=$(( HI - LO ))
if [[ "$span" -le 0 ]]; then echo "HI must exceed LO"; exit 2; fi
echo "launches at n=$LO : $a"
echo "launches at n=$HI : $b"
echo "per decode token  : $(( (b - a) / span ))"
