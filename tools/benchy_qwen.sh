#!/bin/bash
# llama-benchy pp4096/tg32 against grimoire-server, one config per call.
# EXTRA holds newline-separated K=V passed to the server.
#
# A benchmark number is worthless unless you can say WHICH BINARY produced
# it.  Two ways this script used to lose that:
#   - no `pipefail`, so a benchy that died was hidden by the trailing grep
#     and the run looked like "no output" rather than "it failed";
#   - `uvx llama-benchy` unpinned, so two numbers a week apart could come
#     from two different harnesses.
# Both are closed below, and the binary + commit are stamped on every run.
set -uo pipefail
PORT="${PORT:-8099}"
REPO=/mnt/storage/isos/grimoire-fuse
cd "$REPO"

# ---- identity of the thing being measured ---------------------------
BIN=bin/grimoire-server
if [[ ! -x "$BIN" ]]; then
  echo "no $BIN -- build first (tools/build_bridges_b70.sh), nothing to measure"
  exit 1
fi
echo "=== measuring ==="
echo "binary : $BIN"
echo "sha256 : $(sha256sum "$BIN" | cut -d' ' -f1)"
echo "built  : $(date -r "$BIN" '+%Y-%m-%d %H:%M:%S')"
echo "commit : $(git rev-parse --short HEAD 2>/dev/null || echo '?')$(git diff --quiet 2>/dev/null || echo ' +dirty')"
echo "harness: llama-benchy ${BENCHY_VERSION:-<unpinned>}"
echo "extra  : ${EXTRA:-<none>}"
echo "================="

EXTRA="${EXTRA:-}" PORT="$PORT" bash tools/serve_qwen.sh >/tmp/serve_$$.log 2>&1 || {
  echo "server failed"; tail -12 /tmp/serve_$$.log; exit 1; }
cd /tmp
# Pin the harness: BENCHY_VERSION=0.4.2 -> uvx llama-benchy@0.4.2.  Unset
# means "whatever uvx resolves today", which is fine for a smoke test and
# not fine for a number you are going to quote.
BENCHY="llama-benchy${BENCHY_VERSION:+@${BENCHY_VERSION}}"
RAW=/tmp/benchy_$$.md
uvx "$BENCHY" --base-url "http://localhost:${PORT}/v1" \
  --model /models/Qwen3.8-27B-MXFP4-GRIMOIRE \
  --pp 4096 --tg 32 --runs "${RUNS:-3}" --latency-mode generation --format md \
  >"$RAW" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "=== BENCHY FAILED (exit $rc) -- these are NOT results ==="
  tail -20 "$RAW"
  exit "$rc"
fi
grep -E "pp4096|tg32|Coherence|FAILED|model +\|" "$RAW" || {
  echo "=== benchy produced no pp/tg rows -- full output: ==="
  tail -20 "$RAW"
  exit 1
}
