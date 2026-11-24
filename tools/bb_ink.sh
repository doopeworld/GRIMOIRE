#!/bin/bash
# BetterBench against the running Inkling server (serve_ink.sh): same config as the earlier vLLM / GRIMOIRE runs
# (bench-1022/betterbench-vllm1111.json: temp 0.7, top-p 0.95, top-k 20, prefill 2K / 8K, concurrency 1/2/4/8).
#   bb_ink.sh TAG [betterbench run args, e.g. --quick --decode]
# Client pinned to CPUs 14,15.  Prints BetterBench's summary and the engine's own step statistics for the run.
X=/mnt/storage/isos/grimoire-runs/ink-1123
B=/mnt/storage/isos/grimoire-runs/bench-1022
TAG=$1; shift
L=$X/logs/serve-requests.jsonl
n0=$(wc -l < $L 2>/dev/null || echo 0)
cd $B/betterbench1109 || exit 1
PYTHONPATH=. taskset -c 14,15 $X/venv/bin/python -u -m betterbench.cli run --endpoint http://127.0.0.1:${PORT:-6911}/v1 --model inkling-2-mini \
  --config $B/betterbench-vllm1111.json --max-model-len 32768 --name ink-$TAG --out $X/logs/bb-$TAG.json --no-update-check "$@" \
  > $X/logs/bb-$TAG.out 2>&1 < /dev/null
echo "betterbench exit $?"
$X/venv/bin/python - "$L" "$n0" <<'EOF'
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])][int(sys.argv[2]):]
E = [r["end"] for r in L if r.get("end")]
single = [e for e in E if e["gen"] > 32]
g = sum(e["gen"] for e in single); st = sum(e["steps"] for e in single); dm = sum(e["decode_ms"] for e in single)
print("engine: %d requests (%d errors); generations > 32 tokens: %.2f tokens/step, %.1f ms/step, %.1f tok/s decode"
      % (len(L), sum(1 for r in L if r.get("err")), g / max(1, st), dm / max(1, st), 1000.0 * g / max(1e-9, dm)))
EOF
