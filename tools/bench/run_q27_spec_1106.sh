#!/bin/bash
# 11-06: Qwen3.8-27B on ONE B70 (gpu0), INT4, one user: the model's own MTP head (the GRIMOIRE-QWEN3.8 profile
# settings) vs the DFlash2 drafter (z-lab, /models2/Qwen3.8-27B-DFlash2) -- which speculation the one-card INT4
# profile should use.  Never measured on one card at INT4:
#   10-27 two cards FP8 (q38_spec.sh): code probe MTP 88.8-89.2 tok/s (4.40 tok/step) vs DFlash2 126.0-128.4
#         (6.55 tok/step); llama-benchy tg128 at 4K / 8K 56.2 / 52.9 vs 59.3 / 67.1.
#   11-01 one card native GGUF UD-Q4_K_XL: llama-benchy tg128 MTP 59.4 vs DFlash2 51.6.
# Same INT4 weights in both runs (the GPTQ-Int4-MTP-BF16 checkpoint: symmetric g128, the layout the AutoRound file
# also has, so the kernels are the same), the release-candidate tree (grimoire-release-1105b), one server at a time.
# Answers are listed for review; a failed run stops the script.  Log: logs/q27-spec-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=${SPEC_TREE:-/mnt/storage/isos/grimoire-release-1105b}
M=/models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16
cd "$B" || exit 9
exec >>"$B/logs/q27-spec-1106.out" 2>&1
echo "Q27_SPEC_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
[ -x "$T/bin/grimoire-server" ] || { echo "Q27_SPEC_STOP no server in $T"; exit 9; }
[ -d /mnt/models/Qwen3.8-27B-DFlash2 ] || { echo "Q27_SPEC_STOP no DFlash2 drafter"; exit 9; }
waitq() { until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-\|^grim-get-")" ] &&
                ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done; }
export HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 CODE_LENGTHS=4000 CODE_REPEATS=1
run() { # name model proj ctx env tests benchy...
  local name=$1; shift
  waitq; echo "== $(date +%T) $name"
  GRIM_TREE=$T bash ./dual_engine_1102.sh "$name" G0 "$@" > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|answer:|answers pass|Coherence|tests=|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-150
  grep -hiE "speculat|dflash|mtp|accept" "logs/$name.server.log" 2>/dev/null | head -6 | cut -c1-150 | sed 's/^/   | /'
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "Q27_SPEC_STOP $name rc=$rc"; exit 9
  fi
}
P="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8"
BQ="--pp 4096 8192 --tg 128 --concurrency 1 --runs 2"
run q27s-mtp "$M" int4 16384 "$P GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536" ctest,code,benchy $BQ
run q27s-df2 "$M" int4 16384 "$P GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2" ctest,code,benchy $BQ
for n in q27s-mtp q27s-df2; do
  echo "-- $n"
  grep -E "^\| " "logs/$n.benchy.md" 2>/dev/null | cut -c1-150
  python3 - "logs/$n.code.json" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
for k, v in d.items():
    if isinstance(v, dict):
        print('   code', k, {x: v[x] for x in v if x in ('tok_s', 'tokens_per_s', 'decode_tok_s', 'tokens', 'pass', 'ok')})
PY
done
echo "Q27_SPEC_DONE $(date +%T)"
