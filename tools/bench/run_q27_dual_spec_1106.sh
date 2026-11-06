#!/bin/bash
# 11-06: Qwen3.8-27B on TWO B70s (Megatron TP), INT8 (Ian: two cards INT8 / FP8; the official FP8 checkpoint as
# --proj int8 = top-1 98.0% vs the BF16 original, as --proj fp8 94.9%, 10-29), one user: the model's own MTP head
# vs the DFlash2 drafter (z-lab, /models2/Qwen3.8-27B-DFlash2) -- where GRIMOIRE stands against Radiance's
# headline (2x R9700, TP2: MXFP4 67 tok/s plain, DFlash2 x7 222 greedy; FP8 container decode-only 169).
# GRIMOIRE 10-27, two cards FP8: code probe MTP 88.8-89.2 tok/s vs DFlash2 126.0-128.4; benchy tg128 at 4K / 8K
# MTP 56.2 / 52.9, DFlash2 59.3 / 67.1.  10-29 two cards INT8 + MTP: pp4096 3,566, tg128 53.6.
#   mtp    GRIMOIRE_MTP=1 (the profile's speculation; its default depth)             ctest,code,benchy
#   df2    GRIMOIRE_DFLASH_MODEL (MTP off)                                           ctest,code,benchy
#   df2-ar df2 + GRIMOIRE_TP_SMALL_AR_TIME=1 (diagnostic: where a verify step's all-reduces go; code only)
# gpu1 runs at Gen3 x4 today (Ian: "be careful"): dual_engine_1102.sh's link guard stops a run on any uncorrectable
# error; this script stops at the first stopped or failed run. Holds logs/HOLD-q27dual. Log: logs/q27-dual-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=${SPEC_TREE:-/mnt/storage/isos/grimoire-runs/fnmtp-verify-1106}
M=/models2/Qwen3.8-27B-FP8
cd "$B" || exit 9
exec >>"$B/logs/q27-dual-1106.out" 2>&1
echo "Q27_DUAL_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
[ -x "$T/bin/grimoire-server" ] || { echo "Q27_DUAL_STOP no server in $T"; exit 9; }
[ -d /mnt/models/Qwen3.8-27B-DFlash2 ] || { echo "Q27_DUAL_STOP no DFlash2 drafter"; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-q27dual
trap 'rm -f "$B/logs/HOLD-q27dual"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 CODE_LENGTHS=4000 CODE_REPEATS=1
run() { # name env tests [benchy args]
  local name=$1 env=$2; shift 2
  echo "== $(date +%T) $name [$env]"
  bash ./dual_engine_1102.sh "$name" TP "$M" int8 16384 "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 $env" "$@" \
    > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|answer:|answers pass|Coherence|tests=|server stopped|LEFT RUNNING|STOP|link (before|after)|RxErr during" \
    "logs/$name.run" | cut -c1-170
  grep -hiE "speculat|dflash|mtp|accept|TP small all-reduce" "logs/$name.server.log" 2>/dev/null | tail -6 | cut -c1-230 |
    sed 's/^/   | /'
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "Q27_DUAL_STOP $name rc=$rc"; exit 9
  fi
}
BQ="--pp 4096 --tg 128 --concurrency 1 --runs 2"
run q27d-mtp "GRIMOIRE_MTP=1" ctest,code,benchy $BQ
run q27d-df2 "GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2" ctest,code,benchy $BQ
run q27d-df2-ar "GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_TP_SMALL_AR_TIME=1" code
for n in q27d-mtp q27d-df2; do
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
echo "Q27_DUAL_DONE $(date +%T)"
