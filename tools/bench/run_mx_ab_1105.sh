#!/bin/bash
# 11-05: MXFP4 small-M (verify / batched decode) A/B on one B70 (gpu0), one server at a time, the SAME binary
# (grimoire-mx-smallm-1105, server, G31): GRIMOIRE_MX_SMALLM_256=1 = the library kernel (v1.10.3 behaviour) vs
# the default 128-register kernel with (k, k+4) nibble pairs.
#   Qwen3.8-27B MXFP4 (GRIMOIRE's conversion), the 11-04 MXFP4-vs-Swift settings: MTP K=6, draft vocabulary 65,536,
#   4 sequence slots, --ctx 36864; Paris, coding at 4K / 16K (unit-tested), 4 at once, llama-benchy pp4096 / 16384.
#   Ornith-1.5-35B-A3B MXFP4 (the ORNITH profile's model and settings), 1 and 8 users.
# Waits for the build and for a quiet machine; removes logs/HOLD-v1104 at the end (the v1.10.4 gate waits on it).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
trap 'rm -f "$B/logs/HOLD-v1104"' EXIT
until grep -qE "^BUILD_(PASS|FAIL)" logs/build-mx-1105.out 2>/dev/null; do sleep 15; done
grep -qx BUILD_PASS logs/build-mx-1105.out || { echo "MXAB_STOP build failed"; exit 9; }
until [ -z "$(docker ps -q)" ]; do sleep 10; done
export GRIM_TREE=/mnt/storage/isos/grimoire-mx-smallm-1105 HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
run() { # name model proj ctx env tests benchy...
  local name=$1; shift
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" G0 "$@" > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|answer:|answers pass|Coherence|tests=|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-150
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "MXAB_STOP $name rc=$rc"; exit 9
  fi
}
Q="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=4 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536"
O="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8"
QM=/models2/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16
OM=/models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
export CODE_LENGTHS=4000,16000 CODE_REPEATS=1
run mx1105-q-new "$QM" mxfp4 36864 "$Q" ctest,code,conc4,benchy --pp 4096 16384 --tg 128 --concurrency 1 2 4 --runs 2
run mx1105-q-old "$QM" mxfp4 36864 "$Q GRIMOIRE_MX_SMALLM_256=1" ctest,code,conc4,benchy --pp 4096 16384 --tg 128 --concurrency 1 2 4 --runs 2
unset CODE_LENGTHS CODE_REPEATS
run mx1105-o-new "$OM" mxfp4 32000 "$O" ctest,conc4,seq4,benchy --pp 4096 --tg 128 --concurrency 1 8 --runs 2
run mx1105-o-old "$OM" mxfp4 32000 "$O GRIMOIRE_MX_SMALLM_256=1" ctest,conc4,seq4,benchy --pp 4096 --tg 128 --concurrency 1 8 --runs 2
python3 "$B/compare_ab_1105.py" "$B/logs" library=mx1105-q-old new=mx1105-q-new
python3 "$B/compare_ab_1105.py" "$B/logs" library=mx1105-o-old new=mx1105-o-new
python3 - "$B/logs" <<'PY'
import json, sys
d = sys.argv[1]
for a, b in (('mx1105-o-old', 'mx1105-o-new'),):
    for kind in ('seq4', 'conc4'):
        A, Bj = json.load(open(f'{d}/{a}.{kind}.json')), json.load(open(f'{d}/{b}.{kind}.json'))
        same = sum(A[k]['text'] == Bj.get(k, {}).get('text') for k in A)
        print(f'Ornith {kind}: library vs new kernel, identical answers {same}/{len(A)}')
PY
echo "MXAB_DONE $(date +%T)"
