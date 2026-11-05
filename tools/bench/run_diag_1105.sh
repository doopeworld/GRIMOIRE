#!/bin/bash
# 11-05: where a speculative step's time goes at 4 users, one card vs two cards (diagnosis for the two-card 4-user
# decode: 139 vs 177 tok/s on one card, Qwen3.8-27B GPTQ + MTP). Candidate build (grimoire-release-1105b), B70s only.
# GRIMOIRE_SPEC_TIMING=1 (host wall time of draft / verify / commit per step, a sync at each boundary), 4 coding answers
# at once (conc4); then two cards once more with GRIMOIRE_TP_SMALL_AR_TIME=1 (two waits per small all-reduce: the split
# of one all-reduce, not a speed). Runs right after the served A/B (NEXT_AB_DONE); holds logs/HOLD-diag so the release
# gate (which waits for Claude's review of the A/B anyway) starts after it.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105b
cd "$B" || exit 9
exec >>"$B/logs/diag-1105.out" 2>&1
echo "DIAG_START $(date +%T)"
until grep -qE "^NEXT_AB_(DONE|STOP)" logs/next-ab-1105.out 2>/dev/null; do sleep 20; done
touch logs/HOLD-diag
trap 'rm -f "$B/logs/HOLD-diag"' EXIT
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ]; do sleep 10; done
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
G="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_SPEC_TIMING=1"
M=/models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16
for spec in "g0 G0 " "tp TP " "tpar TP GRIMOIRE_TP_SMALL_AR_TIME=1"; do
  set -- $spec
  name=diag1105-$1
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" "$2" "$M" int4 16384 "$G ${3:-}" conc4 > "logs/$name.run" 2>&1
  echo "rc=$?"; grep -E "CONC|STOP|refusing|LEFT RUNNING|server stopped" "logs/$name.run" | head -4 | cut -c1-150
  grep -E "^spec timing:" "logs/$name.server.log" | cut -c1-230
  grep -E "TP small all-reduce" "logs/$name.server.log" | tail -4 | cut -c1-240
done
# the grouped speculative commit (fnmtp-1105 1058c74 = candidate + 1e12ee8: one replay / rebuild launch per layer for all
# conversations), one card, the same test as diag1105-g0: answers must equal, the commit time should drop
if grep -qx FNMTP_BUILD_PASS logs/fnmtp-build-1105.out 2>/dev/null; then
  for spec in "g0-grouped " "tp-grouped TP"; do
    set -- $spec
    name=diag1105-$1
    echo "== $(date +%T) $name"
    GRIM_TREE=/mnt/storage/isos/grimoire-runs/fnmtp-1105 bash ./dual_engine_1102.sh "$name" "${2:-G0}" "$M" int4 16384 "$G" conc4 \
      > "logs/$name.run" 2>&1
    echo "rc=$?"; grep -E "CONC|STOP|refusing|LEFT RUNNING|server stopped" "logs/$name.run" | head -4 | cut -c1-150
    grep -E "^spec timing:" "logs/$name.server.log" | cut -c1-230
  done
  python3 - "$B/logs" <<'PY'
import json, sys
d = sys.argv[1]
for a, b in (('diag1105-g0', 'diag1105-g0-grouped'), ('diag1105-tp', 'diag1105-tp-grouped')):
    try:
        A, Bj = (json.load(open(f'{d}/{x}.conc4.json')) for x in (a, b))
    except Exception as e:
        print(a, b, 'no conc4 json', e); continue
    same = sum(A[k]['text'] == Bj.get(k, {}).get('text') for k in A)
    print(f'{a} vs {b}: identical concurrent answers {same}/{len(A)}')
PY
fi
echo "DIAG_END $(date +%T)"
