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
# Where one verify pass (7 rows: MTP K=6) spends its time, per region summed over all 64 layers: CLI, the 6K Sherlock
# prompt, 96 tokens, GRIMOIRE_TIME_LAYER=all (drains the queue after every region: region = its kernels + launch
# latency, so small regions read high). MXFP4 and Swift INT4.
exec 9>/mnt/user/isos/grimoire-runs/gpu0.lock
if flock -n 9; then
  cd /mnt/storage/isos/grimoire-fuse && for spec in "mx /models2/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16 mxfp4" \
      "swift /models2/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8 int4"; do
    set -- $spec
    EXTRA_ENV="GRIMOIRE_POWER_LIMIT_W=0
GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=6
GRIMOIRE_MTP_DRAFT_VOCAB=65536
GRIMOIRE_TIME_LAYER=all" GRIM_TREE=$T LIM=900 TAILN=5 \
      bash tools/g0run.sh "vregion-$1" /grimoire/bin/grimoire -m "$2" --proj "$3" --ctx 8192 -p "$(cat real4k_ascii.txt)" -n 96 9>&- > /dev/null
    cp "/tmp/grim-vregion-$1.log" "$B/logs/diag1105-vregion-$1.log"
    echo "== verify regions $1: $(grep -oE "elapsed=[0-9.]+" "$B/logs/diag1105-vregion-$1.log" | tail -1)"
    python3 - "$B/logs/diag1105-vregion-$1.log" <<'PY'
import re, sys, collections
tabs, cur = [], None
for line in open(sys.argv[1], errors='replace'):
    m = re.match(r'\s+host region budget \((\d+) tokens', line)
    if m:
        cur = {'M': int(m.group(1)), 'r': collections.OrderedDict()}; tabs.append(cur); continue
    m = re.match(r'\s{6}(.+?)\s+([0-9.]+) ms', line)
    if cur is not None and m and not m.group(1).startswith('TOTAL'):
        cur['r'][m.group(1).strip()] = float(m.group(2))
v = [t for t in tabs if 2 <= t['M'] <= 16]
if not v:
    print('no verify tables'); sys.exit()
rows = collections.Counter(t['M'] for t in v)
acc = collections.OrderedDict()
for t in v:
    for k, x in t['r'].items():
        acc[k] = acc.get(k, 0.0) + x
n = len(v); tot = sum(acc.values()) / n
print(f'{n} verify passes (rows: {dict(rows)}), timed total {tot:.2f} ms per pass')
for k, x in sorted(acc.items(), key=lambda kv: -kv[1])[:24]:
    print(f'  {k:<30s} {x / n:8.3f} ms  {100 * x / n / tot:5.1f}%')
PY
  done
  cd "$B"
  flock -u 9
fi
exec 9>&-
echo "DIAG_END $(date +%T)"
