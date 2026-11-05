#!/bin/bash
# 11-05 17:40: diagnostics on gpu0 while the CPU-only release build runs (cores 6-13; the GPUs are idle until it ends),
# v1.10.4's release build (grimoire-release-1105). Holds logs/HOLD-diagnow so the queued two-card experiments wait.
#   1. where one verify pass (7 rows: MTP K=6) spends its time, per region summed over all 64 layers: CLI, the 6K
#      Sherlock prompt, 96 tokens, GRIMOIRE_TIME_LAYER=all (drains the queue after every region: region = its kernels +
#      launch latency, so small regions read high). MXFP4 and Swift INT4.
#   2. the speculative step split at 4 users, one card (GRIMOIRE_SPEC_TIMING=1, 4 coding answers at once), Qwen3.8-27B
#      GPTQ + MTP: the one-card side of the two-card 4-user question (run_diag_1105.sh runs both after the A/B)
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105
cd "$B" || exit 9
exec >>"$B/logs/diag-1105.out" 2>&1
echo "DIAGNOW_START $(date +%T)"
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-diagnow
trap 'rm -f "$B/logs/HOLD-diagnow"' EXIT
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
print(f'{len(tabs)} timed prefill calls, {len(v)} of 2-16 rows')
if v:
    rows = collections.Counter(t['M'] for t in v)
    acc = collections.OrderedDict()
    for t in v:
        for k, x in t['r'].items():
            acc[k] = acc.get(k, 0.0) + x
    n = len(v); tot = sum(acc.values()) / n
    print(f'{n} verify passes (rows: {dict(rows)}), timed total {tot:.2f} ms per pass')
    for k, x in sorted(acc.items(), key=lambda kv: -kv[1])[:26]:
        print(f'  {k:<30s} {x / n:8.3f} ms  {100 * x / n / tot:5.1f}%')
PY
  done
  cd "$B"
  flock -u 9
else
  echo "gpu0.lock held: no verify region runs"
fi
exec 9>&-
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
name=diag1105-g0-v1104
echo "== $(date +%T) $name"
bash ./dual_engine_1102.sh "$name" G0 /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_SPEC_TIMING=1" \
  conc4 > "logs/$name.run" 2>&1
echo "rc=$?"; grep -E "CONC|STOP|refusing|LEFT RUNNING|server stopped" "logs/$name.run" | head -4 | cut -c1-150
grep -E "^spec timing:" "logs/$name.server.log" | cut -c1-230
echo "DIAGNOW_END $(date +%T)"
