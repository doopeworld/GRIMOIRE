#!/bin/bash
# 11-05: two B70s, Megatron TP, the grimoire-mx-smallm-1105 server: 4-bit FFN weights on the TP prompt wavefront
# (default) vs the old per-layer path (GRIMOIRE_TP_WAVE_4BIT=0), Qwen3.8-27B GPTQ-Int4 + MTP with the one-card
# release row's settings; then Qwen3.8-27B MXFP4 + MTP on two cards (wavefront), coding answers + pp4096.
# One server at a time; stops at the first failure. Called by run_mx_ab_1105.sh before the v1.10.4 gate.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
export GRIM_TREE=/mnt/storage/isos/grimoire-mx-smallm-1105 HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
run() { # name model proj ctx env tests benchy...
  local name=$1; shift
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" TP "$@" > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|answers pass|Coherence|tests=|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-150
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "WAVEAB_STOP $name rc=$rc"; return 9
  fi
}
Q="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536"
G=/models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16
X=/models2/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16
run w1105-gptq-wave   "$G" int4 16384 "$Q" code,conc4,benchy --pp 4096 --tg 128 --concurrency 1 4 --runs 2 || exit 9
run w1105-gptq-layer  "$G" int4 16384 "$Q GRIMOIRE_TP_WAVE_4BIT=0" code,conc4,benchy --pp 4096 --tg 128 --concurrency 1 4 --runs 2 || exit 9
run w1105-mxfp4-wave  "$X" mxfp4 16384 "$Q" code,benchy --pp 4096 --tg 128 --concurrency 1 --runs 2 || exit 9
python3 "$B/compare_ab_1105.py" "$B/logs" per-layer=w1105-gptq-layer wavefront=w1105-gptq-wave
python3 "$B/compare_ab_1105.py" "$B/logs" mxfp4-two-cards=w1105-mxfp4-wave
python3 - "$B/logs" <<'PY'
import json, sys
d = sys.argv[1]
a = [json.loads(l) for l in open(f'{d}/w1105-gptq-layer.code.json') if l.strip()]
b = [json.loads(l) for l in open(f'{d}/w1105-gptq-wave.code.json') if l.strip()]
same = sum(x.get('text') == y.get('text') for x, y in zip(a, b))
print(f'GPTQ two cards, coding answers per-layer vs wavefront: identical text {same}/{min(len(a), len(b))}')
PY
echo "WAVEAB_DONE $(date +%T)"
