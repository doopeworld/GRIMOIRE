#!/bin/bash
# v1.10.4 combined gate, continued (11-05). run_v1104b_all.sh stopped at 11:26 because the candidate's two-card
# Flash-Next text with fp32 partials differed from v1.10.3; run_fn_det_1105.sh then measured whether two-card
# Flash-Next text repeats at all. Claude writes CONTINUE (with the reason) into logs/fn-det-decision.txt after reading
# it; anything else stops here. Everything before that step passed and is not rerun.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105
exec >>"$B/logs/v1104b-all.out" 2>&1
stop() { echo "V1104B_ALL_STOP $1 $(date +%T)"; exit 9; }
echo "V1104B_ALL_CONTINUE $(date +%T)"
until grep -q "^FN_DET_DONE" "$B/logs/fn-det-1105.out" 2>/dev/null; do sleep 15; done
until [ -e "$B/logs/fn-det-decision.txt" ]; do sleep 15; done
head -1 "$B/logs/fn-det-decision.txt" | grep -q "^CONTINUE" || stop "two-card Flash-Next: $(head -1 "$B/logs/fn-det-decision.txt")"
cat "$B/logs/fn-det-decision.txt"
until [ -z "$(docker ps -q)" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
# the peer partial is lossless: the same Qwen answers with it off
DUAL_EXTRA=GRIMOIRE_TP_SMALL_PEER=0 GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:v1.10.3 \
  bash "$B/run_dual_1104.sh" v1104np qwen > "$B/logs/dual-v1104np-qwen.out" 2>&1 || stop "two-card Qwen, peer partial off"
cat "$B/logs/dual-v1104np-qwen.out"
python3 - "$B/logs" <<'PY' || stop "two-card Qwen answers differ with the peer partial"
import json, sys
d = sys.argv[1]
a = [json.loads(l) for l in open(f'{d}/d-v1104np-qwen-tp.code.json') if l.strip()]
b = [json.loads(l) for l in open(f'{d}/d-v1104-qwen-tp.code.json') if l.strip()]
ca, cb = (json.load(open(f'{d}/d-{t}-qwen-tp.conc4.json')) for t in ('v1104np', 'v1104'))
s1 = sum(x['text'] == y['text'] for x, y in zip(a, b))
s2 = sum(ca[k]['text'] == cb.get(k, {}).get('text') for k in ca)
print(f'two cards, Qwen FP8 + MTP, peer partial off vs on: coding identical {s1}/{len(a)}, concurrent identical {s2}/{len(ca)}')
sys.exit(0 if s1 == len(a) == len(b) and s2 == len(ca) else 1)
PY
GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 \
  bash "$B/dual_engine_1102.sh" d1104b-gptq-tp TP /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536" \
  code,conc4,benchy --pp 4096 --tg 128 --concurrency 1 4 --runs 2 > "$B/logs/d1104b-gptq-tp.run" 2>&1 || stop "two-card GPTQ"
grep -E "ready=|answers pass|Coherence|tests=|server stopped|LEFT RUNNING|STOP" "$B/logs/d1104b-gptq-tp.run" | cut -c1-150
grep -q "server stopped gracefully" "$B/logs/d1104b-gptq-tp.run" || stop "two-card GPTQ did not stop gracefully"

# 3. full regression (text differences listed, not gated)
bash "$B/run_regress_v1104b.sh"; rc=$?
bash "$B/regress_compare_v1104b.sh" > "$B/logs/regress-compare-v1104b.out" 2>&1
cat "$B/logs/regress-compare-v1104b.out"
[ "$rc" = 0 ] || stop "regression rc=$rc"
grep -q "^EMPTY_ANSWER" "$B/logs/regress-compare-v1104b.out" && stop "regression: empty answer"

# 4. image
bash "$B/assemble-image-v1104b.sh" > "$B/logs/assemble-v1104b.out" 2>&1 || stop "assembly"
tail -2 "$B/logs/assemble-v1104b.out"
bash "$B/validate_v1104.sh" > "$B/logs/validate-v1104.out" 2>&1
cat "$B/logs/validate-v1104.out"
grep -qx IMAGE_LOG_CHECK_PASS "$B/logs/validate-v1104.out" || stop "image validation"
echo "V1104B_ALL_DONE $(date +%T)"
