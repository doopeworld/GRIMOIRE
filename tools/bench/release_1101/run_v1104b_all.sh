#!/bin/bash
# v1.10.4 gate, combined candidate (11-05, Ian away): mxfp4-smallm-1105 dd33cbe = 35201c2 (CPU-expert pool fix, bf16
# two-card exchange, expert budget) + MXFP4 small-M kernels (verify / batched decode, one card) + 4-bit FFN weights on
# the two-card prompt wavefront + the peer's bf16 partial in the two-card small all-reduce. Built in
# grimoire-release-1105 (G31+G21, libgrimoire_gemm.so rebuilt). One GPU job at a time; stops at the first failure.
#   1. one card: served Flash-Next quiet / 4 busy (v1.10.3 code vs candidate), Ornith GGUF default grouping
#   2. two cards: Flash-Next v1.10.3 vs candidate (bf16 text may differ: then fp32 partials must equal v1.10.3);
#      Qwen3.8-27B-FP8 + MTP with and without the peer partial: every answer byte-identical (lossless claim);
#      Qwen3.8-27B GPTQ + MTP (the release table)
#   3. the full 44-case regression against 11-02's v1.10.3 run (text differences are listed for review, not gated:
#      MXFP4 speculation changes its summation order)
#   4. assemble v1.10.4-rc1 (v1.10.3 runtime + new executables + new gemm library) and validate its own binaries
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105
exec >>"$B/logs/v1104b-all.out" 2>&1
stop() { echo "V1104B_ALL_STOP $1 $(date +%T)"; exit 9; }
echo "V1104B_ALL_START $(date +%T)"
until grep -qE "^RELEASE_BUILD_(PASS|FAIL)" "$B/logs/release-build-1105.out" 2>/dev/null; do sleep 30; done
grep -qx RELEASE_BUILD_PASS "$B/logs/release-build-1105.out" || stop "build failed"
until [ -z "$(docker ps -q)" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
echo "candidate $(git -C "$T" rev-parse --short HEAD) $(date +%T)"

# 1. one card
SKIP_MATRIX=1 NEW_TREE=$T BUILD_LOG=release-build-1105.out bash "$B/run_v1104_checks.sh"; rc=$?
grep -E "^\||IDENTICAL|DIFFERS|identical answers|tg128|V1104_CHECKS_END" "$B/logs/v1104-checks.out" | cut -c1-160
[ "$rc" = 0 ] || stop "one-card checks rc=$rc"

# 2. two cards
IMG=grimoire-b70:v1.10.3 HOSTBIN=0 bash "$B/run_dual_1104.sh" v1103 fn > "$B/logs/dual-v1103-fn.out" 2>&1 || stop "two-card Flash-Next on v1.10.3"
cat "$B/logs/dual-v1103-fn.out"
GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:v1.10.3 bash "$B/run_dual_1104.sh" v1104 fn qwen > "$B/logs/dual-v1104-fn.out" 2>&1 ||
  stop "two-card Flash-Next / Qwen on the candidate"
cat "$B/logs/dual-v1104-fn.out"
same() { # TAG: the candidate's two-card Flash-Next answers vs v1.10.3's
python3 - "$B/logs/d-v1103-fn-tp.gen.json" "$B/logs/d-$1-fn-tp.gen.json" "$1" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = sum(a[k]["text"] == b.get(k, {}).get("text") for k in a)
for k in a:
    print(k, "two cards, v1.10.3 vs", sys.argv[3] + ":", "IDENTICAL" if a[k]["text"] == b[k]["text"] else "DIFFERS",
          "seconds %.2f -> %.2f" % (a[k]["seconds"], b[k]["seconds"]))
sys.exit(0 if same == len(a) else 1)
PY
}
if ! same v1104; then
  # bf16 partials may move a near-tie token; the same binary with fp32 partials must equal v1.10.3 exactly
  DUAL_EXTRA=GRIMOIRE_TP_SMALL_BF16=0 GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:v1.10.3 \
    bash "$B/run_dual_1104.sh" v1104f32 fn > "$B/logs/dual-v1104f32-fn.out" 2>&1 || stop "two-card Flash-Next, fp32 partials"
  cat "$B/logs/dual-v1104f32-fn.out"
  same v1104f32 || stop "two-card Flash-Next answers differ with fp32 partials too"
  echo "TWO_CARD_BF16_TEXT_DIFFERS: review logs/d-v1104-fn-tp.gen.json against logs/d-v1103-fn-tp.gen.json"
fi
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
