#!/bin/bash
# v1.10.5 gate (11-05): candidate a3f81a1 in grimoire-release-1105b (G31+G21) = v1.10.4 + the CPU split in tiered-expert
# decode (23f2a24) + small-M split plans (a3f81a1). One GPU job at a time, the two B70s only; stops at the first failure.
#   0. waits for the served A/B (run_next_ab_1105.sh) and Claude's review of it: logs/v1105-decision.txt, first line
#      CONTINUE (anything else stops)
#   1. two cards, Qwen3.8-27B GPTQ + MTP (Megatron runs the INT4 small-M plans): code, 4 at once, llama-benchy, against
#      v1.10.4's gate run d1104b-gptq-tp
#   2. the full 44-case regression against v1.10.4's (regress-1105-1154; text differences listed for review: the split
#      plans change summation order)
#   3. assemble v1.10.5-rc1 (v1.10.4 runtime + new executables + new gemm library) and validate its own binaries
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105b
exec >>"$B/logs/v1105-all.out" 2>&1
stop() { echo "V1105_ALL_STOP $1 $(date +%T)"; exit 9; }
echo "V1105_ALL_START $(date +%T)"
until grep -qE "^NEXT_AB_(DONE|STOP)" "$B/logs/next-ab-1105.out" 2>/dev/null; do sleep 30; done
grep -q "^NEXT_AB_DONE" "$B/logs/next-ab-1105.out" || stop "served A/B stopped"
until [ -e "$B/logs/v1105-decision.txt" ]; do sleep 30; done
head -1 "$B/logs/v1105-decision.txt" | grep -q "^CONTINUE" || stop "decision: $(head -1 "$B/logs/v1105-decision.txt")"
cat "$B/logs/v1105-decision.txt"
waitq() { until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done; }
waitq
echo "candidate $(git -C "$T" rev-parse --short HEAD) $(date +%T)"

# 1. two cards
GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 \
  bash "$B/dual_engine_1102.sh" d1105-gptq-tp TP /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536" \
  code,conc4,benchy --pp 4096 --tg 128 --concurrency 1 4 --runs 2 > "$B/logs/d1105-gptq-tp.run" 2>&1 || stop "two-card GPTQ"
grep -E "ready=|answers pass|Coherence|tests=|server stopped|LEFT RUNNING|STOP" "$B/logs/d1105-gptq-tp.run" | cut -c1-150
grep -q "server stopped gracefully" "$B/logs/d1105-gptq-tp.run" || stop "two-card GPTQ did not stop gracefully"
python3 "$B/compare_ab_1105.py" "$B/logs" v1.10.4=d1104b-gptq-tp candidate=d1105-gptq-tp

# 2. full regression (text differences listed, not gated)
waitq
bash "$B/run_regress_v1105.sh"; rc=$?
bash "$B/regress_compare_v1105.sh" > "$B/logs/regress-compare-v1105.out" 2>&1
cat "$B/logs/regress-compare-v1105.out"
[ "$rc" = 0 ] || stop "regression rc=$rc"
grep -q "^EMPTY_ANSWER" "$B/logs/regress-compare-v1105.out" && stop "regression: empty answer"

# 3. image
bash "$B/assemble-image-v1105.sh" > "$B/logs/assemble-v1105.out" 2>&1 || stop "assembly"
tail -2 "$B/logs/assemble-v1105.out"
bash "$B/validate_v1105.sh" > "$B/logs/validate-v1105.out" 2>&1
cat "$B/logs/validate-v1105.out"
grep -qx IMAGE_LOG_CHECK_PASS "$B/logs/validate-v1105.out" || stop "image validation"
echo "V1105_ALL_DONE $(date +%T)"
