#!/bin/bash
# 11-05 17:20 (Ian: "resume"): the remaining environment-only experiments on v1.10.4's release build
# (grimoire-release-1105), one GPU job at a time, the two B70s only (the B580 is Ian's ComfyUI card: never used).
#   1. burst admission: Qwen3.8-27B MXFP4 + MTP, 4 users x 16K prompts, three benchy runs each with the default 8 ms
#      hold and with GRIMOIRE_ADMIT_HOLD_MS=100; GRIMOIRE_BATCH_HOST_TIMING=1 logs every admission batch
#   2. where a speculative step's time goes (GRIMOIRE_SPEC_TIMING=1): MXFP4 vs Swift INT4, one card, coding
#   3. two cards, GPTQ + MTP: small all-reduce limit 16 rows (81,920 floats) with the peer partial, vs the default --
#      after the CPU build of the next release (run_release_build_v1105b.sh) ends: that all-reduce runs on the CPU
# Every run waits until no other container but a CPU build runs and no logs/HOLD-* exists.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105
cd "$B" || exit 9
exec >>"$B/logs/resume-exp-1105.out" 2>&1
echo "RESUME_EXP_START $(date +%T) (restart: hold100, spec timing, two-card)"
waitq() { until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done; }
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
Q="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=4 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_BATCH_HOST_TIMING=1"
for spec in "hold100 GRIMOIRE_ADMIT_HOLD_MS=100"; do   # hold8 ran 17:20 (first queue)
  set -- $spec
  name=exp1105-burst-$1
  waitq; echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" G0 /models2/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16 mxfp4 36864 "$Q ${2:-}" \
    benchy --pp 16384 --tg 128 --concurrency 4 --runs 3 > "logs/$name.run" 2>&1
  echo "rc=$?"; grep -E "STOP|refusing|LEFT RUNNING|server stopped" "logs/$name.run" | head -3
  grep -E "c4\)" "logs/$name.benchy.md" | cut -c1-140
  grep -c "scheduler admit batch" "logs/$name.server.log" 2>/dev/null
  grep "scheduler admit batch" "logs/$name.server.log" 2>/dev/null | head -12 | cut -c1-140
done
for spec in "mxfp4 /models2/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16 mxfp4" "swift /models2/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8 int4"; do
  set -- $spec
  name=exp1105-spect-$1
  waitq; echo "== $(date +%T) $name"
  CODE_LENGTHS=4000 CODE_REPEATS=1 bash ./dual_engine_1102.sh "$name" G0 "$2" "$3" 36864 \
    "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=4 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_SPEC_TIMING=1" \
    code > "logs/$name.run" 2>&1
  echo "rc=$?"; grep -E "STOP|refusing|LEFT RUNNING|server stopped" "logs/$name.run" | head -3
  grep -E "code\]  " "logs/$name.run" | cut -c1-150
  grep -E "^spec timing:" "logs/$name.server.log" | tail -4 | cut -c1-220
done
until grep -qE "^RELEASE_BUILD_(PASS|FAIL)" logs/release-build-1105b.out 2>/dev/null; do sleep 20; done
G="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536"
for spec in "sm64k " "sm16rows GRIMOIRE_TP_SHM_CPU_MAX=81920"; do
  set -- $spec
  name=exp1105-tp-$1
  waitq; echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" TP /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 "$G ${2:-}" \
    conc4,benchy --pp 4096 --tg 128 --concurrency 1 2 4 --runs 2 > "logs/$name.run" 2>&1
  echo "rc=$?"; grep -E "STOP|refusing|LEFT RUNNING|server stopped" "logs/$name.run" | head -3
  grep -E "CONC|Coherence" "logs/$name.run" | cut -c1-120
  grep -E "tg128" "logs/$name.benchy.md" | cut -c1-140
  grep -E "TP small all-reduce \(BAR" "logs/$name.server.log" | tail -2 | cut -c1-240
done
echo "RESUME_EXP_END $(date +%T)"
