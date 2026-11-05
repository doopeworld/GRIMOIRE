#!/bin/bash
# 11-05, after v1.10.4 is published (start by hand): environment-only experiments on the release build
# (grimoire-release-1105), one GPU job at a time, plus the INT4 split-plan sweep.
#   1. INT4 small-M plan sweep (run_i4_sweep_1105.sh; needs I4PROBE_BUILD_PASS)
#   2. Burst admission: Qwen3.8-27B MXFP4 + MTP, 4 users x 16K prompts, three benchy runs each with the default 8 ms
#      hold and with GRIMOIRE_ADMIT_HOLD_MS=100; GRIMOIRE_BATCH_HOST_TIMING=1 logs every admission batch
#   3. Two cards, GPTQ + MTP: small all-reduce limit 16 rows (81,920 floats) with the peer partial, vs the default
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105
cd "$B" || exit 9
exec >>"$B/logs/after-publish-exp-1105.out" 2>&1
echo "AFTER_PUBLISH_EXP_START $(date +%T)"
until [ -z "$(docker ps -q)" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
if grep -qx I4PROBE_BUILD_PASS logs/build-i4probe-1105.out 2>/dev/null; then
  bash "$B/run_i4_sweep_1105.sh"; tail -1 logs/i4-sweep-1105.out
fi
# GGUF native small-M split plan (tools/bench_native_smallm; M <= 8 is the engine's 128-register kernel, its 9-16
# rows here are a 128-register copy, not the library's)
if grep -qx GGUFBENCH_BUILD_PASS logs/build-ggufbench-1105.out 2>/dev/null; then
  touch "$B/logs/HOLD-gguf"
  (
    exec 9>/mnt/storage/isos/grimoire-runs/gpu0.lock
    flock -n 9 || exit 75
    cd /mnt/storage/isos/grimoire-fuse || exit 9
    for ks in 0 1 2 4 8 12 16 20 24; do
      ex=""; [ "$ks" != 0 ] && ex="GRIMOIRE_GGUF_SMALLM_KS=$ks"
      echo "== GGUF KS=$ks $(date +%T)"
      EXTRA_ENV="$ex" GRIM_TREE=/mnt/storage/isos/grimoire-mx-lib-1105 LIM=600 TAILN=100 \
        bash tools/g0run.sh "gguf-ks$ks" /grimoire/bin/bench_native_smallm 9>&- > /dev/null
      grep -E "M=(5|8) " "/tmp/grim-gguf-ks$ks.log" | cut -c1-120
      grep -E "^(OK|FAIL)" "/tmp/grim-gguf-ks$ks.log"
    done
  )
  rm -f "$B/logs/HOLD-gguf"
fi
# B580: the tiered-expert CPU split in decode (build_split_g21_1105.sh), against today's v1.10.4 run b5-v1104-grimoire
until grep -qE "^SPLIT_BUILD_(PASS|FAIL)" logs/build-split-g21-1105.out 2>/dev/null; do sleep 20; done
if grep -qx SPLIT_BUILD_PASS logs/build-split-g21-1105.out; then
  echo "== $(date +%T) B580 CPU split"
  GTREE=/mnt/storage/isos/grimoire-runs/split-tree-1105 NCPU=26 bash "$B/run_b580_1104.sh" split grimoire grimoire-default
fi
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
Q="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=4 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_BATCH_HOST_TIMING=1"
for spec in "hold8 " "hold100 GRIMOIRE_ADMIT_HOLD_MS=100"; do
  set -- $spec
  name=exp1105-burst-$1
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" G0 /models2/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16 mxfp4 36864 "$Q ${2:-}" \
    benchy --pp 16384 --tg 128 --concurrency 4 --runs 3 > "logs/$name.run" 2>&1
  grep -E "c4\)" "logs/$name.benchy.md" | cut -c1-140
  grep -c "scheduler admit batch" "logs/$name.server.log" 2>/dev/null
  grep "scheduler admit batch" "logs/$name.server.log" 2>/dev/null | head -8 | cut -c1-140
done
# where a speculative step's time goes now (draft / verify / commit, host wall time per step), one card, coding
for spec in "mxfp4 /models2/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16 mxfp4" "swift /models2/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8 int4"; do
  set -- $spec
  name=exp1105-spect-$1
  echo "== $(date +%T) $name"
  CODE_LENGTHS=4000 CODE_REPEATS=1 bash ./dual_engine_1102.sh "$name" G0 "$2" "$3" 36864 \
    "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=4 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_SPEC_TIMING=1" \
    code > "logs/$name.run" 2>&1
  grep -E "code\]  " "logs/$name.run" | cut -c1-150
  grep -iE "spec.*(draft|verify)" "logs/$name.server.log" | tail -3 | cut -c1-220
done
G="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536"
for spec in "sm64k " "sm16rows GRIMOIRE_TP_SHM_CPU_MAX=81920"; do
  set -- $spec
  name=exp1105-tp-$1
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" TP /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 "$G ${2:-}" \
    conc4,benchy --pp 4096 --tg 128 --concurrency 1 2 4 --runs 2 > "logs/$name.run" 2>&1
  grep -E "CONC|Coherence" "logs/$name.run" | cut -c1-120
  grep -E "tg128" "logs/$name.benchy.md" | cut -c1-140
  grep -E "TP small all-reduce \(BAR" "logs/$name.server.log" | tail -2 | cut -c1-240
done
echo "AFTER_PUBLISH_EXP_END $(date +%T)"
