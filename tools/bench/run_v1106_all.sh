#!/bin/bash
# v1.10.5-rc2 gate (11-06): candidate WANT in grimoire-release-1106 (G31+G21, run_release_build_v1106.sh). One GPU job at
# a time, the two B70s only (gpu1 at Gen3 x4 today; the link guard of dual_engine_1102.sh stops on uncorrectable errors);
# stops at the first failure.
#   1. two cards: Qwen3.8-27B GPTQ + MTP (against v1.10.4's gate run d1104b-gptq-tp); Qwen3.8-27B FP8 checkpoint as
#      --proj int8 + DFlash2 (the recommended DUAL-QWEN3.8 setup; dev build q27g-b: code 130.4 / 130.9); Flash-Next
#      INT8 170 a layer, plain (heat engine stays off with several ranks; 11-06: tg128 52.9)
#   2. the full 44-case regression against v1.10.4's (regress-1105-1154; text differences listed for review -- the
#      Flash-Next cases run with the heat engine now on by default and are expected to reword near ties)
#   3. assemble v1.10.5-rc2 (v1.10.4 runtime + new executables + new gemm library) and validate its own binaries
#   run_v1106_all.sh WANT
set -u
WANT=${1:?commit the release tree must be at}
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1106
PD=/mnt/storage/Models/grimoire-ple
exec >>"$B/logs/v1106-all.out" 2>&1
stop() { echo "V1106_ALL_STOP $1 $(date +%T)"; exit 9; }
echo "V1106_ALL_START $(date +%T) want=$WANT"
until grep -q "source=$WANT" "$B/logs/release-build-1106.out" 2>/dev/null &&
      grep -qE "^RELEASE_BUILD_(PASS|FAIL)" "$B/logs/release-build-1106.out"; do sleep 30; done
grep -qx RELEASE_BUILD_PASS "$B/logs/release-build-1106.out" || stop "release build failed"
waitq() { until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done; }
waitq
touch "$B/logs/HOLD-v1106"
trap 'rm -f "$B/logs/HOLD-v1106"; cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
echo "candidate $(git -C "$T" rev-parse --short HEAD) $(date +%T)"
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
dual() { # name model proj ctx env tests [args]
  local name=$1; shift
  echo "== $(date +%T) $name"
  bash "$B/dual_engine_1102.sh" "$name" TP "$@" > "$B/logs/$name.run" 2>&1 || stop "$name rc=$?"
  grep -E "ready=|answer:|answers pass|Coherence|tests=|tg128|pp512|pp4096|server stopped|LEFT RUNNING|STOP|RxErr during" \
    "$B/logs/$name.run" | cut -c1-150
  grep -q "server stopped gracefully" "$B/logs/$name.run" || stop "$name did not stop gracefully"
  [ -e "$B/logs/$name.STOP" ] && stop "$name link guard"
  return 0
}

# 1. two cards
cd "$B" || stop "no bench dir"
dual d1106-gptq-tp /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536" \
  code,conc4,benchy --pp 4096 --tg 128 --concurrency 1 4 --runs 2
python3 "$B/compare_ab_1105.py" "$B/logs" v1.10.4=d1104b-gptq-tp candidate=d1106-gptq-tp
CODE_LENGTHS=4000 CODE_REPEATS=1 dual d1106-q27-df2 /models2/Qwen3.8-27B-FP8 int8 16384 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_SPEC_STATS=1" \
  code,benchy --pp 512 4096 --tg 128 --concurrency 1 --runs 2
grep -hE "spec: DFlash2" "$B/logs/d1106-q27-df2.server.log" | grep rank0 | tail -1 | cut -c1-150
cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
dual d1106-fn-tp /models2/Qwen3.8-Flash-Next-NVFP4 int8 8192 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=170 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1" \
  pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2
grep -hE "expert heat engine" "$B/logs/d1106-fn-tp.server.log" | head -2 | cut -c1-150
rm -f "$B/logs/HOLD-v1106"

# 2. full regression (text differences listed, not gated)
waitq
bash "$B/run_regress_v1106.sh"; rc=$?
bash "$B/regress_compare_v1106.sh" > "$B/logs/regress-compare-v1106.out" 2>&1
cat "$B/logs/regress-compare-v1106.out"
[ "$rc" = 0 ] || stop "regression rc=$rc"
grep -q "^EMPTY_ANSWER" "$B/logs/regress-compare-v1106.out" && stop "regression: empty answer"

# 3. image
bash "$B/assemble-image-v1106.sh" "$WANT" > "$B/logs/assemble-v1106.out" 2>&1 || stop "assembly"
tail -2 "$B/logs/assemble-v1106.out"
bash "$B/validate_v1106.sh" > "$B/logs/validate-v1106.out" 2>&1
cat "$B/logs/validate-v1106.out"
grep -qx IMAGE_LOG_CHECK_PASS "$B/logs/validate-v1106.out" || stop "image validation"
echo "V1106_ALL_DONE $(date +%T)"
