#!/bin/bash
# Ian, 11-04: Qwen3.8-27B converted from BF16 to MXFP4 by GRIMOIRE vs Wrapzii's Swift-1.5 GPTQ-Int4 (baked), which is
# faster at pp4096 / 8K / 16K / 32K and tg128, 1 / 2 / 4 users, every answer checked. One B70 (gpu0), the published
# v1.10.3 image's own binaries, the same settings for both: MTP K=6, draft vocabulary 65,536, 4 sequence slots,
# --ctx 36864. Per model, in order: Paris; the coding probe at 4K / 8K / 16K / 32K-token prompts (fresh + repeat,
# unit-tested answers); 4 coding answers at once (two rounds); llama-benchy (its coherence check first).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
# wait for the Swift download and for the GPU
until grep -qE "^SWIFT_FETCH_(DONE|FAIL)" logs/fetch-swift-1104.out; do sleep 10; done
grep -qx SWIFT_FETCH_DONE logs/fetch-swift-1104.out || { echo "MS_STOP Swift download failed"; exit 9; }
until [ -z "$(docker ps -q)" ] && ! pgrep -f "^bash $B/run_pin_matrix_1102.sh" >/dev/null; do sleep 10; done
export IMG=grimoire-b70:v1.10.3 HOSTBIN=0 LOAD_MAX=1500 CLIENT_CPUS=14,15 CODE_LENGTHS=4000,8000,16000,32000 CODE_REPEATS=1
E="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=4 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536"
BEN="--pp 4096 8192 16384 32768 --tg 128 --concurrency 1 2 4 --runs 2"
run() { # name model proj
  local name=$1 model=$2 proj=$3
  echo "== $(date +%T) $name $model --proj $proj"
  bash ./dual_engine_1102.sh "$name" G0 "$model" "$proj" 36864 "$E" ctest,code,conc4,benchy $BEN > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|answer:|answers pass|Coherence|tests=|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-150
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "MS_STOP $name rc=$rc"; exit 9
  fi
}
run ms1104-mxfp4 /models2/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16 mxfp4
run ms1104-swift /models2/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8 int4
python3 "$B/compare_mxfp4_swift_1104.py" "$B/logs"
echo "MS_DONE $(date +%T)"
