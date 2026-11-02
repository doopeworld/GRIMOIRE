#!/bin/bash
# v1.10.3-rc1 validation: the image's OWN binaries, gpu0 only, one server at a time.
# The five v1.10.2 configurations with the same tests, then each new option, then its paired baseline.
# Added to the Ornith runs: conc4 + seq4 = the same four coding prompts together and one at a time, on
# three weight formats, to tell a model mistake from a batching effect (none of it gates the release).
# Start by hand after the regression and the release build have ended; nothing waits behind it.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd "$B" || exit 9
export IMG=${IMG:-grimoire-b70:v1.10.3-rc1} HOSTBIN=0 LOAD_MAX=1500 CLIENT_CPUS=14,15
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "VALIDATE_STOP no image $IMG"; exit 9; }
S="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8"
MTP="GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536"
Q=/models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16
QG=/models2/unsloth-Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
OG=/models2/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf
QTOK=/mnt/models/Qwen3.8-27B-FP8
OTOK=/mnt/storage/Models/Ornith-1.5-35B-A3B
BEN="--pp 4096 --tg 128"
run() { # name tokenizer-dir model proj ctx env tests [benchy args]
  local name=$1 tok=$2; shift 2
  echo "== $(date +%T) $name"
  TOKDIR=$tok bash ./dual_engine_1102.sh "$name" G0 "$@" > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|answer:|answers pass|Coherence|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-140
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "VALIDATE_STOP $name rc=$rc"; exit 9
  fi
}
run r1102-img-q38       ""     "$Q"  int4   16384 "$S $MTP" code,conc4,benchy $BEN --concurrency 1 4 --runs 2
run r1102-img-orn       ""     /models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE mxfp4 32000 "$S" ctest,conc4,seq4,benchy $BEN --concurrency 1 8 --runs 1
run r1102-img-qg-mtp    "$QTOK" "$QG" native 16384 "$S $MTP" code,conc4,benchy $BEN --concurrency 1 4 --runs 2
run r1102-img-og-native "$OTOK" "$OG" native 32000 "$S" ctest,conc4,seq4,benchy $BEN --concurrency 1 8 --runs 1
run r1102-img-og-mxfp4  "$OTOK" "$OG" mxfp4  32000 "$S" ctest,conc4,seq4,benchy $BEN --concurrency 1 8 --runs 1
run r1102-opt-moe       "$OTOK" "$OG" native 32000 "$S GRIMOIRE_MOE_NATIVE_SMALLM=1" ctest,conc4,seq4,benchy $BEN --concurrency 1 8 --runs 1
run r1102-opt-mtp       ""     "$Q"  int4   16384 "$S $MTP GRIMOIRE_MTP_GPU_DRAFT=1" ctest,code,conc4,ragged,benchy $BEN --concurrency 1 4 --runs 2
W="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=1 GRIMOIRE_MTP=0"
run r1102-opt-w8-off    ""     /models2/Qwen3.8-27B-FP8 int8 8192 "$W" ctest,code,prose,benchy $BEN --concurrency 1 --runs 3
run r1102-opt-w8-wide   ""     /models2/Qwen3.8-27B-FP8 int8 8192 "$W GRIMOIRE_ONEDNN_W8A8=3" ctest,code,prose,benchy $BEN --concurrency 1 --runs 3
echo "VALIDATE_RUNS_DONE $(date +%T)"
python3 "$B/verify_image_logs_1102.py" "$B/logs"
