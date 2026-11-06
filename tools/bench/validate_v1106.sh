#!/bin/bash
# v1.10.5-rc2 validation: the image's OWN binaries, gpu0 only, one server at a time. The v1.10.4 rows with the same
# tests (r1104-img-*), plus: Flash-Next one card with the BF16 dense part and the heat engine OFF (answers must equal
# v1.10.4's), Flash-Next one card INT4 with the heat engine ON BY DEFAULT (the new one-card default; frozen hits as in
# the dev pairs), and the small-card row (Ornith NVFP4, tiered experts, 6 GB budget). The B580 is Ian's ComfyUI card and
# is never used. Nothing waits behind it.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
PD=/mnt/storage/Models/grimoire-ple
cd "$B" || exit 9
export IMG=${IMG:-grimoire-b70:v1.10.5-rc2} HOSTBIN=0 LOAD_MAX=1500 CLIENT_CPUS=14,15
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "VALIDATE_STOP no image $IMG"; exit 9; }
S="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8"
MTP="GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536"
Q=/models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16
QG=/models2/unsloth-Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
OG=/models2/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf
F=/models2/Qwen3.8-Flash-Next-NVFP4
QTOK=/mnt/models/Qwen3.8-27B-FP8
OTOK=/mnt/storage/Models/Ornith-1.5-35B-A3B
FE="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=112 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_CX_STATS=1"
FE4="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=120 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_CX_STATS=1"
BEN="--pp 4096 --tg 128"
trap 'cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
run() { # name tokenizer-dir model proj ctx env tests [benchy args]
  local name=$1 tok=$2; shift 2
  until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
  echo "== $(date +%T) $name"
  TOKDIR=$tok bash ./dual_engine_1102.sh "$name" G0 "$@" > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|answer:|answers pass|Coherence|server stopped|LEFT RUNNING|STOP|tg128|pp512" "logs/$name.run" | cut -c1-140
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "VALIDATE_STOP $name rc=$rc"; exit 9
  fi
}
# Flash-Next and the tiered row (CPU experts, the CPU-sensitive rows) first
run r1106-img-fn        ""     "$F"  bf16   8192  "$FE GRIMOIRE_EXPERT_HEAT=0" pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2
cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
run r1106-img-fn4       ""     "$F"  int4   8192  "$FE4" pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2
grep -hE "expert heat engine: (on|off)|expert heat engine: [0-9]+ steps" logs/r1106-img-fn4.server.log | tail -2 | cut -c1-160
export CODE_LENGTHS=4000 CODE_REPEATS=1
run r1106-img-split     "$OTOK" /models/Ornith-1.5-35B-A3B-NVFP4 mxfp4 8192 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_TIERED_MOE=1 GRIMOIRE_EXPERT_VRAM_GB=6 GRIMOIRE_CX_STATS=1" \
  ctest,code,benchy --pp 512 2048 --tg 128 --concurrency 1 --runs 2
unset CODE_LENGTHS CODE_REPEATS
run r1106-img-q38      ""     "$Q"  int4   16384 "$S $MTP" code,conc4,benchy $BEN --concurrency 1 4 --runs 2
run r1106-img-orn       ""     /models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE mxfp4 32000 "$S" ctest,conc4,seq4,benchy $BEN --concurrency 1 8 --runs 1
run r1106-img-qg-mtp    "$QTOK" "$QG" native 16384 "$S $MTP" code,conc4,benchy $BEN --concurrency 1 4 --runs 2
run r1106-img-og-native "$OTOK" "$OG" native 32000 "$S" ctest,conc4,seq4,benchy $BEN --concurrency 1 8 --runs 1
run r1106-img-og-mxfp4  "$OTOK" "$OG" mxfp4  32000 "$S" ctest,conc4,seq4,benchy $BEN --concurrency 1 8 --runs 1
echo "VALIDATE_RUNS_DONE $(date +%T)"
python3 "$B/verify_image_logs_v1106.py" "$B/logs"
