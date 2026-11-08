#!/bin/bash
# Resume only missing candidate-image checks for Qwen3.8-27B and Flash-Next.
# Preserve all 11-06 artifacts; no rebuild, broad regression, deployment or publishing.
set -euo pipefail
B=/mnt/storage/isos/grimoire-runs/bench-1022
PD=/mnt/storage/Models/grimoire-ple
cd "$B"
exec >>logs/resume-1108.out 2>&1
exec 6>logs/resume-1108.lock
flock -n 6 || exit 9
export IMG=grimoire-b70:v1.10.5-rc2 HOSTBIN=0 LOAD_MAX=1500 CLIENT_CPUS=14,15
export GRIM_TREE=/mnt/storage/isos/grimoire-release-1106
test "$(docker image inspect "$IMG" --format '{{.Id}}')" = sha256:717960860fad7cd194ef9a7bf6d35787059f7c7dc084e8b04348c9290b112210
echo "RESUME_START $(date -Iseconds) image=$IMG gpu1=Gen4x4"
trap 'cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
run() {
  local name=$1; shift
  if [ -f "logs/$name.VERIFIED" ]; then echo "SKIP verified $name"; return; fi
  # Never overwrite a partial attempt: inspect it before choosing a new run name.
  test ! -e "logs/$name.run"
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "CASE_START $name $(date -Iseconds)"
  bash ./dual_engine_1102.sh "$name" "$@" >"logs/$name.run" 2>&1 || { echo "RESUME_STOP $name harness"; exit 9; }
  python3 ./verify_resume_1108.py "$B/logs/$name"
  touch "logs/$name.VERIFIED"
  grep -E 'ready=|answers pass|tests=|Coherence|tg128|pp512|pp2048|pp4096|server stopped' "logs/$name.run" | cut -c1-180
  echo "CASE_PASS $name $(date -Iseconds)"
}
F=/models2/Qwen3.8-Flash-Next-NVFP4
FE="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_CX_STATS=1"
FH="$FE GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab"
run r1108-img-fn G0 "$F" bf16 8192 "$FE GRIMOIRE_EXPERT_VRAM_PER_LAYER=112 GRIMOIRE_EXPERT_HEAT=0" pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2
run r1108-img-fn4 G0 "$F" int4 8192 "$FH GRIMOIRE_EXPERT_VRAM_PER_LAYER=120" pleready,gen,benchy --pp 512 --tg 128 --concurrency 1 --runs 2
export CODE_LENGTHS=4000 CODE_REPEATS=1
run r1108-img-q38 G0 /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536" code,conc4,benchy --pp 4096 --tg 128 --concurrency 1 4 --runs 2
run r1108-img-q27-df2 TP /models2/Qwen3.8-27B-FP8 int8 16384 "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_SPEC_STATS=1" code,benchy --pp 512 4096 --tg 128 --concurrency 1 --runs 2
run r1108-img-fn-tp TP "$F" int8 8192 "$FH GRIMOIRE_EXPERT_VRAM_PER_LAYER=170" pleready,gen,benchy --pp 512 2048 --tg 128 --concurrency 1 --runs 2
run r1108-fnpp-340 PP "$F" int8 8192 "$FH GRIMOIRE_PP_LAYERS=24,24 GRIMOIRE_EXPERT_VRAM_PER_LAYER=340" pleready,gen,benchy --pp 512 2048 --tg 128 --concurrency 1 --runs 2
run r1108-fnpp-340-h8 PP "$F" int8 8192 "$FH GRIMOIRE_PP_LAYERS=24,24 GRIMOIRE_EXPERT_VRAM_PER_LAYER=340 GRIMOIRE_EXPERT_HEAT=8 GRIMOIRE_EXPERT_HEAT_TP=1" pleready,gen,benchy --pp 512 2048 --tg 128 --concurrency 1 --runs 2
echo "RESUME_CHECKS_PASS $(date -Iseconds)"
