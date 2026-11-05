#!/bin/bash
# 11-05: served A/B of the next release candidate (grimoire-release-1105b = a3f81a1) against v1.10.4
# (grimoire-release-1105 = 315167c), one B70 (gpu0), one server at a time, v1.10.4 first in every pair.
# The B580 is Ian's ComfyUI card and is never used: the small-card case runs on gpu0 with a 6 GB expert budget.
#   gptq   Qwen3.8-27B GPTQ-Int4 + MTP, the GRIMOIRE-QWEN3.8 profile settings: INT4 split plans (<= 8, 9-16 rows)
#   gguf   Qwen3.8-27B UD-Q4_K_XL native GGUF + MTP, the GRIMOIRE-QWEN3.8-GGUF settings: GGUF split plan (<= 8 rows)
#   mx     Qwen3.8-27B MXFP4 + MTP, the 11-04 MXFP4-vs-Swift settings: MXFP4 o / z split plan
#   omx    Ornith-1.5-35B-A3B MXFP4, the GRIMOIRE-ORNITH settings, 1 and 8 users
#   split  Ornith-1.5-35B-A3B NVFP4, tiered experts, GRIMOIRE_EXPERT_VRAM_GB=6 (2/3 of the experts in RAM, as on a
#          12 GB B580 or a 24 GB B60): the CPU split in decode (23f2a24) vs v1.10.4 reading every RAM expert over PCIe
# Waits for RELEASE_BUILD_PASS and for the experiment queue (RESUME_EXP_END). Stops at a failed run (not at a wrong
# answer: answers are listed for review).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
OLD=/mnt/storage/isos/grimoire-release-1105
NEW=/mnt/storage/isos/grimoire-release-1105b
cd "$B" || exit 9
exec >>"$B/logs/next-ab-1105.out" 2>&1
echo "NEXT_AB_START $(date +%T)"
until grep -qE "^RELEASE_BUILD_(PASS|FAIL)" logs/release-build-1105b.out 2>/dev/null; do sleep 20; done
grep -qx RELEASE_BUILD_PASS logs/release-build-1105b.out || { echo "NEXT_AB_STOP build failed"; exit 9; }
until grep -q "^RESUME_EXP_END" logs/resume-exp-1105.out 2>/dev/null; do sleep 20; done
waitq() { until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done; }
export HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
run() { # tree name model proj ctx env tests benchy...
  local tree=$1 name=$2; shift 2
  waitq; echo "== $(date +%T) $name"
  GRIM_TREE=$tree bash ./dual_engine_1102.sh "$name" G0 "$@" > "logs/$name.run" 2>&1
  local rc=$?
  grep -E "ready=|answer:|answers pass|Coherence|tests=|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-150
  if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then
    echo "NEXT_AB_STOP $name rc=$rc"; exit 9
  fi
}
pair() { # tag model proj ctx env tests benchy...
  local tag=$1; shift
  run "$OLD" "nx1105-$tag-old" "$@"
  run "$NEW" "nx1105-$tag-new" "$@"
  python3 "$B/compare_ab_1105.py" "$B/logs" v1.10.4="nx1105-$tag-old" candidate="nx1105-$tag-new"
}
P="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536"
Q="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=4 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536"
BQ="--pp 4096 --tg 128 --concurrency 1 2 4 --runs 2"
export CODE_LENGTHS=4000 CODE_REPEATS=1
pair gptq /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 "$P" ctest,code,conc4,benchy $BQ
export TOKDIR=/mnt/models/Qwen3.8-27B-GPTQ-Int4-MTP-BF16
pair gguf /models2/unsloth-Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf native 16384 "$P" ctest,code,conc4,benchy $BQ
unset TOKDIR
pair mx /models2/Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16 mxfp4 36864 "$Q" ctest,code,conc4,benchy $BQ
pair omx /models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE mxfp4 32000 "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8" \
  ctest,conc4,seq4,benchy --pp 4096 --tg 128 --concurrency 1 8 --runs 2
export TOKDIR=/mnt/storage/Models/Ornith-1.5-35B-A3B
pair split /models/Ornith-1.5-35B-A3B-NVFP4 mxfp4 8192 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_TIERED_MOE=1 GRIMOIRE_EXPERT_VRAM_GB=6 GRIMOIRE_CX_STATS=1" \
  ctest,code,benchy --pp 512 2048 --tg 128 --concurrency 1 --runs 2
grep -hE "tiered experts|CPU experts" logs/nx1105-split-old.server.log logs/nx1105-split-new.server.log | head -8 | cut -c1-160
python3 - "$B/logs" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
for tag in ('gptq', 'gguf', 'mx', 'omx', 'split'):
    for kind in ('conc4', 'seq4', 'code'):
        a, b = d / f'nx1105-{tag}-old.{kind}.json', d / f'nx1105-{tag}-new.{kind}.json'
        if a.exists() and b.exists():
            A, Bj = json.load(open(a)), json.load(open(b))
            same = sum(A[k].get('text') == Bj.get(k, {}).get('text') for k in A if isinstance(A[k], dict))
            print(f'{tag} {kind}: identical answers {same}/{len(A)}')
PY
echo "NEXT_AB_DONE $(date +%T)"
