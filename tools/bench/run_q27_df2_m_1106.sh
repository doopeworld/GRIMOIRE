#!/bin/bash
# 11-06 evening: the DFlash2 verify width on two B70s for a DENSE model (Qwen3.8-27B, FP8 checkpoint as --proj int8,
# Megatron TP).  GRIMOIRE_DFLASH_M defaults to 8 rows (7 drafts + the bonus), chosen 09-30 on Ornith, a MoE whose
# verify pays ~3 ms an extra row in experts; here the code probe accepted 5.50 of 7 drafts (78.6%) -- the cap binds.
# The drafter fills its 16-token block in one pass either way, so only the verify widens.  11-06 step split at M=8:
# draft 9.0 + verify 40.6 + snapshot/commit 1.1 = 50.7 ms for 6.5 tokens (128 tok/s).
#   q27m-8 / q27m-12 / q27m-16   code probe + llama-benchy pp512 / tg128 x2 (+ GRIMOIRE_SPEC_STATS)
# radiance-heat-1106b build. Holds logs/HOLD-q27m. Log: logs/q27-df2-m-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
M=/models2/Qwen3.8-27B-FP8
cd "$B" || exit 9
exec >>"$B/logs/q27-df2-m-1106.out" 2>&1
echo "Q27M_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-q27m
trap 'rm -f "$B/logs/HOLD-q27m"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 CODE_LENGTHS=4000 CODE_REPEATS=1
for m in 8 12 16; do
  name=q27m-$m
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" TP "$M" int8 16384 \
    "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_DFLASH_M=$m GRIMOIRE_SPEC_STATS=1" \
    code,benchy --pp 512 --tg 128 --concurrency 1 --runs 2 > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "answer:|tests=|tg128|pp512|server stopped|STOP|RxErr during" "logs/$name.run" | cut -c1-200
  grep -hE "spec: DFlash2" "logs/$name.server.log" | tail -2 | cut -c1-200
  [ -e "logs/$name.STOP" ] && { echo "Q27M_STOP link guard"; exit 1; }
done
echo "Q27M_END $(date +%T)"
