#!/bin/bash
# 11-06 evening: where a DFlash2 step goes on two B70s (Qwen3.8-27B, the FP8 checkpoint as --proj int8, Megatron TP):
# 128.1 tok/s on the code probe at 6.48 tokens a step = ~50.6 ms a step; Radiance's FP8 figure on 2x R9700: 169.
# The verify's all-reduces (q27d-df2-ar): 75 us a call x 128 a step, rank 1's copy out 35 us vs rank 0's 13
# (gpu1 at Gen3 x4).  This run: GRIMOIRE_SPEC_TIME (fenced phases: draft / verify / commit) + GRIMOIRE_SPEC_STATS,
# code probe only.  radiance-heat-1106b build.  Holds logs/HOLD-q27t. Log: logs/q27-df2-time-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
M=/models2/Qwen3.8-27B-FP8
cd "$B" || exit 9
exec >>"$B/logs/q27-df2-time-1106.out" 2>&1
echo "Q27T_START $(date +%T) tree=$T ($(git -C "$T" rev-parse --short HEAD 2>/dev/null))"
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-q27t
trap 'rm -f "$B/logs/HOLD-q27t"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 CODE_LENGTHS=4000 CODE_REPEATS=1
name=q27t-df2-time
echo "== $(date +%T) $name"
bash ./dual_engine_1102.sh "$name" TP "$M" int8 16384 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_SPEC_TIME=1 GRIMOIRE_SPEC_STATS=1" \
  code > "logs/$name.run" 2>&1
echo "rc=$?"
grep -E "answer:|tests=|server stopped|STOP|RxErr during" "logs/$name.run" | cut -c1-200
grep -hE "spec time|spec:|dflash|draft" "logs/$name.server.log" | tail -10 | cut -c1-230
echo "Q27T_END $(date +%T)"
