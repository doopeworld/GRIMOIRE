#!/bin/bash
# 11-06 evening: why a 12-row DFlash2 verify costs ~30 ms more a step than an 8-row one on two B70s (Qwen3.8-27B
# INT8, dense: the weights are read once whatever the rows).  q27m-8 / q27m-12: code 127.7 / 95.6 tok/s at 6.48 /
# 7.78 tokens a step -> ~50.7 / ~81.4 ms a step.  The FP8 / INT8 small-M kernel takes M <= 8 only.  Code probe with
# GRIMOIRE_SPEC_TIME (draft / verify / commit), GRIMOIRE_DFLASH_TIME (the drafter's stages) and
# GRIMOIRE_TP_SMALL_AR_TIME (the all-reduces by row class), at M = 12 and M = 8.  Holds logs/HOLD-q27m12.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
M=/models2/Qwen3.8-27B-FP8
cd "$B" || exit 9
exec >>"$B/logs/q27-df2-m12-time-1106.out" 2>&1
echo "Q27M12_START $(date +%T)"
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-q27m12
trap 'rm -f "$B/logs/HOLD-q27m12"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 CODE_LENGTHS=4000 CODE_REPEATS=1
for m in 12 8; do
  name=q27mt-$m
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" TP "$M" int8 16384 \
    "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_DFLASH_M=$m GRIMOIRE_SPEC_STATS=1 GRIMOIRE_SPEC_TIME=1 GRIMOIRE_DFLASH_TIME=1 GRIMOIRE_TP_SMALL_AR_TIME=1" \
    code > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "tests=|server stopped|STOP|RxErr during" "logs/$name.run" | cut -c1-200
  grep -hE "spec time per step|dflash draft|draft time|TP small all-reduce \(BAR" "logs/$name.server.log" | grep "rank0" | tail -6 | cut -c1-260
  grep -hE "dflash.*ms|draft .*ms" "logs/$name.server.log" | grep -v "spec time" | tail -6 | cut -c1-260
  [ -e "logs/$name.STOP" ] && { echo "Q27M12_STOP link guard"; exit 1; }
done
echo "Q27M12_END $(date +%T)"
