#!/bin/bash
# 11-06 evening: GRIMOIRE_DFLASH_DRAFT_VOCAB on two B70s (Qwen3.8-27B, FP8 checkpoint as --proj int8, Megatron TP,
# DFlash2, M = 8).  The drafter's head over the first n token ids instead of all 248K (2.2 of a 9.3 ms draft).
# Speculation is exact: the code answers must be IDENTICAL to the whole-vocabulary run.
#   q27dv-full / q27dv-64k / q27dv-32k   code probe + GRIMOIRE_DFLASH_TIME (draft stages) + llama-benchy pp512/tg128 x2
# Waits for run_heat_build2_1106.sh <WANT>. Holds logs/HOLD-q27dv. Log: logs/q27-df2-dv-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-runs/radiance-heat-1106b
M=/models2/Qwen3.8-27B-FP8
WANT=${1:?commit the build must be at}
cd "$B" || exit 9
exec >>"$B/logs/q27-df2-dv-1106.out" 2>&1
echo "Q27DV_START $(date +%T) want=$WANT"
until grep -q "source=$WANT" logs/heat-build2-1106.out 2>/dev/null && grep -qE "^HEAT2_BUILD_(PASS|FAIL)" logs/heat-build2-1106.out; do sleep 20; done
grep -qx HEAT2_BUILD_PASS logs/heat-build2-1106.out || { echo "Q27DV_STOP build failed"; grep -m8 error logs/heat-build2-1106.out; exit 9; }
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-q27dv
trap 'rm -f "$B/logs/HOLD-q27dv"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 CODE_LENGTHS=4000 CODE_REPEATS=1
for v in full 65536 32768; do
  name=q27dv-$v
  env="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_SPEC_STATS=1 GRIMOIRE_DFLASH_TIME=1"
  [ "$v" = full ] || env="$env GRIMOIRE_DFLASH_DRAFT_VOCAB=$v"
  echo "== $(date +%T) $name"
  bash ./dual_engine_1102.sh "$name" TP "$M" int8 16384 "$env" code,benchy --pp 512 --tg 128 --concurrency 1 --runs 2 \
    > "logs/$name.run" 2>&1
  echo "rc=$?"
  grep -E "answer:|tests=|tg128|pp512|server stopped|STOP|RxErr during" "logs/$name.run" | cut -c1-200
  grep -hE "spec: DFlash2" "logs/$name.server.log" | grep rank0 | tail -1 | cut -c1-200
  grep -h -A15 "\[rank0\]   DFlash native timing call 30," "logs/$name.server.log" | grep -E "lm head|TOTAL" | cut -c1-120
  [ -e "logs/$name.STOP" ] && { echo "Q27DV_STOP link guard"; exit 1; }
done
python3 - "$B/logs" <<'PY'
import json, sys
d = sys.argv[1]
def load(n):
    try: return json.load(open(f"{d}/{n}.code.json"))
    except Exception as e: return None
base = load("q27dv-full")
for n in ("q27dv-65536", "q27dv-32768"):
    a = load(n)
    if not base or not a: print("  no code json for", n); continue
    def texts(x): return [v.get("text", "") for k, v in sorted(x.items()) if isinstance(v, dict)]
    print("  %s vs full: code answers %s" % (n, "IDENTICAL" if texts(a) == texts(base) else "DIFFER"))
PY
echo "Q27DV_END $(date +%T)"
