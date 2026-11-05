#!/bin/bash
# v1.10.4 candidate checks on GPU0, one job at a time; stops at the first failure. Start by hand after the build passed.
# Binaries: grimoire-release-1102b = v1.10.3 source + CPU-expert pool fix + one card's decode thread free
# + native GGUF experts grouped by default (+ since 11-05: bf16 two-card small all-reduce, 35201c2).
# Runtime: grimoire-b70:latest (v1.10.3) with the host bin/.
#   1. Flash-Next CLI short case, five loads x decode thread pinned / free: text must match regress-1102-1701.
#   2. Flash-Next served, v1.10.3 code vs candidate, quiet and with 4 busy processes: text must match.
#   3. Ornith Q4_K_M native with no option set: alone answers must equal r1102-opt-moe (option on, 11-02).
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
NEW=/mnt/storage/isos/grimoire-release-1102b
cd "$B" || exit 9
exec >"$B/logs/v1104-checks.out" 2>&1
end() { echo "V1104_CHECKS_END rc=$1 $(date +%T)"; exit "$1"; }
echo "V1104_CHECKS_START $(date +%T) source=$(git -C "$NEW" rev-parse --short HEAD)"
grep -qx RELEASE_BUILD_PASS logs/release-build-1102b.out || { echo "build did not pass"; end 9; }
[ -x "$NEW/bin/grimoire" ] && [ -x "$NEW/bin/grimoire-server" ] || { echo "no binaries"; end 9; }

# SKIP_MATRIX=1 (11-05): the matrix passed on 11-04 19:26 with the 1881830 build; 35201c2 adds only the two-card
# bf16 small all-reduce and an expert budget cap that is unchanged on a B70, so one-card code is the same
if [ "${SKIP_MATRIX:-0}" = 1 ]; then
  echo "pin matrix: not rerun, passed 11-04 on 1881830 (one-card code unchanged in 35201c2):"
else
  POOL_TREE=$NEW MATRIX_TAG=v1104 bash "$B/run_pin_matrix_1102.sh"
  rc=$?
  [ "$rc" = 0 ] || { grep -E "^\||PIN_MATRIX_END" logs/pin-matrix-v1104-1102.out; end 9; }
fi
grep -E "^\||PIN_MATRIX_END" logs/pin-matrix-v1104-1102.out

for spec in "v1104-quiet 0" "v1104-busy4 4"; do
  set -- $spec
  NEW_TREE=$NEW bash "$B/run_fn_serve_1104.sh" "$1" "$2"
  rc=$?
  grep -E "^==|tg128|IDENTICAL|DIFFERS|FN_SERVE" "logs/fn-serve-$1.out" | cut -c1-150
  [ "$rc" = 0 ] || end 9
done

OG=/models2/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf
OTOK=/mnt/storage/Models/Ornith-1.5-35B-A3B
name=v1104-og-native
GRIM_TREE=$NEW HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15 TOKDIR=$OTOK \
  bash ./dual_engine_1102.sh "$name" G0 "$OG" native 32000 "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8" \
  ctest,conc4,seq4,benchy --pp 4096 --tg 128 --concurrency 1 8 --runs 1 > "logs/$name.run" 2>&1
rc=$?
grep -E "ready=|answer:|answers pass|Coherence|server stopped|LEFT RUNNING|STOP" "logs/$name.run" | cut -c1-140
grep -E "tg128|pp4096" "logs/$name.benchy.md" | cut -c1-120
if [ "$rc" -ne 0 ] || [ -e "logs/$name.STOP" ] || ! grep -q "server stopped gracefully" "logs/$name.run"; then end 9; fi
python3 - "logs/r1102-opt-moe" "logs/$name" <<'PY'
import json, sys
a, b = sys.argv[1], sys.argv[2]
for kind in ('seq4', 'conc4'):
    A, Bj = json.load(open(f'{a}.{kind}.json')), json.load(open(f'{b}.{kind}.json'))
    same = sum(A[k]['text'] == Bj.get(k, {}).get('text') for k in A)
    print(f'{kind}: option-on 11-02 vs default now, identical answers {same}/{len(A)}')
PY
end 0
