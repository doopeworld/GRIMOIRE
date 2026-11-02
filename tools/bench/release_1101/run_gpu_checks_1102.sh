#!/bin/bash
# One guarded job on gpu0, one thing at a time, stops at the first failure:
#  A. Flash-Next short case again on the combined build (page cache warm now): why was fn-s 16.7 s?
#  B. wide-INT4 register probe: one small shape first, then all shapes. No engine dispatch changes.
set -u
R=/mnt/storage/isos/grimoire-runs
L=$R/bench-1022/logs/gpu-checks-1102.out
E=/mnt/storage/isos/grimoire-engine-1101
W=/mnt/storage/isos/grimoire-verify-grf-1102
exec >"$L" 2>&1
end() { echo "GPU_CHECKS_END rc=$1 $(date -Iseconds)"; exit "$1"; }
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "gpu0.lock held"; end 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then echo "renderD128 busy"; end 75; fi
cd /mnt/storage/isos/grimoire-fuse || end 9

SHORT="Explain in two sentences why the sky is blue."
FN="GRIMOIRE_EXPERT_VRAM_PER_LAYER=112
GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin"
for rep in 1 2; do
  echo "== $(date +%T) fn-s warm rep $rep"
  GRIM_TREE=$E EXTRA_ENV="$FN" LIM=900 bash tools/g0run.sh rx-fn-s-warm$rep -m /models2/Qwen3.8-Flash-Next-NVFP4 \
    --proj bf16 --ctx 8192 -p "$SHORT" -n 64 9>&- || end 9
  cp "/tmp/grim-rx-fn-s-warm$rep.log" "$R/bench-1022/logs/fn-s-warm$rep-1102.log"
  grep -E "elapsed=|CPU experts: [0-9]+ layer calls" "/tmp/grim-rx-fn-s-warm$rep.log" | cut -c1-110
done
txt() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
for rep in 1 2; do
  if cmp -s <(txt "$R/bench-1022/logs/fn-s-warm$rep-1102.log") <(txt "$R/regress-1102-1701/fn-s.log"); then
    echo "fn-s warm rep $rep text vs the regression's fn-s: IDENTICAL"
  else
    echo "fn-s warm rep $rep text vs the regression's fn-s: differs"
  fi
done

echo "== $(date +%T) probe smoke: shape 0 only"
GRIM_TREE=$W LIM=300 TAILN=20 bash tools/g0run.sh vgrf-smoke /grimoire/bin/i4wide_grf_probe 0 9>&-
rc=$?
cp /tmp/grim-vgrf-smoke.log "$R/bench-1022/logs/vgrf-smoke-1102.log"
[ "$rc" -eq 0 ] || end 9
grep -q "failures=0 PASS" /tmp/grim-vgrf-smoke.log || { echo "probe smoke did not pass"; end 8; }

echo "== $(date +%T) probe: all shapes"
GRIM_TREE=$W LIM=900 TAILN=70 bash tools/g0run.sh vgrf-all /grimoire/bin/i4wide_grf_probe 9>&-
rc=$?
cp /tmp/grim-vgrf-all.log "$R/bench-1022/logs/vgrf-all-1102.log"
[ "$rc" -eq 0 ] || end 9
grep -q "failures=0 PASS" /tmp/grim-vgrf-all.log || { echo "probe did not pass"; end 8; }
end 0
