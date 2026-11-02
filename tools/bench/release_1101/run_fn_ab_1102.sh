#!/bin/bash
# run_fn_ab_1102.sh TREE TAG: the regression's Flash-Next short case on gpu0 with TREE's binaries, same settings.
# Holds gpu0.lock without waiting. Prints the time, the CPU-expert stats and the CPU load during the run.
set -u
TREE=$1 TAG=$2
R=/mnt/storage/isos/grimoire-runs
L=$R/bench-1022/logs/fn-ab-$TAG-1102.out
exec >"$L" 2>&1
end() { echo "FN_AB_END tag=$TAG rc=$1 $(date -Iseconds)"; exit "$1"; }
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "gpu0.lock held"; end 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then echo "renderD128 busy"; end 75; fi
cd /mnt/storage/isos/grimoire-fuse || end 9
SHORT="Explain in two sentences why the sky is blue."
FN="GRIMOIRE_EXPERT_VRAM_PER_LAYER=112
GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin"
echo "== $(date +%T) tree=$TREE builder running: $(docker ps --format '{{.Names}}' | grep -c build)"
GRIM_TREE=$TREE EXTRA_ENV="$FN" LIM=900 bash tools/g0run.sh rx-fn-$TAG -m /models2/Qwen3.8-Flash-Next-NVFP4 \
  --proj bf16 --ctx 8192 -p "$SHORT" -n 64 9>&- || end 9
cp "/tmp/grim-rx-fn-$TAG.log" "$R/bench-1022/logs/fn-$TAG-1102.log"
grep -E "elapsed=|CPU experts: [0-9]+ layer calls|pool threads" "/tmp/grim-rx-fn-$TAG.log" | cut -c1-120
uptime
txt() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
if cmp -s <(txt "/tmp/grim-rx-fn-$TAG.log") <(txt "$R/regress-1102-1701/fn-s.log"); then
  echo "text vs the regression's fn-s: IDENTICAL"
else
  echo "text vs the regression's fn-s: differs"
fi
end 0
