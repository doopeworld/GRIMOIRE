#!/bin/bash
# The regression cases the v1.10.4 change can reach, gpu0, the same commands as tools/regress_all_g0.sh, compared to
# regress-1102-1701 (the v1.10.3 source). The change is cpu_experts.cpp, the one-rank decode-thread pin (both only
# under tiered experts) and the native GGUF expert grouping (2-16 rows, GGUF only): of the 44 cases only Flash-Next's
# fn-s / fn-l / fn-h run tiered experts, and the suite has no GGUF case (that path is checked by run_v1104_checks.sh
# and the image validation).
set -u
T=/mnt/storage/isos/grimoire-release-1102b
R=/mnt/storage/isos/grimoire-runs
BASE=$R/regress-1102-1701
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "REGRESSION_REFUSED gpu0.lock held $(date -Iseconds)"; exit 75; }
if fuser /dev/dri/renderD128 >/dev/null 2>&1; then echo "REGRESSION_REFUSED renderD128 busy"; exit 75; fi
cd /mnt/storage/isos/grimoire-fuse || exit 9
export GRIM_TREE=$T
OUT=$R/regress-fn-$(date +%m%d-%H%M); mkdir -p "$OUT"
echo "results: $OUT source=$(git -C "$T" rev-parse --short HEAD) start=$(date -Iseconds)"
SHORT="Explain in two sentences why the sky is blue."
LONG="$(cat real4k_ascii.txt)"
MOVED=$(for d in /mnt/models/*/; do b=$(basename "$d"); [ -e "/mnt/storage/Models/$b" ] || echo "$b"; done)
fixp() { local s=$1 b; for b in $MOVED; do s=$(printf '%s' "$s" | sed -E "s#/models/(${b//./\\.})(/|\$|[[:space:]])#/models2/\\1\\2#g"); done; printf '%s' "$s"; }
run() { # name model proj n prompt [env]
  local nm=$1 m=$2 pr=$3 n=$4 p=$5 ex=${6:-} r
  ex=$(fixp "$ex")
  r=$(EXTRA_ENV="$ex" LIM=900 bash tools/g0run.sh rx-$nm -m "$(fixp "/models/$m")" --proj $pr --ctx 8192 -p "$p" -n $n 9>&-) \
    || { echo "STOP: gpu0 guard tripped (or run failed) at $nm: $r"; cp /tmp/grim-rx-$nm.log "$OUT/$nm.log" 2>/dev/null; exit 9; }
  cp /tmp/grim-rx-$nm.log "$OUT/$nm.log"
  echo "  $nm: ${r:-NO RESULT LINE}"
}
txt() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
el() { grep -oE "elapsed=[0-9.]+" "$1" | tail -1 | cut -d= -f2; }
FN="GRIMOIRE_EXPERT_VRAM_PER_LAYER=112
GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin"
run fn-s Qwen3.8-Flash-Next-NVFP4 bf16 64 "$SHORT" "$FN"
run fn-l Qwen3.8-Flash-Next-NVFP4 bf16 24 "$LONG" "$FN"
run fn-h Qwen3.8-Flash-Next-NVFP4 bf16 64 "$SHORT" "$FN
GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
bad=0
for n in fn-s fn-l fn-h; do
  [ -n "$(txt "$OUT/$n.log")" ] || { echo "$n EMPTY_ANSWER"; bad=1; continue; }
  if cmp -s <(txt "$OUT/$n.log") <(txt "$BASE/$n.log"); then s=IDENTICAL; else s=DIFFERS; bad=1; fi
  echo "$n text vs 11-02: $s   elapsed $(el "$OUT/$n.log") s (11-02: $(el "$BASE/$n.log") s)"
done
echo "REGRESSION_FN_EXIT=$bad end=$(date -Iseconds)"
exit "$bad"
