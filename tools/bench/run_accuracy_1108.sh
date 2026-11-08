#!/bin/bash
# Previously prepared imatrix experiments; independent of the frozen rc2 candidate.
set -euo pipefail
B=/mnt/storage/isos/grimoire-runs/bench-1022
X=/mnt/storage/isos/grimoire-runs/dev-1106
A=/mnt/storage/Models/grimoire-acc
PD=/mnt/storage/Models/grimoire-ple
cd "$B"
exec >>logs/accuracy-1108.out 2>&1
exec 6>logs/accuracy-1108.lock
flock -n 6 || exit 9
test "$(git -C "$X" rev-parse --short HEAD)" = ff77f2b
grep -qx IMX_BUILD_PASS logs/imx-build-1106.out
for f in ref-bf16.bin sherlock.ids imatrix-qwen38-27b.gguf cmp_logits.py fn-bf16.bin fn512.ids imatrix-flash-next.gguf; do
  test -s "$A/$f"
done
export GRIM_TREE=$X GRIM_IMAGE=grimoire-b70:v1.10.4 HOSTBIN=1
echo "ACCURACY_START $(date -Iseconds) source=ff77f2b"
trap 'cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"' EXIT
run() {
  local name=$1 model=$2 proj=$3 ids=$4 ref=$5 extra=$6
  local out="$A/acc1108-$name.bin" log="$B/logs/acc1108-$name.log"
  if [ -f "$B/logs/acc1108-$name.VERIFIED" ]; then echo "SKIP verified $name"; return; fi
  test ! -e "$out" && test ! -e "$log"
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  echo "SCORE_START $name $(date -Iseconds)"
  EXTRA_ENV="GRIMOIRE_POWER_LIMIT_W=0
$extra" LIM=1500 TAILN=6 flock -n /mnt/storage/isos/grimoire-runs/gpu0.lock \
    bash /mnt/storage/isos/grimoire-fuse/tools/g0run.sh "acc1108-$name" /grimoire/bin/grimoire \
      -m "$model" --proj "$proj" --ctx 1024 --score-ids "/models/grimoire-acc/$ids" \
      --score-out "/models/grimoire-acc/acc1108-$name.bin"
  cp "/tmp/grim-acc1108-$name.log" "$log"
  python3 - "$A/$ids" "$A/$ref" "$out" <<'VERIFY'
from pathlib import Path
import sys
ids, ref, out = map(Path, sys.argv[1:])
n = len(ids.read_text().split())
assert ref.stat().st_size == out.stat().st_size == n * 248320 * 4, (n, ref.stat().st_size, out.stat().st_size)
VERIFY
  if [[ "$extra" == *GRIMOIRE_IMATRIX=* ]]; then
    grep -E 'imatrix.*(hit|matched|round|tensor|entries)|importance' "$log" | tail -8
    if [ "$proj" = int4 ]; then
      grep -Eq 'imatrix: [1-9][0-9]* INT4 weights rounded by their output error' "$log"
    fi
  fi
  if [[ "$extra" == *GRIMOIRE_IMATRIX_FOLD=* ]]; then
    grep -Eq "imatrix fold .*: [1-9][0-9]* layers' input norms folded" "$log"
  fi
  uv run --no-project --with numpy python "$A/cmp_logits.py" "$A/$ref" "$out" 248320 "$name"
  touch "$B/logs/acc1108-$name.VERIFIED"
  echo "SCORE_PASS $name $(date -Iseconds)"
}
M=/models2/Qwen3.8-27B
IM=GRIMOIRE_IMATRIX=/models/grimoire-acc/imatrix-qwen38-27b.gguf
run q27-int4 "$M" int4 sherlock.ids ref-bf16.bin ""
run q27-int4-imx "$M" int4 sherlock.ids ref-bf16.bin "$IM"
run q27-int4-imx-g64 "$M" int4 sherlock.ids ref-bf16.bin "$IM
GRIMOIRE_INT4_RTN_GROUP=64"
run q27-int4-fold "$M" int4 sherlock.ids ref-bf16.bin "$IM
GRIMOIRE_IMATRIX_FOLD=0.4"
run q27-int4-fold-g64 "$M" int4 sherlock.ids ref-bf16.bin "$IM
GRIMOIRE_IMATRIX_FOLD=0.4
GRIMOIRE_INT4_RTN_GROUP=64"
run q27-int8 "$M" int8 sherlock.ids ref-bf16.bin ""
run q27-int8-fold "$M" int8 sherlock.ids ref-bf16.bin "$IM
GRIMOIRE_IMATRIX_FOLD=0.2"
run q27-autoround /models2/Qwen3.8-27B-int4-AutoRound int4 sherlock.ids ref-bf16.bin ""
F=/models2/Qwen3.8-Flash-Next-NVFP4
FE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin
GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab
GRIMOIRE_EXPERT_VRAM_PER_LAYER=90
GRIMOIRE_EXPERT_HEAT=0"
run fn-int4 "$F" int4 fn512.ids fn-bf16.bin "$FE"
run fn-int4-imx "$F" int4 fn512.ids fn-bf16.bin "$FE
GRIMOIRE_IMATRIX=/models/grimoire-acc/imatrix-flash-next.gguf"
run fn-int4-imx-g64 "$F" int4 fn512.ids fn-bf16.bin "$FE
GRIMOIRE_IMATRIX=/models/grimoire-acc/imatrix-flash-next.gguf
GRIMOIRE_INT4_RTN_GROUP=64"
echo "ACCURACY_CHECKS_PASS $(date -Iseconds)"
