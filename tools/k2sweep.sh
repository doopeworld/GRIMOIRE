#!/bin/bash
# K2 decode: exactness vs the previous binary's text, timeline, R sweep. gpu0 only.
cd /mnt/storage/isos/grimoire-fuse
M=/models/K2-Horizon-MoVA-36B-A4B; P="Explain in two sentences why the sky is blue."
txt() { awk '/norms /{f=1;next} f' /tmp/grim-$1.log | grep -v '^\s\|^$\|prompt=\|greedy'; }
r=$(EXTRA_ENV="GRIMOIRE_PRINT_IDS=1" LIM=900 bash tools/g0run.sh k2-s0 -m $M --proj mxfp4 --ctx 8192 -p "$P" -n 256) || { echo STOP; exit 9; }
echo "default: $r"
if [ "$(txt k2-s0)" = "$(txt k2-d256)" ]; then echo "text IDENTICAL to previous binary (256 tokens)"; else echo "text DIFFERS from previous binary"; diff <(txt k2-s0) <(txt k2-d256) | head -5; fi
r=$(EXTRA_ENV="GRIMOIRE_TIMELINE=1
GRIMOIRE_TIMELINE_LAYER=3" LIM=900 bash tools/g0run.sh k2-tl2 -m $M --proj mxfp4 --ctx 8192 -p "$P" -n 8) || { echo STOP; exit 9; }
sed -n '/device timeline/,/incr_pos/p' /tmp/grim-k2-tl2.log | grep -vE '^\s+L[0-9]{2} full'
grep -E 'TOTAL' /tmp/grim-k2-tl2.log | tail -1
for v in "B70_MOE_DN_SLOTS=2" "B70_MOE_DN_SLOTS=4" "B70_MOVA_ROWS=1" "B70_MOVA_ROWS=4" "B70_MOE_SLOTS=8"; do
  r=$(EXTRA_ENV="$v" LIM=900 bash tools/g0run.sh k2-sw -m $M --proj mxfp4 --ctx 8192 -p "$P" -n 256) || { echo STOP; exit 9; }
  s=$( [ "$(txt k2-sw)" = "$(txt k2-s0)" ] && echo same-text || echo TEXT-DIFFERS )
  echo "$v: $r $s"
done
