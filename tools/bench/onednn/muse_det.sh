#!/bin/bash
# 10-29: is Muse deterministic on one build?  The regression's mu-s / z-mumx texts flip between two
# variants across builds (10-28 A, 10-29 pre-merge B, 10-29 merge A).  Same build (merge, bin/),
# same command as regress_all_g0.sh's mu-s, three times; compare the texts.
cd /mnt/storage/isos/grimoire-fuse || exit 2
L=/mnt/storage/isos/grimoire-runs/bench-1022/logs
echo "$(date '+%F %T') muse_det.sh (gpu0: Muse INT4 short prompt x3, same build)" >> $L/claude-commands-1028.log
{
for i in 1 2 3; do
  flock /mnt/user/isos/grimoire-runs/gpu0.lock env LIM=900 bash tools/g0run.sh mudet-$i -m /models/Muse-Glimmer-30B-INT4-W4A16 \
    --proj int4 --ctx 8192 -p "Explain in two sentences why the sky is blue." -n 64 || { echo "STOP run $i"; exit 9; }
  awk '/norms /{f=1;next} f' /tmp/grim-mudet-$i.log | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids|prompt attention' > /tmp/mudet-$i.txt
  echo "run $i: $(md5sum < /tmp/mudet-$i.txt | cut -c1-12)  $(tr '\n' ' ' < /tmp/mudet-$i.txt | cut -c120-230)"
done
echo "== MUDET_DONE"
} > $L/muse_det.outer 2>&1
