#!/bin/bash
# ab_ink.sh "tag:ENV=V ENV2=V" ... -- per arm: serve_ink.sh start (2 cards) with the env, BetterBench quick decode, stop
X=/mnt/storage/isos/grimoire-runs/ink-1123
cd $X
for arm in "$@"; do
  tag=${arm%%:*}; envs=${arm#*:}
  env $envs ./serve_ink.sh start ${CARDS:-2} 32768 ${CHUNK:-2048} 6911 > logs/ab-$tag-start.log 2>&1 || { echo "$tag START FAILED"; tail -5 logs/ab-$tag-start.log; ./serve_ink.sh stop > /dev/null 2>&1; continue; }
  ./bb_ink.sh $tag ${BBARGS:---quick --decode} > logs/ab-$tag.res 2>&1
  grep -hE "^\*\*Combined|^engine:|^\| (2000|8000) |^\| (chat|code|file_edit|json|math|prose|reasoning|summarization) " logs/bb-$tag.out logs/ab-$tag.res | sed "s/^/$tag /"
  ./serve_ink.sh stop > logs/ab-$tag-stop.log 2>&1
  grep -E "GPU HELD|still running" logs/ab-$tag-stop.log
done
echo AB_DONE
