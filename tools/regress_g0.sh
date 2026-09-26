#!/bin/bash
# regress_g0.sh NAME MODEL PROJ [EXTRA_ENV] -- one short and one long
# generation on gpu0 through tools/g0run.sh (gpu0 only, stops on any new
# AER/xe error); prints the capability lines, speed and the generated text.
set -u
cd /mnt/storage/isos/grimoire-fuse
n=$1 m=$2 pr=$3 ex=${4:-}
SHORT="Explain in two sentences why the sky is blue."
for kind in short long; do
  if [ $kind = short ]; then P="$SHORT"; N=64; else P="$(cat real4k_ascii.txt)"; N=24; fi
  r=$(EXTRA_ENV="$ex" LIM=900 bash tools/g0run.sh rg-$n-$kind -m /models/$m --proj $pr --ctx 8192 -p "$P" -n $N) \
    || { echo "STOP: gpu0 guard tripped at $n $kind"; exit 9; }
  L=/tmp/grim-rg-$n-$kind.log
  echo "== $n [$kind] ${r:-NO RESULT LINE}"
  grep -E "^\s+(parallel|speculation|batching|prefill|norms) |spec:|accepted|error|failed|abort|Abort" $L | head -8 | sed 's/^/     /'
  awk '/norms /{f=1;next} f' $L | grep -vE "^\s|^$|prompt=" | tr "\n" " " | cut -c1-320 | sed 's/^/     text: /'; echo
done
