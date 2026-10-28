#!/bin/bash
# Qwen3.8-27B-FP8 + DFlash2 on both B70s: the 8-row verify's FP8 GEMMs through GRIMOIRE's
# small-M kernels (a) or through oneDNN (b, GRIMOIRE_SMALLM_ONEDNN=1); spec phases fenced.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
M=/models2/Qwen3.8-27B-FP8
DF="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2 GRIMOIRE_SPEC_TIME=1"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && { echo "STOPPED at $1"; exit 3; }; sleep 3; }
run tp-df4a TP $M fp8 16384 "$DF" code
run tp-df4b TP $M fp8 16384 "$DF GRIMOIRE_SMALLM_ONEDNN=1" code
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
for n in tp-df4a tp-df4b; do
  echo "== $n"; grep -E "\[code\]  +len" logs/$n.out | tail -2 | cut -c1-150
  grep -h "spec time" logs/$n.server.log | grep rank0 | tail -2
done
echo ALL_DONE
