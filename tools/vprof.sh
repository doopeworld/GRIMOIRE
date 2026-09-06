#!/bin/bash
# M=4 verify region attribution: MTP k=3 with per-region host timing.
set -u
cd /mnt/storage/isos/grimoire-fuse
PROMPT=$(cat real4k.txt)
GPU=${GPU:-gpu0} LIM=${LIM:-900} EXTRA_ENV="GRIMOIRE_W4A8=1
GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_EXACT_VERIFY=1
GRIMOIRE_MTP_DRAFT_VOCAB=131072
GRIMOIRE_MTP_PROFILE=1
${TLENV:-GRIMOIRE_TIME_LAYER=all}" \
  bash tools/tune.sh ${NAME:-vprof} /grimoire/bin/grimoire \
  -m /models/Qwen3.8-27B-MXFP4-GRIMOIRE --proj mxfp4 --ctx 8192 \
  -p "$PROMPT" -n ${NPRED:-8}
