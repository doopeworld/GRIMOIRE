#!/bin/bash
# Sweep GRIMOIRE_VERIFY_KEYS_PER_SPLIT on the real 4k-context MTP decode.
set -u
cd /mnt/storage/isos/grimoire-fuse
PROMPT=$(cat real4k.txt)
KPS=${KPS:?keys per split}
GPU=${GPU:-gpu0} LIM=${LIM:-900} EXTRA_ENV="GRIMOIRE_W4A8=1
GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_EXACT_VERIFY=1
GRIMOIRE_MTP_DRAFT_VOCAB=131072
GRIMOIRE_MTP_PROFILE=1
GRIMOIRE_VERIFY_KEYS_PER_SPLIT=$KPS" \
  bash tools/tune.sh vk$KPS /grimoire/bin/grimoire \
  -m /models/Qwen3.8-27B-MXFP4-GRIMOIRE --proj mxfp4 --ctx 8192 \
  -p "$PROMPT" -n ${NPRED:-64}
