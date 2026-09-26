#!/bin/bash
# fetch_models.sh -- download the checkpoints GRIMOIRE supports but never ran on
# the B70 (earlier sessions had no HuggingFace access).  No GPU is attached to
# this container.  Log: /mnt/storage/Models/_downloads.log
set -u
LOG=/mnt/storage/Models/_downloads.log
for r in unsloth/Qwen3.8-27B-NVFP4 ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
         IFM/K2-Horizon-MoVA-36B-A4B Agnes-AI/Agnes-3.0-Flash; do
  d=/models/$(basename $r)
  echo "$(date +%T) START $r -> $d" >> $LOG
  docker run --rm --name grimdl-$(basename $r | tr 'A-Z.' 'a-z-') --entrypoint bash \
    -v /mnt/storage/Models:/models my-vllm-xpu:latest -lc \
    "hf download $r --local-dir $d --max-workers 8 > /dev/null 2>&1; echo exit \$?" >> $LOG 2>&1
  echo "$(date +%T) DONE  $r ($(du -sh /mnt/storage/Models/$(basename $r) | cut -f1))" >> $LOG
done
echo "$(date +%T) ALL DONE" >> $LOG
