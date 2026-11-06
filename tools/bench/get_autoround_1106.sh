#!/bin/bash
# 11-06: download CySpiegel/Qwen3.8-27B-Int4-AutoRound (19 GB, Apache-2.0) to /mnt/models (= /models2 in the run
# containers) for run_imx_acc_1106.sh's "autoround" variant.  CPU / network only, no GPU: the grimoire-ref:cpu image
# (transformers -> huggingface_hub), resumable, refuses below 30 GB free.  Log: logs/get-autoround-1106.out.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
D=/mnt/models/Qwen3.8-27B-Int4-AutoRound
exec >>"$B/logs/get-autoround-1106.out" 2>&1
echo "GET_AR_START $(date +%T)"
free_gb=$(df -BG --output=avail /mnt/models | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 30 ] || { echo "GET_AR_STOP only ${free_gb}G free on /mnt/models"; exit 9; }
docker run --rm --name grim-get-autoround --cpuset-cpus 6-13 -v /mnt/models:/models grimoire-ref:cpu \
  python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('CySpiegel/Qwen3.8-27B-Int4-AutoRound', local_dir='/models/Qwen3.8-27B-Int4-AutoRound')
print('downloaded to', p)
"
rc=$?
du -sh "$D" 2>&1
ls "$D" 2>&1 | head -30
echo "GET_AR_END $(date +%T) rc=$rc"
