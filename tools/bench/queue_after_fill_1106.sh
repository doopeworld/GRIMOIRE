#!/bin/bash
# 11-06 evening queue (one GPU job at a time): after run_fill_ab_1106.sh ends -> run_fn_mtp_fill_1106.sh (one card,
# Flash-Next MTP with heat + fill) -> run_v1106_all.sh WANT (the v1.10.5-rc2 gate). Stops if the fill A/B stopped.
#   queue_after_fill_1106.sh WANT
set -u
WANT=${1:?commit}
B=/mnt/storage/isos/grimoire-runs/bench-1022
exec >>"$B/logs/queue-after-fill-1106.out" 2>&1
echo "QUEUE_START $(date +%T) want=$WANT"
until grep -qE "^FILL_AB_(END|STOP)" "$B/logs/fill-ab-1106.out" 2>/dev/null; do sleep 20; done
grep -q "^FILL_AB_END" "$B/logs/fill-ab-1106.out" || { echo "QUEUE_STOP fill A/B stopped $(date +%T)"; exit 9; }
echo "fill A/B ended $(date +%T)"
bash "$B/run_fn_mtp_fill_1106.sh"
echo "fn mtp ended $(date +%T)"
bash "$B/run_v1106_all.sh" "$WANT"
echo "QUEUE_END $(date +%T)"
