#!/bin/bash
# 11-05 17:45: the 4-user verify pass by region (one card, v1.10.4 build, while the CPU build runs): Qwen3.8-27B GPTQ + MTP
# served, 4 coding answers at once, GRIMOIRE_TIME_LAYER=all (a table per verify call; the queue drained after every
# region). One-user tables came from diag1105-vregion-*: 7 rows 38.9 ms; spec timing at 4 users: verify 44.4 ms for
# ~15.5 rows, commit 4.9 ms. Holds logs/HOLD-diagnow2.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
T=/mnt/storage/isos/grimoire-release-1105
cd "$B" || exit 9
exec >>"$B/logs/diag-1105.out" 2>&1
echo "DIAGNOW2_START $(date +%T)"
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-diagnow2
trap 'rm -f "$B/logs/HOLD-diagnow2"' EXIT
export GRIM_TREE=$T HOSTBIN=1 IMG=grimoire-b70:latest LOAD_MAX=1500 CLIENT_CPUS=14,15
name=diag1105-g0-vregion4
echo "== $(date +%T) $name"
bash ./dual_engine_1102.sh "$name" G0 /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=6 GRIMOIRE_MTP_DRAFT_VOCAB=65536 GRIMOIRE_TIME_LAYER=all" \
  conc4 > "logs/$name.run" 2>&1
echo "rc=$?"; grep -E "CONC|STOP|refusing|LEFT RUNNING|server stopped" "logs/$name.run" | head -4 | cut -c1-150
echo "DIAGNOW2_END $(date +%T)"
