#!/bin/bash
# 10-28: batched MTP drafting on two cards (Qwen3.8-27B-FP8, Megatron TP, MTP, 8 slots).
# GRIMOIRE_SPEC_MAX_SEQS=8: every step up to 8 users speculates, so c2 / c4 / c8 all use the
# batched drafts.  Coding probe (answers + unit tests) + llama-benchy pp4096 / tg128 c1/2/4/8.
# Before (row-by-row drafts, 10-28): c1 56.5, c2 62.9, c4 88.9 speculating / 95.0 plain, c8 161.9 plain.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
echo "$(date '+%F %T') q38_bd.sh (two cards via dual.sh: batched drafting, SPEC_MAX_SEQS=8)" >> logs/claude-commands-1028.log
RP="RP_WIN=1000 RP_RUN=1000000"
{
env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh bd-mtp TP /models2/Qwen3.8-27B-FP8 fp8 18432 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=1 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_SPEC_MAX_SEQS=8 GRIMOIRE_SPEC_TIMING=1" \
  code,benchy --pp 4096 --tg 128 --concurrency 1 2 4 8 --runs 1 > logs/bd-mtp.run 2>&1
grep -E "\[code\].*trial|STOP|LEFT|ready=0" logs/bd-mtp.run | cut -c1-170
grep -E "MTP batched drafting|scheduler:" logs/bd-mtp.server.log | head -3 | cut -c1-160
grep -E "spec timing" logs/bd-mtp.server.log | grep rank0 | tail -4 | cut -c1-220
grep -E "^\| /" logs/bd-mtp.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-14s %-18s\n", $3, $4}'
grep -iE "request failed|exception|throw|abort" logs/bd-mtp.server.log | head -5 | cut -c1-200
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo "== $(date +%T) BD_DONE"
} > logs/bd.outer 2>&1
