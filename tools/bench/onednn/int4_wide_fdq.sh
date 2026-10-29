#!/bin/bash
# 10-29: main 91d43d5 + INT4 small-M fix (9..16 rows through the fast-dequant kernel).
# gpu0 ONLY.  Build, then the GRIMOIRE-QWEN3.8 settings (GPTQ Int4 + MTP, draft vocab 64K, 8 slots):
#   a. K=6, old path past 8 rows (GRIMOIRE_I4_WIDE_FDQ=0)  b. K=6, fix  c. K=8, fix
#   each: coding probe, prose probe, 4 coding answers at once, llama-benchy pp4096 / tg128 c1 / c2 / c4
B=/mnt/storage/isos/grimoire-runs/bench-1022
cd $B || exit 2
echo "$(date '+%F %T') fx1.sh (build main + int4 wide fdq; gpu0: K=6 old/new, K=8 new)" >> logs/claude-commands-1028.log
{
date '+%T build (branch bd-main + fix)'
[ "$(cd /mnt/storage/isos/grimoire-fuse && git branch --show-current)" = bd-main ] || { echo "STOP: not on bd-main"; exit 1; }
bash /mnt/user/isos/grimoire-runs/bench-1022/tower_build.sh all > /tmp/fx1b.txt 2>&1 || { echo "STOP: build failed"; grep -B2 -A8 "error:" /tmp/fx1b.txt | head -40; exit 1; }
tail -1 /tmp/fx1b.txt
RP="RP_WIN=1000 RP_RUN=1000000"
run() {  # name extra_env
  echo "== $(date +%T) $1 [$2]"
  env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh $1 G0 /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 \
    "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_DRAFT_VOCAB=65536 $2" \
    code,prose,conc4,benchy --pp 4096 --tg 128 --concurrency 1 2 4 --runs 2 > logs/$1.run 2>&1
  grep -E "\[(code|prose)\].*trial|\[conc4\] round [01]:|CONC|FAIL|STOP|LEFT|ready=0" logs/$1.run | cut -c1-170
  grep -E "^\| /" logs/$1.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-14s %-18s\n", $3, $4}'
  grep -iE "request failed|exception|throw|abort" logs/$1.server.log | head -5 | cut -c1-200
  grep -q STOP logs/$1.run && exit 9
  sleep 3
}
run fx1-k6-old "GRIMOIRE_MTP_K=6 GRIMOIRE_I4_WIDE_FDQ=0"
run fx1-k6-new "GRIMOIRE_MTP_K=6"
run fx1-k8-new "GRIMOIRE_MTP_K=8"
python3 - logs/fx1-k6-new.conc4.json logs/fx1-k6-old.conc4.json <<'EOF'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
same = sum(1 for k in a if k in b and a[k]['text'] == b[k]['text'])
print(f"conc4: {same}/{len(a)} answers identical fix vs old (K=6)")
EOF
echo "== $(date +%T) FX1_DONE"
} > logs/fx1.outer 2>&1
