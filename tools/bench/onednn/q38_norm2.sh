#!/bin/bash
# 10-28: build with the split RMSNorm's partials loaded up front (opt-in: B70_NORM_PREG=1), then on
# Qwen3.8-27B-FP8 two cards: (1) one-slot timeline + answers (compared with q38-tl1, same prompts),
# (2) llama-benchy pp4096 tg128, one user, 8 slots, old vs new.
S=/mnt/user/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-fuse
cd $S || exit 2
O=$S/logs/q38-norm.outer
[ -z "$(docker ps -q --filter ancestor=grimoire-b70:latest)" ] || { echo "a GPU container is up" > $O; exit 1; }
t0=$(date +%s)
(cd $R && bash tools/dev_build.sh server) > logs/build-norm.log 2>&1
if ! grep -q "built: bin/grimoire-server" logs/build-norm.log || [ "$(stat -c %Y $R/bin/grimoire-server)" -le "$t0" ]; then
  { echo BUILD_FAILED; grep -E "error:" -A 3 logs/build-norm.log | head -30; } > $O; exit 1
fi
cp $R/bin/grimoire-server $S/grimoire-server.norm
RP="RP_WIN=1000 RP_RUN=1000000"
M=/models2/Qwen3.8-27B-FP8
{
env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh q38-tl2 TP $M fp8 18432 \
  "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0 GRIMOIRE_SEQ_SLOTS=1 GRIMOIRE_TIMELINE=1 B70_NORM_PREG=1" gen > logs/q38-tl2.run 2>&1
echo "== q38-tl2 (one slot, timeline)"
grep -E "\[gen\]|STOP|LEFT" logs/q38-tl2.run | cut -c1-160
grep -E "^\[rank0\]   (TOTAL|    in_norm|    post_norm|lm_head)" logs/q38-tl2.server.log | head -8 | cut -c1-90
python3 - <<'EOF'
import json
a = json.load(open("logs/q38-tl1.gen.json")); b = json.load(open("logs/q38-tl2.gen.json"))
for k in a: print("  %-8s %s" % (k, "SAME as before" if a[k]["text"] == b.get(k, {}).get("text") else "DIFFERENT"))
EOF
sleep 3
for v in old:"" new:"B70_NORM_PREG=1"; do
  t=${v%%:*}; e=${v#*:}
  env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh q38-n$t TP $M fp8 18432 \
    "GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_MTP=0 GRIMOIRE_SEQ_SLOTS=8 $e" benchy --pp 4096 --tg 128 --runs 2 > logs/q38-n$t.run 2>&1
  echo "== q38-n$t [$e]"
  grep -E "STOP|LEFT|ready=0" logs/q38-n$t.run | cut -c1-160
  grep -E "^\| /" logs/q38-n$t.benchy.md 2>/dev/null | awk -F'|' '{printf "     %-8s %-18s\n", $3, $4}'
  sleep 3
done
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo Q38_NORM_DONE
} > $O 2>&1
