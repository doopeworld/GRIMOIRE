#!/bin/bash
# 10-28: Flash-Next MTP head (mtp4), step 1 = shadow mode.  At every decode step the head drafts
# the next token; the log counts how often the target then produced it (GRIMOIRE_MTP4_SHADOW=1).
# No speculation: the answers stay the target's.  The head takes ~1.5 GB a card, so 160 experts
# a layer stay in VRAM instead of 170 (a different VRAM / RAM split: the text may differ from the
# 170-expert runs, the same way it drifts with the placement).
#   arm 1: full draft vocabulary; arm 2: GRIMOIRE_MTP_DRAFT_VOCAB=65536 (rank 0 alone)
S=/mnt/user/isos/grimoire-runs/bench-1022; R=/mnt/storage/isos/grimoire-fuse
cd $S || exit 2
O=$S/logs/fn-mtp1.outer
[ -z "$(docker ps -q --filter ancestor=grimoire-b70:latest)" ] || { echo "a GPU container is up" > $O; exit 1; }
t0=$(date +%s)
(cd $R && bash tools/dev_build.sh server) > logs/build-mtp1.log 2>&1
if ! grep -q "built: bin/grimoire-server" logs/build-mtp1.log || [ "$(stat -c %Y $R/bin/grimoire-server)" -le "$t0" ]; then
  { echo BUILD_FAILED; grep -E "error:" -A 3 logs/build-mtp1.log | head -30; } > $O; exit 1
fi
PD=/mnt/storage/Models/grimoire-ple
B="GRIMOIRE_POWER_LIMIT_W=0 GRIMOIRE_EXPERT_VRAM_PER_LAYER=160"
F=/models2/Qwen3.8-Flash-Next-NVFP4
RP="RP_WIN=1000 RP_RUN=1000000"
{
for v in full: v64k:GRIMOIRE_MTP_DRAFT_VOCAB=65536; do
  t=${v%%:*}; e=${v#*:}
  cp $PD/flash-next.hits.frozen-1028 $PD/flash-next.hits.ab
  env $RP LOAD_MAX=1500 CLIENT_CPUS=14,15 ./dual.sh fn-mtp1-$t TP $F bf16 65536 \
    "$B GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab GRIMOIRE_MTP=1 GRIMOIRE_MTP4_SHADOW=1 $e" \
    pleready,gen > logs/fn-mtp1-$t.run 2>&1
  echo "== fn-mtp1-$t [$e]"
  grep -E "\[gen\]|STOP|LEFT|ready=0" logs/fn-mtp1-$t.run | cut -c1-160
  grep -E "mtp4 head|speculation|MTP4 shadow|error|failed" logs/fn-mtp1-$t.server.log | head -12 | cut -c1-200
  sleep 3
done
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo FN_MTP1_DONE
} > $O 2>&1
