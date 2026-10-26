#!/bin/bash
# Ian 10-26: llama-benchy speed table, Ornith / Qwen3.8 / Flash-Next, two cards and one card, today's
# code (= image grimoire-b70:v1.10-dev2), each with its Unraid container's settings. pp2048 + pp4096,
# tg128, c1, --runs 2.
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
B="GRIMOIRE_POWER_LIMIT_W=0"
Q=/models2/Qwen3.8-27B-FP8
O2=/models/Ornith-1.5-35B-A3B-FP8
F=/models2/Qwen3.8-Flash-Next-NVFP4
PLE="GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
BEN="benchy --pp 2048 4096 --tg 128 --runs 2"
run() { ./dual.sh "$@" 2>&1; [ -e logs/$1.STOP ] && echo "STOPPED at $1 (guard) -- continuing with the next one"; sleep 3; }
# two cards
run st-q38-tp   TP $Q fp8 16384 "$B GRIMOIRE_MTP=0" $BEN
run st-q38-mtp  TP $Q fp8 16384 "$B GRIMOIRE_MTP=1" $BEN
run st-q38-df   TP $Q fp8 16384 "$B GRIMOIRE_MTP=0 GRIMOIRE_DFLASH_MODEL=/models2/Qwen3.8-27B-DFlash2" $BEN
run st-orn-pp   PP $O2 fp8 32000 "$B GRIMOIRE_PP_SPLIT=20 GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_DEFER_MOE_GATHER=1 GRIMOIRE_BF16_QKV=1 GRIMOIRE_BF16_DN_QKV=1" $BEN
run st-fn-pp    PP $F bf16 65536 "$B GRIMOIRE_PP_SPLIT=24 GRIMOIRE_EXPERT_VRAM_PER_LAYER=360 $PLE" code,$BEN
# one card (gpu0)
run st-orn-g0   G0 /models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE mxfp4 32000 "$B GRIMOIRE_SEQ_SLOTS=8" $BEN
run st-q38-g0   G0 /models2/Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4 16384 "$B GRIMOIRE_SEQ_SLOTS=8 GRIMOIRE_MTP=1 GRIMOIRE_MTP_K=4" $BEN
run st-fn-g0    G0 $F bf16 65536 "$B GRIMOIRE_EXPERT_VRAM_PER_LAYER=112 $PLE" code,$BEN
for p in $(pgrep -f "http[.]server 8999"); do kill $p; done
echo ALL_DONE
