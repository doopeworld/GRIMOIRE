#!/bin/bash
# 10-29: native GGUF dense ESIMD GEMV -- (a) Q4_K / Q6_K past K = 16384 (ffn_down 17408), (b) new
# Q5_K kernel (GRIMOIRE_GGUF_ESGEMV_Q5K=0 = off).  Build main + both, then gpu0, Qwen3.8-27B
# UD-Q4_K_XL --proj native:
#   1. decode timelines: Q5K=0 (only (a)), default ((a)+(b))  [before today: 79.9 ms a token]
#   2. 64-token answers: GRIMOIRE_GGUF_ESGEMV=0 (the generic kernels) vs default -- texts compared
#   3. accuracy score vs the BF16 original (afternoon, generic: top-1 94.5%, KL 0.063)
B=/mnt/storage/isos/grimoire-runs/bench-1022
R=/mnt/storage/isos/grimoire-fuse
A=/mnt/storage/Models/grimoire-acc
L=$B/logs
G=/models2/unsloth-Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
NL='
'
echo "$(date '+%F %T') nq2.sh (build native esgemv K>16384 + Q5_K; gpu0 timelines, texts, score)" >> $L/claude-commands-1028.log
g0() { (cd $R && flock /mnt/user/isos/grimoire-runs/gpu0.lock env EXTRA_ENV="$2" LIM=900 bash tools/g0run.sh $1 "${@:3}") || { echo "STOP $1"; exit 9; }; }
{
date '+%T build'
bash /mnt/user/isos/grimoire-runs/bench-1022/tower_build.sh all > /tmp/nq2b.txt 2>&1 || { echo "STOP: build failed"; grep -B2 -A8 "error:" /tmp/nq2b.txt | head -40; exit 1; }
tail -1 /tmp/nq2b.txt
for v in q5off def; do
  e="GRIMOIRE_TIMELINE=1${NL}GRIMOIRE_DECODE_GRAPH=0${NL}GRIMOIRE_TIMELINE_LAYER=0"
  [ $v = q5off ] && e="$e${NL}GRIMOIRE_GGUF_ESGEMV_Q5K=0"
  echo "== $(date +%T) timeline $v"
  g0 nq2-tl-$v "$e" -m $G --proj native --ctx 2048 -p "Explain in two sentences why the sky is blue." -n 32
  cp /tmp/grim-nq2-tl-$v.log $L/nq2-tl-$v.log
  grep -E "device span|sum of" $L/nq2-tl-$v.log
  python3 $B/layer_types.py /tmp/udq4.types $L/nq2-tl-$v.log | tail -10
done
for v in generic def; do
  e=""; [ $v = generic ] && e="GRIMOIRE_GGUF_ESGEMV=0"
  g0 nq2-txt-$v "$e" -m $G --proj native --ctx 2048 -p "Write a Python function that returns the n-th Fibonacci number iteratively." -n 64
  awk '/norms /{f=1;next} f' /tmp/grim-nq2-txt-$v.log | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids|prompt attention|GGUF' > /tmp/nq2-txt-$v.txt
  echo "text $v: $(grep -oE 'elapsed=[0-9.]+s' /tmp/grim-nq2-txt-$v.log | tail -1) $(md5sum < /tmp/nq2-txt-$v.txt | cut -c1-12) $(tr '\n' ' ' < /tmp/nq2-txt-$v.txt | cut -c1-140)"
done
cmp -s /tmp/nq2-txt-generic.txt /tmp/nq2-txt-def.txt && echo "texts IDENTICAL generic vs ESIMD" || echo "texts DIFFER generic vs ESIMD"
g0 acc-nq2 "" -m $G --proj native --ctx 1024 --score-ids /models/grimoire-acc/sherlock.ids --score-out /models/grimoire-acc/g-gguf-udq4-native-nq2.bin
grep -h "SCORE:" /tmp/grim-acc-nq2.log
docker run --rm --cpuset-cpus 6-13 -v $A:/acc grimoire-ref:cpu python /acc/cmp_logits.py /acc/ref-bf16.bin /acc/g-gguf-udq4-native-nq2.bin 248320 udq4-native-nq2
echo "== $(date +%T) NQ2_DONE"
} > $L/nq2.outer 2>&1
