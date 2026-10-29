#!/bin/bash
# 10-29: GGUF accuracy against the BF16 original (the morning's reference ref-bf16.bin, 256 tokens of
# "A Scandal in Bohemia", transformers on the CPU).  gpu0 only, main eb6526f (GGUF merged), host bin.
#   Qwen3.8-27B Q8_0 (Unsloth) as --proj int8 and --proj mxfp4; UD-Q4_K_XL as --proj mxfp4 and native.
A=/mnt/storage/Models/grimoire-acc
R=/mnt/storage/isos/grimoire-fuse
L=/mnt/storage/isos/grimoire-runs/bench-1022/logs
G=/models2/unsloth-Qwen3.8-27B-GGUF
echo "$(date '+%F %T') acc_gguf.sh (gpu0: 4 GGUF score runs + compare)" >> $L/claude-commands-1028.log
{
run() {  # name model proj
  date "+%T grimoire $1"
  (cd $R && flock /mnt/user/isos/grimoire-runs/gpu0.lock env LIM=900 bash tools/g0run.sh acc-$1 -m $2 --proj $3 --ctx 1024 \
      --score-ids /models/grimoire-acc/sherlock.ids --score-out /models/grimoire-acc/g-$1.bin > /dev/null) \
    || { echo "STOP: run $1 failed"; tail -6 /tmp/grim-acc-$1.log; return 1; }
  grep -h "SCORE:" /tmp/grim-acc-$1.log | sed 's/^/   /'
}
run gguf-q8-int8     $G/Qwen3.8-27B-Q8_0.gguf       int8
run gguf-q8-mxfp4    $G/Qwen3.8-27B-Q8_0.gguf       mxfp4
run gguf-udq4-mxfp4  $G/Qwen3.8-27B-UD-Q4_K_XL.gguf mxfp4
run gguf-udq4-native $G/Qwen3.8-27B-UD-Q4_K_XL.gguf native
for n in gguf-q8-int8 gguf-q8-mxfp4 gguf-udq4-mxfp4 gguf-udq4-native; do
  [ -s $A/g-$n.bin ] && docker run --rm --cpuset-cpus 6-13 -v $A:/acc grimoire-ref:cpu \
    python /acc/cmp_logits.py /acc/ref-bf16.bin /acc/g-$n.bin 248320 $n
done
echo "== $(date +%T) ACCG_DONE"
} > $L/acc_gguf.outer 2>&1
