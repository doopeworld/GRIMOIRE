#!/bin/bash
# 11-06: what activation statistics buy Qwen3.8-27B converted from its BF16 original at load, one card (gpu0),
# dev build grimoire-runs/dev-1106 (run_imx_build_1106.sh).  Same reference and tokens as the 10-29 table
# (tools/bench/accuracy/README.md): ref-bf16.bin = the BF16 original in transformers on the CPU, 256 tokens of
# "A Scandal in Bohemia" (sherlock.ids), fed one at a time through the decode path (grimoire --score-ids).
# Unsloth's imatrix for this model (imatrix_unsloth.gguf, 1.15M calibration tokens) is copied to
# $A/imatrix-qwen38-27b.gguf first.  10-29: --proj int4 88.3% top-1, --proj int8 98.4%, UD-Q4_K_XL native 94.5%.
#   runs: int4 (same-binary baseline), + GRIMOIRE_IMATRIX, + groups of 64, + the input-norm fold (alpha 0.4),
#         + both; int8 and int8 + fold (alpha 0.2: per-row INT8 wants a gentler fold -- offline, layers 10-14,
#         the input projections' error 0.26-0.62 of plain INT8 at 0.2, while 0.4 made in_proj_z 1.3x WORSE).
#         Holds HOLD-imxacc; gpu0 lock.
set -u
B=/mnt/storage/isos/grimoire-runs/bench-1022
X=/mnt/storage/isos/grimoire-runs/dev-1106
A=/mnt/storage/Models/grimoire-acc
M=/models2/Qwen3.8-27B
cd "$B" || exit 9
exec >>"$B/logs/imx-acc-1106.out" 2>&1
echo "IMX_ACC_START $(date +%T)"
[ -x "$X/bin/grimoire" ] || { echo "IMX_ACC_STOP no dev build"; exit 9; }
for f in ref-bf16.bin sherlock.ids imatrix-qwen38-27b.gguf cmp_logits.py; do
  [ -s "$A/$f" ] || { echo "IMX_ACC_STOP missing $A/$f"; exit 9; }
done
until [ -z "$(docker ps --format "{{.Names}}" | grep -v "^grim-build-")" ] && ! ls "$B"/logs/HOLD-* >/dev/null 2>&1; do sleep 10; done
touch logs/HOLD-imxacc
trap 'rm -f "$B/logs/HOLD-imxacc"' EXIT
cd /mnt/storage/isos/grimoire-fuse || exit 9
run() {  # name proj extra-env (one K=V per line)
  echo "== $(date +%T) score $1"
  EXTRA_ENV="GRIMOIRE_POWER_LIMIT_W=0
$3" GRIM_TREE=$X LIM=1500 TAILN=6 \
    flock /mnt/storage/isos/grimoire-runs/gpu0.lock bash tools/g0run.sh "imx-$1" /grimoire/bin/grimoire -m "$M" --proj "$2" \
      --ctx 1024 --score-ids /models/grimoire-acc/sherlock.ids --score-out "/models/grimoire-acc/q27-$1.bin"
  grep -h "imatrix" "/tmp/grim-imx-$1.log" 2>/dev/null | sed 's/^/   /'
  ls -la "$A/q27-$1.bin" 2>&1 | cut -c1-120
}
IM=GRIMOIRE_IMATRIX=/models/grimoire-acc/imatrix-qwen38-27b.gguf
run int4          int4 ""
run int4-imx      int4 "$IM"
run int4-imx-g64  int4 "$IM
GRIMOIRE_INT4_RTN_GROUP=64"
run int4-fold     int4 "$IM
GRIMOIRE_IMATRIX_FOLD=0.4"
run int4-fold-g64 int4 "$IM
GRIMOIRE_IMATRIX_FOLD=0.4
GRIMOIRE_INT4_RTN_GROUP=64"
run int8          int8 ""
run int8-fold     int8 "$IM
GRIMOIRE_IMATRIX_FOLD=0.2"
cd "$B"
for n in int4 int4-imx int4-imx-g64 int4-fold int4-fold-g64 int8 int8-fold; do
  [ -s "$A/q27-$n.bin" ] && docker run --rm --network none --cpuset-cpus 6-13 -v "$A":/acc grimoire-ref:cpu \
    python3 /acc/cmp_logits.py /acc/ref-bf16.bin "/acc/q27-$n.bin" 248320 "Qwen3.8-27B $n"
done

# Flash-Next, one card: the dense part at INT4 rounded by Unsloth's Flash-Next imatrix, against GRIMOIRE's own
# BF16 dense part (fn-bf16.bin, 11-05; experts NVFP4 in every run, frozen placement) -- 11-05: int4 96.9% top-1.
# (No fold: Flash-Next has no input norm before its projections, the hyper-connections stand there.)
PD=/mnt/storage/Models/grimoire-ple
if [ -s "$A/imatrix-flash-next.gguf" ] && [ -s "$A/fn-bf16.bin" ] && [ -s "$A/fn512.ids" ]; then
  cd /mnt/storage/isos/grimoire-fuse || exit 9
  for v in int4 int4-imx int4-imx-g64; do
    cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
    case $v in int4) E="";; int4-imx) E="GRIMOIRE_IMATRIX=/models/grimoire-acc/imatrix-flash-next.gguf";;
      int4-imx-g64) E="GRIMOIRE_IMATRIX=/models/grimoire-acc/imatrix-flash-next.gguf
GRIMOIRE_INT4_RTN_GROUP=64";; esac
    echo "== $(date +%T) score fn-$v"
    EXTRA_ENV="GRIMOIRE_POWER_LIMIT_W=0
GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin
GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits.ab
GRIMOIRE_EXPERT_VRAM_PER_LAYER=90
$E" GRIM_TREE=$X LIM=1500 TAILN=6 \
      flock /mnt/storage/isos/grimoire-runs/gpu0.lock bash tools/g0run.sh "imx-fn-$v" /grimoire/bin/grimoire \
        -m /models2/Qwen3.8-Flash-Next-NVFP4 --proj int4 --ctx 1024 \
        --score-ids /models/grimoire-acc/fn512.ids --score-out "/models/grimoire-acc/fn-imx-$v.bin"
    grep -h "imatrix" "/tmp/grim-imx-fn-$v.log" 2>/dev/null | sed 's/^/   /'
  done
  cp "$PD/flash-next.hits.frozen-1028" "$PD/flash-next.hits.ab"
  cd "$B"
  for v in int4 int4-imx int4-imx-g64; do
    [ -s "$A/fn-imx-$v.bin" ] && docker run --rm --network none --cpuset-cpus 6-13 -v "$A":/acc grimoire-ref:cpu \
      python3 /acc/cmp_logits.py /acc/fn-bf16.bin "/acc/fn-imx-$v.bin" 248320 "Flash-Next dense $v vs bf16"
  done
fi
echo "IMX_ACC_END $(date +%T)"
