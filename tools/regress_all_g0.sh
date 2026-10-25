#!/bin/bash
# regress_all_g0.sh -- the pure-mode regression pass, gpu0 ONLY.
# Every run goes through tools/g0run.sh: it refuses any GPU but 03:00.0 and
# exits 9 at the first new AER / xe error, which stops this whole script.
# Prints one summary block per model; full logs land in $OUT.
set -u
cd /mnt/storage/isos/grimoire-fuse
OUT=/mnt/storage/isos/grimoire-runs/regress-$(date +%m%d-%H%M); mkdir -p "$OUT"; echo "results: $OUT"
SHORT="Explain in two sentences why the sky is blue."
STORY="Write a detailed story about a lighthouse keeper."
LONG="$(cat real4k_ascii.txt)"
# Checkpoints moved to the new disk (/mnt/models, mounted at /models2 by b70run.sh since
# 2026-10-22) are found there, by their top directory, in -m and in the env strings.
MOVED=$(for d in /mnt/models/*/; do b=$(basename "$d"); [ -e "/mnt/storage/Models/$b" ] || echo "$b"; done)
fixp() { local s=$1 b; for b in $MOVED; do s=$(printf '%s' "$s" | sed -E "s#/models/(${b//./\\.})(/|\$|[[:space:]])#/models2/\\1\\2#g"); done; printf '%s' "$s"; }
# SUITE_ENV (newline-separated K=V) is added to every run's env, e.g. SUITE_ENV=GRIMOIRE_ONEDNN=1.
run() { # name model proj n prompt [env]
  local nm=$1 m=$2 pr=$3 n=$4 p=$5 ex=${6:-} r
  ex=$(fixp "$ex${SUITE_ENV:+
$SUITE_ENV}")
  r=$(EXTRA_ENV="$ex" LIM=900 bash tools/g0run.sh rx-$nm -m "$(fixp "/models/$m")" --proj $pr --ctx 8192 -p "$p" -n $n) \
    || { echo "STOP: gpu0 guard tripped (or run failed) at $nm: $r"; cp /tmp/grim-rx-$nm.log "$OUT/$nm.log" 2>/dev/null; exit 9; }
  cp /tmp/grim-rx-$nm.log "$OUT/$nm.log"
  echo "  $nm: ${r:-NO RESULT LINE}"
}
txt() { awk '/norms /{f=1;next} f' "$OUT/$1.log" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
show() { echo "    text: $(txt $1 | tr '\n' ' ' | cut -c1-200)"; }
el() { grep -oE "elapsed=[0-9.]+" "$OUT/$1.log" | tail -1 | cut -d= -f2; }
tps() { python3 -c "a=float('$(el $1)');b=float('$(el $2)');per=(b-a)/$3;print(f'{per*1000:.2f} ms/token = {1/per:.1f} tok/s')"; }
same() { cmp -s <(txt $1) <(txt $2) && echo "IDENTICAL" || echo "differs"; }

Q=Qwen3.8-27B-MXFP4-GRIMOIRE
echo "== Qwen3.8-27B MXFP4 (baseline: pp4096 1.972-1.976 s; Sherlock n=24 == ref)"
run q-pp1 $Q mxfp4 1 "$(cat p4096.txt)"
run q-pp2 $Q mxfp4 1 "$(cat p4096.txt)"
run q-c1 $Q mxfp4 24 "$LONG"
cmp -s <(txt q-c1) ref-sherlock-5987-n24.txt && echo "    Sherlock n=24: IDENTICAL to ref-sherlock-5987-n24.txt" \
  || { echo "    Sherlock n=24: DIFFERS from reference"; diff <(txt q-c1) ref-sherlock-5987-n24.txt | head -4; }
run q-d32 $Q mxfp4 32 "$STORY"
run q-d160 $Q mxfp4 160 "$STORY"
echo "    plain decode: $(tps q-d32 q-d160 128)"; show q-d160

QM=Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16
MTP="GRIMOIRE_MTP=1
GRIMOIRE_MTP_K=3
GRIMOIRE_MTP_DRAFT_VOCAB=131072
GRIMOIRE_SPEC_STATS=1"
echo "== MTP ($QM)"
run m-p160 $QM mxfp4 160 "$STORY"
run m-32 $QM mxfp4 32 "$STORY" "$MTP"
run m-160 $QM mxfp4 160 "$STORY" "$MTP"
echo "    mtp decode: $(tps m-32 m-160 128)  text vs plain: $(same m-p160 m-160)"
grep -h "spec:" "$OUT/m-160.log" | head -2 | sed 's/^/    /'
run m-e32 $QM mxfp4 32 "$STORY" "$MTP
GRIMOIRE_MTP_EXACT_VERIFY=1"
run m-e160 $QM mxfp4 160 "$STORY" "$MTP
GRIMOIRE_MTP_EXACT_VERIFY=1"
echo "    mtp exact: $(tps m-e32 m-e160 128)  text vs plain: $(same m-p160 m-e160)"
grep -h "spec:" "$OUT/m-e160.log" | head -2 | sed 's/^/    /'

DF="GRIMOIRE_DFLASH_MODEL=/models/Qwen3.8-27B-DFlash2
GRIMOIRE_SPEC_STATS=1"
echo "== DFlash2 on Qwen3.8-27B"
run f-32 $Q mxfp4 32 "$STORY" "$DF"
run f-160 $Q mxfp4 160 "$STORY" "$DF"
echo "    dflash decode: $(tps f-32 f-160 128)  text vs plain: $(same q-d160 f-160)"
grep -h "spec:\|DFlash\|dflash" "$OUT/f-160.log" | grep -v unavailable | head -3 | sed 's/^/    /'

O=Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
echo "== Ornith-1.5-35B-A3B MXFP4"
run o-s $O mxfp4 64 "$SHORT"; show o-s
run o-l $O mxfp4 24 "$LONG"; show o-l
run o-d32 $O mxfp4 32 "$STORY"
run o-d160 $O mxfp4 160 "$STORY"
echo "    plain decode: $(tps o-d32 o-d160 128)"
ODF="GRIMOIRE_DFLASH_MODEL=/models/Ornith-1.5-35B-A3B-DFlash2
GRIMOIRE_SPEC_STATS=1"
run of-32 $O mxfp4 32 "$STORY" "$ODF"
run of-160 $O mxfp4 160 "$STORY" "$ODF"
echo "    dflash2 decode: $(tps of-32 of-160 128)  text vs plain: $(same o-d160 of-160)"
grep -h "spec:" "$OUT/of-160.log" | head -2 | sed 's/^/    /'

MU=Muse-Glimmer-30B-INT4-W4A16
echo "== Muse-Glimmer-30B INT4-W4A16"
run mu-s $MU int4 64 "$SHORT"; show mu-s
run mu-l $MU int4 24 "$LONG"; show mu-l

echo "== Agnes-3.0-Flash (baseline long 4.691 s)"
run a-s Agnes-3.0-Flash mxfp4 64 "$SHORT"; show a-s
run a-l Agnes-3.0-Flash mxfp4 24 "$LONG"; show a-l

echo "== INT4 W4A16 checkpoints (group 64 AutoRound GPTQ; fixed f8b7198)"
run w-s Qwen3.8-27B-W4A16 int4 64 "$SHORT"; show w-s
run w-l Qwen3.8-27B-W4A16 int4 24 "$LONG"; show w-l

echo "== NVFP4 / FP8 checkpoints (short)"
run n-orn Ornith-1.5-35B-A3B-NVFP4 mxfp4 64 "$SHORT"; show n-orn
run n-q Qwen3.8-27B-NVFP4 mxfp4 64 "$SHORT"; show n-q
run n-q8 Qwen3.8-27B-FP8 mxfp4 64 "$SHORT"; show n-q8

FN="GRIMOIRE_EXPERT_VRAM_PER_LAYER=112
GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin"
echo "== Qwen3.8-Flash-Next-NVFP4, tiered VRAM+RAM+SSD (previous long 9.2 s)"
run fn-s Qwen3.8-Flash-Next-NVFP4 bf16 64 "$SHORT" "$FN"; show fn-s
run fn-l Qwen3.8-Flash-Next-NVFP4 bf16 24 "$LONG" "$FN"; show fn-l
run fn-h Qwen3.8-Flash-Next-NVFP4 bf16 64 "$SHORT" "$FN
GRIMOIRE_EXPERT_HITS=/models/grimoire-ple/flash-next.hits"
echo "    hot-expert placement (hits file): short $(el fn-h) s vs $(el fn-s) s  text: $(same fn-s fn-h)"
echo "== Full zoo sweep 2026-10-02: every remaining checkpoint in /models, short prompt"
run z-mugptq  Muse-Glimmer-30B-GPTQ-INT4             int4 40 "$SHORT"
run z-mumx    Muse-Glimmer-30B-MXFP4                 mxfp4 40 "$SHORT"
run z-ornbare Ornith-1.5-35B-A3B                     mxfp4 40 "$SHORT"
run z-orngptq Ornith-1.5-35B-A3B-GPTQ-Int4           int4 40 "$SHORT"
run z-ornar   Ornith-1.5-35B-A3B-INT4-W4A16-AutoRound int4 40 "$SHORT"
run z-ornmtpfx Ornith-1.5-35B-A3B-MTPFIX             mxfp4 40 "$SHORT"
run z-q36gptq Qwen3.6-35B-A3B-GPTQ-Int4              int4 40 "$SHORT"
run z-q38bare Qwen3.8-27B                            mxfp4 40 "$SHORT"
run z-q38gm   Qwen3.8-27B-GPTQ-Int4-MTP-BF16         int4 40 "$SHORT"
run z-q38ar   Qwen3.8-27B-MXFP4-AutoRound/Qwen3.8-27B-mxfp-w4g32 mxfp4 40 "$SHORT"
run z-q38i4ar Qwen3.8-27B-int4-AutoRound             int4 40 "$SHORT"
# Qwen3.8-27B-int4-ov is skipped on purpose: OpenVINO IR format (.xml/.bin,
# vision-capable), the vLLM/OpenVINO int4-ov reference baseline this project
# measures ITSELF against (see grimoire-vllm-baseline-measured) -- not a
# GRIMOIRE-loadable checkpoint and never will be (no vLLM/OpenVINO, ever).
echo "== DFlash draft pairings not otherwise covered above"
run z-q36df Qwen3.6-35B-A3B-GPTQ-Int4 int4 48 "$STORY" "GRIMOIRE_DFLASH_MODEL=/models/Qwen3.6-35B-A3B-DFlash
GRIMOIRE_SPEC_STATS=1"
grep -h "spec:" "$OUT/z-q36df.log" | tail -1 | sed 's/^/    /'
run z-q35df Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE mxfp4 48 "$STORY" "GRIMOIRE_DFLASH_MODEL=/models/Qwen3.5-35B-A3B-DFlash
GRIMOIRE_SPEC_STATS=1"
grep -h "spec:" "$OUT/z-q35df.log" | tail -1 | sed 's/^/    /'
# K2-Horizon-MoVA-36B-A4B: own architecture (K2), own sweep script
# (tools/k2sweep.sh); smoke-tested here so a shared-code regression catches
# it too (it is NOT a Qwen3.5-MoE derivative and shares no weights with
# anything else in this list -- it caught real bugs in 2026-09-28 that a
# Qwen-family-only regression would have missed entirely).
run z-k2 K2-Horizon-MoVA-36B-A4B mxfp4 64 "$SHORT"; show z-k2
echo "ALL DONE"
