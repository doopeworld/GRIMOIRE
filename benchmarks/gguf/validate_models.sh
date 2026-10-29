#!/bin/bash
# validate_models.sh -- "download from Hugging Face and run": GRIMOIRE on bare .gguf FILES (no
# tokenizer.json / config.json next to them), gpu0 only, through tools/b70run.sh.
#
#   validate_models.sh gate  ST_DIR GGUF_FILE [PROJ]   token-identity gate: safetensors twin vs GGUF
#   validate_models.sh run   GGUF_FILE [PROJ]          banner + 2 prompts, text saved for reading
#
# Paths: ST_DIR under /mnt/storage/Models, GGUF_FILE under /mnt/models (both as seen on the host).
# Every run prints the generated text: read it (CLAUDE.md rule 8).
set -u
MODE=${1:?gate|run}; shift
REPO=$(cd "$(dirname "$0")/../.." && pwd)
NODE=$(bash "$REPO/tools/gpunode.sh" gpu0) || exit 2
OUTROOT=/mnt/user/isos/grimoire-runs/gguf-validate; mkdir -p "$OUTROOT"
PROMPTS=("Explain in three sentences why the sky is blue."
         "Write a Python function that returns the n-th Fibonacci number iteratively.")
ENVS="GRIMOIRE_PRINT_IDS=1
ZE_AFFINITY_MASK=0"

one() {  # one TAG MODELS_ROOT MODEL_PATH_IN_CONTAINER PROJ PROMPT_INDEX OUTDIR
  local tag=$1 root=$2 path=$3 proj=$4 i=$5 out=$6
  GRIM_ENV="$ENVS" GRIM_TREE="$REPO" GRIM_MODELS="$root" \
    bash "$REPO/tools/b70run.sh" "$NODE" 3600 "val-$tag-$i" \
    /grimoire/bin/grimoire -m "$path" --proj "$proj" --ctx 4096 -p "${PROMPTS[$i]}" -n 64 \
    > "$out/$tag-$i.launcher" 2>&1
  local rc=$?
  cp "/tmp/grim-val-$tag-$i.log" "$out/$tag-$i.log" 2>/dev/null
  docker rm "grim-val-$tag-$i" >/dev/null 2>&1
  return $rc
}
ids()  { grep -h "^greedy ids" "$1" 2>/dev/null | cut -d: -f2; }
text() { awk '/^prompt ids/{f=1;next} /^greedy ids/{f=0} f' "$1" 2>/dev/null |
         grep -vE "unavailable|^    |^  [a-zA-Z]" | tr '\n' ' ' | cut -c1-240; }

case $MODE in
gate)
  ST=${1:?st dir}; GG=${2:?gguf file}; PROJ=${3:-mxfp4}
  OUT=$OUTROOT/gate-$(basename "$GG" .gguf)-$PROJ-$(date +%H%M%S); mkdir -p "$OUT"
  echo "GATE $ST  vs  $GG  (--proj $PROJ)  gpu0=$NODE  commit=$(git -C "$REPO" rev-parse --short HEAD)"
  pass=0
  for i in 0 1; do
    one st /mnt/storage/Models "/models/$ST" "$PROJ" $i "$OUT"; rs=$?
    one gg /mnt/models "/models/${GG#/mnt/models/}" "$PROJ" $i "$OUT"; rg=$?
    a=$(ids "$OUT/st-$i.log"); b=$(ids "$OUT/gg-$i.log")
    if [ $rs = 0 ] && [ $rg = 0 ] && [ -n "$a" ] && [ "$a" = "$b" ]; then
      echo "  prompt $i: IDENTICAL ($(echo $a | wc -w) tokens)"; pass=$((pass+1))
    else echo "  prompt $i: DIFFERENT (rc st=$rs gguf=$rg)"; fi
  done
  grep -hE "GGUF (source|tokens|tokenizer|round-trip)|ships no MTP|tokenizer from GGUF|sidecar" "$OUT/gg-1.log" | sort -u | cut -c1-200
  echo "  text: $(text "$OUT/gg-0.log")"
  echo "GATE RESULT: $pass/2 identical -> $OUT"
  [ $pass = 2 ] ;;
run)
  GG=${1:?gguf file}; PROJ=${2:-mxfp4}
  OUT=$OUTROOT/run-$(basename "$GG" .gguf)-$PROJ-$(date +%H%M%S); mkdir -p "$OUT"
  echo "RUN $GG (--proj $PROJ) gpu0=$NODE"
  ok=0
  for i in 0 1; do
    one gg /mnt/models "/models/${GG#/mnt/models/}" "$PROJ" $i "$OUT" && ok=$((ok+1))
    echo "  prompt $i rc=$? $(grep -h '^prompt=' "$OUT/gg-$i.log" | head -1)"
    echo "    $(text "$OUT/gg-$i.log")"
  done
  grep -hE "GGUF source|native|LOSSY|ships no MTP|tokenizer from GGUF|refus|unsupported" "$OUT/gg-0.log" | grep -v "^    native" | sort -u | cut -c1-220
  a=$(ids "$OUT/gg-0.log"); b=$(ids "$OUT/gg-1.log")
  [ -n "$a" ] && [ "$a" != "$b" ] && echo "  outputs depend on input: yes" || echo "  outputs depend on input: NO (rule 12!)"
  echo "RUN RESULT: $ok/2 ok -> $OUT" ;;
*) echo "mode must be gate or run"; exit 2 ;;
esac
