#!/bin/bash
# gate_bf16.sh -- Phase 2 Gate A: GRIMOIRE reading the BF16 GGUF natively must generate exactly the
# same greedy token ids as GRIMOIRE reading the same model's BF16 safetensors, both with --proj PROJ.
#
# Why equality is the right bar: docs/gguf/FORMAT_SUPPORT.md section 8 proved every GGUF tensor
# reconstructs the safetensors bf16 values bit-exactly after the inverse transforms, so both sources
# must produce identical device weights and therefore identical tokens.  Anything less is a bug.
#
#   gate_bf16.sh [PROJ=mxfp4] [N=64]
# gpu0 is resolved by PCI id at run time (render node numbers change across boots).
set -u
PROJ=${1:-mxfp4}; NTOK=${2:-64}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=/mnt/user/isos/grimoire-runs/gguf-gate/bf16-$PROJ-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
NODE=$(bash "$REPO/tools/gpunode.sh" gpu0) || exit 2
PROMPTS=(
  "Explain in three sentences why the sky is blue."
  "Write a Python function that returns the n-th Fibonacci number iteratively."
  "Translate to French: The weather is lovely today and we are going to the beach."
  "List the planets of the solar system in order from the sun."
)
ENVS="GRIMOIRE_PRINT_IDS=1
ZE_AFFINITY_MASK=0
SYCL_CACHE_PERSISTENT=1"
run() {  # run TAG MODELS_ROOT MODEL_DIR PROMPT_INDEX
  local tag=$1 root=$2 dir=$3 i=$4
  GRIM_ENV="$ENVS" GRIM_TREE="$REPO" GRIM_MODELS="$root" \
    bash "$REPO/tools/b70run.sh" "$NODE" 3600 "gate-$tag-$i" \
    /grimoire/bin/grimoire -m "/models/$dir" --proj "$PROJ" --ctx 4096 -p "${PROMPTS[$i]}" -n "$NTOK" \
    > "$OUT/$tag-$i.launcher" 2>&1
  local rc=$?
  cp "/tmp/grim-gate-$tag-$i.log" "$OUT/$tag-$i.log" 2>/dev/null
  docker rm "grim-gate-$tag-$i" >/dev/null 2>&1
  return $rc
}
echo "gpu0=$NODE proj=$PROJ n=$NTOK out=$OUT  commit=$(git -C "$REPO" rev-parse --short HEAD)"
pass=0; fail=0
for i in "${!PROMPTS[@]}"; do
  run st /mnt/storage/Models Ornith-1.5-35B-A3B "$i"; rs=$?
  run gguf /mnt/models ornith-gguf-bf16 "$i"; rg=$?
  a=$(grep -h "^greedy ids" "$OUT/st-$i.log" 2>/dev/null | cut -d: -f2)
  b=$(grep -h "^greedy ids" "$OUT/gguf-$i.log" 2>/dev/null | cut -d: -f2)
  pa=$(grep -h "^prompt ids" "$OUT/st-$i.log" 2>/dev/null | cut -d: -f2)
  pb=$(grep -h "^prompt ids" "$OUT/gguf-$i.log" 2>/dev/null | cut -d: -f2)
  if [ $rs -eq 0 ] && [ $rg -eq 0 ] && [ -n "$a" ] && [ "$a" = "$b" ] && [ "$pa" = "$pb" ]; then
    echo "prompt $i: IDENTICAL ($(echo $a | wc -w) tokens)"; pass=$((pass+1))
  else
    echo "prompt $i: DIFFERENT  rc st=$rs gguf=$rg  prompt-ids-equal=$([ "$pa" = "$pb" ] && echo yes || echo no)"
    python3 - "$a" "$b" <<'EOF'
import sys
a, b = sys.argv[1].split(), sys.argv[2].split()
d = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
print(f"   first divergence at token {d} of {len(a)}/{len(b)}: st={a[d:d+5]} gguf={b[d:d+5]}")
EOF
    fail=$((fail+1))
  fi
done
echo "--- GGUF loader report (last gguf run):"
grep -hE "GGUF (source|tokens|tokenizer|eos|round-trip|bf16)|LOSSY|WARNING" "$OUT/gguf-$((${#PROMPTS[@]}-1)).log" | sort -u
echo "--- generated text, gguf, prompt 0 (READ IT):"
sed -n '/^greedy ids/q;p' "$OUT/gguf-0.log" | grep -vE "^\s|^prompt ids|unavailable|^  " | tail -8
echo "GATE A ($PROJ): $pass identical, $fail different -> $OUT"
[ $fail -eq 0 ]
