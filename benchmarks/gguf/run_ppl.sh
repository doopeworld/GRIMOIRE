#!/bin/bash
# run_ppl.sh -- accuracy of each way to run Ornith-1.5 on gpu0, WikiText-2 test, llama-perplexity
# protocol (ctx 512, first 64 chunks, second half of each chunk scored, no BOS).
#   1 grimoire  BF16 GGUF  -> --proj mxfp4  (GRIMOIRE's existing path; Gate A: == safetensors)
#   2 grimoire  Q4_K_M GGUF -> --proj mxfp4 (decode-at-load requant, the thing being judged)
#   3 llama.cpp Q4_K_M native Q4_K          (the file's own accuracy)
#   4 llama.cpp BF16 (experts on CPU)        (full-precision reference)
# llama.cpp is a reference here only; nothing of it runs inside GRIMOIRE.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=/mnt/user/isos/grimoire-runs/gguf-ppl/$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
NODE=$(bash "$REPO/tools/gpunode.sh" gpu0) || exit 2
TXT=/models/eval/wiki.test.raw; CTX=512; CH=${CHUNKS:-64}
LCPP=ghcr.io/ggml-org/llama.cpp:full-intel@sha256:c68de96b6d45634398b648b1ad9efb2f5cb0d1d7e95fa45e9657cd78610d427f
echo "gpu0=$NODE commit=$(git -C "$REPO" rev-parse --short HEAD) ctx=$CTX chunks=$CH out=$OUT"

grim() {  # grim TAG MODELDIR [extra flags]
  local tag=$1 dir=$2; shift 2
  GRIM_ENV="ZE_AFFINITY_MASK=0" GRIM_TREE="$REPO" GRIM_MODELS=/mnt/models \
    bash "$REPO/tools/b70run.sh" "$NODE" 7200 "ppl-$tag" /grimoire/bin/grimoire -m "/models/$dir" --proj mxfp4 \
    --ppl $TXT --ppl-ctx $CTX --ppl-chunks $CH --ppl-dump "/models/eval/dump-$tag.tsv" "$@" > "$OUT/$tag.launcher" 2>&1
  cp "/tmp/grim-ppl-$tag.log" "$OUT/$tag.log"; docker rm "grim-ppl-$tag" >/dev/null 2>&1
  echo "$tag: $(grep -h '^PPL:' "$OUT/$tag.log")  $(grep -h 'selfcheck' "$OUT/$tag.log")"
}
lcpp() {  # lcpp TAG GGUF [extra flags]  -- llama-perplexity in its own container, b70run-style lifecycle
  local tag=$1 f=$2; shift 2; local name=gguf-ppl-$tag
  docker rm -f $name >/dev/null 2>&1
  docker run -d --name $name --init --stop-timeout 300 --device /dev/dri/$NODE --group-add "$(stat -c %g /dev/dri/$NODE)" \
    -v /mnt/models:/models:ro -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
    --entrypoint /usr/bin/timeout $LCPP --signal=TERM --kill-after=60 7200 \
    /app/llama-perplexity -m "$f" -f $TXT -c $CTX --chunks $CH -b $CTX -fa on "$@" >/dev/null
  docker wait $name >/dev/null; docker logs $name > "$OUT/$tag.log" 2>&1; docker rm $name >/dev/null
  echo "$tag: $(grep -hE 'Final estimate' "$OUT/$tag.log")  $(grep -hoE 'tokenizing the input .*|[0-9]+ tokens' "$OUT/$tag.log" | head -1)"
}
grim  grim-bf16-mxfp4  ornith-gguf-bf16 --ppl-selfcheck
grim  grim-q4km-mxfp4  ornith-gguf-q4-k-m
lcpp  lcpp-q4km        /models/ornith-gguf-q4-k-m/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99
lcpp  lcpp-bf16        /models/ornith-gguf-bf16/Ornith-1.5-35B-BF16.gguf -ngl 99 --cpu-moe
echo "--- GRIMOIRE Q4_K_M->mxfp4 vs GRIMOIRE BF16->mxfp4, token level:"
python3 "$REPO/tools/ppl_compare.py" /mnt/models/eval/dump-grim-bf16-mxfp4.tsv /mnt/models/eval/dump-grim-q4km-mxfp4.tsv
echo "PPL RUN DONE -> $OUT"
