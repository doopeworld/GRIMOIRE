#!/bin/bash
# pfetch.sh URL FILE SIZE SHA256 [PARTS]
# Segmented, resumable HTTP range download.  Hugging Face throttled a single connection to
# ~3 MB/s for the 71 GB BF16 GGUF on 2026-10-15, while 8 parallel ranges reached ~200 MB/s.
# Parts live in FILE.parts/NNN (an existing partial FILE can be moved to FILE.parts/000 first);
# each part resumes from its current length.  The result is concatenated and accepted only if it
# matches the pinned sha256.
set -u
U=${1:?url} F=${2:?file} SIZE=${3:?size} WANT=${4:?sha256} N=${5:-16}
mkdir -p "$F.parts"
CH=$(( (SIZE + N - 1) / N ))
echo "$(date -Is) START $F size=$SIZE parts=$N"
for i in $(seq 0 $((N - 1))); do
  lo=$((i * CH)); hi=$((lo + CH - 1)); [ $hi -ge $SIZE ] && hi=$((SIZE - 1))
  want=$((hi - lo + 1)); p=$F.parts/$(printf %03d $i)
  (
    for try in $(seq 1 40); do
      have=$(stat -c %s "$p" 2>/dev/null || echo 0)
      [ "$have" -ge "$want" ] && break
      curl -sL --fail --connect-timeout 20 --speed-limit 100000 --speed-time 60 \
           -r $((lo + have))-$hi "$U" >> "$p" || sleep 5
    done
    have=$(stat -c %s "$p" 2>/dev/null || echo 0)
    if [ "$have" -eq "$want" ]; then echo "$(date -Is) part $i ok"; else echo "$(date -Is) part $i SIZE_BAD have=$have want=$want"; fi
  ) &
done
wait
tot=$(du -cb "$F".parts/* | tail -1 | cut -f1)
[ "$tot" -eq "$SIZE" ] || { echo "$(date -Is) TOTAL_BAD $tot != $SIZE"; exit 1; }
cat "$F".parts/* > "$F.tmp" && rm -rf "$F.parts"
got=$(sha256sum "$F.tmp" | cut -d' ' -f1)
if [ "$got" = "$WANT" ]; then mv "$F.tmp" "$F"; echo "$(date -Is) SHA_OK $F $got"
else echo "$(date -Is) SHA_MISMATCH $F got=$got want=$WANT (kept as $F.tmp)"; exit 1; fi
