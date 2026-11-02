#!/bin/bash
# Read only: SYCL kernel names that exist in more than one production binary. Two images holding one
# kernel name leave the runtime to pick one, whatever register file or flags each was compiled with.
E=/mnt/storage/isos/grimoire-engine-1101/bin
T=$(mktemp -d)
names() { grep -a -o -E '_ZTS[A-Za-z0-9_$]*(nd_item|nd_range|item|handler)[A-Za-z0-9_$]*' "$1" | grep -E 'b70|GLOBAL__N' | sort -u; }
for b in grimoire-server grimoire libgrimoire_gemm.so libgrimoire_attn.so; do
  names "$E/$b" > "$T/$b.txt"
  echo "$b: $(wc -l < "$T/$b.txt") candidate kernel names"
done
short() { sed -E 's/^_ZTSZZN3b70//; s/^_ZTSZN3b70//' | grep -oE '^(12_GLOBAL__N_1)?[0-9]*[a-zA-Z_][A-Za-z0-9_]*(I[A-Za-z0-9_]*E)?' | cut -c1-90 | sort | uniq -c; }
echo "--- in both the server and the 256-GRF gemm library"
comm -12 "$T/grimoire-server.txt" "$T/libgrimoire_gemm.so.txt" > "$T/a.txt"; wc -l < "$T/a.txt"; short < "$T/a.txt"
echo "--- in both the server and the attention library"
comm -12 "$T/grimoire-server.txt" "$T/libgrimoire_attn.so.txt" > "$T/b.txt"; wc -l < "$T/b.txt"; short < "$T/b.txt"
echo "--- in both libraries"
comm -12 "$T/libgrimoire_gemm.so.txt" "$T/libgrimoire_attn.so.txt" > "$T/c.txt"; wc -l < "$T/c.txt"; short < "$T/c.txt"
echo "--- first full name in server+gemm"
head -2 "$T/a.txt" | cut -c1-260
rm -rf "$T"
