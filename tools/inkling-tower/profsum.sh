#!/bin/bash
# profsum.sh LOG [N]: unitrace -d kernel table, sorted by device time (names may contain commas)
awk -F, '/Kernel, *Calls/{f=1;next} /=== Kernel Properties/{f=0} f && NF>=7 {n=$1; for(i=2;i<=NF-6;i++) n=n","$i; gsub(/^ +/,"",n); gsub(/"/,"",n); printf "%10.1f ms %8d  %s\n", $(NF-4)/1e6, $(NF-5), substr(n,1,90)}' $1 | sort -nr | head -${2:-50}
