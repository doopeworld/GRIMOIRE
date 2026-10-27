#!/bin/bash
# 10-27: after the caller-pin tests (cxs_chain.sh), one timeline run (fn_tl.sh: where a decode
# token's time goes, GRIMOIRE_TIMELINE=1 + the small all-reduce timer) with today's build.
S=/mnt/user/isos/grimoire-runs/bench-1022
cd $S || exit 2
for i in $(seq 1 360); do grep -qE "CHAIN_DONE|BUILD_FAILED|not started" logs/cxs_chain.out 2>/dev/null && break; sleep 10; done
grep -q CHAIN_DONE logs/cxs_chain.out || { echo "cxs chain did not finish -- timeline not run" > logs/tl_chain.out; exit 1; }
sleep 5
./fn_tl.sh > logs/tl_chain.out 2>&1
echo TL_DONE >> logs/tl_chain.out
