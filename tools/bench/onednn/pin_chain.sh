#!/bin/bash
# 10-27 evening: rebuild the server with the CPU pool pinning (the 15:57 build stopped before the
# server link), then the pinning A/B (fn_pin.sh).  The build is CPU-only (no GPU device).
cd /mnt/user/isos/grimoire-runs/bench-1022 || exit 2
/mnt/storage/isos/grimoire-fuse/tools/dev_build.sh server > logs/build-pin2.log 2>&1
if ! grep -q "built: bin/grimoire-server" logs/build-pin2.log; then
  echo BUILD_FAILED > logs/pin_chain.out
  tail -20 logs/build-pin2.log >> logs/pin_chain.out
  exit 1
fi
echo BUILD_OK > logs/pin_chain.out
./fn_pin.sh >> logs/pin_chain.out 2>&1
echo CHAIN_DONE >> logs/pin_chain.out
