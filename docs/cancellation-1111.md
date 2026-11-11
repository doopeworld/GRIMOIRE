# Ranked cancellation, 2026-11-11

Resumed after discovering newer November10 work. Coupled sampling was GPU validated but slower (107.28 vs118.31 in five-pass smoke); retain OFF and K6. Matched c8 aggregate139.3tok/s,48/48, below reference365.5. Another GPU1 drop occurred in sustained16K prefill; cause remains unproven. Do not repeat sustained GPU1 prefill while owner cannot reboot.

Existing a493563 shutdown fix passed GPU0-only idle and mid-generation tests, but a cancelled TP request still ran its entire budget (186seconds for3000tokens). This can exceed Docker's stop timer for larger requests.

New generation_control.hpp broadcasts rank0 callback continuation at each emitted token. TP uses root-to-peer control sockets; PP forwards root-to-last, after pending hidden-state sends drain. The word has a phase tag and validated boolean. Workers cannot independently terminate a request. Scheduler no longer overrides cancellation to true for ranked generation. EOS/length remain identical on all ranks; no fabricated EOS is exposed to clients. Single-GPU decoding has no added control traffic.

CPU production-generation tests use real Unix sockets with three ranks in TP-star and PP-chain layouts. Depth0/1/6, cancellation at token1/3/10, cancellation inside accepted blocks, and a next-request header/generation pass. Invalid control words fail explicitly. Existing generation/sampling suites pass. This tests control flow, not GPU numerical equivalence.

Plan: isolated CPU-only G31 AOT build, then short GPU0-only T0 check. GPU1 remains unmapped. No resets/retraining/reboots, no forced stop timers, no sustained GPU1 tests. Do not deploy this into user services until runtime validation passes.
