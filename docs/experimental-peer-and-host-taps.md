# Experimental P2P transport and DFlash host history

These options are under validation in the experimental branch. They are off by
default and do not change the saved user container. This document describes the
implementation, not a claim that physical P2P bandwidth has been measured.

## GPU peer transport

`GRIMOIRE_P2P=auto` negotiates Level Zero IPC on each applicable Unix socket.
If either endpoint cannot export/import the device allocation, both use the
existing host transport. `GRIMOIRE_P2P=1` (or `required`) instead refuses startup
when direct IPC is unavailable. Unset or `0` keeps the existing transport.
`GRIMOIRE_P2P_CHUNK_MB` sets the bounded allocation size, 1–64 MiB, default 4.

The implementation handles two-rank Megatron tensor parallel and adjacent
pipeline stages. Legacy TP and TP with more than two ranks retain host
transport in auto mode; required mode rejects those configurations.

Each worker retains its own GPU mask. Unix sockets transfer dma-buf descriptors
using SCM_RIGHTS and carry readiness/consumption messages. Native Level Zero
copy commands transfer GPU payloads. A local snapshot and receive buffer keep
kernels away from imported memory. There are no cross-device atomics or
GPU polling loops. Tensor reductions use FP32 payloads and rank-ordered sums;
the existing host BF16/INT8 communication settings do not quantize this path.
CPU expert partials are added before the FP32 snapshot.

After a transfer starts, an error terminates that socket. It never silently
switches an in-flight collective to a different protocol. Control waits and
native copy synchronization have 120-second limits.

`tools/test_peer_copy.cpp` re-execs two workers on one visible GPU and explicitly
allows same-device IPC for the test. It verifies multi-chunk FP32/BF16 sums,
CPU partial additions and pipeline round trips. Production setup rejects two
ranks selecting the same GPU. This test proves protocol correctness only.
`tools/test_ipc_socket.cpp` checks usable/CLOEXEC descriptors, malformed
multi-descriptor rejection, and short I/O on a large socket payload.

The Tower's root-port topology has previously refused PCIe P2P. Successful
fallback on that machine cannot establish performance on a P2P-capable tower.

## DFlash target-feature history in host RAM

`GRIMOIRE_DFLASH_TAPS_HOST=1` keeps the full target-feature history as lossless
FP32 in pinned system RAM. Target kernels capture into that history. Before
DFlash context projection, the engine copies one context chunk to a bounded
device staging buffer. Chunk size remains `GRIMOIRE_DFLASH_CTX_CHUNK`, default 512
for Qwen, clamped to 16–4096 rows. Device staging addresses remain stable.

For Qwen 27B's five taps at hidden5120, the full buffer is 12.5 GiB per rank at
131072 tokens and 25 GiB per rank at 262144. Host staging removes that full-history
VRAM allocation; the bounded staging buffer at 512 rows occupies 50 MiB per rank.
The FP8 target KV cache and DFlash's own caches are separate allocations.

Host allocation checks available system RAM, keeps an 8 GiB reserve, and first
touches the buffer while holding a local-worker allocation lock. Additional
sequence slots get independent histories. Host bytes are excluded from GPU
memory accounting. Automatic multi-slot host-budget fitting is still under
review; initial long-context validation uses one slot.

This retains a full history in system RAM. It does not implement windowed
history eviction. Long-context cache allocation alone is not proof that a
full-length request has been processed. No 256K user-container change is part
of this work; the saved user context remains 131072.
