# Experimental TP2 verify rendezvous

`GRIMOIRE_TP_VERIFY_ONE_RENDEZVOUS=1` opts in on both ranks. Unset is the
existing per-piece protocol. This only covers the synchronous BF16 BAR
peer-partial path for single-sequence candidate verification of up to seven
rows (one anchor plus up to six drafts). It does not opt ordinary prompt
prefill, scheduler batches, asynchronous work, FP32, or the host-add path in.
Small-allreduce timing mode disables this option on both ranks. Look for
`single BF16 BAR rendezvous dispatched in verify` from **both** ranks before
counting a run as enabled. An unsupported route does not change protocols.

The producer has already completed the entire BF16 partial at this point.
Publish the full partial with one release store, advance the counter by the
original piece count, wait for the peer with one acquire, and then copy the
peer partial to BAR as before. Buffer ownership, parity, byte count,
rounding, GPU addition, BAR fence/readback and other transports are unchanged.
Both ranks agree on the option before inference.

In an instrumented earlier B60 experiment on v1.10.10, two GPUs with Swift
INT4, 8,192 exact input IDs, 128 output commits, context 8,336 and greedy MTP6
used 20 verification rounds and 2,560 reductions per rank. Peer waits went
from **10,112 to 2,560** (−74.7%); all output IDs, proposals and acceptance
counts matched. Warmed decode median went from **1,004.54 to 995.23 ms**
(−0.93%) in one ordered pair. Prefill drifted +3.51%; neither repeatable
speedup nor a benefit on narrow PCIe/OCuLink has been established. Reducing
wait operations by 75% does not reduce bytes or total latency by 75%.

This patch keeps the earlier algorithm as a small, default-off experiment.
The standalone host test covers 2,400 skewed rounds with mixed widths,
counter continuity and exact peer bytes. The older live run validates the
instrumented predecessor, **not** a live result on this exact patch. Please
record topology, require dispatch on both ranks, verify exact ordered output
IDs and report both run orders against the stock path. Keep it off by default
until there is a useful and repeatable result.

```sh
c++ -std=c++17 -O2 -pthread -Iinclude tests/test_tp_verify_pieces.cpp -o /tmp/test_tp_verify_pieces
/tmp/test_tp_verify_pieces
```
