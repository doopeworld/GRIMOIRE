# Server base decode is 3x off roofline; the decode graph is never built

Measured 2026-09-04, Qwen3.8-27B-MXFP4-GRIMOIRE, llama-benchy pp4096/tg32:

| engine   | config              | tg32      |
|----------|---------------------|----------:|
| vLLM     | plain decode, no MTP| 27.95 t/s |
| GRIMOIRE | plain decode, no MTP| 12.20 t/s |
| GRIMOIRE | + MTP k=3           | 18.55 t/s |

vLLM's number is WITHOUT speculation. So the like-for-like comparison is
12.20 vs 27.95: GRIMOIRE's base decode forward is 2.3x slower, and MTP was
masking that rather than causing it.

Roofline: 15.55 GiB of weights at the measured 602 GB/s is ~28 ms/token,
i.e. ~36 t/s. vLLM at 27.95 is 77% of that. GRIMOIRE at 12.20 is 82 ms/token
= 34% of roofline. A 3x shortfall at 4 bits/weight is not bandwidth, it is
per-token launch and fusion overhead. 568b12c reached the same conclusion
from the other direction: "our forward floor is ~22 ms of weight reads".

Both engines are ~27B at ~4 bits/weight (MXFP4 carries an e8m0 scale per 32
values, int4-AutoRound a scale per 128: 4.25 vs 4.13 bits/weight), so decode
speed is comparable at the level that matters even though the checkpoints
and quantizations differ.

## The lever that exists and is not wired in

c957f2f added an opt-in device-resident decode graph that submits all ~800
per-token kernel launches as a single command list. The CLI uses it:

    src/grimoire.cpp:7101   if (e.graph_ok) e.step(); else e.forward(tok);

grimoire_serve_generate does not. Its greedy loop only ever calls
e.forward(tok), and there is no e.step() call anywhere in the server path.

Setting GRIMOIRE_DECODE_GRAPH=1 on the container does NOTHING, because
e.build_graph() is only called from the CLI generate path at
src/grimoire.cpp:6852. The server never builds the graph, so graph_ok stays
false and the branch above can never be taken. Grep confirms: e.step() has
exactly three call sites (3725 probe, 6759 definition, 7101 CLI).

## Next steps

- Build the decode graph once in grimoire_serve_generate after prefill and
  replay it in the greedy loop, mirroring the CLI at 7101. Guard it with
  GRIMOIRE_DECODE_GRAPH so it stays opt-in until measured.
- Re-measure plain decode (GRIMOIRE_MTP unset) against the 12.20 baseline
  before touching speculation again. Target is the ~28 ms/token roofline.
- Only then revisit MTP. Speculation multiplies whatever the base forward
  costs, so a 3x-slow forward caps every speculative number too.
- Note the comment at src/grimoire.cpp:6851 says "Speculative paths call
  forward() directly and are unaffected" -- so the graph helps plain decode
  first; making the verify forward use it is a separate, later question.
