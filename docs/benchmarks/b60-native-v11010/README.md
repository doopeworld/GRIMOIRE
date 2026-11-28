# B60 TP2 native benchmark evidence

These captures cover two historical **off → on** pairs, each with one complete excluded warmup and three measured requests per arm. They do not qualify the exact source in draft PR #5, measure a narrow link, or benchmark HTTP serving.

| Evidence | oneDNN off → on | BF16 BAR rendezvous off → on |
|---|---|---|
| Original native JSON stdout | [off](onednn-0.llama-bench.json) / [on](onednn-1.llama-bench.json) | [off](rendezvous-0.llama-bench.json) / [on](rendezvous-1.llama-bench.json) |
| Both-rank verification records, warmup and all ordered output IDs | [off](onednn-0.verified.json) / [on](onednn-1.verified.json) | [off](rendezvous-0.verified.json) / [on](rendezvous-1.verified.json) |
| Machine-readable comparison | [Combined measured rows](native-measurements.csv) | [Host spans and counters by rank, with warmup marked](rendezvous-rank-spans.csv) |

`verified.json` files are parsed rank-log records, not GPU event traces. Their `prefill_ns`, `draft` and `ids` keys are request indices, with `0` the excluded warmup. The primary JSON stdout contains only measured requests. The CSV keeps integer ns and calculates input throughput independently for each rank; use neither the sum of ranks nor first-commit elapsed to calculate prompt throughput. Decode is `127e9 / post_first_to_both_done_ns`. In the rendezvous CSV, producer completion and host publication are nested inside `small_ns`, itself part of `verify_ns`; do not add nested spans or sum ranks. Producer completion includes already queued device work.

## Reproduce the comparison

Use the [exact 8,192 prompt IDs](counting-prompt.ids) (one decimal ID per line), Swift INT4, one slot, FP8 KV, and two B60 GPUs with the native TP2 adapter launched through its generation worker. There is no HTTP client or prefix reuse. The benchmark argv used inside that worker was:

```sh
llama-bench --engine grimoire --tensor-parallel-size 2 --model /checkpoint \
  --decode-mode generation --prompt-ids /prompt.ids --mtp-depth 6 \
  --decode-tokens 128 --context-capacity 8336 --grimoire-onednn 0 \
  --repetitions 3 -o json
```

For oneDNN, change only `--grimoire-onednn 0` to `1`. For the instrumented rendezvous predecessor, hold oneDNN at `1` and append `--grimoire-verify-rendezvous 0` or `1`; this was a bench-adapter flag, not the draft PR's runtime opt-in. The generation worker used separate 300 s budgets for load, warmup, and measurement. The two pairs used different instrumented builds: compare **within** each pair. To check repeatability, run both orders on fresh loads and preserve the per-rank dispatch and ID checks. Do not use a production serving port for this benchmark.

- Grimoire base: `acc75ea3d74cbedfc3f2857a96787409be88dd0e` (v1.10.10); native source builds with bench-only hooks, not the published server executable.
- Same official userspace runtime image in both pairs: `sha256:f98202719037d89450a10d685ec985289909a3cf06abace99363baecf7b0ffb2`. Intel `icpx` 2026.2; oneDNN 3.14; recreated release-profile G31/G21 engine for oneDNN, G21 candidate build for rendezvous.
- oneDNN engine SHA-256: `75e9d65cde7a222c959d24385d761d298d9a533309e95f8fb8d8a4ddff6d3ea4`.
- Rendezvous predecessor engine SHA-256: `122c9df7ea7c41f6e16cf7dd8dd166bb8228c2742863de83d77b844b1480806f`; adapter SHA-256 `111fcf16f0f7f06864fe170754f9d72617cbabdb0ea971a49c44e5be74444849`.
- Prompt text-file SHA-256: `9b15d17968642e220b2e2bb87150e20c7d2e93e396f207cd63803b3dfcb4fa5d`. All stored ordered 128-ID lists are identical in both pairs, including warmups; SHA-256 of their compact JSON array is `c8192a2d623b644f3f1ce210b82e659a546a221700dd36a0af43e288951c8395`. Every request judged 119 draft tokens and accepted 108.

## Separate 16K packed-KV comparison

[Original rows](packed-v4-16k.native-rows.csv) from an FP8-V → INT4-V → INT4-V → FP8-V A/B. This is Grimoire `31fb1c3` on the same pinned OCI image and Swift INT4 with the 16K test-only adapter lane; both arms used INT8 K, ctx16528, MTP6, oneDNN on, one slot and the same 16,384 prompt IDs. The lane uses 1,024-row prefill waves and 32-row non-team DPAS attention slices in both arms; do not extrapolate its prefill rate from the 8K FP8-KV benchmark. `B70_NATIVE_K8_LONG=1` in both arms, `B70_NATIVE_PACKED_V4_SMOKE=0|1` for the V format. The prompt had markers near 10%, 51% and 91% of its token length. All 16 requests recovered all three in order and matched the same 128-ID sequence (SHA-256 of comma-separated decimal IDs: `555279097dc935f4588dfff1aaf2fa91c319aef8f2ea19297b4d9c66c19cb204`). The CSV publishes observations, not the evaluation prompt or output text. One tested token count and one slot are not a maximum usable capacity measurement.

The original production container was restored after the historical windows. Recovery checks are not throughput samples. File hashes for the public extracts are in [SHA256SUMS](SHA256SUMS). See the [results summary](../../b60-native-decode-screen.md) for interpretation and negative results.
