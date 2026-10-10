# 2026-10-10 prefill work — checkpoint

## Findings

- Ian resumed. Tower rebooted; 03:00.0 [8086:e223] remains renderD128. gpu0 only,
  gpu1 untouched. Builds use grimoire-dev:latest without GPU devices. GPU tests
  run foreground through g0run/srv2 and containers stop at completion.
- Confirmed production ESIMD prefill spill: 4544 bytes with cl-256-GRF. doubleGRF
  removes it but alone drops normal joint_matrix kernels to 128 registers.
  Combined flags retain their 256-register allocation and eliminate flash spill.
- Production library benchmark now checks sampled fp64 causal attention, ragged
  tails, resumed and long contexts, all-output finiteness and full output hashes.
  Initial kernel 16K 12.603 -> 9.322 ms; 128K 134.883 -> 110.528 ms.
- Matched served prompts, ctx 135168, slots 2, MTP6/vocab65536, fixed chunk2048:
  baseline 4K/16K TTFT 2.53/10.41 s; correct-GRF ESIMD 2.48/9.82 s. All answers
  PASS and identical. Throughput gains are smaller than isolated kernel gains.
- Eight query rows and 128-key blocks outperform 32/64 blocks. Four-row variants
  are substantially slower and discarded. Ragged/resumed/long oracle checks pass.
- Served eight-row/128-key trial: 4K/16K/64K/128K TTFT 2.47/9.80/56.08/163.82 s,
  all PASS. Matched baseline long 64K/128K 65.39/199.01 s, all PASS. Thus long
  prompt speed improves ~16.6%/21.5% with fixed cache/chunk/prompt. Decode step
  comparisons must account for changed answers/acceptance (128K baseline/trial
  63.2/63.0 tok/s, equal 5.23 tokens/update).
- Blocked temporary packed keys give only ~3% in the isolated 128K test, with
  bit-identical output hashes. This does not establish translation as the primary
  limiter. Testing interleaved independent value-product accumulators next.
- g0run.sh swallowed tune.sh failures; a blocked setup exposed this when a
  missing executable was reported as successful. Fixed return-status propagation;
  negative control returns 1, no GPU work occurred, stopped containers removed.
  Only rows with actual oracle PASS/behavior PASS are measurements.
- Main inference library remains unchanged. Candidates are in the detached
  /mnt/storage/isos/grimoire-prefill-20261010 worktree and ignored bin/prefill1010-*.
  Logs/scripts in /mnt/storage/isos/grimoire-runs/bench-1010.

## Next steps

1. Finish interleaved-accumulator comparison; retain only verified wins.
2. Evaluate the fast verify kernel on the repaired single-slot generation path,
   which currently uses the older attention loop and cannot exploit freed KV memory.
   Candidate is opt-in GRIMOIRE_SOLO_VERIFY_DPAS=1; server is building, unverified.
3. Revalidate actual coded answers at every context before changing defaults.
4. Inspect feed-forward GEMM costs; the 16K diagnostic put them at 46.4%. Do not
   adopt extra activation quantization without numerical and behavioral validation.
