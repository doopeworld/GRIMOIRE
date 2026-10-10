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

## Follow-up findings

- Interleaved output accumulators slowed both shapes (10.571/107.867 ms) and
  were reverted. Four-row flash and packed temporary keys were discarded too.
- Tried blocked BF16 weight panels without extra activation quantization. Toy
  ragged GEMM/SwiGLU/residual oracle hashes match. Real FFN 2048x34816x5120,
  GS128: flat 5.813/6.073/6.391 ms, blocked 6.020/7.237/6.590 ms. Rejected.
- Fast DPAS verify now supports the non-scheduler, one-slot generation path,
  using inclusive per-row cache lengths. Existing exact and unsupported paths
  remain available; SOLO_VERIFY_DPAS=0 restores the older batched attention.
- Found an independent old causal-count error: launch_flash_decode_batched was
  given pos rather than pos+1, and exact fallback device lengths omitted self.
  set_cursor includes self; these alternatives now agree with its convention.
- Real Qwen GPTQ/MTP K4 96-token story fixture: ordinary greedy, exact verify and
  legacy exact verify now have IDENTICAL output token ids. This is a fixture,
  not proof of greedy parity for every model/prompt. Approximate verify is not
  promised bit-identical to the plain GEMV path.
- Final clean library enables eight-row/128-key ESIMD for head_dim256 by default;
  other widths, QSA and sliding-window stay joint_matrix. Both GRF flags are
  necessary. Default library fp64 checks PASS at T1/start0, T17/start29,
  T1024/start12288 and T1024/start130048. Full output hashes match the winning
  prototype. Final kernel timings: 9.292/97.397 ms at the long shapes.
- Five-context one-slot K6/auto-budget sweep: 10/10 syntax+behavior PASS,
  all natural EOS. 4K/8K/16K/64K/128K fresh prompt tok/s:
  1975/2011/1868/1267/834; repeated decode:119.4/116.1/114.2/88.6/65.8.
  128K fresh TTFT157.405, repeat160.494; actual131252tokens, peak31.17GiB.
  This changes chunk budget/slot count from Oct9, so use fixed2048/two-slot A/B
  above to attribute the attention gain. All servers stopped immediately after tests.
- Final default K4 Qwen one-slot 4K/8K fresh+repeat PASS (101.9/99.5 decode).
  Ornith K4 final-default 8K PASS; 4K answer coherently omits empty-list handling,
  0/2 behavior PASS, its decode rates excluded. Saved-build control pending.
- Two K4 launches used a space-delimited --lengths argument by mistake; probes
  refused it, servers were stopped, no measurement retained. Correct comma list
  was rerun. cmp_g0 now propagates probe process failures after stopping server.
- Final clean CLI/server built in /mnt/storage/isos/grimoire-final-20261010
  (base8bac772); both libs are immutable there. Main bin still original at this
  checkpoint; install only after controls, then full regress_all_g0.sh.

## Next steps

1. Finish Ornith saved-build control; do not report its failing code as a speed win.
2. Install validated clean source/binaries and run tools/regress_all_g0.sh, inspect
   Sherlock reference, prose parity comparisons and every checkpoint's output.
3. Save updated context report, raw receipts, CLAUDE banner and memory; commit with
   Findings/Next steps and push longctx-attn-wip.
4. Further work: FFN remains the largest measured prefill cost. No W4A8 quality
   equivalence has been established. Intel cloud How to Connect SSH line pending.

### Ornith control and narrower default

- Saved main build with identical prompts: 4K/8K all four answers PASS.
- Candidate with SOLO_VERIFY_DPAS=0: 4K still FAIL, 8K PASS. Candidate with MTP=0:
  4K still FAIL, 8K PASS. Both produce coherent code missing the empty-list case;
  this is not skipped/repeated chunk corruption.
- Candidate with FLASH_ESIMD=0 and fast one-slot verify enabled: 4K/8K all four
  answers PASS, repeated decode157.7/147.1 tok/s. This isolates the sampled
  behavioral loss to the prompt-kernel arithmetic on Ornith, not the new verify
  routing. Numerical tolerance alone is not a broad quality-equivalence claim.
- Narrowed ESIMD auto selection to validated geometry: H24/KVH4/D256. Other
  shapes use joint_matrix unless explicitly opting into GRIMOIRE_FLASH_ESIMD=1.
  Ornith H16/KVH2/D256 retains its prior prefill path. Library rebuilt; verify
  source/binaries unchanged. Recheck default dispatch and behavior next.

- Real negative control: tools/regress_solo_verify_g0.sh on saved old main CLI
  returns1; exact verify's 96 token IDs differ from plain. Corrected candidate
  exact and legacy-exact both match. Logs bench-1010/negative-old/ and final-*.log.
  The test is a real engine fixture, not a mirrored host implementation.

### Installed checkpoint

- Scoped default library passes all numerical probes. Qwen hashes identical to
  winning kernel; H16/KVH2 default hash equals explicit FLASH_ESIMD=0.
- Scoped default served K4 Qwen and Ornith: all8/8 4K/8K fresh/repeat coding
  answers PASS. Qwen repeats101.8/99.2; Ornith157.5/146.7 tok/s.
- Installed final CLI/server/libs into main bin while no GPU holders or containers
  were active. Originals preserved bench-1010/main-before-install, hashes in
  bench-1010/installed-sha256.txt. No release image or asset changed.
- tools/regress_all_g0.sh now running FOREGROUND; output bench-1010/regress-final.out.
  Never launch a second GPU job or overwrite these binaries until it completes.
