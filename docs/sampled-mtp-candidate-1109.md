# Experimental sampled MTP candidate

`sampled-tp-1109` adds temperature, top-k (1–64), top-p, and a per-request seed
for dense Qwen targets on one GPU or Megatron TP. Greedy remains the default.
The target sampler runs on GPU; TP rank zero's existing selected-token messages
keep workers in step. Each job owns its settings and each draw is keyed by
absolute output position, rather than by speculative attempts or batch order.
The first candidate retains deterministic greedy proposals and samples the
verified target rows. Synthetic tests preserve every target draw across 7,680
seeded speculative paths. GPU analytical-distribution tests passed on B70.

Real-model repeated seeds reproduce within the same serial configuration.
Changing batch width or enabling verification changes numerical execution;
observed 64-token sampled transcripts did not all match across those layouts.
The differences are preserved in the evidence, including an exact-verification
flag experiment. A full numerical attribution is not established. Do not claim
cross-layout bitwise sampled-text identity from these results.

## Acceptance optimization, not yet performance validated

`GRIMOIRE_SAMPLE_COUPLED_MTP=1` opts into sampled proposals and rejection correction.
It follows Algorithm 1 of [Leviathan et al., ICML 2023](https://arxiv.org/abs/2211.17192):
for a proposal from q, accept with min(1,p/q), otherwise sample normalized
max(p-q,0). The existing prefix comparison/commit protocol then handles the
accepted candidates and the corrected bonus. The target distribution remains
the requested temperature/top-k/top-p distribution.

Each proposal retains its sparse normalized top-k/top-p distribution on GPU.
Target verification reads it directly. A bounded 16-entry ring per physical
sequence slot is protected by request/block epochs, position and seed; a stale
record cannot turn a bonus or a new request into an acceptance check. Proposal,
acceptance and residual draws use separate random streams. Only the sampling
rank keeps these records; existing TP control messages remain unchanged.

The option is off by default. Without it, the first sampler's CDF calculation is
preserved. CPU analytical tests currently verify target marginals, expected
acceptance, disjoint support, identical distributions and invalid proposals.
GPU distribution and real-model performance validation are still pending.
This candidate has not yet beaten the supplied reference results.
