# K2-Horizon-MoVA-36B-A4B on one B70 — status

2026-09-28. Checkpoint `/mnt/storage/Models/K2-Horizon-MoVA-36B-A4B` (BF16, 36B total,
~4B active). Pure GRIMOIRE (SYCL/L0/C++), gpu0 only, `--proj mxfp4` (fits 32 GB VRAM).

## Architecture (read from its own `modeling_k2_horizon.py`, not assumed)

- 48 layers, hidden 2560, 32 q heads / 8 kv heads, head_dim 128, full RoPE (theta 1e7,
  nested under `rope_parameters`), `query_key_norm: false` (no q/k norm weights at all).
- Grouped RMSNorm (`layernorm_num_groups` 2), weight applied directly (no `1 + w`).
- Softplus output gate (`beta = ln 2`) from a SEPARATE `self_attn.gate_proj`, on every layer.
- Layers 0-2: dense MLP (6144). Layers 3-47: MoVA attention + MoE.
  - MoVA: no v_proj; value = sum_j w_j * silu(v_experts[e_j] @ x), 64 experts, top-4,
    sigmoid scores, `v_router.bias` steers selection only, normalize, x2.5.
  - MoE: 100 experts, top-8, same sigmoid + selection-bias + normalize + x2.5 router,
    one shared expert (768) with NO gate.

## Correctness — matches the HF reference

`tools/k2_reference.py` runs the model's own code on the CPU (no devices in the container;
experts disk-offloaded — one device-map entry per expert ModuleList, or transformers'
`expand_device_map` spins for 15+ min). Same 20 ids (K2's own template, BOS 0):

| | first tokens |
|---|---|
| HF bf16 (CPU) | `User asks: "Explain in two sentences` — top-5 User 20.75, The 20.125, 1 19.0, We 18.5, Okay 14.75 |
| GRIMOIRE mxfp4 (gpu0) | `The user asks: "Explain in two` (the reference's #2, 0.6 behind) |

Bugs that made it garbage (all silent), fixed in a39d304:
1. softplus gate never applied (only Muse paths used `d.o_gate`);
2. decode MoE routed with the softmax top-k instead of `launch_router_topk_k2`;
3. `launch_rmsnorm_residual2` (fused MoE-join norm, on by default) ignored grouped norms.
Hang (0eebde1): RoPE kernels read the null q/k-norm weights.

## Speed (gpu0, MXFP4)

| | before | now |
|---|---|---|
| 5768-token prefill | 78.6 s (MoVA per-token kernels = 97.5%) | 2.32 s incl. first token (~2,490 tok/s) — grouped MoVA, 9e09003 |
| decode, 30-token prompt | 8.2 tok/s | ~57 tok/s (256 tokens 4.50 s) — sub-group MoVA kernel 9cd0bda, register top-k |

One decode token, sparse layer 3 (`GRIMOIRE_TIMELINE=1 GRIMOIRE_TIMELINE_LAYER=3`), us:
q 15.1, k+MoVA 48.6, flash 34.6 (30-token context), o_gate 16.1, o 20.3, router topk 6.5
(was 16.9), moe_gate_up 41.6, moe_down 59.2, shared 21.1, norms ~8 each; lm_head 570;
token 14.25 ms. Weight traffic alone is ~96 us/layer at 550 GB/s -> ~3x headroom.

Sweeps (256 tokens, bit-identical text): `B70_MOE_DN_SLOTS=2` 4.254 s (default 1: 4.504;
4: 4.350) — only meaningful after the moe_down launch-shape fix (R > 1 used to recompute
neighbouring rows); `B70_MOVA_ROWS` 1: 4.375 (now default), 2: 4.504, 4: 4.776;
`B70_MOE_SLOTS=8` 4.634 (default 4).

## Next

- flash_decode: key scoring was one strided FP8 load per dependent FMA (latency-bound,
  34.6 us at a 30-token context). Loads-first batching written (bit-identical by
  construction) — verify, then measure on Qwen/Muse at depth (the "decode wall").
- Decide `B70_MOE_DN_SLOTS` default after measuring Qwen/Ornith (affects every MoE model).
- moe_down 141 GB/s and moe_gate_up 400 GB/s at K2's shapes; shared expert 3 kernels.
