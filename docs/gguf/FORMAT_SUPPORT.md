# GGUF format support — Ornith-1.5-35B-A3B (`qwen35moe`)

Status: **GGUF adapter and native Q4_K/Q6_K path implemented; B70 execution pending.** Transforms were verified against real tensor values (2026-10-15, §7). Everything marked *(header)* was read from
the real GGUF headers (first 16 MiB of each file, fetched by HTTP range request on
2026-10-15). Everything marked *(converter)* was read from llama.cpp's converter source at
`ggml-org/llama.cpp@711780ffc75b` (`conversion/qwen.py`). Items marked **UNVERIFIED** must be
checked against tensor VALUES on the Tower before any code relies on them (CLAUDE.md rule 21:
a transform verified only against our own reading of it is not verified).

## 1. Target files

| | |
|---|---|
| Repository | `ornith-ai/Ornith-1.5-35B-A3B-GGUF` |
| Revision | `12393612fd4f730ff5aadc23e9b8f9648aa49ceb` (2026-08-24) |
| `Ornith-1.5-35B-Q4_K_M.gguf` | 21,713,463,040 B, sha256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f` |
| `Ornith-1.5-35B-BF16.gguf` | 71,066,994,400 B, sha256 `044d7f8b55580720dae0d0d8b58f824024901ed5da62f93c011f6bd8dc45a42b` |
| Q8_0 / Q6_K / Q5_K_M | also published (37.8 / 29.2 / 25.3 GB); not downloaded |

There is **no Unsloth-published Ornith-1.5 GGUF** (Unsloth published Ornith-1.0 only). The
official file embeds a chat template ending in `{#- Unsloth fixes - developer role, tool
calling #}`, i.e. Unsloth's template, with a standard llama.cpp Q4_K_M recipe
(`general.file_type = 15`, imatrix-calibrated, 3,636 chunks).

## 2. Tensor-type inventory *(header)*

GGUF v3, 753 tensors, alignment 32, `general.architecture = qwen35moe`, 41 blocks
(40 transformer layers + 1 MTP/nextn layer at `blk.40`).

| File | Type | Tensors | GiB |
|---|---|---:|---:|
| Q4_K_M | Q4_K | 379 | 15.23 |
| Q4_K_M | Q6_K | 64 | 4.90 |
| Q4_K_M | F32 | 310 | 0.08 |
| BF16 | BF16 | 443 | 66.09 |
| BF16 | F32 | 310 | 0.08 |

Per-role types in Q4_K_M (`blk.N.` prefix stripped; shapes are GGUF `ne`, innermost first):

| Role | Q4_K_M types | ne |
|---|---|---|
| `ffn_gate_exps`, `ffn_up_exps` | Q4_K ×41 | [2048, 512, 256] |
| `ffn_down_exps` | **Q6_K ×21, Q4_K ×20** | [512, 2048, 256] |
| `ffn_gate_shexp`, `ffn_up_shexp` | Q4_K ×41 | [2048, 512] |
| `ffn_down_shexp` | Q6_K ×21, Q4_K ×20 | [512, 2048] |
| `ffn_gate_inp` (router) | F32 | [2048, 256] |
| `ffn_gate_inp_shexp` | F32 | **[2048] in Q4_K_M, [2048, 1] in BF16** |
| `attn_qkv` (DeltaNet) | Q6_K ×14, Q4_K ×16 | [2048, 8192] |
| `attn_gate` (DeltaNet z) | Q4_K ×30 | [2048, 4096] |
| `ssm_alpha`, `ssm_beta` | Q4_K ×30 | [2048, 32] |
| `ssm_out` | Q4_K ×30 | [4096, 2048] |
| `ssm_a`, `ssm_dt.bias` | F32 | [32] |
| `ssm_conv1d` | F32 | [4, 8192] |
| `ssm_norm` | F32 | [128] |
| `attn_q` (full attn, q+gate) | Q4_K ×11 | [2048, 8192] |
| `attn_k` | Q4_K ×11 | [2048, 512] |
| `attn_v` | Q6_K ×7, Q4_K ×4 | [2048, 512] |
| `attn_output` | Q4_K ×11 | [4096, 2048] |
| `attn_q_norm`, `attn_k_norm` | F32 | [256] |
| `attn_norm`, `post_attention_norm` | F32 | [2048] |
| `token_embd` | Q4_K | [2048, 248320] |
| `output` | Q6_K | [2048, 248320] |
| `output_norm` | F32 | [2048] |
| `nextn.eh_proj` | Q4_K | [4096, 2048] |
| `nextn.{enorm,hnorm,shared_head_norm}` | F32 | [2048] |

**Implication:** a native Q4_K_M path needs **Q4_K and Q6_K** kernels for both dense and
routed-expert GEMM; ~24% of the quantized bytes are Q6_K, including half of all expert-down
weights. Every K dimension (2048, 4096, 512) is a multiple of the 256-element super-block.

## 3. Metadata → `Qwen35Config` *(header + converter)*

| GGUF key | Value | GRIMOIRE / HF field |
|---|---|---|
| `qwen35moe.block_count` | 41 | `num_hidden_layers` = 41 − `nextn_predict_layers` (1) = **40** |
| `qwen35moe.embedding_length` | 2048 | `hidden_size` |
| `qwen35moe.attention.head_count` / `head_count_kv` | 16 / 2 | `num_attention_heads` / `num_key_value_heads` |
| `qwen35moe.attention.key_length` / `value_length` | 256 / 256 | `head_dim` |
| `qwen35moe.rope.freq_base` | 1e7 | `rope_theta` |
| `qwen35moe.rope.dimension_count` | 64 | `head_dim × partial_rotary_factor` → factor 0.25 |
| `qwen35moe.rope.dimension_sections` | [11,11,10,0] | MRoPE sections (text-only: positions identical across sections) |
| `qwen35moe.attention.layer_norm_rms_epsilon` | 1e-6 | `rms_norm_eps` |
| `qwen35moe.expert_count` / `expert_used_count` | 256 / 8 | `num_experts` / `num_experts_per_tok` |
| `qwen35moe.expert_feed_forward_length` | 512 | `moe_intermediate_size` |
| `qwen35moe.expert_shared_feed_forward_length` | 512 | `shared_expert_intermediate_size` |
| `qwen35moe.ssm.group_count` | 16 | `linear_num_key_heads` |
| `qwen35moe.ssm.time_step_rank` | 32 | `linear_num_value_heads` |
| `qwen35moe.ssm.state_size` | 128 | `linear_key_head_dim` |
| `qwen35moe.ssm.inner_size` | 4096 | `linear_value_head_dim × linear_num_value_heads` → 128 |
| `qwen35moe.ssm.conv_kernel` | 4 | `linear_conv_kernel_dim` |
| `qwen35moe.full_attention_interval` | 4 | `layer_types`: layer *i* is full attention iff `(i+1) % 4 == 0` → 10 full, 30 linear; **matches HF config.json** (full at 3,7,…,39) |
| `qwen35moe.nextn_predict_layers` | 1 | MTP head present at `blk.40` |
| `tokenizer.ggml.*` | gpt2 BPE, pre `qwen35`, 248,320 tokens, 247,587 merges, eos 248046, bos/pad 248044, add_bos false | see §6 |

Not in GGUF and needed by GRIMOIRE: `attn_output_gate` (true for this family: `attn_q` is
8192 = 16 heads × 256 × 2), `tie_word_embeddings` (false: `output.weight` is present).
`recurrent_layers` metadata is written by newer converters but is **absent** in this file;
derive layer types from `full_attention_interval`.

## 4. Tensor name mapping and converter transforms

HF names are GRIMOIRE's (`docs/gguf/ARCHITECTURE.md` §4). `L` = layer, `P` =
`model.language_model.` or `model.`. Converter behaviour is from
`Qwen3_5MoeTextModel = _Qwen35MRopeMixin + _LinearAttentionVReorderBase(Qwen3NextModel)`
and `Qwen2MoeModel.modify_tensors`.

### 4.1 Value transforms applied by the converter (must be inverted or absorbed)

| # | Transform *(converter, `conversion/qwen.py`)* | Applies to | Inverse / handling |
|---|---|---|---|
| T1 | `A = -exp(A_log)` (l.396-397), stored F32; **verified exact** vs f32(-exp(bf16 A_log)) | `ssm_a` | Feed `A` directly where the engine computes `-exp(A_log)`; `A_log = log(-A)` is not bit-exact. Which form GRIMOIRE's DeltaNet consumes is checked in the adapter task. |
| T2 | `w + 1` for every `*norm.weight` **except** `linear_attn.norm.weight` (l.402-403). **Verified: stored as exact f32(1 + w)** — NOT bf16-rounded (that hypothesis differs by up to 7.8e-3 and is the control) | `attn_norm`, `post_attention_norm`, `output_norm`, `attn_q_norm`, `attn_k_norm`, `nextn.{enorm,hnorm,shared_head_norm}` | GRIMOIRE applies `(1 + w)` in f32 (CLAUDE.md rule 11): use the GGUF value under the plain-`w` convention for these norms. Since f32(1+w) is exactly what the engine forms from the bf16 `w`, **exact token equality with the safetensors path is possible.** `ssm_norm` is NOT shifted (verified). |
| T3 | `conv1d` squeezed `[C,1,4] → [C,4]` (l.400-401) | `ssm_conv1d` | ne = [4, 8192]: row per channel, 4 taps contiguous. |
| T4 | **V-head reorder grouped → tiled** (`_reorder_v_heads`, l.454-646); applies because `num_k_heads (16) ≠ num_v_heads (32)`, `num_v_per_k = 2` | V rows of `attn_qkv` (rows 4096..8191); rows of `attn_gate` (z); rows of `ssm_alpha`, `ssm_beta`; `ssm_a`; `ssm_dt.bias`; V channels of `ssm_conv1d` (4096..8191); **columns** of `ssm_out` | Row-side (all but `ssm_out`): exact inverse = row permutation; for quantized tensors whole rows (blocks along K) move, lossless. **Column-side (`ssm_out`) cannot be undone on quantized blocks** (a 256-element block spans two 128-wide V heads). Strategy A decodes and permutes columns at load. Native Q4_K/Q6_K keeps the blocks intact and gathers the grouped-order activation into tiled V-head order immediately before the `ssm_out` projection. |
| T5 | `dt_bias` renamed to `dt_proj.bias` → GGUF `ssm_dt.bias` (l.398-399) | | name only (plus T4 permutation) |
| T6 | Fused experts `gate_up_proj [E, 2I, H]` split: gate = rows `[0, I)`, up = rows `[I, 2I)` (`Qwen2MoeModel`, l.105-118) | `ffn_gate_exps`, `ffn_up_exps` | Concatenate gate then up per expert = GRIMOIRE's existing `[E*2I, H]` gate-then-up layout. `down_proj [E, H, I]` unchanged → `ffn_down_exps` ne [512, 2048, 256]. |

Grouped vs tiled order, with `r = num_v_per_k = 2`, `G = num_k_heads = 16`:
grouped (HF) index of V head = `g * r + j`; tiled (GGUF) index = `j * G + g`
(`g ∈ [0,G)`, `j ∈ [0,r)`), each head `head_dim` (128) elements wide; for `ssm_alpha/beta`,
`ssm_a`, `ssm_dt.bias` the head width is 1.

**Version risk:** this file was converted on/before 2026-08-24 with an unknown llama.cpp
commit. The transforms above are from today's master. Each one is a claim about the FILE
and must be checked against values (§7), not assumed.

### 4.2 Name table

| GGUF | HF / GRIMOIRE | Notes |
|---|---|---|
| `token_embd.weight` | `P embed_tokens.weight` | |
| `output.weight` | `lm_head.weight` | not tied |
| `output_norm.weight` | `P norm.weight` | T2 |
| `blk.L.attn_norm.weight` | `P layers.L.input_layernorm.weight` | T2 |
| `blk.L.post_attention_norm.weight` | `P layers.L.post_attention_layernorm.weight` | T2 |
| `blk.L.attn_qkv.weight` | `… linear_attn.in_proj_qkv.weight` | T4 on V rows |
| `blk.L.attn_gate.weight` | `… linear_attn.in_proj_z.weight` | T4 |
| `blk.L.ssm_alpha.weight` | `… linear_attn.in_proj_a.weight` | T4 (width 1); alpha=a / beta=b verified (swap control fails) |
| `blk.L.ssm_beta.weight` | `… linear_attn.in_proj_b.weight` | T4 (width 1) |
| `blk.L.ssm_conv1d.weight` | `… linear_attn.conv1d.weight` | T3, T4 on V channels |
| `blk.L.ssm_a` | `… linear_attn.A_log` | T1, T4 |
| `blk.L.ssm_dt.bias` | `… linear_attn.dt_bias` | T5, T4 |
| `blk.L.ssm_norm.weight` | `… linear_attn.norm.weight` | no T2 |
| `blk.L.ssm_out.weight` | `… linear_attn.out_proj.weight` | T4 on columns |
| `blk.L.attn_q.weight` | `… self_attn.q_proj.weight` | q+gate, 8192 rows, verified bit-identical (no reorder) |
| `blk.L.attn_k.weight` / `attn_v.weight` | `… self_attn.k_proj` / `v_proj` | |
| `blk.L.attn_output.weight` | `… self_attn.o_proj.weight` | |
| `blk.L.attn_q_norm` / `attn_k_norm` | `… self_attn.q_norm` / `k_norm` | T2 |
| `blk.L.ffn_gate_inp.weight` | `… mlp.gate.weight` | F32 router |
| `blk.L.ffn_gate_inp_shexp.weight` | `… mlp.shared_expert_gate.weight` | shape [2048] or [2048,1] |
| `blk.L.ffn_{gate,up,down}_shexp.weight` | `… mlp.shared_expert.{gate,up,down}_proj.weight` | |
| `blk.L.ffn_{gate,up}_exps.weight` | `… mlp.experts.gate_up_proj` (split) | T6 |
| `blk.L.ffn_down_exps.weight` | `… mlp.experts.down_proj` | |
| `blk.40.*` (+ `nextn.*`) | `mtp.*` (one-layer MTP head) | MTP is optional (`GRIMOIRE_MTP`); map after the main path works |

## 5. Quantization strategies

**Phase 3 step 1 implementation:** `--proj native` preserves Q4_K/Q6_K
blocks in device memory and decodes them in projection kernels. See
[NATIVE_QK.md](NATIVE_QK.md) for storage, T4 handling and validation.

- **Strategy A (load-time decode → `--proj`)**: reuses every existing kernel; the path to the
  correctness milestone. Requantizing Q4_K/Q6_K → MXFP4/INT4 is **lossy** and must pass an
  accuracy gate against the GGUF's own dequantized values, not just "generates text".
- **Strategy B (native Q4_K/Q6_K kernels)**: implemented as an opt-in path
  with raw super-blocks, decode GEMV, prefill GEMM, MoE dispatch and an
  `ssm_out` activation gather for T4(b). B70 compilation, generation, KLD
  and throughput remain to be checked.

## 6. Tokenizer

The GGUF embeds a gpt2-style BPE (`tokenizer.ggml.model = gpt2`, `pre = qwen35`).
GRIMOIRE now builds its tokenizer from GGUF metadata when `tokenizer.json` is absent.
The supported GGUF pre-tokenizer is `qwen35`; unknown pre names are refused by name.
`llama` SentencePiece BPE uses scores, the `▁` marker and byte fallback; its
synthetic host fixture has hand-checked expectations. If `tokenizer.json` is
present it remains the source, with GGUF vocabulary and ordered merges checked.
The Qwen3.8 header's first 248,070 IDs and all 247,587 merge ranks match the
Ornith HF tokenizer; its 250 additional IDs are CONTROL/UNUSED. The GGUF-built
tokenizer matches all 297,193 HF IDs for `wiki.test.raw` and the golden cases.
B70 compilation and real GGUF generation remain to be checked on the card.

## 7. Verification plan (Tower, before any adapter code is trusted)

Using the BF16 GGUF and `/mnt/storage/Models/Ornith-1.5-35B-A3B` (BF16 safetensors):

1. For one linear layer and one full layer: compare every GGUF tensor against its HF source
   after applying the forward transforms T1–T6 (expect bit-equality for BF16 weights; ULP-level
   for T1/T2 in f32), and report max-abs diff per tensor. This checks the transforms against
   the FILE, independently of our adapter.
2. Confirm `layer_types` from `config.json` equals the `full_attention_interval` rule.
3. Confirm the `ssm_alpha ↔ in_proj_a` assignment and the `attn_q` q/gate interleave.
4. Confirm the sidecar tokenizer equals the GGUF tokenizer arrays.

## 8. Verification results (2026-10-15)

`tests/gguf/verify_transforms.py` fetched byte ranges of the published BF16 GGUF
(`@12393612`) and of the original HF checkpoint `ornith-ai/Ornith-1.5-35B-A3B@10fbf86f`
(BF16 safetensors) and compared values after applying llama.cpp's forward transforms to the HF
side. **44 comparisons, 0 unexpected** — every transform row is bit-exact, every control
(no reorder; alpha/beta swapped; bf16-rounded norm) differs:

- T1+T4 `ssm_a`, T5+T4 `ssm_dt.bias`, T3+T4 `ssm_conv1d`, T4 `ssm_alpha`/`ssm_beta`/
  `attn_gate`/`attn_qkv` (V rows) and `ssm_out` (columns): exact.
- T2 norms (`attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm`, `output_norm`):
  exact as f32(1+w); `ssm_norm` exact with no shift.
- Full attention q/k/v/o, router, shared expert gate/gate/down, routed experts 0/137/255
  (gate, up, down), embedding and head rows 0/151643/248319: bit-identical.

Not covered by this check: MTP (`blk.40.nextn.*`) names/transforms, and Q4_K_M-specific
behaviour (the quantizer may change which rows/types are used — the inventory in §2 is from the
real header). HF `config.json` says `eos_token_id = 248044`; the GGUF says 248046
(`<|im_end|>`) — resolve in the tokenizer check.

## 9. Dense Qwen3.8-27B (`qwen35`)

The same GGUF adapter accepts `general.architecture = qwen35`. Its metadata uses the
`qwen35.` prefix and `feed_forward_length` in place of MoE keys. The synthesized text
config has `model_type = qwen3_5_text`, `intermediate_size = 17408`, 64 layers after
subtracting one `nextn_predict_layers` block, and 48 linear V heads over 16 K heads
(`num_v_per_k = 3`). The root model type is `qwen3_5`. The dense FFN names
`ffn_gate`, `ffn_up`, and `ffn_down` map to `mlp.{gate,up,down}_proj` without value
transforms. Gate and up are concatenated for the existing dense SwiGLU path.

`tests/test_gguf_qwen35.cpp` checks every synthesized field against the independent
Qwen3.8-27B HF config using the real Unsloth Q8_0 header at revision `4ca72078`;
it also checks that every non-MTP tensor in the 866-tensor header maps. The r=3
inverse is tested against an independently written forward permutation, including
a no-reorder negative control. `tests/test_gguf_native.cpp` checks the r=3 native
`ssm_out` activation gather and raw Q4_K/Q6_K FFN gate/up concatenation and down
row reads, including K=17408. The header and host tests establish mapping and
geometry; dense tensor *values* and B70 generation remain to be compared with the
safetensors checkpoint. T1–T5 are the same converter transforms verified above
on Ornith; T6 applies only to MoE experts.
