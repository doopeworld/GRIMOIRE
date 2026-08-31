"""Dump the first DFlash proposal cycle's tensors from Fusion, for parity
comparison against Grimoire's GRIMOIRE_DFLASH_DUMP output.

Files are written as raw little-endian float32, name-matched to Grimoire's
`g_*.f32` with an `f_` prefix.
"""
import os, sys, json

os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

DUMP = os.environ["FUSION_DUMP"]
os.makedirs(DUMP, exist_ok=True)
TARGET = os.environ.get("TARGET", "/models/Muse-Glimmer-30B-INT4-W4A16")
DRAFT = os.environ.get("DRAFT", "/models/Muse-Glimmer-assistant")
K = int(os.environ.get("K", "15"))
PROMPT_IDS = os.environ.get("PROMPT_IDS", "")

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.model_executor.models import qwen3_dflash as QD

_state = {"cycle": 0, "ctx_chunk": 0}


def save(name, t):
    if _state["cycle"] != 0:
        return
    a = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    a.tofile(os.path.join(DUMP, "f_%s.f32" % name))
    print("  fusion dump: %s %s" % (name, tuple(t.shape)), flush=True)


# ---- stage 1/2: aux concat + fc -------------------------------------------
_orig_combine = QD.DFlashQwen3ForCausalLM.combine_hidden_states


def combine_hidden_states(self, hidden_states):
    save("01_aux_0", hidden_states)
    out = _orig_combine(self, hidden_states)
    save("02_fc_0", out)
    return out


QD.DFlashQwen3ForCausalLM.combine_hidden_states = combine_hidden_states

# ---- stage 3: context K/V --------------------------------------------------
_orig_proj = QD.DFlashQwen3Model._project_context_kv
_orig_precompute = QD.DFlashQwen3Model.precompute_and_store_context_kv


def _project_context_kv(self, context_states, num_ctx, num_layers, nkv, hd):
    import vllm._custom_ops as ops
    normed = torch.empty_like(context_states)
    ops.rms_norm(normed, context_states, self._hidden_norm_weight,
                 self._rms_norm_eps)
    save("03_ctxnorm_0", normed)
    import torch.nn.functional as F
    all_kv_flat = F.linear(normed, self._fused_kv_weight, self._fused_kv_bias)
    save("04_ctxkv_0", all_kv_flat)
    all_kv = (all_kv_flat.view(num_ctx, num_layers, 2, nkv, hd)
              .permute(2, 1, 0, 3, 4).contiguous())
    return all_kv[0], all_kv[1]


QD.DFlashQwen3Model._project_context_kv = _project_context_kv


def precompute_and_store_context_kv(self, context_states, context_positions,
                                    context_slot_mapping=None):
    if context_slot_mapping is not None:
        save("00b_ctxstates", context_states)
    r = _orig_precompute(self, context_states, context_positions,
                         context_slot_mapping)
    return r


QD.DFlashQwen3Model.precompute_and_store_context_kv = \
    precompute_and_store_context_kv

# ---- stage 4/5/6: draft block forward -------------------------------------
_orig_fwd = QD.DFlashQwen3Model.forward


def forward(self, input_ids, positions, input_embeds=None):
    if input_embeds is None:
        input_embeds = self.embed_input_ids(input_ids)
    save("07_blockembed", input_embeds)
    save("07b_positions", positions.to(torch.float32))
    save("07c_input_ids", input_ids.to(torch.float32))

    hidden_states = input_embeds
    residual = None
    for li, layer in enumerate(self.layers):
        attn = layer.self_attn
        if residual is None:
            normed, res = layer.input_layernorm(hidden_states), hidden_states
        else:
            normed, res = layer.input_layernorm(hidden_states, residual)
        save("08_L%d_innorm" % li, normed)
        qkv = attn.qkv_proj(normed)
        if isinstance(qkv, tuple):
            qkv = qkv[0]
        save("09_L%d_qkv" % li, qkv)
        hidden_states, residual = layer(positions=positions,
                                        hidden_states=hidden_states,
                                        residual=residual)
        save("15_L%d_mlp" % li, hidden_states)
        save("15b_L%d_residual" % li, residual)
    hidden_states, _ = self.norm(hidden_states, residual)
    save("16_finalnorm", hidden_states)
    return hidden_states


QD.DFlashQwen3Model.forward = forward

_orig_logits = QD.DFlashQwen3ForCausalLM.compute_logits


def compute_logits(self, hidden_states):
    out = _orig_logits(self, hidden_states)
    save("17_logits", out)
    if _state["cycle"] == 0:
        ids = out.argmax(dim=-1).tolist()
        with open(os.path.join(DUMP, "f_17_argmax.txt"), "w") as f:
            f.write(json.dumps(ids))
        print("  fusion dump: draft argmax %s" % ids, flush=True)
        _state["cycle"] = 1
    return out


QD.DFlashQwen3ForCausalLM.compute_logits = compute_logits

# ---- run -------------------------------------------------------------------
llm = LLM(
    model=TARGET,
    dtype="float16",
    max_model_len=12288,
    gpu_memory_utilization=0.90,
    max_num_seqs=16,
    max_num_batched_tokens=16384,
    block_size=64,
    trust_remote_code=True,
    enforce_eager=True,
    speculative_config={"method": "dflash", "model": DRAFT,
                        "num_speculative_tokens": K},
)

if PROMPT_IDS:
    ids = [int(x) for x in PROMPT_IDS.replace("\n", "").split(",") if x.strip()]
    with open(os.path.join(DUMP, "f_00_prompt_ids.txt"), "w") as f:
        f.write(",".join(str(i) for i in ids))
    prompt = TokensPrompt(prompt_token_ids=ids)
else:
    prompt = "Write a haiku about the ocean."

out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))
print("OUTPUT TOKENS:", out[0].outputs[0].token_ids)
print("OUTPUT TEXT:", repr(out[0].outputs[0].text))
