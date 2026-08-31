"""Per-stage prefill timing on Fusion's TARGET model, for direct comparison
against Grimoire's GRIMOIRE_MUSE_TIME_LAYER=all output at the same prompt
length. Hooks the same stage boundaries Grimoire times: qkv proj, attention,
o_proj, FFN gate_up/swiglu/down -- summed across all layers, host-clock only
(no torch profiler autograd overhead skewing the numbers), matching Grimoire's
drain-and-clock methodology so the two are comparable.
"""
import os, time, json

os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_XPU_ENABLE_XPU_GRAPH", "0")  # eager: hooks need real forward calls

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

TARGET = os.environ.get("TARGET", "/models/Muse-Glimmer-30B-INT4-W4A16")
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "3672"))

sums = {}
order = []


def add(name, dt):
    if name not in sums:
        sums[name] = 0.0
        order.append(name)
    sums[name] += dt


def sync():
    torch.xpu.synchronize()


def wrap(mod, name):
    orig = mod.forward

    def timed(*a, **kw):
        sync()
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        sync()
        add(name, (time.perf_counter() - t0) * 1000.0)
        return out

    mod.forward = timed


llm = LLM(
    model=TARGET,
    dtype="float16",
    max_model_len=8192,
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.95")),
    max_num_seqs=16,
    max_num_batched_tokens=16384,
    block_size=64,
    trust_remote_code=True,
    enforce_eager=True,
)

driver = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker
model = driver.worker.model_runner.model
layers = model.model.layers if hasattr(model, "model") else model.layers
print("layers: %d" % len(layers), flush=True)

for li, layer in enumerate(layers):
    tag = "L%d" % li
    attn = layer.self_attn
    wrap(attn.qkv_proj, "qkv proj")
    # Wrap the inner Attention op only, not the whole attn module -- wrapping
    # both would double-count qkv_proj/o_proj time inside the outer sum.
    if hasattr(attn, "attn"):
        wrap(attn.attn, "attention (core)")
    wrap(attn.o_proj, "o_proj")
    mlp = layer.mlp
    for cand in ("gate_up_proj", "up_proj"):
        if hasattr(mlp, cand):
            wrap(getattr(mlp, cand), "FFN gate_up")
            break
    if hasattr(mlp, "down_proj"):
        wrap(mlp.down_proj, "FFN down")
    if hasattr(mlp, "act_fn"):
        wrap(mlp.act_fn, "FFN act")

ids = list(range(200000, 200000 + PROMPT_TOKENS))
prompt = TokensPrompt(prompt_token_ids=ids)

t0 = time.perf_counter()
out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1))
sync()
total_ms = (time.perf_counter() - t0) * 1000.0

print("\nFusion prefill per-stage sums over %d tokens (wall total incl. non-hooked: %.1f ms):"
      % (PROMPT_TOKENS, total_ms), flush=True)
grand = sum(sums.values())
for name in order:
    print("  %-24s %10.2f ms  (%5.1f%% of hooked)" % (name, sums[name],
                                                       100.0 * sums[name] / grand))
print("  %-24s %10.2f ms" % ("SUM (hooked)", grand))
print("\nJSON=" + json.dumps(sums))
