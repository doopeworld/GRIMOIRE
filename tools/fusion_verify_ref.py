"""Ground truth for Grimoire's DFlash verifier.

Runs Fusion's TARGET model (no speculative config) over the exact token
sequence Grimoire verifies -- the 64 prompt ids followed by the 16 candidate
rows -- and reports the target's argmax at every position 64..79. That is,
element for element, what Grimoire's `verified[]` must contain.
"""
import os, json

os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_XPU_ENABLE_XPU_GRAPH", "1")

TARGET = os.environ.get("TARGET", "/models/Muse-Glimmer-30B-INT4-W4A16")
PROMPT_IDS = [int(x) for x in os.environ["PROMPT_IDS"].replace("\n", "").split(",") if x.strip()]
CANDS = [int(x) for x in os.environ["CANDIDATES"].split(",") if x.strip()]

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(
    model=TARGET,
    dtype="float16",
    max_model_len=int(os.environ.get("MAX_LEN", "2048")),
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.95")),
    max_num_seqs=16,
    max_num_batched_tokens=16384,
    block_size=64,
    trust_remote_code=True,
)

seq = PROMPT_IDS + CANDS
print("prompt=%d cands=%d total=%d" % (len(PROMPT_IDS), len(CANDS), len(seq)), flush=True)

out = llm.generate(
    [TokensPrompt(prompt_token_ids=seq)],
    SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1),
)
pl = out[0].prompt_logprobs

# prompt_logprobs[i] describes position i; the top-1 entry there is the token
# the target would emit given positions [0, i). So the prediction made AT
# position p is found at index p+1.
ref = []
for p in range(len(PROMPT_IDS) - 1, len(seq) - 1):
    d = pl[p + 1]
    top = max(d.items(), key=lambda kv: kv[1].logprob)
    ref.append((p, top[0]))

print("\nTarget argmax at each verified position:")
for p, t in ref:
    print("  pos %3d -> %d" % (p, t))
print("\nREF_VERIFIED=" + ",".join(str(t) for _, t in ref))
