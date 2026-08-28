# Generic speculative comparison. Spec config is built from plain env vars so
# no JSON has to survive shell quoting.
import os, time
os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams

MODEL  = os.environ["MODEL"]
NTOK   = int(os.environ.get("NTOK", "128"))
METHOD = os.environ.get("SPEC_METHOD", "").strip()
DRAFT  = os.environ.get("SPEC_MODEL", "").strip()
K      = int(os.environ.get("SPEC_K", "0"))
UTIL   = float(os.environ.get("GPU_UTIL", "0.92"))

def main():
    kw = dict(model=MODEL, dtype="bfloat16", max_model_len=int(os.environ.get("MAX_LEN","4096")),
              max_num_seqs=1, max_num_batched_tokens=4096, block_size=64,
              trust_remote_code=True, enable_prefix_caching=False,
              gpu_memory_utilization=UTIL, kv_cache_dtype="fp8")
    if METHOD:
        spec = {"method": METHOD, "num_speculative_tokens": K}
        if DRAFT:
            spec["model"] = DRAFT
        kw["speculative_config"] = spec
        print("SPEC:", spec, flush=True)
    else:
        print("SPEC: none (baseline)", flush=True)
    llm = LLM(**kw)
    prompt = "Write a Python function that implements merge sort, with comments."
    sp = SamplingParams(max_tokens=NTOK, temperature=0, ignore_eos=True)
    llm.generate([prompt], sp)
    best = 0.0
    for _ in range(2):
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        ms = (time.perf_counter() - t0) * 1e3
        n = len(out[0].outputs[0].token_ids)
        best = max(best, 1000.0 * n / ms)
    tag = "%s k=%d" % (METHOD, K) if METHOD else "baseline"
    print("\nRESULT %s  %d tokens  %.1f tok/s  (%.2f ms/token)"
          % (tag, NTOK, best, 1000.0 / best), flush=True)

if __name__ == "__main__":
    main()
