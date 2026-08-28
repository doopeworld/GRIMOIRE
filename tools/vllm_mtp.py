# vLLM with MTP, measured -- the reference for what speculation is actually
# worth on THIS box.  Runs baseline and speculative back to back so the only
# variable is the speculative config.
import os, time, json, sys
os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "/models/Qwen3.8-27B-GPTQ-Int4-MTP-BF16")
NTOK  = int(os.environ.get("NTOK", "128"))
K     = int(os.environ.get("SPEC_K", "0"))     # 0 = baseline

def main():
    kw = dict(model=MODEL, dtype="bfloat16", max_model_len=8192,
              max_num_seqs=1, max_num_batched_tokens=8192, block_size=64,
              trust_remote_code=True, enable_prefix_caching=False)
    if K > 0:
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": K}
    print("MODEL=%s  spec_k=%d" % (MODEL, K), flush=True)
    llm = LLM(**kw)
    prompt = "Write a short explanation of how a four-stroke engine works."
    sp = SamplingParams(max_tokens=NTOK, temperature=0, ignore_eos=True)
    llm.generate([prompt], sp)                       # warm
    best = None
    for _ in range(2):
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        ms = (time.perf_counter() - t0) * 1e3
        n = len(out[0].outputs[0].token_ids)
        r = 1000.0 * n / ms
        best = r if best is None else max(best, r)
    print("\nRESULT spec_k=%d  %d tokens  %.1f tok/s  (%.2f ms/token)"
          % (K, NTOK, best, 1000.0 / best), flush=True)

if __name__ == "__main__":
    main()
