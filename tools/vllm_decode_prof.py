# vLLM DECODE profile, per operator, to line up against GRIMOIRE's decode timeline.
import os, torch, time
os.environ["VLLM_TARGET_DEVICE"]="xpu"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
from vllm import LLM, SamplingParams
from torch.profiler import profile, ProfilerActivity

def main():
    MODEL=os.environ.get("MODEL","/models/Qwen3.8-27B-int4-AutoRound")
    NTOK=int(os.environ.get("NTOK","48"))
    llm=LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, max_num_seqs=1,
            max_num_batched_tokens=8192, block_size=64, trust_remote_code=True,
            enable_prefix_caching=False, kv_cache_dtype="fp8")
    tok=llm.get_tokenizer()
    prompt="The capital of France is"          # short: decode dominates
    sp=SamplingParams(max_tokens=NTOK, temperature=0, ignore_eos=True)
    llm.generate([prompt],sp)                   # warmup
    t0=time.perf_counter(); llm.generate([prompt],sp); wall=(time.perf_counter()-t0)*1e3
    print("\nWALL %d decode tokens: %.1f ms -> %.2f ms/token -> %.1f tok/s"
          %(NTOK,wall,wall/NTOK,1000*NTOK/wall),flush=True)
    acts=[ProfilerActivity.CPU]
    if hasattr(ProfilerActivity,"XPU"): acts.append(ProfilerActivity.XPU)
    with profile(activities=acts,record_shapes=False) as prof:
        llm.generate([prompt],sp)
    rows=[]
    for e in prof.key_averages():
        dev=0
        for a in ("self_device_time_total","self_xpu_time_total","self_cuda_time_total"):
            dev=getattr(e,a,0) or dev
            if dev: break
        if dev>0: rows.append((dev/1000.0,e.count,e.key))
    rows.sort(reverse=True)
    tot=sum(r[0] for r in rows)
    print("\n=== vLLM DECODE, %d tokens, device time by op ==="%NTOK)
    print("%-52s %9s %7s %9s %7s"%("op","ms","count","us/tok","%"))
    for ms,c,k in rows[:26]:
        print("%-52s %9.2f %7d %9.1f %6.1f%%"%(k[:52],ms,c,ms*1000/NTOK,100*ms/tot))
    print("%-52s %9.2f %7s %9.1f"%("TOTAL device",tot,"",tot*1000/NTOK))

if __name__=="__main__":
    main()
