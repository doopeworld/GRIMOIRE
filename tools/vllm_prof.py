import os, torch
os.environ["VLLM_TARGET_DEVICE"]="xpu"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"   # in-process, so the profiler sees the ops
from vllm import LLM, SamplingParams
from torch.profiler import profile, ProfilerActivity

def main():
    MODEL="/models/Qwen3.8-27B-int4-AutoRound"
    llm=LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, max_num_seqs=1,
            max_num_batched_tokens=8192, block_size=64, trust_remote_code=True,
            enable_prefix_caching=False)
    tok=llm.get_tokenizer()
    word=("The industrial revolution transformed manufacturing through steam power and "
          "mechanised textile production across Britain and later the world. ")
    def mk(seed):
        ids=tok((word+f"Variant {seed}. ")*400).input_ids[:4096]
        return tok.decode(ids)
    warm=mk(1); prompt=mk(2)          # DIFFERENT prompts: no prefix-cache reuse
    print("warm tokens:",len(tok(warm).input_ids),
          "measured tokens:",len(tok(prompt).input_ids),flush=True)
    sp=SamplingParams(max_tokens=1,temperature=0)
    llm.generate([warm],sp)
    acts=[ProfilerActivity.CPU]
    if hasattr(ProfilerActivity,"XPU"): acts.append(ProfilerActivity.XPU)
    with profile(activities=acts,record_shapes=False) as prof:
        llm.generate([prompt],sp)
    rows=[]
    for e in prof.key_averages():
        dev=0
        for attr in ("self_device_time_total","self_xpu_time_total","self_cuda_time_total"):
            dev=getattr(e,attr,0) or dev
            if dev: break
        if dev>0: rows.append((dev/1000.0,e.count,e.key))
    rows.sort(reverse=True)
    tot=sum(r[0] for r in rows)
    print("\n=== vLLM PREFILL 4096 — device time by op (ms) ===")
    print("%-56s %9s %6s %6s"%("op","ms","count","%"))
    for ms,c,k in rows[:30]:
        print("%-56s %9.2f %6d %5.1f%%"%(k[:56],ms,c,100*ms/tot))
    print("%-56s %9.2f"%("TOTAL device",tot))

if __name__=="__main__":
    main()
