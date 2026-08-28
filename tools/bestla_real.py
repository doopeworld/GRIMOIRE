import torch, time, json, glob, os
from safetensors import safe_open
import vllm, auto_round_kernel as ark
dev="xpu"; M=4096
MD="/models/Qwen3.8-27B-int4-AutoRound"
idx=json.load(open(glob.glob(MD+"/*.index.json")[0]))["weight_map"]
qcfg=json.load(open(MD+"/config.json")).get("quantization_config",{})
G=qcfg.get("group_size",128); sym=qcfg.get("sym",True)
print("checkpoint group_size=%s sym=%s bits=%s"%(G,sym,qcfg.get("bits")))

def load(base):
    out={}
    for suf in ("qweight","scales","qzeros"):
        k=base+"."+suf
        if k not in idx: continue
        with safe_open(os.path.join(MD,idx[k]),framework="pt") as f:
            out[suf]=f.get_tensor(k)
    return out

# real Qwen FFN tensors
CAND=[("ffn-gate(up-half)","model.language_model.layers.0.mlp.gate_proj"),
      ("ffn-down","model.language_model.layers.0.mlp.down_proj")]
GRIM={"ffn-gate(up-half)":936.2/64/2,"ffn-down":446.4/64}
print("%-18s %7s %7s %10s %9s | %9s %9s | %s"%("shape","N","K","bestla ms","TFLOP/s","grim ms","TFLOP/s","ratio"))
for name,base in CAND:
    t=load(base)
    if "qweight" not in t: print("  %-16s tensors not found"%name); continue
    qw=t["qweight"].to(dev); sc=t["scales"].to(dev)
    qz=t.get("qzeros"); qz=qz.to(dev) if qz is not None else None
    K=qw.shape[0]*8 if qw.dtype==torch.int32 else qw.shape[0]
    N=sc.shape[1] if sc.dim()==2 else qw.shape[1]
    # derive from scales: scales is [K//G, N]
    if sc.dim()==2: K=sc.shape[0]*G; N=sc.shape[1]
    print("   %s qweight%s %s  scales%s"%(name,tuple(qw.shape),qw.dtype,tuple(sc.shape)),flush=True)
    x=torch.randn(M,K,dtype=torch.bfloat16,device=dev)
    # repack wants UNPACKED int8 [K,N] (it reads K from QB.shape[0]); the checkpoint
    # stores GPTQ int32 [K/8,N] with 8 nibbles per word along K.
    if qw.dtype==torch.int32:
        sh=torch.arange(8,device=dev,dtype=torch.int32)*4
        u=((qw.unsqueeze(1)>>sh.view(1,8,1))&0xF)          # [K/8, 8, N]
        qw8=u.reshape(qw.shape[0]*8, qw.shape[1]).to(torch.int8).contiguous()
        print("   unpacked -> %s %s"%(tuple(qw8.shape),qw8.dtype),flush=True)
    else:
        qw8=qw
    packed=None
    for wt in ("int4","int4_clip"):
        for zz in (qz,None):
            try:
                packed=ark.repack_quantized_weight(qw8,sc.to(torch.bfloat16),zz,G,"bf16",wt,"bf16",not sym)
                WT=wt; break
            except Exception as e: err=e
        if packed is not None: break
    if packed is None:
        print("   repack failed: %s"%str(err)[:130]); continue
    def run():
        y=ark.woqgemm_linear(x,packed,None,N,K,G,"bf16",WT,"bf16",not sym)
        torch.xpu.synchronize(); return y
    try: run()
    except Exception as e:
        print("   call failed: %s"%str(e)[:130]); continue
    best=1e9
    for _ in range(6):
        s=time.perf_counter(); run(); best=min(best,(time.perf_counter()-s)*1e3)
    fl=2*M*N*K; g=GRIM[name]
    print("%-18s %7d %7d %10.3f %9.1f | %9.3f %9.1f | %.2fx"%(name,N,K,best,fl/best*1e-9,g,fl/g*1e-9,g/best))
