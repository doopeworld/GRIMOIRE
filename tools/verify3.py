import torch, json, glob, os, itertools
from safetensors import safe_open
import auto_round_kernel as ark
MD="/models/Qwen3.8-27B-int4-AutoRound"; G=128; M=8
idx=json.load(open(glob.glob(MD+"/*.index.json")[0]))["weight_map"]
base="model.language_model.layers.0.mlp.gate_proj"
t={}
for suf in ("qweight","scales"):
    k=base+"."+suf
    with safe_open(os.path.join(MD,idx[k]),framework="pt") as f: t[suf]=f.get_tensor(k)
qw=t["qweight"]; sc32=t["scales"].to(torch.float32)
K=sc32.shape[0]*G; N=sc32.shape[1]
SH=torch.arange(8,dtype=torch.int32)*4
u=((qw.unsqueeze(1)>>SH.view(1,8,1))&0xF).reshape(qw.shape[0]*8,qw.shape[1])
x=torch.randn(M,K,dtype=torch.bfloat16)
ref=(x.to(torch.float32)@((u.to(torch.float32)-8.0)*sc32.repeat_interleave(G,dim=0)))
print("ref |mean|=%.4f  K=%d N=%d"%(ref.abs().mean(),K,N))
for ct,st,sdt in itertools.product(("fp16","bf16"),("fp16","bf16"),("fp16","bf16")):
    qi=(u.to(torch.int16)-8).to(torch.int8).contiguous().to("xpu")
    s=sc32.to(torch.float16 if sdt=="fp16" else torch.bfloat16).contiguous().to("xpu")
    try:
        packed=ark.repack_quantized_weight(qi,s,None,G,ct,"int4",st,False)
        out=ark.woqgemm_linear(x.to("xpu"),packed,None,N,K,G,ct,"int4",st,False)
        torch.xpu.synchronize()
        o=out.to("cpu").to(torch.float32)
        cos=torch.nn.functional.cosine_similarity(o.flatten(),ref.flatten(),dim=0).item()
        print("  compute=%-4s scale=%-4s sdtype=%-4s -> cosine=%+.4f |out|=%.4f"%(ct,st,sdt,cos,o.abs().mean()))
    except Exception as e:
        print("  compute=%-4s scale=%-4s sdtype=%-4s -> %s"%(ct,st,sdt,str(e)[:60]))
