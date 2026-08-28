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
u=((qw.unsqueeze(1)>>SH.view(1,8,1))&0xF).reshape(qw.shape[0]*8,qw.shape[1])  # [K,N]
x=torch.randn(M,K,dtype=torch.bfloat16)
w_ref=((u.to(torch.float32)-8.0)*sc32.repeat_interleave(G,dim=0))
ref=(x.to(torch.float32)@w_ref)
print("ref |mean|=%.4f   K=%d N=%d"%(ref.abs().mean(),K,N))
best=None
for zp,wT,sT in itertools.product((8,0),(False,True),(False,True)):
    qi=(u.to(torch.int16)-zp).to(torch.int8)
    s=sc32.to(torch.bfloat16)
    if wT: qi=qi.t()
    if sT: s=s.t()
    try:
        packed=ark.repack_quantized_weight(qi.contiguous().to("xpu"),
                 s.contiguous().to("xpu"),None,G,"bf16","int4","bf16",False)
        out=ark.woqgemm_linear(x.to("xpu"),packed,None,N,K,G,"bf16","int4","bf16",False)
        torch.xpu.synchronize()
        o=out.to("cpu").to(torch.float32)
        cos=torch.nn.functional.cosine_similarity(o.flatten(),ref.flatten(),dim=0).item()
        print("  ZP=%d w%-5s s%-5s -> cosine=%+.4f  |out|=%.4f"%(
              zp,"T" if wT else ".","T" if sT else ".",cos,o.abs().mean()))
        if best is None or cos>best[0]: best=(cos,zp,wT,sT)
    except Exception as e:
        print("  ZP=%d w%-5s s%-5s -> %s"%(zp,"T" if wT else ".","T" if sT else ".",str(e)[:70]))
print("BEST:",best)
