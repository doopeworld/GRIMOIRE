import torch, json, glob, os
from safetensors import safe_open
import auto_round_kernel as ark
MD="/models/Qwen3.8-27B-int4-AutoRound"; G=128; M=8
idx=json.load(open(glob.glob(MD+"/*.index.json")[0]))["weight_map"]
base="model.language_model.layers.0.mlp.gate_proj"
t={}
for suf in ("qweight","scales","qzeros"):
    k=base+"."+suf
    if k in idx:
        with safe_open(os.path.join(MD,idx[k]),framework="pt") as f: t[suf]=f.get_tensor(k)
qw=t["qweight"]; sc=t["scales"].to(torch.float32)
print("qweight",tuple(qw.shape),qw.dtype," scales",tuple(sc.shape)," qzeros",("qzeros" in t))
K=sc.shape[0]*G; N=sc.shape[1]
SH=torch.arange(8,dtype=torch.int32)*4
u=((qw.unsqueeze(1)>>SH.view(1,8,1))&0xF).reshape(qw.shape[0]*8,qw.shape[1])   # [K,N] 0..15
print("nibble range",int(u.min()),int(u.max()))
x=torch.randn(M,K,dtype=torch.bfloat16)
# reference: dequantise on CPU in fp32.  symmetric -> centre at 8
w_ref=((u.to(torch.float32)-8.0)*sc.repeat_interleave(G,dim=0))            # [K,N]
ref=(x.to(torch.float32)@w_ref)
for zp,tag in ((8,"centred (u-8)"),(0,"raw u")):
    qi=(u.to(torch.int16)-zp).to(torch.int8).contiguous().to("xpu")
    packed=ark.repack_quantized_weight(qi,sc.to(torch.bfloat16).to("xpu"),None,G,
                                       "bf16","int4","bf16",False)
    out=ark.woqgemm_linear(x.to("xpu"),packed,None,N,K,G,"bf16","int4","bf16",False)
    torch.xpu.synchronize()
    o=out.to("cpu").to(torch.float32)
    num=(o-ref).abs().mean().item(); den=ref.abs().mean().item()
    cos=torch.nn.functional.cosine_similarity(o.flatten(),ref.flatten(),dim=0).item()
    print("  ZP=%d %-14s mean|err|=%.4f  ref scale=%.4f  cosine=%.4f"%(zp,tag,num,den,cos))
