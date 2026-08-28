import torch, json, glob, os, time
from safetensors import safe_open
import auto_round_kernel as ark
MD="/models/Qwen3.8-27B-int4-AutoRound"; G=128
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
# vLLM (qlinear.py): symmetric -> intweight -= qbias (8), to int8; zeros = EMPTY int8
iw=(u.to(torch.int16)-8).to(torch.int8).contiguous().to("xpu")
zeros=torch.empty(0,dtype=torch.int8).to("xpu")
sc16=sc32.to(torch.float16).contiguous().to("xpu")
GRIM={"gate":936.2/64}
for M in (8,4096):
    x=torch.randn(M,K,dtype=torch.bfloat16)
    ref=(x.to(torch.float32)@((u.to(torch.float32)-8.0)*sc32.repeat_interleave(G,dim=0))) if M==8 else None
    print("--- M=%d ---"%M)
    for cdt in ("int8","bf16","fp16"):
        try:
            packed=ark.repack_quantized_weight(iw,sc16,zeros,G,cdt,"int4","fp16",False)
            xx=x.to("xpu")
            o=ark.woqgemm_linear(xx,packed,None,N,K,G,cdt,"int4","fp16",False); torch.xpu.synchronize()
            best=1e9
            for _ in range(5):
                s=time.perf_counter()
                ark.woqgemm_linear(xx,packed,None,N,K,G,cdt,"int4","fp16",False)
                torch.xpu.synchronize(); best=min(best,(time.perf_counter()-s)*1e3)
            msg="cdt=%-5s %8.3f ms  %7.1f TFLOP/s"%(cdt,best,2*M*N*K/best*1e-9)
            if ref is not None:
                c=torch.nn.functional.cosine_similarity(o.to("cpu").to(torch.float32).flatten(),ref.flatten(),dim=0).item()
                msg+="   cosine=%+.4f"%c
            if M==4096: msg+="   (grimoire 14.63 ms)"
            print("   "+msg)
        except Exception as e:
            print("   cdt=%-5s FAILED: %s"%(cdt,str(e)[:70]))
