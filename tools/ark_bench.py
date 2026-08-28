import torch, time, itertools, auto_round_kernel as ark
dev="xpu"
# Qwen3.8-27B real prefill shapes, M=4096 (same as grimoire's autotune + production)
SHAPES=[("ffn-gate-up",34816,5120),("ffn-down",5120,17408),("dn-qkv",10240,5120)]
M=4096; G=32
print("kernel: auto_round_kernel.woqgemm_linear   M=%d group=%d"%(M,G))
print("%-14s %7s %7s %10s %12s"%("shape","N","K","ms","TFLOP/s"))
for name,N,K in SHAPES:
    x=torch.randn(M,K,dtype=torch.bfloat16,device=dev)
    w=torch.randn(N,K,dtype=torch.bfloat16,device=dev)
    packed=None; err=None
    for wt in ("int4_clip","int4","sym_int4","int4_fullrange"):
        try:
            packed=ark.repack_quantized_weight(w,N,K,G,"bf16",wt,"bf16",False)
            weight_type=wt; break
        except Exception as e:
            err=e
    if packed is None:
        print("  %-12s repack failed: %s"%(name,str(err)[:110])); continue
    def run():
        y=ark.woqgemm_linear(x,packed,None,N,K,G,"bf16",weight_type,"bf16",False)
        torch.xpu.synchronize(); return y
    try:
        run()
    except Exception as e:
        print("  %-12s call failed: %s"%(name,str(e)[:130])); continue
    best=1e9
    for _ in range(5):
        t=time.perf_counter(); run(); best=min(best,(time.perf_counter()-t)*1e3)
    fl=2*M*N*K
    print("%-14s %7d %7d %10.3f %12.1f   (wt=%s)"%(name,N,K,best,fl/best*1e-9,weight_type))
