import torch, time, vllm, auto_round_kernel as ark
dev="xpu"; M=4096; G=128
SH=[("ffn-gate-up",34816,5120),("ffn-down",5120,17408),("dn-qkv",10240,5120)]
# grimoire production ms/layer (region budget)
GRIM={"ffn-gate-up":936.2/64,"ffn-down":446.4/64,"dn-qkv":202.0/48}
print("BesTLA (ark.woqgemm_linear) vs GRIMOIRE cutlass MXFP4, M=%d G=%d"%(M,G))
print("%-13s %7s %7s %10s %9s | %9s %9s | %s"%("shape","N","K","bestla ms","TFLOP/s","grim ms","TFLOP/s","ratio"))
for name,N,K in SH:
    x=torch.randn(M,K,dtype=torch.bfloat16,device=dev)
    QB=torch.randint(-2**31,2**31-1,(K,N//8),dtype=torch.int32,device=dev)
    sc=torch.randn(K//G,N,dtype=torch.bfloat16,device=dev).abs().add_(0.01)
    for wt in ("int4","int4_clip","sym_int4"):
        try:
            packed=ark.repack_quantized_weight(QB,sc,None,G,"bf16",wt,"bf16",False)
            break
        except Exception as e: last=e; packed=None
    if packed is None:
        print("  %-12s repack failed: %s"%(name,str(last)[:110])); continue
    def run():
        y=ark.woqgemm_linear(x,packed,None,N,K,G,"bf16",wt,"bf16",False)
        torch.xpu.synchronize(); return y
    try: run()
    except Exception as e:
        print("  %-12s call failed: %s"%(name,str(e)[:110])); continue
    best=1e9
    for _ in range(6):
        t=time.perf_counter(); run(); best=min(best,(time.perf_counter()-t)*1e3)
    fl=2*M*N*K; g=GRIM[name]
    print("%-13s %7d %7d %10.3f %9.1f | %9.3f %9.1f | %.2fx  (wt=%s)"%(
        name,N,K,best,fl/best*1e-9,g,fl/g*1e-9,g/best,wt))
