import torch, time, vllm  # vllm import registers _xpu_C
dev="xpu"; M=4096
SHAPES=[("ffn-gate-up",34816,5120),("ffn-down",5120,17408),("dn-qkv",10240,5120)]
# grimoire MXFP4, measured in production today (ms per layer, M=4096)
GRIM={"ffn-gate-up":934.845/64,"ffn-down":446.749/64,"dn-qkv":204.683/64}
for G in (128,32):
    print("\n=== int4_gemm_w4a16 (vLLM/AutoRound kernel), group_size=%d, M=%d ==="%(G,M))
    print("%-13s %7s %7s %9s %10s | %9s %10s | %s"%("shape","N","K","ms","TFLOP/s","grim ms","grim TF","verdict"))
    for name,N,K in SHAPES:
        try:
            x=torch.randn(M,K,dtype=torch.bfloat16,device=dev)
            qw=torch.randint(-2**31,2**31-1,(N,K//8),dtype=torch.int32,device=dev).t()
            sc=torch.randn(K//G,N,dtype=torch.bfloat16,device=dev)
            qz=torch.tensor([8],dtype=torch.int8,device=dev)
            def run():
                y=torch.ops._xpu_C.int4_gemm_w4a16(x,qw,None,sc,qz,G,None)
                torch.xpu.synchronize(); return y
            run()
            best=1e9
            for _ in range(5):
                t=time.perf_counter(); run(); best=min(best,(time.perf_counter()-t)*1e3)
            tf=2*M*N*K/best*1e-9
            g=GRIM[name]; gtf=2*M*N*K/g*1e-9
            print("%-13s %7d %7d %9.3f %10.1f | %9.3f %10.1f | %s"%(
                name,N,K,best,tf,g,gtf,
                "int4 %.2fx FASTER"%(g/best) if best<g else "grimoire %.2fx faster"%(best/g)))
        except Exception as e:
            print("%-13s %7d %7d  FAILED: %s"%(name,N,K,str(e)[:90]))
