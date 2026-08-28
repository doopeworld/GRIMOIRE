import torch, time, vllm
dev="xpu"; M=4096; G=128
SH=[("ffn-gate-up",34816,5120),("ffn-down",5120,17408)]
print("vLLM int4_gemm_w4a16 : 1 matrix (cache-hot) vs 64 distinct (production-like)")
print("%-13s %11s %12s %12s"%("shape","1 matrix","64 distinct","delta/layer"))
for name,N,K in SH:
    x=torch.randn(M,K,dtype=torch.bfloat16,device=dev)
    qz=torch.tensor([8],dtype=torch.int8,device=dev)
    W=[];S=[]
    L=64
    try:
        for i in range(L):
            W.append(torch.randint(-2**31,2**31-1,(N,K//8),dtype=torch.int32,device=dev).t())
            S.append(torch.randn(K//G,N,dtype=torch.bfloat16,device=dev))
    except RuntimeError as e:
        L=len(W); print("   (only %d matrices fit)"%L)
    def one(w,s):
        torch.ops._xpu_C.int4_gemm_w4a16(x,w,None,s,qz,G,None)
    one(W[0],S[0]); torch.xpu.synchronize()
    a=1e9
    for _ in range(6):
        t=time.perf_counter(); one(W[0],S[0]); torch.xpu.synchronize()
        a=min(a,(time.perf_counter()-t)*1e3)
    b=1e9
    for _ in range(3):
        t=time.perf_counter()
        for i in range(L): one(W[i],S[i])
        torch.xpu.synchronize()
        b=min(b,(time.perf_counter()-t)*1e3/L)
    print("%-13s %8.3f ms %9.3f ms %+9.3f ms"%(name,a,b,b-a))
    del W,S; torch.xpu.empty_cache()
