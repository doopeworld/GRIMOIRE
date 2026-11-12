"""oneCCL all-reduce latency on the Tower's two B70s with Ian's old vLLM "p2p" settings (2026-11-12).
Two processes, one per card, both cards visible (ZE_AFFINITY_MASK=0,1 as vLLM sets it); torch.distributed over
xccl (or ccl), bf16 tensors at Grimoire's verify size (7 x 5120) and a few others; 20 warm-up + 500 timed calls.
oneCCL's log (CCL_LOG_LEVEL=info) names the transport it picked.  Run inside intel/llm-scaler-vllm."""
import os, sys, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [(1, 5120), (7, 5120), (16, 5120), (256, 5120), (1024, 5120)]


def worker(rank, world):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29561')
    torch.xpu.set_device(rank)
    backend = 'xccl' if 'xccl' in dist.Backend.backend_list else 'ccl'
    if backend == 'ccl':
        import oneccl_bindings_for_pytorch  # noqa: F401
    dist.init_process_group(backend, rank=rank, world_size=world, device_id=torch.device('xpu', rank))
    if rank == 0:
        print('backend', backend, 'torch', torch.__version__, flush=True)
    for rows, cols in SIZES:
        x = torch.ones(rows, cols, dtype=torch.bfloat16, device='xpu') * (rank + 1)
        for _ in range(20):
            dist.all_reduce(x)
        torch.xpu.synchronize()
        x.fill_(rank + 1)
        n = 500
        t0 = time.perf_counter()
        for _ in range(n):
            dist.all_reduce(x)
        torch.xpu.synchronize()
        us = (time.perf_counter() - t0) / n * 1e6
        ok = bool(torch.isfinite(x.float()).all().item())
        if rank == 0:
            print(f'all_reduce bf16 {rows}x{cols} ({rows * cols * 2 / 1024:.0f} KiB): {us:.1f} us per call, finite={ok}',
                  flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    print('CCL_PROBE_DONE', flush=True)
