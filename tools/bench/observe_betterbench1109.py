"""Durable progress outside timed requests; official BetterBench source unchanged."""
import json,os,statistics,sys
from pathlib import Path
from betterbench import cli,runner
phase='';warmup=0;counts={};measured={}
trace=Path(os.environ['BB_PROGRESS'])
original_one=runner._one
async def one(*args,**kwargs):
    result=await original_one(*args,**kwargs)
    if phase=='single':
        cat=result.category;counts[cat]=counts.get(cat,0)+1
        row=result.as_dict();row['warmup']=counts[cat]<=warmup
        with trace.open('a') as f:f.write(json.dumps(row)+'\n')
        if not row['warmup']:
            measured.setdefault(cat,[]).append(row)
            good=[r['decode_tps'] for r in measured[cat] if r.get('ok') and r.get('decode_tps')]
            if good and len(measured[cat])%5==0:
                print(f"[measured] {cat} {len(measured[cat])} passes median {statistics.median(good):.2f} tok/s",flush=True)
    return result
runner._one=one
# Preserve each completed prefill request, rather than waiting until
# the entire depth sweep returns. The first two rows per depth are warmup.
original_stream=runner.stream_chat
async def stream(*args,**kwargs):
    result=await original_stream(*args,**kwargs)
    if kwargs.get('category')=='prefill':
        with trace.with_suffix('.prefill-progress.jsonl').open('a') as f:
            f.write(json.dumps(result.as_dict())+'\n')
    return result
runner.stream_chat=stream
original_single=cli.single_stream
async def single(*args,**kwargs):
    global phase,warmup
    phase='single';warmup=args[3].warmup
    result=await original_single(*args,**kwargs)
    trace.with_suffix('.single.json').write_text(json.dumps(result,indent=2))
    phase='';return result
cli.single_stream=single
original_prefill=cli.prefill_sweep
async def prefill(*args,**kwargs):
    result=await original_prefill(*args,**kwargs)
    trace.with_suffix('.prefill.json').write_text(json.dumps(result,indent=2))
    return result
cli.prefill_sweep=prefill
original_concurrency=cli.concurrency_sweep
async def concurrency(*args,**kwargs):
    result=await original_concurrency(*args,**kwargs)
    trace.with_suffix('.concurrency.json').write_text(json.dumps(result,indent=2))
    return result
cli.concurrency_sweep=concurrency
cli.main()
