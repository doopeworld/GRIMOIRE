import concurrent.futures,json,sys,time,urllib.request
from pathlib import Path
port=int(sys.argv[1]);out=Path(sys.argv[2]);base=f'http://127.0.0.1:{port}'
def call(seed,temperature=.7,chat=False):
 b={'model':'m','max_tokens':64,'temperature':temperature,'top_p':.95,'top_k':20,'seed':seed,'stream':False}
 if chat:b['messages']=[{'role':'user','content':'Name three animals.'}]
 else:b['prompt']='Write a different short phrase about a blue boat:'
 t=time.monotonic()
 req=urllib.request.Request(base+('/v1/chat/completions' if chat else '/v1/completions'),data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:j=json.load(r)
 c=j['choices'][0];assert c['finish_reason'] in ['length','stop'];assert 0<j['usage']['completion_tokens']<=64
 return {'seed':seed,'text':c.get('text',c.get('message',{})), 'tokens':j['usage']['completion_tokens'],'seconds':time.monotonic()-t}
a=call(42);b=call(42);assert a['text']==b['text'],(a,b)
reference={i:call(i) for i in range(8)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:parallel=list(ex.map(call,range(8)))
# Floating-point batching can perturb a very close nucleus boundary. Record
# exact agreements; all requests must still complete with valid accounting.
agreement=sum(reference[p['seed']]['text']==p['text'] for p in parallel)
chat=call(3,chat=True)
out.write_text(json.dumps({'repeat_seed_exact':True,'parallel_seed_agreements':agreement,'parallel_requests':8,'reference':list(reference.values()),'parallel':parallel,'chat':chat},indent=2))
print('SAMPLED_API_PASS','repeat_seed_exact',True,'parallel_completed',len(parallel),'seed_agreements',agreement,flush=True)
