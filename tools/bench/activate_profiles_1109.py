"""Recreate through dockerMan, check each profile, leave only the user's Qwen running."""
import json,re,subprocess,time,urllib.request
from pathlib import Path
b=Path('/mnt/storage/isos/grimoire-runs/bench-1022')
assert 'CANDIDATE_SCOPE_PASS' in (b/'logs/candidate-validation1109.out').read_text()
meta=json.loads((b/'logs/profile-update1109.json').read_text());image_id=meta['image_id']
profiles=meta['profiles']
def inspect(name):
    p=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
    return json.loads(p.stdout)[0] if p.returncode==0 else None
def run(*args):return subprocess.run(list(args),check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True).stdout
def health(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:return json.load(r).get('status')=='ok'
    except Exception:return False
def counts():
    result={}
    for addr in ['0000:03:00.0','0000:07:00.0']:
        for part in (Path('/sys/bus/pci/devices')/addr).resolve().parts:
            if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]',part):continue
            for kind in ['nonfatal','fatal']:
                f=Path('/sys/bus/pci/devices')/part/('aer_dev_'+kind)
                if f.exists():
                    for line in f.read_text().splitlines():
                        if line.startswith('TOTAL_ERR_'):result[(part,kind)]=int(line.split()[1])
    assert result, "No GPU-path PCIe AER counters found"
    return result
results=[]
for profile in profiles:
    name,port=profile['name'],profile['port'];before=counts();syslog=Path('/var/log/syslog');offset=syslog.stat().st_size
    old=inspect(name)
    if old and old['State']['Running']:run('docker','stop','--time','300',name)
    create=run('php','/mnt/storage/isos/grimoire-runs/bench-1003/create_from_template.php',name)
    (b/'logs'/('profile-create1109-'+name+'.log')).write_text(create)
    run('docker','start',name)
    ready=False
    for _ in range(500):
        c=inspect(name)
        if not c or not c['State']['Running']:break
        if counts()!=before:raise RuntimeError('GPU path error during '+name)
        if health(port):ready=True;break
        time.sleep(3)
    log=run('docker','logs',name)
    (b/'logs'/('profile-server1109-'+name+'.log')).write_text(log)
    assert ready, 'Profile did not become healthy; inspect without killing ongoing GPU work: '+name
    if 'FLASH-NEXT' in name:
        # Let the existing PLE background load finish before the stop/recreate cycle.
        for _ in range(150):
            log=run('docker','logs',name)
            if 'PLE table: copying' not in log or 'PLE table in memory' in log:break
            time.sleep(2)
    c=inspect(name);args=c['Args'];model=args[args.index('--model')+1]
    assert c['Image']==image_id and c['Config']['Labels']['net.unraid.docker.managed']=='dockerman'
    assert args[args.index('--ctx')+1]==str(profile['ctx']) and args[args.index('--proj')+1]==profile['proj']
    assert 'pci 0000:03:00.0' in log and 'pci 0000:0b:00.0' not in log
    if profile['cards']==2:assert 'pci 0000:07:00.0' in log
    payload={'model':model,'messages':[{'role':'user','content':'What is the capital of France? Answer with only the city name.'}],'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}}
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response:answer=json.load(response)
    assert answer['model']==model and 'Paris' in answer['choices'][0]['message']['content']
    with syslog.open('rb') as f:f.seek(offset);new=f.read().decode(errors='replace')
    bad=[s for s in new.splitlines() if re.search(r'xe 0000:(03:00\.0|07:00\.0)',s) and re.search(r'\b(?:reset\w*|wedged|fault\w*|error\w*|fail\w*|timedout|lost)\b',s,re.I)]
    assert counts()==before and not bad,bad
    final=name=='GRIMOIRE-DUAL-QWEN3.8'
    results.append({**profile,'healthy':True,'answer':answer['choices'][0]['message']['content'],'running':final})
    print('PROFILE_PASS',json.dumps(results[-1]),flush=True)
    if not final:run('docker','stop','--time','300',name)
(b/'logs/profile-activation1109.json').write_text(json.dumps(results,indent=2)+'\n')
print('SCOPED_PROFILES_PASS current=GRIMOIRE-DUAL-QWEN3.8 port=6897 ctx=131072 MTP=6',flush=True)
