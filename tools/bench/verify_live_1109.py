import datetime,json,re,subprocess,time
from pathlib import Path
b=Path('/mnt/storage/isos/grimoire-runs/bench-1022')
meta=json.loads((b/'logs/profile-update1109.json').read_text())
start=Path(meta['backup']).stat().st_mtime-120
errors=[]
for line in Path('/var/log/syslog').read_text(errors='replace').splitlines():
 if not re.search(r'xe 0000:(03:00\.0|07:00\.0)',line):continue
 if not re.search(r'\b(?:reset\w*|wedged|fault\w*|error\w*|fail\w*|timedout|lost)\b',line,re.I):continue
 fields=line.split(None,3)
 try:stamp=datetime.datetime.strptime(str(datetime.datetime.now().year)+' '+' '.join(fields[:3]),'%Y %b %d %H:%M:%S').timestamp()
 except ValueError:continue
 if stamp>=start:errors.append(line)
assert not errors,errors
counts={}
for addr in ['0000:03:00.0','0000:07:00.0']:
 for part in (Path('/sys/bus/pci/devices')/addr).resolve().parts:
  if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]',part):continue
  for kind in ['nonfatal','fatal']:
   f=Path('/sys/bus/pci/devices')/part/('aer_dev_'+kind)
   if f.exists():
    for line in f.read_text().splitlines():
     if line.startswith('TOTAL_ERR_'):counts[part+'/'+kind]=int(line.split()[1])
assert len(counts)>=4 and all(x==0 for x in counts.values()),counts
states={}
for name in ['GRIMOIRE-FLASH-NEXT','GRIMOIRE-DUAL-FLASH-NEXT','GRIMOIRE-DUAL-QWEN3.8','open-webui']:
 r=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
 states[name]={'running':r['State']['Running'],'image':r['Config']['Image']}
assert states['GRIMOIRE-DUAL-QWEN3.8']['running']
assert not states['GRIMOIRE-FLASH-NEXT']['running'] and not states['GRIMOIRE-DUAL-FLASH-NEXT']['running']
assert not states['open-webui']['running']
result={'states':states,'gpu_path_aer':counts,'new_gpu_errors':errors,'verified_at':time.strftime('%Y-%m-%d %H:%M:%S%z')}
(b/'logs/live-verification1109.json').write_text(json.dumps(result,indent=2)+'\n')
print('LIVE_PROFILES_VERIFIED',json.dumps(result))
