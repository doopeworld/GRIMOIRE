"""Apply only the three scoped, validated Grimoire profiles; preserve WebUI."""
import copy,json,os,shlex,subprocess,time
from pathlib import Path
import xml.etree.ElementTree as ET
os.umask(0o077)
b=Path('/mnt/storage/isos/grimoire-runs/bench-1022')
assert 'CANDIDATE_SCOPE_PASS' in (b/'logs/candidate-validation1109.out').read_text()
proof=json.loads((b/'logs/candidate-image-provenance1109.json').read_text())
image='grimoire-b70:v1.10.5-rc3'
info=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
assert info['Id']==proof['image_id']
root=Path('/boot/config/plugins/dockerMan/templates-user')
q=ET.parse(root/'my-GRIMOIRE-DUAL-QWEN3.8.xml').getroot()
qenv={n.get('Target'):n for n in q.findall('Config')}
backup=b/('profiles-backup1109-'+time.strftime('%Y%m%d-%H%M%S'));backup.mkdir(mode=0o700)
specs=[('GRIMOIRE-FLASH-NEXT','int4',65536,6893,1),('GRIMOIRE-DUAL-FLASH-NEXT','int8',65536,6898,2),('GRIMOIRE-DUAL-QWEN3.8','int8',131072,6897,2)]
prepared=[]
for name,proj,ctx,port,cards in specs:
    path=root/('my-'+name+'.xml');tree=ET.parse(path);r=tree.getroot()
    assert r.findtext('Name')==name and r.findtext('Privileged')=='false'
    args=shlex.split(r.findtext('PostArgs'))
    def value(key):return args[args.index(key)+1]
    assert int(value('--ctx'))==ctx and int(value('--port'))==port
    extra=r.findtext('ExtraParams') or ''
    assert 'pci-0000:03:00.0-render' in extra and '0b:00.0' not in extra
    assert ('pci-0000:07:00.0-render' in extra)==(cards==2)
    env={n.get('Target'):n for n in r.findall('Config')}
    def set_env(key,val):
        if key not in env:
            env[key]=copy.deepcopy(qenv[key]);r.append(env[key])
        env[key].text=val;env[key].set('Default',val)
    if 'FLASH-NEXT' in name:
        assert value('--model')=='/models2/Qwen3.8-Flash-Next-NVFP4'
        assert value('--proj')=='bf16'
        args[args.index('--proj')+1]=proj
        r.find('PostArgs').text=shlex.join(args)
        set_env('GRIMOIRE_SEQ_SLOTS','1');set_env('GRIMOIRE_MTP','0')
        r.find('Overview').text=(f'Qwen3.8-Flash-Next NVFP4 experts with {proj.upper()} projections on {cards} B70 GPU(s). '
            f'65536-token context, one conversation at a time, API port {port}. '
            'Uses the validated native Grimoire candidate; MTP remains disabled for this profile. The B580 is excluded.')
    else:
        assert value('--model')=='/models2/Qwen3.8-27B-Uncensored-FP8' and value('--proj')=='int8'
        for key,val in {'GRIMOIRE_SEQ_SLOTS':'1','GRIMOIRE_MTP':'1','GRIMOIRE_MTP_K':'6','GRIMOIRE_MTP_DRAFT_VOCAB':'65536','GRIMOIRE_DFLASH_MODEL':''}.items():
            assert (env[key].text or '')==val,(name,key)
        # Keep model, MTP settings, context and all WebUI defaults exactly as configured.
    r.find('Repository').text=image
    (backup/path.name).write_bytes(path.read_bytes())
    before=subprocess.run(['docker','inspect',name],capture_output=True)
    if before.returncode==0:(backup/(name+'-container-before.json')).write_bytes(before.stdout)
    ET.indent(tree,space='  ');prepared.append((path,tree,{'name':name,'image':image,'proj':proj,'ctx':ctx,'port':port,'cards':cards}))
# Publish template writes only after every profile passed preconditions.
for path,tree,meta in prepared:
    tree.write(path,encoding='utf-8',xml_declaration=True);print(json.dumps(meta),flush=True)
(b/'logs/profile-update1109.json').write_text(json.dumps({'backup':str(backup),'image_id':proof['image_id'],'profiles':[p[2] for p in prepared]},indent=2)+'\n')
print('SCOPED_PROFILE_TEMPLATES_READY backup='+str(backup),flush=True)
