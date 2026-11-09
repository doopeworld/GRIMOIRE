"""Read a stopped container to prove packaged executables match this build."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
b=Path('/mnt/storage/isos/grimoire-runs/bench-1022');x=Path('/mnt/storage/isos/grimoire-release-1109')
image='grimoire-b70:v1.10.5-rc3';name='grimoire-provenance1109'
info=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
source=(x/'SOURCE_REVISION').read_text().strip()
assert info['Config']['Labels']['org.opencontainers.image.revision']==source
subprocess.run(['docker','create','--name',name,image],check=True,stdout=subprocess.DEVNULL)
try:
    with tempfile.TemporaryDirectory(prefix='grimoire-provenance1109-') as d:
        root=Path(d)
        subprocess.run(['docker','cp',name+':/usr/share/doc/grimoire/build-manifest.json',str(root/'manifest.json')],check=True)
        manifest=json.loads((root/'manifest.json').read_text());assert manifest['source_commit']==source
        for exe in ['grimoire','grimoire-server']:
            dest=root/exe
            subprocess.run(['docker','cp',name+':/grimoire/bin/'+exe,str(dest)],check=True)
            digest=hashlib.sha256(dest.read_bytes()).hexdigest()
            assert digest==manifest['rebuilt'][exe]==hashlib.sha256((x/'bin'/exe).read_bytes()).hexdigest()
        result={'image':image,'image_id':info['Id'],'source_commit':source,'executables':manifest['rebuilt'],'targets':manifest['targets']}
        (b/'logs/candidate-image-provenance1109.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PACKAGED_IMAGE_PROVENANCE_PASS',json.dumps(result))
finally:
    subprocess.run(['docker','rm',name],check=True,stdout=subprocess.DEVNULL)
