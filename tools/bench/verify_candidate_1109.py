"""Validate packaged-image artifacts before progressing to another case."""
import json,re,sys
from pathlib import Path
p=Path(sys.argv[1]);run=p.with_suffix('.run').read_text();server=p.with_suffix('.server.log').read_text()
for text in ["image grimoire-b70:v1.10.5-rc3, the image's own binaries",'ready=1','server stopped gracefully','uncorrectable(all ports)=0','gpu0 path correctable=0']:
    assert text in run,(p.name,text)
assert not p.with_suffix('.STOP').exists(),p.name
assert not re.search(r'\b(?:FAIL|FAILED|STOP|LEFT RUNNING|Traceback)\b',run),p.name
g=json.loads(p.with_suffix('.gen.json').read_text())
assert len(g)==3 and all(x['tokens']>0 for x in g.values())
assert 'Paris' in g['capital']['text']
rates=p.with_suffix('.benchy.md').read_text()
assert 'tg128' in rates and 'Coherence test PASSED' in run
if '-orca-' in p.name or '-host-' in p.name:
    rows=[json.loads(x) for x in p.with_suffix('.code.json').read_text().splitlines() if x.strip()]
    assert len(rows)>=2 and all(x['quality']['syntax'] and x['quality']['tests'] for x in rows)
if '-orca-' in p.name:
    assert 'ORCA_API_CHECKS_PASS' in run
    assert server.count('KV cache: 1 slot x --ctx 131072, 2.00 GiB on this card')==2
if '-host-' in p.name:
    assert server.count('DFlash taps: 12.50 GiB pinned system RAM per rank')==2
    assert server.count('distributed drafter serves one request at a time -> 1 slot (requested 8)')==2
    assert server.count('KV cache: 1 slot x --ctx 131072, 2.00 GiB on this card')==2
    assert any(x['prompt_tokens']>=128000 for x in rows)
    assert 'TP P2P: host fallback: peer IPC import unsupported by driver/topology' in server
if '-fn-tp-' in p.name:
    assert 'TP P2P: host fallback: peer IPC import unsupported by driver/topology' in server
if '-fn-single' in p.name or '-fn-imx' in p.name:
    assert 'expert heat engine: on' in server
    assert re.search(r'expert heat engine: \d+ steps, [1-9]\d* experts moved',server)
if '-fn-imx' in p.name:
    assert re.search(r'imatrix: [1-9][0-9]* INT4 weights rounded',server)
print('CANDIDATE_CASE_PASS',p.name)
