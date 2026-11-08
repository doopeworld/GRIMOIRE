"""Validate each new run before advancing; historical artifacts stay untouched."""
import json
from pathlib import Path
import re
import sys

p = Path(sys.argv[1])
run = p.with_suffix('.run').read_text()
server = p.with_suffix('.server.log').read_text()
assert not p.with_suffix('.STOP').exists(), 'link guard stopped the run'
for needed in ('image grimoire-b70:v1.10.5-rc2, the image\'s own binaries',
               'ready=1', 'Coherence test PASSED', 'server stopped gracefully',
               'uncorrectable(all ports)=0', 'gpu0 path correctable=0'):
    assert needed in run, f'{p.name}: missing {needed}'
assert not re.search(r'\b(?:FAIL|FAILED|STOP|LEFT RUNNING|Traceback)\b', run), p.name
rates = p.with_suffix('.benchy.md').read_text()
assert 'tg128' in rates and 'pp' in rates
if '-q' in p.name:
    rows = [json.loads(line) for line in p.with_suffix('.code.json').read_text().splitlines() if line.strip()]
    assert rows and all(r['quality']['syntax'] and r['quality']['tests'] for r in rows), 'coding test'
    if p.name.endswith('-q38'):
        rows = json.loads(p.with_suffix('.conc4.json').read_text())
        assert len(rows) == 8 and all(r['tests']['syntax'] and r['tests']['tests'] for r in rows.values()), 'concurrent code'
else:
    gen = json.loads(p.with_suffix('.gen.json').read_text())
    assert 'Paris' in gen['capital']['text'], 'capital'
    assert 'F(1) = 1' in gen['code']['text'], 'code'
    if p.name.endswith('-img-fn'):
        old = json.loads((p.parent / 'r1104-img-fn.gen.json').read_text())
        assert gen.keys() == old.keys() and all(gen[k]['text'] == old[k]['text'] for k in gen), 'fixed baseline texts'
    if p.name.endswith('-fn4') or p.name.endswith('-h8'):
        assert 'expert heat engine: on' in server, 'heat engine inactive'
        assert re.search(r'expert heat engine: \d+ steps, [1-9]\d* experts moved', server), 'no heat moves'
    if 'fnpp' in p.name:
        assert server.count('tiered experts: 24 layers x 512') == 2, 'expected two 24-layer stages'
print(f'ARTIFACT_CHECK_PASS {p.name}')
