"""Summarize verified 11-08 measurements without mixing historical link speeds."""
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
cases = [
    ('r1108-img-fn', 'Flash-Next, 1 B70, BF16, heat off'),
    ('r1108-img-fn4', 'Flash-Next, 1 B70, INT4, heat + fill'),
    ('r1108-img-q38', 'Qwen 27B, 1 B70, GPTQ + MTP'),
    ('r1108-img-q27-df2', 'Qwen 27B, 2 B70, INT8 + DFlash2'),
    ('r1108-img-fn-tp', 'Flash-Next, 2 B70 TP, INT8'),
    ('r1108-fnpp-340', 'Flash-Next, 2 B70 PP 24/24, INT8'),
    ('r1108-fnpp-340-h8', 'Flash-Next, 2 B70 PP 24/24, INT8 + heat'),
]
print('All rows use rc2 image 717960860fad, own binaries, gpu1 Gen4 x4.\n')
print('| Configuration | Test | tok/s |')
print('|---|---|---:|')
data = {}
for name, label in cases:
    stem = root / name
    if not stem.with_suffix('.VERIFIED').exists():
        continue
    records = []
    for line in stem.with_suffix('.benchy.md').read_text().splitlines():
        cols = [s.strip() for s in line.split('|')]
        if len(cols) < 4 or not re.fullmatch(r'(?:pp\d+|tg128)(?: \(c\d+\))?', cols[2]):
            continue
        print(f'| {label} | {cols[2]} | {cols[3]} |')
        records.append({'test': cols[2], 'rate': cols[3]})
    coding = stem.with_suffix('.code.json')
    data[name] = {'label': label, 'rates': records}
    if coding.exists():
        rows = [json.loads(s) for s in coding.read_text().splitlines() if s.strip()]
        data[name]['code'] = [{'decode_tok_s': r['decode_tok_s'], 'quality': r['quality']} for r in rows]
print('\nCoding checks:')
for name, record in data.items():
    if 'code' in record:
        print(f"- {record['label']}: " + ', '.join(f"{r['decode_tok_s']:.1f} tok/s" for r in record['code']))
(root / 'summary-1108.json').write_text(json.dumps(data, indent=2) + '\n')
