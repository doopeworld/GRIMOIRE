#!/usr/bin/env python3
"""One table from the two runs of run_mxfp4_vs_swift_1104.sh: speed side by side, then every correctness check."""
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
M = {'MXFP4 (GRIMOIRE from BF16)': 'ms1104-mxfp4', 'Swift GPTQ-Int4 (Wrapzii)': 'ms1104-swift'}


def rates(stem):
    # llama-benchy prints each tg128 row after the pp row it follows, both labelled with the user count:
    # key a tg row by the last prompt length seen at the same user count
    out, last_pp = {}, {}
    for line in (root / f'{stem}.benchy.md').read_text().splitlines():
        parts = [p.strip() for p in line.split('|')]
        if len(parts) > 4:
            m = re.fullmatch(r'(pp|tg)(\d+)(?: \(c(\d+)\))?', parts[2])
            if m:
                c = int(m.group(3) or 1)
                tot = float(parts[3].split()[0])      # total throughput
                per = float(parts[4].split()[0]) if parts[4] and m.group(3) else tot   # per request
                if m.group(1) == 'pp':
                    last_pp[c] = int(m.group(2))
                    out[('pp', last_pp[c], c)] = (tot, per)
                else:
                    out[('tg', last_pp.get(c, 0), c)] = (tot, per)
    return out


R = {k: rates(v) for k, v in M.items()}
names = list(M)
pps = sorted({k[1] for k in R[names[0]] if k[0] == 'pp'})
cs = sorted({k[2] for k in R[names[0]]})
print('| Prompt | Users | Prompt tok/s, MXFP4 | Prompt tok/s, Swift | Decode tok/s total, MXFP4 | Decode tok/s total, Swift |')
print('|---:|---:|---:|---:|---:|---:|')
for p in pps:
    for c in cs:
        row = [f'{p:,}', str(c)]
        for kind in ('pp', 'tg'):
            for n in names:
                v = R[n].get((kind, p, c))
                row.append(f'{v[0]:,.1f}' if v else '-')
        print('| ' + ' | '.join(row) + ' |')
print()
for n, stem in M.items():
    run = (root / f'{stem}.run').read_text()
    code = [json.loads(l) for l in (root / f'{stem}.code.json').read_text().splitlines() if l.strip()]
    conc = json.loads((root / f'{stem}.conc4.json').read_text())
    by_len = {}
    for r in code:
        by_len.setdefault(r['target'], []).append(r)
    parts = [f"{t // 1000}K {sum(1 for r in rs if r['quality']['tests'])}/{len(rs)} "
             f"({max(r['decode_tok_s'] for r in rs):.1f} tok/s)" for t, rs in sorted(by_len.items())]
    ok_conc = sum(1 for r in conc.values() if r['tests'].get('syntax') and r['tests'].get('tests'))
    print(f"{n}: Paris {'yes' if 'answer: ' + chr(39) + 'Paris' in run else 'NO'}; "
          f"coherence {'PASS' if 'Coherence test PASSED' in run else 'FAIL'}; "
          f"coding by prompt length {'; '.join(parts)}; 4 users at once {ok_conc}/{len(conc)}")
