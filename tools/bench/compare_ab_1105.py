#!/usr/bin/env python3
"""Side-by-side llama-benchy rates and correctness checks of dual_engine runs (2026-11-05).
   compare_ab_1105.py LOGDIR LABEL=STEM [LABEL=STEM ...]
The first run is the baseline: the last column is the change of the last run against it."""
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
M = dict(a.split('=', 1) for a in sys.argv[2:])


def rates(stem):
    # llama-benchy prints each tg128 row after the pp row it follows, both labelled with the user count:
    # key a tg row by the last prompt length seen at the same user count
    out, last_pp = {}, {}
    p = root / f'{stem}.benchy.md'
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        parts = [x.strip() for x in line.split('|')]
        if len(parts) > 4:
            m = re.fullmatch(r'(pp|tg)(\d+)(?: \(c(\d+)\))?', parts[2])
            if m:
                c = int(m.group(3) or 1)
                tot = float(parts[3].split()[0])
                if m.group(1) == 'pp':
                    last_pp[c] = int(m.group(2))
                    out[('pp', last_pp[c], c)] = tot
                else:
                    out[('tg', last_pp.get(c, 0), c)] = tot
    return out


R = {k: rates(v) for k, v in M.items()}
names = list(M)
keys = sorted({k for r in R.values() for k in r}, key=lambda k: (k[0] != 'pp', k[1], k[2]))
print('| Test | Users | ' + ' | '.join(names) + ' | change |')
print('|---|---:|' + '---:|' * len(names) + '---:|')
for k in keys:
    vals = [R[n].get(k) for n in names]
    ch = f'{100 * (vals[-1] / vals[0] - 1):+.1f}%' if vals[0] and vals[-1] else '-'
    print(f'| {k[0]}{k[1]} | {k[2]} | ' + ' | '.join(f'{v:,.1f}' if v else '-' for v in vals) + f' | {ch} |')
print()
for n, stem in M.items():
    run = (root / f'{stem}.run').read_text() if (root / f'{stem}.run').exists() else ''
    parts = []
    cp = root / f'{stem}.code.json'
    if cp.exists():
        code = [json.loads(l) for l in cp.read_text().splitlines() if l.strip()]
        by_len = {}
        for r in code:
            by_len.setdefault(r['target'], []).append(r)
        parts.append('coding ' + '; '.join(
            f"{t // 1000}K {sum(1 for r in rs if r['quality']['tests'])}/{len(rs)} "
            f"({max(r['decode_tok_s'] for r in rs):.1f} tok/s)" for t, rs in sorted(by_len.items())))
    qp = root / f'{stem}.conc4.json'
    if qp.exists():
        conc = json.loads(qp.read_text())
        ok = sum(1 for r in conc.values() if r['tests'].get('syntax') and r['tests'].get('tests'))
        parts.append(f'4 users at once {ok}/{len(conc)}')
    sp = root / f'{stem}.seq4.json'
    if sp.exists():
        seq = json.loads(sp.read_text())
        parts.append(f'seq4 {len(seq)} answers')
    paris = "answer: 'Paris" in run or 'answer: "Paris' in run
    print(f"{n}: Paris {'yes' if paris else 'NO'}; coherence {'PASS' if 'Coherence test PASSED' in run else 'FAIL'}; "
          + '; '.join(parts))
