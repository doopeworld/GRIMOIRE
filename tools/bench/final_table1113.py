#!/usr/bin/env python3
"""Collect the 2026-11-13 results into one JSON (numbers straight from the logs, nothing typed by hand).
Run from bench-1022/logs with PYTHONPATH=../betterbench1109 and numpy available (uv run --with numpy).
Usage: final_table1113.py OUT.json"""
import glob, json, os, re, statistics as st, sys

from betterbench import report  # noqa: F401  (sum_bb1111 needs it importable)
sys.path.insert(0, '..')


def bb(tag):
    p = os.path.join(tag, 'betterbench.json')
    if not os.path.exists(p):
        return None
    import subprocess
    out = subprocess.run([sys.executable, '../sum_bb1111.py', p], capture_output=True, text=True).stdout.strip()
    return json.loads(out.splitlines()[-1]) if out else None


def diag(tag):
    p = os.path.join(tag, 'diag.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    r = {}
    for s, name in ((True, 'sampled'), (False, 'greedy')):
        rows = [x for x in d if x['sampled'] == s]
        r[name] = round(st.mean(x['decode_tps'] for x in rows), 1)
        for c in ('code', 'prose', 'json'):
            r[name + '_' + c] = round(st.mean(x['decode_tps'] for x in rows if x['category'] == c), 1)
    r['ttft_ms'] = round(st.median(x['ttft_ms'] for x in d))
    return r


def benchy(tag):
    p = os.path.join(tag, 'benchy.md')
    if not os.path.exists(p):
        return None
    r, depth = {}, None
    for line in open(p):
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 4:
            continue
        m = re.match(r'(pp|tg)(\d+) \(c(\d+)\)', cells[1])
        if not m:
            continue
        kind, n, c = m.group(1), int(m.group(2)), int(m.group(3))
        if kind == 'pp':
            depth = n
        total = float(cells[2].split('±')[0]); req = float(cells[3].split('±')[0])
        key = f'{kind}{n if kind == "pp" else 128}@{depth}k' if kind == 'tg' else f'pp{n}'
        r.setdefault(key, {})[f'c{c}'] = {'total': total, 'per_req': req}
    return r


res = {}
for d in sorted(glob.glob('dual1112-*')):
    if not os.path.isdir(d):
        continue
    tag = d[len('dual1112-'):]
    res[tag] = {k: v for k, v in (('bb', bb(d)), ('diag', diag(d)), ('benchy', benchy(d))) if v}
for v in glob.glob('vllm1112-psu750'):
    res['vllm-tp2'] = {'bb': bb(v)}
json.dump(res, open(sys.argv[1], 'w'), indent=1)
print('wrote', sys.argv[1], len(res), 'runs')
