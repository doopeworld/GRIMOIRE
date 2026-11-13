#!/usr/bin/env python3
"""Table of diag_client1112.py runs (2026-11-13): per run, decode tok/s by category for sampled and greedy requests,
median TTFT, and whether the greedy texts match a reference run's.
Usage: diag_table1113.py LOGDIR REF_TAG TAG [TAG ...]   (TAG = dual1112-<tag> directory under LOGDIR)"""
import json, os, statistics as st, sys

logdir, ref, tags = sys.argv[1], sys.argv[2], sys.argv[3:]


def load(tag):
    p = os.path.join(logdir, 'dual1112-' + tag, 'diag.json')
    return json.load(open(p)) if os.path.exists(p) else None


R = load(ref)
refg = {r['tag']: r['text'] for r in R if not r['sampled']} if R else {}
print(f"{'run':28s} {'samp':>6s} {'code':>6s} {'prose':>6s} {'json':>6s} | {'greedy':>6s} {'code':>6s} {'prose':>6s} {'json':>6s} | ttft  same-greedy-text")
for tag in tags:
    d = load(tag)
    if not d:
        print(f'{tag:28s} (no diag.json)')
        continue
    out = []
    for s in (True, False):
        rows = [r for r in d if r['sampled'] == s]
        by = {}
        for r in rows:
            by.setdefault(r['category'], []).append(r['decode_tps'])
        out.append((st.mean(r['decode_tps'] for r in rows), *(st.mean(by[c]) for c in ('code', 'prose', 'json'))))
    ttft = st.median(r['ttft_ms'] for r in d)
    g = {r['tag']: r['text'] for r in d if not r['sampled']}
    same = sum(1 for k, v in g.items() if refg.get(k) == v)
    print(f'{tag:28s} ' + ' '.join(f'{x:6.1f}' for x in out[0]) + ' | ' + ' '.join(f'{x:6.1f}' for x in out[1]) +
          f' | {ttft:4.0f}  {same}/{len(g)}')
