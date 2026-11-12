#!/usr/bin/env python3
"""Flag garbage answers in run_conc1112.py tables (2026-11-12).  The prompts are English, so an answer
whose text has more than 2% CJK / unusual-script characters, or junk tokens the 8-user bug produced
('councill', 'TokenName', 'taboola', 'TRGL'), is flagged.  Usage: check_texts_1112.py DIR [DIR...]"""
import json, pathlib, sys

JUNK = ('councill', 'TokenName', 'taboola', 'TRGL', 'BHY')


def odd(ch):
    o = ord(ch)
    return o >= 0x0590 and not (0x2000 <= o <= 0x2BFF) and not (0x1F300 <= o <= 0x1FAFF)


for d in sys.argv[1:]:
    tab = json.load(open(pathlib.Path(d) / 'table.json'))
    by = {}
    for rec in tab:
        for r in rec['rows']:
            t = r.get('text', '')
            frac = sum(odd(c) for c in t) / max(1, len(t))
            bad = frac > 0.02 or any(j in t for j in JUNK)
            s = by.setdefault(rec['users'], [0, 0, []])
            s[0] += 1
            s[1] += bad
            if bad and len(s[2]) < 2:
                s[2].append(repr(t[:120]))
    print('==', d)
    for u, (n, b, ex) in sorted(by.items()):
        print(f'  {u} users: {b} of {n} answers flagged')
        for e in ex:
            print('     ', e)
