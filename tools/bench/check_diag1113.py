#!/usr/bin/env python3
"""Flag garbage answers in diag_client1112.py output (2026-11-13), same rule as check_texts_1112.py: more than 2%
CJK / unusual-script characters, or the junk tokens of the 8-user bug.  Usage: check_diag1113.py DIAG.json [...]"""
import json, sys

JUNK = ('councill', 'TokenName', 'taboola', 'TRGL', 'BHY')


def odd(ch):
    o = ord(ch)
    return o >= 0x0590 and not (0x2000 <= o <= 0x2BFF) and not (0x1F300 <= o <= 0x1FAFF)


for p in sys.argv[1:]:
    d = json.load(open(p))
    bad = [r for r in d if sum(odd(c) for c in r['text']) / max(1, len(r['text'])) > 0.02 or any(j in r['text'] for j in JUNK)]
    print(f'{p}: {len(bad)} of {len(d)} answers flagged')
    for r in bad[:2]:
        print('   ', r['tag'], repr(r['text'][:120]))
    g = [r for r in d if not r['sampled'] and r['category'] == 'code'][:1]
    for r in g:
        print('    greedy code sample:', repr(r['text'][-160:]))
