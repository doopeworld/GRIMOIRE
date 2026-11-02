#!/usr/bin/env python3
# orn_compare_1102.py LOGDIR TAG... : the same coding prompts together and alone, answer by answer.
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
for tag in sys.argv[2:]:
    c, s = root / f'{tag}.conc4.json', root / f'{tag}.seq4.json'
    if not (c.exists() and s.exists()):
        print(tag, 'not finished')
        continue
    C, S = json.loads(c.read_text()), json.loads(s.read_text())
    sample = C[sorted(C)[0]]
    print(f'== {tag}; fields: {sorted(sample)}')

    def toks(r):
        for key in ('tokens', 'completion_tokens', 'n'):
            if key in r:
                return r[key]
        return len(r.get('text', ''))

    def sorted_call(text):
        return 'sort' in text

    print('| Answer | Together | Alone | Same text | Length together | Length alone | Mentions sort, together / alone |')
    print('|---|---|---|---|---:|---:|---|')
    for k in sorted(C):
        a, b = C[k], S.get(k, {})
        ok = lambda r: 'pass' if r.get('tests', {}).get('syntax') and r.get('tests', {}).get('tests') else 'FAIL'
        print(f"| {k} | {ok(a)} | {ok(b)} | {a['text'] == b.get('text')} | {toks(a)} | {toks(b)} "
              f"| {sorted_call(a['text'])} / {sorted_call(b.get('text', ''))} |")
    n = len([k for k in C if k.startswith('0-')])
    print('round 0 = round 1: together',
          sum(C[f'0-{i}']['text'] == C[f'1-{i}']['text'] for i in range(n)), f'/{n}; alone',
          sum(S[f'0-{i}']['text'] == S[f'1-{i}']['text'] for i in range(n)), f'/{n}')
    fails = [k for k in sorted(S) if not (S[k].get('tests', {}).get('syntax') and S[k].get('tests', {}).get('tests'))]
    if fails:
        print(f'--- first failing answer alone ({fails[0]}):')
        print(S[fails[0]]['text'][:900])
