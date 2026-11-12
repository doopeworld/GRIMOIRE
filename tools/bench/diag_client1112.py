#!/usr/bin/env python3
"""Single-user step diagnostic client (2026-11-12): sequential streaming requests, sampled (temp 0.7 / top-p 0.95 /
top-k 20, the BetterBench settings) then greedy, on code / prose / json prompts.  Per request: TTFT, decode tok/s
(first to last token), tokens, update gaps.  Server side: GRIMOIRE_SPEC_TIMING / GRIMOIRE_SPEC_STATS lines.
Usage: diag_client1112.py ENDPOINT MODEL OUT.json"""
import json, statistics as st, sys, time, urllib.request

EP, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPTS = {
    'code': 'Write a Python class implementing an LRU cache with get and put in O(1), using a dict and a doubly '
            'linked list, with type hints and three unit tests.',
    'prose': 'Write a vivid 400-word short story about a lighthouse keeper who finds a message in a bottle.',
    'json': 'Return a JSON array of 15 objects, each with fields id (int), name (string), email (string), '
            'active (bool) and tags (array of 2 strings). Output only the JSON.',
}


def one(prompt, sampled, tag):
    body = {'model': MODEL, 'stream': True, 'max_tokens': 512,
            'messages': [{'role': 'user', 'content': f'[{tag} {time.time_ns()}] ' + prompt}]}
    if sampled:
        body.update(temperature=0.7, top_p=0.95, top_k=20)
    else:
        body.update(temperature=0)
    req = urllib.request.Request(EP + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.monotonic(); first = last = None; usage = {}; stamps = []; parts = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b'data: ') or b'[DONE]' in line:
                continue
            d = json.loads(line[6:])
            if d.get('usage'):
                usage = d['usage']
            for c in d.get('choices') or []:
                dl = c.get('delta') or {}
                if (dl.get('content') or '') + (dl.get('reasoning_content') or ''):
                    now = time.monotonic(); first = first or now; last = now; stamps.append(now)
                    parts.append((dl.get('reasoning_content') or '') + (dl.get('content') or ''))
    n = usage.get('completion_tokens', 0)
    gaps = [1000 * (b - a) for a, b in zip(stamps, stamps[1:])]
    return {'tag': tag, 'sampled': sampled, 'tokens': n, 'text': ''.join(parts), 'ttft_ms': 1000 * (first - t0) if first else None,
            'decode_tps': (n - 1) / (last - first) if first and last and last > first else None,
            'gap_p50_ms': st.median(gaps) if gaps else None,
            'gap_p99_ms': sorted(gaps)[int(0.99 * (len(gaps) - 1))] if gaps else None}


rows = []
for sampled in (True, False):
    for cat, prompt in PROMPTS.items():
        for rep in range(3):
            r = one(prompt, sampled, f'{cat}-{rep}')
            r['category'] = cat
            rows.append(r)
            print(json.dumps(r), flush=True)
            json.dump(rows, open(OUT, 'w'), indent=1)
print('DIAG_CLIENT_DONE', flush=True)
