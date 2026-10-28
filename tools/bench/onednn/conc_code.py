#!/usr/bin/env python3
# conc_code.py PORT USERS OUT.json -- USERS copies of coding_probe.py's merge_intervals task sent at
# once (greedy, thinking off), each with its own first line so the prompts differ; every answer
# must pass the probe's unit tests.  Two rounds.  Prints one line per answer (tokens, seconds,
# tests, md5 of the text) and writes the texts to OUT.json for comparing two server runs.
import hashlib, json, sys, threading, time, types, urllib.request
port, users, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# coding_probe.py's TASK and check(): its source up to the main loop (its argparse needs --base / --model)
src = open('/mnt/storage/isos/grimoire-runs/bench-1003/coding_probe.py').read().split('\nrows = []')[0]
cp = types.SimpleNamespace()
sys.argv = [sys.argv[0], '--base', 'x', '--model', 'm']
ns = {'__name__': 'coding_probe', '__builtins__': __builtins__}
exec(compile(src, 'coding_probe.py', 'exec'), ns)
cp.TASK, cp.check = ns['TASK'], ns['check']
url = 'http://127.0.0.1:%s/v1/chat/completions' % port


def ask(i, rnd, res):
    context = 'Background note: interval endpoints are integers. Sort before merging.\n' * 150
    prompt = f'User {i} asks.\nReference notes follow.\n' + context + '\nTask:\n' + cp.TASK
    body = {'model': 'm', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
            'max_tokens': 768, 'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1200))
        text = r['choices'][0]['message'].get('content') or ''
        res[(rnd, i)] = {'text': text, 'tokens': r.get('usage', {}).get('completion_tokens'),
                         's': time.time() - t0, 'tests': cp.check(text)}
    except Exception as e:
        res[(rnd, i)] = {'text': '', 'tokens': 0, 's': time.time() - t0, 'tests': {'tests': False, 'error': repr(e)[:120]}}


res, bad = {}, 0
for rnd in range(2):
    th = [threading.Thread(target=ask, args=(i, rnd, res)) for i in range(users)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    toks = sum(res[(rnd, i)]['tokens'] or 0 for i in range(users))
    for i in range(users):
        r = res[(rnd, i)]
        ok = r['tests'].get('tests')
        bad += 0 if ok else 1
        print(f"round {rnd} user {i}: {r['tokens']} tok {r['s']:.1f}s tests={'PASS' if ok else 'FAIL ' + str(r['tests'].get('error', r['tests'].get('skipped', '')))} "
              f"md5 {hashlib.md5(r['text'].encode()).hexdigest()[:10]}", flush=True)
    print(f"round {rnd}: {users} users, {toks} tokens in {wall:.1f}s = {toks / wall:.1f} tok/s total", flush=True)
json.dump({f'{k[0]}-{k[1]}': v for k, v in res.items()}, open(out, 'w'))
print(f"CONC {users} users: {2 * users - bad}/{2 * users} answers pass", flush=True)
