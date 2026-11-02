#!/usr/bin/env python3
# seq_code_1102.py PORT USERS OUT.json -- conc_code.py's exact prompts, sent ONE AT A TIME, two rounds.
# Compared with conc_code.py's answers it shows whether running together changes an answer, and
# whether a failing answer also fails alone (the model's mistake, not the batch's).
import hashlib, json, sys, time, types, urllib.request
port, users, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
src = open('/mnt/storage/isos/grimoire-runs/bench-1003/coding_probe.py').read().split('\nrows = []')[0]
cp = types.SimpleNamespace()
sys.argv = [sys.argv[0], '--base', 'x', '--model', 'm']
ns = {'__name__': 'coding_probe', '__builtins__': __builtins__}
exec(compile(src, 'coding_probe.py', 'exec'), ns)
cp.TASK, cp.check = ns['TASK'], ns['check']
url = 'http://127.0.0.1:%s/v1/chat/completions' % port


def ask(i):
    context = 'Background note: interval endpoints are integers. Sort before merging.\n' * 150
    prompt = f'User {i} asks.\nReference notes follow.\n' + context + '\nTask:\n' + cp.TASK
    body = {'model': 'm', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
            'max_tokens': 768, 'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        text = r['choices'][0]['message'].get('content') or ''
        return {'text': text, 'tokens': r.get('usage', {}).get('completion_tokens'),
                's': time.time() - t0, 'tests': cp.check(text)}
    except Exception as e:
        return {'text': '', 'tokens': 0, 's': time.time() - t0, 'tests': {'tests': False, 'error': repr(e)[:120]}}


res, bad = {}, 0
for rnd in range(2):
    for i in range(users):
        r = res[f'{rnd}-{i}'] = ask(i)
        ok = r['tests'].get('tests')
        bad += 0 if ok else 1
        print(f"round {rnd} user {i}: {r['tokens']} tok {r['s']:.1f}s tests={'PASS' if ok else 'FAIL ' + str(r['tests'].get('error', r['tests'].get('skipped', '')))} "
              f"md5 {hashlib.md5(r['text'].encode()).hexdigest()[:10]}", flush=True)
json.dump(res, open(out, 'w'))
print(f"SEQ {users} prompts: {2 * users - bad}/{2 * users} answers pass", flush=True)
