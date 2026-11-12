#!/usr/bin/env python3
"""INT4 prompt GEMM as int8 x int8 in the Hadamard-128 domain (GRIMOIRE_I4A8) -- gpu0-only A/B (2026-11-12).

One B70 (gpu0, 03:00.0), Swift INT4 + MTP K6, one server per variant.  Measures
  * prompt speed: ~2K/4K/8K/16K-token prompts, max_tokens 1, 2 reps each (fresh nonce = no cache reuse);
  * answers: 8 real-text prompts (repo docs, ~3-6K tokens) + a question, greedy, 48 tokens -> texts saved,
    compared across variants by compare mode.  The OFF run repeats the set to show run-to-run identity.
Variant: --env KEY=VAL.  Safety as run_conc1111.py: gpu1 never mapped, guard (xe resets, PCI config ffff,
fatal/nonfatal AER), TERM + wait stop, a failed server is left in place.
"""
import argparse, fcntl, hashlib, json, os, pathlib, re, statistics as st, subprocess, time, urllib.request

BASE = pathlib.Path('/mnt/storage/isos/grimoire-runs/bench-1022')
P = argparse.ArgumentParser()
P.add_argument('--tree', required=True)
P.add_argument('--tag', required=True)
P.add_argument('--env', action='append', default=[])
P.add_argument('--ctx', type=int, default=32768)
P.add_argument('--repeat-answers', action='store_true')
P.add_argument('--docs', default='/mnt/storage/isos/grimoire-runs/fast-1111b')
O = P.parse_args()
assert re.fullmatch(r'[a-z0-9-]+', O.tag)
TREE = pathlib.Path(O.tree)
OUT = BASE / 'logs' / ('i4a8-1112-' + O.tag)
NAME = 's1112-i4a8-' + O.tag
PORT = 6902
MODEL = '/models2/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8'
OUT.mkdir(exist_ok=False)


def command(*a):
    return subprocess.check_output(a, text=True, stderr=subprocess.STDOUT).strip()


BAD = re.compile(r'xe 0000:(?:03|07):00.0.*(?:engine reset|timedout|wedged|unreliable|force.?wake.*fail|fault|error)', re.I)
kernel0 = set(command('dmesg').splitlines())


def aer():
    t = {}
    for card in ('0000:03:00.0', '0000:07:00.0'):
        ep = pathlib.Path('/sys/bus/pci/devices') / card
        for dev in (ep.resolve(), *ep.resolve().parents):
            if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', dev.name):
                continue
            for kind in ('fatal', 'nonfatal'):
                f = dev / ('aer_dev_' + kind)
                if f.exists():
                    for line in f.read_text().splitlines():
                        if line.startswith('TOTAL_ERR_'):
                            t[str(f)] = int(line.split()[-1])
    return t


aer0 = aer()
assert all(v == 0 for v in aer0.values()), 'existing fatal/nonfatal AER: do not start'


def healthy():
    if aer() != aer0:
        return False
    fresh = [x for x in command('dmesg').splitlines() if x not in kernel0]
    (OUT / 'kernel-new.log').write_text('\n'.join(fresh) + '\n')
    if any(BAD.search(x) for x in fresh):
        return False
    try:
        for pci in ('0000:03:00.0', '0000:07:00.0'):
            with open('/sys/bus/pci/devices/' + pci + '/config', 'rb', buffering=0) as f:
                if f.read(4) in (b'\xff' * 4, b''):
                    return False
    except OSError:
        return False
    return True


src = (TREE / 'BUILD_SOURCE').read_text().strip()
sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TREE / 'bin').glob('*')) if p.is_file()}
lock = open('/mnt/user/isos/grimoire-runs/gpu0.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not command('docker', 'ps', '-q'), 'another container is running (ComfyUI?): do not start'
assert not (BASE / 'USER_HOLD_B70').exists(), 'USER_HOLD_B70 present: Ian has not released the GPUs'
node = pathlib.Path('/dev/dri/by-path/pci-0000:03:00.0-render').resolve(strict=True)
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        for fd in (proc / 'fd').iterdir():
            try:
                if os.readlink(fd) == str(node):
                    raise RuntimeError('gpu0 held by ' + str(proc))
            except FileNotFoundError:
                pass
    except (FileNotFoundError, PermissionError):
        pass
assert healthy()

env = {'ONEAPI_DEVICE_SELECTOR': 'level_zero:gpu', 'ZE_AFFINITY_MASK': '0', 'GRIMOIRE_SEQ_SLOTS': '8',
       'GRIMOIRE_MTP': '1', 'GRIMOIRE_MTP_K': '6', 'GRIMOIRE_MTP_DRAFT_VOCAB': '65536',
       'GRIMOIRE_SAMPLE_COUPLED_MTP': '0', 'GRIMOIRE_SERVER_WARMUP': '0', 'GRIMOIRE_SPEC_STATS': '1'}
for kv in O.env:
    k, v = kv.split('=', 1)
    env[k] = v
args = ['docker', 'run', '-d', '--name', NAME, '--init', '--network', 'host', '--ipc=private', '--shm-size', '1g',
        '--restart', 'no', '--device', str(node),
        '-v', str(TREE / 'bin') + ':/grimoire/bin:ro',
        '-v', str(TREE / 'docker-entrypoint.sh') + ':/usr/local/bin/docker-entrypoint.sh:ro', '-v', '/mnt/models:/models2:ro']
for k, v in env.items():
    args += ['-e', k + '=' + v]
args += ['grimoire-b70:dev-cancel-1111', 'server', '--model', MODEL, '--proj', 'int4', '--ctx', str(O.ctx),
         '--host', '127.0.0.1', '--port', str(PORT)]
command(*args)
meta = {'tag': O.tag, 'source': src, 'sha256': sha, 'env': env, 'ctx': O.ctx, 'gpu': '03:00.0 only'}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=2))

# real text: the repo's own docs, in a fixed order
DOCS = sorted(p for p in pathlib.Path(O.docs).glob('*.md') if p.stat().st_size > 20000)[:12]
TEXT = '\n\n'.join(p.read_text(errors='replace') for p in DOCS)
assert len(TEXT) > 200000, len(TEXT)


def chat(content, max_tokens, temperature=0.0):
    body = {'model': 'm', 'stream': False, 'max_tokens': max_tokens, 'temperature': temperature,
            'messages': [{'role': 'user', 'content': content}]}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    dt = time.monotonic() - t0
    msg = d['choices'][0]['message']
    return dt, d.get('usage', {}), (msg.get('content') or '') + '|' + (msg.get('reasoning_content') or '')


def guarded(fn, *a, **k):
    # requests are synchronous; the guard runs between them (a request is at most a few seconds of GPU)
    assert healthy(), 'guard: leave server in place'
    r = fn(*a, **k)
    assert healthy(), 'guard: leave server in place'
    return r


try:
    t_end = time.monotonic() + 300
    while True:
        assert healthy(), 'guard during load: leave container in place'
        assert json.loads(command('docker', 'inspect', NAME))[0]['State']['Running'], 'server exited during load'
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        assert time.monotonic() < t_end, 'load deadline: leave in place'
        time.sleep(1)
    guarded(chat, 'nonce warm ' + TEXT[:2000] + '\nSummarise in one line.', 8)   # first prompt GEMMs + checks
    res = {'pp': [], 'answers': []}
    for chars in (8000, 16000, 32000, 64000):
        for rep in range(2):
            start = (chars * 7 + rep * 3331) % (len(TEXT) - chars)
            prompt = f'[{O.tag} {time.time_ns()}] ' + TEXT[start:start + chars] + '\n\nWhat is this text about?'
            dt, usage, _ = guarded(chat, prompt, 1)
            rec = {'chars': chars, 'rep': rep, 'prompt_tokens': usage.get('prompt_tokens'), 'seconds': dt,
                   'pp_tps': (usage.get('prompt_tokens') or 0) / dt}
            res['pp'].append(rec)
            print(json.dumps(rec), flush=True)
            (OUT / 'result.json').write_text(json.dumps(res, indent=1))
    QUESTIONS = ['Summarise the text above in five bullet points.',
                 'What is the single most important rule stated in the text above? Quote it.',
                 'List three numbers that appear in the text above and what each one measures.',
                 'Write one Python function that the text above would need, with a docstring.']
    sets = 2 if O.repeat_answers else 1
    for s in range(sets):
        for i in range(8):
            chars = 12000 + 2500 * i
            start = (i * 104729) % (len(TEXT) - chars)
            prompt = TEXT[start:start + chars] + '\n\n' + QUESTIONS[i % 4]
            dt, usage, text = guarded(chat, prompt, 48)
            rec = {'set': s, 'i': i, 'prompt_tokens': usage.get('prompt_tokens'), 'text': text}
            res['answers'].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != 'text'}), flush=True)
            (OUT / 'result.json').write_text(json.dumps(res, indent=1))
    (OUT / 'server.log').write_text(command('docker', 'logs', NAME))
    assert healthy()
    command('docker', 'kill', '--signal=TERM', NAME)
    t_end = time.monotonic() + 60
    while json.loads(command('docker', 'inspect', NAME))[0]['State']['Running']:
        assert healthy() and time.monotonic() < t_end, 'shutdown: leave in place, no forced kill'
        time.sleep(0.25)
    state = json.loads(command('docker', 'inspect', NAME))[0]['State']
    assert state['ExitCode'] == 0, state
    (OUT / 'server.log').write_text(command('docker', 'logs', NAME))
    command('docker', 'rm', NAME)   # Ian 11-11: no leftover test containers on the Tower
    by = {}
    for r in res['pp']:
        by.setdefault(r['chars'], []).append(r)
    res['summary'] = {str(c): {'prompt_tokens': st.median(x['prompt_tokens'] for x in rs),
                               'pp_tps': max(x['pp_tps'] for x in rs)} for c, rs in by.items()}
    res['status'] = 'PASS'
    (OUT / 'result.json').write_text(json.dumps(res, indent=1))
    print('I4A8_PASS', json.dumps(res['summary']), flush=True)
except Exception:
    try:
        (OUT / 'server.log').write_text(command('docker', 'logs', NAME))
    except Exception:
        pass
    (OUT / 'FAILED').write_text('Leave the container in place. No forced kill or automatic recovery.\n')
    raise
