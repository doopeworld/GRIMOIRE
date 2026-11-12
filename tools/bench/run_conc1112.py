#!/usr/bin/env python3
"""GPU0-only multi-user A/B (2026-11-12 copy of run_conc1111.py: also saves each answer's text).

One B70 (gpu0, 03:00.0), Swift INT4 + MTP K6, one server per variant, 1/2/4/8 simultaneous streaming
requests -> total tok/s, per-request decode tok/s, TTFT (the shape of Ian's friend's table).
Variants: --env KEY=VAL (repeatable), e.g. GRIMOIRE_SPEC_MAX_SEQS=4 = the old one-card default.
--check adds GRIMOIRE_SMALLM_VERIFY=1 (DPAS small-M vs decode GEMV, once per shape; syncs -> no timing).
Safety: gpu1 is never mapped; health guard (xe resets, PCI config, fatal/nonfatal AER) before/while
running; stop = TERM + wait, no forced kill; a failed server is left in place.
"""
import argparse, fcntl, hashlib, json, os, pathlib, re, statistics as st, subprocess, threading, time, urllib.request

BASE = pathlib.Path('/mnt/storage/isos/grimoire-runs/bench-1022')
P = argparse.ArgumentParser()
P.add_argument('--tree', required=True)            # build tree with bin/ and docker-entrypoint.sh
P.add_argument('--tag', required=True)
P.add_argument('--env', action='append', default=[])
P.add_argument('--check', action='store_true')
P.add_argument('--t0', action='store_true')   # Megatron TP, BOTH ranks on gpu0 (gpu1 never mapped)
P.add_argument('--ctx', type=int, default=32768)
P.add_argument('--slots', type=int, default=8)
P.add_argument('--levels', default='1,2,4,8')
P.add_argument('--rounds', type=int, default=2)
P.add_argument('--max-tokens', type=int, default=384)
O = P.parse_args()
assert re.fullmatch(r'[a-z0-9-]+', O.tag)
TREE = pathlib.Path(O.tree)
OUT = BASE / 'logs' / ('conc1112-' + O.tag)
NAME = 's1112-conc-' + O.tag
PORT = 6901
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


# provenance + exclusivity before mapping the GPU
src = (TREE / 'BUILD_SOURCE').read_text().strip()
sha = hashlib.sha256((TREE / 'bin/grimoire-server').read_bytes()).hexdigest()
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

env = {'ONEAPI_DEVICE_SELECTOR': 'level_zero:gpu', 'ZE_AFFINITY_MASK': '0', 'GRIMOIRE_SEQ_SLOTS': str(O.slots),
       'GRIMOIRE_MTP': '1', 'GRIMOIRE_MTP_K': '6', 'GRIMOIRE_MTP_DRAFT_VOCAB': '65536',
       'GRIMOIRE_SAMPLE_COUPLED_MTP': '0', 'GRIMOIRE_SERVER_WARMUP': '0', 'GRIMOIRE_SPEC_STATS': '1'}
for kv in O.env:
    k, v = kv.split('=', 1)
    env[k] = v
if O.check:
    env['GRIMOIRE_SMALLM_VERIFY'] = '1'
if O.t0:
    env.update(GRIMOIRE_MULTI_GPUS='2', GRIMOIRE_MULTI_DEVICES='0,0', GRIMOIRE_P2P='0', GRIMOIRE_TP_AR_BAR='0')
args = ['docker', 'run', '-d', '--name', NAME, '--init', '--network', 'host', '--ipc=private', '--shm-size', '1g',
        '--restart', 'no', '--device', str(node),
        '-v', str(TREE / 'bin') + ':/grimoire/bin:ro',
        '-v', str(TREE / 'docker-entrypoint.sh') + ':/usr/local/bin/docker-entrypoint.sh:ro', '-v', '/mnt/models:/models2:ro']
for k, v in env.items():
    args += ['-e', k + '=' + v]
args += ['grimoire-b70:dev-cancel-1111'] + (['multi', 'TP'] if O.t0 else ['server']) + ['--model', MODEL, '--proj', 'int4', '--ctx', str(O.ctx),
         '--host', '127.0.0.1', '--port', str(PORT)]
command(*args)
meta = {'tag': O.tag, 'source': src, 'server_sha256': sha, 'env': env, 'ctx': O.ctx, 'gpu': '03:00.0 only',
        'layout': 'T0: two TP ranks on gpu0' if O.t0 else 'G0: one card'}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=2))

PROMPTS = [
    'Write Python code for an LRU cache with get and put methods using OrderedDict. Include tests.',
    'Return a JSON array of twelve objects each with id, name, enabled and tags fields. No explanations.',
    'Describe a rainy walk through a coastal town in clear vivid prose, about 300 words.',
    'Write a C function that parses an IPv4 address string into four bytes, with error checks.',
    'Explain step by step how to compute the determinant of a 3x3 matrix, with one worked example.',
    'Write a bash script that backs up a directory into a dated tar.gz and keeps the last seven copies.',
    'Summarise the causes and consequences of the printing press in Europe in two paragraphs.',
    'Write a TypeScript function that debounces another function, with a short usage example.',
]


def one(i, rows, tag):
    body = {'model': 'm', 'stream': True, 'max_tokens': O.max_tokens, 'temperature': 0.7, 'top_p': 0.95, 'top_k': 20,
            'messages': [{'role': 'user', 'content': f'[{tag}-{i}-{time.time_ns()}] ' + PROMPTS[i % len(PROMPTS)]}]}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.monotonic(); first = last = None; usage = {}; parts = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b'data: ') or b'[DONE]' in line:
                continue
            d = json.loads(line[6:])
            if d.get('error'):
                raise RuntimeError(d)
            if d.get('usage'):
                usage = d['usage']
            for c in d.get('choices') or []:
                dl = c.get('delta') or {}
                if (dl.get('content') or '') + (dl.get('reasoning_content') or ''):
                    parts.append((dl.get('reasoning_content') or '') + (dl.get('content') or ''))
                    now = time.monotonic(); first = first or now; last = now
    n = usage.get('completion_tokens', 0)
    rows.append({'i': i, 'start': t0, 'end': last or time.monotonic(), 'ttft_ms': (first - t0) * 1000 if first else None,
                 'tokens': n, 'text': ''.join(parts), 'decode_tps': (n - 1) / (last - first) if first and last and last > first else None})


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
    one(0, [], 'warmup')
    table = []
    for level in [int(x) for x in O.levels.split(',')]:
        for rnd in range(O.rounds):
            rows = []
            th = [threading.Thread(target=one, args=(i, rows, f'c{level}r{rnd}')) for i in range(level)]
            for t in th:
                t.start()
            while any(t.is_alive() for t in th):
                assert healthy(), 'guard during requests: leave server in place'
                time.sleep(1)
            assert len(rows) == level and all(r['tokens'] > 0 for r in rows), rows
            wall = max(r['end'] for r in rows) - min(r['start'] for r in rows)
            rec = {'users': level, 'round': rnd, 'total_tps': sum(r['tokens'] for r in rows) / wall,
                   'each_tps_median': st.median(r['decode_tps'] for r in rows if r['decode_tps']),
                   'ttft_ms_median': st.median(r['ttft_ms'] for r in rows), 'rows': rows}
            table.append(rec)
            (OUT / 'table.json').write_text(json.dumps(table, indent=1))
            print(json.dumps({k: v for k, v in rec.items() if k != 'rows'}), flush=True)
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
    summary = {}
    for rec in table:
        summary.setdefault(rec['users'], []).append(rec)
    out = {u: {'total_tps': st.median(r['total_tps'] for r in rs), 'each_tps': st.median(r['each_tps_median'] for r in rs),
               'ttft_ms': st.median(r['ttft_ms_median'] for r in rs)} for u, rs in summary.items()}
    (OUT / 'result.json').write_text(json.dumps({'status': 'PASS', 'meta': meta, 'summary': out}, indent=2))
    print('CONC_PASS', json.dumps(out), flush=True)
except Exception:
    try:
        (OUT / 'server.log').write_text(command('docker', 'logs', NAME))
    except Exception:
        pass
    (OUT / 'FAILED').write_text('Leave the container in place. No forced kill or automatic recovery.\n')
    raise
