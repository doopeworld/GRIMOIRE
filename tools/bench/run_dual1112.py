#!/usr/bin/env python3
"""Grimoire on BOTH B70s, against vLLM TP2's BetterBench (same client + betterbench-vllm1111.json), 2026-11-12.
Ian 11-12: "gpu 1 is not dropping anymore so maybe could try dual gpu to measure against vllm and also use stacker
container to do like my friend use same model on 2 gpu".
  --mode tp       one server, "multi TP" over gpu0 + gpu1 (BAR receive stays off, copy-engine receive)
  --mode replica  one one-card server per GPU (gpu0 -> :3551, gpu1 -> :3552) behind Ian's lmstack-router (:3557)
  --client betterbench | benchy (llama-benchy 0.4.0, the friend's tool)
Safety as run_conc1111.py: gpu1 never mapped; guard (xe resets, PCI config ffff, fatal/nonfatal AER) before
and during; a guard trip stops only the CPU client, the server is left in place; stop = TERM + wait."""
import argparse, fcntl, hashlib, json, os, pathlib, re, signal, subprocess, time, urllib.request

B = pathlib.Path('/mnt/storage/isos/grimoire-runs/bench-1022')
P = argparse.ArgumentParser()
P.add_argument('--tree', required=True)
P.add_argument('--tag', required=True)
P.add_argument('--env', action='append', default=[])
P.add_argument('--ctx', type=int, default=32768)
P.add_argument('--config', default=str(B / 'betterbench-vllm1111.json'))
P.add_argument('--mode', choices=['tp', 'replica', 'one'], required=True)   # one = gpu0 only
P.add_argument('--client', choices=['betterbench', 'benchy', 'diag'], default='betterbench')
P.add_argument('--bb-args', default='')   # e.g. '--prefill' = BetterBench's prompt phase only
O = P.parse_args()
assert re.fullmatch(r'[a-z0-9-]+', O.tag)
TREE = pathlib.Path(O.tree)
OUT = B / 'logs' / ('dual1112-' + O.tag)
NAMES = (['s1112-dual-' + O.tag] if O.mode == 'tp' else ['s1112-one-' + O.tag] if O.mode == 'one'
         else ['s1112-rep0-' + O.tag, 's1112-rep1-' + O.tag])
PORTS = [6904] if O.mode == 'tp' else [6905] if O.mode == 'one' else [3551, 3552]
PORT = 3557 if O.mode == 'replica' else PORTS[0]
ROUTER = 'lmstack-router'
MODEL = '/models2/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8'
OUT.mkdir(exist_ok=False)


def sh(*a):
    return subprocess.check_output(a, text=True, stderr=subprocess.STDOUT).strip()


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


BAD = re.compile(r'xe 0000:(?:03|07):00.0.*(?:engine reset|timedout|wedged|unreliable|force.?wake.*fail|fault|error)', re.I)
kernel0 = set(sh('dmesg').splitlines())


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
    fresh = [x for x in sh('dmesg').splitlines() if x not in kernel0]
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


def running():
    return all(json.loads(sh('docker', 'inspect', n))[0]['State']['Running'] for n in NAMES)


src = (TREE / 'BUILD_SOURCE').read_text().strip()
sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TREE / 'bin').glob('*')) if p.is_file()}
lock = open('/mnt/user/isos/grimoire-runs/gpu0.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not sh('docker', 'ps', '-q'), 'another container is running (ComfyUI?): do not start'
assert not (B / 'USER_HOLD_B70').exists(), 'USER_HOLD_B70 present'
nodes = [str(pathlib.Path('/dev/dri/by-path/pci-0000:%s-render' % c).resolve(strict=True)) for c in ('03:00.0', '07:00.0')]
assert nodes[0] != nodes[1]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        for fd in (proc / 'fd').iterdir():
            try:
                if os.readlink(fd) in nodes:
                    raise RuntimeError('a B70 is held by ' + str(proc))
            except FileNotFoundError:
                pass
    except (FileNotFoundError, PermissionError):
        pass
lock1 = open('/mnt/user/isos/grimoire-runs/gpu1.lock', 'w')
fcntl.flock(lock1, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not json.loads(sh('docker', 'inspect', ROUTER))[0]['State']['Running'], 'lmstack-router already running'
assert healthy()

env = {'ONEAPI_DEVICE_SELECTOR': 'level_zero:gpu', 'ZE_AFFINITY_MASK': '0', 'GRIMOIRE_SEQ_SLOTS': '8',
       'GRIMOIRE_MTP': '1', 'GRIMOIRE_MTP_K': '6', 'GRIMOIRE_MTP_DRAFT_VOCAB': '65536',
       'GRIMOIRE_SAMPLE_COUPLED_MTP': '0', 'GRIMOIRE_SPEC_STATS': '1', 'GRIMOIRE_POWER_LIMIT_W': '0'}
for kv in O.env:
    k, v = kv.split('=', 1)
    env[k] = v
def start(name, devs, entry, port, aff, extra):
    a = ['docker', 'run', '-d', '--name', name, '--init', '--network', 'host', '--ipc=host',
         '--restart', 'no']
    for d in devs:
        a += ['--device', d]
    a += ['-v', str(TREE / 'bin') + ':/grimoire/bin:ro',
          '-v', str(TREE / 'docker-entrypoint.sh') + ':/usr/local/bin/docker-entrypoint.sh:ro', '-v', '/mnt/models:/models2:ro']
    e = dict(env, ZE_AFFINITY_MASK=aff, **extra)
    for k, v in e.items():
        a += ['-e', k + '=' + v]
    a += ['grimoire-b70:dev-cancel-1111'] + entry + ['--model', MODEL, '--proj', 'int4', '--ctx', str(O.ctx),
          '--host', '127.0.0.1', '--port', str(port)]
    sh(*a)


if O.mode == 'tp':
    start(NAMES[0], nodes, ['multi', 'TP'], PORTS[0], '0,1', {'GRIMOIRE_MULTI_GPUS': '2'})
elif O.mode == 'one':
    start(NAMES[0], [nodes[0]], ['server'], PORTS[0], '0', {})   # gpu1 never mapped
else:
    start(NAMES[0], [nodes[0]], ['server'], PORTS[0], '0', {})
    start(NAMES[1], [nodes[1]], ['server'], PORTS[1], '0', {})
meta = {'tag': O.tag, 'mode': O.mode, 'client': O.client, 'source': src, 'sha256': sha, 'env': env, 'ctx': O.ctx,
        'gpus': '03:00.0 only' if O.mode == 'one' else '03:00.0 + 07:00.0', 'config': O.config}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=2))
log('servers started', NAMES)
try:
    def wait_health(port, what, t_end):
        while True:
            assert healthy(), 'guard during load: servers left in place'
            assert running(), 'a server exited during load'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            assert time.monotonic() < t_end, what + ' load deadline: servers left in place'
            time.sleep(2)
    t_end = time.monotonic() + 900
    for p in PORTS:
        wait_health(p, 'backend %d' % p, t_end)
    if O.mode == 'replica':
        sh('docker', 'start', ROUTER)
        log('router started')
        t_r = time.monotonic() + 120
        while True:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models', timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            assert time.monotonic() < t_r, 'router did not come up'
            time.sleep(2)
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models', timeout=10) as r:
        model_id = json.load(r)['data'][0]['id']
    log('ready, model id', model_id)
    body = {'model': model_id, 'max_tokens': 400, 'temperature': 0,
            'messages': [{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}]}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        sanity = json.load(r)
    (OUT / 'sanity.json').write_text(json.dumps(sanity, indent=1))
    text = (sanity['choices'][0]['message'].get('content') or '') + (sanity['choices'][0]['message'].get('reasoning_content') or '')
    assert 'Paris' in text, 'sanity answer has no Paris'
    if O.client == 'betterbench':
        cmd = ['taskset', '-c', '14,15', 'uv', 'run', '--offline', '--no-project', '--with', 'numpy',
               'python', '-u', str(B / 'observe_betterbench1109.py'), 'run',
               '--endpoint', f'http://127.0.0.1:{PORT}/v1', '--model', model_id,
               '--config', O.config, '--max-model-len', str(O.ctx),
               '--out', str(OUT / 'betterbench.json'), '--no-update-check'] + O.bb_args.split()
        book = None
    elif O.client == 'diag':
        cmd = ['taskset', '-c', '14,15', 'python3', '-u', str(B / 'diag_client1112.py'),
               f'http://127.0.0.1:{PORT}/v1', model_id, str(OUT / 'diag.json')]
        book = None
    else:
        cmd = ['taskset', '-c', '14,15', 'uvx', 'llama-benchy@0.4.0',
               '--base-url', f'http://127.0.0.1:{PORT}/v1', '--model', model_id,
               '--tokenizer', '/mnt/models/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8',
               '--book-url', 'http://127.0.0.1:8999/1661-0.txt',
               '--pp', '4096', '16384', '--tg', '128', '--concurrency', '1', '2', '4', '8',
               '--runs', '3', '--no-cache', '--format', 'md', '--save-result', str(OUT / 'benchy.md')]
        book = subprocess.Popen(['python3', '-m', 'http.server', '8999', '--bind', '127.0.0.1'],
                                cwd='/mnt/storage/isos/grimoire-runs/bench-1003/book',
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    cenv = dict(os.environ, BB_PROGRESS=str(OUT / 'progress.jsonl'), UV_CACHE_DIR='/mnt/storage/isos/grimoire-cache/uv',
                PYTHONPATH=str(B / 'betterbench1109'))
    try:
        with (OUT / 'benchmark.out').open('w') as bl:
            client = subprocess.Popen(cmd, cwd=B, env=cenv, stdout=bl, stderr=subprocess.STDOUT, start_new_session=True)
            log('client started', O.client, client.pid)
            deadline = time.monotonic() + 9000
            while client.poll() is None:
                if not healthy() or not running() or time.monotonic() > deadline:
                    os.killpg(client.pid, signal.SIGTERM)   # the CPU HTTP client only
                    raise RuntimeError('benchmark client stopped by guard/exit/deadline; servers left in place')
                time.sleep(2)
        assert client.returncode == 0, 'benchmark client failed: see benchmark.out'
    finally:
        if book:
            os.killpg(book.pid, signal.SIGTERM)
    log('client done')
    if O.mode == 'replica':
        (OUT / 'router.log').write_text(sh('docker', 'logs', ROUTER))
        sh('docker', 'stop', ROUTER)          # CPU only: Ian's router back to stopped
    for n in NAMES:
        (OUT / (n + '.log')).write_text(sh('docker', 'logs', n))
    assert healthy()
    for n in NAMES:
        sh('docker', 'kill', '--signal=TERM', n)
    t_end = time.monotonic() + 90
    while any(json.loads(sh('docker', 'inspect', n))[0]['State']['Running'] for n in NAMES):
        assert healthy() and time.monotonic() < t_end, 'shutdown: servers left in place, no forced kill'
        time.sleep(0.25)
    for n in NAMES:
        state = json.loads(sh('docker', 'inspect', n))[0]['State']
        (OUT / (n + '.log')).write_text(sh('docker', 'logs', n))
        assert state['ExitCode'] == 0, (n, state)
    for n in NAMES:
        sh('docker', 'rm', n)   # Ian 11-11: no leftover test containers
    (OUT / 'result.json').write_text(json.dumps({'status': 'PASS'}, indent=2))
    log('DUAL_PASS')
except Exception as e:
    for n in NAMES:
        try:
            (OUT / (n + '.log')).write_text(sh('docker', 'logs', n))
        except Exception:
            pass
    if O.mode == 'replica':
        try:
            (OUT / 'router.log').write_text(sh('docker', 'logs', ROUTER))
            sh('docker', 'stop', ROUTER)      # the router holds no GPU
        except Exception:
            pass
    (OUT / 'FAILED').write_text(f'{e!r}\nLeave the container in place. No forced kill or automatic recovery.\n')
    log('FAILED', repr(e))
    raise
