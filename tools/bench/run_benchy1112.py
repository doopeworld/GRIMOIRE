#!/usr/bin/env python3
"""llama-benchy 0.4.0 (the tool Ian's friend uses) on ONE B70 (gpu0) with a Grimoire build, 2026-11-12.
Flags: --pp 4096 16384 --tg 128 --concurrency 1 2 4 8 --runs 3 --no-cache (his reported points: prompt 4K/16K,
1/2/4 users; his exact command line is not known).  Book text served locally (bench-1003/book).
Safety as run_conc1111.py: gpu1 never mapped; guard (xe resets, PCI config ffff, fatal/nonfatal AER) before
and during; a guard trip stops only the CPU client, the server is left in place; stop = TERM + wait."""
import argparse, fcntl, hashlib, json, os, pathlib, re, signal, subprocess, time, urllib.request

B = pathlib.Path('/mnt/storage/isos/grimoire-runs/bench-1022')
P = argparse.ArgumentParser()
P.add_argument('--tree', required=True)
P.add_argument('--tag', required=True)
P.add_argument('--env', action='append', default=[])
P.add_argument('--ctx', type=int, default=32768)
O = P.parse_args()
assert re.fullmatch(r'[a-z0-9-]+', O.tag)
TREE = pathlib.Path(O.tree)
OUT = B / 'logs' / ('benchy1112-' + O.tag)
NAME = 's1112-benchy-' + O.tag
PORT = 6903
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
    return json.loads(sh('docker', 'inspect', NAME))[0]['State']['Running']


src = (TREE / 'BUILD_SOURCE').read_text().strip()
sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TREE / 'bin').glob('*')) if p.is_file()}
lock = open('/mnt/user/isos/grimoire-runs/gpu0.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not sh('docker', 'ps', '-q'), 'another container is running (ComfyUI?): do not start'
assert not (B / 'USER_HOLD_B70').exists(), 'USER_HOLD_B70 present'
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
       'GRIMOIRE_SAMPLE_COUPLED_MTP': '0', 'GRIMOIRE_SPEC_STATS': '1', 'GRIMOIRE_POWER_LIMIT_W': '0'}
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
sh(*args)
meta = {'tag': O.tag, 'source': src, 'sha256': sha, 'env': env, 'ctx': O.ctx, 'gpu': '03:00.0 only',
        'client': 'llama-benchy 0.4.0 --pp 4096 16384 --tg 128 --concurrency 1 2 4 8 --runs 3 --no-cache'}
(OUT / 'meta.json').write_text(json.dumps(meta, indent=2))
log('server started', NAME)
try:
    t_end = time.monotonic() + 600
    while True:
        assert healthy(), 'guard during load: server left in place'
        assert running(), 'server exited during load'
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        assert time.monotonic() < t_end, 'load deadline: server left in place'
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
    book = subprocess.Popen(['python3', '-m', 'http.server', '8999', '--bind', '127.0.0.1'],
                            cwd='/mnt/storage/isos/grimoire-runs/bench-1003/book',
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    cenv = dict(os.environ, UV_CACHE_DIR='/mnt/storage/isos/grimoire-cache/uv')
    try:
        with (OUT / 'benchmark.out').open('w') as bl:
            client = subprocess.Popen(['taskset', '-c', '14,15', 'uvx', 'llama-benchy@0.4.0',
                                       '--base-url', f'http://127.0.0.1:{PORT}/v1', '--model', model_id,
                                       '--tokenizer', '/mnt/models/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8',
                                       '--book-url', 'http://127.0.0.1:8999/1661-0.txt',
                                       '--pp', '4096', '16384', '--tg', '128', '--concurrency', '1', '2', '4', '8',
                                       '--runs', '3', '--no-cache', '--format', 'md',
                                       '--save-result', str(OUT / 'benchy.md')],
                                      cwd=B, env=cenv, stdout=bl, stderr=subprocess.STDOUT, start_new_session=True)
            log('llama-benchy started pid', client.pid)
            deadline = time.monotonic() + 7200
            while client.poll() is None:
                if not healthy() or not running() or time.monotonic() > deadline:
                    os.killpg(client.pid, signal.SIGTERM)   # the CPU HTTP client only
                    raise RuntimeError('benchmark client stopped by guard/exit/deadline; server left in place')
                time.sleep(2)
        assert client.returncode == 0, 'benchmark client failed: see benchmark.out'
    finally:
        os.killpg(book.pid, signal.SIGTERM)
    log('llama-benchy done')
    (OUT / 'server.log').write_text(sh('docker', 'logs', NAME))
    assert healthy()
    sh('docker', 'kill', '--signal=TERM', NAME)
    t_end = time.monotonic() + 60
    while running():
        assert healthy() and time.monotonic() < t_end, 'shutdown: server left in place, no forced kill'
        time.sleep(0.25)
    state = json.loads(sh('docker', 'inspect', NAME))[0]['State']
    (OUT / 'server.log').write_text(sh('docker', 'logs', NAME))
    assert state['ExitCode'] == 0, state
    sh('docker', 'rm', NAME)   # Ian 11-11: no leftover test containers
    (OUT / 'result.json').write_text(json.dumps({'status': 'PASS'}, indent=2))
    log('BB_PASS')
except Exception as e:
    try:
        (OUT / 'server.log').write_text(sh('docker', 'logs', NAME))
    except Exception:
        pass
    (OUT / 'FAILED').write_text(f'{e!r}\nLeave the container in place. No forced kill or automatic recovery.\n')
    log('FAILED', repr(e))
    raise
