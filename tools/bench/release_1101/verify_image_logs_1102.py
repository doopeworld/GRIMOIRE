#!/usr/bin/env python3
"""Check the v1.10.3 candidate's run artifacts before promoting or uploading it.

The release gate covers the five v1.10.2 configurations. New options stay off by default, so
their answer counts are printed as facts, not gated, except the checks every server must pass.
"""
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
# key: label, users, single-user coding gate, concurrent coding gate, v1.10.2 pp / tg / total
RELEASE = {
    'img-q38': ('Qwen GPTQ + MTP', 4, True, True, (2020.07, 67.08, 180.47)),
    'img-orn': ('Ornith safetensors MXFP4', 8, False, False, (8803.61, 174.72, 378.93)),
    'img-qg-mtp': ('Qwen GGUF native + MTP', 4, True, True, (1972.94, 59.41, 141.55)),
    'img-og-native': ('Ornith GGUF native', 8, False, False, (7258.69, 155.19, 133.78)),
    'img-og-mxfp4': ('Ornith GGUF as MXFP4', 8, False, False, (8752.84, 175.14, 378.44)),
}
OPTIONS = {
    'opt-moe': ('Ornith GGUF native + NATIVE_SMALLM', 8),
    'opt-mtp': ('Qwen GPTQ + MTP_GPU_DRAFT', 4),
    'opt-w8-off': ('Qwen FP8 as INT8, W8A8=0', 1),
    'opt-w8-wide': ('Qwen FP8 as INT8, W8A8=3', 1),
}


def server_checks(stem, label, allow_answer_fail):
    run = stem.with_suffix('.run').read_text()
    for needed in ("the image's own binaries", 'Coherence test PASSED', 'server stopped gracefully',
                   'gpu0 path correctable=0', 'uncorrectable(all ports)=0'):
        assert needed in run, f'{label}: missing {needed!r}'
    text = run
    if allow_answer_fail:
        text = '\n'.join(l for l in run.splitlines() if 'tests=FAIL' not in l)
    bad = re.search(r'\b(?:FAIL|FAILED|STOP|LEFT RUNNING|Traceback)\b', text)
    assert not bad, f'{label}: {bad.group(0)}'
    return run


def rates(stem, label):
    out = {}
    for line in stem.with_suffix('.benchy.md').read_text().splitlines():
        parts = [part.strip() for part in line.split('|')]
        if len(parts) > 3:
            m = re.fullmatch(r'(pp4096|tg128)(?: \(c(\d+)\))?', parts[2])
            if m:
                out[(m.group(1), int(m.group(2) or 1))] = float(parts[3].split()[0])
    assert out and all(v > 0 for v in out.values()), f'{label}: no rates'
    return out


def conc_count(stem):
    path = stem.with_suffix('.conc4.json')
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    return sum(1 for r in rows.values() if r['tests'].get('syntax') and r['tests'].get('tests')), len(rows)


def pct(new, old):
    return f'{100 * (new - old) / old:+.1f}%'


print('| Model | Prompt pp4096: v1.10.2 -> now | Decode, 1 user | Total, many users | Checks |')
print('|---|---|---|---|---|')
worst = 0.0
for key, (label, users, code, conc, old) in RELEASE.items():
    stem = root / f'r1102-{key}'
    run = server_checks(stem, label, allow_answer_fail=not conc)
    r = rates(stem, label)
    checks = ['coherence']
    if code:
        rows = [json.loads(l) for l in stem.with_suffix('.code.json').read_text().splitlines() if l]
        assert len(rows) == 2 and all(x['quality']['syntax'] and x['quality']['tests'] for x in rows), label
        checks.append('code 2/2')
    if conc:
        c = conc_count(stem)
        assert c == (8, 8), f'{label}: concurrent coding {c}'
        checks.append('concurrent code 8/8')
    elif conc_count(stem):
        checks.append('concurrent code %d/%d (not gated)' % conc_count(stem))
    if not code:
        assert "[ctest] answer: 'Paris'" in run, label
        checks.append('Paris')
    new = (r[('pp4096', 1)], r[('tg128', 1)], r[('tg128', users)])
    worst = min(worst, *[(n - o) / o for n, o in zip(new, old)])
    print(f'| {label} | {old[0]:.0f} -> {new[0]:.0f} ({pct(new[0], old[0])}) '
          f'| {old[1]:.1f} -> {new[1]:.1f} ({pct(new[1], old[1])}) '
          f'| {old[2]:.1f} -> {new[2]:.1f} ({pct(new[2], old[2])}, {users} users) | {"; ".join(checks)} |')
print(f'worst speed change vs v1.10.2: {100 * worst:+.1f}%')
print('IMAGE_LOG_CHECK_PASS')
print()
print('| New option (off by default) | Prompt pp4096 | Decode, 1 user | Total, many users | Answers |')
print('|---|---:|---:|---:|---|')
for key, (label, users) in OPTIONS.items():
    stem = root / f'r1102-{key}'
    server_checks(stem, label, allow_answer_fail=True)
    r = rates(stem, label)
    many = f"{r[('tg128', users)]:.1f} ({users} users)" if users > 1 else '-'
    answers = []
    code = stem.with_suffix('.code.json')
    if code.exists():
        rows = [json.loads(l) for l in code.read_text().splitlines() if l]
        answers.append('code %d/%d' % (sum(1 for x in rows if x['quality']['syntax'] and x['quality']['tests']), len(rows)))
    if conc_count(stem):
        answers.append('concurrent code %d/%d' % conc_count(stem))
    print(f"| {label} | {r[('pp4096', 1)]:.0f} | {r[('tg128', 1)]:.1f} | {many} | {'; '.join(answers) or '-'} |")
print()
print('| Same 4 coding prompts on Ornith, 2 rounds | Pass together | Pass alone | Together = alone | Round 0 = round 1, alone |')
print('|---|---:|---:|---:|---:|')
for key, label in (('img-orn', 'Safetensors MXFP4'), ('img-og-mxfp4', 'GGUF as MXFP4'),
                   ('img-og-native', 'GGUF native'), ('opt-moe', 'GGUF native + NATIVE_SMALLM')):
    stem = root / f'r1102-{key}'
    conc = json.loads(stem.with_suffix('.conc4.json').read_text())
    seq = json.loads(stem.with_suffix('.seq4.json').read_text())
    ok = lambda rows: sum(1 for x in rows.values() if x['tests'].get('tests'))
    same = sum(1 for k in conc if conc[k]['text'] == seq[k]['text'])
    stable = sum(1 for i in range(4) if seq[f'0-{i}']['text'] == seq[f'1-{i}']['text'])
    print(f'| {label} | {ok(conc)}/{len(conc)} | {ok(seq)}/{len(seq)} | {same}/{len(conc)} | {stable}/4 |')
