#!/usr/bin/env python3
"""Check the v1.10.4 candidate's run artifacts before promoting or uploading it.

Same gate as v1.10.3 (verify_image_logs_1102.py), measured against v1.10.3's own runs (r1102-img-*):
every server passes coherence, stops gracefully with no new link errors; Qwen rows need code 2/2 and
concurrent code 8/8; the others answer Paris. Flash-Next is new: its three fixed answers must equal the
same binaries' host run (fn-serve-new-v1104-quiet). Ornith concurrent coding is printed, not gated.
"""
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
# key: label, users, single-user coding gate, concurrent coding gate
RELEASE = {
    'img-q38': ('Qwen GPTQ + MTP', 4, True, True),
    'img-orn': ('Ornith safetensors MXFP4', 8, False, False),
    'img-qg-mtp': ('Qwen GGUF native + MTP', 4, True, True),
    'img-og-native': ('Ornith GGUF native (experts grouped by default)', 8, False, False),
    'img-og-mxfp4': ('Ornith GGUF as MXFP4', 8, False, False),
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
            m = re.fullmatch(r'(pp\d+|tg128)(?: \(c(\d+)\))?', parts[2])
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


print('| Model | Prompt pp4096: v1.10.3 -> v1.10.4 | Decode, 1 user | Total, many users | Checks |')
print('|---|---|---|---|---|')
worst = 0.0
for key, (label, users, code, conc) in RELEASE.items():
    stem = root / f'r1104-{key}'
    run = server_checks(stem, label, allow_answer_fail=not conc)
    r, o = rates(stem, label), rates(root / f'r1102-{key}', label + ' (v1.10.3)')
    checks = ['coherence']
    if code:
        rows = [json.loads(l) for l in stem.with_suffix('.code.json').read_text().splitlines() if l]
        assert len(rows) == 2 and all(x['quality']['syntax'] and x['quality']['tests'] for x in rows), label
        checks.append('code 2/2 at %.1f tok/s' % max(x['decode_tok_s'] for x in rows))
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
    old = (o[('pp4096', 1)], o[('tg128', 1)], o[('tg128', users)])
    worst = min(worst, *[(n - p) / p for n, p in zip(new, old)])
    print(f'| {label} | {old[0]:.0f} -> {new[0]:.0f} ({pct(new[0], old[0])}) '
          f'| {old[1]:.1f} -> {new[1]:.1f} ({pct(new[1], old[1])}) '
          f'| {old[2]:.1f} -> {new[2]:.1f} ({pct(new[2], old[2])}, {users} users) | {"; ".join(checks)} |')
print(f'worst speed change vs v1.10.3: {100 * worst:+.1f}%')

stem = root / 'r1104-img-fn'
server_checks(stem, 'Flash-Next', allow_answer_fail=False)
r = rates(stem, 'Flash-Next')
img = json.loads(stem.with_suffix('.gen.json').read_text())
host = json.loads((root / 'fn-serve-new-v1104-quiet.gen.json').read_text())
same = sum(img[k]['text'] == host.get(k, {}).get('text') for k in img)
assert same == len(img) == len(host), f'Flash-Next: {same}/{len(img)} answers equal the host run'
print(f"Flash-Next NVFP4, one card: pp512 {r[('pp512', 1)]:.1f}, tg128 {r[('tg128', 1)]:.2f} tok/s; "
      f'fixed answers equal the host run {same}/{len(img)}')

stem = root / 'r1104-opt-moe-off'
run = server_checks(stem, 'Ornith GGUF native, GRIMOIRE_MOE_NATIVE_SMALLM=0', allow_answer_fail=True)
assert "[ctest] answer: 'Paris'" in run
r = rates(stem, 'opt-out')
print(f"Opt-out GRIMOIRE_MOE_NATIVE_SMALLM=0: pp4096 {r[('pp4096', 1)]:.0f}, tg128 {r[('tg128', 1)]:.1f}, "
      f"8 users {r[('tg128', 8)]:.1f} tok/s total")
print('IMAGE_LOG_CHECK_PASS')
