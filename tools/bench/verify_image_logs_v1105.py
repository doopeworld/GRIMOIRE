#!/usr/bin/env python3
"""Check the v1.10.5 candidate's run artifacts before promoting or uploading it.

Same gate as v1.10.4 (verify_image_logs_v1104.py), measured against v1.10.4's own image runs (r1104-img-*):
every server passes coherence, stops gracefully with no new link errors; Qwen rows need code 2/2 and
concurrent code 8/8; the others answer Paris. Flash-Next (its code path did not change) must give v1.10.4's
fixed answers exactly. The small-card row (Ornith NVFP4, tiered experts, 6 GB expert budget: the CPU split
in decode) must answer Paris and pass its coding test; its speed is printed against v1.10.4's host run
(nx1105-split-old). Ornith concurrent coding is printed, not gated.
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
    'img-og-native': ('Ornith GGUF native', 8, False, False),
    'img-og-mxfp4': ('Ornith GGUF as MXFP4', 8, False, False),
}


def server_checks(stem, label, allow_answer_fail, own_binaries=True):
    run = stem.with_suffix('.run').read_text()
    needed = ['Coherence test PASSED', 'server stopped gracefully', 'gpu0 path correctable=0',
              'uncorrectable(all ports)=0']
    if own_binaries:
        needed.append("the image's own binaries")
    for n in needed:
        assert n in run, f'{label}: missing {n!r}'
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


def code_rows(stem):
    return [json.loads(l) for l in stem.with_suffix('.code.json').read_text().splitlines() if l]


def pct(new, old):
    return f'{100 * (new - old) / old:+.1f}%'


print('| Model | Prompt pp4096: v1.10.4 -> v1.10.5 | Decode, 1 user | Total, many users | Checks |')
print('|---|---|---|---|---|')
worst = 0.0
for key, (label, users, code, conc) in RELEASE.items():
    stem = root / f'r1105-{key}'
    run = server_checks(stem, label, allow_answer_fail=not conc)
    r, o = rates(stem, label), rates(root / f'r1104-{key}', label + ' (v1.10.4)')
    checks = ['coherence']
    if code:
        rows = code_rows(stem)
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
print(f'worst speed change vs v1.10.4: {100 * worst:+.1f}%')

stem = root / 'r1105-img-fn'
server_checks(stem, 'Flash-Next', allow_answer_fail=False)
r, o = rates(stem, 'Flash-Next'), rates(root / 'r1104-img-fn', 'Flash-Next (v1.10.4)')
img = json.loads(stem.with_suffix('.gen.json').read_text())
old = json.loads((root / 'r1104-img-fn.gen.json').read_text())
same = sum(img[k]['text'] == old.get(k, {}).get('text') for k in img)
assert same == len(img) == len(old), f'Flash-Next: {same}/{len(img)} answers equal v1.10.4'
print(f"Flash-Next NVFP4, one card: pp512 {o[('pp512', 1)]:.1f} -> {r[('pp512', 1)]:.1f}, "
      f"tg128 {o[('tg128', 1)]:.2f} -> {r[('tg128', 1)]:.2f} tok/s; fixed answers equal v1.10.4 {same}/{len(img)}")

stem = root / 'r1105-img-split'
run = server_checks(stem, 'Ornith NVFP4 tiered 6 GB', allow_answer_fail=False)
assert "[ctest] answer: 'Paris'" in run
rows = code_rows(stem)
assert rows and all(x['quality']['syntax'] and x['quality']['tests'] for x in rows), 'split: coding test'
r = rates(stem, 'split')
oh = root / 'nx1105-split-old'
o = rates(oh, 'split, v1.10.4 host run') if oh.with_suffix('.benchy.md').exists() else {}
row = ', '.join(f"{t} {o[(t, 1)]:.1f} -> {r[(t, 1)]:.1f} ({pct(r[(t, 1)], o[(t, 1)])})" if (t, 1) in o
                else f"{t} {r[(t, 1)]:.1f}" for t in ('pp512', 'pp2048', 'tg128'))
print(f'Ornith NVFP4, tiered experts, 6 GB expert budget (CPU split; v1.10.4 = host run): {row}; '
      f'Paris; code {len(rows)}/{len(rows)} at {max(x["decode_tok_s"] for x in rows):.1f} tok/s')
print('IMAGE_LOG_CHECK_PASS')
